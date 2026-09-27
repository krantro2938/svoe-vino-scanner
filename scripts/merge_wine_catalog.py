#!/usr/bin/env python3
"""Build a separate expanded catalogue without changing organiser identities."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_image(root, relative):
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path


def merge_catalog(official, snapshot, official_images, snapshot_root, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    official_rows, web_rows = read_jsonl(official), read_jsonl(snapshot)
    web = {}
    for row in web_rows:
        slug = row.get('slug')
        if not isinstance(slug, str) or not slug.strip() or not row.get('name'):
            raise ValueError('snapshot record requires slug and name')
        if slug in web:
            raise ValueError(f'duplicate snapshot slug: {slug}')
        web[slug] = row
    # Verify all candidate bytes before allowing any gate recovery. Include blocked
    # identities in collision checks: they are still distinct catalogue labels.
    verified_web = {}
    all_hash_labels, asset_labels = defaultdict(set), defaultdict(set)
    for row in official_rows:
        source = safe_image(official_images, row.get('photo_name'))
        if source:
            all_hash_labels[hashlib.sha256(source.read_bytes()).hexdigest()].add(row['slug'])
        asset = str(row.get('media_path') or '')
        if '/uploads/' in asset:
            asset_labels[Path(asset).name].add(row['slug'])
    for row in web_rows:
        source = safe_image(snapshot_root, row.get('image_path') or row.get('photo_name'))
        if source:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if row.get('image_sha256') and row['image_sha256'] != digest:
                raise ValueError(f'website image checksum mismatch: {row["slug"]}')
            all_hash_labels[digest].add(row['slug'])
            if row.get('image_sha256') == digest and row.get('index_ready'):
                verified_web[row['slug']] = (source, digest)
        asset = str(row.get('source_image_path') or '')
        if asset.startswith('/uploads/'):
            asset_labels[Path(asset).name].add(row['slug'])
    def normalized(value):
        return ' '.join(unicodedata.normalize('NFKC', value or '').casefold().split())
    def recoverable(original):
        slug = original['slug']
        candidate = web.get(slug, {})
        if original.get('index_ready') or original.get('media_status') not in {'ambiguous', 'unresolved'}:
            return False
        if slug not in verified_web:
            return False
        if any(not normalized(original.get(field)) or normalized(original.get(field)) != normalized(candidate.get(field))
               for field in ('name', 'winery')):
            return False
        asset = str(candidate.get('source_image_path') or '')
        if not asset.startswith('/uploads/'):
            return False
        return (all_hash_labels[verified_web[slug][1]] == {slug}
                and asset_labels[Path(asset).name] == {slug})
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'images'
    images.mkdir(exist_ok=True)
    report = {'official_records':len(official_rows), 'snapshot_records':len(web_rows),
              'recovered_slugs':[], 'added_slugs':[], 'matched_slugs':[], 'missing_images':[], 'duplicate_image_groups':[],
              'official_sha256':hashlib.sha256(official.read_bytes()).hexdigest(),
              'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    result, hashes = [], defaultdict(list)
    seen = set()
    official_slugs = {o['slug'] for o in official_rows}
    for original in official_rows + [r for s,r in sorted(web.items()) if s not in official_slugs]:
        row = dict(original)
        slug = row['slug']
        if slug in seen:
            raise ValueError(f'duplicate official slug: {slug}')
        seen.add(slug)
        is_official = slug in official_slugs
        if is_official:
            row['catalog_origin'] = 'organizer'
            source = safe_image(official_images, row.get('photo_name'))
            if slug in web:
                row['website_snapshot'] = web[slug]
                report['matched_slugs'].append(slug)
            if recoverable(original):
                source, digest = verified_web[slug]
                row['index_ready'] = True
                row['reference_origin'] = 'website'
                row['reference_recovery'] = {
                    'reason': 'exact_slug_name_winery_unique_verified_website_image',
                    'original_index_ready': original.get('index_ready'),
                    'original_media_status': original.get('media_status'),
                    'source_url': web[slug].get('source_url', ''),
                    'source_image_path': web[slug]['source_image_path'],
                    'image_sha256': digest,
                    'observed_at': web[slug].get('observed_at', ''),
                }
                report['recovered_slugs'].append(slug)
        else:
            row['catalog_origin'] = 'website'
            source = safe_image(snapshot_root, row.get('image_path') or row.get('photo_name'))
            report['added_slugs'].append(slug)
            row['index_ready'] = bool(row.get('index_ready', True))
        if source:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            filename = digest + source.suffix.lower()
            destination = images / filename
            if not destination.exists():
                shutil.copyfile(source, destination)
            row['original_photo_name'] = row.get('photo_name')
            row['photo_name'] = filename
            row['image_sha256'] = digest
            hashes[digest].append(row)
        else:
            row['index_ready'] = False
            row['photo_name'] = None
            report['missing_images'].append(slug)
        result.append(row)
    # Keep exact-slug alternate website photographs as extra training views.
    candidates = []
    by_slug = {row['slug']: row for row in result}
    for row in result:
        if row.get('photo_name') and row.get('index_ready'):
            candidates.append(dict(slug=row['slug'], image_path=row['photo_name'],
                source_sha256=row['image_sha256'], source_kind=row.get('reference_origin', row['catalog_origin']),
                source_archive_path='' if row.get('reference_origin') == 'website' else row.get('media_path', ''),
                source_url=row.get('reference_recovery', {}).get('source_url', row.get('source_url', '')),
                retrieved_at=row.get('reference_recovery', {}).get('observed_at', row.get('observed_at', ''))))
    for slug, row in web.items():
        source = safe_image(snapshot_root, row.get('image_path'))
        if not source or not row.get('index_ready'):
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if row.get('image_sha256') and row['image_sha256'] != digest:
            raise ValueError(f'website image checksum mismatch: {slug}')
        filename = digest + source.suffix.lower()
        if not (images / filename).exists():
            shutil.copyfile(source, images / filename)
        # Include even blocked records in the collision audit, but never in training.
        if all(r['slug'] != slug for r in hashes[digest]):
            hashes[digest].append(by_slug[slug])
        if by_slug[slug].get('index_ready'):
            candidates.append(dict(slug=slug, image_path=filename, source_sha256=digest,
                source_kind='website', source_url=row.get('source_url', ''),
                retrieved_at=row.get('observed_at', '')))
    # Website renditions have different bytes; canonical upload paths catch these too.
    report['shared_source_assets'] = []
    for asset, slugs in sorted(asset_labels.items()):
        if len(slugs) > 1:
            report['shared_source_assets'].append({'asset':asset, 'slugs':sorted(slugs)})
            for slug in slugs:
                by_slug[slug]['index_ready'] = False
                by_slug[slug]['training_blockers'] = sorted(set(by_slug[slug].get('training_blockers', []) + ['shared_source_asset']))
    for digest, rows in hashes.items():
        if len(rows) > 1:
            report['duplicate_image_groups'].append({'sha256':digest, 'slugs':[r['slug'] for r in rows]})
            # Identical pixels cannot teach the model which conflicting identity is correct.
            for row in rows:
                row['index_ready'] = False
                row['training_blockers'] = sorted(set(row.get('training_blockers', []) + ['shared_image_bytes']))
    clean_candidates, seen_candidates = [], set()
    for candidate in candidates:
        key = (candidate['slug'], candidate['source_sha256'])
        if len({r['slug'] for r in hashes[candidate['source_sha256']]}) > 1 or key in seen_candidates:
            continue
        # A known ambiguous original identity remains quarantined across all views.
        if not by_slug[candidate['slug']].get('index_ready'):
            continue
        seen_candidates.add(key)
        family = by_slug[candidate['slug']].get('winery') or candidate['slug']
        candidate.update(leakage_group=family, hard_negative_group=family)
        clean_candidates.append(candidate)
    clean_candidates.sort(key=lambda r: (r['slug'], r['image_path']))
    (output/'training_references.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in clean_candidates),encoding='utf-8')
    report['training_references'] = len(clean_candidates)
    report['training_classes'] = len({r['slug'] for r in clean_candidates})
    report['expanded_records'] = len(result)
    report['index_ready_records'] = sum(bool(r.get('index_ready')) for r in result)
    report['recovered_count'] = len(report['recovered_slugs'])
    report['added_count'] = len(report['added_slugs'])
    report['matched_count'] = len(report['matched_slugs'])
    (output/'catalog.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in result), encoding='utf-8')
    (output/'merge_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official',type=Path,default=Path('data/generated/catalog/catalog.jsonl'))
    parser.add_argument('--official-images',type=Path,default=Path('data/catalog-images'))
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--snapshot-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('data/generated/expanded'))
    args=parser.parse_args()
    report=merge_catalog(args.official,args.snapshot,args.official_images,args.snapshot_root,args.output)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
