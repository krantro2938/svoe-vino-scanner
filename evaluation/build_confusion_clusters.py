#!/usr/bin/env python3
"""Build deterministic metadata candidate groups, never inferred wine labels."""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

YEARS = re.compile(r'^(?:19|20)\d{2}$')
STOP = {'вино', 'wine', 'vino', 'белое', 'красное', 'розовое', 'сухое', 'полусухое',
        'сладкое', 'полусладкое', 'брют', 'beloe', 'krasnoe', 'rozovoe', 'suhoe',
        'polusuhoe', 'polusladkoe', 'bryut', 'г', 'год', 'года'}


def tokens(value):
    return tuple(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', str(value or '')).casefold(), re.U))


def without_year(value):
    return tuple(t for t in tokens(value) if not YEARS.fullmatch(t))


def make_clusters(rows, max_group=40, max_neighbors=16):
    if max_group < 2 or max_neighbors < 1:
        raise ValueError('max_group must be >=2 and max_neighbors >=1')
    rows = sorted((r for r in rows if r.get('index_ready')), key=lambda r: r['slug'])
    by_slug = {r['slug']: r for r in rows}
    if len(by_slug) != len(rows):
        raise ValueError('duplicate eligible slug')
    families = defaultdict(list)
    buckets = defaultdict(set)
    for row in rows:
        winery = tokens(row.get('winery'))
        if not winery:
            continue
        slug = row['slug']
        families[winery].append(row)
        name = without_year(row.get('name'))
        slug_words = without_year(slug)
        grapes = tuple(sorted(set(tokens(row.get('grapes')))))
        category = tokens(row.get('category'))
        # These are token signatures only, never a claim that labels are equivalent.
        if name:
            buckets[('same_name_without_vintage', winery, name)].add(slug)
        if slug_words:
            buckets[('same_slug_without_vintage', winery, slug_words)].add(slug)
        if grapes and category:
            buckets[('same_grapes_category', winery, grapes + ('|',) + category)].add(slug)
        removed = set(grapes) | set(category) | set(winery) | STOP
        series = tuple(t for t in name if t not in removed and not t.isdigit())
        if len(series) >= 2:
            buckets[('same_series_tokens', winery, series)].add(slug)
    # Near-duplicate names/slug series provide bounded pairs without transitive
    # connected components that would merge an entire winery into one cluster.
    for winery, members in sorted(families.items()):
        for left, right in itertools.combinations(members, 2):
            for field in ('name', 'slug'):
                a, b = set(without_year(left.get(field))), set(without_year(right.get(field)))
                a -= set(winery) | STOP
                b -= set(winery) | STOP
                overlap = a & b
                if len(overlap) >= 2 and len(overlap) / len(a | b) >= 0.65:
                    signature = (left['slug'], right['slug'])
                    buckets[('near_' + field + '_tokens', winery, signature)].update(signature)
    clusters = []
    oversized = Counter()
    weights = {'same_name_without_vintage':1.0, 'same_slug_without_vintage':1.0,
               'same_series_tokens':0.9, 'near_name_tokens':0.8,
               'near_slug_tokens':0.75, 'same_grapes_category':0.6}
    neighbors = defaultdict(dict)
    for (reason, winery, signature), members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        if len(members) > max_group:
            oversized[reason] += 1
            continue
        members = sorted(members)
        identity = json.dumps([reason, winery, signature, members], ensure_ascii=False, separators=(',', ':'))
        cluster_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
        clusters.append({'cluster_id':cluster_id, 'reason':reason, 'winery_tokens':list(winery),
                         'signature':list(signature), 'slugs':members})
        for source, target in itertools.permutations(members, 2):
            candidate = neighbors[source].setdefault(target, {'slug':target, 'score':0, 'reasons':set()})
            candidate['score'] = max(candidate['score'], weights[reason])
            candidate['reasons'].add(reason)
    references = []
    for slug in sorted(by_slug):
        candidates = sorted(neighbors[slug].values(), key=lambda r: (-r['score'], r['slug']))[:max_neighbors]
        references.append({'slug':slug, 'neighbors':[dict(c, reasons=sorted(c['reasons'])) for c in candidates]})
    covered = sum(bool(r['neighbors']) for r in references)
    report = {'schema_version':1, 'eligible_classes':len(rows), 'covered_classes':covered,
              'coverage_fraction':round(covered / len(rows), 6) if rows else 0,
              'uncovered_classes':len(rows)-covered, 'clusters':len(clusters),
              'clusters_by_reason':dict(sorted(Counter(c['reason'] for c in clusters).items())),
              'oversized_groups_excluded':dict(sorted(oversized.items())),
              'directed_neighbor_pairs':sum(len(r['neighbors']) for r in references),
              'max_group':max_group, 'max_neighbors':max_neighbors,
              'interpretation':'Metadata candidate groups only; scores are deterministic priorities, not model probabilities or verified visual confusion.',
              'leakage_policy':'Use training members only for supervised negative sampling; this manifest does not assign or change dataset splits.'}
    return clusters, references, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('data/generated/expanded/catalog.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/artifacts/confusion_clusters'))
    parser.add_argument('--max-group', type=int, default=40)
    parser.add_argument('--max-neighbors', type=int, default=16)
    args = parser.parse_args()
    source = args.catalog.read_bytes()
    rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
    clusters, neighbors, report = make_clusters(rows, args.max_group, args.max_neighbors)
    report['catalog_sha256'] = hashlib.sha256(source).hexdigest()
    report['catalog_path'] = str(args.catalog)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in [('clusters', clusters), ('neighbors', neighbors)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in records), encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
