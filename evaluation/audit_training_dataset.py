#!/usr/bin/env python3
"""Audit generated training files, class/source isolation and centre annotations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def audit(root: Path):
    root = root.resolve()
    manifest = root / 'queries.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    errors = []
    ids = set()
    checksum_count = 0
    classes, sources, families = defaultdict(set), defaultdict(set), defaultdict(set)
    split_counts, scene_counts = Counter(), Counter()
    for row in rows:
        query = row['query_id']
        if query in ids:
            errors.append(f'duplicate query id: {query}')
        ids.add(query)
        split = row['split']
        split_counts[split] += 1
        scene_counts[row.get('scene_type', 'unknown')] += 1
        if split not in {'train','validation','test'}:
            errors.append(f'invalid split: {query}')
        image = (root / row['image_path']).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            errors.append(f'missing or unsafe image: {query}')
        else:
            expected_hash = row.get('sha256', row.get('image_sha256'))
            if expected_hash:
                checksum_count += 1
                if hashlib.sha256(image.read_bytes()).hexdigest() != expected_hash:
                    errors.append(f'image checksum mismatch: {query}')
            elif row.get('generator_version', 0) >= 3:
                errors.append(f'missing image checksum: {query}')
        classes[row['expected_slug']].add(split)
        sources[row['source_sha256']].add(split)
        if row.get('leakage_group'):
            families[row['leakage_group']].add(split)
        objects = row.get('objects', [])
        for obj in objects:
            classes[obj['slug']].add(split)
            sources[obj['source_sha256']].add(split)
        if row.get('scene_type') == 'center_target_three_bottles':
            targets = [obj for obj in objects if obj.get('is_target')]
            if len(targets) != 1 or targets[0]['slug'] != row['expected_slug']:
                errors.append(f'invalid centre label: {query}')
                continue
            if len(objects) != 3 or len({obj['slug'] for obj in objects}) != 3:
                errors.append(f'invalid distractors: {query}')
            center_x, center_y = row['width']/2, row['height']/2
            distances = []
            for obj in objects:
                x1,y1,x2,y2 = obj['bbox_xyxy']
                if not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                    errors.append(f'invalid bounds: {query}')
                distances.append((((x1+x2)/2-center_x)**2+((y1+y2)/2-center_y)**2,obj['is_target']))
            if not min(distances)[1]:
                errors.append(f'target is not closest to image centre: {query}')
    for kind, mapping in [('class', classes),('source',sources),('family',families)]:
        for key, splits in mapping.items():
            if len(splits) > 1:
                errors.append(f'{kind} crosses splits: {key}')
    summary_path = root/'dataset.json'
    if summary_path.exists():
        expected=json.loads(summary_path.read_text())
        if expected['query_count'] != len(rows):
            errors.append('dataset summary query_count mismatch')
    return {'valid':not errors, 'queries':len(rows),'classes':len(classes),'source_images':len(sources),
            'split_counts':dict(split_counts),'scene_counts':dict(scene_counts),
            'checksums_verified':checksum_count,
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=audit(args.dataset)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['valid'] else 1

if __name__=='__main__':
    raise SystemExit(main())
