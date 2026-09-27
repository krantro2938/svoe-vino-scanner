"""Standalone SIFT retrieval experiment. Validation only; never selects test rows.

Catalogue references are the retrieval gallery, not fine-tuning data. Cache uses
NPZ with no pickle. Scores are geometric evidence, not calibrated confidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

PARAMETERS = dict(max_side=900, nfeatures=700, contrast_threshold=0.025,
                  ratio=0.78, shortlist=20, ransac_pixels=5.0,
                  center_width=0.46, flann_checks=64)


def read_rgb(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGBA')
        bg = Image.new('RGBA', image.size, 'white')
        bg.alpha_composite(image)
        image = bg.convert('RGB')
        image.thumbnail((PARAMETERS['max_side'], PARAMETERS['max_side']))
        return np.array(image)


class LocalFeatures:
    def __init__(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(42)
        self.sift = cv2.SIFT_create(nfeatures=PARAMETERS['nfeatures'],
                                    contrastThreshold=PARAMETERS['contrast_threshold'])
        self.refs = []

    def features(self, image, query=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = None
        h, w = gray.shape
        if query and w / h > 0.85:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            left = round(w * (1 - PARAMETERS['center_width']) / 2)
            mask[:, left:w-left] = 255
        kp, desc = self.sift.detectAndCompute(gray, mask)
        points = np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 2)
        if desc is None:
            desc = np.empty((0, 128), dtype=np.float32)
        # RootSIFT suppresses bursty strong gradients while retaining local detail.
        desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-8))
        return points, desc

    def build(self, manifest, images, cache):
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        pts, vectors, offsets, slugs = [], [], [0], []
        def extract(row):
            # Separate detector per worker; evaluation stays strictly one thread.
            worker = LocalFeatures()
            return worker.features(read_rgb(images / row['image_path']))
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (row, (points, desc)) in enumerate(zip(rows, executor.map(extract, rows))):
                pts.append(points); vectors.append(desc); offsets.append(offsets[-1] + len(desc))
                slugs.append(row['slug'])
                if (i + 1) % 200 == 0:
                    print(f'Indexed {i+1}/{len(rows)} references', flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, points=np.concatenate(pts), descriptors=np.concatenate(vectors),
                            offsets=np.array(offsets), slugs=np.array(slugs),
                            parameters=json.dumps(PARAMETERS),
                            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())

    def load(self, cache):
        data = np.load(cache, allow_pickle=False)
        if json.loads(str(data['parameters'])) != PARAMETERS:
            raise ValueError('Cache parameters differ; rebuild the index')
        self.points, self.desc = data['points'], data['descriptors']
        self.offsets, self.slugs = data['offsets'], data['slugs'].tolist()
        self.owners = np.repeat(np.arange(len(self.slugs)), np.diff(self.offsets))
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=PARAMETERS['flann_checks']))
        self.flann.add([self.desc]); self.flann.train()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def predict(self, image):
        points, desc = self.features(image, query=True)
        if len(desc) < 2:
            return {'slug': None, 'candidates': [], 'query_features': len(desc)}
        pairs = self.flann.knnMatch(desc, k=2)
        votes = Counter()
        for pair in pairs:
            if len(pair) < 2: continue
            a, b = pair
            # Soft votes keep near-identical label families in the shortlist.
            votes[int(self.owners[a.trainIdx])] += 1.0
            if a.distance < PARAMETERS['ratio'] * b.distance:
                votes[int(self.owners[a.trainIdx])] += 2.0
        ranked = []
        for index, vote in votes.most_common(PARAMETERS['shortlist']):
            lo, hi = self.offsets[index:index+2]
            reference = self.desc[lo:hi]
            if len(reference) < 2: continue
            pairs = self.bf.knnMatch(desc, reference, k=2)
            matches = [a for a,b in pairs if a.distance < PARAMETERS['ratio'] * b.distance]
            # One feature in the reference must not explain repeated query detail.
            unique = {}
            for match in matches:
                if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
                    unique[match.trainIdx] = match
            matches = list(unique.values())
            inliers = 0
            if len(matches) >= 4:
                src = np.float32([self.points[lo+m.trainIdx] for m in matches])
                dst = np.float32([points[m.queryIdx] for m in matches])
                matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, PARAMETERS['ransac_pixels'], maxIters=1000, confidence=0.995)
                if matrix is not None and mask is not None:
                    inliers = int(mask.sum())
            score = inliers + 0.05 * len(matches) + 0.001 * vote
            ranked.append({'slug': self.slugs[index], 'score': round(score, 4),
                           'inliers': inliers, 'matches': len(matches)})
        ranked.sort(key=lambda r: (-r['score'], r['slug']))
        return {'slug': ranked[0]['slug'] if ranked else None,
                'candidates': ranked[:5], 'query_features': len(desc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', type=Path, default=Path('data/generated/expanded/training_references.jsonl'))
    p.add_argument('--images', type=Path, default=Path('data/generated/expanded/images'))
    p.add_argument('--cache', type=Path, default=Path('evaluation/artifacts/local-features/index.npz'))
    p.add_argument('--dataset', type=Path, default=Path('evaluation/artifacts/training-expanded-v3'))
    p.add_argument('--output', type=Path, default=Path('evaluation/artifacts/local-features/validation.json'))
    p.add_argument('--per-scene', type=int, default=100)
    p.add_argument('--build', action='store_true')
    args = p.parse_args()
    engine = LocalFeatures()
    if args.build or not args.cache.exists(): engine.build(args.references, args.images, args.cache)
    start = time.perf_counter(); engine.load(args.cache); load_time = time.perf_counter()-start
    rows = [json.loads(line) for line in (args.dataset/'queries.jsonl').read_text().splitlines()]
    chosen = []
    for scene in ('single_reference', 'center_target_three_bottles'):
        selected = sorted((r for r in rows if r['split']=='validation' and r['scene_type']==scene),key=lambda r:r['query_id'])
        chosen.extend(selected[:args.per_scene])
    out = {'parameters': PARAMETERS, 'index_load_seconds': load_time, 'references':len(engine.slugs),
           'descriptor_count': len(engine.desc), 'split':'validation',
           'selection':'query_id-sorted first N per scene, validation only',
           'reference_manifest_sha256':hashlib.sha256(args.references.read_bytes()).hexdigest(),
           'query_manifest_sha256':hashlib.sha256((args.dataset/'queries.jsonl').read_bytes()).hexdigest(),
           'opencv_version':cv2.__version__,
           'caveat':'Synthetic catalogue derivatives; not independent real-world accuracy.', 'results':[]}
    for i,row in enumerate(chosen):
        start=time.perf_counter(); prediction=engine.predict(read_rgb(args.dataset/row['image_path'])); elapsed=(time.perf_counter()-start)*1000
        out['results'].append(dict(query_id=row['query_id'], scene_type=row['scene_type'], expected_slug=row['expected_slug'],
                                   correct=prediction['slug']==row['expected_slug'], latency_ms=round(elapsed,2), **prediction))
        if (i+1)%20==0: print(f'Evaluated {i+1}/{len(chosen)}',flush=True)
    out['summary']={}
    for scene in ('all', 'single_reference', 'center_target_three_bottles'):
        subset=[r for r in out['results'] if scene=='all' or r['scene_type']==scene]
        latency=[r['latency_ms'] for r in subset]
        out['summary'][scene]=dict(count=len(subset), correct=sum(r['correct'] for r in subset),
                                   top1=sum(r['correct'] for r in subset)/len(subset),
                                   p50_ms=float(np.percentile(latency,50)),p95_ms=float(np.percentile(latency,95)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['summary']),flush=True)

if __name__=='__main__': main()
