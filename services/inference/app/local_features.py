"""Optional CPU RootSIFT gallery retrieval with geometric verification."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

INDEX_VERSION = 'rootsift-flann-ransac-v1'


def catalog_fingerprint(catalog):
    records = [catalog[k].model_dump(mode='json') for k in sorted(catalog)]
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


PARAMETERS = dict(max_side=900, nfeatures=700, contrast_threshold=0.025,
                  ratio=0.78, shortlist=20, ransac_pixels=5.0,
                  center_width=0.46, flann_checks=64)


def has_side_objects(image):
    """Detect edge-bearing objects on both sides of a wide centre target.

    The 1% minimum-side edge density was selected on validation data. It cleanly
    separates the generated three-bottle shelves from wide single-reference views;
    it remains a routing hint, not a bottle detector or accuracy claim.
    """
    height, width = image.shape[:2]
    if width / height <= 0.85:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 70, 140) > 0
    side = 77  # 30% of the validation resolution.
    return min(float(edges[:, :side].mean()), float(edges[:, -side:].mean())) > 0.01


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

    def build(self, manifest, images, cache, catalog):
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if any(row['slug'] not in catalog for row in rows):
            raise ValueError('Reference slug absent from catalog')
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
                            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            index_version=INDEX_VERSION, catalog_sha256=catalog_fingerprint(catalog))
        checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
        cache.with_suffix(cache.suffix+".sha256").write_text(checksum+"\n")

    def load(self, cache, catalog):
        checksum_path = cache.with_suffix(cache.suffix + '.sha256')
        expected = checksum_path.read_text().strip()
        if hashlib.sha256(cache.read_bytes()).hexdigest() != expected:
            raise ValueError('Local-feature index checksum mismatch')
        data = np.load(cache, allow_pickle=False)
        if str(data['index_version']) != INDEX_VERSION:
            raise ValueError('Unsupported local-feature index version')
        if str(data['catalog_sha256']) != catalog_fingerprint(catalog):
            raise ValueError('Local-feature index catalog mismatch')
        if json.loads(str(data['parameters'])) != PARAMETERS:
            raise ValueError('Cache parameters differ; rebuild the index')
        self.points, self.desc = data['points'], data['descriptors']
        self.offsets, self.slugs = data['offsets'], data['slugs'].tolist()
        if (self.desc.ndim != 2 or self.desc.shape[1] != 128
                or self.points.shape != (len(self.desc), 2)
                or len(self.offsets) != len(self.slugs) + 1
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.desc)
                or np.any(np.diff(self.offsets) < 0)
                or not np.isfinite(self.desc).all()
                or any(slug not in catalog for slug in self.slugs)):
            raise ValueError('Invalid local-feature descriptor arrays')
        data.close()
        self.owners = np.repeat(np.arange(len(self.slugs)), np.diff(self.offsets))
        self.slots_of = {}
        for i, slug in enumerate(self.slugs):
            self.slots_of.setdefault(slug, []).append(i)
        # The catalogue-wide FLANN index (~1 GB) only serves the legacy predict()
        # fallback; the fusion pipeline calls verify() on a shortlist instead.
        self._flann = None
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def _inliers(self, points, desc, index):
        """RANSAC homography inliers between the query and one reference."""
        lo, hi = self.offsets[index:index+2]
        reference = self.desc[lo:hi]
        if len(reference) < 2 or len(desc) < 2:
            return 0, 0
        pairs = self.bf.knnMatch(desc, reference, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < PARAMETERS['ratio'] * p[1].distance]
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
        return inliers, len(matches)

    def verify(self, image, slugs):
        """Geometric evidence for an externally supplied candidate list."""
        rgb = image.convert('RGB') if hasattr(image, 'convert') else Image.fromarray(image)
        rgb = rgb.copy()
        rgb.thumbnail((PARAMETERS['max_side'], PARAMETERS['max_side']))
        points, desc = self.features(np.asarray(rgb), query=True)
        result = {}
        for slug in slugs:
            best = (0, 0)
            for index in self.slots_of.get(slug, ()):
                best = max(best, self._inliers(points, desc, index))
            result[slug] = {'inliers': best[0], 'matches': best[1]}
        return result

    def predict(self, image):
        points, desc = self.features(image, query=True)
        if len(desc) < 2:
            return {'slug': None, 'candidates': [], 'query_features': len(desc)}
        if self._flann is None:
            self._flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=PARAMETERS['flann_checks']))
            self._flann.add([self.desc]); self._flann.train()
        pairs = self._flann.knnMatch(desc, k=2)
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
            inliers, matched = self._inliers(points, desc, index)
            score = inliers + 0.05 * matched + 0.001 * vote
            ranked.append({'slug': self.slugs[index], 'score': round(score, 4),
                           'inliers': inliers, 'matches': matched})
        ranked.sort(key=lambda r: (-r['score'], r['slug']))
        return {'slug': ranked[0]['slug'] if ranked else None,
                'candidates': ranked[:5], 'query_features': len(desc)}



def strong_match(prediction):
    """Validation-selected gate; confidence is deliberately capped, not a probability.

    On the fixed200 validation cases this gate accepted162, all exact-slug correct.
    The30% margin rejects duplicate product identities with competing slugs.
    """
    candidates = prediction['candidates']
    if len(candidates) < 2:
        return None
    first, second = candidates[:2]
    margin = (first['score'] - second['score']) / max(first['score'], 1e-8)
    if first['inliers'] < 10 or margin < 0.30:
        return None
    return first['slug'], min(0.94, 0.76 + 0.18 * margin), margin
