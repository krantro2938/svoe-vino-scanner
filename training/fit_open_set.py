#!/usr/bin/env python3
"""Fit the "is this wine in the catalogue?" model on absolute evidence.

Rows are queries. Features of the ranker's top candidate: p(top-1),
log RootSIFT inliers, best visual similarity, text coverage. Negatives are
real photos of wines absent from the catalogue (labelled NOT_IN_CATALOG);
positives are real in-catalogue photos plus down-weighted field-like renders.
Real-photo performance is reported with leave-one-out, so every reported
number comes from a model that did not see that photo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))
from app.pipeline import OPEN_SET_FEATURES, FusionModel  # noqa: E402


def rows(features: Path, queries: Path | None, model: FusionModel):
    labels = {}
    if queries:
        labels = {json.loads(l)["query_id"]: json.loads(l)["expected_slug"] for l in queries.read_text().splitlines()}
    out = []
    for line in features.read_text().splitlines():
        row = json.loads(line)
        cands = row["candidates"]
        p = model.probabilities([{k: v for k, v in c.items() if k not in ("slug", "label")} for c in cands])
        top = cands[int(np.argmax(p))]
        x = [float(p.max())] + [float(top[f]) for f in OPEN_SET_FEATURES[1:]]
        known = labels.get(row["query_id"], row["expected"]) != "NOT_IN_CATALOG"
        out.append((x, float(known), bool(top["label"])))
    return out


def fit(x, y, w, l2=0.05, steps=4000, lr=0.1):
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    theta = np.zeros(x.shape[1] + 1)
    for _ in range(steps):
        logits = theta[0] + z @ theta[1:]
        p = 1 / (1 + np.exp(-logits))
        g = p - y
        theta -= lr * (np.concatenate([[np.sum(w * g)], z.T @ (w * g)]) / w.sum()
                       + l2 * np.concatenate([[0.0], theta[1:]]))
    weights = theta[1:] / std
    return theta[0] - np.sum(weights * mean), weights


def class_weight(y: np.ndarray) -> np.ndarray:
    """Balance the few real in-catalogue photos against the many imports."""
    positives = max(1.0, y.sum())
    return np.where(y == 1, (len(y) - positives) / positives, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-features", type=Path, required=True)
    parser.add_argument("--real-queries", type=Path, required=True)
    parser.add_argument("--synthetic-features", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, default=ROOT / "services/inference/data/fusion.json")
    parser.add_argument("--synthetic-weight", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="in-catalogue probability below which the UI says 'not found'")
    parser.add_argument("--write", action="store_true", help="store the model in the fusion file")
    args = parser.parse_args()
    model = FusionModel.load(args.fusion)
    real = rows(args.real_features, args.real_queries, model)
    synthetic = [r for r in rows(args.synthetic_features, None, model) if r[1]]
    xr = np.array([r[0] for r in real]); yr = np.array([r[1] for r in real])
    xs = np.array([r[0] for r in synthetic]); ys = np.ones(len(synthetic))
    print(f"real: {int(yr.sum())} in catalogue, {int((1 - yr).sum())} not; synthetic positives: {len(xs)}")

    # Leave-one-out over the real photos.
    scores = np.zeros(len(real))
    for i in range(len(real)):
        keep = np.arange(len(real)) != i
        x = np.vstack([xr[keep], xs]); y = np.concatenate([yr[keep], ys])
        w = np.concatenate([class_weight(yr[keep]), np.full(len(xs), args.synthetic_weight)])
        bias, weights = fit(x, y, w)
        scores[i] = 1 / (1 + np.exp(-(bias + xr[i] @ weights)))
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
        flagged_unknown = np.mean(scores[yr == 0] < thr)
        flagged_known = np.mean(scores[yr == 1] < thr)
        print(f"LOO threshold {thr}: not_found for {flagged_unknown:.0%} of off-catalogue photos, "
              f"{flagged_known:.0%} of in-catalogue photos")

    x = np.vstack([xr, xs]); y = np.concatenate([yr, ys])
    w = np.concatenate([class_weight(yr), np.full(len(xs), args.synthetic_weight)])
    bias, weights = fit(x, y, w)
    synthetic_scores = 1 / (1 + np.exp(-(bias + xs @ weights)))
    print(f"field-v1 renders wrongly flagged at 0.5: {np.mean(synthetic_scores < 0.5):.1%}")
    open_set = {"bias": round(float(bias), 5), **{f: round(float(v), 5) for f, v in zip(OPEN_SET_FEATURES, weights)}}
    print(json.dumps(open_set))
    if args.write:
        raw = json.loads(args.fusion.read_text())
        raw["open_set"] = open_set
        raw["open_set_threshold"] = args.threshold
        raw["open_set_note"] = ("logistic on top-candidate evidence; negatives are real photos of wines "
                                "not in the catalogue; LOO-validated (training/fit_open_set.py)")
        args.fusion.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
