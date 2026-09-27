#!/usr/bin/env python3
"""Fit the conditional-logit fusion ranker on dumped candidate features.

Each query is a softmax over its shortlisted candidates; the loss is the
negative log-probability of the expected slug (queries whose expected slug is
not shortlisted cannot be learned from and are only counted in evaluation).
Split is by slug hash, so held-out wines are never seen during fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))
from app.pipeline import FEATURES  # noqa: E402


def holdout(slug: str, fraction: float) -> bool:
    return int(hashlib.sha256(slug.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def load(paths: list[Path]):
    queries = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            x = np.array([[c[f] for f in FEATURES] for c in row["candidates"]], dtype=np.float64)
            y = np.array([c["label"] for c in row["candidates"]])
            queries.append((row["expected"], x, y))
    return queries


VISUAL = [FEATURES.index(f) for f in FEATURES if f.startswith(("v_", "sift_"))]


def mismatch_augment(queries, fraction, seed=20260927):
    """Simulate a shelf label that no longer matches its catalogue photo.

    Real bottles often carry a newer label or vintage than the (small) reference
    image, so the true wine can look *less* similar than a sibling of the same
    series. For a fraction of queries the true candidate's visual and geometric
    features are swapped with a close sibling's; text evidence is unchanged, so
    the ranker must learn when the label text should overrule appearance.
    """
    rng = np.random.default_rng(seed)
    augmented = []
    for slug, x, y in queries:
        if not y.any() or len(y) < 3 or rng.random() >= fraction:
            continue
        positive = int(np.argmax(y))
        by_visual = np.argsort(-x[:, FEATURES.index("v_mean")])
        siblings = [i for i in by_visual[:6] if i != positive]
        j = int(rng.choice(siblings))
        x2 = x.copy()
        x2[positive, VISUAL], x2[j, VISUAL] = x[j, VISUAL], x[positive, VISUAL]
        augmented.append((slug, x2, y))
    return augmented


def fit(queries, l2=1e-3, steps=3000, lr=0.05):
    usable = [(x, y) for _, x, y in queries if y.any()]
    scale = np.concatenate([x for x, _ in usable]).std(axis=0) + 1e-6
    w = np.zeros(len(FEATURES))
    m = np.zeros_like(w); v = np.zeros_like(w)
    for step in range(1, steps + 1):   # Adam on the mean listwise NLL
        grad = l2 * w
        for x, y in usable:
            z = (x / scale) @ w
            p = np.exp(z - z.max()); p /= p.sum()
            grad += ((x / scale).T @ (p - y)) / len(usable)
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad**2
        w -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return w / scale


def evaluate(queries, w, name):
    top1 = top5 = conf_ok = 0
    confident = []
    for _, x, y in queries:
        z = x @ w
        p = np.exp(z - z.max()); p /= p.sum()
        order = np.argsort(-p)
        top1 += bool(y[order[0]]); top5 += bool(y[order[:5]].any())
        confident.append((p[order[0]], bool(y[order[0]])))
    n = len(queries)
    confident.sort(key=lambda t: -t[0])
    high = [ok for pr, ok in confident if pr >= 0.8]
    print(f"{name:14s} n={n} acc@1={top1/n:.3f} R@5={top5/n:.3f} "
          f"p>=0.8: {len(high)/n:.2f} of queries at {np.mean(high) if high else float('nan'):.3f} precision")
    return top1 / n


def open_set(queries, w, reject_rate=0.03):
    """Simulate unknown wines by deleting the true slug from each shortlist.

    Returns the not-found threshold that flags at most `reject_rate` of known
    wines, and the share of simulated unknown wines it catches.
    """
    known, unknown = [], []
    for _, x, y in queries:
        z = x @ w
        p = np.exp(z - z.max()); p /= p.sum()
        known.append(p.max())
        if y.any() and len(y) > 1:
            keep = y == 0
            zu = z[keep]
            pu = np.exp(zu - zu.max()); pu /= pu.sum()
            unknown.append(pu.max())
    threshold = float(np.quantile(known, reject_rate))
    caught = float(np.mean(np.array(unknown) < threshold))
    print(f"open-set: not_found threshold p<{threshold:.3f} flags {reject_rate:.0%} of known wines "
          f"and catches {caught:.1%} of simulated unknown wines (median unknown p={np.median(unknown):.2f})")
    return threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--holdout", type=float, default=0.35)
    parser.add_argument("--final", action="store_true", help="refit on all queries after reporting holdout")
    parser.add_argument("--eval", type=Path, nargs="*", help="separate query set: train on all --features, report here")
    parser.add_argument("--mismatch", type=float, default=0.0, help="fraction of reference-mismatch augmentations")
    args = parser.parse_args()
    queries = load(args.features)
    if args.eval:
        train, test = queries, load(args.eval)
        seen = {q[0] for q in train}
        args.holdout = 0.0
    else:
        train = [q for q in queries if not holdout(q[0], args.holdout)]
        test = [q for q in queries if holdout(q[0], args.holdout)]
    for feature in ("v_mean", "v_max", "sift_inliers", "t_score"):
        w = np.zeros(len(FEATURES)); w[FEATURES.index(feature)] = 50.0
        evaluate(test, w, f"only {feature}")
    w = fit(train + mismatch_augment(train, args.mismatch))
    evaluate(train, w, "fusion train")
    evaluate(mismatch_augment(test, 1.0), w, "eval mismatch")
    evaluate(test, w, "fusion eval" if args.eval else "fusion holdout")
    if args.eval:
        evaluate([q for q in test if q[0] not in seen], w, "eval unseen")
    threshold = open_set(test, w)
    if args.final:
        w = fit(queries)
        evaluate(queries, w, "fusion all")
    if args.output:
        args.output.write_text(json.dumps({
            "model": "conditional-logit-v1",
            "features": list(FEATURES),
            "weights": {f: round(float(v), 6) for f, v in zip(FEATURES, w)},
            "trained_on": [str(p.relative_to(ROOT)) if p.is_absolute() else str(p) for p in args.features],
            "queries": len(queries),
            "holdout_fraction": 0.0 if args.final else args.holdout,
            "not_found_threshold": round(threshold, 4),
        }, indent=2))
    print(json.dumps({f: round(float(v), 3) for f, v in zip(FEATURES, w)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
