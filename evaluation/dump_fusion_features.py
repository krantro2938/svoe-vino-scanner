#!/usr/bin/env python3
"""Dump per-candidate recognition features for a labelled query set.

Runs the production Recognizer ranking (visual shortlist, RootSIFT verification,
text evidence) and records every shortlisted candidate's features plus whether
it is the expected slug. Query embeddings and RootSIFT results are cached per
query, so re-dumping after a text or feature change only computes what is new.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from app.catalog import load_catalog  # noqa: E402
from app.label_text import Word  # noqa: E402
from app.local_features import LocalFeatures  # noqa: E402
from app.pipeline import Recognizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ocr-words", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="default: <output>.cache.jsonl")
    parser.add_argument("--expanded", type=Path, default=ROOT / "data/generated/expanded")
    args = parser.parse_args()
    cache_path = args.cache or args.output.with_suffix(".cache.jsonl")

    catalog = load_catalog(args.expanded / "catalog.jsonl")
    local = LocalFeatures()
    local.load(args.expanded / "local_features.npz", catalog)
    recognizer = Recognizer(catalog, args.expanded / "siglip", None, local)
    words = {}
    for line in args.ocr_words.read_text().splitlines():
        row = json.loads(line)
        words[row["query_id"]] = [Word(*w) for w in row["words"]]
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[row["query_id"]] = row
    rows = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    started = time.perf_counter()
    computed = 0
    with args.output.open("w") as handle:
        for i, row in enumerate(rows):
            entry = cache.setdefault(row["query_id"], {"query_id": row["query_id"], "sift": {}})
            image = None

            def load_image():
                nonlocal image
                if image is None:
                    with Image.open(args.queries.parent / row["image_path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                return image

            if "emb" not in entry:
                embedding = recognizer.visual.embed_query(load_image())
                entry["emb"] = {k: np.round(v, 5).tolist() for k, v in embedding.items()}
            visual = recognizer.visual.scores({k: np.asarray(v, dtype=np.float32) for k, v in entry["emb"].items()})

            def verify(slugs):
                nonlocal computed
                missing = [s for s in slugs if s not in entry["sift"]]
                if missing:
                    computed += len(missing)
                    for slug, value in recognizer.verify(load_image(), missing).items():
                        entry["sift"][slug] = [value["inliers"], value["matches"]]
                return {s: {"inliers": entry["sift"][s][0], "matches": entry["sift"][s][1]}
                        for s in slugs if s in entry["sift"]}

            slugs, features, _ = recognizer.rank(visual, words.get(row["query_id"], []), verify)
            handle.write(json.dumps({
                "query_id": row["query_id"], "expected": row["expected_slug"],
                "candidates": [{"slug": s, "label": int(s == row["expected_slug"]), **f}
                               for s, f in zip(slugs, features)],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)} {time.perf_counter() - started:.0f}s new-sift={computed}", flush=True)
    with cache_path.open("w") as handle:
        for entry in cache.values():
            handle.write(json.dumps(entry) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
