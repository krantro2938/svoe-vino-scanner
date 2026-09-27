#!/usr/bin/env python3
"""Cache PP-OCRv5 words for a labelled query set (for offline fusion tuning)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "inference"))

from PIL import Image, ImageOps  # noqa: E402

from app.label_text import LabelReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    reader = LabelReader(ROOT / "services/inference/data/ocr")
    rows = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    rows = rows[: args.limit] if args.limit else rows
    done = {}
    if args.output.exists():
        done = {r["query_id"]: r for r in map(json.loads, args.output.read_text().splitlines())}
    with args.output.open("a") as handle:
        for i, row in enumerate(rows):
            if row["query_id"] in done:
                continue
            with Image.open(args.queries.parent / row["image_path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
            started = time.perf_counter()
            words = reader.read(image)
            handle.write(json.dumps({"query_id": row["query_id"], "ms": round((time.perf_counter() - started) * 1000),
                                     "words": [[w.text, w.score, w.weight] for w in words]}, ensure_ascii=False) + "\n")
            handle.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
