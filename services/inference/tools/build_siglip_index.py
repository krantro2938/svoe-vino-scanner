#!/usr/bin/env python3
"""Embed every index-ready catalogue reference with the ONNX image encoder.

Writes gallery.npz (one L2-normalised row per slug and gallery view) and
gallery.json (checksums tying the gallery to the encoder and catalogue).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.encoder import ImageEncoder, slug_fingerprint  # noqa: E402
from app.views import GALLERY_VIEWS, gallery_views  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("index_ready") and (args.images_dir / r["photo_name"]).is_file()]
    encoder = ImageEncoder(args.encoder_dir, threads=10)

    def load(row):
        with Image.open(args.images_dir / row["photo_name"]) as image:
            image.load()
            return gallery_views(image)

    started = time.time()
    embeddings = {name: [] for name in GALLERY_VIEWS}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(rows), args.batch):
            views = list(pool.map(load, rows[offset: offset + args.batch]))
            for name in GALLERY_VIEWS:
                embeddings[name].append(encoder.encode([v[name] for v in views]))
            if (offset // args.batch) % 20 == 0:
                print(f"{offset + len(views)}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    gallery = args.encoder_dir / "gallery.npz"
    slugs = [r["slug"] for r in rows]
    np.savez(gallery, slugs=np.array(slugs), **{k: np.concatenate(v).astype(np.float16) for k, v in embeddings.items()})
    (args.encoder_dir / "gallery.json").write_text(json.dumps({
        "schema_version": 1,
        "views": list(GALLERY_VIEWS),
        "count": len(slugs),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "catalog_slug_fingerprint": slug_fingerprint(slugs),
        "model_sha256": encoder.config["model_sha256"],
        "gallery_sha256": hashlib.sha256(gallery.read_bytes()).hexdigest(),
    }, indent=2))
    print(f"indexed {len(slugs)} references in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
