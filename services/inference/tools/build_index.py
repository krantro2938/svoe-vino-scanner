from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from app.catalog import load_catalog
from app.features import DESCRIPTOR_VERSION, extract_descriptors
from app.imaging import prepare_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the dependency-free CPU visual index.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    items = []
    missing = []
    for wine in catalog.values():
        if not wine.index_ready:
            missing.append(wine.slug)
            continue
        if not wine.photo_name:
            missing.append(wine.slug)
            continue
        image_path = args.images_dir / wine.photo_name
        if not image_path.is_file():
            missing.append(wine.slug)
            continue
        try:
            with Image.open(image_path) as source:
                image = prepare_rgb(source)
                descriptors = [descriptor.tolist() for descriptor in extract_descriptors(image)]
        except (OSError, ValueError):
            missing.append(wine.slug)
            continue
        items.append({"slug": wine.slug, "descriptors": descriptors})

    manifest = {
        "model_version": "cpu-visual-baseline-v1",
        "descriptor_version": DESCRIPTOR_VERSION,
        "catalog_count": len(catalog),
        "indexed_count": len(items),
        "missing_slugs": missing,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"indexed {len(items)}/{len(catalog)} wines; missing {len(missing)}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
