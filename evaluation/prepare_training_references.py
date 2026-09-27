#!/usr/bin/env python3
"""Build verified reference rows, quarantining missing/ambiguous/conflicting images."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def prepare(catalog: Path, images_dir: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    candidates = []
    rejected = []
    root = images_dir.resolve()
    for row in rows:
        slug = row.get("slug")
        image_path = row.get("photo_name") or row.get("local_path")
        reason = None
        if not slug or not image_path:
            reason = "missing_slug_or_image_path"
        elif row.get("index_ready") is False:
            reason = "not_index_ready"
        else:
            path = (root / image_path).resolve()
            if not path.is_relative_to(root):
                reason = "unsafe_image_path"
            elif not path.is_file():
                reason = "missing_image"
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        if min(image.size) < 16:
                            reason = "image_too_small"
                except (OSError, ValueError):
                    reason = "unreadable_image"
        if reason:
            rejected.append({"slug": slug, "image_path": image_path, "reason": reason})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        family = str(row.get("winery") or "").strip()
        # Producer-level grouping prevents nearly identical labels crossing splits.
        candidates.append({
            "slug": slug, "image_path": image_path, "source_sha256": digest,
            "hard_negative_group": family or slug,
            "leakage_group": family or slug,
            "source_kind": row.get("source_kind", "organizer_catalog"),
            "source_archive_path": row.get("media_path", ""),
            **{key: row[key] for key in ("source_url", "retrieved_at") if row.get(key)},
        })
    labels = defaultdict(set)
    for row in candidates:
        labels[row["source_sha256"]].add(row["slug"])
    clean = []
    seen = set()
    for row in candidates:
        if len(labels[row["source_sha256"]]) > 1:
            rejected.append({"slug": row["slug"], "image_path": row["image_path"], "reason": "identical_bytes_conflicting_labels"})
        elif (row["slug"], row["source_sha256"]) not in seen:
            seen.add((row["slug"], row["source_sha256"]))
            clean.append(row)
    clean.sort(key=lambda row: (row["slug"], row["image_path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean))
    report = {"catalog": str(catalog), "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
              "catalog_rows": len(rows), "references": len(clean), "unique_slugs": len({row["slug"] for row in clean}),
              "rejection_counts": dict(Counter(row["reason"] for row in rejected)), "rejected": rejected}
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "rejected"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.catalog, args.images_dir, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
