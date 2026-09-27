#!/usr/bin/env python3
"""Build deterministic, labeled field-like queries from catalog reference images.

Split assignment happens at connected-group level before augmentation. Rows sharing a
slug, source bytes, or an explicit leakage group can therefore never cross splits,
and changing the augmentation seed cannot move them to another split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    from .evaluate import InputError, _first, read_records
except ImportError:  # Direct script execution.
    from evaluate import InputError, _first, read_records


SLUG_FIELDS = ("expected_slug", "true_slug", "ground_truth_slug", "label", "slug")
IMAGE_FIELDS = ("image_path", "reference_image", "path", "image")
LEAKAGE_FIELDS = ("leakage_group", "source_group")
HARD_NEGATIVE_FIELDS = ("hard_negative_group", "label_family", "confusion_group")
ALLOWED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Source:
    row_number: int
    image_path: str
    absolute_path: Path
    slug: str
    sha256: str
    declared_split: str | None
    leakage_groups: tuple[str, ...]
    hard_negative_group: str | None
    provenance: dict[str, Any]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, *, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"row {row_number}: {field} must be a non-empty string")
    normalized = value.strip().replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise InputError(f"row {row_number}: {field} must be a safe relative path")
    return normalized


def _optional_text(row: dict[str, Any], fields: Iterable[str]) -> str | None:
    value = _first(row, fields)
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_sources(manifest: Path, images_dir: Path) -> list[Source]:
    """Load and validate references without decoding their image content."""
    rows = read_records(manifest)
    if not rows:
        raise InputError(f"{manifest}: reference manifest has no rows")
    root = images_dir.resolve()
    sources: list[Source] = []
    digest_labels: dict[str, str] = {}
    seen_rows: dict[tuple[str, str], int] = {}
    for row_number, row in enumerate(rows, 2):
        image_path = _safe_relative(_first(row, IMAGE_FIELDS), field="image_path", row_number=row_number)
        slug = _optional_text(row, SLUG_FIELDS)
        if slug is None:
            raise InputError(f"row {row_number}: no ground-truth slug was found")
        absolute = (images_dir / image_path).resolve()
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise InputError(f"row {row_number}: image_path escapes images directory") from exc
        if not absolute.is_file():
            raise InputError(f"row {row_number}: reference image not found: {absolute}")
        digest = _sha256(absolute)
        declared_digest = _optional_text(row, ("source_sha256",))
        if declared_digest is not None and digest != declared_digest.lower():
            raise InputError(f"row {row_number}: source_sha256 mismatch for {image_path}")
        old_label = digest_labels.setdefault(digest, slug)
        if old_label != slug:
            raise InputError(
                f"row {row_number}: identical image bytes have conflicting slugs: {old_label!r} and {slug!r}"
            )
        identity = (digest, slug)
        declared_split = _optional_text(row, ("split",))
        if declared_split is not None:
            declared_split = declared_split.lower()
            if declared_split not in ALLOWED_SPLITS:
                raise InputError(f"row {row_number}: split must be one of {', '.join(ALLOWED_SPLITS)}")
        leakage_groups = tuple(
            f"{field}:{value.strip()}"
            for field in LEAKAGE_FIELDS
            if isinstance((value := _first(row, (field,))), str) and value.strip()
        )
        candidate = Source(
            row_number=row_number,
            image_path=image_path,
            absolute_path=absolute,
            slug=slug,
            sha256=digest,
            declared_split=declared_split,
            leakage_groups=leakage_groups,
            hard_negative_group=_optional_text(row, HARD_NEGATIVE_FIELDS),
            provenance={key: row[key] for key in ("source_url", "source_archive_path", "source_kind", "retrieved_at") if key in row},
        )
        if identity in seen_rows:
            index = seen_rows[identity]
            previous = sources[index]
            if previous.declared_split and declared_split and previous.declared_split != declared_split:
                raise InputError(f"row {row_number}: duplicate reference has conflicting declared splits")
            sources[index] = replace(previous,
                declared_split=previous.declared_split or declared_split,
                leakage_groups=tuple(sorted(set(previous.leakage_groups + leakage_groups))))
        else:
            seen_rows[identity] = len(sources)
            sources.append(candidate)
    return sources


def parse_split_weights(specification: str) -> dict[str, float]:
    """Parse ``validation=0.5,test=0.5`` and require an exact probability mass."""
    weights: dict[str, float] = {}
    try:
        for item in specification.split(","):
            name, raw_weight = item.split("=", 1)
            name = name.strip().lower()
            weight = float(raw_weight)
            if name not in ALLOWED_SPLITS or name in weights or not math.isfinite(weight) or weight < 0:
                raise ValueError
            weights[name] = weight
    except ValueError as exc:
        raise InputError("splits must look like validation=0.5,test=0.5 with unique supported names") from exc
    if not weights or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise InputError("split weights must sum to 1.0")
    return weights


def assign_splits(sources: list[Source], weights: dict[str, float], *, split_seed: str) -> list[str]:
    """Assign connected leakage components to splits, independently of augmentation seed."""
    groups = _DisjointSet(len(sources))
    seen_keys: dict[str, int] = {}
    for index, source in enumerate(sources):
        # Slugs and byte digests are always isolation keys. An explicit leakage
        # group can additionally keep a producer/label family in one split.
        keys = (f"slug:{source.slug}", f"sha256:{source.sha256}", *source.leakage_groups)
        for key in keys:
            if key in seen_keys:
                groups.union(index, seen_keys[key])
            else:
                seen_keys[key] = index

    components: dict[int, list[int]] = {}
    for index in range(len(sources)):
        components.setdefault(groups.find(index), []).append(index)
    assignments = [""] * len(sources)
    ordered_weights = [(name, weight) for name, weight in weights.items() if weight > 0]
    for indices in components.values():
        declared = {sources[index].declared_split for index in indices if sources[index].declared_split}
        if len(declared) > 1:
            rows = ", ".join(str(sources[index].row_number) for index in indices)
            raise InputError(f"connected leakage group has conflicting declared splits (rows {rows})")
        if declared:
            split = declared.pop()
            if split not in weights or weights[split] == 0:
                raise InputError(f"declared split {split!r} has no positive configured weight")
        else:
            component_key = min(
                [f"slug:{sources[index].slug}" for index in indices]
                + [f"sha256:{sources[index].sha256}" for index in indices]
                + [group for index in indices for group in sources[index].leakage_groups]
            )
            value = int.from_bytes(
                hashlib.sha256(f"{split_seed}\0{component_key}".encode()).digest()[:8], "big"
            ) / 2**64
            cumulative = 0.0
            split = ordered_weights[-1][0]
            for name, weight in ordered_weights:
                cumulative += weight
                if value < cumulative:
                    split = name
                    break
        for index in indices:
            assignments[index] = split
    return assignments


def _variant_parameters(source: Source, augmentation_seed: str, variant: int) -> dict[str, Any]:
    seed_material = f"{augmentation_seed}\0{source.sha256}\0{source.slug}\0{variant}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:16], "big")
    rng = random.Random(seed)
    return {
        "angle": round(rng.uniform(-8.0, 8.0), 3),
        "crop_fraction": round(rng.uniform(0.01, 0.09), 4),
        "brightness": round(rng.uniform(0.72, 1.28), 4),
        "contrast": round(rng.uniform(0.78, 1.30), 4),
        "blur_radius": round(rng.uniform(0.0, 1.15), 3),
        "occlusion_fraction": round(rng.uniform(0.0, 0.075), 4),
        "occlusion_side": rng.choice(("left", "right", "top", "bottom")),
        "jpeg_quality": rng.randint(68, 91),
        "perspective_x": round(rng.uniform(-0.065, 0.065), 4),
        "perspective_y": round(rng.uniform(-0.025, 0.025), 4),
        "glare_alpha": rng.randint(12, 46),
        "glare_center": round(rng.uniform(0.25, 0.75), 4),
        "glare_width": round(rng.uniform(0.04, 0.13), 4),
    }


@lru_cache(maxsize=64)
def _open_reference(path: Path, source_sha256: str):
    from PIL import Image, ImageOps
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        return image.convert("RGBA")


def _render(source: Source, output: Path, parameters: dict[str, Any]) -> tuple[int, int]:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise InputError(
            "Pillow is required to render synthetic queries; use the inference environment or install Pillow"
        ) from exc

    rgba = _open_reference(source.absolute_path, source.sha256)
    image = Image.new("RGB", rgba.size, (238, 235, 228))
    image.paste(rgba, mask=rgba.getchannel("A"))
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    width, height = image.size
    if width < 16 or height < 16:
        raise InputError(f"reference image is too small to transform: {source.image_path}")
    crop_x = min(width // 4, max(1, round(width * parameters["crop_fraction"])))
    crop_y = min(height // 4, max(1, round(height * parameters["crop_fraction"])))
    image = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    image = image.rotate(parameters["angle"], resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(238, 235, 228))
    w, h = image.size
    px, py = w * parameters["perspective_x"], h * parameters["perspective_y"]
    image = image.transform(image.size, Image.Transform.QUAD,
                            (px, py, -px, h - py, w + px, h + py, w - px, -py),
                            resample=Image.Resampling.BICUBIC, fillcolor=(238, 235, 228))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glare = ImageDraw.Draw(overlay)
    center = w * parameters["glare_center"]
    radius = w * parameters["glare_width"]
    glare.polygon(((center - radius, 0), (center + radius, 0),
                   (center + radius + w * 0.18, h), (center - radius + w * 0.18, h)),
                  fill=(255, 255, 255, parameters["glare_alpha"]))
    image = Image.alpha_composite(image.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(max(1, w * 0.025)))).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(parameters["brightness"])
    image = ImageEnhance.Contrast(image).enhance(parameters["contrast"])
    if parameters["blur_radius"] >= 0.15:
        image = image.filter(ImageFilter.GaussianBlur(parameters["blur_radius"]))
    fraction = parameters["occlusion_fraction"]
    if fraction >= 0.01:
        draw = ImageDraw.Draw(image, "RGBA")
        transformed_width, transformed_height = image.size
        side = parameters["occlusion_side"]
        if side == "left":
            box = (0, 0, round(transformed_width * fraction), transformed_height)
        elif side == "right":
            box = (round(transformed_width * (1 - fraction)), 0, transformed_width, transformed_height)
        elif side == "top":
            box = (0, 0, transformed_width, round(transformed_height * fraction))
        else:
            box = (0, round(transformed_height * (1 - fraction)), transformed_width, transformed_height)
        draw.rectangle(box, fill=(18, 18, 18, 105))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=parameters["jpeg_quality"], optimize=False, progressive=False)
    return image.size


def _render_shelf(source: Source, distractors: list[Source], output: Path,
                  parameters: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    """Composite distinct bottles around the centred target, preserving alpha masks.

    Bounding boxes describe rendered reference extents (not hand-labelled bottle masks).
    Distractors are chosen within the target's split by the caller.
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

    width, height = 640, 640
    canvas = Image.new("RGB", (width, height), (218, 207, 190))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 565, width, height), fill=(133, 103, 78))
    annotations = []
    # Draw target last and give every bottle a disjoint horizontal slot.
    placements = [(other, x, False) for other, x in zip(distractors, (99, 541))]
    placements.append((source, 320, True))
    for reference, center_x, target in placements:
        bottle = _open_reference(reference.absolute_path, reference.sha256).copy()
        alpha = bottle.getchannel("A")
        if alpha.getextrema()[0] < 255:
            bounds = alpha.getbbox()
            if bounds:
                bottle = bottle.crop(bounds)
        bottle.thumbnail((190, 480 if target else 515), Image.Resampling.LANCZOS)
        x, y = center_x - bottle.width // 2, 565 - bottle.height
        canvas.paste(bottle, (x, y), bottle)
        annotations.append({
            "slug": reference.slug, "is_target": target,
            "bbox_xyxy": [x, y, x + bottle.width, y + bottle.height],
            "source_image_path": reference.image_path,
            "source_sha256": reference.sha256,
            "provenance": reference.provenance,
        })
    canvas = ImageEnhance.Brightness(canvas).enhance(parameters["brightness"])
    canvas = ImageEnhance.Contrast(canvas).enhance(parameters["contrast"])
    if parameters["blur_radius"] >= 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(parameters["blur_radius"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=parameters["jpeg_quality"])
    return width, height, annotations


def _choose_distractors(source: Source, pool: list[Source], seed: str, variant: int) -> list[Source]:
    # Family matches are the difficult negatives; never reuse target or another split.
    candidates = [other for other in pool if other.slug != source.slug]
    candidates.sort(key=lambda other: (
        not (source.hard_negative_group and source.hard_negative_group == other.hard_negative_group),
        hashlib.sha256(f"{seed}\0{source.sha256}\0{variant}\0{other.sha256}".encode()).hexdigest(),
    ))
    chosen = []
    for other in candidates:
        if other.slug not in {item.slug for item in chosen}:
            chosen.append(other)
        if len(chosen) == 2:
            break
    return chosen


def generate_dataset(
    reference_manifest: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    variants_per_image: int = 3,
    augmentation_seed: str = "1",
    split_seed: str = "cifr-v1",
    split_weights: dict[str, float] | None = None,
    shelf_variants_per_image: int = 0,
) -> dict[str, Any]:
    """Render a synthetic dataset and return its machine-readable build summary."""
    if shelf_variants_per_image < 0:
        raise InputError("shelf_variants_per_image must be non-negative")
    if variants_per_image <= 0:
        raise InputError("variants_per_image must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InputError(f"output directory is not empty: {output_dir}")
    weights = split_weights or {"validation": 0.5, "test": 0.5}
    sources = load_sources(reference_manifest, images_dir)
    assignments = assign_splits(sources, weights, split_seed=split_seed)
    pools = {name: [source for source, split in zip(sources, assignments) if split == name] for name in weights}
    if shelf_variants_per_image:
        for split, pool in pools.items():
            if pool and len({source.slug for source in pool}) < 3:
                raise InputError(f"split {split!r} needs at least three distinct slugs for shelf scenes")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "queries.jsonl"
    records: list[dict[str, Any]] = []
    for source, split in zip(sources, assignments):
        for variant in range(variants_per_image + shelf_variants_per_image):
            is_shelf = variant >= variants_per_image
            parameters = _variant_parameters(source, augmentation_seed, variant)
            identity = hashlib.sha256(
                f"v3\0{source.sha256}\0{source.slug}\0{augmentation_seed}\0{variant}\0{is_shelf}".encode()
            ).hexdigest()[:20]
            query_id = f"syn-{identity}"
            relative = f"images/{split}/{query_id}.jpg"
            objects = []
            if is_shelf:
                distractors = _choose_distractors(source, pools[split], augmentation_seed, variant)
                if len(distractors) < 2:
                    raise InputError(f"split {split!r} needs at least three distinct slugs for shelf scenes")
                width, height, objects = _render_shelf(source, distractors, output_dir / relative, parameters)
                parameters = {key: parameters[key] for key in ("brightness", "contrast", "blur_radius", "jpeg_quality")}
            else:
                width, height = _render(source, output_dir / relative, parameters)
            record: dict[str, Any] = {
                "query_id": query_id,
                "sha256": _sha256(output_dir / relative),
                "generator_version": 3,
                "image_path": relative,
                "expected_slug": source.slug,
                "split": split,
                "source_image_path": source.image_path,
                "source_sha256": source.sha256,
                "augmentation_seed": str(augmentation_seed),
                "variant": variant,
                "transform": parameters,
                "width": width,
                "height": height,
                "scene_type": "center_target_three_bottles" if is_shelf else "single_reference",
                "is_synthetic": True,
                "provenance": source.provenance,
                "objects": objects,
                "hard_negative_slugs": [item["slug"] for item in objects if not item["is_target"]],
            }
            if source.leakage_groups:
                record["leakage_group"] = source.leakage_groups[0].split(":", 1)[1]
            if source.hard_negative_group:
                record["hard_negative_group"] = source.hard_negative_group
            records.append(record)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 3,
        "reference_manifest": str(reference_manifest),
        "reference_count": len(sources),
        "query_count": len(records),
        "augmentation_seed": str(augmentation_seed),
        "split_seed": str(split_seed),
        "variants_per_image": variants_per_image,
        "shelf_variants_per_image": shelf_variants_per_image,
        "scene_counts": {kind: sum(record["scene_type"] == kind for record in records) for kind in ("single_reference", "center_target_three_bottles")},
        "reference_manifest_sha256": _sha256(reference_manifest),
        "evaluation_caveat": "Synthetic catalogue derivatives are not independent real-world accuracy evidence. Use held-out real phone photos for final evaluation.",
        "splits": {name: sum(record["split"] == name for record in records) for name in weights},
        "manifest": str(manifest_path),
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path, help="labeled CSV/TSV/JSON(L) references")
    parser.add_argument("--images-dir", required=True, type=Path, help="root for reference image_path values")
    parser.add_argument("--output-dir", required=True, type=Path, help="new or empty dataset directory")
    parser.add_argument("--variants", type=int, default=3, help="queries per unique reference image")
    parser.add_argument("--shelf-variants", type=int, default=0, help="additional centred-target three-bottle scenes per reference")
    parser.add_argument("--augmentation-seed", default="1", help="changes transforms but never split membership")
    parser.add_argument("--split-seed", default="cifr-v1", help="changes group-level split assignment")
    parser.add_argument("--splits", default="validation=0.5,test=0.5", help="comma-separated split weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_dataset(
            args.references,
            args.images_dir,
            args.output_dir,
            variants_per_image=args.variants,
            shelf_variants_per_image=args.shelf_variants,
            augmentation_seed=args.augmentation_seed,
            split_seed=args.split_seed,
            split_weights=parse_split_weights(args.splits),
        )
    except (InputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
