#!/usr/bin/env python3
"""Render field-like close-up label photos from catalogue reference images.

The v3 synthetic set keeps the studio framing of the references (whole bottle,
flat background), which makes it far easier than real shelf photos. Public field
photos are portrait phone shots where the label fills most of the frame, the
label is curved around the bottle, neighbours are cut off at the edges, and there
is glare, colour cast and blur. This generator reproduces those properties so
that recognition changes can be compared on a harder, closer-to-real benchmark.

It is still synthetic: numbers measured here are a proxy, never a claim about
the organiser's private set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

GENERATOR_VERSION = "field-closeup-v1"


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def bottle_cutout(path: Path, max_side: int = 1400) -> Image.Image:
    """Return an RGBA cut-out of the bottle, cropped to its bounding box."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        if (alpha < 250).mean() > 0.05:
            box = Image.fromarray(alpha).point(lambda v: 255 if v > 16 else 0).getbbox()
            return rgba.crop(box) if box else rgba
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).sum(axis=2)
    mask = distance > 36
    # Close small gaps (dark glass on a dark background) column-wise: a bottle is
    # a vertically convex object, so fill between the first and last hit per row.
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1)
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    columns = np.arange(mask.shape[1])[None, :]
    filled = any_hit[:, None] & (columns >= first[:, None]) & (columns <= last[:, None])
    alpha = Image.fromarray((filled * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    box = alpha.getbbox()
    return rgba.crop(box) if box else rgba


def cylinder_warp(image: Image.Image, strength: float) -> Image.Image:
    """Compress the sides like a label wrapped around a cylinder, with shading."""
    array = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height, width = array.shape[:2]
    xs = np.linspace(-1.0, 1.0, width)
    # Output column u samples source column sin(u*theta)/sin(theta).
    theta = strength * np.pi / 2
    source = np.sin(xs * theta) / np.sin(theta)
    columns = np.clip(((source + 1) / 2 * (width - 1)).round().astype(int), 0, width - 1)
    warped = array[:, columns]
    shade = 0.55 + 0.45 * np.cos(xs * theta) ** 0.8
    warped[..., :3] *= shade[None, :, None]
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))


def background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    kind = rng.choice(["dark_shelf", "fridge", "table", "store"])
    base = {
        "dark_shelf": (rng.randint(15, 45),) * 3,
        "fridge": (rng.randint(150, 210), rng.randint(160, 215), rng.randint(170, 225)),
        "table": (rng.randint(120, 180), rng.randint(85, 130), rng.randint(50, 90)),
        "store": (rng.randint(60, 140), rng.randint(60, 140), rng.randint(60, 140)),
    }[kind]
    canvas = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(canvas)
    for _ in range(rng.randint(8, 30)):
        x0, y0 = rng.randint(-width // 4, width), rng.randint(-height // 4, height)
        x1, y1 = x0 + rng.randint(width // 10, width // 2), y0 + rng.randint(height // 20, height // 3)
        shift = rng.randint(-50, 50)
        colour = tuple(max(0, min(255, c + shift + rng.randint(-12, 12))) for c in base)
        draw.rectangle((x0, y0, x1, y1), fill=colour)
    if kind in {"dark_shelf", "store"}:
        for _ in range(rng.randint(1, 3)):  # shelf rails
            y = rng.randint(height // 2, height)
            draw.rectangle((0, y, width, y + rng.randint(8, 30)), fill=(rng.randint(120, 200),) * 3)
    return canvas.filter(ImageFilter.GaussianBlur(rng.uniform(6, 18)))


def glare(image: Image.Image, rng: random.Random, bottle_box: tuple[int, int, int, int]) -> Image.Image:
    overlay = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = bottle_box
    for _ in range(rng.randint(1, 3)):
        cx = rng.uniform(left + 0.15 * (right - left), right - 0.15 * (right - left))
        w = rng.uniform(0.02, 0.08) * (right - left)
        draw.ellipse((cx - w, top + rng.uniform(0, 0.3) * (bottom - top), cx + w,
                      bottom - rng.uniform(0, 0.3) * (bottom - top)), fill=rng.randint(90, 220))
    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(4, 16)))
    white = Image.new("RGB", image.size, (255, 255, 250))
    return Image.composite(white, image, overlay)


def place(canvas: Image.Image, bottle: Image.Image, centre_x: float, label_y: float,
          body_width: float, rng: random.Random, curvature: float) -> tuple[int, int, int, int]:
    """Scale so the bottle body spans body_width pixels, centre it, return its box."""
    scale = body_width / bottle.width
    resized = bottle.resize((max(1, round(bottle.width * scale)), max(1, round(bottle.height * scale))),
                            Image.Resampling.LANCZOS)
    resized = cylinder_warp(resized, curvature)
    angle = rng.uniform(-6, 6)
    resized = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    x = round(centre_x - resized.width / 2)
    # Labels sit in the lower ~55% of a bottle; anchor that area to label_y.
    y = round(label_y - resized.height * 0.66)
    canvas.paste(resized, (x, y), resized)
    return x, y, x + resized.width, y + resized.height


def render(target: Path, neighbours: list[Path], seed: str, size=(960, 1280)) -> tuple[Image.Image, dict]:
    rng = _rng(seed)
    width, height = size
    canvas = background(size, rng)
    fill = rng.uniform(0.55, 0.95)          # bottle body width as fraction of frame
    body = width * fill
    curvature = rng.uniform(0.35, 0.8)
    for side, path in zip((-1, 1), neighbours):
        other = bottle_cutout(path)
        offset = rng.uniform(0.62, 0.9) * body + body * 0.5
        place(canvas, other, width / 2 + side * offset, height * rng.uniform(0.5, 0.7),
              body * rng.uniform(0.85, 1.1), rng, curvature)
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    bottle = bottle_cutout(target)
    box = place(canvas, bottle, width / 2 + rng.uniform(-0.06, 0.06) * width,
                height * rng.uniform(0.4, 0.58), body, rng, curvature)
    # Perspective: mild keystone via a quad transform.
    k = rng.uniform(0, 0.06) * width
    quad = (rng.uniform(-k, k), rng.uniform(-k, k), rng.uniform(-k, k), height + rng.uniform(-k, k),
            width + rng.uniform(-k, k), height + rng.uniform(-k, k), width + rng.uniform(-k, k), rng.uniform(-k, k))
    canvas = canvas.transform(size, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC,
                             fillcolor=tuple(int(v) for v in np.asarray(canvas).reshape(-1, 3).mean(axis=0)))
    if rng.random() < 0.8:
        canvas = glare(canvas, rng, box)
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.55, 1.25))
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.7, 1.2))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.7, 1.3))
    r, g, b = canvas.split()   # white-balance cast (tungsten/fluorescent)
    warm = rng.uniform(-0.12, 0.12)
    r = r.point(lambda v: min(255, v * (1 + warm)))
    b = b.point(lambda v: min(255, v * (1 - warm)))
    canvas = Image.merge("RGB", (r, g, b))
    blur = rng.choice(["none", "gauss", "motion"])
    if blur == "gauss":
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.2)))
    elif blur == "motion":
        n = rng.choice([3, 5])
        kernel = [0.0] * (n * n)
        for i in range(n):
            kernel[(n // 2) * n + i] = 1.0 / n
        canvas = canvas.filter(ImageFilter.Kernel((n, n), kernel))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1, 7), (height, width, 3))
    canvas = Image.fromarray(np.clip(np.asarray(canvas, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    ys, xs = np.mgrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    falloff = 1.0 - rng.uniform(0.15, 0.45) * np.clip((xs**2 + ys**2) / 2.0, 0, 1)
    canvas = Image.fromarray(np.clip(np.asarray(canvas, dtype=np.float32) * falloff[..., None], 0, 255).astype(np.uint8))
    return canvas, {"fill": round(fill, 3), "curvature": round(curvature, 3), "blur": blur}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/generated/expanded/catalog.jsonl"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/generated/expanded/images"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/artifacts/field-v1"))
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", default="field-v1")
    parser.add_argument("--slugs", type=Path, help="optional file with one slug per line to render")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.catalog.read_text().splitlines()]
    rows = [r for r in rows if r.get("index_ready") and (args.images_dir / r["photo_name"]).is_file()]
    rng = _rng(args.seed, "selection")
    if args.slugs:
        wanted = set(args.slugs.read_text().split())
        chosen = [r for r in rows if r["slug"] in wanted]
    else:
        chosen = rng.sample(rows, min(args.count, len(rows)))
    (args.output / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(sorted(chosen, key=lambda r: r["slug"])):
        local = _rng(args.seed, row["slug"])
        others = local.sample([r for r in rows if r["slug"] != row["slug"]], 2)
        image, params = render(args.images_dir / row["photo_name"],
                               [args.images_dir / r["photo_name"] for r in others],
                               f"{args.seed}|{row['slug']}")
        name = f"images/{hashlib.sha256((args.seed + row['slug']).encode()).hexdigest()[:16]}.jpg"
        image.save(args.output / name, quality=local.randint(70, 92))
        records.append({"query_id": f"field-{index:05d}", "image_path": name, "expected_slug": row["slug"],
                        "neighbour_slugs": [r["slug"] for r in others], "generator_version": GENERATOR_VERSION,
                        **params})
        if (index + 1) % 100 == 0:
            print(f"rendered {index + 1}/{len(chosen)}", flush=True)
    with (args.output / "queries.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output / "queries.tsv").open("w") as handle:
        handle.write("query_id\timage_path\n")
        for record in records:
            handle.write(f"{record['query_id']}\t{record['image_path']}\n")
    print(json.dumps({"count": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
