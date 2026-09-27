"""Image views shared by gallery indexing and query encoding.

Catalogue references are studio shots of the whole bottle; field photos are
close-ups of the label. Both sides are therefore encoded as several views and
compared view-to-view, so a label close-up can meet the label region of its
reference instead of a tiny whole-bottle thumbnail.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

GALLERY_VIEWS = ("bottle", "label")
QUERY_VIEWS = ("full", "centre")


def bottle_cutout(image: Image.Image, max_side: int = 1400) -> Image.Image:
    """Cut the bottle out of a flat studio background (RGBA, tight crop)."""
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        if (alpha < 250).mean() > 0.05:
            box = Image.fromarray(alpha).point(lambda v: 255 if v > 16 else 0).getbbox()
            return rgba.crop(box) if box else rgba
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    distance = np.abs(rgb - np.median(border, axis=0)).sum(axis=2)
    mask = distance > 36
    # A bottle is convex row-wise: fill between the outermost hits so dark
    # glass on a dark background does not leave holes.
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


def square(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def gallery_views(reference: Image.Image) -> dict[str, Image.Image]:
    cut = bottle_cutout(ImageOps.exif_transpose(reference))
    flat = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    flat.alpha_composite(cut)
    bottle = flat.convert("RGB")
    width, height = bottle.size
    top = int(height * 0.42)   # label band of a standing bottle
    label = bottle.crop((0, top, width, min(height, top + int(width * 1.3))))
    return {"bottle": square(bottle, (255, 255, 255)), "label": square(label, (255, 255, 255))}


def query_views(image: Image.Image) -> dict[str, Image.Image]:
    """Whole frame plus a centred label-sized crop (targets sit in the middle)."""
    width, height = image.size
    crop_w, crop_h = int(width * 0.8), int(height * 0.6)
    left, top = (width - crop_w) // 2, int(height * 0.5 - crop_h / 2)
    centre = image.crop((left, top, left + crop_w, top + crop_h))
    return {"full": square(image, (0, 0, 0)), "centre": square(centre, (0, 0, 0))}
