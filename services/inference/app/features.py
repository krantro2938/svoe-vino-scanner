from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


DESCRIPTOR_VERSION = "pillow-grid-color-edge-v1"


def _single_descriptor(image: Image.Image) -> np.ndarray:
    # Preserve absolute label colors: per-channel autocontrast amplifies tiny JPEG
    # artifacts in otherwise flat areas and can reverse the nearest neighbour.
    rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    values = np.asarray(rgb, dtype=np.float32) / 255.0

    color_parts: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(values[:, :, channel], bins=16, range=(0.0, 1.0))
        color_parts.append(histogram.astype(np.float32) / values[:, :, channel].size)

    grid_parts: list[np.ndarray] = []
    for y in range(4):
        for x in range(4):
            cell = values[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16]
            grid_parts.extend((cell.mean(axis=(0, 1)), cell.std(axis=(0, 1))))

    edges = np.asarray(rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_grid = edges.reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
    descriptor = np.concatenate((*color_parts, *grid_parts, edge_grid)).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm else descriptor


def extract_descriptors(image: Image.Image) -> list[np.ndarray]:
    """Return full-frame and center-crop descriptors without modifying the source."""
    width, height = image.size
    variants = [image]
    for fraction in (0.80, 0.62):
        crop_width, crop_height = int(width * fraction), int(height * fraction)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        variants.append(image.crop((left, top, left + crop_width, top + crop_height)))
    return [_single_descriptor(variant) for variant in variants]


def image_quality(image: Image.Image) -> tuple[float, str | None]:
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    contrast = float(gray.std())
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    sharpness = float((horizontal + vertical) / 2.0)
    score = min(1.0, (contrast / 38.0) * 0.55 + (sharpness / 15.0) * 0.45)
    if contrast < 8:
        return score, "Add light and keep the front label visible."
    if sharpness < 3:
        return score, "Hold the camera steady and move closer to the label."
    return score, None
