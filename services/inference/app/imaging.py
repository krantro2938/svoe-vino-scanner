from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageInputError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class DecodedImage:
    image: Image.Image
    format: str
    width: int
    height: int


def center_wine_view(image: Image.Image) -> Image.Image:
    """Limit wide scenes to the central bottle-sized column.

    This is a framing prior, not an object detector. Keep portrait bottle/label
    photos intact; wide shelf photos must not let a larger side bottle dominate
    either OCR or visual retrieval. Preserve full height for neck labels.
    """
    width, height = image.size
    if width / height <= 0.85:
        return image
    # Validation-only sweep on centred three-bottle scenes selected 0.26:
    # it keeps the middle label while excluding both neighbours. The previous
    # 0.70 crop admitted large parts of the side bottles and diluted retrieval.
    crop_width = max(1, round(height * 0.26))
    left = (width - crop_width) // 2
    return image.crop((left, 0, left + crop_width, height))


def prepare_rgb(source: Image.Image) -> Image.Image:
    """Apply the exact same orientation and alpha policy for indexing and queries."""
    oriented = ImageOps.exif_transpose(source)
    oriented.load()
    if oriented.mode == "RGB":
        return oriented.copy()
    converted = Image.new("RGB", oriented.size, "white")
    if "A" in oriented.getbands():
        converted.paste(oriented, mask=oriented.getchannel("A"))
    else:
        converted.paste(oriented.convert("RGB"))
    return converted


def decode_image(
    payload: bytes,
    *,
    max_pixels: int,
    min_side: int,
) -> DecodedImage:
    if not payload:
        raise ImageInputError("empty_image", "The uploaded image is empty.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                source = Image.open(io.BytesIO(payload))
                detected_format = source.format or "unknown"
                source.verify()
                source = Image.open(io.BytesIO(payload))
                oriented = prepare_rgb(source)
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
                raise ImageInputError(
                    "invalid_image", "The upload is not a supported, decodable image."
                ) from error
            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                raise ImageInputError(
                    "image_too_large",
                    f"The decoded image exceeds the {max_pixels:,}-pixel safety limit.",
                    413,
                ) from error

        width, height = oriented.size
        if width * height > max_pixels:
            raise ImageInputError(
                "image_too_large",
                f"The decoded image exceeds the {max_pixels:,}-pixel safety limit.",
                413,
            )
        if min(width, height) < min_side:
            raise ImageInputError(
                "image_too_small",
                f"Both image dimensions must be at least {min_side} pixels.",
            )
        return DecodedImage(oriented, detected_format, width, height)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
