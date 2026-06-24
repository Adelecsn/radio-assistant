"""Image loading and deterministic preprocessing without clinical inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom import dcmread

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


SUPPORTED_SUFFIXES = {".dcm", ".dicom", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    original_format: str
    original_width: int
    original_height: int
    burned_in_annotation: str
    technical_review: str


def _to_uint8(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite pixel value")

    valid = values[finite]
    low, high = np.percentile(valid, (0.5, 99.5))
    if high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        raise ValueError("Image has no usable intensity range")

    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    values = np.clip(values, low, high)
    values = (values - low) / (high - low)
    return np.rint(values * 255).astype(np.uint8)


def _technical_review(width: int, height: int) -> str:
    if width < 256 or height < 256:
        return "review_required_low_resolution"
    return "pass_basic_checks"


def _load_dicom(path: Path) -> LoadedImage:
    dataset = dcmread(path)
    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError("Only single-frame grayscale DICOM images are supported")

    pixels = apply_modality_lut(pixels, dataset)
    try:
        pixels = apply_voi_lut(pixels, dataset)
    except (IndexError, TypeError, ValueError):
        # Window metadata is optional; percentile normalization remains deterministic.
        pass

    normalized = _to_uint8(pixels)
    if str(dataset.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        normalized = 255 - normalized

    height, width = normalized.shape
    burned_in = str(dataset.get("BurnedInAnnotation", "UNKNOWN")).upper()
    if burned_in not in {"YES", "NO"}:
        burned_in = "UNKNOWN"

    return LoadedImage(
        image=Image.fromarray(normalized),
        original_format="dicom",
        original_width=width,
        original_height=height,
        burned_in_annotation=burned_in,
        technical_review=_technical_review(width, height),
    )


def _load_raster(path: Path) -> LoadedImage:
    with Image.open(path) as source:
        source.load()
        if source.mode in {"I", "I;16", "I;16L", "I;16B", "F"}:
            normalized = _to_uint8(np.asarray(source))
        else:
            normalized = np.asarray(source.convert("L"), dtype=np.uint8)

    height, width = normalized.shape
    return LoadedImage(
        image=Image.fromarray(normalized),
        original_format=path.suffix.lower().lstrip("."),
        original_width=width,
        original_height=height,
        burned_in_annotation="UNKNOWN",
        technical_review=_technical_review(width, height),
    )


def load_image(path: str | Path) -> LoadedImage:
    """Load a supported image without retaining source metadata."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {suffix or '<none>'}")
    if suffix in {".dcm", ".dicom"}:
        return _load_dicom(source)
    return _load_raster(source)


def resize_with_padding(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize while preserving aspect ratio and pad without cropping anatomy."""
    if size <= 0:
        raise ValueError("Target size must be positive")

    resized = image.copy()
    resized.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), color=0)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas
