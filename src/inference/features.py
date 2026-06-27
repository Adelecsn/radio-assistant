"""Deterministic image feature extraction for the baseline model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageFeatures:
    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    foreground_ratio: float
    dark_ratio: float
    bright_ratio: float
    central_bright_ratio: float
    horizontal_asymmetry: float
    edge_density: float
    local_texture_max: float

    def to_dict(self) -> dict[str, int | float]:
        """Return rounded JSON-serializable feature values."""
        return asdict(self)


def _local_texture_max(values: np.ndarray, grid: int = 6) -> float:
    """Highest local contrast among central patches.

    A localized consolidation (opacity) is a homogeneous region with low local
    contrast, while healthy lung keeps high-contrast vascular markings. Reporting
    the maximum patch contrast over the central lung field gives an auxiliary,
    deterministic signal that global statistics average away.
    """
    height, width = values.shape
    center = values[height // 6 : (5 * height) // 6, width // 6 : (5 * width) // 6]
    patch_height = max(center.shape[0] // grid, 1)
    patch_width = max(center.shape[1] // grid, 1)
    contrasts: list[float] = []
    for row in range(grid):
        for col in range(grid):
            patch = center[
                row * patch_height : (row + 1) * patch_height,
                col * patch_width : (col + 1) * patch_width,
            ]
            foreground = patch[patch > 5]
            if foreground.size >= 10:
                contrasts.append(float(foreground.std()))
    if not contrasts:
        return 0.0
    return max(contrasts)


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.any():
        return float(values[mask].mean())
    return 0.0


def extract_image_features(
    image_path: str | Path,
    *,
    bright_pixel_threshold: int = 205,
    edge_threshold: float = 16.0,
) -> ImageFeatures:
    """Extract non-clinical grayscale statistics from a preprocessed image."""
    with Image.open(image_path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32)

    if values.ndim != 2:
        raise ValueError("Inference expects a single grayscale image")

    height, width = values.shape
    foreground_mask = values > 5
    foreground = values[foreground_mask] if foreground_mask.any() else values.ravel()

    center = values[height // 5 : (height * 4) // 5, width // 5 : (width * 4) // 5]
    center_mask = center > 5
    center_foreground = center[center_mask] if center_mask.any() else center.ravel()

    left = values[:, : width // 2]
    right = values[:, width - width // 2 :]
    left_mean = _masked_mean(left, left > 5)
    right_mean = _masked_mean(right, right > 5)

    gradient_x = np.abs(np.diff(values, axis=1))
    gradient_y = np.abs(np.diff(values, axis=0))
    edge_density = (
        float((gradient_x >= edge_threshold).mean())
        + float((gradient_y >= edge_threshold).mean())
    ) / 2

    return ImageFeatures(
        width=int(width),
        height=int(height),
        mean_intensity=_rounded(float(foreground.mean())),
        std_intensity=_rounded(float(foreground.std())),
        foreground_ratio=_rounded(float(foreground_mask.mean())),
        dark_ratio=_rounded(float((foreground <= 30).mean())),
        bright_ratio=_rounded(float((foreground >= bright_pixel_threshold).mean())),
        central_bright_ratio=_rounded(float((center_foreground >= bright_pixel_threshold).mean())),
        horizontal_asymmetry=_rounded(abs(left_mean - right_mean) / 255),
        edge_density=_rounded(edge_density),
        local_texture_max=_rounded(_local_texture_max(values)),
    )
