"""Inference hyperparameters kept outside the model code."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the deterministic image-statistics baseline."""

    model_version: str = "image-stat-baseline-v0.2"
    prompt_version: str = "v1.0"
    confidence_threshold: float = 0.60
    opacity_threshold: float = 0.72
    normal_threshold: float = 0.72
    poor_quality_std_threshold: float = 12.0
    min_foreground_ratio: float = 0.20
    bright_pixel_threshold: int = 205
    edge_threshold: float = 16.0
    normal_asymmetry_max: float = 0.07
    normal_central_bright_max: float = 0.36
    normal_edge_density_max: float = 0.026
    opacity_asymmetry_min: float = 0.22
    opacity_central_bright_min: float = 0.28
    opacity_edge_density_min: float = 0.032

    def __post_init__(self) -> None:
        ratio_fields = {
            "confidence_threshold": self.confidence_threshold,
            "opacity_threshold": self.opacity_threshold,
            "normal_threshold": self.normal_threshold,
            "min_foreground_ratio": self.min_foreground_ratio,
            "normal_asymmetry_max": self.normal_asymmetry_max,
            "normal_central_bright_max": self.normal_central_bright_max,
            "normal_edge_density_max": self.normal_edge_density_max,
            "opacity_asymmetry_min": self.opacity_asymmetry_min,
            "opacity_central_bright_min": self.opacity_central_bright_min,
            "opacity_edge_density_min": self.opacity_edge_density_min,
        }
        for name, value in ratio_fields.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.poor_quality_std_threshold < 0:
            raise ValueError("poor_quality_std_threshold must be non-negative")
        if not 0 <= self.bright_pixel_threshold <= 255:
            raise ValueError("bright_pixel_threshold must be between 0 and 255")
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold must be non-negative")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable hyperparameters for run metadata."""
        return asdict(self)
