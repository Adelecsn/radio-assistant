"""Conservative deterministic baseline for the inference workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.contracts import WARNING_TEXT, validate_prediction

from .config import InferenceConfig
from .features import ImageFeatures, extract_image_features


@dataclass(frozen=True)
class ImagePrediction:
    """Prediction plus internal evidence saved by the batch pipeline."""

    prediction: dict[str, Any]
    features: dict[str, int | float]
    quality_reasons: list[str]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _quality_reasons(features: ImageFeatures, config: InferenceConfig) -> list[str]:
    reasons: list[str] = []
    if features.std_intensity < config.poor_quality_std_threshold:
        reasons.append("contraste global insuffisant")
    if features.foreground_ratio < config.min_foreground_ratio:
        reasons.append("zone utile trop faible après prétraitement")
    return reasons


def _scores(features: ImageFeatures, config: InferenceConfig) -> tuple[float, float]:
    bright_signal = _clamp(
        features.central_bright_ratio / max(config.opacity_threshold / 4, 0.01),
        0,
        1,
    )
    asymmetry_signal = _clamp(features.horizontal_asymmetry / 0.18, 0, 1)
    texture_signal = _clamp(features.std_intensity / 80, 0, 1)

    opacity_score = (0.45 * bright_signal) + (0.35 * asymmetry_signal) + (0.20 * texture_signal)
    normal_score = 1 - ((0.65 * bright_signal) + (0.35 * asymmetry_signal))
    return opacity_score, normal_score


def _decision_from_thresholds(
    features: ImageFeatures,
    config: InferenceConfig,
) -> tuple[str, float, str]:
    """Return class, confidence and reason from explicit baseline thresholds."""
    normal_match = (
        features.horizontal_asymmetry <= config.normal_asymmetry_max
        and features.central_bright_ratio <= config.normal_central_bright_max
        and features.edge_density <= config.normal_edge_density_max
    )
    if normal_match:
        asymmetry_margin = config.normal_asymmetry_max - features.horizontal_asymmetry
        bright_margin = config.normal_central_bright_max - features.central_bright_ratio
        edge_margin = config.normal_edge_density_max - features.edge_density
        confidence = config.confidence_threshold + min(
            0.24,
            0.8 * asymmetry_margin + 0.25 * bright_margin + 3.0 * edge_margin,
        )
        return (
            "normal",
            _clamp(confidence, config.confidence_threshold, 0.84),
            "asymétrie, signal clair central et contours sous les seuils normaux",
        )

    opacity_match = (
        features.horizontal_asymmetry >= config.opacity_asymmetry_min
        or features.central_bright_ratio >= config.opacity_central_bright_min
        or features.edge_density >= config.opacity_edge_density_min
    )
    if opacity_match:
        asymmetry_signal = _clamp(features.horizontal_asymmetry / config.opacity_asymmetry_min, 0, 1)
        bright_signal = _clamp(
            features.central_bright_ratio / config.opacity_central_bright_min,
            0,
            1,
        )
        edge_signal = _clamp(features.edge_density / config.opacity_edge_density_min, 0, 1)
        confidence = config.confidence_threshold + (0.28 * max(asymmetry_signal, bright_signal, edge_signal))
        return (
            "suspected_opacity",
            _clamp(confidence, config.confidence_threshold, 0.88),
            "au moins un seuil de signal clair, asymétrie ou contours est dépassé",
        )

    return (
        "uncertain",
        min(config.confidence_threshold - 0.01, 0.59),
        "mesures situées entre les seuils normal et opacité",
    )


def _findings(predicted_class: str, quality: str, features: ImageFeatures) -> list[str]:
    findings = [
        f"Contraste mesuré: écart-type {features.std_intensity:.1f}.",
        f"Zone utile estimée: {features.foreground_ratio:.1%} de l'image.",
    ]
    if quality == "poor":
        findings.append("La qualité mesurée ne permet pas une conclusion automatique fiable.")
    elif predicted_class == "suspected_opacity":
        findings.append("La baseline détecte une concentration centrale de pixels clairs.")
    elif predicted_class == "normal":
        findings.append("Aucun signal clair dominant selon les seuils de la baseline.")
    else:
        findings.append("Les mesures restent ambiguës pour cette baseline.")
    return findings


def assemble_prediction(
    *,
    image_quality: str,
    predicted_class: str,
    confidence_score: float,
    decision_reason: str,
    features: ImageFeatures,
    quality_reasons: list[str],
    config: InferenceConfig,
    latency_ms: int,
    justification_prefix: str,
) -> ImagePrediction:
    """Build a contract-valid prediction shared by the baseline and improved variants."""
    prediction: dict[str, Any] = {
        "image_quality": image_quality,
        "predicted_class": predicted_class,
        "confidence_score": round(float(confidence_score), 3),
        "visual_findings": _findings(predicted_class, image_quality, features),
        "justification": f"{justification_prefix}: {decision_reason}.",
        "limitations": [
            "Baseline sans segmentation anatomique, sans contexte clinique et sans entraînement médical.",
            "Résultat destiné uniquement à tester le workflow logiciel.",
        ],
        "warning": WARNING_TEXT,
        "model_version": config.model_version,
        "prompt_version": config.prompt_version,
        "inference_latency_ms": latency_ms,
    }

    errors = validate_prediction(prediction)
    if errors:
        raise ValueError(f"Internal inference output does not match the contract: {errors}")

    return ImagePrediction(
        prediction=prediction,
        features=features.to_dict(),
        quality_reasons=quality_reasons,
    )


def predict_image(
    image_path: str | Path,
    config: InferenceConfig | None = None,
) -> ImagePrediction:
    """Run the baseline on one preprocessed image and return a valid prediction."""
    active_config = config or InferenceConfig()
    started = perf_counter()
    features = extract_image_features(
        image_path,
        bright_pixel_threshold=active_config.bright_pixel_threshold,
        edge_threshold=active_config.edge_threshold,
    )
    quality_reasons = _quality_reasons(features, active_config)
    image_quality = "poor" if quality_reasons else "good"

    if image_quality == "poor":
        predicted_class = "uncertain"
        confidence_score = min(active_config.confidence_threshold - 0.01, 0.45)
        decision_reason = "; ".join(quality_reasons)
    else:
        predicted_class, confidence_score, decision_reason = _decision_from_thresholds(
            features,
            active_config,
        )

    latency_ms = int(round((perf_counter() - started) * 1000))
    return assemble_prediction(
        image_quality=image_quality,
        predicted_class=predicted_class,
        confidence_score=confidence_score,
        decision_reason=decision_reason,
        features=features,
        quality_reasons=quality_reasons,
        config=active_config,
        latency_ms=latency_ms,
        justification_prefix="Décision produite par une baseline statistique non clinique",
    )
