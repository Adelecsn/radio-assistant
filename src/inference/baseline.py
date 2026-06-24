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

    opacity_score, normal_score = _scores(features, active_config)
    if image_quality == "poor":
        predicted_class = "uncertain"
        confidence_score = min(active_config.confidence_threshold - 0.01, 0.45)
        decision_reason = "; ".join(quality_reasons)
    elif opacity_score >= active_config.opacity_threshold:
        predicted_class = "suspected_opacity"
        confidence_score = _clamp(opacity_score, active_config.confidence_threshold, 0.88)
        decision_reason = "score d'opacité supérieur au seuil baseline"
    elif normal_score >= active_config.normal_threshold:
        predicted_class = "normal"
        confidence_score = _clamp(normal_score, active_config.confidence_threshold, 0.86)
        decision_reason = "score normal supérieur au seuil baseline"
    else:
        predicted_class = "uncertain"
        confidence_score = min(active_config.confidence_threshold - 0.01, 0.59)
        decision_reason = "scores baseline trop proches des seuils"

    latency_ms = int(round((perf_counter() - started) * 1000))
    prediction: dict[str, Any] = {
        "image_quality": image_quality,
        "predicted_class": predicted_class,
        "confidence_score": round(float(confidence_score), 3),
        "visual_findings": _findings(predicted_class, image_quality, features),
        "justification": (
            "Décision produite par une baseline statistique non clinique: "
            f"{decision_reason}."
        ),
        "limitations": [
            "Baseline sans segmentation anatomique, sans contexte clinique et sans entraînement médical.",
            "Résultat destiné uniquement à tester le workflow logiciel.",
        ],
        "warning": WARNING_TEXT,
        "model_version": active_config.model_version,
        "prompt_version": active_config.prompt_version,
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
