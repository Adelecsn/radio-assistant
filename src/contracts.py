"""Shared output contract used by inference, evaluation and the web app."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


WARNING_TEXT = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)
ALLOWED_QUALITIES = {"good", "poor"}
ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_FIELDS = {
    "image_quality",
    "predicted_class",
    "confidence_score",
    "visual_findings",
    "justification",
    "limitations",
    "warning",
    "model_version",
    "prompt_version",
    "inference_latency_ms",
}


def validate_prediction(payload: Mapping[str, Any]) -> list[str]:
    """Return contract errors without mutating the model output."""
    errors: list[str] = []
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")

    quality = payload.get("image_quality")
    predicted_class = payload.get("predicted_class")
    confidence = payload.get("confidence_score")

    if quality not in ALLOWED_QUALITIES:
        errors.append("image_quality must be 'good' or 'poor'")
    if predicted_class not in ALLOWED_CLASSES:
        errors.append("predicted_class is not allowed")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        errors.append("confidence_score must be numeric")
    elif not 0 <= float(confidence) <= 1:
        errors.append("confidence_score must be between 0 and 1")
    elif float(confidence) < 0.60 and predicted_class != "uncertain":
        errors.append("confidence below 0.60 requires predicted_class='uncertain'")

    if quality == "poor" and predicted_class != "uncertain":
        errors.append("poor image quality requires predicted_class='uncertain'")
    if payload.get("warning") != WARNING_TEXT:
        errors.append("warning is missing or differs from the required text")

    for field in ("visual_findings", "limitations"):
        value = payload.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            errors.append(f"{field} must be a list of non-empty strings")

    for field in ("justification", "model_version", "prompt_version"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    latency = payload.get("inference_latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        errors.append("inference_latency_ms must be a non-negative integer")

    return errors


def is_valid_prediction(payload: Mapping[str, Any]) -> bool:
    """Return whether a prediction follows the shared output contract."""
    return not validate_prediction(payload)
