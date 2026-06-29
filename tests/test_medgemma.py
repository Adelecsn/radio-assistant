from __future__ import annotations

import json

from src.contracts import WARNING_TEXT, is_valid_prediction
from src.inference import InferenceConfig
from src.inference.medgemma import extract_json_object, parse_medgemma_output


def _config() -> InferenceConfig:
    return InferenceConfig.medgemma()


def _model_json(**overrides: object) -> str:
    payload = {
        "image_quality": "good",
        "predicted_class": "suspected_opacity",
        "confidence_score": 0.82,
        "visual_findings": ["opacité du lobe inférieur droit"],
        "justification": "opacité focale visible",
        "limitations": ["échantillon limité"],
        "warning": WARNING_TEXT,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_extract_json_object_handles_surrounding_text() -> None:
    text = "Voici l'analyse:\n```json\n{\"a\": 1, \"b\": [2, 3]}\n```\nFin."
    assert extract_json_object(text) == {"a": 1, "b": [2, 3]}


def test_parse_valid_output_preserves_clinical_fields() -> None:
    prediction = parse_medgemma_output(_model_json(), _config(), latency_ms=1200)

    assert is_valid_prediction(prediction)
    assert prediction["predicted_class"] == "suspected_opacity"
    assert prediction["visual_findings"] == ["opacité du lobe inférieur droit"]
    assert prediction["warning"] == WARNING_TEXT
    assert prediction["model_version"] == "google/medgemma-4b-it"
    assert prediction["prompt_version"] == "medgemma-v1.0"
    assert prediction["inference_latency_ms"] == 1200


def test_poor_quality_forces_uncertain() -> None:
    prediction = parse_medgemma_output(
        _model_json(image_quality="poor", predicted_class="normal"),
        _config(),
        latency_ms=10,
    )
    assert prediction["predicted_class"] == "uncertain"
    assert is_valid_prediction(prediction)


def test_low_confidence_forces_uncertain() -> None:
    prediction = parse_medgemma_output(
        _model_json(predicted_class="normal", confidence_score=0.4),
        _config(),
        latency_ms=10,
    )
    assert prediction["predicted_class"] == "uncertain"
    assert is_valid_prediction(prediction)


def test_invalid_class_falls_back_to_uncertain() -> None:
    prediction = parse_medgemma_output(
        _model_json(predicted_class="pneumonia"),
        _config(),
        latency_ms=10,
    )
    assert prediction["predicted_class"] == "uncertain"
    assert is_valid_prediction(prediction)


def test_garbage_output_returns_valid_uncertain() -> None:
    prediction = parse_medgemma_output("le modèle a planté, pas de json", _config(), latency_ms=5)

    assert is_valid_prediction(prediction)
    assert prediction["predicted_class"] == "uncertain"
    assert prediction["image_quality"] == "poor"
