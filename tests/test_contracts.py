from __future__ import annotations

import json
from pathlib import Path

from src.contracts import WARNING_TEXT, is_valid_prediction, validate_prediction


ROOT = Path(__file__).resolve().parents[1]


def valid_payload() -> dict:
    return {
        "image_quality": "good",
        "predicted_class": "normal",
        "confidence_score": 0.81,
        "visual_findings": ["No focal opacity identified by the prototype."],
        "justification": "The output is limited to visible image features.",
        "limitations": ["No clinical context."],
        "warning": WARNING_TEXT,
        "model_version": "baseline-placeholder",
        "prompt_version": "v1.0",
        "inference_latency_ms": 42,
    }


def test_valid_prediction_contract() -> None:
    assert is_valid_prediction(valid_payload())


def test_low_confidence_requires_uncertainty() -> None:
    payload = valid_payload()
    payload["confidence_score"] = 0.42

    assert "confidence below 0.60" in " ".join(validate_prediction(payload))


def test_poor_quality_requires_uncertainty() -> None:
    payload = valid_payload()
    payload["image_quality"] = "poor"

    assert "poor image quality" in " ".join(validate_prediction(payload))


def test_json_schema_matches_python_contract() -> None:
    schema = json.loads((ROOT / "prompts" / "output_schema.json").read_text())

    assert set(schema["required"]) == set(valid_payload())
    assert schema["properties"]["warning"]["const"] == WARNING_TEXT
