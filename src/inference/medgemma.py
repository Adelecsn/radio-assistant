"""Pluggable MedGemma vision-language backend behind the shared output contract.

This is the multimodal baseline targeted by the call for tenders. It slots into
the existing pipeline without changing the JSON contract: MedGemma proposes the
clinical fields, and this module *enforces* the contract on top of the raw model
output (literal warning, our versions/latency, and the mandatory fall-back to
``uncertain`` when the JSON is invalid, the quality is poor or the confidence is
too low).

The heavy dependencies (``torch`` / ``transformers``) are imported lazily inside
the model functions, so importing this module, running the statistical variants
and the test suite never require a GPU or the gated model weights. Only the parts
that genuinely need MedGemma load it, and a model load/setup failure raises a
clear ``RuntimeError`` instead of silently rejecting every case.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from src.contracts import (
    ALLOWED_CLASSES,
    ALLOWED_QUALITIES,
    WARNING_TEXT,
    validate_prediction,
)

from .baseline import ImagePrediction
from .config import InferenceConfig

# Cache loaded models/processors by (model_id, device) to avoid reloading the
# ~8 GB weights for every image in a batch run.
_BACKEND_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

_USER_INSTRUCTION = (
    "Analyse cette radiographie thoracique frontale et réponds uniquement par "
    "l'objet JSON demandé, sans aucun texte autour."
)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object found in free model text."""
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _string_list(value: Any, fallback: str) -> list[str]:
    if isinstance(value, list):
        items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if items:
            return items
    return [fallback]


def _fallback_uncertain(config: InferenceConfig, latency_ms: int, reason: str) -> dict[str, Any]:
    """A guaranteed contract-valid abstention when the model output is unusable."""
    return {
        "image_quality": "poor",
        "predicted_class": "uncertain",
        "confidence_score": round(min(config.confidence_threshold - 0.01, 0.45), 3),
        "visual_findings": ["Sortie du modèle inexploitable : aucune observation retenue."],
        "justification": f"Bascule de sécurité vers uncertain : {reason}.",
        "limitations": [
            "Sortie de modèle multimodal non clinique, à vérifier par un professionnel.",
        ],
        "warning": WARNING_TEXT,
        "model_version": config.model_version,
        "prompt_version": config.prompt_version,
        "inference_latency_ms": latency_ms,
    }


def parse_medgemma_output(
    text: str,
    config: InferenceConfig,
    latency_ms: int,
) -> dict[str, Any]:
    """Coerce raw MedGemma text into a contract-valid prediction (testable, no GPU).

    The clinical fields come from the model; the warning, versions and latency are
    set by us. Uncertainty rules are enforced: invalid JSON, poor quality or
    confidence below the threshold all force ``uncertain``.
    """
    payload = extract_json_object(text)
    if payload is None:
        return _fallback_uncertain(config, latency_ms, "JSON absent ou non parsable")

    quality = str(payload.get("image_quality", "")).strip().lower()
    if quality not in ALLOWED_QUALITIES:
        quality = "poor"

    predicted = str(payload.get("predicted_class", "")).strip().lower()
    if predicted not in ALLOWED_CLASSES:
        predicted = "uncertain"

    try:
        confidence = float(payload.get("confidence_score"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    justification = payload.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        justification = "Justification non fournie par le modèle."

    # Mandatory uncertainty guardrails.
    if quality == "poor" and predicted != "uncertain":
        predicted = "uncertain"
    if confidence < config.confidence_threshold and predicted != "uncertain":
        predicted = "uncertain"

    prediction = {
        "image_quality": quality,
        "predicted_class": predicted,
        "confidence_score": round(confidence, 3),
        "visual_findings": _string_list(
            payload.get("visual_findings"),
            "Aucune observation visuelle exploitable fournie par le modèle.",
        ),
        "justification": justification.strip(),
        "limitations": _string_list(
            payload.get("limitations"),
            "Sortie de modèle multimodal non clinique, à vérifier.",
        ),
        "warning": WARNING_TEXT,
        "model_version": config.model_version,
        "prompt_version": config.prompt_version,
        "inference_latency_ms": latency_ms,
    }

    if validate_prediction(prediction):
        return _fallback_uncertain(config, latency_ms, "sortie non conforme au contrat")
    return prediction


def _load_backend(config: InferenceConfig) -> tuple[Any, Any]:
    """Lazily load and cache the MedGemma processor and model."""
    key = (config.medgemma_model_id, config.device)
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "MedGemma nécessite torch + transformers (+ accelerate). "
            "Installe requirements.txt puis connecte-toi à Hugging Face."
        ) from error

    try:
        processor = AutoProcessor.from_pretrained(config.medgemma_model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            config.medgemma_model_id,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
        )
    except Exception as error:  # noqa: BLE001 - surface any load/auth failure clearly
        raise RuntimeError(
            f"Chargement de MedGemma '{config.medgemma_model_id}' impossible : {error}. "
            "Vérifie l'acceptation de la licence et ton HF_TOKEN."
        ) from error

    _BACKEND_CACHE[key] = (processor, model)
    return processor, model


def _run_model(image_path: str | Path, system_prompt: str, config: InferenceConfig) -> str:
    """Run MedGemma on one image and return its raw text output."""
    from PIL import Image

    processor, model = _load_backend(config)
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _USER_INSTRUCTION},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]
    generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)
    return processor.decode(generated[0][input_length:], skip_special_tokens=True)


def predict_image_medgemma(
    image_path: str | Path,
    config: InferenceConfig | None = None,
) -> ImagePrediction:
    """Run the MedGemma backend on one image, enforcing the output contract."""
    active_config = config if config is not None else InferenceConfig.medgemma()
    system_prompt = Path(active_config.prompt_path).read_text(encoding="utf-8")

    started = perf_counter()
    try:
        raw_text = _run_model(image_path, system_prompt, active_config)
        latency_ms = int(round((perf_counter() - started) * 1000))
        prediction = parse_medgemma_output(raw_text, active_config, latency_ms)
    except RuntimeError:
        # Setup/auth/load failure: propagate so the user sees the real cause.
        raise
    except Exception as error:  # noqa: BLE001 - per-image failure must not crash a batch
        latency_ms = int(round((perf_counter() - started) * 1000))
        prediction = _fallback_uncertain(active_config, latency_ms, f"erreur d'inférence ({error})")

    return ImagePrediction(
        prediction=prediction,
        features={"backend": "medgemma", "model_id": active_config.medgemma_model_id},
        quality_reasons=[],
    )
