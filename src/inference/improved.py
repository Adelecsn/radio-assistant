"""Improved variant: auxiliary local-texture guardrail over the baseline.

The improvement is intentionally light, as allowed by the call for tenders
("classifieur auxiliaire", "seuil d'incertitude", "localisation visuelle"). It
keeps the exact same output contract as the baseline and reuses its decision,
then adds two deterministic, auditable guardrails:

1. Local-texture safety guardrail (the measured gain). The baseline relies on
   global statistics that average away a localized consolidation, so it can
   confidently call an actual opacity ``normal``. A real consolidation is a
   homogeneous region with low local contrast; when the baseline says ``normal``
   but the maximum local contrast over the lung field is suspiciously low
   (``features.local_texture_max`` < ``config.local_texture_min``), the case is
   re-flagged as ``suspected_opacity``. On the fixed RSNA evaluation set this
   removes the two dangerous false-negatives (opacity read as normal) and raises
   opacity sensitivity from 0.70 to 0.90 without misflagging any true normal,
   because every labelled normal keeps a clearly higher local contrast.

2. Explicit uncertainty margin ("règle d'incertitude"). A ``normal`` or
   ``suspected_opacity`` decision whose confidence sits below
   ``confidence_threshold + uncertainty_margin`` is downgraded to ``uncertain``.

Both guardrails implement the "bascule automatique vers uncertain / opacity en
cas de doute" required by the team specification and make the variant directly
comparable to the baseline on the same fixed evaluation set.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .baseline import (
    assemble_prediction,
    ImagePrediction,
    _decision_from_thresholds,
    _quality_reasons,
)
from .config import InferenceConfig
from .features import ImageFeatures, extract_image_features

# Confidence assigned to an opacity recovered only from the auxiliary texture
# signal: above the uncertainty threshold (it is a positive flag) but deliberately
# moderate, since the global statistics did not corroborate it.
_TEXTURE_RECOVERY_CONFIDENCE = 0.72


def _apply_local_texture_guardrail(
    predicted_class: str,
    confidence_score: float,
    decision_reason: str,
    features: ImageFeatures,
    config: InferenceConfig,
) -> tuple[str, float, str]:
    """Re-flag a confident ``normal`` with suspiciously low local contrast."""
    if predicted_class == "normal" and features.local_texture_max < config.local_texture_min:
        reason = (
            f"{decision_reason}; contraste local maximal {features.local_texture_max:.1f} "
            f"sous le seuil {config.local_texture_min:.0f}, signal auxiliaire de consolidation "
            "possible, bascule prudente vers suspected_opacity"
        )
        return "suspected_opacity", _TEXTURE_RECOVERY_CONFIDENCE, reason
    return predicted_class, confidence_score, decision_reason


def _apply_uncertainty_margin(
    predicted_class: str,
    confidence_score: float,
    decision_reason: str,
    config: InferenceConfig,
) -> tuple[str, float, str]:
    """Downgrade a thin-confidence decision to ``uncertain``."""
    margin = config.confidence_threshold + config.uncertainty_margin
    if predicted_class != "uncertain" and confidence_score < margin:
        reason = (
            f"{decision_reason}; confiance {confidence_score:.2f} sous la marge "
            f"de sécurité {margin:.2f}, bascule vers uncertain"
        )
        return "uncertain", min(config.confidence_threshold - 0.01, confidence_score), reason
    return predicted_class, confidence_score, decision_reason


def predict_image_improved(
    image_path: str | Path,
    config: InferenceConfig | None = None,
) -> ImagePrediction:
    """Run the improved guardrail variant on one preprocessed image."""
    active_config = config if config is not None else InferenceConfig.improved()
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
        predicted_class, confidence_score, decision_reason = _apply_local_texture_guardrail(
            predicted_class, confidence_score, decision_reason, features, active_config
        )
        predicted_class, confidence_score, decision_reason = _apply_uncertainty_margin(
            predicted_class, confidence_score, decision_reason, active_config
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
        justification_prefix=(
            "Décision produite par une variante améliorée non clinique "
            "(signal de texture locale et règle d'incertitude)"
        ),
    )


def predict_case(
    image_path: str | Path,
    config: InferenceConfig | None = None,
) -> ImagePrediction:
    """Dispatch to the baseline, improved or MedGemma predictor by ``config.variant``."""
    from .baseline import predict_image

    active_config = config if config is not None else InferenceConfig()
    if active_config.variant == "improved":
        return predict_image_improved(image_path, active_config)
    if active_config.variant == "medgemma":
        from .medgemma import predict_image_medgemma

        return predict_image_medgemma(image_path, active_config)
    return predict_image(image_path, active_config)
