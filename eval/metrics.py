"""Evaluation metrics for labeled prediction outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CLASS_ORDER = ("normal", "suspected_opacity", "uncertain")

ERROR_COMMENTS = {
    "normal_as_uncertain": (
        "Baseline prudente: cas normal placé entre les seuils, à relire "
        "visuellement pour confirmer absence de signal pathologique évident."
    ),
    "normal_as_suspected_opacity": (
        "Faux positif probable: signal clair/asymétrie interprété comme opacité "
        "par les seuils statistiques."
    ),
    "suspected_opacity_as_normal": (
        "Faux négatif probable: opacité RSNA peu captée par les features simples, "
        "besoin de modèle ou segmentation plus robuste."
    ),
    "suspected_opacity_as_uncertain": (
        "Signal d'opacité insuffisant ou ambigu pour la baseline, classé uncertain "
        "par prudence."
    ),
    "uncertain_as_normal": (
        "Cas RSNA non normal sans opacité franche: ressemble à un normal pour les "
        "features globales."
    ),
    "uncertain_as_suspected_opacity": (
        "Cas non normal confondu avec opacité: signal clair/asymétrique dépasse "
        "les seuils statistiques."
    ),
}


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round((2 * precision * recall) / (precision + recall), 4)


def _mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return round(sum(defined) / len(defined), 4)


def _record_from_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(payload.get("case_id", "")),
        "split": str(payload.get("split", "")),
        "label": str(payload.get("label", "")),
        "prediction": payload.get("prediction", {}),
    }


def load_prediction_records(predictions_dir: str | Path) -> list[dict[str, Any]]:
    """Load case records from an inference output directory."""
    root = Path(predictions_dir)
    cases_dir = root / "cases"
    if cases_dir.is_dir():
        records = []
        for path in sorted(cases_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(_record_from_case_payload(payload))
        return records

    index_path = root / "predictions.jsonl"
    if not index_path.is_file():
        return []

    records = []
    with index_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            case_json = record.get("case_json")
            if case_json:
                case_path = root / str(case_json)
                if case_path.is_file():
                    records.append(
                        _record_from_case_payload(
                            json.loads(case_path.read_text(encoding="utf-8"))
                        )
                    )
                    continue
            records.append(record)
    return records


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy, macro metrics, confusion matrix and error rows."""
    confusion = {
        ground_truth: {predicted: 0 for predicted in CLASS_ORDER} for ground_truth in CLASS_ORDER
    }
    skipped_unlabeled = 0
    skipped_invalid_label = 0
    evaluated = 0
    correct = 0
    errors: list[dict[str, Any]] = []

    for record in records:
        label = str(record.get("label", "")).strip()
        if not label:
            skipped_unlabeled += 1
            continue
        if label not in CLASS_ORDER:
            skipped_invalid_label += 1
            continue

        prediction = record.get("prediction", {})
        if not isinstance(prediction, dict):
            prediction = {}
        predicted_class = str(prediction.get("predicted_class", "")).strip()
        if predicted_class not in CLASS_ORDER:
            predicted_class = "uncertain"

        confusion[label][predicted_class] += 1
        evaluated += 1
        if label == predicted_class:
            correct += 1
        else:
            error_type = f"{label}_as_{predicted_class}"
            errors.append(
                {
                    "case_id": record.get("case_id", ""),
                    "split": record.get("split", ""),
                    "ground_truth": label,
                    "predicted_class": predicted_class,
                    "confidence_score": prediction.get("confidence_score"),
                    "image_quality": prediction.get("image_quality"),
                    "prompt_version": prediction.get("prompt_version"),
                    "model_version": prediction.get("model_version"),
                    "error_type": error_type,
                    "reviewer_comment": ERROR_COMMENTS.get(
                        error_type,
                        "Erreur à relire visuellement.",
                    ),
                    "status": "commented_needs_review",
                }
            )

    per_class: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_ORDER:
        tp = confusion[class_name][class_name]
        support = sum(confusion[class_name].values())
        predicted_count = sum(confusion[truth][class_name] for truth in CLASS_ORDER)
        fp = predicted_count - tp
        fn = support - tp
        precision = _safe_divide(tp, predicted_count)
        recall = _safe_divide(tp, support)
        per_class[class_name] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall_sensitivity": recall,
            "f1": _f1(precision, recall),
        }

    macro_precision = _mean([metrics["precision"] for metrics in per_class.values()])
    macro_recall = _mean([metrics["recall_sensitivity"] for metrics in per_class.values()])
    macro_f1 = _mean([metrics["f1"] for metrics in per_class.values()])

    return {
        "classes": list(CLASS_ORDER),
        "evaluated_cases": evaluated,
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_invalid_label": skipped_invalid_label,
        "accuracy": _safe_divide(correct, evaluated),
        "macro_precision": macro_precision,
        "macro_recall_sensitivity": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "errors": errors,
    }


def evaluate_prediction_dir(predictions_dir: str | Path) -> dict[str, Any]:
    """Evaluate all labeled predictions in an inference output directory."""
    return evaluate_records(load_prediction_records(predictions_dir))


def write_evaluation_outputs(
    evaluation: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON report and CSV error register from evaluation results."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "metrics_report.json"
    errors_path = destination / "error_register.csv"

    report_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "case_id",
        "split",
        "ground_truth",
        "predicted_class",
        "confidence_score",
        "error_type",
        "image_quality",
        "prompt_version",
        "model_version",
        "reviewer_comment",
        "status",
    ]
    with errors_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluation.get("errors", []))

    return {"report_json": report_path, "error_register_csv": errors_path}
