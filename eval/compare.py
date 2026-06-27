"""Compare two prediction directories (baseline vs improved) on the same cases.

Produces a measured, reproducible comparison required by the call for tenders
("comparaison baseline/amélioration chiffrée"): a metrics table with deltas and a
per-case table flagging which variant is right on each labelled case.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .metrics import evaluate_prediction_dir, load_prediction_records

# Headline metrics carried into the comparison table, in display order.
COMPARED_METRICS = (
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall_sensitivity",
    "macro_specificity",
    "json_validity_rate",
    "uncertainty_rate",
    "unfounded_justification_rate",
    "mean_confidence",
    "mean_latency_ms",
)


def _delta(baseline: float | None, improved: float | None) -> float | None:
    if baseline is None or improved is None:
        return None
    return round(improved - baseline, 4)


def _predictions_by_case(predictions_dir: str | Path) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for record in load_prediction_records(predictions_dir):
        case_id = str(record.get("case_id", "")).strip()
        if case_id:
            by_case[case_id] = record
    return by_case


def _dangerous_false_negatives(evaluation: dict[str, Any]) -> int:
    """Opacities confidently cleared as normal: the safety-critical error."""
    confusion = evaluation.get("confusion_matrix", {})
    return int(confusion.get("suspected_opacity", {}).get("normal", 0))


def build_comparison(
    baseline_dir: str | Path,
    improved_dir: str | Path,
) -> dict[str, Any]:
    """Return a side-by-side comparison of two prediction directories."""
    baseline_eval = evaluate_prediction_dir(baseline_dir)
    improved_eval = evaluate_prediction_dir(improved_dir)

    metrics_table = []
    for name in COMPARED_METRICS:
        base_value = baseline_eval.get(name)
        improved_value = improved_eval.get(name)
        metrics_table.append(
            {
                "metric": name,
                "baseline": base_value,
                "improved": improved_value,
                "delta": _delta(base_value, improved_value),
            }
        )

    baseline_pred = _predictions_by_case(baseline_dir)
    improved_pred = _predictions_by_case(improved_dir)

    per_case = []
    changed = 0
    improved_fixed = 0
    improved_broke = 0
    for case_id in sorted(set(baseline_pred) & set(improved_pred)):
        base_record = baseline_pred[case_id]
        improved_record = improved_pred[case_id]
        label = str(base_record.get("label", "")).strip()
        base_class = str(base_record.get("prediction", {}).get("predicted_class", ""))
        improved_class = str(improved_record.get("prediction", {}).get("predicted_class", ""))
        base_correct = bool(label) and base_class == label
        improved_correct = bool(label) and improved_class == label
        case_changed = base_class != improved_class
        if case_changed:
            changed += 1
        if label and not base_correct and improved_correct:
            improved_fixed += 1
        if label and base_correct and not improved_correct:
            improved_broke += 1
        per_case.append(
            {
                "case_id": case_id,
                "label": label,
                "baseline_class": base_class,
                "improved_class": improved_class,
                "changed": case_changed,
                "baseline_correct": base_correct,
                "improved_correct": improved_correct,
            }
        )

    return {
        "baseline_dir": str(baseline_dir),
        "improved_dir": str(improved_dir),
        "baseline_model_version": _model_version(baseline_pred),
        "improved_model_version": _model_version(improved_pred),
        "metrics": metrics_table,
        "safety": {
            "baseline_dangerous_false_negatives": _dangerous_false_negatives(baseline_eval),
            "improved_dangerous_false_negatives": _dangerous_false_negatives(improved_eval),
        },
        "case_changes": {
            "changed": changed,
            "improved_fixed": improved_fixed,
            "improved_broke": improved_broke,
        },
        "per_case": per_case,
    }


def _model_version(by_case: dict[str, dict[str, Any]]) -> str:
    for record in by_case.values():
        version = record.get("prediction", {}).get("model_version")
        if version:
            return str(version)
    return ""


def write_comparison_outputs(
    comparison: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the comparison report (JSON), metrics CSV and per-case CSV."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "comparison_report.json"
    metrics_csv = destination / "comparison_metrics.csv"
    cases_csv = destination / "comparison_cases.csv"

    report_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with metrics_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "baseline", "improved", "delta"])
        writer.writeheader()
        writer.writerows(comparison["metrics"])

    with cases_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_id",
                "label",
                "baseline_class",
                "improved_class",
                "changed",
                "baseline_correct",
                "improved_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison["per_case"])

    return {
        "report_json": report_path,
        "metrics_csv": metrics_csv,
        "cases_csv": cases_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and improved predictions.")
    parser.add_argument("--baseline-dir", default="data/predictions/baseline_v1")
    parser.add_argument("--improved-dir", default="data/predictions/improved_v1")
    parser.add_argument("--output-dir", default="eval/outputs/comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = build_comparison(args.baseline_dir, args.improved_dir)
    outputs = write_comparison_outputs(comparison, args.output_dir)
    accuracy = next(row for row in comparison["metrics"] if row["metric"] == "accuracy")
    macro_f1 = next(row for row in comparison["metrics"] if row["metric"] == "macro_f1")
    print(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "case_changes": comparison["case_changes"],
                "safety": comparison["safety"],
                "report_json": str(outputs["report_json"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
