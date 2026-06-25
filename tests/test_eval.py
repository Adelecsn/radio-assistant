from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from eval.metrics import evaluate_prediction_dir, evaluate_records, write_evaluation_outputs


def _record(case_id: str, label: str, predicted: str) -> dict:
    return {
        "case_id": case_id,
        "split": "test",
        "label": label,
        "prediction": {
            "predicted_class": predicted,
            "confidence_score": 0.8,
            "image_quality": "good",
            "prompt_version": "v1.0",
            "model_version": "baseline-test",
        },
    }


def test_evaluate_records_computes_metrics_and_errors() -> None:
    evaluation = evaluate_records(
        [
            _record("case_1", "normal", "normal"),
            _record("case_2", "suspected_opacity", "uncertain"),
            _record("case_3", "uncertain", "uncertain"),
        ]
    )

    assert evaluation["evaluated_cases"] == 3
    assert evaluation["accuracy"] == 0.6667
    assert evaluation["confusion_matrix"]["suspected_opacity"]["uncertain"] == 1
    assert evaluation["per_class"]["normal"]["recall_sensitivity"] == 1.0
    assert evaluation["errors"][0]["error_type"] == "suspected_opacity_as_uncertain"


def test_write_evaluation_outputs(tmp_path: Path) -> None:
    evaluation = evaluate_records(
        [
            _record("case_1", "normal", "suspected_opacity"),
            _record("case_2", "uncertain", "uncertain"),
        ]
    )

    outputs = write_evaluation_outputs(evaluation, tmp_path / "outputs")
    report = json.loads(outputs["report_json"].read_text(encoding="utf-8"))
    with outputs["error_register_csv"].open(encoding="utf-8", newline="") as stream:
        errors = list(csv.DictReader(stream))

    assert report["evaluated_cases"] == 2
    assert errors[0]["case_id"] == "case_1"
    assert errors[0]["status"] == "commented_needs_review"
    assert errors[0]["reviewer_comment"]


def test_evaluate_cli_runs_on_prediction_directory(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    cases = predictions / "cases"
    cases.mkdir(parents=True)
    (cases / "case_1.json").write_text(
        json.dumps(_record("case_1", "normal", "normal")),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.evaluate",
            "--predictions-dir",
            str(predictions),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["evaluated_cases"] == 1
    assert evaluate_prediction_dir(predictions)["accuracy"] == 1.0
    assert (output_dir / "metrics_report.json").exists()
