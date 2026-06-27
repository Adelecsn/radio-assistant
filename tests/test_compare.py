from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.compare import build_comparison, write_comparison_outputs
from src.inference import InferenceConfig, run_batch_inference


def _smooth_low_texture_image(path: Path) -> None:
    rows, cols = np.indices((512, 512))
    radius = np.sqrt((rows - 256) ** 2 + (cols - 256) ** 2)
    values = 70 + np.clip(radius / 256, 0, 1) * 90
    Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).save(path)


def _write_manifest(path: Path, image_path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = Path("../processed/images/case_test.png")
    assert (path.parent / relative).resolve() == image_path.resolve()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case_id", "processed_path", "split", "label"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_test",
                "processed_path": relative.as_posix(),
                "split": "validation",
                "label": label,
            }
        )


def _build_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _smooth_low_texture_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path, label="suspected_opacity")

    baseline_dir = tmp_path / "baseline"
    improved_dir = tmp_path / "improved"
    run_batch_inference(manifest, baseline_dir, config=InferenceConfig())
    run_batch_inference(manifest, improved_dir, config=InferenceConfig.improved())
    return baseline_dir, improved_dir


def test_build_comparison_reports_metrics_and_case_changes(tmp_path: Path) -> None:
    baseline_dir, improved_dir = _build_pair(tmp_path)

    comparison = build_comparison(baseline_dir, improved_dir)

    metric_names = {row["metric"] for row in comparison["metrics"]}
    assert {"accuracy", "macro_f1", "macro_specificity"} <= metric_names
    # the homogeneous opacity is missed by the baseline and recovered by improved
    assert comparison["case_changes"]["improved_fixed"] == 1
    assert comparison["case_changes"]["improved_broke"] == 0
    assert comparison["safety"]["baseline_dangerous_false_negatives"] == 1
    assert comparison["safety"]["improved_dangerous_false_negatives"] == 0


def test_write_comparison_outputs(tmp_path: Path) -> None:
    baseline_dir, improved_dir = _build_pair(tmp_path)
    comparison = build_comparison(baseline_dir, improved_dir)

    outputs = write_comparison_outputs(comparison, tmp_path / "comparison")
    report = json.loads(outputs["report_json"].read_text(encoding="utf-8"))
    with outputs["metrics_csv"].open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    with outputs["cases_csv"].open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))

    assert report["case_changes"]["improved_fixed"] == 1
    assert any(row["metric"] == "accuracy" for row in metrics)
    assert cases[0]["case_id"] == "case_test"
