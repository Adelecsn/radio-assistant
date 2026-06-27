from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.contracts import is_valid_prediction
from src.inference import (
    InferenceConfig,
    predict_image,
    predict_image_improved,
    run_batch_inference,
)


def _normal_like_image(path: Path) -> None:
    rows, cols = np.indices((512, 512))
    radius = np.sqrt((rows - 256) ** 2 + (cols - 256) ** 2)
    values = 70 + np.clip(radius / 256, 0, 1) * 80
    texture = ((rows % 17) - 8) * 2
    Image.fromarray(np.clip(values + texture, 0, 255).astype(np.uint8)).save(path)


def _smooth_low_texture_image(path: Path) -> None:
    """Normal-looking globally, but homogeneous (low local contrast)."""
    rows, cols = np.indices((512, 512))
    radius = np.sqrt((rows - 256) ** 2 + (cols - 256) ** 2)
    values = 70 + np.clip(radius / 256, 0, 1) * 90
    Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).save(path)


def _opacity_like_image(path: Path) -> None:
    rows, cols = np.indices((512, 512))
    radius = np.sqrt((rows - 256) ** 2 + (cols - 256) ** 2)
    values = 65 + np.clip(radius / 256, 0, 1) * 70
    values[160:360, 80:270] = 235
    Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).save(path)


def _write_manifest(path: Path, image_path: Path) -> None:
    relative = Path(os.path.relpath(image_path, start=path.parent))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case_id", "processed_path", "split", "label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_test",
                "processed_path": relative.as_posix(),
                "split": "test",
                "label": "normal",
            }
        )


def test_predict_image_returns_valid_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _normal_like_image(image_path)

    result = predict_image(image_path)

    assert is_valid_prediction(result.prediction)
    assert result.prediction["image_quality"] == "good"
    assert result.prediction["model_version"] == "image-stat-baseline-v0.2"
    assert "std_intensity" in result.features


def test_poor_quality_image_becomes_uncertain(tmp_path: Path) -> None:
    image_path = tmp_path / "flat.png"
    Image.new("L", (512, 512), color=40).save(image_path)

    result = predict_image(image_path)

    assert is_valid_prediction(result.prediction)
    assert result.prediction["image_quality"] == "poor"
    assert result.prediction["predicted_class"] == "uncertain"
    assert result.quality_reasons


def test_bright_asymmetric_signal_can_be_flagged_by_baseline(tmp_path: Path) -> None:
    image_path = tmp_path / "signal.png"
    _opacity_like_image(image_path)

    result = predict_image(image_path)

    assert is_valid_prediction(result.prediction)
    assert result.prediction["predicted_class"] == "suspected_opacity"


def test_improved_variant_is_valid_and_versioned(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _normal_like_image(image_path)

    result = predict_image_improved(image_path)

    assert is_valid_prediction(result.prediction)
    assert result.prediction["model_version"] == "image-stat-improved-v0.3"
    assert result.prediction["prompt_version"] == "v2.0"
    assert "local_texture_max" in result.features


def test_local_texture_guardrail_reflags_homogeneous_normal(tmp_path: Path) -> None:
    image_path = tmp_path / "smooth.png"
    _smooth_low_texture_image(image_path)

    baseline = predict_image(image_path)
    improved = predict_image_improved(image_path)

    assert baseline.prediction["predicted_class"] == "normal"
    assert improved.prediction["predicted_class"] == "suspected_opacity"
    assert is_valid_prediction(improved.prediction)


def test_batch_inference_improved_variant(tmp_path: Path) -> None:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _smooth_low_texture_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path)

    result = run_batch_inference(
        manifest, tmp_path / "predictions", config=InferenceConfig.improved()
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.processed == 1
    assert metadata["hyperparameters"]["variant"] == "improved"
    assert metadata["hyperparameters"]["model_version"] == "image-stat-improved-v0.3"


def test_batch_inference_writes_json_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _normal_like_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path)

    result = run_batch_inference(manifest, tmp_path / "predictions")
    case_json = result.output_dir / "cases" / "case_test.json"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    case_payload = json.loads(case_json.read_text(encoding="utf-8"))
    index_lines = result.index_path.read_text(encoding="utf-8").strip().splitlines()

    assert result.processed == 1
    assert result.rejected == 0
    assert is_valid_prediction(case_payload["prediction"])
    assert metadata["hyperparameters"]["confidence_threshold"] == 0.60
    assert len(index_lines) == 1


def test_inference_cli_runs_end_to_end(tmp_path: Path) -> None:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _normal_like_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "predictions"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["processed"] == 1
    assert Path(payload["index_path"]).exists()


def test_inference_cli_improved_variant(tmp_path: Path) -> None:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _normal_like_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference",
            "--variant",
            "improved",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "predictions"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metadata = json.loads(
        (tmp_path / "predictions" / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["hyperparameters"]["variant"] == "improved"
