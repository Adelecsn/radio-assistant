from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.contracts import WARNING_TEXT
from src.inference import run_batch_inference
from src.webapp import PredictionRepository, log_case_view, read_log_summary
from src.webapp.render import (
    error_register_csv,
    render_case_detail,
    render_dashboard,
    render_error_review,
    render_report,
)


def _normal_like_image(path: Path) -> None:
    rows, cols = np.indices((512, 512))
    radius = np.sqrt((rows - 256) ** 2 + (cols - 256) ** 2)
    values = 70 + np.clip(radius / 256, 0, 1) * 80
    texture = ((cols % 19) - 9) * 2
    Image.fromarray(np.clip(values + texture, 0, 255).astype(np.uint8)).save(path)


def _write_manifest(path: Path, image_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = Path("../processed/images/case_test.png")
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
    assert (path.parent / relative).resolve() == image_path.resolve()


def _build_predictions(tmp_path: Path) -> Path:
    image_path = tmp_path / "processed" / "images" / "case_test.png"
    image_path.parent.mkdir(parents=True)
    _normal_like_image(image_path)
    manifest = tmp_path / "manifests" / "ingest.csv"
    _write_manifest(manifest, image_path)
    output_dir = tmp_path / "predictions"
    run_batch_inference(manifest, output_dir)
    return output_dir


def test_prediction_repository_summarizes_outputs(tmp_path: Path) -> None:
    output_dir = _build_predictions(tmp_path)
    repository = PredictionRepository(output_dir)

    cases = repository.list_cases()
    summary = repository.summary()

    assert len(cases) == 1
    assert summary["total_cases"] == 1
    assert summary["average_confidence"] is not None
    assert repository.resolve_image_path(cases[0]) == (
        tmp_path / "processed" / "images" / "case_test.png"
    ).resolve()


def test_webapp_sqlite_logs_case_views(tmp_path: Path) -> None:
    output_dir = _build_predictions(tmp_path)
    case = PredictionRepository(output_dir).list_cases()[0]
    database = tmp_path / "logs" / "webapp.sqlite"

    log_case_view(database, case)
    log_summary = read_log_summary(database)

    assert log_summary["total_views"] == 1
    assert case.prediction["predicted_class"] in log_summary["views_by_class"]


def test_dashboard_and_case_html_include_warning(tmp_path: Path) -> None:
    output_dir = _build_predictions(tmp_path)
    repository = PredictionRepository(output_dir)
    cases = repository.list_cases()

    dashboard_html = render_dashboard(
        repository.summary(),
        repository.evaluation(),
        cases,
        repository.metadata(),
        {"total_views": 0, "last_viewed_at": None, "views_by_class": {}},
    )
    case_html = render_case_detail(cases[0], image_url="/cases/case_test/image")

    assert WARNING_TEXT in dashboard_html
    assert "case_test" in dashboard_html
    assert "Évaluation baseline" in dashboard_html
    assert WARNING_TEXT in case_html
    assert "JSON complet" in case_html


def test_error_review_html_displays_error_images(tmp_path: Path) -> None:
    output_dir = _build_predictions(tmp_path)
    repository = PredictionRepository(output_dir)
    case = repository.list_cases()[0]
    errors = [
        {
            "case_id": case.case_id,
            "ground_truth": "normal",
            "predicted_class": "suspected_opacity",
            "confidence_score": 0.8,
            "error_type": "normal_as_suspected_opacity",
        }
    ]

    html = render_error_review(errors, {case.case_id: case})

    assert "Revue visuelle des erreurs" in html
    assert f"/cases/{case.case_id}/image" in html
    assert "normal_as_suspected_opacity" in html


def test_report_html_and_error_csv_exports(tmp_path: Path) -> None:
    output_dir = _build_predictions(tmp_path)
    repository = PredictionRepository(output_dir)
    evaluation = repository.evaluation()

    report_html = render_report(evaluation, repository.metadata())
    errors_csv = error_register_csv(evaluation)

    assert "Rapport d'évaluation" in report_html
    assert "/api/evaluation/errors.csv" in report_html
    assert errors_csv.startswith("case_id,split,ground_truth")
