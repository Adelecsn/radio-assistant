"""FastAPI application for visualizing local inference outputs."""

from __future__ import annotations

from pathlib import Path

from .render import (
    error_register_csv,
    render_case_detail,
    render_dashboard,
    render_error_review,
    render_report,
)
from .repository import PredictionRepository
from .storage import init_db, log_case_view, read_log_summary


def create_app(
    predictions_dir: str | Path = "data/predictions/baseline_v1",
    db_path: str | Path = "logs/webapp.sqlite",
):
    """Create the local dashboard application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    except ImportError as error:
        raise RuntimeError(
            "Les dépendances web ne sont pas installées. "
            "Installe requirements.txt pour lancer la webapp."
        ) from error

    repository = PredictionRepository(predictions_dir)
    database = Path(db_path)
    init_db(database)

    app = FastAPI(
        title="Radio Assistant",
        description="Dashboard pédagogique local pour les sorties d'inférence.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(
            repository.summary(),
            repository.evaluation(),
            repository.list_cases(),
            repository.metadata(),
            read_log_summary(database),
        )

    @app.get("/errors", response_class=HTMLResponse)
    def error_review() -> str:
        cases = repository.list_cases()
        return render_error_review(
            repository.evaluation().get("errors", []),
            {case.case_id: case for case in cases},
        )

    @app.get("/report", response_class=HTMLResponse)
    def report() -> str:
        return render_report(repository.evaluation(), repository.metadata())

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case_detail(case_id: str) -> str:
        case = repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        log_case_view(database, case)
        image_path = repository.resolve_image_path(case)
        image_url = f"/cases/{case.case_id}/image" if image_path and image_path.is_file() else None
        return render_case_detail(case, image_url=image_url)

    @app.get("/cases/{case_id}/image")
    def case_image(case_id: str):
        case = repository.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        image_path = repository.resolve_image_path(case)
        if image_path is None or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path, media_type="image/png")

    @app.get("/api/summary")
    def api_summary() -> dict[str, object]:
        return {
            "summary": repository.summary(),
            "evaluation": repository.evaluation(),
            "logs": read_log_summary(database),
            "metadata": repository.metadata(),
        }

    @app.get("/api/evaluation")
    def api_evaluation() -> dict[str, object]:
        return repository.evaluation()

    @app.get("/api/evaluation/errors.csv", response_class=PlainTextResponse)
    def api_error_register_csv() -> str:
        return error_register_csv(repository.evaluation())

    @app.get("/api/cases")
    def api_cases() -> list[dict[str, object]]:
        return [
            {
                "case_id": case.case_id,
                "prediction": case.prediction,
                "features": case.features,
                "split": case.split,
                "label": case.label,
            }
            for case in repository.list_cases()
        ]

    return app
