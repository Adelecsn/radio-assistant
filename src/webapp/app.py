"""FastAPI application for visualizing local inference outputs."""

from __future__ import annotations

from pathlib import Path

from src.inference import InferenceConfig

from .live import analyze_upload
from .render import (
    error_register_csv,
    render_case_detail,
    render_about,
    render_dashboard,
    render_error_review,
    render_report,
    render_upload_form,
    render_upload_result,
)
from .repository import PredictionRepository
from .storage import init_db, log_case_view, log_inference_event, read_log_summary


def create_app(
    predictions_dir: str | Path = "data/predictions/baseline_v1",
    db_path: str | Path = "logs/webapp.sqlite",
    *,
    variant: str = "improved",
    upload_dir: str | Path = "logs/uploads",
    prompt_path: str | None = None,
    prompt_version: str | None = None,
):
    """Create the local dashboard application."""
    try:
        from fastapi import FastAPI, File, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    except ImportError as error:
        raise RuntimeError(
            "Les dépendances web ne sont pas installées. "
            "Installe requirements.txt pour lancer la webapp."
        ) from error

    repository = PredictionRepository(predictions_dir)
    database = Path(db_path)
    init_db(database)
    if variant == "improved":
        live_config = InferenceConfig.improved()
    elif variant == "medgemma":
        # Live MedGemma needs a prompt that names the JSON fields, otherwise the
        # model returns keys we don't expect and every upload falls back to
        # uncertain. Default to baseline_v1 but allow overriding via the CLI.
        medgemma_overrides: dict[str, object] = {}
        if prompt_path:
            medgemma_overrides["prompt_path"] = prompt_path
        if prompt_version:
            medgemma_overrides["prompt_version"] = prompt_version
        live_config = InferenceConfig.medgemma(**medgemma_overrides)
    else:
        live_config = InferenceConfig()
    uploads = Path(upload_dir)

    app = FastAPI(
        title="Radio Assistant",
        description="Dashboard pédagogique local pour les sorties d'inférence.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/about", response_class=HTMLResponse)
    def about() -> str:
        return render_about(repository.summary(), repository.evaluation(), repository.metadata())

    @app.get("/upload", response_class=HTMLResponse)
    def upload_form() -> str:
        return render_upload_form(
            variant=live_config.variant,
            model_version=live_config.model_version,
        )

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(file: bytes = File(...)):
        try:
            result = analyze_upload(
                file,
                config=live_config,
                upload_dir=uploads,
            )
        except ValueError as error:
            return HTMLResponse(
                render_upload_form(
                    variant=live_config.variant,
                    model_version=live_config.model_version,
                    error=str(error),
                ),
                status_code=400,
            )
        log_inference_event(
            database,
            case_id=result.case_id,
            prediction=result.prediction,
            source="upload",
        )
        return render_upload_result(
            case_id=result.case_id,
            prediction=result.prediction,
            features=result.features,
            quality_reasons=result.quality_reasons,
            image_url=f"/uploads/{result.case_id}/image",
        )

    @app.get("/uploads/{case_id}/image")
    def upload_image(case_id: str):
        if not case_id.replace("_", "").isalnum():
            raise HTTPException(status_code=404, detail="Image not found")
        image_path = uploads / f"{case_id}.png"
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path, media_type="image/png")

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
