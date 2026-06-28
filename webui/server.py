"""Interface design "Assistant radiologue virtuel" — vitrine connectée au vrai moteur.

Cette application est volontairement SÉPARÉE du tableau de bord Streamlit noté
(src/webapp). Elle réutilise exactement la même analyse réelle (analyze_upload),
mais propose une présentation moderne et un sélecteur de public.

Règle de sécurité non négociable : quel que soit le public sélectionné,
l'avertissement non clinique reste identique et visible, et aucune sortie n'est
présentée comme un diagnostic. Seul le NIVEAU D'EXPLICATION change, jamais le sens
médical de la réponse.

Lancement :
    uvicorn webui.server:app --reload --port 8100
ou
    python -m webui.server
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.webapp.live import analyze_upload

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"

app = FastAPI(title="Assistant radiologue virtuel — interface design")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    try:
        result = analyze_upload(data, filename=file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # On renvoie l'image prétraitée en base64 pour l'afficher sans exposer de fichier.
    image_b64 = base64.b64encode(
        result.processed_image_path.read_bytes()
    ).decode("ascii")

    return JSONResponse(
        {
            "case_id": result.case_id,
            "prediction": result.prediction,
            "features": result.features,
            "quality_reasons": result.quality_reasons,
            "image_b64": image_b64,
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
