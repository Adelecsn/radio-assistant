"""Interface unifiée "Assistant radiologue virtuel" : analyse + dashboard complet.

Une seule application FastAPI qui réunit, dans la même interface :
- l'analyse d'image (page d'accueil, design multi-public) ;
- le dashboard d'évaluation COMPLET (toutes les infos de l'ancien tableau de bord) ;
- les pages détail d'un cas, rapport et "à propos / limites".

Elle réutilise le moteur réel du projet (analyze_upload, PredictionRepository,
evaluate_records) et les rendus existants, pour ne perdre aucune information.

Lancement :
    python -m uvicorn webui.server:app --reload --port 8100
Variables d'environnement optionnelles :
    RADIO_PRED_DIR  (def. data/predictions/baseline_v1)
    RADIO_DB_PATH   (def. logs/webapp.sqlite)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from src.webapp.live import analyze_upload
from src.webapp.repository import PredictionRepository
from src.webapp.render import (
    _bar_svg,
    _confusion_matrix_table,
    _error_table,
    _per_class_table,
    error_register_csv,
    render_about,
    render_case_detail,
    render_report,
)
from src.webapp.storage import init_db, read_log_summary

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
PRED_DIR = os.environ.get("RADIO_PRED_DIR", "data/predictions/baseline_v1")
DB_PATH = os.environ.get("RADIO_DB_PATH", "logs/webapp.sqlite")

WARNING = ("Prototype pédagogique. Non destiné au diagnostic. "
           "Validation par un professionnel qualifié requise.")

app = FastAPI(title="Assistant radiologue virtuel — interface unifiée")

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
try:
    init_db(DB_PATH)
except Exception:
    pass


def _repo() -> PredictionRepository:
    return PredictionRepository(PRED_DIR)


def _pct(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _nav(active: str) -> str:
    def link(href, label, key):
        on = "background:var(--primary);border-color:var(--primary);color:#fff" if key == active \
            else "background:var(--surface);color:var(--ink)"
        return (f'<a href="{href}" style="text-decoration:none;border:1px solid var(--line);'
                f'{on};padding:9px 16px;border-radius:999px;font-size:.92rem;font-weight:600">{label}</a>')
    return ('<nav class="top" style="display:flex;gap:8px;flex-wrap:wrap;margin:18px 0">'
            + link("/", "Analyser une image", "analyse")
            + link("/dashboard", "Dashboard d'évaluation", "dashboard")
            + link("/report", "Rapport", "report")
            + link("/about", "À propos & limites", "about")
            + "</nav>")


# ---------------------------------------------------------------- Analyse
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
    image_b64 = base64.b64encode(result.processed_image_path.read_bytes()).decode("ascii")
    return JSONResponse({
        "case_id": result.case_id,
        "prediction": result.prediction,
        "features": result.features,
        "quality_reasons": result.quality_reasons,
        "image_b64": image_b64,
    })


# ---------------------------------------------------------------- Dashboard complet
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    repo = _repo()
    cases = repo.list_cases()
    if not cases:
        return HTMLResponse(_shell("Dashboard", _nav("dashboard") + f"""
          <div class="warning"><span class="ic">⚠️</span><span>{WARNING}</span></div>
          <div class="card"><h2>Aucune prédiction trouvée</h2>
          <p class="muted">Le dossier <code>{PRED_DIR}</code> ne contient pas de
          <code>predictions.jsonl</code>. Génère-les d'abord :</p>
          <pre>python -m src.inference --manifest data/manifests/ingest_manifest.csv --output-dir data/predictions/baseline_v1</pre>
          </div>"""))

    summary = repo.summary()
    ev = repo.evaluation()
    try:
        logs = read_log_summary(DB_PATH)
    except Exception:
        logs = {"total_views": 0, "total_inferences": 0}

    # 5 cartes du haut
    top = "".join(
        f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>'
        for l, v in [
            ("Cas analysés", summary.get("total_cases", 0)),
            ("Confiance moyenne", summary.get("average_confidence", "n/a")),
            ("Latence moyenne", f'{summary.get("average_latency_ms","?")} ms'),
            ("Consultations loggées", logs.get("total_views", 0)),
            ("Analyses uploadées", logs.get("total_inferences", 0)),
        ])
    acc_note = ""
    if summary.get("baseline_accuracy") is not None:
        acc_note = (f'<p class="muted">Accuracy baseline sur labels disponibles : '
                    f'{summary.get("baseline_accuracy")} ({summary.get("labeled_cases")} cas labellisés).</p>')

    # Cartes évaluation baseline
    evcards = "".join(
        f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>'
        for l, v in [
            ("Accuracy", _pct(ev.get("accuracy"))),
            ("Macro-F1", _pct(ev.get("macro_f1"))),
            ("Macro-rappel", _pct(ev.get("macro_recall_sensitivity"))),
            ("Macro-spécificité", _pct(ev.get("macro_specificity"))),
            ("Validité JSON", _pct(ev.get("json_validity_rate"))),
            ("Taux d'incertitude", _pct(ev.get("uncertainty_rate"))),
            ("Erreurs à revoir", len(ev.get("errors", []))),
            ("Cas évalués", ev.get("evaluated_cases", 0)),
        ])

    # Tableau complet des cas
    rows = "".join(
        "<tr>"
        f'<td><a href="/cases/{c.case_id}">{c.case_id}</a></td>'
        f"<td>{c.prediction.get('predicted_class','')}</td>"
        f"<td>{c.prediction.get('confidence_score','')}</td>"
        f"<td>{c.prediction.get('image_quality','')}</td>"
        f"<td>{c.split}</td><td>{c.label}</td></tr>"
        for c in cases)
    cases_table = ('<div class="table-wrap"><table><thead><tr><th>Cas</th><th>Classe</th>'
                   '<th>Confiance</th><th>Qualité</th><th>Split</th><th>Label</th></tr></thead><tbody>'
                   + rows + "</tbody></table></div>")

    body = _nav("dashboard") + f"""
      <div class="warning"><span class="ic">⚠️</span><span>{WARNING}</span></div>
      <div class="card"><h2>Synthèse</h2><div class="stats">{top}</div>{acc_note}</div>

      <div class="grid">
        <div class="card"><h2>Classes prédites</h2>{_bar_svg(summary.get("by_class", {}), "Classes prédites")}</div>
        <div class="card"><h2>Qualité image</h2>{_bar_svg(summary.get("by_quality", {}), "Qualité image")}</div>
      </div>

      <div class="card"><h2>Évaluation baseline</h2><div class="stats">{evcards}</div></div>

      <div class="grid">
        <div class="card"><h2>Matrice de confusion</h2>{_confusion_matrix_table(ev)}</div>
        <div class="card"><h2>Métriques par classe</h2>{_per_class_table(ev)}</div>
      </div>

      <div class="card"><h2>Erreurs à analyser</h2>
        <p class="muted"><a href="/report">Voir le rapport</a> · export :
        <a href="/api/evaluation/errors.csv">registre CSV</a></p>
        {_error_table(ev)}
      </div>

      <div class="card"><h2>Cas ({len(cases)})</h2>{cases_table}</div>
      <div class="card"><h2>Métadonnées du run</h2>
        <pre>{json.dumps(repo.metadata(), ensure_ascii=False, indent=2)}</pre></div>

      <div class="warning" style="margin-top:18px"><span class="ic">⚠️</span><span>{WARNING}</span></div>
    """
    return HTMLResponse(_shell("Dashboard d'évaluation", body))


# ---------------------------------------------------------------- Détail d'un cas
@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(case_id: str) -> HTMLResponse:
    repo = _repo()
    case = repo.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Cas introuvable")
    image_url = f"/cases/{case_id}/image" if repo.resolve_image_path(case) else None
    html = render_case_detail(case, image_url=image_url)
    return HTMLResponse(_relink(html))


@app.get("/cases/{case_id}/image")
def case_image(case_id: str):
    repo = _repo()
    case = repo.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Cas introuvable")
    path = repo.resolve_image_path(case)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(str(path))


# ---------------------------------------------------------------- Rapport & À propos
@app.get("/report", response_class=HTMLResponse)
def report() -> HTMLResponse:
    repo = _repo()
    return HTMLResponse(_relink(render_report(repo.evaluation(), repo.metadata())))


@app.get("/about", response_class=HTMLResponse)
def about() -> HTMLResponse:
    repo = _repo()
    return HTMLResponse(_relink(render_about(repo.summary(), repo.evaluation(), repo.metadata())))


@app.get("/api/evaluation")
def api_evaluation() -> JSONResponse:
    return JSONResponse(_repo().evaluation())


@app.get("/api/evaluation/errors.csv", response_class=PlainTextResponse)
def api_errors_csv() -> PlainTextResponse:
    return PlainTextResponse(error_register_csv(_repo().evaluation()), media_type="text/csv")


# ---------------------------------------------------------------- Helpers
def _relink(html: str) -> str:
    """Réoriente les liens des pages réutilisées vers la nav de webui."""
    html = html.replace('href="/upload"', 'href="/"')
    html = html.replace('href="/"', 'href="/dashboard"')  # 'Retour au dashboard'
    return html


def _shell(subtitle: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Assistant radiologue virtuel — {subtitle}</title>
<style>
  :root{{--bg:#eef4f9;--surface:#fff;--ink:#0f2a3f;--muted:#5b7185;--line:#dce7f0;
  --primary:#1668a8;--primary-deep:#0b4a73;--soft:#e3eef7;--warn-bg:#fff7e6;
  --warn-line:#f0c674;--warn-ink:#7a5600;--radius:16px;--shadow:0 6px 24px rgba(11,74,115,.08);}}
  *{{box-sizing:border-box}} html,body{{overflow-x:hidden}}
  body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial,sans-serif;line-height:1.55}}
  .wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 64px}}
  header.app{{display:flex;align-items:center;gap:14px;margin-bottom:8px}}
  .logo{{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,var(--primary),var(--primary-deep));
  display:grid;place-items:center;color:#fff;font-weight:800;box-shadow:var(--shadow)}}
  h1{{font-size:1.55rem;margin:0}} .sub{{color:var(--muted);font-size:.95rem;margin:2px 0 0}}
  .warning{{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);
  padding:12px 16px;border-radius:12px;font-size:.92rem;font-weight:600;display:flex;gap:10px;
  align-items:flex-start;margin-bottom:20px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
  .grid>*{{min-width:0}} @media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
  .card{{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:20px;box-shadow:var(--shadow);margin-bottom:18px}}
  .card h2{{font-size:1.05rem;margin:0 0 14px}} .card h3{{font-size:.98rem;margin:0 0 10px}}
  .muted{{color:var(--muted);overflow-wrap:anywhere}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
  @media(max-width:820px){{.stats{{grid-template-columns:repeat(2,1fr)}}}}
  .stat{{background:#f7fafc;border:1px solid var(--line);border-radius:12px;padding:14px}}
  .stat-v{{font-size:1.4rem;font-weight:800;color:var(--primary-deep)}}
  .stat-l{{font-size:.8rem;color:var(--muted);margin-top:2px}}
  .table-wrap{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.86rem}}
  th,td{{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left}}
  th{{color:var(--primary-deep);font-weight:700}} td a{{color:var(--primary)}}
  svg.chart{{max-width:100%;height:auto}}
  pre{{background:#0f2a3f;color:#d7e6f2;padding:14px;border-radius:12px;font-size:.8rem;
  white-space:pre-wrap;word-break:break-word}}
  footer{{margin-top:30px;color:var(--muted);font-size:.82rem;text-align:center}}
</style></head><body><div class="wrap">
  <header class="app"><div class="logo">R</div>
    <div><h1>Assistant radiologue virtuel</h1>
    <p class="sub">Prototype pédagogique d'analyse prudente de radiographies thoraciques</p></div>
  </header>
  {body}
  <footer>Interface de démonstration · moteur d'inférence et d'évaluation réels du projet ·
  aucune donnée patient nominative conservée.</footer>
</div></body></html>"""


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
