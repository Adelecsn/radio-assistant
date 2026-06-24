"""HTML rendering helpers for the local dashboard."""

from __future__ import annotations

import html
import json
from typing import Any

from src.contracts import WARNING_TEXT

from .repository import CaseRecord


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _metric(value: object) -> str:
    return "n/a" if value is None else _esc(value)


def _bar_svg(counts: dict[str, int], title: str) -> str:
    if not counts:
        return "<p>Aucune donnée à afficher.</p>"

    max_count = max(counts.values()) or 1
    rows = []
    for index, (label, count) in enumerate(counts.items()):
        width = int((count / max_count) * 280)
        y = 34 + index * 34
        rows.append(
            f'<text x="0" y="{y}" class="chart-label">{_esc(label)}</text>'
            f'<rect x="130" y="{y - 16}" width="{width}" height="20" rx="4" />'
            f'<text x="{140 + width}" y="{y}" class="chart-count">{count}</text>'
        )

    height = 48 + len(counts) * 34
    return (
        f'<svg class="chart" viewBox="0 0 460 {height}" role="img" '
        f'aria-label="{_esc(title)}">'
        f'<title>{_esc(title)}</title>'
        + "".join(rows)
        + "</svg>"
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7fb;
      color: #172033;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px; }}
    a {{ color: #2358d5; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .warning {{
      background: #fff4d6;
      border: 1px solid #e1bf63;
      border-radius: 12px;
      padding: 14px 16px;
      margin: 18px 0;
      font-weight: 650;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
    .card {{
      background: white;
      border: 1px solid #dde2ee;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 1px 4px rgba(21, 32, 59, 0.06);
    }}
    .metric {{ display: block; font-size: 2rem; font-weight: 750; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #e7ebf3; text-align: left; vertical-align: top; }}
    th {{ background: #edf1f8; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    code, pre {{ background: #edf1f8; border-radius: 8px; }}
    pre {{ padding: 14px; overflow: auto; }}
    .chart rect {{ fill: #4a6cf0; }}
    .chart-label, .chart-count {{ font-size: 14px; fill: #172033; }}
    .muted {{ color: #667086; }}
    .image-preview {{ max-width: 100%; border-radius: 12px; border: 1px solid #dde2ee; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render_dashboard(
    summary: dict[str, Any],
    cases: list[CaseRecord],
    metadata: dict[str, Any],
    log_summary: dict[str, Any],
) -> str:
    cards = f"""
<section class="grid">
  <div class="card">Cas analysés<span class="metric">{_metric(summary.get("total_cases"))}</span></div>
  <div class="card">Confiance moyenne<span class="metric">{_metric(summary.get("average_confidence"))}</span></div>
  <div class="card">Latence moyenne<span class="metric">{_metric(summary.get("average_latency_ms"))} ms</span></div>
  <div class="card">Consultations loggées<span class="metric">{_metric(log_summary.get("total_views"))}</span></div>
</section>
"""
    accuracy = summary.get("baseline_accuracy")
    if accuracy is not None:
        cards += (
            f'<p class="muted">Accuracy baseline sur labels disponibles: '
            f'{_esc(accuracy)} ({_esc(summary.get("labeled_cases"))} cas labellisés).</p>'
        )

    rows = []
    for case in cases:
        prediction = case.prediction
        rows.append(
            "<tr>"
            f'<td><a href="/cases/{_esc(case.case_id)}">{_esc(case.case_id)}</a></td>'
            f"<td>{_esc(prediction.get('predicted_class'))}</td>"
            f"<td>{_esc(prediction.get('confidence_score'))}</td>"
            f"<td>{_esc(prediction.get('image_quality'))}</td>"
            f"<td>{_esc(case.split)}</td>"
            f"<td>{_esc(case.label)}</td>"
            "</tr>"
        )

    table = (
        "<table><thead><tr><th>Cas</th><th>Classe</th><th>Confiance</th>"
        "<th>Qualité</th><th>Split</th><th>Label</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    if not cases:
        table = "<p>Aucune prédiction trouvée. Lance d'abord <code>python -m src.inference</code>.</p>"

    body = f"""
<h1>Radio Assistant - Dashboard</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
{cards}
<section class="grid">
  <div class="card">
    <h2>Classes prédites</h2>
    {_bar_svg(summary.get("by_class", {}), "Répartition des classes prédites")}
  </div>
  <div class="card">
    <h2>Qualité image</h2>
    {_bar_svg(summary.get("by_quality", {}), "Répartition des qualités image")}
  </div>
</section>
<h2>Cas</h2>
{table}
<h2>Run</h2>
<pre>{_esc(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>
"""
    return _page("Radio Assistant - Dashboard", body)


def render_case_detail(case: CaseRecord, *, image_url: str | None = None) -> str:
    prediction = case.prediction
    image_block = ""
    if image_url:
        image_block = (
            '<div class="card"><h2>Image prétraitée</h2>'
            f'<img class="image-preview" src="{_esc(image_url)}" alt="Image prétraitée du cas">'
            "</div>"
        )

    findings = "".join(f"<li>{_esc(item)}</li>" for item in prediction.get("visual_findings", []))
    limitations = "".join(f"<li>{_esc(item)}</li>" for item in prediction.get("limitations", []))
    reasons = "".join(f"<li>{_esc(item)}</li>" for item in case.quality_reasons)
    if not reasons:
        reasons = "<li>Aucun motif de qualité bloquant.</li>"

    body = f"""
<p><a href="/">Retour au dashboard</a></p>
<h1>Cas {_esc(case.case_id)}</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
<section class="grid">
  <div class="card">Classe<span class="metric">{_esc(prediction.get("predicted_class"))}</span></div>
  <div class="card">Confiance<span class="metric">{_esc(prediction.get("confidence_score"))}</span></div>
  <div class="card">Qualité<span class="metric">{_esc(prediction.get("image_quality"))}</span></div>
  <div class="card">Latence<span class="metric">{_esc(prediction.get("inference_latency_ms"))} ms</span></div>
</section>
<section class="grid">
  {image_block}
  <div class="card">
    <h2>Observations visuelles</h2>
    <ul>{findings}</ul>
    <h2>Limites</h2>
    <ul>{limitations}</ul>
    <h2>Contrôle qualité</h2>
    <ul>{reasons}</ul>
  </div>
</section>
<h2>Justification</h2>
<p>{_esc(prediction.get("justification"))}</p>
<h2>Features</h2>
<pre>{_esc(json.dumps(case.features, ensure_ascii=False, indent=2))}</pre>
<h2>JSON complet</h2>
<pre>{_esc(json.dumps(prediction, ensure_ascii=False, indent=2))}</pre>
"""
    return _page(f"Cas {case.case_id}", body)
