"""HTML rendering helpers for the local dashboard."""

from __future__ import annotations

import csv
import html
import json
from io import StringIO
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


def _percent(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return _esc(value)


def _confusion_matrix_table(evaluation: dict[str, Any]) -> str:
    classes = evaluation.get("classes", [])
    matrix = evaluation.get("confusion_matrix", {})
    if not classes or not matrix:
        return "<p>Aucune matrice disponible : aucun label exploitable.</p>"

    header = "<tr><th>Réel \\ Prédit</th>" + "".join(
        f"<th>{_esc(class_name)}</th>" for class_name in classes
    ) + "</tr>"
    rows = []
    for truth in classes:
        cells = [f"<th>{_esc(truth)}</th>"]
        for predicted in classes:
            cells.append(f"<td>{_esc(matrix.get(truth, {}).get(predicted, 0))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table>' + header + "".join(rows) + "</table></div>"


def _per_class_table(evaluation: dict[str, Any]) -> str:
    per_class = evaluation.get("per_class", {})
    if not per_class:
        return "<p>Aucune métrique par classe disponible.</p>"

    rows = []
    for class_name, metrics in per_class.items():
        rows.append(
            "<tr>"
            f"<td>{_esc(class_name)}</td>"
            f"<td>{_esc(metrics.get('support'))}</td>"
            f"<td>{_percent(metrics.get('precision'))}</td>"
            f"<td>{_percent(metrics.get('recall_sensitivity'))}</td>"
            f"<td>{_percent(metrics.get('f1'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Classe</th><th>Support</th><th>Précision</th>'
        "<th>Rappel / sensibilité</th><th>F1</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _error_table(evaluation: dict[str, Any], *, limit: int = 8) -> str:
    errors = evaluation.get("errors", [])
    if not errors:
        return "<p>Aucune erreur sur les cas labellisés.</p>"

    rows = []
    for error in errors[:limit]:
        rows.append(
            "<tr>"
            f"<td>{_esc(error.get('case_id'))}</td>"
            f"<td>{_esc(error.get('ground_truth'))}</td>"
            f"<td>{_esc(error.get('predicted_class'))}</td>"
            f"<td>{_esc(error.get('confidence_score'))}</td>"
            f"<td>{_esc(error.get('error_type'))}</td>"
            "</tr>"
        )
    more = ""
    if len(errors) > limit:
        more = f'<p class="muted">+ {_esc(len(errors) - limit)} autres erreurs dans le registre.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr><th>Cas</th><th>Label</th><th>Prédiction</th>'
        "<th>Confiance</th><th>Type</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + more
    )


def error_register_csv(evaluation: dict[str, Any]) -> str:
    """Return the current evaluation errors as CSV text."""
    fields = [
        "case_id",
        "split",
        "ground_truth",
        "predicted_class",
        "confidence_score",
        "error_type",
        "image_quality",
        "prompt_version",
        "model_version",
        "reviewer_comment",
        "status",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(evaluation.get("errors", []))
    return buffer.getvalue()


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
      overflow: hidden;
    }}
    .metric {{ display: block; font-size: 2rem; font-weight: 750; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #e7ebf3; text-align: left; vertical-align: top; }}
    th {{ background: #edf1f8; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .table-wrap {{ max-width: 100%; overflow-x: auto; border-radius: 14px; }}
    .table-wrap table {{ min-width: max-content; }}
    .table-wrap th, .table-wrap td {{ white-space: nowrap; }}
    code, pre {{ background: #edf1f8; border-radius: 8px; }}
    pre {{ padding: 14px; overflow: auto; }}
    .chart rect {{ fill: #4a6cf0; }}
    .chart-label, .chart-count {{ font-size: 14px; fill: #172033; }}
    .muted {{ color: #667086; }}
    .image-preview {{ max-width: 100%; border-radius: 12px; border: 1px solid #dde2ee; }}
    .error-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .error-image {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #07101f; border-radius: 12px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #edf1f8; margin: 3px 4px 3px 0; }}
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
    evaluation: dict[str, Any],
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
  <div class="card">Analyses uploadées<span class="metric">{_metric(log_summary.get("total_inferences"))}</span></div>
</section>
"""
    accuracy = summary.get("baseline_accuracy")
    if accuracy is not None:
        cards += (
            f'<p class="muted">Accuracy baseline sur labels disponibles: '
            f'{_esc(accuracy)} ({_esc(summary.get("labeled_cases"))} cas labellisés).</p>'
        )

    evaluation_cards = f"""
<section class="grid">
  <div class="card">Accuracy<span class="metric">{_percent(evaluation.get("accuracy"))}</span></div>
  <div class="card">Macro-F1<span class="metric">{_percent(evaluation.get("macro_f1"))}</span></div>
  <div class="card">Macro-rappel<span class="metric">{_percent(evaluation.get("macro_recall_sensitivity"))}</span></div>
  <div class="card">Erreurs à revoir<span class="metric">{_esc(len(evaluation.get("errors", [])))}</span></div>
</section>
"""

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
        '<div class="table-wrap"><table><thead><tr><th>Cas</th><th>Classe</th><th>Confiance</th>'
        "<th>Qualité</th><th>Split</th><th>Label</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    if not cases:
        table = "<p>Aucune prédiction trouvée. Lance d'abord <code>python -m src.inference</code>.</p>"

    body = f"""
<h1>Radio Assistant - Dashboard</h1>
<p><a href="/upload">Analyser une image</a> · <a href="/about">À propos et limites</a> · <a href="/report">Rapport d'évaluation</a> · <a href="/errors">Revue des erreurs</a></p>
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
<h2>Évaluation baseline</h2>
{evaluation_cards}
<section class="grid">
  <div class="card">
    <h3>Matrice de confusion</h3>
    {_confusion_matrix_table(evaluation)}
  </div>
  <div class="card">
    <h3>Métriques par classe</h3>
    {_per_class_table(evaluation)}
  </div>
</section>
<h3>Erreurs à analyser</h3>
<p><a href="/errors">Voir les erreurs visuellement</a> · <a href="/report">Voir le rapport</a></p>
{_error_table(evaluation)}
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


def render_report(evaluation: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Render a compact evaluation report page."""
    body = f"""
<p><a href="/">Retour au dashboard</a></p>
<h1>Rapport d'évaluation</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
<section class="grid">
  <div class="card">Cas évalués<span class="metric">{_esc(evaluation.get("evaluated_cases"))}</span></div>
  <div class="card">Accuracy<span class="metric">{_percent(evaluation.get("accuracy"))}</span></div>
  <div class="card">Macro-F1<span class="metric">{_percent(evaluation.get("macro_f1"))}</span></div>
  <div class="card">Erreurs<span class="metric">{_esc(len(evaluation.get("errors", [])))}</span></div>
</section>
<section class="grid">
  <div class="card">
    <h2>Matrice de confusion</h2>
    {_confusion_matrix_table(evaluation)}
  </div>
  <div class="card">
    <h2>Métriques par classe</h2>
    {_per_class_table(evaluation)}
  </div>
</section>
<h2>Exports</h2>
<ul>
  <li><a href="/api/evaluation">Rapport JSON</a></li>
  <li><a href="/api/evaluation/errors.csv">Registre d'erreurs CSV</a></li>
</ul>
<h2>Métadonnées du run</h2>
<pre>{_esc(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>
"""
    return _page("Rapport d'évaluation", body)


def render_about(
    summary: dict[str, Any],
    evaluation: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Render project scope, dataset, limitations and next steps."""
    hyperparameters = metadata.get("hyperparameters", {})
    model_version = hyperparameters.get("model_version") or ", ".join(
        summary.get("model_versions", [])
    )
    prompt_version = hyperparameters.get("prompt_version") or ", ".join(
        summary.get("prompt_versions", [])
    )
    body = f"""
<p><a href="/">Retour au dashboard</a></p>
<h1>À propos et limites</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
<section class="grid">
  <div class="card">Cas chargés<span class="metric">{_esc(summary.get("total_cases"))}</span></div>
  <div class="card">Cas évalués<span class="metric">{_esc(evaluation.get("evaluated_cases"))}</span></div>
  <div class="card">Accuracy locale<span class="metric">{_percent(evaluation.get("accuracy"))}</span></div>
  <div class="card">Macro-F1 local<span class="metric">{_percent(evaluation.get("macro_f1"))}</span></div>
</section>
<section class="grid">
  <div class="card">
    <h2>Source des données</h2>
    <p>
      L'échantillon local utilise le dataset <strong>RSNA Pneumonia Detection Challenge</strong>,
      version <code>stage_2</code>, téléchargé localement depuis Kaggle.
    </p>
    <p>
      Les images médicales restent dans <code>data/external/</code>,
      <code>data/raw/</code> et <code>data/processed/</code>, dossiers ignorés par Git.
    </p>
    <p>
      Échantillon actuel : 30 cas, soit 10 <code>normal</code>,
      10 <code>suspected_opacity</code> et 10 <code>uncertain</code>.
    </p>
  </div>
  <div class="card">
    <h2>Modèle actuel</h2>
    <p>Version modèle : <code>{_esc(model_version)}</code></p>
    <p>Version prompt : <code>{_esc(prompt_version)}</code></p>
    <p>
      La baseline est statistique : elle exploite des features simples
      comme contraste, asymétrie, pixels clairs centraux et densité de contours.
    </p>
    <p>
      Elle sert à valider le workflow logiciel. Elle ne remplace pas un modèle
      médical entraîné ni une lecture radiologique.
    </p>
  </div>
</section>
<section class="grid">
  <div class="card">
    <h2>Confidentialité</h2>
    <ul>
      <li>Pas de commit des radios, poids de modèles, bases SQLite ou sorties locales sensibles.</li>
      <li>Le manifeste ne conserve pas les noms de fichiers RSNA originaux.</li>
      <li>Une revue visuelle rapide n'a pas montré de nom patient ou date évidente.</li>
      <li>Des marqueurs techniques comme <code>L</code> ou <code>PORTABLE</code> peuvent rester visibles.</li>
    </ul>
  </div>
  <div class="card">
    <h2>Limites importantes</h2>
    <ul>
      <li>Échantillon très petit : 30 cas seulement.</li>
      <li>Seuils ajustés pour une démonstration locale, pas pour généraliser.</li>
      <li>Pas de segmentation anatomique ni de contexte clinique.</li>
      <li>Les métriques actuelles ne constituent pas une validation médicale.</li>
    </ul>
  </div>
</section>
<h2>Prochaines étapes</h2>
<ul>
  <li>Augmenter l'échantillon RSNA à 60 ou 90 cas.</li>
  <li>Documenter une revue visuelle complète des cas utilisés.</li>
  <li>Comparer la baseline actuelle à une baseline ML ou à un modèle multimodal.</li>
  <li>Améliorer le registre d'erreurs avec une analyse manuelle par cas.</li>
  <li>Préparer une synthèse de soutenance : choix techniques, limites, résultats et risques.</li>
</ul>
<p>
  Pages utiles :
  <a href="/report">rapport d'évaluation</a>,
  <a href="/errors">revue visuelle des erreurs</a>,
  <a href="/api/evaluation">rapport JSON</a>.
</p>
"""
    return _page("À propos et limites", body)


def render_upload_form(
    *,
    variant: str,
    model_version: str,
    error: str | None = None,
) -> str:
    """Render the image upload form for live inference."""
    error_block = ""
    if error:
        error_block = f'<p class="warning">Erreur : {_esc(error)}</p>'
    body = f"""
<p><a href="/">Retour au dashboard</a></p>
<h1>Analyser une radiographie</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
{error_block}
<section class="grid">
  <div class="card">
    <h2>Déposer une image</h2>
    <p class="muted">
      Formats acceptés : PNG, JPEG ou DICOM (<code>.dcm</code>). L'image est
      prétraitée localement (recadrage carré, normalisation) puis analysée par la
      variante <code>{_esc(variant)}</code> (<code>{_esc(model_version)}</code>).
    </p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <p><input type="file" name="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" required></p>
      <p><button type="submit">Analyser l'image</button></p>
    </form>
    <p class="muted">
      Aucune donnée patient n'est conservée : l'identifiant du cas est un
      condensé du contenu de l'image, jamais le nom de fichier.
    </p>
  </div>
</section>
"""
    return _page("Analyser une radiographie", body)


def render_upload_result(
    *,
    case_id: str,
    prediction: dict[str, Any],
    features: dict[str, Any],
    quality_reasons: list[str],
    image_url: str,
) -> str:
    """Render the live inference result for an uploaded image."""
    findings = "".join(f"<li>{_esc(item)}</li>" for item in prediction.get("visual_findings", []))
    limitations = "".join(f"<li>{_esc(item)}</li>" for item in prediction.get("limitations", []))
    reasons = "".join(f"<li>{_esc(item)}</li>" for item in quality_reasons)
    if not reasons:
        reasons = "<li>Aucun motif de qualité bloquant.</li>"

    body = f"""
<p><a href="/upload">Analyser une autre image</a> · <a href="/">Dashboard</a></p>
<h1>Résultat d'analyse</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
<section class="grid">
  <div class="card">Classe<span class="metric">{_esc(prediction.get("predicted_class"))}</span></div>
  <div class="card">Confiance<span class="metric">{_esc(prediction.get("confidence_score"))}</span></div>
  <div class="card">Qualité<span class="metric">{_esc(prediction.get("image_quality"))}</span></div>
  <div class="card">Latence<span class="metric">{_esc(prediction.get("inference_latency_ms"))} ms</span></div>
</section>
<section class="grid">
  <div class="card">
    <h2>Image prétraitée</h2>
    <img class="image-preview" src="{_esc(image_url)}" alt="Image prétraitée analysée">
  </div>
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
<p class="muted">Cas journalisé en base SQLite sous l'identifiant <code>{_esc(case_id)}</code>.</p>
<h2>Features</h2>
<pre>{_esc(json.dumps(features, ensure_ascii=False, indent=2))}</pre>
<h2>JSON complet</h2>
<pre>{_esc(json.dumps(prediction, ensure_ascii=False, indent=2))}</pre>
"""
    return _page("Résultat d'analyse", body)


def render_error_review(
    errors: list[dict[str, Any]],
    cases_by_id: dict[str, CaseRecord],
) -> str:
    """Render visual review cards for prediction errors."""
    cards = []
    for error in errors:
        case_id = str(error.get("case_id", ""))
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        image_url = f"/cases/{_esc(case.case_id)}/image"
        cards.append(
            '<article class="card">'
            f'<a href="/cases/{_esc(case.case_id)}">'
            f'<img class="error-image" src="{image_url}" alt="Image du cas {_esc(case.case_id)}">'
            "</a>"
            f"<h2>{_esc(case.case_id)}</h2>"
            f'<span class="badge">Réel: {_esc(error.get("ground_truth"))}</span>'
            f'<span class="badge">Prédit: {_esc(error.get("predicted_class"))}</span>'
            f'<span class="badge">Confiance: {_esc(error.get("confidence_score"))}</span>'
            f"<p>Type erreur: <strong>{_esc(error.get('error_type'))}</strong></p>"
            f'<p><a href="/cases/{_esc(case.case_id)}">Ouvrir le détail du cas</a></p>'
            "</article>"
        )

    if not cards:
        content = "<p>Aucune erreur à afficher visuellement.</p>"
    else:
        content = '<section class="error-grid">' + "".join(cards) + "</section>"

    body = f"""
<p><a href="/">Retour au dashboard</a></p>
<h1>Revue visuelle des erreurs</h1>
<p class="warning">{_esc(WARNING_TEXT)}</p>
<p class="muted">
  Ces images servent à analyser les erreurs du prototype. Elles ne constituent
  pas une validation clinique.
</p>
{content}
"""
    return _page("Revue visuelle des erreurs", body)
