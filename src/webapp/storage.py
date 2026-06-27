"""SQLite logging for local webapp consultations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3

from .repository import CaseRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS case_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    image_quality TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    viewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_case_views_case_id
ON case_views(case_id);

CREATE TABLE IF NOT EXISTS inference_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    source TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    image_quality TEXT NOT NULL,
    confidence_score REAL,
    model_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    inference_latency_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inference_events_case_id
ON inference_events(case_id);
"""


def init_db(path: str | Path) -> None:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA)


def log_case_view(path: str | Path, case: CaseRecord) -> None:
    init_db(path)
    prediction = case.prediction
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO case_views (
                case_id,
                predicted_class,
                image_quality,
                model_version,
                prompt_version
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                case.case_id,
                str(prediction.get("predicted_class", "")),
                str(prediction.get("image_quality", "")),
                str(prediction.get("model_version", "")),
                str(prediction.get("prompt_version", "")),
            ),
        )


def log_inference_event(
    path: str | Path,
    *,
    case_id: str,
    prediction: dict[str, object],
    source: str = "upload",
) -> None:
    """Log a live inference (e.g. an uploaded image) without any patient data."""
    init_db(path)
    confidence = prediction.get("confidence_score")
    latency = prediction.get("inference_latency_ms")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO inference_events (
                case_id,
                source,
                predicted_class,
                image_quality,
                confidence_score,
                model_version,
                prompt_version,
                inference_latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                source,
                str(prediction.get("predicted_class", "")),
                str(prediction.get("image_quality", "")),
                float(confidence) if isinstance(confidence, (int, float)) else None,
                str(prediction.get("model_version", "")),
                str(prediction.get("prompt_version", "")),
                int(latency) if isinstance(latency, int) else None,
            ),
        )


def read_log_summary(path: str | Path) -> dict[str, object]:
    database = Path(path)
    if not database.is_file():
        return {
            "total_views": 0,
            "last_viewed_at": None,
            "views_by_class": {},
            "total_inferences": 0,
            "inferences_by_class": {},
        }

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT predicted_class, viewed_at FROM case_views ORDER BY viewed_at"
        ).fetchall()
        inference_rows = connection.execute(
            "SELECT predicted_class FROM inference_events"
        ).fetchall()

    counts = Counter(str(row[0]) for row in rows)
    inference_counts = Counter(str(row[0]) for row in inference_rows)
    last_viewed_at = rows[-1][1] if rows else None
    return {
        "total_views": len(rows),
        "last_viewed_at": last_viewed_at,
        "views_by_class": dict(sorted(counts.items())),
        "total_inferences": len(inference_rows),
        "inferences_by_class": dict(sorted(inference_counts.items())),
    }
