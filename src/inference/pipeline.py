"""Batch inference pipeline that stores JSON outputs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import predict_image
from .config import InferenceConfig


@dataclass(frozen=True)
class BatchInferenceResult:
    processed: int
    rejected: int
    output_dir: Path
    index_path: Path
    metadata_path: Path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"case_id", "processed_path"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows


def _resolve_processed_path(manifest_path: Path, processed_path: str) -> Path:
    path = Path(processed_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_batch_inference(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: InferenceConfig | None = None,
) -> BatchInferenceResult:
    """Run inference for every manifest row and store per-case JSON files."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest}")

    active_config = config or InferenceConfig()
    destination = Path(output_dir)
    cases_dir = destination / "cases"
    index_path = destination / "predictions.jsonl"
    metadata_path = destination / "run_metadata.json"
    rows = _read_manifest(manifest)

    processed = 0
    rejected = 0
    index_records: list[dict[str, Any]] = []

    for row in rows:
        case_id = row["case_id"]
        image_path = _resolve_processed_path(manifest, row["processed_path"])
        if not image_path.is_file():
            rejected += 1
            continue

        try:
            result = predict_image(image_path, active_config)
        except (OSError, ValueError):
            rejected += 1
            continue

        case_path = cases_dir / f"{case_id}.json"
        case_record: dict[str, Any] = {
            "case_id": case_id,
            "processed_path": row["processed_path"],
            "split": row.get("split", ""),
            "label": row.get("label", ""),
            "prediction": result.prediction,
            "features": result.features,
            "quality_reasons": result.quality_reasons,
        }
        _write_json(case_path, case_record)
        index_records.append(
            {
                "case_id": case_id,
                "case_json": case_path.relative_to(destination).as_posix(),
                "prediction": result.prediction,
                "features": result.features,
            }
        )
        processed += 1

    destination.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary_index.open("w", encoding="utf-8") as stream:
        for record in index_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_index, index_path)

    metadata = {
        "manifest_path": str(manifest.resolve()),
        "contract_version": "prediction-v1",
        "processed": processed,
        "rejected": rejected,
        "hyperparameters": active_config.to_dict(),
    }
    _write_json(metadata_path, metadata)

    return BatchInferenceResult(
        processed=processed,
        rejected=rejected,
        output_dir=destination,
        index_path=index_path,
        metadata_path=metadata_path,
    )
