"""Batch ingestion pipeline and de-identified manifest generation."""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from src.contracts import ALLOWED_CLASSES

from .config import SourceConfig
from .images import SUPPORTED_SUFFIXES, load_image, resize_with_padding


MANIFEST_FIELDS = [
    "case_id",
    "processed_path",
    "source_name",
    "source_version",
    "source_license",
    "source_access_url",
    "redistribution_allowed",
    "split",
    "label",
    "original_format",
    "original_width",
    "original_height",
    "processed_width",
    "processed_height",
    "content_sha256",
    "technical_review",
    "burned_in_annotation",
    "privacy_review_status",
]


@dataclass(frozen=True)
class IngestionResult:
    processed: int
    rejected: int
    duplicates: int
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _load_labels(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_filename", "label", "split"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Labels CSV must contain columns: {sorted(required)}")

    labels: dict[str, tuple[str, str]] = {}
    for row in rows:
        label = row["label"].strip()
        split = row["split"].strip()
        if label not in ALLOWED_CLASSES:
            raise ValueError(f"Invalid label in labels CSV: {label}")
        if not split:
            raise ValueError("Every labels CSV row must define a split")
        labels[row["source_filename"]] = (label, split)
    return labels


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def ingest_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    source: SourceConfig,
    *,
    labels_csv: str | Path | None = None,
    target_size: int = 512,
) -> IngestionResult:
    """Preprocess images and write a manifest without source filenames."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    manifest = Path(manifest_path)
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")

    files = _discover(input_path)
    if not files:
        raise ValueError("No supported image found in the input directory")

    labels = _load_labels(Path(labels_csv) if labels_csv else None)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    rejected = 0
    duplicates = 0

    for source_path in files:
        content_hash = _sha256(source_path)
        if content_hash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(content_hash)

        try:
            loaded = load_image(source_path)
            processed = resize_with_padding(loaded.image, target_size)
        except (OSError, RuntimeError, ValueError):
            rejected += 1
            continue

        case_id = f"case_{content_hash[:16]}"
        destination = image_dir / f"{case_id}.png"
        processed.save(destination, format="PNG", optimize=True)
        label, split = labels.get(source_path.name, ("", "unassigned"))
        processed_path = os.path.relpath(destination, start=manifest.parent)

        records.append(
            {
                "case_id": case_id,
                "processed_path": Path(processed_path).as_posix(),
                "source_name": source.name,
                "source_version": source.version,
                "source_license": source.license,
                "source_access_url": source.access_url,
                "redistribution_allowed": str(source.redistribution_allowed).lower(),
                "split": split,
                "label": label,
                "original_format": loaded.original_format,
                "original_width": loaded.original_width,
                "original_height": loaded.original_height,
                "processed_width": target_size,
                "processed_height": target_size,
                "content_sha256": content_hash,
                "technical_review": loaded.technical_review,
                "burned_in_annotation": loaded.burned_in_annotation,
                "privacy_review_status": "pending_manual_pixel_review",
            }
        )

    _write_manifest(manifest, records)
    return IngestionResult(
        processed=len(records),
        rejected=rejected,
        duplicates=duplicates,
        manifest_path=manifest,
    )
