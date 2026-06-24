"""Utilities for preparing a local RSNA Pneumonia sample."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.contracts import ALLOWED_CLASSES


RSNA_SOURCE_CONFIG = {
    "name": "RSNA Pneumonia Detection Challenge",
    "version": "stage_2",
    "license": "Kaggle competition terms / RSNA Pneumonia Challenge terms",
    "access_url": "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data",
    "redistribution_allowed": False,
}


@dataclass(frozen=True)
class RsnaExtractionResult:
    selected: int
    skipped_missing_images: int
    labels_csv: Path
    source_config: Path
    selected_cases_csv: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _labels_from_rsna(root: Path) -> dict[str, str]:
    labels_path = root / "stage_2_train_labels.csv"
    class_info_path = root / "stage_2_detailed_class_info.csv"
    if not labels_path.is_file():
        raise ValueError(f"Missing RSNA labels file: {labels_path}")

    target_by_patient: dict[str, int] = {}
    for row in _read_csv(labels_path):
        patient_id = row["patientId"].strip()
        target = int(float(row["Target"]))
        target_by_patient[patient_id] = max(target_by_patient.get(patient_id, 0), target)

    detailed_class: dict[str, str] = {}
    if class_info_path.is_file():
        for row in _read_csv(class_info_path):
            detailed_class[row["patientId"].strip()] = row["class"].strip()

    labels: dict[str, str] = {}
    for patient_id, target in target_by_patient.items():
        if target == 1:
            labels[patient_id] = "suspected_opacity"
        elif detailed_class.get(patient_id) == "Normal":
            labels[patient_id] = "normal"
        else:
            labels[patient_id] = "uncertain"
    return labels


def _select_cases(labels: dict[str, str], per_class: int, seed: int) -> list[tuple[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for patient_id, label in labels.items():
        if label not in ALLOWED_CLASSES:
            continue
        grouped[label].append(patient_id)

    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label in ("normal", "suspected_opacity", "uncertain"):
        patient_ids = sorted(grouped[label])
        rng.shuffle(patient_ids)
        chosen = patient_ids if per_class <= 0 else patient_ids[:per_class]
        selected.extend((patient_id, label) for patient_id in chosen)
    return selected


def _write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_filename", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_cases(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source_filename", "rsna_patient_id", "label", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_rsna_sample(
    rsna_dir: str | Path,
    output_dir: str | Path,
    *,
    labels_csv: str | Path,
    source_config: str | Path,
    selected_cases_csv: str | Path,
    per_class: int = 10,
    split: str = "validation",
    seed: int = 42,
) -> RsnaExtractionResult:
    """Copy a balanced RSNA DICOM sample and write labels for the ingest pipeline."""
    root = Path(rsna_dir)
    image_dir = root / "stage_2_train_images"
    if not image_dir.is_dir():
        raise ValueError(
            "Missing RSNA image directory. Expected: "
            f"{image_dir}. Download and unzip the Kaggle dataset first."
        )
    if not split.strip():
        raise ValueError("split must be non-empty")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = _labels_from_rsna(root)
    selected_cases = _select_cases(labels, per_class=per_class, seed=seed)

    label_rows: list[dict[str, str]] = []
    case_rows: list[dict[str, str]] = []
    skipped = 0
    copied = 0

    for patient_id, label in selected_cases:
        source_image = image_dir / f"{patient_id}.dcm"
        if not source_image.is_file():
            skipped += 1
            continue

        output_name = f"rsna_{copied + 1:05d}.dcm"
        shutil.copy2(source_image, destination / output_name)
        label_rows.append({"source_filename": output_name, "label": label, "split": split})
        case_rows.append(
            {
                "source_filename": output_name,
                "rsna_patient_id": patient_id,
                "label": label,
                "split": split,
            }
        )
        copied += 1

    _write_labels(Path(labels_csv), label_rows)
    _write_selected_cases(Path(selected_cases_csv), case_rows)
    Path(source_config).parent.mkdir(parents=True, exist_ok=True)
    Path(source_config).write_text(
        json.dumps(RSNA_SOURCE_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return RsnaExtractionResult(
        selected=copied,
        skipped_missing_images=skipped,
        labels_csv=Path(labels_csv),
        source_config=Path(source_config),
        selected_cases_csv=Path(selected_cases_csv),
    )
