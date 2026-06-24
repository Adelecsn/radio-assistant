from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, PngImagePlugin
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.ingest.config import SourceConfig
from src.ingest.images import load_image
from src.ingest.pipeline import ingest_directory


def source_config() -> SourceConfig:
    return SourceConfig(
        name="synthetic-test-source",
        version="1.0",
        license="test-only",
        access_url="https://example.test/dataset",
        redistribution_allowed=False,
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_source_config_requires_complete_provenance(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"name": "incomplete"}), encoding="utf-8")

    try:
        SourceConfig.from_json(path)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Incomplete provenance should be rejected")


def test_png_ingestion_removes_metadata_and_source_name(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source_name = "patient_jane_doe_1980.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("PatientName", "Jane Doe")
    Image.new("L", (320, 280), color=120).save(raw / source_name, pnginfo=metadata)
    # Add a small gradient so robust intensity normalization has a usable range.
    array = np.tile(np.arange(320, dtype=np.uint16) % 256, (280, 1)).astype(np.uint8)
    Image.fromarray(array).save(raw / source_name, pnginfo=metadata)

    manifest = tmp_path / "manifests" / "ingest.csv"
    result = ingest_directory(raw, tmp_path / "processed", manifest, source_config())
    rows = read_manifest(manifest)

    assert result.processed == 1
    assert rows[0]["case_id"].startswith("case_")
    assert source_name not in manifest.read_text(encoding="utf-8")
    assert rows[0]["privacy_review_status"] == "pending_manual_pixel_review"

    output = (manifest.parent / rows[0]["processed_path"]).resolve()
    with Image.open(output) as image:
        assert image.size == (512, 512)
        assert image.mode == "L"
        assert "PatientName" not in image.info


def test_labels_are_joined_but_source_filename_is_not_exported(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    array = np.arange(300 * 300, dtype=np.uint32).reshape(300, 300) % 256
    Image.fromarray(array.astype(np.uint8)).save(raw / "private-id.png")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "source_filename,label,split\nprivate-id.png,normal,train\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"

    ingest_directory(
        raw,
        tmp_path / "processed",
        manifest,
        source_config(),
        labels_csv=labels,
    )
    row = read_manifest(manifest)[0]

    assert row["label"] == "normal"
    assert row["split"] == "train"
    assert "private-id.png" not in manifest.read_text(encoding="utf-8")


def test_duplicate_content_is_written_once(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    array = np.arange(300 * 300, dtype=np.uint32).reshape(300, 300) % 256
    image = Image.fromarray(array.astype(np.uint8))
    image.save(raw / "first.png")
    image.save(raw / "second.png")

    result = ingest_directory(
        raw,
        tmp_path / "processed",
        tmp_path / "manifest.csv",
        source_config(),
    )

    assert result.processed == 1
    assert result.duplicates == 1


def test_dicom_pixels_are_loaded_without_exporting_patient_metadata(tmp_path: Path) -> None:
    path = tmp_path / "patient-name.dcm"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Sensitive^Name"
    dataset.Rows = 300
    dataset.Columns = 280
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.BurnedInAnnotation = "NO"
    pixels = np.arange(300 * 280, dtype=np.uint16).reshape(300, 280) % 4096
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, write_like_original=False)

    loaded = load_image(path)

    assert loaded.original_format == "dicom"
    assert loaded.image.size == (280, 300)
    assert loaded.burned_in_annotation == "NO"


def test_ingestion_cli_runs_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    array = np.arange(300 * 300, dtype=np.uint32).reshape(300, 300) % 256
    Image.fromarray(array.astype(np.uint8)).save(raw / "case.png")
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {
                "name": "cli-test",
                "version": "1",
                "license": "test-only",
                "access_url": "https://example.test/cli",
                "redistribution_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifests" / "ingest.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ingest",
            "--input-dir",
            str(raw),
            "--output-dir",
            str(tmp_path / "processed"),
            "--manifest",
            str(manifest),
            "--source-config",
            str(config),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["processed"] == 1
    assert manifest.exists()
