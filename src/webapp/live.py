"""Live inference for uploaded images, reusing the ingest and inference modules.

This powers the web upload required by the call for tenders ("démo web avec
upload"): an image enters, it is deterministically preprocessed (same path as the
batch pipeline), analysed by the selected variant and returned as a contract JSON.
No source metadata is retained; the case id is a content hash, never a filename.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile
from typing import Any

from src.ingest.images import load_image, resize_with_padding
from src.inference import InferenceConfig, predict_case


@dataclass(frozen=True)
class LiveResult:
    case_id: str
    prediction: dict[str, Any]
    features: dict[str, Any]
    quality_reasons: list[str]
    processed_image_path: Path


def _sniff_suffix(data: bytes, filename: str = "") -> str:
    """Detect the image format from content (safer than trusting the filename)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(data) >= 132 and data[128:132] == b"DICM":
        return ".dcm"
    # Some DICOM files omit the 128-byte preamble; fall back to the declared name.
    name_suffix = Path(filename).suffix.lower()
    if name_suffix in {".dcm", ".dicom"}:
        return ".dcm"
    raise ValueError(
        "Format non supporté ou illisible. Formats acceptés : PNG, JPEG ou DICOM (.dcm)."
    )


def analyze_upload(
    data: bytes,
    filename: str = "",
    *,
    config: InferenceConfig | None = None,
    upload_dir: str | Path = "logs/uploads",
    image_size: int = 512,
) -> LiveResult:
    """Preprocess and analyse an uploaded image, returning a contract prediction."""
    if not data:
        raise ValueError("Le fichier reçu est vide.")

    suffix = _sniff_suffix(data, filename)
    active_config = config if config is not None else InferenceConfig.improved()
    case_id = "upload_" + sha256(data).hexdigest()[:16]
    destination = Path(upload_dir)
    destination.mkdir(parents=True, exist_ok=True)
    processed_path = destination / f"{case_id}.png"

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / f"upload{suffix}"
        raw_path.write_bytes(data)
        loaded = load_image(raw_path)
        processed = resize_with_padding(loaded.image, size=image_size)
        processed.save(processed_path)

    result = predict_case(processed_path, active_config)
    return LiveResult(
        case_id=case_id,
        prediction=result.prediction,
        features=result.features,
        quality_reasons=result.quality_reasons,
        processed_image_path=processed_path,
    )
