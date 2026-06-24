"""Read prediction artifacts produced by the inference pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    case_json_path: Path
    prediction: dict[str, Any]
    features: dict[str, Any]
    split: str
    label: str
    processed_path: str
    quality_reasons: list[str]


class PredictionRepository:
    """Repository over local JSON outputs, without model or web framework logic."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / "predictions.jsonl"
        self.metadata_path = self.output_dir / "run_metadata.json"

    def metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def list_cases(self) -> list[CaseRecord]:
        if not self.index_path.is_file():
            return []

        cases: list[CaseRecord] = []
        with self.index_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                case = self._case_from_index_record(record)
                if case is not None:
                    cases.append(case)
        return cases

    def get_case(self, case_id: str) -> CaseRecord | None:
        for case in self.list_cases():
            if case.case_id == case_id:
                return case
        return None

    def summary(self) -> dict[str, Any]:
        cases = self.list_cases()
        class_counts = Counter(case.prediction.get("predicted_class", "unknown") for case in cases)
        quality_counts = Counter(case.prediction.get("image_quality", "unknown") for case in cases)
        confidences = [
            float(case.prediction["confidence_score"])
            for case in cases
            if isinstance(case.prediction.get("confidence_score"), (int, float))
        ]
        latencies = [
            int(case.prediction["inference_latency_ms"])
            for case in cases
            if isinstance(case.prediction.get("inference_latency_ms"), int)
        ]

        labeled_cases = [case for case in cases if case.label]
        correct = sum(
            1 for case in labeled_cases if case.label == case.prediction.get("predicted_class")
        )

        return {
            "total_cases": len(cases),
            "by_class": dict(sorted(class_counts.items())),
            "by_quality": dict(sorted(quality_counts.items())),
            "average_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences
            else None,
            "average_latency_ms": round(sum(latencies) / len(latencies), 1)
            if latencies
            else None,
            "labeled_cases": len(labeled_cases),
            "baseline_accuracy": round(correct / len(labeled_cases), 3)
            if labeled_cases
            else None,
            "model_versions": sorted(
                {
                    str(case.prediction.get("model_version"))
                    for case in cases
                    if case.prediction.get("model_version")
                }
            ),
            "prompt_versions": sorted(
                {
                    str(case.prediction.get("prompt_version"))
                    for case in cases
                    if case.prediction.get("prompt_version")
                }
            ),
        }

    def resolve_image_path(self, case: CaseRecord) -> Path | None:
        if not case.processed_path:
            return None
        processed = Path(case.processed_path)
        if processed.is_absolute():
            return processed

        manifest_path = self.metadata().get("manifest_path")
        if not manifest_path:
            return None

        manifest = Path(str(manifest_path))
        if not manifest.is_absolute():
            manifest = (Path.cwd() / manifest).resolve()
        return (manifest.parent / processed).resolve()

    def _case_from_index_record(self, record: dict[str, Any]) -> CaseRecord | None:
        case_id = str(record.get("case_id", "")).strip()
        if not case_id:
            return None

        case_json = record.get("case_json")
        case_json_path = self.output_dir / str(case_json) if case_json else None
        if case_json_path and case_json_path.is_file():
            payload = json.loads(case_json_path.read_text(encoding="utf-8"))
            prediction = payload.get("prediction", {})
            features = payload.get("features", {})
            return CaseRecord(
                case_id=case_id,
                case_json_path=case_json_path,
                prediction=prediction if isinstance(prediction, dict) else {},
                features=features if isinstance(features, dict) else {},
                split=str(payload.get("split", "")),
                label=str(payload.get("label", "")),
                processed_path=str(payload.get("processed_path", "")),
                quality_reasons=list(payload.get("quality_reasons", [])),
            )

        prediction = record.get("prediction", {})
        features = record.get("features", {})
        return CaseRecord(
            case_id=case_id,
            case_json_path=self.output_dir / "cases" / f"{case_id}.json",
            prediction=prediction if isinstance(prediction, dict) else {},
            features=features if isinstance(features, dict) else {},
            split="",
            label="",
            processed_path="",
            quality_reasons=[],
        )
