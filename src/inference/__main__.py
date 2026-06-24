"""Command line entry point for batch inference."""

from __future__ import annotations

import argparse
import json

from .config import InferenceConfig
from .pipeline import run_batch_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline inference on an ingest manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the ingestion manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON predictions.")
    parser.add_argument("--model-version", default=InferenceConfig.model_version)
    parser.add_argument("--prompt-version", default=InferenceConfig.prompt_version)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=InferenceConfig.confidence_threshold,
    )
    parser.add_argument("--opacity-threshold", type=float, default=InferenceConfig.opacity_threshold)
    parser.add_argument("--normal-threshold", type=float, default=InferenceConfig.normal_threshold)
    parser.add_argument(
        "--poor-quality-std-threshold",
        type=float,
        default=InferenceConfig.poor_quality_std_threshold,
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=InferenceConfig.min_foreground_ratio,
    )
    parser.add_argument(
        "--bright-pixel-threshold",
        type=int,
        default=InferenceConfig.bright_pixel_threshold,
    )
    parser.add_argument("--edge-threshold", type=float, default=InferenceConfig.edge_threshold)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = InferenceConfig(
        model_version=args.model_version,
        prompt_version=args.prompt_version,
        confidence_threshold=args.confidence_threshold,
        opacity_threshold=args.opacity_threshold,
        normal_threshold=args.normal_threshold,
        poor_quality_std_threshold=args.poor_quality_std_threshold,
        min_foreground_ratio=args.min_foreground_ratio,
        bright_pixel_threshold=args.bright_pixel_threshold,
        edge_threshold=args.edge_threshold,
    )
    result = run_batch_inference(args.manifest, args.output_dir, config=config)
    print(
        json.dumps(
            {
                "processed": result.processed,
                "rejected": result.rejected,
                "output_dir": str(result.output_dir),
                "index_path": str(result.index_path),
                "metadata_path": str(result.metadata_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
