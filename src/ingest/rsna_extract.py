"""Command-line entry point for RSNA sample extraction."""

from __future__ import annotations

import argparse
import json

from .rsna import extract_rsna_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a local sample from the RSNA Pneumonia dataset."
    )
    parser.add_argument(
        "--rsna-dir",
        required=True,
        help="Directory containing stage_2_train_images and RSNA CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/rsna_sample",
        help="Destination for copied DICOM files.",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/raw/rsna_sample_labels.csv",
        help="Labels CSV compatible with src.ingest.",
    )
    parser.add_argument(
        "--source-config",
        default="data/source.local.json",
        help="Source provenance JSON to generate.",
    )
    parser.add_argument(
        "--selected-cases-csv",
        default="data/raw/rsna_sample_selected_cases.csv",
        help="Local traceability CSV mapping copied files to RSNA patient IDs.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="Number of cases per class. Use 0 to copy all available cases.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = extract_rsna_sample(
        args.rsna_dir,
        args.output_dir,
        labels_csv=args.labels_csv,
        source_config=args.source_config,
        selected_cases_csv=args.selected_cases_csv,
        per_class=args.per_class,
        split=args.split,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "selected": result.selected,
                "skipped_missing_images": result.skipped_missing_images,
                "labels_csv": str(result.labels_csv),
                "source_config": str(result.source_config),
                "selected_cases_csv": str(result.selected_cases_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
