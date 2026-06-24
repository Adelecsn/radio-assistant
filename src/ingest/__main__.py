"""Command-line entry point for the ingestion pipeline."""

from __future__ import annotations

import argparse
import json

from .config import SourceConfig
from .pipeline import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest authorized chest X-ray images")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--labels-csv")
    parser.add_argument("--target-size", type=int, default=512)
    args = parser.parse_args()

    source = SourceConfig.from_json(args.source_config)
    result = ingest_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        source=source,
        labels_csv=args.labels_csv,
        target_size=args.target_size,
    )
    print(
        json.dumps(
            {
                "processed": result.processed,
                "rejected": result.rejected,
                "duplicates": result.duplicates,
                "manifest": str(result.manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
