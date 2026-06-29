"""Run the local web dashboard."""

from __future__ import annotations

import argparse

from .app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Radio Assistant local dashboard.")
    parser.add_argument(
        "--predictions-dir",
        default="data/predictions/baseline_v1",
        help="Directory produced by python -m src.inference.",
    )
    parser.add_argument(
        "--db-path",
        default="logs/webapp.sqlite",
        help="SQLite database used to log local consultations.",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "improved", "medgemma"),
        default="improved",
        help="Inference variant used for live uploads (POST /predict).",
    )
    parser.add_argument(
        "--upload-dir",
        default="logs/uploads",
        help="Directory for preprocessed uploaded images (kept out of Git).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="MedGemma live uploads only: system prompt file. Use "
        "prompts/improved_v1.txt for real predictions (baseline_v1 returns uncertain).",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="MedGemma live uploads only: prompt version label logged with events.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "uvicorn n'est pas installé. Installe requirements.txt pour lancer la webapp."
        ) from error

    app = create_app(
        predictions_dir=args.predictions_dir,
        db_path=args.db_path,
        variant=args.variant,
        upload_dir=args.upload_dir,
        prompt_path=args.prompt_file,
        prompt_version=args.prompt_version,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
