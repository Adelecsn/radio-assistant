"""Command-line entry point for prediction evaluation."""

from __future__ import annotations

import argparse
import json

from .metrics import evaluate_prediction_dir, write_evaluation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate labeled prediction JSON outputs.")
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory produced by python -m src.inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/outputs/baseline_v1",
        help="Directory for metrics_report.json and error_register.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation = evaluate_prediction_dir(args.predictions_dir)
    outputs = write_evaluation_outputs(evaluation, args.output_dir)
    print(
        json.dumps(
            {
                "evaluated_cases": evaluation["evaluated_cases"],
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "report_json": str(outputs["report_json"]),
                "error_register_csv": str(outputs["error_register_csv"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
