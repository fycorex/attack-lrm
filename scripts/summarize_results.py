#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import write_metrics_bundle
from src.utils.io import read_jsonl
from src.utils.schemas import DialogueSummaryRecord, TurnRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a defensive safety evaluation run.")
    parser.add_argument(
        "--input",
        required=True,
        help="Run directory, for example outputs/latest or outputs/run_20260101T000000Z_abcd1234.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input).resolve()
    turn_records_path = input_dir / "turn_records.jsonl"
    dialogue_summaries_path = input_dir / "dialogue_summaries.jsonl"
    if not turn_records_path.exists() or not dialogue_summaries_path.exists():
        raise SystemExit(
            f"Expected {turn_records_path.name} and {dialogue_summaries_path.name} in {input_dir}"
        )

    turn_records = [
        TurnRecord.model_validate(row)
        for row in read_jsonl(turn_records_path)
    ]
    dialogue_summaries = [
        DialogueSummaryRecord.model_validate(row)
        for row in read_jsonl(dialogue_summaries_path)
    ]
    metrics = write_metrics_bundle(
        output_dir=str(input_dir),
        turn_records=turn_records,
        dialogue_summaries=dialogue_summaries,
    )

    overall = metrics["overall"]
    print(f"input_dir={input_dir}")
    print(f"turn_count={overall['turn_count']}")
    print(f"robust_refusal_rate={overall['robust_refusal_rate']}")
    print(f"disclaimer_leakage_rate={overall['disclaimer_leakage_rate']}")
    print(f"mean_policy_drift_score={overall['mean_policy_drift_score']}")
    print(f"mean_overcompliance_score={overall['mean_overcompliance_score']}")
    print(f"safe_redirection_rate={overall['safe_redirection_rate']}")
    print(f"cross_model_gap_pairs={len(metrics['cross_model_vulnerability_gap'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

