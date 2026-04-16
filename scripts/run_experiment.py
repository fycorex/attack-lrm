#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.openai_compatible import OpenAICompatibleAdapter
from src.datasets.safe_boundary_scenarios import get_scenario
from src.evaluation.judge import JudgeEnsemble, SafetyJudge, build_default_prompts
from src.evaluation.metrics import write_metrics_bundle
from src.orchestrator.run_dialogue import run_dialogue
from src.orchestrator.run_matrix import run_matrix
from src.utils.io import (
    generate_run_id,
    load_experiment_config,
    load_models_config,
    load_prompts_config,
    make_output_directory,
    refresh_latest_pointer,
    resolve_config_path,
    write_csv,
    write_json,
)
from src.utils.logging import configure_logger
from src.utils.schemas import ModelProfile, PromptConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run defensive multi-turn cross-model safety evaluations.",
    )
    parser.add_argument("--config", help="Path to matrix experiment YAML config.")
    parser.add_argument("--models-config", help="Path to model profile YAML config.")
    parser.add_argument(
        "--prompts-config",
        help="Path to prompts YAML config.",
    )
    parser.add_argument("--probe-model", help="Direct probe model name for single-run mode.")
    parser.add_argument("--target-model", help="Direct target model name for single-run mode.")
    parser.add_argument(
        "--judge-model",
        help="Direct judge model name for single-run mode. Use a comma-separated list for a judge ensemble.",
    )
    parser.add_argument("--scenario", help="Scenario id for single-run mode, for example s001.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL override.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_COMPAT_API_KEY",
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for model calls.")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum scored turns.")
    parser.add_argument(
        "--stop-on-risk-score",
        type=int,
        choices=range(1, 6),
        help="Single-run mode only. Stop early once max(policy_drift_score, overcompliance_score) reaches this value.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--output-dir",
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Optional custom header in Key=Value form. Repeatable.",
    )
    return parser


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in raw_headers:
        if "=" not in item:
            raise ValueError(f"Invalid --header value: {item}")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def load_prompts(path: str | None) -> PromptConfig:
    if path and Path(path).exists():
        return load_prompts_config(path)
    return PromptConfig.model_validate(build_default_prompts())


def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logger()

    if args.config:
        experiment_path = Path(args.config).resolve()
        experiment = load_experiment_config(experiment_path)
        models_path = (
            Path(args.models_config).resolve()
            if args.models_config
            else resolve_config_path(experiment_path, experiment.models_config)
        )
        prompts_path = (
            Path(args.prompts_config).resolve()
            if args.prompts_config
            else resolve_config_path(experiment_path, experiment.prompts_config)
        )

        if args.max_turns != 10:
            experiment.max_turns = args.max_turns
        if args.output_dir:
            experiment.output_root = args.output_dir

        models_config = load_models_config(models_path)
        prompts = load_prompts(str(prompts_path))
        run_id = generate_run_id("matrix")
        output_dir = make_output_directory(experiment.output_root, run_id)
        result = run_matrix(
            run_id=run_id,
            output_dir=output_dir,
            experiment=experiment,
            models_config=models_config,
            prompts=prompts,
        )
        logger.info("Completed matrix run %s in %s", result["run_id"], result["output_dir"])
        return 0

    scenario_id = args.scenario
    probe_model = args.probe_model or os.getenv("PROBE_MODEL")
    target_model = args.target_model or os.getenv("TARGET_MODEL")
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL")
    if not scenario_id or not probe_model or not target_model or not judge_model:
        raise SystemExit(
            "Single-run mode requires --scenario plus probe/target/judge model names "
            "via CLI or PROBE_MODEL/TARGET_MODEL/JUDGE_MODEL environment variables."
        )
    judge_models = [item.strip() for item in judge_model.split(",") if item.strip()]

    prompts = load_prompts(args.prompts_config or str(ROOT / "config" / "prompts.example.yaml"))
    headers = parse_headers(args.header)
    output_root = args.output_dir or str(ROOT / "outputs")

    probe_profile = ModelProfile(
        base_url=args.base_url,
        base_url_env=None if args.base_url else "OPENAI_COMPAT_BASE_URL",
        api_key_env=args.api_key_env,
        model=probe_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        headers=headers,
    )
    target_profile = ModelProfile(
        base_url=args.base_url,
        base_url_env=None if args.base_url else "OPENAI_COMPAT_BASE_URL",
        api_key_env=args.api_key_env,
        model=target_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        headers=headers,
    )
    run_id = generate_run_id("dialogue")
    output_dir = make_output_directory(output_root, run_id)
    write_json(
        output_dir / "manifest.json",
        {
            "run_id": run_id,
            "mode": "single",
            "paper_reference": "arXiv:2508.04039 / Nature Communications (2026) structural alignment",
            "scenario_id": scenario_id,
            "max_turns": args.max_turns,
            "stop_on_risk_score": args.stop_on_risk_score,
            "probe_model": probe_model,
            "target_model": target_model,
            "judge_models": judge_models,
        },
    )

    probe_adapter = OpenAICompatibleAdapter("probe_single", probe_profile)
    target_adapter = OpenAICompatibleAdapter("target_single", target_profile)
    judges: list[SafetyJudge] = []
    for index, model_name in enumerate(judge_models, start=1):
        judge_profile = ModelProfile(
            base_url=args.base_url,
            base_url_env=None if args.base_url else "OPENAI_COMPAT_BASE_URL",
            api_key_env=args.api_key_env,
            model=model_name,
            timeout=args.timeout,
            max_retries=args.max_retries,
            temperature=0.0,
            headers=headers,
            response_format_json=True,
        )
        judge_adapter = OpenAICompatibleAdapter(f"judge_single_{index}", judge_profile)
        judges.append(
            SafetyJudge(
                adapter=judge_adapter,
                turn_system_prompt=prompts.judge_turn_system_prompt,
                dialogue_system_prompt=prompts.judge_dialogue_system_prompt,
            )
        )
    judge = judges[0] if len(judges) == 1 else JudgeEnsemble(judges)
    result = run_dialogue(
        run_id=run_id,
        output_dir=output_dir,
        scenario=get_scenario(scenario_id),
        probe_adapter=probe_adapter,
        target_adapter=target_adapter,
        judge=judge,
        prompts=prompts,
        max_turns=args.max_turns,
        stop_on_risk_score=args.stop_on_risk_score,
    )

    write_csv(
        output_dir / "turn_records.csv",
        [record.model_dump(mode="json") for record in result.turn_records],
    )
    write_csv(
        output_dir / "dialogue_summaries.csv",
        [result.dialogue_summary.model_dump(mode="json")],
    )
    write_metrics_bundle(
        output_dir=str(output_dir),
        turn_records=result.turn_records,
        dialogue_summaries=[result.dialogue_summary],
    )
    refresh_latest_pointer(output_root, output_dir)
    logger.info("Completed single dialogue run %s in %s", run_id, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
