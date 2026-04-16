#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Qianfan-friendly model and experiment configs from a small list of model names.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model names available on the endpoint, for example kimi-k2.5,deepseek-v3.2,minimax-m2.5",
    )
    parser.add_argument(
        "--probe-models",
        help="Optional comma-separated probe models. Defaults to --models.",
    )
    parser.add_argument(
        "--target-models",
        help="Optional comma-separated target models. Defaults to --models.",
    )
    parser.add_argument(
        "--judge-models",
        help="Optional comma-separated judge models. Defaults to the first 3 models from --models.",
    )
    parser.add_argument(
        "--base-url-env",
        default="OPENAI_COMPAT_BASE_URL",
        help="Environment variable name holding the shared OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_COMPAT_API_KEY",
        help="Environment variable name holding the shared API key.",
    )
    parser.add_argument(
        "--prefix",
        default="qianfan_selected",
        help="Output filename prefix under config/generated/.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum scored turns for the generated experiment config.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=70,
        help="Number of safe benchmark scenarios to include from the start of the paper-shaped set.",
    )
    return parser


def parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return slug or "model"


def make_models_payload(
    *,
    probe_models: list[str],
    target_models: list[str],
    judge_models: list[str],
    base_url_env: str,
    api_key_env: str,
) -> dict[str, object]:
    profiles: dict[str, object] = {}

    for model_name in probe_models:
        profiles[f"probe_{slugify_model_name(model_name)}"] = {
            "base_url_env": base_url_env,
            "api_key_env": api_key_env,
            "model": model_name,
            "timeout": 45,
            "max_retries": 3,
            "temperature": 0.0,
        }

    for model_name in target_models:
        profiles[f"target_{slugify_model_name(model_name)}"] = {
            "base_url_env": base_url_env,
            "api_key_env": api_key_env,
            "model": model_name,
            "timeout": 45,
            "max_retries": 3,
            "temperature": 0.0,
        }

    for model_name in judge_models:
        profiles[f"judge_{slugify_model_name(model_name)}"] = {
            "base_url_env": base_url_env,
            "api_key_env": api_key_env,
            "model": model_name,
            "timeout": 60,
            "max_retries": 3,
            "temperature": 0.0,
            "response_format_json": True,
        }

    return {"profiles": profiles}


def make_experiment_payload(
    *,
    prefix: str,
    models_filename: str,
    probe_models: list[str],
    target_models: list[str],
    judge_models: list[str],
    max_turns: int,
    scenario_count: int,
) -> dict[str, object]:
    scenario_ids = [f"s{index:03d}" for index in range(1, scenario_count + 1)]
    return {
        "experiment_name": prefix,
        "models_config": models_filename,
        "prompts_config": "../prompts.example.yaml",
        "output_root": "outputs",
        "max_turns": max_turns,
        "probe_profiles": [f"probe_{slugify_model_name(model_name)}" for model_name in probe_models],
        "target_profiles": [f"target_{slugify_model_name(model_name)}" for model_name in target_models],
        "judge_profiles": [f"judge_{slugify_model_name(model_name)}" for model_name in judge_models],
        "scenario_ids": scenario_ids,
    }


def main() -> int:
    args = build_parser().parse_args()
    shared_models = parse_model_list(args.models)
    if not shared_models:
        raise SystemExit("--models must contain at least one model name.")

    probe_models = parse_model_list(args.probe_models) or shared_models
    target_models = parse_model_list(args.target_models) or shared_models
    judge_models = parse_model_list(args.judge_models) or shared_models[: min(3, len(shared_models))]

    if not judge_models:
        raise SystemExit("At least one judge model is required.")
    if not 1 <= args.scenario_count <= 70:
        raise SystemExit("--scenario-count must be between 1 and 70.")

    generated_dir = ROOT / "config" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    models_filename = f"{args.prefix}.models.yaml"
    experiments_filename = f"{args.prefix}.experiments.yaml"
    models_path = generated_dir / models_filename
    experiments_path = generated_dir / experiments_filename

    models_payload = make_models_payload(
        probe_models=probe_models,
        target_models=target_models,
        judge_models=judge_models,
        base_url_env=args.base_url_env,
        api_key_env=args.api_key_env,
    )
    experiments_payload = make_experiment_payload(
        prefix=args.prefix,
        models_filename=models_filename,
        probe_models=probe_models,
        target_models=target_models,
        judge_models=judge_models,
        max_turns=args.max_turns,
        scenario_count=args.scenario_count,
    )

    with models_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(models_payload, handle, sort_keys=False)
    with experiments_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiments_payload, handle, sort_keys=False)

    print(f"models_config={models_path}")
    print(f"experiments_config={experiments_path}")
    print(f"probe_models={','.join(probe_models)}")
    print(f"target_models={','.join(target_models)}")
    print(f"judge_models={','.join(judge_models)}")
    print("run_command=" + f"python scripts/run_experiment.py --config {experiments_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
