from __future__ import annotations

import itertools
from pathlib import Path

from src.adapters.openai_compatible import OpenAICompatibleAdapter
from src.datasets.safe_boundary_scenarios import get_scenario
from src.evaluation.judge import JudgeEnsemble, SafetyJudge
from src.evaluation.metrics import write_metrics_bundle
from src.orchestrator.run_dialogue import run_dialogue
from src.utils.io import refresh_latest_pointer, write_csv, write_json
from src.utils.logging import configure_logger
from src.utils.schemas import ExperimentMatrixConfig, ModelsConfig, PromptConfig


def run_matrix(
    *,
    run_id: str,
    output_dir: str | Path,
    experiment: ExperimentMatrixConfig,
    models_config: ModelsConfig,
    prompts: PromptConfig,
) -> dict[str, object]:
    output_path = Path(output_dir)
    logger = configure_logger()
    all_turn_records = []
    all_dialogue_summaries = []
    total_dialogues = (
        len(experiment.probe_profiles)
        * len(experiment.target_profiles)
        * len(experiment.scenario_ids)
    )

    write_json(
        output_path / "manifest.json",
        {
            "run_id": run_id,
            "mode": "matrix",
            "paper_reference": "arXiv:2508.04039 / Nature Communications (2026) structural alignment",
            "experiment_name": experiment.experiment_name,
            "probe_profiles": experiment.probe_profiles,
            "target_profiles": experiment.target_profiles,
            "judge_profiles": experiment.judge_profiles,
            "scenario_ids": experiment.scenario_ids,
            "max_turns": experiment.max_turns,
        },
    )
    logger.info(
        "Starting matrix run %s | output_dir=%s | probes=%d | targets=%d | judges=%d | scenarios=%d | total_dialogues=%d",
        run_id,
        output_path,
        len(experiment.probe_profiles),
        len(experiment.target_profiles),
        len(experiment.judge_profiles),
        len(experiment.scenario_ids),
        total_dialogues,
    )

    judges = [
        SafetyJudge(
            adapter=OpenAICompatibleAdapter(
                label=judge_name,
                profile=models_config.profiles[judge_name],
            ),
            turn_system_prompt=prompts.judge_turn_system_prompt,
            dialogue_system_prompt=prompts.judge_dialogue_system_prompt,
        )
        for judge_name in experiment.judge_profiles
    ]
    judge_runner: SafetyJudge | JudgeEnsemble
    if len(judges) == 1:
        judge_runner = judges[0]
    else:
        judge_runner = JudgeEnsemble(judges)

    for dialogue_index, (probe_name, target_name, scenario_id) in enumerate(
        itertools.product(
            experiment.probe_profiles,
            experiment.target_profiles,
            experiment.scenario_ids,
        ),
        start=1,
    ):
        scenario = get_scenario(scenario_id)
        logger.info(
            "[%d/%d] scenario=%s | probe=%s | target=%s",
            dialogue_index,
            total_dialogues,
            scenario_id,
            probe_name,
            target_name,
        )
        probe_adapter = OpenAICompatibleAdapter(
            label=probe_name,
            profile=models_config.profiles[probe_name],
        )
        target_adapter = OpenAICompatibleAdapter(
            label=target_name,
            profile=models_config.profiles[target_name],
        )
        result = run_dialogue(
            run_id=run_id,
            output_dir=output_path,
            scenario=scenario,
            probe_adapter=probe_adapter,
            target_adapter=target_adapter,
            judge=judge_runner,
            prompts=prompts,
            max_turns=experiment.max_turns,
        )
        all_turn_records.extend(result.turn_records)
        all_dialogue_summaries.append(result.dialogue_summary)
        logger.info(
            "[%d/%d] completed | scenario=%s | probe=%s | target=%s | turns=%d | refusal_consistency=%d | policy_drift=%d",
            dialogue_index,
            total_dialogues,
            scenario_id,
            probe_name,
            target_name,
            result.dialogue_summary.turns_completed,
            result.dialogue_summary.overall_refusal_consistency_score,
            result.dialogue_summary.overall_policy_drift_score,
        )

    write_csv(
        output_path / "turn_records.csv",
        [record.model_dump(mode="json") for record in all_turn_records],
    )
    write_csv(
        output_path / "dialogue_summaries.csv",
        [record.model_dump(mode="json") for record in all_dialogue_summaries],
    )
    metrics = write_metrics_bundle(
        output_dir=str(output_path),
        turn_records=all_turn_records,
        dialogue_summaries=all_dialogue_summaries,
    )
    refresh_latest_pointer(experiment.output_root, output_path)
    logger.info(
        "Finished matrix run %s | output_dir=%s | turn_records=%d | dialogue_summaries=%d",
        run_id,
        output_path,
        len(all_turn_records),
        len(all_dialogue_summaries),
    )
    return {
        "run_id": run_id,
        "output_dir": str(output_path),
        "metrics": metrics,
    }
