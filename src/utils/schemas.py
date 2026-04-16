from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AllowedResponsePattern = Literal[
    "refuse_and_redirect",
    "refuse_and_explain_boundary",
    "stay_high_level_with_safe_alt",
    "explain_refusal_without_details",
    "answer_safely_at_high_level",
]


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    base_url_env: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    model_env: str | None = None
    timeout: float = 45.0
    max_retries: int = 3
    temperature: float | None = 0.0
    trust_env: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    response_format_json: bool = False


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, ModelProfile]


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_system_prompt: str
    target_system_prompt: str
    judge_turn_system_prompt: str
    judge_dialogue_system_prompt: str


class ExperimentMatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "defensive-boundary-eval"
    models_config: str = "config/models.example.yaml"
    prompts_config: str = "config/prompts.example.yaml"
    output_root: str = "outputs"
    max_turns: int = Field(default=10, ge=1, le=25)
    probe_profiles: list[str]
    target_profiles: list[str]
    judge_profiles: list[str]
    scenario_ids: list[str]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    title: str
    category: str
    objective: str
    allowed_response_pattern: AllowedResponsePattern
    failure_modes: list[str]


class RawModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    run_id: str
    scenario_id: str
    turn_id: int | None = None
    role: Literal["probe", "target", "judge"]
    model_label: str
    request_messages: list[dict[str, str]]
    response_text: str | None = None
    response_json: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: int | None = None
    error: str | None = None


class TurnJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refusal_flag: bool
    disclaimer_flag: bool
    policy_drift_score: int = Field(ge=0, le=5)
    overcompliance_score: int = Field(ge=0, le=5)
    helpful_safe_redirection_score: int = Field(ge=0, le=5)
    consistency_note: str
    judge_rationale_short: str
    judge_parse_error: bool = False


class TurnRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    probe_model: str
    target_model: str
    judge_model: str
    turn_id: int
    probe_message: str
    target_reply: str
    refusal_flag: bool
    disclaimer_flag: bool
    risk_score: int = Field(default=0, ge=0, le=5)
    policy_drift_score: int = Field(ge=0, le=5)
    overcompliance_score: int = Field(ge=0, le=5)
    helpful_safe_redirection_score: int = Field(ge=0, le=5)
    consistency_note: str
    judge_rationale_short: str
    judge_parse_error: bool = False


class DialogueSummaryJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_refusal_consistency_score: int = Field(ge=0, le=5)
    overall_policy_drift_score: int = Field(ge=0, le=5)
    trajectory_summary: str
    dialogue_rationale_short: str
    judge_parse_error: bool = False


class DialogueSummaryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    probe_model: str
    target_model: str
    judge_model: str
    turns_completed: int
    overall_refusal_consistency_score: int = Field(ge=0, le=5)
    overall_policy_drift_score: int = Field(ge=0, le=5)
    trajectory_summary: str
    dialogue_rationale_short: str
    judge_parse_error: bool = False


class DialogueRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: str
    output_dir: str
    turn_records: list[TurnRecord]
    dialogue_summary: DialogueSummaryRecord
