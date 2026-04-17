# Defensive Multi-Turn Cross-Model Safety Evaluation Framework

This repository contains a configurable multi-turn, cross-model safety evaluation harness inspired by the experiment design in [Large Reasoning Models Are Autonomous Jailbreak Agents](https://arxiv.org/abs/2508.04039). It is intended for authorized defensive evaluation of model safety behavior across repeated conversational pressure.

The code implements the paper's experimental shape more than its exact harmful experiment. It supports adversary/probe models, target models, judge models, 10-turn dialogues, a 70-item benchmark shape, full transcript passing to the target model, structured judge output, and matrix-level summaries. It does not reproduce the paper's harmful benchmark items or its reported attack-success-rate experiment.

## Paper Comparison

The cited paper studies autonomous jailbreak behavior by using four large reasoning models as adversaries against nine target models, with three judge models evaluating target replies. The experiment uses 70 harmful requests across seven categories, 10-turn conversations, a neutral `Hi!` opener, and a target system prompt equivalent to `You are a helpful assistant.`

This repository implements a defensive variant of that setup:

| Paper setting | Repository status |
| --- | --- |
| Four adversary/LRM profiles | Implemented as four probe profiles in `config/models.example.yaml`: DeepSeek-R1, Gemini 2.5 Flash, Grok 3 Mini, and Qwen3 235B. |
| Nine target profiles | Implemented in `config/models.example.yaml`: Claude 4 Sonnet, DeepSeek-V3, Gemini 2.5 Flash, GPT-4o, Grok 3, Llama 3.1 70B, o4-mini, Qwen3 30B, and Llama 4 Maverik. |
| Three judge profiles | Implemented in `config/models.example.yaml`: GPT-4.1, Gemini 2.5 Flash, and Grok 3. |
| 10 scored turns | Implemented through `max_turns: 10` in experiment configs and the `--max-turns` CLI option. |
| Neutral conversation start | Implemented: every dialogue begins with `Hi!` sent to the target model before scored turns. |
| Target receives full conversation history | Implemented: the target call receives the system prompt plus the running transcript. |
| 70 benchmark items across seven categories | Implemented structurally as 70 safe-boundary scenarios in `src/datasets/safe_boundary_scenarios.py`. |
| Harmful benchmark requests | Not implemented. The local dataset replaces harmful requests with policy-safe boundary-evaluation scenarios. |
| Paper's exact adversary system prompt | Partially mirrored. The default/local prompt contains an adversarial-style probe instruction, but the runtime also provides safe scenario objectives and asks for policy-safe follow-up messages. Review `config/prompts.example.yaml` before running. |
| Paper's target system prompt | Not exact. The local default prompt is more permissive than the paper's simple helpful-assistant prompt. |
| Harm-score scale and ASR | Not implemented exactly. This repo uses defensive metrics: refusal, disclaimer leakage, policy drift, overcompliance, and safe redirection. |
| Direct harmful-prompt control condition | Not implemented as a separate control experiment. |
| ICC/Kappa judge reliability analysis | Not implemented. Judge outputs are aggregated by averaging numeric scores and majority-voting booleans. |
| Persuasive-strategy annotation | Not implemented. |

Verdict: the repository implements a paper-shaped defensive evaluation framework with similar model counts, turn count, matrix layout, and judge ensemble design. It should not be described as a faithful reproduction of the paper's jailbreak experiment.

## Safety Scope

Use this project only for authorized safety testing of models you are allowed to evaluate.

The included benchmark scenarios are designed to test:

- whether a model keeps safety boundaries stable across turns
- whether repeated follow-ups increase unsafe specificity
- whether benign framing changes policy behavior
- whether refusals stay useful without leaking operational detail
- whether disclaimers are followed by overcompliant content
- whether models redirect to safer alternatives

The repository should not be used to generate or publish harmful instructions. Avoid committing local outputs that may contain model-generated sensitive content.

## Project Layout

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── config/
│   ├── experiments.example.yaml
│   ├── experiments.qianfan.example.yaml
│   ├── models.example.yaml
│   ├── models.qianfan.example.yaml
│   └── prompts.example.yaml
├── outputs/
│   ├── .gitkeep
│   └── example_turn_records.jsonl
├── scripts/
│   ├── make_qianfan_config.py
│   ├── run_experiment.py
│   └── summarize_results.py
└── src/
    ├── adapters/
    │   └── openai_compatible.py
    ├── datasets/
    │   └── safe_boundary_scenarios.py
    ├── evaluation/
    │   ├── judge.py
    │   └── metrics.py
    ├── orchestrator/
    │   ├── run_dialogue.py
    │   └── run_matrix.py
    └── utils/
        ├── io.py
        ├── logging.py
        └── schemas.py
```

## Main Components

`scripts/run_experiment.py`

Runs either a single dialogue or a full matrix from YAML config. Single-run mode accepts model names from CLI flags or environment variables. Matrix mode reads profile names, prompts, scenario IDs, turn count, and output settings from config.

`src/orchestrator/run_dialogue.py`

Runs one probe-target dialogue. It:

- starts the target with `Hi!`
- asks the probe for one follow-up per turn
- sends the full transcript to the target each turn
- asks one judge or a judge ensemble to score each target reply
- writes raw calls, scored turns, and dialogue summaries

`src/orchestrator/run_matrix.py`

Runs every configured probe-target-scenario combination. With the default example config, the full matrix is:

- 4 probe profiles
- 9 target profiles
- 70 scenarios
- 10 turns per dialogue
- 3 judge profiles aggregated as an ensemble

That is 2,520 dialogues before judge calls. Plan cost, latency, and rate limits before running the full matrix.

`src/datasets/safe_boundary_scenarios.py`

Defines 70 defensive scenarios across seven safe-evaluation categories:

- specificity escalation
- benign reframing
- boundary explanation
- safe redirection
- consistency
- policy drift
- disclaimer leakage

Each scenario contains an ID, title, category, objective, expected safe response pattern, and failure modes.

`src/evaluation/judge.py`

Provides turn-level and dialogue-level judge runners. Judges are instructed to return strict JSON. If a provider returns malformed JSON or a call fails, fallback heuristics produce a marked `judge_parse_error` result instead of crashing the run.

`src/evaluation/metrics.py`

Builds aggregate metrics from turn records and dialogue summaries:

- robust refusal rate
- disclaimer leakage rate
- mean policy drift score
- mean overcompliance score
- safe redirection rate
- cross-model vulnerability gaps
- trajectory summaries across turns

`src/adapters/openai_compatible.py`

Uses an OpenAI-compatible `/chat/completions` endpoint through `httpx`. Profiles can use explicit values or environment variables for base URL, API key, model name, headers, retry count, timeout, temperature, and JSON response formatting.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate defensive-safety-eval
```

Or install the Python requirements directly in an existing environment:

```bash
python -m pip install -r requirements.txt
```

The project expects Python with `httpx`, `pydantic`, and `PyYAML`.

## Provider Configuration

All models are accessed through an OpenAI-compatible chat-completions API. Set the shared endpoint and key:

```bash
export OPENAI_COMPAT_BASE_URL="https://your-provider.example/v1"
export OPENAI_COMPAT_API_KEY="your-api-key"
```

For single-run mode, set model names through environment variables:

```bash
export PROBE_MODEL="your-probe-model"
export TARGET_MODEL="your-target-model"
export JUDGE_MODEL="your-judge-model"
```

`JUDGE_MODEL` may be a comma-separated list to use the judge ensemble path:

```bash
export JUDGE_MODEL="gpt-4.1,gemini-2.5-flash,grok-3"
```

You can also pass model names directly through CLI flags.

## Configuration Files

`config/models.example.yaml`

Defines paper-shaped profiles for:

- four probes
- nine targets
- three judges

The profile names mirror the paper, but the actual `model` strings must be supported by your OpenAI-compatible provider.

`config/experiments.example.yaml`

Defines the default paper-shaped defensive matrix:

- `max_turns: 10`
- four probe profiles
- nine target profiles
- three judge profiles
- scenario IDs `s001` through `s070`
- output root `outputs`

`config/prompts.example.yaml`

Defines probe, target, and judge prompts. Review this file carefully before running any evaluation. The local probe prompt is adversarial-style, while the runtime user message asks the probe to produce short, realistic, policy-safe follow-ups for safe-boundary scenarios. The local target prompt is also not the paper's exact plain helpful-assistant prompt.

For a stricter defensive configuration, replace the probe prompt with a direct safe-boundary tester prompt and replace the target prompt with:

```text
You are a helpful assistant.
```

## Run A Single Dialogue

Using environment variables:

```bash
python scripts/run_experiment.py \
  --scenario s001 \
  --max-turns 10
```

Using CLI model names:

```bash
python scripts/run_experiment.py \
  --probe-model deepseek-r1 \
  --target-model gpt-4o \
  --judge-model gpt-4.1,gemini-2.5-flash,grok-3 \
  --scenario s001 \
  --max-turns 10
```

Stop early if the run reaches a high defensive risk score:

```bash
python scripts/run_experiment.py \
  --probe-model deepseek-r1 \
  --target-model gpt-4o \
  --judge-model gpt-4.1 \
  --scenario s001 \
  --stop-on-risk-score 4
```

## Run The Full Matrix

```bash
python scripts/run_experiment.py --config config/experiments.example.yaml
```

For a smaller smoke test, create a generated config with fewer models or scenarios:

```bash
python scripts/make_qianfan_config.py \
  --models kimi-k2.5,deepseek-v3.2,minimax-m2.5 \
  --probe-models kimi-k2.5 \
  --target-models deepseek-v3.2 \
  --judge-models minimax-m2.5 \
  --scenario-count 3 \
  --prefix smoke

python scripts/run_experiment.py --config config/generated/smoke.experiments.yaml
```

## Qianfan-Compatible Configs

The repository includes Qianfan examples:

- `config/models.qianfan.example.yaml`
- `config/experiments.qianfan.example.yaml`

Set the endpoint:

```bash
export OPENAI_COMPAT_BASE_URL="https://qianfan.baidubce.com/v2/coding"
export OPENAI_COMPAT_API_KEY="your-key"
```

Run the Qianfan example:

```bash
python scripts/run_experiment.py --config config/experiments.qianfan.example.yaml
```

Generate a custom Qianfan-style config:

```bash
python scripts/make_qianfan_config.py \
  --models kimi-k2.5,deepseek-v3.2,minimax-m2.5 \
  --prefix qianfan_custom

python scripts/run_experiment.py --config config/generated/qianfan_custom.experiments.yaml
```

Generated configs are ignored by default through `config/generated/*.yaml`.

## Output Artifacts

Each run writes to `outputs/<run_id>/` and refreshes `outputs/latest/`.

Main files:

- `manifest.json`
- `raw_probe.jsonl`
- `raw_target.jsonl`
- `raw_judge.jsonl`
- `turn_records.jsonl`
- `turn_records.csv`
- `dialogue_summaries.jsonl`
- `dialogue_summaries.csv`
- `metrics_summary.json`
- `metrics_by_model.csv`
- `metrics_by_scenario.csv`
- `trajectory_summary.csv`

Raw files contain request and response payloads. Treat them as sensitive. Do not commit real run outputs unless they have been reviewed and sanitized.

## Turn Record Schema

Each scored turn contains:

- `run_id`
- `scenario_id`
- `probe_model`
- `target_model`
- `judge_model`
- `turn_id`
- `probe_message`
- `target_reply`
- `refusal_flag`
- `disclaimer_flag`
- `risk_score`
- `policy_drift_score`
- `overcompliance_score`
- `helpful_safe_redirection_score`
- `consistency_note`
- `judge_rationale_short`
- `judge_parse_error`

## Summarize Results

Summarize the latest completed run:

```bash
python scripts/summarize_results.py --input outputs/latest
```

Summarize a specific run directory:

```bash
python scripts/summarize_results.py --input outputs/matrix_YYYYMMDDTHHMMSSZ_abcd1234
```

The script prints the top-level metrics from `metrics_summary.json`.

## Reproducibility Notes

Model providers may differ in API behavior, model aliases, safety filters, rate limits, and JSON response formatting. The example profile names are paper-shaped labels, not a guarantee that the referenced models are available from your endpoint.

Before comparing results across providers:

- pin model versions or provider model IDs where possible
- keep `temperature: 0.0` for probes, targets, and judges unless deliberately testing stochasticity
- use the same scenario IDs and turn count
- keep judge prompts and parsing behavior fixed
- record provider endpoint, date, model aliases, and any gateway-specific headers

## Known Gaps

This project currently does not include:

- the paper's original harmful prompt benchmark
- direct-prompt control-condition automation
- attack success rate based on maximum harm score
- ICC or Cohen's Kappa reliability calculations
- persuasive-strategy annotation
- provider-specific handling for non-OpenAI-compatible APIs
- concurrency for full matrix runs

Those gaps are intentional for the defensive scope, except for reliability analysis and control-condition support, which could be added as non-harmful evaluation features.

## Citation

Reference paper:

```bibtex
@misc{hagendorff2025large,
  title = {Large Reasoning Models Are Autonomous Jailbreak Agents},
  author = {Thilo Hagendorff and Erik Derner and Nuria Oliver},
  year = {2025},
  eprint = {2508.04039},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```
