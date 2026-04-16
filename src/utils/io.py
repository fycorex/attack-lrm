from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import yaml

from src.utils.schemas import ExperimentMatrixConfig, ModelsConfig, PromptConfig


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_models_config(path: str | Path) -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml_file(path))


def load_prompts_config(path: str | Path) -> PromptConfig:
    return PromptConfig.model_validate(load_yaml_file(path))


def load_experiment_config(path: str | Path) -> ExperimentMatrixConfig:
    return ExperimentMatrixConfig.model_validate(load_yaml_file(path))


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_utc() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def make_output_directory(root: str | Path, run_id: str) -> Path:
    return ensure_directory(Path(root) / run_id)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    ensure_directory(target.parent)

    if not materialized:
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def resolve_config_path(base_file: str | Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    base_resolved = Path(base_file).resolve()
    for ancestor in (base_resolved.parent, *base_resolved.parents):
        resolved = ancestor / candidate
        if resolved.exists():
            return resolved

    return base_resolved.parent / candidate


def resolve_value(
    explicit: str | None,
    env_name: str | None = None,
    default_env_names: tuple[str, ...] = (),
) -> str | None:
    if explicit:
        return explicit
    if env_name and os.getenv(env_name):
        return os.getenv(env_name)
    for candidate in default_env_names:
        if os.getenv(candidate):
            return os.getenv(candidate)
    return None


def refresh_latest_pointer(output_root: str | Path, run_dir: str | Path) -> Path:
    root = ensure_directory(output_root)
    latest = root / "latest"
    run_path = Path(run_dir).resolve()

    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    try:
        latest.symlink_to(run_path, target_is_directory=True)
    except OSError:
        shutil.copytree(run_path, latest)

    return latest
