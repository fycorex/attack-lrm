from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.utils.io import resolve_value
from src.utils.schemas import ModelProfile


@dataclass(frozen=True)
class ResolvedModelProfile:
    label: str
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_retries: int
    temperature: float | None
    trust_env: bool
    headers: dict[str, str]
    response_format_json: bool


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    raw_response: dict[str, Any]
    finish_reason: str | None
    latency_ms: int


def resolve_model_profile(label: str, profile: ModelProfile) -> ResolvedModelProfile:
    base_url = resolve_value(
        explicit=profile.base_url,
        env_name=profile.base_url_env,
        default_env_names=("OPENAI_COMPAT_BASE_URL",),
    )
    api_key = resolve_value(
        explicit=profile.api_key,
        env_name=profile.api_key_env,
        default_env_names=("OPENAI_COMPAT_API_KEY",),
    )
    model = resolve_value(
        explicit=profile.model,
        env_name=profile.model_env,
    )

    missing: list[str] = []
    if not base_url:
        missing.append("base_url")
    if not api_key:
        missing.append("api_key")
    if not model:
        missing.append("model")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Model profile '{label}' is missing required values: {joined}")

    return ResolvedModelProfile(
        label=label,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout=profile.timeout,
        max_retries=profile.max_retries,
        temperature=profile.temperature,
        trust_env=profile.trust_env,
        headers=profile.headers,
        response_format_json=profile.response_format_json,
    )


class OpenAICompatibleAdapter:
    def __init__(self, label: str, profile: ModelProfile) -> None:
        self.settings = resolve_model_profile(label=label, profile=profile)

    @property
    def model_name(self) -> str:
        return self.settings.model

    @property
    def label(self) -> str:
        return self.settings.label

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format_json: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        url = f"{self.settings.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }
        chosen_temperature = self.settings.temperature if temperature is None else temperature
        if chosen_temperature is not None:
            payload["temperature"] = chosen_temperature

        use_json_response = (
            self.settings.response_format_json
            if response_format_json is None
            else response_format_json
        )
        if use_json_response:
            payload["response_format"] = {"type": "json_object"}
        if extra_body:
            payload.update(extra_body)

        # Handle both raw tokens and Bearer-prefixed tokens
        api_key = self.settings.api_key
        if not api_key.startswith("Bearer "):
            api_key = f"Bearer {api_key}"
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
        }
        headers.update(self.settings.headers)

        with httpx.Client(
            timeout=self.settings.timeout,
            trust_env=self.settings.trust_env,
        ) as client:
            for attempt in range(self.settings.max_retries + 1):
                started = time.perf_counter()
                try:
                    response = client.post(url, headers=headers, json=payload)
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.HTTPStatusError(
                            f"Retryable status code {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    raw_payload = response.json()
                    choice = raw_payload.get("choices", [{}])[0]
                    finish_reason = choice.get("finish_reason")
                    message = choice.get("message", {})
                    content = self._normalize_content(message.get("content"))
                    return ChatCompletionResult(
                        text=content,
                        raw_response=raw_payload,
                        finish_reason=finish_reason,
                        latency_ms=latency_ms,
                    )
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                    if attempt >= self.settings.max_retries:
                        raise RuntimeError(
                            f"{self.settings.label} request failed after {attempt + 1} attempts: {exc}"
                        ) from exc
                    # Check for Retry-After header on 429 errors
                    sleep_seconds = min(2**attempt, 60) + random.random() * 0.5
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_seconds = max(int(retry_after), 2)
                    time.sleep(sleep_seconds)

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text", "")
                    if isinstance(text_value, str):
                        fragments.append(text_value)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    fragments.append(item["text"])
            return "\n".join(fragment.strip() for fragment in fragments if fragment.strip()).strip()
        return ""
