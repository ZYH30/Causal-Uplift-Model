"""Minimal OpenAI-compatible HTTP client for auditable TLCD LLM calls."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ApiCallError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    endpoint: str
    api_key: str
    max_tokens_field: str
    supports_prompt_cache_key: bool = False


def provider_config(name: str, api_key: str) -> ProviderConfig:
    if name == "kimi":
        return ProviderConfig(
            name="kimi",
            model="kimi-k2.6",
            endpoint="https://api.moonshot.cn/v1/chat/completions",
            api_key=api_key,
            max_tokens_field="max_completion_tokens",
            supports_prompt_cache_key=True,
        )
    if name == "deepseek":
        return ProviderConfig(
            name="deepseek",
            model="deepseek-v4-flash",
            endpoint="https://api.deepseek.com/chat/completions",
            api_key=api_key,
            max_tokens_field="max_tokens",
        )
    raise ValueError(f"Unknown provider {name!r}")


class CompatibleChatClient:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        timeout_seconds: float = 300.0,
        max_transport_retries: int = 3,
    ) -> None:
        self.config = config
        self.max_transport_retries = max_transport_retries
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=30.0),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "CompatibleChatClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        thinking_enabled: bool,
        prompt_cache_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
            self.config.max_tokens_field: max_completion_tokens,
        }
        if self.config.supports_prompt_cache_key and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        return payload

    def create(
        self,
        messages: list[dict[str, str]],
        *,
        max_completion_tokens: int,
        thinking_enabled: bool = True,
        prompt_cache_key: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        payload = self.build_payload(
            messages,
            max_completion_tokens=max_completion_tokens,
            thinking_enabled=thinking_enabled,
            prompt_cache_key=prompt_cache_key,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_transport_retries + 1):
            try:
                response = self.client.post(self.config.endpoint, json=payload)
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                # The provider may still have completed and billed a long-running
                # generation. Do not resend automatically after response reading
                # has begun; callers can inspect state before an explicit resume.
                raise ApiCallError(
                    f"{self.config.name} response stream failed without a safe automatic retry: {exc}"
                ) from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                if attempt >= self.max_transport_retries:
                    break
                time.sleep(min(2 ** attempt, 10))
                continue

            if response.status_code < 400:
                try:
                    return response.json(), attempt
                except json.JSONDecodeError as exc:
                    raise ApiCallError(
                        f"{self.config.name} returned non-JSON HTTP content",
                        status_code=response.status_code,
                        response_body=response.text[:2000],
                    ) from exc

            body = response.text[:4000]
            error = ApiCallError(
                f"{self.config.name} API returned HTTP {response.status_code}",
                status_code=response.status_code,
                response_body=body,
            )
            if response.status_code not in RETRYABLE_STATUS or attempt >= self.max_transport_retries:
                raise error
            retry_after = response.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else min(2 ** attempt, 10)
            except ValueError:
                delay = min(2 ** attempt, 10)
            time.sleep(delay)
            last_error = error

        raise ApiCallError(f"{self.config.name} transport failed after retries: {last_error}") from last_error


def response_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Response is missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Response content is empty")
    return content


def response_reasoning_content(response: dict[str, Any]) -> str | None:
    try:
        value = response["choices"][0]["message"].get("reasoning_content")
    except (KeyError, IndexError, TypeError):
        return None
    return value if isinstance(value, str) and value else None


def response_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage", {}) or {}
    keys = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]
    out = {}
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            out[key] = int(value)
    if "total_tokens" not in out:
        out["total_tokens"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
    return out
