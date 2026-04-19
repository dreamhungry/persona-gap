"""LLM backend abstraction via litellm.

Provides a unified interface for calling any LLM (OpenAI, Claude, local models, etc.)
with JSON mode, automatic retry, and token usage tracking.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class UsageStats:
    """Cumulative token usage statistics."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    total_retries: int = 0
    total_latency_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "avg_latency_seconds": (
                round(self.total_latency_seconds / self.total_calls, 2)
                if self.total_calls > 0
                else 0
            ),
        }


class LLMBackend:
    """Unified LLM calling interface backed by litellm.

    Features:
    - JSON mode (response_format) for structured outputs
    - Automatic retry on parse/API failures (up to max_retries)
    - Cumulative token usage tracking
    - Call latency recording
    - Support for api_key / api_base / extra_params without relying on env vars
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
        json_mode: bool = True,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.api_key = api_key or None  # convert "" to None
        self.api_base = api_base
        self.timeout = timeout
        self.extra_params = extra_params or {}
        self.usage = UsageStats()

    @classmethod
    def from_llm_config(
        cls,
        llm_config: Any,
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> "LLMBackend":
        """Create an LLMBackend from an LLMConfig data model.

        Per-agent overrides take precedence over the global LLM config.
        """
        return cls(
            model=model_override or llm_config.model,
            temperature=(
                temperature_override
                if temperature_override is not None
                else llm_config.temperature
            ),
            max_tokens=(
                max_tokens_override
                if max_tokens_override is not None
                else llm_config.max_tokens
            ),
            max_retries=llm_config.max_retries,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
            timeout=llm_config.timeout,
            extra_params=llm_config.extra_params,
        )

    @classmethod
    def judge_from_llm_config(cls, llm_config: Any) -> "LLMBackend":
        """Create an LLMBackend configured for LLM-as-judge evaluation."""
        return cls(
            model=llm_config.judge_model or llm_config.model,
            temperature=llm_config.judge_temperature,
            max_tokens=256,
            max_retries=llm_config.max_retries,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
            timeout=llm_config.timeout,
            extra_params=llm_config.extra_params,
        )

    def call(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> dict[str, Any]:
        """Call the LLM and return the parsed JSON response.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Override instance default.
            max_tokens: Override instance default.
            json_mode: Override instance default.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        use_json = json_mode if json_mode is not None else self.json_mode

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            **self.extra_params,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = litellm.completion(**kwargs)
                elapsed = time.monotonic() - t0
                self.usage.total_latency_seconds += elapsed
                self.usage.total_calls += 1

                # Track token usage
                usage = response.usage
                if usage:
                    self.usage.total_prompt_tokens += usage.prompt_tokens or 0
                    self.usage.total_completion_tokens += (
                        usage.completion_tokens or 0
                    )

                content = response.choices[0].message.content or ""

                if use_json:
                    result = json.loads(content)
                    return result
                else:
                    return {"content": content}

            except json.JSONDecodeError as e:
                elapsed = time.monotonic() - t0
                self.usage.total_latency_seconds += elapsed
                self.usage.total_retries += 1
                last_error = e
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                self.usage.total_latency_seconds += elapsed
                self.usage.total_retries += 1
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def call_text(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Convenience: call LLM and return raw text (no JSON parsing)."""
        result = self.call(messages, json_mode=False, **kwargs)
        return result.get("content", "")
