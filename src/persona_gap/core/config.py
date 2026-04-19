"""TOML configuration loader with Pydantic validation.

Sensitive credentials (api_key, api_base, …) are read from a ``.env`` file
at the project root so they never leak into version control.  Values set in
the TOML ``[llm]`` section take precedence over ``.env`` (useful for
quick per-experiment overrides).

Priority (highest → lowest):
  1. Explicit non-empty value in ``[llm]`` section of the TOML file
  2. ``LLM_*`` environment variables (loaded from ``.env`` via python-dotenv)
  3. Model defaults defined in ``LLMConfig``

Usage:
    from persona_gap.core.config import load_config
    config = load_config("configs/experiment.toml")
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import (
    ActionAnnotation,
    AgentConfig,
    BatchExperimentConfig,
    ExperimentConfig,
    LLMConfig,
    PersonalityVector,
)

# ---------------------------------------------------------------------------
# .env loading — executed once at import time
# ---------------------------------------------------------------------------

# Walk up from this file to find the project root .env
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/persona_gap/core -> root
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def load_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file and return a dict."""
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# LLM config merging: .env + TOML
# ---------------------------------------------------------------------------

# Maps LLMConfig field name -> environment variable name
_ENV_VAR_MAP: dict[str, str] = {
    "api_key": "LLM_API_KEY",
    "api_base": "LLM_API_BASE",
    "model": "LLM_MODEL",
    "temperature": "LLM_TEMPERATURE",
    "max_tokens": "LLM_MAX_TOKENS",
    "max_retries": "LLM_MAX_RETRIES",
    "timeout": "LLM_TIMEOUT",
    "judge_model": "LLM_JUDGE_MODEL",
    "judge_temperature": "LLM_JUDGE_TEMPERATURE",
}

# Fields that should be cast to specific types from env var strings
_FLOAT_FIELDS = {"temperature", "timeout", "judge_temperature"}
_INT_FIELDS = {"max_tokens", "max_retries"}


def _build_llm_config(toml_llm: dict[str, Any] | None) -> dict[str, Any]:
    """Merge .env environment variables with TOML ``[llm]`` section.

    Strategy:
    - Start with env vars as the base layer.
    - TOML values override env vars when present and non-empty.
    - This means you can put safe defaults (model, temperature…) in TOML
      and keep secrets (api_key) only in .env.
    """
    merged: dict[str, Any] = {}

    # Layer 1: environment variables (from .env)
    for field, env_var in _ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is not None and value.strip():
            value = value.strip()
            if field in _FLOAT_FIELDS:
                merged[field] = float(value)
            elif field in _INT_FIELDS:
                merged[field] = int(value)
            else:
                merged[field] = value

    # Layer 2: TOML [llm] section overrides (only non-empty values)
    if toml_llm:
        for key, value in toml_llm.items():
            # Skip empty strings / None — they mean "not set, use .env"
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            merged[key] = value

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file.

    The TOML file should have the structure:
        [experiment]
        seed = 42
        ...

        [llm]          # optional — secrets come from .env
        model = "gpt-4o-mini"
        ...

        [env]
        adapter = "rlcard"
        game_name = "leduc-holdem"
        ...

        [[agents]]
        agent_id = "agent_0"
        ...
    """
    raw = load_toml(path)

    # Support both flat layout and nested under [experiment] key
    if "experiment" in raw:
        experiment_data = raw["experiment"]
    else:
        experiment_data = {}

    # Merge top-level keys
    config_dict: dict[str, Any] = {**experiment_data}

    # Build LLM config: .env (base) + TOML [llm] (override)
    config_dict["llm"] = _build_llm_config(raw.get("llm"))

    if "env" in raw:
        config_dict["env"] = raw["env"]
    if "agents" in raw:
        config_dict["agents"] = raw["agents"]

    return ExperimentConfig.model_validate(config_dict)


def load_personality_preset(path: str | Path) -> PersonalityVector:
    """Load a personality preset from a TOML file.

    Expected format:
        [personality]
        risk = 0.8
        aggression = 0.9
        cooperation = 0.2
        deception = 0.5
    """
    raw = load_toml(path)
    data = raw.get("personality", raw)
    return PersonalityVector.model_validate(data)


def load_action_annotations(path: str | Path) -> dict[str, ActionAnnotation]:
    """Load action annotations from a TOML file.

    Expected format:
        [actions.raise]
        is_risky = true
        is_aggressive = true
        is_cooperative = false
        is_deceptive = false

        [actions.call]
        ...
    """
    raw = load_toml(path)
    actions_raw = raw.get("actions", {})
    return {
        name: ActionAnnotation.model_validate(ann)
        for name, ann in actions_raw.items()
    }


def load_batch_config(path: str | Path) -> BatchExperimentConfig:
    """Load a batch experiment configuration from a TOML file."""
    raw = load_toml(path)
    base_data: dict[str, Any] = {}

    if "experiment" in raw:
        base_data.update(raw["experiment"])

    # Build LLM config: .env (base) + TOML [llm] (override)
    base_data["llm"] = _build_llm_config(raw.get("llm"))

    if "env" in raw:
        base_data["env"] = raw["env"]
    if "agents" in raw:
        base_data["agents"] = raw["agents"]

    sweep = raw.get("sweep", {})

    return BatchExperimentConfig(
        base=ExperimentConfig.model_validate(base_data),
        sweep_personalities=sweep.get("personalities", []),
        sweep_memory=sweep.get("memory", [True, False]),
        sweep_communication=sweep.get("communication", [True, False]),
    )
