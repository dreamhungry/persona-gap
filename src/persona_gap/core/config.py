"""TOML configuration loader with Pydantic validation.

Usage:
    from persona_gap.core.config import load_config
    config = load_config("configs/experiment.toml")
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .models import (
    ActionAnnotation,
    AgentConfig,
    BatchExperimentConfig,
    ExperimentConfig,
    LLMConfig,
    PersonalityVector,
)


def load_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file and return a dict."""
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file.

    The TOML file should have the structure:
        [experiment]
        seed = 42
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
    if "llm" in raw:
        config_dict["llm"] = raw["llm"]
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
    if "llm" in raw:
        base_data["llm"] = raw["llm"]
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
