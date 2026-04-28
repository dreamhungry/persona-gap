"""Tests for Pydantic data models and TOML configuration loading."""

from pathlib import Path

import pytest

from persona_gap.core.config import (
    load_action_annotations,
    load_config,
    load_personality_preset,
)
from persona_gap.core.models import (
    ActionAnnotation,
    AgentConfig,
    EnvConfig,
    ExperimentConfig,
    LLMConfig,
    PersonalityVector,
    StepRecord,
)

# ---------------------------------------------------------------------------
# PersonalityVector tests
# ---------------------------------------------------------------------------


class TestPersonalityVector:
    def test_valid_vector(self):
        pv = PersonalityVector(risk=0.5, aggression=0.3, cooperation=0.8, deception=0.1)
        assert pv.risk == 0.5
        assert pv.to_list() == [0.5, 0.3, 0.8, 0.1]

    def test_boundary_values(self):
        pv = PersonalityVector(risk=0.0, aggression=1.0, cooperation=0.0, deception=1.0)
        assert pv.risk == 0.0
        assert pv.aggression == 1.0

    def test_out_of_range_raises(self):
        with pytest.raises(Exception):
            PersonalityVector(risk=1.5, aggression=0.5, cooperation=0.5, deception=0.5)
        with pytest.raises(Exception):
            PersonalityVector(risk=-0.1, aggression=0.5, cooperation=0.5, deception=0.5)

    def test_dim_names(self):
        assert PersonalityVector.dim_names() == [
            "risk", "aggression", "cooperation", "deception"
        ]


# ---------------------------------------------------------------------------
# StepRecord tests
# ---------------------------------------------------------------------------


class TestStepRecord:
    def test_basic_record(self):
        r = StepRecord(
            episode_id=0,
            step=1,
            agent_id="agent_0",
            observation="test obs",
            legal_actions=["call", "raise"],
            action="call",
            reasoning="seemed safe",
        )
        assert r.episode_id == 0
        assert r.is_fallback is False
        assert r.message is None

    def test_json_roundtrip(self):
        r = StepRecord(
            episode_id=1,
            step=2,
            agent_id="agent_1",
            observation="obs",
            legal_actions=["fold"],
            action="fold",
            reasoning="giving up",
            reward=-1.0,
            is_fallback=True,
        )
        json_str = r.model_dump_json()
        r2 = StepRecord.model_validate_json(json_str)
        assert r == r2


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_env_config_defaults_to_text_only_expressed_context(self):
        env = EnvConfig()
        assert env.expressed_context == "text-only"

    def test_load_experiment_config(self, tmp_path: Path):
        toml_content = """
[experiment]
seed = 123
num_episodes = 10
output_dir = "outputs/test"

[llm]
api_key = "sk-test-key"
model = "gpt-4o-mini"

[env]
adapter = "rlcard"
game_name = "leduc-holdem"
action_annotations_path = "configs/envs/leduc_holdem.toml"
expressed_context = "grounded"

[[agents]]
agent_id = "agent_0"


[agents.personality]
risk = 0.8
aggression = 0.9
cooperation = 0.2
deception = 0.5
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content, encoding="utf-8")
        config = load_config(config_file)

        assert config.seed == 123
        assert config.num_episodes == 10
        assert config.llm.api_key == "sk-test-key"
        assert config.llm.model == "gpt-4o-mini"
        assert config.env.adapter == "rlcard"
        assert config.env.game_name == "leduc-holdem"
        assert config.env.expressed_context == "grounded"
        assert len(config.agents) == 1

        assert config.agents[0].personality.risk == 0.8
        assert config.agents[0].model is None  # inherits from llm section

    def test_load_personality_preset(self, tmp_path: Path):
        toml_content = """
[personality]
risk = 0.9
aggression = 0.1
cooperation = 0.7
deception = 0.3
"""
        preset_file = tmp_path / "preset.toml"
        preset_file.write_text(toml_content, encoding="utf-8")
        pv = load_personality_preset(preset_file)
        assert pv.risk == 0.9
        assert pv.cooperation == 0.7

    def test_load_action_annotations(self, tmp_path: Path):
        toml_content = """
[actions.raise]
is_risky = true
is_aggressive = true

[actions.call]
is_cooperative = true

[actions.fold]
"""
        ann_file = tmp_path / "ann.toml"
        ann_file.write_text(toml_content, encoding="utf-8")
        anns = load_action_annotations(ann_file)

        assert "raise" in anns
        assert anns["raise"].is_risky is True
        assert anns["raise"].is_aggressive is True
        assert anns["call"].is_cooperative is True
        assert anns["fold"].is_risky is False  # default
