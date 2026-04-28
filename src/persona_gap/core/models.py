"""Pydantic data models for the Persona-Gap framework.

All core data structures are defined here:
- PersonalityVector: 4-dimensional personality representation
- StepRecord / EpisodeResult: experiment trajectory data
- AgentConfig / EnvConfig / ExperimentConfig: configuration schemas
"""

from __future__ import annotations

from typing import Any, Literal


from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Personality
# ---------------------------------------------------------------------------

class PersonalityVector(BaseModel):
    """4-dimensional personality vector  p = (r, a, c, d) ∈ [0,1]^4."""

    risk: float = Field(ge=0.0, le=1.0, description="Risk-taking tendency")
    aggression: float = Field(ge=0.0, le=1.0, description="Aggression level")
    cooperation: float = Field(ge=0.0, le=1.0, description="Cooperation tendency")
    deception: float = Field(ge=0.0, le=1.0, description="Deception tendency")

    def to_list(self) -> list[float]:
        return [self.risk, self.aggression, self.cooperation, self.deception]

    @staticmethod
    def dim_names() -> list[str]:
        return ["risk", "aggression", "cooperation", "deception"]


# ---------------------------------------------------------------------------
# Action annotation (loaded from TOML config per environment)
# ---------------------------------------------------------------------------

class ActionAnnotation(BaseModel):
    """Personality-dimension tags for a single action."""

    is_risky: bool = False
    is_aggressive: bool = False
    is_cooperative: bool = False
    is_deceptive: bool = False


# ---------------------------------------------------------------------------
# Trajectory data
# ---------------------------------------------------------------------------

class StepRecord(BaseModel):
    """A single step of agent–environment interaction."""

    episode_id: int
    step: int
    agent_id: str
    observation: str
    legal_actions: list[str]
    action: str
    reasoning: str
    message: str | None = None
    reward: float = 0.0
    is_fallback: bool = False


class EpisodeResult(BaseModel):
    """Summary of one episode."""

    episode_id: int
    rewards: dict[str, float]  # agent_id -> total reward
    num_steps: int
    winner: str | None = None


# ---------------------------------------------------------------------------
# Configuration schemas
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """Global LLM configuration — centralizes API credentials and model defaults.

    All LLM-related settings live here instead of environment variables,
    making it easy to manage on Windows and switch between providers.

    Attributes:
        api_key: API key for the LLM provider (e.g. OpenAI / DeepSeek / Anthropic).
        api_base: Custom API base URL. Use this when targeting a compatible endpoint
                  (e.g. Azure, local vLLM, 3rd-party proxy). Set to None for provider defaults.
        model: Default model identifier in litellm format.
                Examples: "gpt-4o-mini", "deepseek/deepseek-chat",
                          "claude-3-5-sonnet-20241022", "openai/my-local-model".
        temperature: Default sampling temperature (can be overridden per-agent).
        max_tokens: Default maximum output tokens (can be overridden per-agent).
        max_retries: Maximum number of retries for failed LLM calls.
        timeout: Request timeout in seconds. None = provider default.
        extra_params: Any additional key-value pairs to forward to litellm.completion().
    """

    api_key: str = Field(
        default="",
        description="API key for the LLM provider",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL (e.g. 'https://api.deepseek.com')",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Default model identifier in litellm format",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    max_retries: int = Field(default=3, ge=1)
    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional params forwarded to litellm.completion()",
    )

    # --- Judge-specific overrides (for LLM-as-judge in metrics) ---
    judge_model: str | None = Field(
        default=None,
        description="Model for LLM-as-judge. Falls back to `model` if None.",
    )
    judge_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for judge calls (low for determinism)",
    )


class AgentConfig(BaseModel):
    """Configuration for a single LLM agent.

    Per-agent model/temperature/max_tokens override the global LLM config.
    Set to None to inherit from the global [llm] section.
    """

    agent_id: str
    personality: PersonalityVector
    model: str | None = Field(
        default=None,
        description="Override model for this agent. None = use global llm.model",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override temperature. None = use global llm.temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Override max_tokens. None = use global llm.max_tokens",
    )
    memory_enabled: bool = True
    communication_enabled: bool = True


class EnvConfig(BaseModel):
    """Environment configuration — generic via adapter + game_name."""

    adapter: str = "rlcard"
    game_name: str = "leduc-holdem"
    action_annotations_path: str = "configs/envs/leduc_holdem.toml"
    behavioral_method: str = Field(
        default="annotation",
        description="Method for behavioral personality extraction: 'annotation', 'scorer', or 'llm'",
    )
    expressed_context: Literal["text-only", "grounded"] = Field(
        default="text-only",
        description="Context mode for expressed personality extraction: 'text-only' or 'grounded'",
    )



class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    seed: int = 42
    num_episodes: int = 50
    output_dir: str = "outputs/"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    agents: list[AgentConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Batch experiment configuration
# ---------------------------------------------------------------------------

class BatchExperimentConfig(BaseModel):
    """Configuration for running a batch of experiments."""

    base: ExperimentConfig
    sweep_personalities: list[str] = Field(
        default_factory=list,
        description="List of personality preset file names to sweep over",
    )
    sweep_memory: list[bool] = Field(default_factory=lambda: [True, False])
    sweep_communication: list[bool] = Field(default_factory=lambda: [True, False])
