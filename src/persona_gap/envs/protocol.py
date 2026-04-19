"""BaseEnv Protocol — the unified interface that all environment adapters must satisfy.

Any class that implements these methods can be used as an environment,
without needing to inherit from a base class (structural subtyping via Protocol).
"""

from __future__ import annotations

from typing import Any, Protocol

from persona_gap.core.models import ActionAnnotation


class BaseEnv(Protocol):
    """Unified environment protocol (Gym-like, turn-based)."""

    @property
    def num_agents(self) -> int:
        """Number of agents in this environment."""
        ...

    @property
    def agent_ids(self) -> list[str]:
        """Ordered list of agent identifiers, e.g. ['agent_0', 'agent_1']."""
        ...

    def reset(self) -> tuple[dict[str, str], str]:
        """Reset the environment for a new episode.

        Returns:
            observations: {agent_id: observation_text} — only the current
                          player's entry is meaningful; others may be empty.
            current_player: agent_id of the player who should act first.
        """
        ...

    def step(self, action: str) -> tuple[dict[str, str], dict[str, float], bool, dict[str, Any]]:
        """Apply *one* action from the current player.

        Args:
            action: The action string chosen by the current player.

        Returns:
            observations: {agent_id: observation_text} for the *next* player.
            rewards: {agent_id: reward} — typically 0 until the game ends.
            done: Whether the episode is over.
            info: Extra metadata (e.g. 'current_player', 'winner').
        """
        ...

    def get_current_player(self) -> str:
        """Return the agent_id of the player whose turn it is."""
        ...

    def get_legal_actions(self, agent_id: str) -> list[str]:
        """Return a list of legal action *strings* for the given agent."""
        ...

    def get_action_annotations(self, agent_id: str) -> dict[str, ActionAnnotation]:
        """Return personality-dimension annotations for each legal action.

        Returns:
            {action_name: ActionAnnotation} for every legal action.
        """
        ...
