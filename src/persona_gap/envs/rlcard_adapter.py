"""Generic RLCard adapter — wraps any rlcard game into the BaseEnv protocol.

Supports all rlcard games (leduc-holdem, uno, blackjack, limit-holdem, etc.)
by accepting ``game_name`` as a parameter.  Action personality annotations
are loaded from a TOML config file, *not* hard-coded.

Usage:
    from persona_gap.envs.registry import create_env

    env = create_env(
        "rlcard",
        game_name="leduc-holdem",
        seed=42,
        action_annotations_path="configs/envs/leduc_holdem.toml",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from persona_gap.core.config import load_action_annotations
from persona_gap.core.models import ActionAnnotation
from persona_gap.envs.registry import register_env

logger = logging.getLogger(__name__)


@register_env("rlcard")
class RLCardAdapter:
    """Adapter that wraps any rlcard environment into the BaseEnv protocol.

    Key design choices
    ------------------
    * ``game_name`` is forwarded to ``rlcard.make()``, so *any* rlcard game
      works without changing code.
    * Action annotations come from a TOML file pointed to by
      ``action_annotations_path``.
    * Observations are formatted as human-readable text from the *raw_obs*
      dict, suitable for LLM prompts.
    """

    def __init__(
        self,
        game_name: str = "leduc-holdem",
        seed: int = 42,
        action_annotations_path: str | None = None,
        num_players: int | None = None,
    ) -> None:
        import rlcard

        config: dict[str, Any] = {"seed": seed}
        if num_players is not None:
            config["game_num_players"] = num_players

        self._env = rlcard.make(game_name, config=config)
        self._game_name = game_name
        self._agent_ids = [f"agent_{i}" for i in range(self._env.num_players)]

        # Current turn state (updated by reset / step)
        self._current_state: dict[str, Any] | None = None
        self._current_player_idx: int = 0

        # Load action annotations from TOML (if provided)
        self._annotations: dict[str, ActionAnnotation] = {}
        if action_annotations_path is not None:
            path = Path(action_annotations_path)
            if path.exists():
                self._annotations = load_action_annotations(path)
                logger.info(
                    "Loaded %d action annotations from %s",
                    len(self._annotations),
                    path,
                )
            else:
                logger.warning(
                    "Action annotations file not found: %s — "
                    "all annotations will be empty",
                    path,
                )

    # ------------------------------------------------------------------
    # BaseEnv protocol implementation
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        return self._env.num_players

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    def reset(self) -> tuple[dict[str, str], str]:
        state, player_idx = self._env.reset()
        self._current_state = state
        self._current_player_idx = player_idx

        current_player = self._agent_ids[player_idx]
        observations = {
            aid: self._format_observation(state) if aid == current_player else ""
            for aid in self._agent_ids
        }
        return observations, current_player

    def step(
        self, action: str
    ) -> tuple[dict[str, str], dict[str, float], bool, dict[str, Any]]:
        # rlcard expects integer action IDs, not raw action strings
        action_id = self._action_to_id(action)
        state, player_idx = self._env.step(action_id)
        self._current_state = state
        self._current_player_idx = player_idx

        done = self._env.is_over()

        # Rewards only meaningful at game end
        if done:
            payoffs = self._env.get_payoffs()
            rewards = {
                self._agent_ids[i]: float(payoffs[i])
                for i in range(self.num_agents)
            }
        else:
            rewards = {aid: 0.0 for aid in self._agent_ids}

        # Build observations
        current_player = self._agent_ids[player_idx] if not done else ""
        observations: dict[str, str] = {}
        for aid in self._agent_ids:
            if not done and aid == current_player:
                observations[aid] = self._format_observation(state)
            else:
                observations[aid] = ""

        info: dict[str, Any] = {
            "current_player": current_player if not done else None,
            "done": done,
        }
        if done:
            # Determine winner (highest payoff)
            winner_idx = int(self._env.get_payoffs().argmax())
            info["winner"] = self._agent_ids[winner_idx]

        return observations, rewards, done, info

    def get_current_player(self) -> str:
        return self._agent_ids[self._current_player_idx]

    def get_legal_actions(self, agent_id: str) -> list[str]:
        if self._current_state is None:
            return []
        return list(self._current_state.get("raw_legal_actions", []))

    def get_action_annotations(
        self, agent_id: str
    ) -> dict[str, ActionAnnotation]:
        legal = self.get_legal_actions(agent_id)
        result: dict[str, ActionAnnotation] = {}
        for action_name in legal:
            if action_name in self._annotations:
                result[action_name] = self._annotations[action_name]
            else:
                # Unknown action — return empty annotation
                result[action_name] = ActionAnnotation()
        return result

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _action_to_id(self, action: str) -> int:
        """Convert an action string (e.g. 'call') to its rlcard integer ID.

        RLCard internally uses integer action IDs. The mapping is stored in
        ``self._env._decode_action`` / ``self._env.actions``. We build a
        reverse lookup from the environment's action list.
        """
        # rlcard envs expose an `actions` list where index = action_id
        actions_list = getattr(self._env, "actions", None)
        if actions_list is not None:
            for idx, act in enumerate(actions_list):
                if act == action:
                    return idx

        # Fallback: if action is already numeric
        try:
            return int(action)
        except (ValueError, TypeError):
            pass

        raise ValueError(
            f"Cannot map action '{action}' to an integer ID. "
            f"Known actions: {actions_list}"
        )

    def _format_observation(self, state: dict[str, Any]) -> str:
        """Convert rlcard raw_obs into a human-readable text for LLM prompts."""
        raw = state.get("raw_obs", {})
        legal = state.get("raw_legal_actions", [])

        # Dispatch to game-specific formatter if available
        formatter_name = f"_format_{self._game_name.replace('-', '_')}"
        formatter = getattr(self, formatter_name, None)
        if formatter is not None:
            return formatter(raw, legal)

        # Generic fallback
        return self._format_generic(raw, legal)

    # --- Game-specific formatters ---

    def _format_leduc_holdem(
        self, raw: dict[str, Any], legal: list[str]
    ) -> str:
        hand = raw.get("hand", "unknown")
        public = raw.get("public_card", None)
        my_chips = raw.get("my_chips", "?")
        all_chips = raw.get("all_chips", [])
        action_record = raw.get("action_record", [])

        lines = [
            f"Your hand: {hand}",
            f"Public card: {public if public else 'not revealed yet'}",
            f"Your chips in pot: {my_chips}",
            f"All players' chips: {all_chips}",
        ]
        if action_record:
            history = ", ".join(
                f"player {pid}: {act}" for act, pid in action_record
            )
            lines.append(f"Action history: {history}")
        lines.append(f"Legal actions: {', '.join(legal)}")
        return "\n".join(lines)

    def _format_generic(
        self, raw: dict[str, Any], legal: list[str]
    ) -> str:
        """Fallback formatter: dump raw_obs as key-value pairs."""
        lines = []
        for key, value in raw.items():
            lines.append(f"{key}: {value}")
        lines.append(f"Legal actions: {', '.join(legal)}")
        return "\n".join(lines)
