"""LLMAgent — the core agent that combines personality, memory, and LLM decision-making.

An LLMAgent:
1. Receives observations and legal actions from the environment.
2. Optionally communicates with other agents (if communication_enabled).
3. Decides on an action via LLM call (with personality prompt + memory context).
4. Falls back to a random legal action if the LLM response is unparseable.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from persona_gap.agents.memory import Memory
from persona_gap.agents.prompts import (
    communication_prompt,
    decision_prompt,
    episode_summary_prompt,
    personality_to_text,
)
from persona_gap.core.models import AgentConfig, PersonalityVector
from persona_gap.llm.backend import LLMBackend

logger = logging.getLogger(__name__)


class LLMAgent:
    """An LLM-powered agent with personality injection and episodic memory.

    Attributes:
        agent_id: Unique identifier (e.g. "agent_0").
        personality: The 4-dimensional personality vector.
        config: Full agent configuration.
        memory: Episodic memory module.
        backend: LLM calling backend.
    """

    def __init__(self, config: AgentConfig, backend: LLMBackend | None = None) -> None:
        self.agent_id = config.agent_id
        self.personality = config.personality
        self.config = config

        self.memory = Memory(max_episodes=10)
        self._personality_text = personality_to_text(config.personality)

        # LLM backend (shared or per-agent)
        self.backend = backend or LLMBackend(
            model=config.model or "gpt-4o-mini",
            temperature=config.temperature if config.temperature is not None else 0.7,
            max_tokens=config.max_tokens if config.max_tokens is not None else 512,
        )

    def act(
        self,
        observation: str,
        legal_actions: list[str],
    ) -> tuple[str, str]:
        """Choose an action given the current observation.

        Args:
            observation: Human-readable text describing the current game state.
            legal_actions: List of valid action strings.

        Returns:
            (action, reasoning) tuple.
            If parsing fails, action is a random legal action and
            reasoning is marked as "[FALLBACK]".
        """
        memory_summary = self.memory.summarize() if self.config.memory_enabled else ""

        messages = decision_prompt(
            personality_text=self._personality_text,
            memory_summary=memory_summary,
            observation=observation,
            legal_actions=legal_actions,
        )

        try:
            result = self.backend.call(messages)
            action = result.get("action", "")
            reasoning = result.get("reasoning", "")

            # Validate action is legal
            if action not in legal_actions:
                logger.warning(
                    "Agent %s chose illegal action '%s'; legal: %s. "
                    "Attempting fuzzy match...",
                    self.agent_id,
                    action,
                    legal_actions,
                )
                action = self._fuzzy_match_action(action, legal_actions)
                if action is None:
                    action = random.choice(legal_actions)
                    reasoning = f"[FALLBACK] Original action not in legal actions. {reasoning}"
                    logger.warning(
                        "Agent %s: fuzzy match failed, falling back to random: %s",
                        self.agent_id,
                        action,
                    )

            return action, reasoning

        except RuntimeError:
            # All LLM retries exhausted
            action = random.choice(legal_actions)
            logger.error(
                "Agent %s: LLM call failed, falling back to random action: %s",
                self.agent_id,
                action,
            )
            return action, "[FALLBACK] LLM call failed after retries."

    def communicate(
        self,
        observation: str,
        opponent_message: str | None = None,
    ) -> tuple[str, str]:
        """Generate a communication message to send to the opponent.

        Args:
            observation: Current game state text.
            opponent_message: Message received from the opponent (if any).

        Returns:
            (message, reasoning) tuple.
        """
        if not self.config.communication_enabled:
            return "", ""

        memory_summary = self.memory.summarize() if self.config.memory_enabled else ""

        messages = communication_prompt(
            personality_text=self._personality_text,
            memory_summary=memory_summary,
            observation=observation,
            opponent_message=opponent_message,
        )

        try:
            result = self.backend.call(messages)
            return result.get("message", ""), result.get("reasoning", "")
        except RuntimeError:
            logger.error(
                "Agent %s: communication LLM call failed.", self.agent_id
            )
            return "", "[FALLBACK] Communication LLM call failed."

    def update_memory(
        self,
        episode_id: int,
        total_reward: float,
        key_actions: list[str],
    ) -> str:
        """Generate and store an episode summary in memory.

        Args:
            episode_id: The episode number.
            total_reward: Agent's total reward for this episode.
            key_actions: Notable actions taken during the episode.

        Returns:
            The generated summary text.
        """
        if not self.config.memory_enabled:
            return ""

        messages = episode_summary_prompt(
            agent_id=self.agent_id,
            episode_id=episode_id,
            total_reward=total_reward,
            key_actions=key_actions,
        )

        try:
            summary = self.backend.call_text(messages)
        except RuntimeError:
            summary = (
                f"Episode {episode_id}: reward={total_reward}, "
                f"actions={len(key_actions)}"
            )
            logger.error(
                "Agent %s: summary generation failed, using fallback.",
                self.agent_id,
            )

        self.memory.add(summary)
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_match_action(
        action: str, legal_actions: list[str]
    ) -> str | None:
        """Try to match the LLM's output to a legal action (case-insensitive)."""
        action_lower = action.strip().lower()
        for la in legal_actions:
            if la.lower() == action_lower:
                return la
        # Check if legal action is contained in the response
        for la in legal_actions:
            if la.lower() in action_lower:
                return la
        return None
