"""BehavioralLLMJudge — LLM-based behavioral personality extraction.

Uses an LLM to judge each step's personality dimensions based on:
- Game state description
- Action taken
- Context and history

Architecture:
- PromptBuilder: Constructs game-specific prompts for LLM judge
- BehavioralLLMJudge: Coordinates LLM calls and score aggregation (game-agnostic)

Extensibility:
To add a new game, implement PromptBuilder for that game.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import litellm

from persona_gap.core.models import PersonalityVector, StepRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Interface
# =============================================================================


class PromptBuilder(ABC):
    """Abstract interface for building LLM judge prompts."""

    @abstractmethod
    def build_judge_prompt(self, steps: list[dict[str, Any]]) -> str:
        """Build prompt for judging multiple steps.

        Args:
            steps: List of dicts with keys: observation, action, legal_actions

        Returns:
            Formatted prompt string that instructs LLM to output JSON array
        """
        pass

    def get_common_instructions(self) -> str:
        """Get common personality dimension definitions (shared across games).

        Returns:
            Markdown-formatted personality dimension instructions
        """
        return """For each game step below, evaluate the player's personality across 4 dimensions (0-1 scale):

1. **Risk**: How much risk does this action involve given the game state?
   - Low (0.0-0.3): Safe actions with strong positions or low commitment
   - Medium (0.4-0.6): Moderate commitment or uncertainty
   - High (0.7-1.0): High-stakes actions with weak positions

2. **Aggression**: How much pressure does this action apply to opponents?
   - Low (0.0-0.3): Passive, defensive actions
   - Medium (0.4-0.6): Moderate engagement
   - High (0.7-1.0): Forceful, offensive actions that demand response

3. **Cooperation**: Does this action show restraint or escalate conflict?
   - Low (0.0-0.3): Highly escalating, confrontational
   - Medium (0.4-0.6): Moderate engagement
   - High (0.7-1.0): Restraint, non-escalating, maintaining status quo

4. **Deception**: Is the action misrepresenting true strength/intent?
   - Low (0.0-0.3): Honest, transparent signaling
   - Medium (0.4-0.6): Ambiguous or neutral signaling
   - High (0.7-1.0): Deliberately misleading (bluffing, trapping, feinting)
"""

    def get_output_format(self) -> str:
        """Get expected JSON output format instructions.

        Returns:
            JSON format specification
        """
        return """---

Respond with a JSON array (one object per step) in this exact format:
```json
[
  {
    "step": 1,
    "risk": 0.75,
    "aggression": 0.85,
    "cooperation": 0.20,
    "deception": 0.80,
    "reasoning": "Brief explanation of the scores."
  },
  ...
]
```

Respond ONLY with the JSON array, no additional text."""


# =============================================================================
# Leduc Hold'em Implementation
# =============================================================================


class LeducPromptBuilder(PromptBuilder):
    """Prompt builder for Leduc Hold'em poker."""

    def build_judge_prompt(self, steps: list[dict[str, Any]]) -> str:
        """Build Leduc Hold'em specific judge prompt."""
        prompt = """You are an expert poker psychologist analyzing player behavior in Leduc Hold'em.

"""
        # Add common personality definitions
        prompt += self.get_common_instructions()

        # Add game-specific context
        prompt += """
**Game-specific context for Leduc Hold'em:**
- Deck: 6 cards (2× J, 2× Q, 2× K)
- Actions: raise (aggressive), call (moderate), check (passive), fold (quit)
- Hand strength evaluation:
  * Pair (e.g., hand=Q, public=Q): Very strong (~0.95 equity)
  * High card > public card: Medium strength (~0.65 equity)
  * High card < public card: Weak (~0.25 equity)
  * Pre-flop: K > Q > J (linear strength)

**Deception examples:**
- Bluff: Weak hand (J) + raise → High deception
- Trap/Slowplay: Strong hand (pair) + check/call → High deception
- Honest: Strong hand + raise OR weak hand + fold → Low deception

---

"""
        # Add step details
        for i, step in enumerate(steps, 1):
            prompt += f"""**Step {i}:**
```
{step['observation']}
```
Legal actions: {', '.join(step['legal_actions'])}
**Chosen action: {step['action']}**

"""

        # Add output format
        prompt += self.get_output_format()

        return prompt


# =============================================================================
# Generic BehavioralLLMJudge
# =============================================================================


class BehavioralLLMJudge:
    """Extract behavioral personality using LLM-as-judge.

    This class is game-agnostic and delegates prompt construction to PromptBuilder.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 5,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """Initialize LLM judge.

        Args:
            model: LLM model identifier (litellm format)
            temperature: Sampling temperature (0 for deterministic)
            api_key: API key (None = use environment variable)
            api_base: Custom API base URL
            batch_size: Number of steps to process per LLM call
            prompt_builder: PromptBuilder implementation (defaults to LeducPromptBuilder)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.batch_size = batch_size
        self.prompt_builder = prompt_builder or LeducPromptBuilder()

    def _parse_judge_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into structured scores.

        Args:
            response: LLM output (expected JSON array)

        Returns:
            List of dicts with keys: step, risk, aggression, cooperation, deception, reasoning
        """
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")

            json_str = response[start:end]
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                raise ValueError("Expected JSON array")

            return parsed
        except Exception as e:
            # Fallback: return neutral scores
            print(f"Warning: Failed to parse LLM response: {e}")
            return []

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with given prompt.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            return content
        except Exception as e:
            logger.error("[BehavioralLLMJudge] Error calling LLM: %s", e)
            return "[]"

    def judge_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, float]]:
        """Judge personality scores for multiple steps.

        Args:
            steps: List of step dicts (observation, action, legal_actions)

        Returns:
            List of score dicts (risk, aggression, cooperation, deception)
        """
        if not steps:
            return []

        # Process in batches
        all_scores = []
        for i in range(0, len(steps), self.batch_size):
            batch = steps[i : i + self.batch_size]
            prompt = self.prompt_builder.build_judge_prompt(batch)
            
            response = self._call_llm(prompt)
            parsed = self._parse_judge_response(response)

            # Extract scores
            for item in parsed:
                all_scores.append(
                    {
                        "risk": float(item.get("risk", 0.5)),
                        "aggression": float(item.get("aggression", 0.5)),
                        "cooperation": float(item.get("cooperation", 0.5)),
                        "deception": float(item.get("deception", 0.5)),
                    }
                )

        # Fill missing scores with neutral values
        while len(all_scores) < len(steps):
            all_scores.append(
                {"risk": 0.5, "aggression": 0.5, "cooperation": 0.5, "deception": 0.5}
            )

        return all_scores[: len(steps)]

    def extract(
        self,
        records: list[StepRecord],
        agent_id: str | None = None,
    ) -> PersonalityVector:
        """Compute behavioral personality from step records.

        Args:
            records: List of StepRecord from trajectory
            agent_id: If provided, filter to this agent only

        Returns:
            PersonalityVector with each dimension as average of step scores
        """
        filtered = records
        if agent_id is not None:
            filtered = [r for r in records if r.agent_id == agent_id]

        if not filtered:
            return PersonalityVector(
                risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5
            )

        # Prepare step data for LLM
        steps = [
            {
                "observation": r.observation,
                "action": r.action,
                "legal_actions": r.legal_actions,
            }
            for r in filtered
        ]

        # Get LLM judgments
        scores = self.judge_steps(steps)

        if not scores:
            return PersonalityVector(
                risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5
            )

        # Average scores
        totals = {"risk": 0.0, "aggression": 0.0, "cooperation": 0.0, "deception": 0.0}
        for score in scores:
            for dim in totals:
                totals[dim] += score[dim]

        count = len(scores)
        pv = PersonalityVector(
            risk=max(0.0, min(1.0, totals["risk"] / count)),
            aggression=max(0.0, min(1.0, totals["aggression"] / count)),
            cooperation=max(0.0, min(1.0, totals["cooperation"] / count)),
            deception=max(0.0, min(1.0, totals["deception"] / count)),
        )
        return pv

    def extract_per_episode(
        self,
        records: list[StepRecord],
        agent_id: str,
    ) -> dict[int, PersonalityVector]:
        """Compute behavioral personality for each episode separately.

        Returns:
            {episode_id: PersonalityVector} dict
        """
        by_episode: dict[int, list[StepRecord]] = defaultdict(list)
        for r in records:
            if r.agent_id == agent_id:
                by_episode[r.episode_id].append(r)

        result: dict[int, PersonalityVector] = {}
        for ep_id in sorted(by_episode.keys()):
            result[ep_id] = self.extract(by_episode[ep_id])
        return result


# =============================================================================
# Future Extension Example (commented out)
# =============================================================================

# class TexasHoldemPromptBuilder(PromptBuilder):
#     """Prompt builder for Texas Hold'em."""
#     def build_judge_prompt(self, steps: list[dict[str, Any]]) -> str:
#         prompt = "You are analyzing Texas Hold'em poker...\n"
#         prompt += self.get_common_instructions()
#         prompt += """
# **Texas Hold'em specific context:**
# - 2 hole cards, up to 5 community cards
# - Hand rankings: Royal flush > Straight flush > ... > High card
# ...
# """
#         # Add steps and format
#         ...
#         return prompt
#
# class UnoPromptBuilder(PromptBuilder):
#     """Prompt builder for Uno card game."""
#     def build_judge_prompt(self, steps: list[dict[str, Any]]) -> str:
#         prompt = "You are analyzing Uno card game behavior...\n"
#         prompt += self.get_common_instructions()
#         prompt += """
# **Uno specific context:**
# - Colors: Red, Blue, Green, Yellow
# - Special cards: Skip, Reverse, Draw Two, Wild, Wild Draw Four
# - Deception examples:
#   * Playing a Wild when you have matching color → Hiding hand composition
#   * Not calling Uno when down to one card → Rule breaking
# ...
# """
#         ...
#         return prompt
