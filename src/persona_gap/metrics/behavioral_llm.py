"""BehavioralLLMJudge — LLM-based behavioral personality extraction.

Uses an LLM to judge each step's personality dimensions based on:
- Game state description
- Action taken
- Context and history

Provides flexibility and semantic understanding at the cost of:
- Higher latency and API cost
- Potential variability in judgments
- Dependency on LLM quality
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import litellm

from persona_gap.core.models import PersonalityVector, StepRecord


class BehavioralLLMJudge:
    """Extract behavioral personality using LLM-as-judge."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 5,
    ) -> None:
        """Initialize LLM judge.

        Args:
            model: LLM model identifier (litellm format)
            temperature: Sampling temperature (0 for deterministic)
            api_key: API key (None = use environment variable)
            api_base: Custom API base URL
            batch_size: Number of steps to process per LLM call
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.batch_size = batch_size

    def _build_judge_prompt(self, steps: list[dict[str, Any]]) -> str:
        """Build prompt for judging multiple steps.

        Args:
            steps: List of dicts with keys: observation, action, legal_actions

        Returns:
            Formatted prompt string
        """
        prompt = """You are an expert poker psychologist analyzing player behavior in Leduc Hold'em.

For each game step below, evaluate the player's personality across 4 dimensions (0-1 scale):

1. **Risk**: How much risk does this action involve given the game state?
   - Low (0.0-0.3): Safe actions with strong hands or low commitment
   - Medium (0.4-0.6): Moderate commitment or uncertainty
   - High (0.7-1.0): Aggressive betting with weak hands or high stakes

2. **Aggression**: How much pressure does this action apply?
   - Low (0.0-0.3): Passive actions (check, fold)
   - Medium (0.4-0.6): Calling or small raises
   - High (0.7-1.0): Strong raises that force opponents to react

3. **Cooperation**: Does this action show restraint or escalate conflict?
   - Low (0.0-0.3): Highly escalating actions
   - Medium (0.4-0.6): Moderate engagement
   - High (0.7-1.0): Non-escalating, maintaining status quo

4. **Deception**: Is the action misrepresenting hand strength?
   - Low (0.0-0.3): Honest signaling (strong hand → raise, weak hand → fold)
   - Medium (0.4-0.6): Ambiguous or neutral signaling
   - High (0.7-1.0): Bluffing (weak hand → raise) or trapping (strong hand → check/call)

**Card strength reference for Leduc Hold'em:**
- Pair (e.g., hand=Q, public=Q): Very strong (~0.95 equity)
- High card > public card: Medium strength (~0.65 equity)
- High card < public card: Weak (~0.25 equity)
- Pre-flop: K > Q > J (linear 0.33-1.0 range)

---

"""
        for i, step in enumerate(steps, 1):
            prompt += f"""**Step {i}:**
```
{step['observation']}
```
Legal actions: {', '.join(step['legal_actions'])}
**Chosen action: {step['action']}**

"""

        prompt += """---

Respond with a JSON array (one object per step) in this exact format:
```json
[
  {
    "step": 1,
    "risk": 0.75,
    "aggression": 0.85,
    "cooperation": 0.20,
    "deception": 0.80,
    "reasoning": "Weak hand (J) with no public card, but chose to raise. High risk and deception (bluff)."
  },
  ...
]
```

Respond ONLY with the JSON array, no additional text."""

        return prompt

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
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
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
            prompt = self._build_judge_prompt(batch)
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
        return PersonalityVector(
            risk=totals["risk"] / count,
            aggression=totals["aggression"] / count,
            cooperation=totals["cooperation"] / count,
            deception=totals["deception"] / count,
        )

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
