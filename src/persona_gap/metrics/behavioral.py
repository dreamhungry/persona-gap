"""BehavioralExtractor — compute behavioral personality from action trajectories.

Given a list of StepRecords and the corresponding action annotations,
calculates the behavioral personality vector as the ratio of personality-tagged
actions across all steps for each agent.

    b_dim = (# actions tagged as dim) / (# total actions)
"""

from __future__ import annotations

from collections import defaultdict

from persona_gap.core.models import ActionAnnotation, PersonalityVector, StepRecord


class BehavioralExtractor:
    """Extract behavioral personality vectors from trajectory data."""

    def __init__(
        self, action_annotations: dict[str, ActionAnnotation]
    ) -> None:
        """
        Args:
            action_annotations: Mapping of action_name -> ActionAnnotation,
                                loaded from the environment's TOML config.
        """
        self.annotations = action_annotations

    def extract(
        self,
        records: list[StepRecord],
        agent_id: str | None = None,
    ) -> PersonalityVector:
        """Compute behavioral personality from step records.

        Args:
            records: List of StepRecord from the trajectory.
            agent_id: If provided, filter to this agent only.

        Returns:
            PersonalityVector with each dimension as a ratio in [0, 1].
        """
        filtered = records
        if agent_id is not None:
            filtered = [r for r in records if r.agent_id == agent_id]

        if not filtered:
            return PersonalityVector(
                risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5
            )

        counts = {
            "risk": 0,
            "aggression": 0,
            "cooperation": 0,
            "deception": 0,
        }
        total = len(filtered)

        for record in filtered:
            ann = self.annotations.get(record.action, ActionAnnotation())
            if ann.is_risky:
                counts["risk"] += 1
            if ann.is_aggressive:
                counts["aggression"] += 1
            if ann.is_cooperative:
                counts["cooperation"] += 1
            if ann.is_deceptive:
                counts["deception"] += 1

        return PersonalityVector(
            risk=counts["risk"] / total,
            aggression=counts["aggression"] / total,
            cooperation=counts["cooperation"] / total,
            deception=counts["deception"] / total,
        )

    def extract_per_episode(
        self,
        records: list[StepRecord],
        agent_id: str,
    ) -> dict[int, PersonalityVector]:
        """Compute behavioral personality for each episode separately.

        Returns:
            {episode_id: PersonalityVector} dict.
        """
        by_episode: dict[int, list[StepRecord]] = defaultdict(list)
        for r in records:
            if r.agent_id == agent_id:
                by_episode[r.episode_id].append(r)

        result: dict[int, PersonalityVector] = {}
        for ep_id in sorted(by_episode.keys()):
            result[ep_id] = self.extract(by_episode[ep_id])
        return result
