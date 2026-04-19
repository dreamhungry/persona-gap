"""TemporalAnalyzer — measure behavioral consistency and drift over episodes.

Metrics:
- Temporal Consistency: 1 - (1/T) * Σ|b_t - b̄|²   (higher = more stable)
- Personality Drift: |b_t - b_1|                     (distance from initial behavior)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from persona_gap.core.models import PersonalityVector


@dataclass
class TemporalResult:
    """Result of temporal consistency analysis."""

    agent_id: str

    # Mean behavioral personality across all episodes
    mean_behavioral: PersonalityVector

    # Consistency score (1 = perfectly stable, 0 = highly variable)
    consistency: float

    # Per-episode drift from the first episode
    drift_series: list[float] = field(default_factory=list)

    # Per-episode behavioral vectors (for plotting)
    episode_vectors: dict[int, PersonalityVector] = field(default_factory=dict)


class TemporalAnalyzer:
    """Analyze behavioral personality stability over time."""

    @staticmethod
    def analyze(
        per_episode: dict[int, PersonalityVector],
        agent_id: str = "",
    ) -> TemporalResult:
        """Compute temporal consistency and drift.

        Args:
            per_episode: {episode_id: behavioral PersonalityVector}.
            agent_id: Agent identifier for the result.

        Returns:
            TemporalResult with consistency score and drift series.
        """
        if not per_episode:
            return TemporalResult(
                agent_id=agent_id,
                mean_behavioral=PersonalityVector(
                    risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5
                ),
                consistency=1.0,
            )

        episodes = sorted(per_episode.keys())
        vectors = [per_episode[ep].to_list() for ep in episodes]
        T = len(vectors)
        ndim = 4

        # Mean behavioral vector
        mean = [sum(v[d] for v in vectors) / T for d in range(ndim)]
        mean_pv = PersonalityVector(
            risk=mean[0],
            aggression=mean[1],
            cooperation=mean[2],
            deception=mean[3],
        )

        # Temporal Consistency = 1 - (1/T) * Σ ||b_t - b̄||²
        variance_sum = 0.0
        for v in vectors:
            sq_dist = sum((v[d] - mean[d]) ** 2 for d in range(ndim))
            variance_sum += sq_dist

        # Normalize: max possible variance per dim is 1.0, so max total is 4.0
        consistency = 1.0 - (variance_sum / T) / ndim if T > 0 else 1.0
        consistency = max(0.0, min(1.0, consistency))

        # Personality Drift: ||b_t - b_1|| for each t
        first = vectors[0]
        drift_series: list[float] = []
        for v in vectors:
            dist = math.sqrt(sum((v[d] - first[d]) ** 2 for d in range(ndim)))
            drift_series.append(round(dist, 4))

        return TemporalResult(
            agent_id=agent_id,
            mean_behavioral=mean_pv,
            consistency=round(consistency, 4),
            drift_series=drift_series,
            episode_vectors=dict(per_episode),
        )
