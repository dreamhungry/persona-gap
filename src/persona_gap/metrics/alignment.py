"""AlignmentCalculator — measure the gap between expressed and behavioral personality.

Core metrics:
- L1 distance (Manhattan distance)
- Cosine similarity
- Per-dimension difference
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from persona_gap.core.models import PersonalityVector


@dataclass
class AlignmentResult:
    """Result of alignment calculation between expressed and behavioral personality."""

    behavioral: PersonalityVector
    expressed: PersonalityVector

    # Overall metrics
    l1_distance: float        # Sum of absolute differences (lower = more aligned)
    cosine_similarity: float  # Range [-1, 1], higher = more aligned
    alignment_score: float    # Negative L1 distance (higher = better, as per paper)

    # Per-dimension differences (behavioral - expressed)
    diff_risk: float
    diff_aggression: float
    diff_cooperation: float
    diff_deception: float


class AlignmentCalculator:
    """Compute alignment between behavioral and expressed personality vectors."""

    @staticmethod
    def compute(
        behavioral: PersonalityVector,
        expressed: PersonalityVector,
    ) -> AlignmentResult:
        """Calculate alignment metrics.

        Args:
            behavioral: Personality estimated from actions.
            expressed: Personality estimated from text.

        Returns:
            AlignmentResult with all metrics.
        """
        b = behavioral.to_list()
        e = expressed.to_list()
        dims = PersonalityVector.dim_names()

        # Per-dimension differences
        diffs = [bi - ei for bi, ei in zip(b, e)]

        # L1 distance
        l1 = sum(abs(d) for d in diffs)

        # Cosine similarity
        dot = sum(bi * ei for bi, ei in zip(b, e))
        norm_b = math.sqrt(sum(x * x for x in b)) or 1e-8
        norm_e = math.sqrt(sum(x * x for x in e)) or 1e-8
        cosine = dot / (norm_b * norm_e)

        # Alignment score (negative L1, as per paper: higher = better)
        alignment = -l1

        return AlignmentResult(
            behavioral=behavioral,
            expressed=expressed,
            l1_distance=round(l1, 4),
            cosine_similarity=round(cosine, 4),
            alignment_score=round(alignment, 4),
            diff_risk=round(diffs[0], 4),
            diff_aggression=round(diffs[1], 4),
            diff_cooperation=round(diffs[2], 4),
            diff_deception=round(diffs[3], 4),
        )
