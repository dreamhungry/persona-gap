"""Tests for metrics: behavioral extraction, alignment, temporal analysis."""

import pytest

from persona_gap.core.models import ActionAnnotation, PersonalityVector, StepRecord
from persona_gap.metrics.alignment import AlignmentCalculator
from persona_gap.metrics.behavioral import BehavioralExtractor
from persona_gap.metrics.temporal import TemporalAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotations() -> dict[str, ActionAnnotation]:
    return {
        "raise": ActionAnnotation(is_risky=True, is_aggressive=True),
        "call": ActionAnnotation(is_cooperative=True),
        "fold": ActionAnnotation(),
        "check": ActionAnnotation(),
    }


def _make_records(actions: list[str], agent_id: str = "agent_0", episode_id: int = 0) -> list[StepRecord]:
    return [
        StepRecord(
            episode_id=episode_id,
            step=i,
            agent_id=agent_id,
            observation="obs",
            legal_actions=["raise", "call", "fold", "check"],
            action=action,
            reasoning="reason",
        )
        for i, action in enumerate(actions)
    ]


# ---------------------------------------------------------------------------
# BehavioralExtractor tests
# ---------------------------------------------------------------------------


class TestBehavioralExtractor:
    def test_all_raises(self):
        ext = BehavioralExtractor(_make_annotations())
        records = _make_records(["raise"] * 10)
        pv = ext.extract(records, agent_id="agent_0")
        assert pv.risk == 1.0
        assert pv.aggression == 1.0
        assert pv.cooperation == 0.0
        assert pv.deception == 0.0

    def test_all_calls(self):
        ext = BehavioralExtractor(_make_annotations())
        records = _make_records(["call"] * 10)
        pv = ext.extract(records, agent_id="agent_0")
        assert pv.risk == 0.0
        assert pv.cooperation == 1.0

    def test_mixed_actions(self):
        ext = BehavioralExtractor(_make_annotations())
        records = _make_records(["raise", "call", "fold", "check"])
        pv = ext.extract(records, agent_id="agent_0")
        assert pv.risk == 0.25  # 1 raise out of 4
        assert pv.aggression == 0.25
        assert pv.cooperation == 0.25
        assert pv.deception == 0.0

    def test_empty_records(self):
        ext = BehavioralExtractor(_make_annotations())
        pv = ext.extract([], agent_id="agent_0")
        # Should return neutral
        assert pv.risk == 0.5

    def test_per_episode(self):
        ext = BehavioralExtractor(_make_annotations())
        records = (
            _make_records(["raise"] * 5, episode_id=0) +
            _make_records(["call"] * 5, episode_id=1)
        )
        per_ep = ext.extract_per_episode(records, "agent_0")
        assert per_ep[0].risk == 1.0
        assert per_ep[1].cooperation == 1.0


# ---------------------------------------------------------------------------
# AlignmentCalculator tests
# ---------------------------------------------------------------------------


class TestAlignmentCalculator:
    def test_perfect_alignment(self):
        pv = PersonalityVector(risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5)
        result = AlignmentCalculator.compute(pv, pv)
        assert result.l1_distance == 0.0
        assert result.alignment_score == 0.0
        assert result.cosine_similarity == 1.0

    def test_maximum_misalignment(self):
        b = PersonalityVector(risk=1.0, aggression=1.0, cooperation=1.0, deception=1.0)
        e = PersonalityVector(risk=0.0, aggression=0.0, cooperation=0.0, deception=0.0)
        result = AlignmentCalculator.compute(b, e)
        assert result.l1_distance == 4.0
        assert result.alignment_score == -4.0

    def test_partial_difference(self):
        b = PersonalityVector(risk=0.8, aggression=0.2, cooperation=0.5, deception=0.5)
        e = PersonalityVector(risk=0.3, aggression=0.7, cooperation=0.5, deception=0.5)
        result = AlignmentCalculator.compute(b, e)
        assert result.diff_risk == pytest.approx(0.5, abs=0.01)
        assert result.diff_aggression == pytest.approx(-0.5, abs=0.01)
        assert result.diff_cooperation == 0.0


# ---------------------------------------------------------------------------
# TemporalAnalyzer tests
# ---------------------------------------------------------------------------


class TestTemporalAnalyzer:
    def test_perfect_consistency(self):
        pv = PersonalityVector(risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5)
        per_ep = {i: pv for i in range(10)}
        result = TemporalAnalyzer.analyze(per_ep, "agent_0")
        assert result.consistency == 1.0
        assert all(d == 0.0 for d in result.drift_series)

    def test_varying_behavior(self):
        per_ep = {
            0: PersonalityVector(risk=0.0, aggression=0.0, cooperation=0.0, deception=0.0),
            1: PersonalityVector(risk=1.0, aggression=1.0, cooperation=1.0, deception=1.0),
        }
        result = TemporalAnalyzer.analyze(per_ep, "agent_0")
        assert result.consistency < 1.0
        assert result.drift_series[0] == 0.0
        assert result.drift_series[1] > 0.0

    def test_empty_input(self):
        result = TemporalAnalyzer.analyze({}, "agent_0")
        assert result.consistency == 1.0
        assert result.drift_series == []
