"""Tests for different behavioral extraction methods.

Validates that all three methods (annotation, scorer, llm) work correctly
and produce reasonable personality vectors.
"""

from __future__ import annotations

import pytest

from persona_gap.core.models import ActionAnnotation, LLMConfig, PersonalityVector, StepRecord
from persona_gap.metrics.behavioral import BehavioralExtractor
from persona_gap.metrics.behavioral_scorer import BehavioralScorer
from persona_gap.metrics.behavioral_factory import create_behavioral_extractor


def _make_annotations() -> dict[str, ActionAnnotation]:
    """Create test action annotations (Leduc-like)."""
    return {
        "raise": ActionAnnotation(is_risky=True, is_aggressive=True),
        "call": ActionAnnotation(is_cooperative=True),
        "check": ActionAnnotation(),
        "fold": ActionAnnotation(),
    }


def _make_leduc_observation(
    hand: str, public_card: str | None, my_chips: int, all_chips: list[int]
) -> str:
    """Create realistic Leduc observation string."""
    public_line = f"Public card: {public_card}" if public_card else "Public card: not revealed yet"
    return f"""Your hand: {hand}
{public_line}
Your chips in pot: {my_chips}
All players' chips: {all_chips}
Legal actions: raise, call, fold"""


def _make_step_record(
    action: str,
    hand: str = "K",
    public_card: str | None = None,
    episode_id: int = 0,
    step: int = 0,
) -> StepRecord:
    """Create a test step record with Leduc observation."""
    obs = _make_leduc_observation(hand, public_card, 2, [2, 2])
    return StepRecord(
        episode_id=episode_id,
        step=step,
        agent_id="agent_0",
        observation=obs,
        legal_actions=["raise", "call", "fold"],
        action=action,
        reasoning="test",
    )


# =============================================================================
# Test Annotation Method (baseline)
# =============================================================================


class TestAnnotationMethod:
    """Tests for original annotation-based extraction."""

    def test_basic_extraction(self):
        ext = BehavioralExtractor(_make_annotations())
        records = [_make_step_record("raise"), _make_step_record("call")]
        pv = ext.extract(records, agent_id="agent_0")
        
        # Raise is risky+aggressive, call is cooperative
        assert pv.risk == 0.5
        assert pv.aggression == 0.5
        assert pv.cooperation == 0.5
        assert pv.deception == 0.0  # No deceptive actions in annotations


# =============================================================================
# Test Scorer Method (state-aware)
# =============================================================================


class TestScorerMethod:
    """Tests for state-aware scoring extraction."""

    def test_weak_hand_raise_is_risky_and_deceptive(self):
        """Weak hand + raise should show high risk and deception (bluff)."""
        scorer = BehavioralScorer()
        records = [_make_step_record("raise", hand="J", public_card=None)]
        pv = scorer.extract(records, agent_id="agent_0")
        
        # Should detect bluff
        assert pv.risk > 0.5
        assert pv.deception > 0.2  # Adjusted: J is weakest but still has some equity pre-flop
        assert pv.aggression > 0.7

    def test_strong_hand_raise_is_low_risk(self):
        """Strong hand + raise should show lower risk."""
        scorer = BehavioralScorer()
        records = [_make_step_record("raise", hand="K", public_card="K")]  # Pair
        pv = scorer.extract(records, agent_id="agent_0")
        
        # Strong hand reduces risk
        assert pv.risk < 0.3
        assert pv.aggression > 0.7
        assert pv.deception < 0.3

    def test_strong_hand_check_is_deceptive_trap(self):
        """Strong hand + check (when raise available) should detect trap."""
        scorer = BehavioralScorer()
        records = [_make_step_record("check", hand="K", public_card="K")]  # Pair
        pv = scorer.extract(records, agent_id="agent_0")
        
        # Should detect trap (slow play)
        assert pv.deception > 0.3
        assert pv.aggression < 0.3
        assert pv.cooperation < 0.7  # Reduced by trap penalty

    def test_deception_not_always_zero(self):
        """Deception should be non-zero for appropriate state-action pairs."""
        scorer = BehavioralScorer()
        
        # Mix of bluffs and honest actions
        records = [
            _make_step_record("raise", hand="J", public_card=None),  # Bluff
            _make_step_record("fold", hand="J", public_card="K"),    # Honest
            _make_step_record("check", hand="K", public_card="K"),   # Trap
        ]
        pv = scorer.extract(records, agent_id="agent_0")
        
        # Overall deception should be non-zero
        assert pv.deception > 0.0

    def test_custom_parser_calculator(self):
        """Test that custom parser/calculator can be injected."""
        from persona_gap.metrics.behavioral_scorer import (
            LeducStateParser,
            PokerScoreCalculator,
        )
        
        # Create scorer with explicit parser and calculator
        parser = LeducStateParser()
        calculator = PokerScoreCalculator()
        scorer = BehavioralScorer(parser=parser, calculator=calculator)
        
        records = [_make_step_record("raise", hand="K")]
        pv = scorer.extract(records, agent_id="agent_0")
        
        # Should work correctly
        assert isinstance(pv, PersonalityVector)
        assert 0.0 <= pv.risk <= 1.0


# =============================================================================
# Test Factory
# =============================================================================


class TestBehavioralFactory:
    """Tests for the factory function."""

    def test_create_annotation_extractor(self):
        ext = create_behavioral_extractor(
            method="annotation",
            action_annotations=_make_annotations(),
        )
        assert isinstance(ext, BehavioralExtractor)

    def test_create_scorer_extractor(self):
        ext = create_behavioral_extractor(method="scorer")
        assert isinstance(ext, BehavioralScorer)

    def test_create_scorer_with_game_name(self):
        """Test that game_name auto-selects appropriate parser/calculator."""
        ext = create_behavioral_extractor(
            method="scorer",
            game_name="leduc-holdem",
        )
        assert isinstance(ext, BehavioralScorer)
        
        # Should use Leduc parser
        from persona_gap.metrics.behavioral_scorer import LeducStateParser
        assert isinstance(ext.parser, LeducStateParser)

    def test_create_llm_extractor(self):
        llm_config = LLMConfig(model="gpt-4o-mini", api_key="test-key")
        ext = create_behavioral_extractor(method="llm", llm_config=llm_config)
        from persona_gap.metrics.behavioral_llm import BehavioralLLMJudge
        assert isinstance(ext, BehavioralLLMJudge)

    def test_invalid_method_raises_error(self):
        with pytest.raises(ValueError, match="Invalid behavioral method"):
            create_behavioral_extractor(method="invalid")

    def test_missing_annotations_raises_error(self):
        with pytest.raises(ValueError, match="action_annotations required"):
            create_behavioral_extractor(method="annotation")

    def test_missing_llm_config_raises_error(self):
        with pytest.raises(ValueError, match="llm_config required"):
            create_behavioral_extractor(method="llm")


# =============================================================================
# Integration Test
# =============================================================================


class TestMethodComparison:
    """Compare outputs across different methods."""

    def test_all_methods_produce_valid_vectors(self):
        """All methods should produce valid PersonalityVector."""
        records = [
            _make_step_record("raise", hand="J"),
            _make_step_record("call", hand="Q"),
            _make_step_record("fold", hand="J"),
        ]

        # Annotation method
        ext_ann = create_behavioral_extractor(
            method="annotation",
            action_annotations=_make_annotations(),
        )
        pv_ann = ext_ann.extract(records, agent_id="agent_0")
        assert isinstance(pv_ann, PersonalityVector)
        assert 0.0 <= pv_ann.risk <= 1.0

        # Scorer method
        ext_scorer = create_behavioral_extractor(method="scorer")
        pv_scorer = ext_scorer.extract(records, agent_id="agent_0")
        assert isinstance(pv_scorer, PersonalityVector)
        assert 0.0 <= pv_scorer.risk <= 1.0

        # Scorer should detect deception, annotation should not
        assert pv_scorer.deception > pv_ann.deception
