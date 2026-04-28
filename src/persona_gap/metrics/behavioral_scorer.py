"""BehavioralScorer — state-aware behavioral personality extraction.

This scorer computes behavioral personality by analyzing the relationship
between game state (hand strength, pot odds, etc.) and chosen actions.

Architecture:
- StateParser: Extracts structured game state from observation string
- ScoreCalculator: Computes 4-dimensional personality scores from state + action
- BehavioralScorer: Coordinates parsing and scoring (game-agnostic)

Extensibility:
To add a new game, implement StateParser and ScoreCalculator for that game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from persona_gap.core.models import PersonalityVector, StepRecord


# =============================================================================
# Abstract Interfaces
# =============================================================================


class GameState(ABC):
    """Abstract representation of game state.
    
    Subclasses should implement game-specific state attributes.
    """

    pass


class StateParser(ABC):
    """Abstract interface for parsing observation strings into GameState."""

    @abstractmethod
    def parse(self, observation: str, legal_actions: list[str]) -> GameState | None:
        """Parse observation string into structured GameState.
        
        Args:
            observation: Raw observation string from environment
            legal_actions: List of legal action strings
            
        Returns:
            Parsed GameState or None if parsing fails
        """
        pass


class ScoreCalculator(ABC):
    """Abstract interface for calculating personality scores from game state."""

    @abstractmethod
    def score_step(self, state: GameState, action: str) -> dict[str, float]:
        """Calculate personality scores for a single step.
        
        Args:
            state: Parsed game state
            action: Chosen action string
            
        Returns:
            Dict with keys: risk, aggression, cooperation, deception
            Each value should be in [0, 1]
        """
        pass


# =============================================================================
# Leduc Hold'em Implementation
# =============================================================================


class LeducState(GameState):
    """Leduc Hold'em specific game state."""

    def __init__(
        self,
        hand: str,
        public_card: str | None,
        my_chips: int,
        all_chips: list[int],
        legal_actions: list[str],
    ):
        self.hand = hand
        self.public_card = public_card
        self.my_chips = my_chips
        self.all_chips = all_chips
        self.legal_actions = legal_actions

    def equity(self) -> float:
        """Estimate hand strength/equity in [0, 1].

        Uses exact rank-based calculation for Leduc Hold'em:
        - Deck: 6 cards (2x J, 2x Q, 2x K)
        - Strong pair > High card > Low card
        """
        rank_values = {"J": 1, "Q": 2, "K": 3}
        hand_rank = rank_values.get(self.hand, 0)

        if self.public_card is None:
            # Pre-flop: only private card matters
            return 0.33 + (hand_rank - 1) * 0.335
        else:
            public_rank = rank_values.get(self.public_card, 0)
            if self.hand == self.public_card:
                return 0.95  # Made pair
            elif hand_rank > public_rank:
                return 0.65  # Higher card
            else:
                return 0.25  # Lower card

    def pot_odds(self) -> float:
        """Calculate current pot odds."""
        if not self.all_chips or len(self.all_chips) < 2:
            return 0.5
        max_chips = max(self.all_chips)
        total_pot = sum(self.all_chips)
        call_amount = max_chips - self.my_chips
        if total_pot == 0:
            return 0.0
        return min(call_amount / total_pot, 1.0)

    def uncertainty(self) -> float:
        """Estimate information uncertainty (higher pre-flop)."""
        return 0.8 if self.public_card is None else 0.3


class LeducStateParser(StateParser):
    """Parse Leduc Hold'em observation strings."""

    def parse(self, observation: str, legal_actions: list[str]) -> LeducState | None:
        """Parse Leduc Hold'em observation string.

        Expected format:
            Your hand: K
            Public card: Q / not revealed yet
            Your chips in pot: 2
            All players' chips: [2, 1]
            Legal actions: raise, call, fold
        """
        import re

        try:
            # Extract hand
            hand_match = re.search(r"Your hand:\s*([JQK])", observation)
            hand = hand_match.group(1) if hand_match else "unknown"

            # Extract public card
            public_match = re.search(r"Public card:\s*([JQK])", observation)
            public_card = public_match.group(1) if public_match else None

            # Extract my chips
            my_chips_match = re.search(r"Your chips in pot:\s*(\d+)", observation)
            my_chips = int(my_chips_match.group(1)) if my_chips_match else 0

            # Extract all chips
            all_chips_match = re.search(
                r"All players' chips:\s*\[([\d,\s]+)\]", observation
            )
            if all_chips_match:
                all_chips = [
                    int(x.strip()) for x in all_chips_match.group(1).split(",")
                ]
            else:
                all_chips = []

            return LeducState(hand, public_card, my_chips, all_chips, legal_actions)
        except Exception:
            return None


class PokerScoreCalculator(ScoreCalculator):
    """Generic poker score calculator.
    
    Suitable for Leduc Hold'em, Texas Hold'em, and other poker variants
    where state provides equity() and uncertainty() methods.
    """

    def score_step(self, state: GameState, action: str) -> dict[str, float]:
        """Score personality dimensions for poker games.
        
        Assumes state has: equity(), uncertainty(), legal_actions attributes
        """
        if not isinstance(state, LeducState):
            # Fallback for non-Leduc states (future Texas Hold'em etc.)
            return self._default_scores(action)

        equity = state.equity()
        uncertainty = state.uncertainty()

        # Action commitment and escalation levels
        commitment = {"raise": 1.0, "call": 0.5, "check": 0.3, "fold": 0.0}
        escalation = {"raise": 1.0, "call": 0.4, "check": 0.2, "fold": 0.0}

        act_commit = commitment.get(action, 0.5)
        act_escalation = escalation.get(action, 0.5)

        # --- Risk: commitment × (1 - equity) × uncertainty ---
        risk_score = act_commit * (1 - equity) * (0.5 + 0.5 * uncertainty)
        risk_score = min(risk_score, 1.0)

        # --- Aggression: escalation level with equity adjustment ---
        aggression_score = act_escalation * (0.7 + 0.3 * (1 - equity))
        aggression_score = min(aggression_score, 1.0)

        # --- Deception: bluff + trap components ---
        deception_score = 0.0

        # Bluff: weak hand + aggressive action
        if action == "raise" and equity < 0.5:
            bluff_strength = (0.5 - equity) * 2
            deception_score += bluff_strength * 0.8

        # Trap: strong hand + passive action (when raise available)
        if (
            action in ["check", "call"]
            and equity > 0.7
            and "raise" in state.legal_actions
        ):
            trap_strength = (equity - 0.7) / 0.3
            deception_score += trap_strength * 0.6

        deception_score = min(deception_score, 1.0)

        # --- Cooperation: restraint (non-escalation) with trap penalty ---
        cooperation_score = 1.0 - act_escalation

        # Reduce cooperation if trapping
        if (
            action in ["check", "call"]
            and equity > 0.7
            and "raise" in state.legal_actions
        ):
            trap_penalty = (equity - 0.7) / 0.3 * 0.5
            cooperation_score -= trap_penalty

        cooperation_score = max(min(cooperation_score, 1.0), 0.0)

        return {
            "risk": risk_score,
            "aggression": aggression_score,
            "cooperation": cooperation_score,
            "deception": deception_score,
        }

    def _default_scores(self, action: str) -> dict[str, float]:
        """Fallback scoring when state type is unknown."""
        escalation = {"raise": 1.0, "call": 0.4, "check": 0.2, "fold": 0.0}
        act_escalation = escalation.get(action, 0.5)

        return {
            "risk": act_escalation * 0.7,
            "aggression": act_escalation,
            "cooperation": 1.0 - act_escalation,
            "deception": 0.0,
        }


# =============================================================================
# Generic BehavioralScorer
# =============================================================================


class BehavioralScorer:
    """Extract behavioral personality using state-aware scoring.
    
    This class is game-agnostic and delegates to parser and calculator.
    """

    def __init__(
        self,
        parser: StateParser | None = None,
        calculator: ScoreCalculator | None = None,
    ) -> None:
        """Initialize the scorer.
        
        Args:
            parser: StateParser implementation (defaults to LeducStateParser)
            calculator: ScoreCalculator implementation (defaults to PokerScoreCalculator)
        """
        self.parser = parser or LeducStateParser()
        self.calculator = calculator or PokerScoreCalculator()

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

        # Accumulate scores
        totals = {"risk": 0.0, "aggression": 0.0, "cooperation": 0.0, "deception": 0.0}
        valid_count = 0

        for record in filtered:
            state = self.parser.parse(record.observation, record.legal_actions)
            if state is None:
                continue

            scores = self.calculator.score_step(state, record.action)
            for dim in totals:
                totals[dim] += scores[dim]
            valid_count += 1

        # Average over valid steps
        if valid_count == 0:
            return PersonalityVector(
                risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5
            )

        return PersonalityVector(
            risk=totals["risk"] / valid_count,
            aggression=totals["aggression"] / valid_count,
            cooperation=totals["cooperation"] / valid_count,
            deception=totals["deception"] / valid_count,
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
        from collections import defaultdict

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

# class TexasHoldemState(GameState):
#     """Texas Hold'em specific state."""
#     def __init__(self, hole_cards: list[str], community_cards: list[str], ...):
#         ...
#
# class TexasStateParser(StateParser):
#     def parse(self, observation: str, legal_actions: list[str]) -> TexasHoldemState | None:
#         ...
#
# class UnoState(GameState):
#     """Uno card game state."""
#     def __init__(self, hand: list[str], top_card: str, ...):
#         ...
#
# class UnoScoreCalculator(ScoreCalculator):
#     def score_step(self, state: GameState, action: str) -> dict[str, float]:
#         # Completely different scoring logic for Uno
#         ...
