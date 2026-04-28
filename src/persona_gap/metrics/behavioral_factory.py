"""Factory for creating behavioral extractors based on configuration.

Supports three extraction methods:
- annotation: Original annotation-based approach (BehavioralExtractor)
- scorer: State-aware scoring (BehavioralScorer)
- llm: LLM-as-judge (BehavioralLLMJudge)

Extensibility:
For scorer method, you can customize parser and calculator:
- game_name: Automatically selects appropriate parser/calculator
- Custom implementations: Pass parser/calculator directly to BehavioralScorer
"""

from __future__ import annotations

from typing import Any

from persona_gap.core.models import ActionAnnotation, LLMConfig
from persona_gap.metrics.behavioral import BehavioralExtractor
from persona_gap.metrics.behavioral_llm import BehavioralLLMJudge
from persona_gap.metrics.behavioral_scorer import (
    BehavioralScorer,
    LeducStateParser,
    PokerScoreCalculator,
    ScoreCalculator,
    StateParser,
)


def create_behavioral_extractor(
    method: str,
    action_annotations: dict[str, ActionAnnotation] | None = None,
    llm_config: LLMConfig | None = None,
    game_name: str | None = None,
    parser: StateParser | None = None,
    calculator: ScoreCalculator | None = None,
) -> BehavioralExtractor | BehavioralScorer | BehavioralLLMJudge:
    """Create a behavioral extractor based on the specified method.

    Args:
        method: Extraction method - "annotation", "scorer", or "llm"
        action_annotations: Required for "annotation" method
        llm_config: Required for "llm" method
        game_name: Game environment name (e.g., "leduc-holdem"). Used to auto-select parser/calculator.
        parser: Custom StateParser (for scorer method). If None, auto-selects based on game_name.
        calculator: Custom ScoreCalculator (for scorer method). If None, auto-selects based on game_name.

    Returns:
        Appropriate extractor instance

    Raises:
        ValueError: If method is invalid or required config is missing
    """
    method = method.lower()

    if method == "annotation":
        if action_annotations is None:
            raise ValueError("action_annotations required for annotation method")
        return BehavioralExtractor(action_annotations)

    elif method == "scorer":
        # Auto-select parser and calculator based on game_name if not provided
        if parser is None and game_name:
            parser = _get_parser_for_game(game_name)
        if calculator is None and game_name:
            calculator = _get_calculator_for_game(game_name)

        return BehavioralScorer(parser=parser, calculator=calculator)

    elif method == "llm":
        if llm_config is None:
            raise ValueError("llm_config required for llm method")

        # Use judge-specific config if available, otherwise fall back to main config
        model = llm_config.judge_model or llm_config.model
        temperature = llm_config.judge_temperature

        return BehavioralLLMJudge(
            model=model,
            temperature=temperature,
            api_key=llm_config.api_key if llm_config.api_key else None,
            api_base=llm_config.api_base,
        )

    else:
        raise ValueError(
            f"Invalid behavioral method: {method}. "
            f"Must be 'annotation', 'scorer', or 'llm'"
        )


def _get_parser_for_game(game_name: str) -> StateParser:
    """Select appropriate StateParser based on game name."""
    game_name = game_name.lower()

    if "leduc" in game_name or "holdem" in game_name:
        return LeducStateParser()
    # Future extensions:
    # elif "texas" in game_name:
    #     return TexasHoldemParser()
    # elif "uno" in game_name:
    #     return UnoParser()
    else:
        # Default to Leduc parser
        return LeducStateParser()


def _get_calculator_for_game(game_name: str) -> ScoreCalculator:
    """Select appropriate ScoreCalculator based on game name."""
    game_name = game_name.lower()

    # Poker games share the same calculator
    if any(
        keyword in game_name
        for keyword in ["leduc", "holdem", "texas", "poker", "omaha"]
    ):
        return PokerScoreCalculator()
    # Future extensions:
    # elif "uno" in game_name:
    #     return UnoScoreCalculator()
    else:
        # Default to poker calculator
        return PokerScoreCalculator()
