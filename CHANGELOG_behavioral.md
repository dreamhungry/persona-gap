# Behavioral Methods Changelog

## v2.2 - Grounded Expressed Personality Context

**Date**: 2026-04-28

### Major Changes

✨ **Expressed Prompt Builder**: Added `ExpressedPromptBuilder` with `TextOnlyExpressedPromptBuilder` and `GroundedExpressedPromptBuilder`.

✨ **Grounded Expressed Mode**: Added `expressed_context = "grounded"` to judge expressed personality with private reasoning, public message, observation, legal actions, and selected action.

✨ **Explicit Reasoning Semantics**: Prompts now define private reasoning as the agent's internal thought or true intent, and public message as outward communication.

### Compatibility

- Default remains `expressed_context = "text-only"` for backward-compatible language-only analysis.
- Existing `ExpressedExtractor(backend)` calls still work and use text-only mode.
- `grounded` mode is opt-in and helps detect deception from private-public or message-action mismatch.

### Bug Fixes

🐛 **Score Clamping**: Fixed `BehavioralLLMJudge.extract` to clamp LLM-returned scores to `[0, 1]` range, preventing `PersonalityVector` validation errors.

---

## v2.1 - LLM Judge Architecture Refactor


**Date**: 2026-04-28

### Major Changes

✨ **Modular LLM Prompt Builder**: Refactored `BehavioralLLMJudge` to support multiple games through `PromptBuilder` interface

**Before:**
```python
class BehavioralLLMJudge:
    def _build_judge_prompt(...):
        # Hardcoded Leduc Hold'em rules in prompt
        prompt = """You are analyzing Leduc Hold'em..."""
```

**After:**
```python
# Game-agnostic judge
class BehavioralLLMJudge:
    def __init__(self, ..., prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder  # Dependency injection

# Game-specific prompt builder
class LeducPromptBuilder(PromptBuilder):
    def build_judge_prompt(self, steps) -> str:
        # Leduc-specific context
        ...
```

### New Abstract Interface

- **`PromptBuilder`**: Interface for building game-specific LLM judge prompts
  - `build_judge_prompt(steps)`: Construct full prompt with game context
  - `get_common_instructions()`: Reusable personality dimension definitions
  - `get_output_format()`: Standard JSON output specification

### Concrete Implementation

- `LeducPromptBuilder`: Leduc Hold'em specific prompts with poker rules

### Factory Enhancements

```python
# Auto-selects prompt builder based on game_name
create_behavioral_extractor(
    method="llm",
    llm_config=config,
    game_name="leduc-holdem",  # Triggers LeducPromptBuilder
)

# Or inject custom implementation
create_behavioral_extractor(
    method="llm",
    llm_config=config,
    prompt_builder=UnoPromptBuilder(),
)
```

### Benefits

✅ **Extensibility**: Add new games by implementing `PromptBuilder`  
✅ **Reusability**: Common personality definitions shared across games  
✅ **Consistency**: All games use same JSON output format  
✅ **Backward Compatible**: Existing code still works (defaults to Leduc)

---

## v2.0 - Architecture Refactor (Game-Agnostic Design)

**Date**: 2026-04-28

### Major Changes

✨ **Modular Architecture**: Refactored `BehavioralScorer` to support multiple games through dependency injection

**Before:**
```python
# Tightly coupled to Leduc Hold'em
class BehavioralScorer:
    def parse_leduc_observation(...):  # Hardcoded
        ...
    def score_step(...):  # Mixed parsing and scoring logic
        ...
```

**After:**
```python
# Game-agnostic coordinator
class BehavioralScorer:
    def __init__(self, parser: StateParser, calculator: ScoreCalculator):
        self.parser = parser       # Dependency injection
        self.calculator = calculator
```

### New Abstract Interfaces

1. **`GameState`**: Abstract base for game-specific state
2. **`StateParser`**: Interface for parsing observations → state
3. **`ScoreCalculator`**: Interface for scoring (state, action) → personality

### Concrete Implementations

- `LeducState` / `LeducStateParser`: Leduc Hold'em support
- `PokerScoreCalculator`: Generic poker scoring (reusable for Texas Hold'em, Omaha, etc.)

### Factory Enhancements

```python
# Auto-selects parser/calculator based on game_name
create_behavioral_extractor(
    method="scorer",
    game_name="leduc-holdem",  # or "texas-holdem", "uno", etc.
)

# Or inject custom implementations
create_behavioral_extractor(
    method="scorer",
    parser=MyGameParser(),
    calculator=MyGameCalculator(),
)
```

### Benefits

✅ **Extensibility**: Add new games without modifying core `BehavioralScorer`  
✅ **Testability**: Each component (parser, calculator) tested independently  
✅ **Reusability**: `PokerScoreCalculator` works for all poker variants  
✅ **Backward Compatible**: Existing code still works (defaults to Leduc)

### Migration Guide

**No changes needed for existing code!** Default behavior preserved:

```python
# Still works
scorer = BehavioralScorer()  # Auto-uses LeducStateParser + PokerScoreCalculator
judge = BehavioralLLMJudge(model="gpt-4o-mini")  # Auto-uses LeducPromptBuilder
```

**To add a new game:**

1. Implement `StateParser` for your game (for scorer method)
2. Implement `PromptBuilder` for your game (for llm method)
3. Optionally implement custom `ScoreCalculator`
4. Register in factory or inject directly

See `docs/behavioral_methods.md` for detailed examples.

---

## v1.0 - Initial Implementation

**Date**: 2026-04-27

### Features

✨ **Three extraction methods**: annotation, scorer, llm  
✨ **Deception detection**: Scorer method detects bluffs and traps  
✨ **Configuration-based switching**: Select method via `behavioral_method` config  

### Files Added

- `src/persona_gap/metrics/behavioral_scorer.py`: State-aware scoring
- `src/persona_gap/metrics/behavioral_llm.py`: LLM-as-judge
- `src/persona_gap/metrics/behavioral_factory.py`: Factory pattern
- `tests/test_behavioral_methods.py`: Comprehensive test suite
- `docs/behavioral_methods.md`: Usage documentation
- `examples/compare_behavioral_methods.py`: Comparison script

### Experimental Validation

Tested on Leduc Hold'em trajectories:

| Method | Deception Detection |
|--------|-------------------|
| Annotation | ❌ Always 0.0 |
| Scorer | ✅ 0.808 |
| LLM | ✅ 0.535 |

**Key Finding**: Annotation method fails to detect context-dependent behaviors like bluffing.
