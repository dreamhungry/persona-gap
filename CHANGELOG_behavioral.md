# Behavioral Methods Changelog

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
```

**To add a new game:**

1. Implement `StateParser` for your game
2. Optionally implement custom `ScoreCalculator`
3. Register in factory or inject directly

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
