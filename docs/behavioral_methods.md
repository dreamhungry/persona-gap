# Behavioral Personality Extraction Methods

This document explains the three available methods for extracting behavioral personality from agent actions.

## Overview

| Method | Description | Key Advantage | Extensibility |
|--------|-------------|---------------|---------------|
| **annotation** | Uses pre-defined action tags | Fast, no parsing needed | Limited to annotated actions |
| **scorer** | State-aware scoring logic | Detects deception, adjusts for context | **Easily extensible to new games** |
| **llm** | LLM-as-judge evaluation | Captures nuanced behavior | Costly, requires API |

## Method Details

### 1. Annotation Method (Baseline)

Uses action annotations defined in `configs/envs/<game>.toml`:

```toml
[actions.raise]
is_risky = true
is_aggressive = true

[actions.call]
is_cooperative = true
```

**Limitations:**
- Cannot detect deception (always 0.0)
- No context awareness (same action = same score)

---

### 2. Scorer Method (State-Aware) ⭐ Recommended

**Architecture:** Modular and game-agnostic

```
BehavioralScorer (coordinator)
  ├── StateParser: observation → GameState
  └── ScoreCalculator: (GameState, action) → scores
```

**Scoring Logic:**

- **Risk**: `commitment × (1 - equity) × uncertainty`
- **Aggression**: Action escalation level (adjusted by equity)
- **Deception**:
  - **Bluff**: Weak hand + aggressive action
  - **Trap**: Strong hand + passive action
- **Cooperation**: Restraint/non-escalation (with trap penalty)

**Example:**

```python
from persona_gap.metrics.behavioral_scorer import BehavioralScorer

scorer = BehavioralScorer()
pv = scorer.extract(records, agent_id="agent_0")
```

**Extending to New Games:**

```python
from persona_gap.metrics.behavioral_scorer import (
    BehavioralScorer,
    GameState,
    StateParser,
    ScoreCalculator,
)

# Step 1: Define game-specific state
class UnoState(GameState):
    def __init__(self, hand: list[str], top_card: str, ...):
        self.hand = hand
        self.top_card = top_card
        # ...

# Step 2: Implement parser
class UnoStateParser(StateParser):
    def parse(self, observation: str, legal_actions: list[str]) -> UnoState | None:
        # Parse Uno observation string
        ...

# Step 3: Implement calculator
class UnoScoreCalculator(ScoreCalculator):
    def score_step(self, state: GameState, action: str) -> dict[str, float]:
        # Calculate scores based on Uno rules
        ...

# Step 4: Use custom implementations
scorer = BehavioralScorer(
    parser=UnoStateParser(),
    calculator=UnoScoreCalculator(),
)
```

**Or use factory with auto-selection:**

```python
from persona_gap.metrics.behavioral_factory import create_behavioral_extractor

# Factory auto-selects parser/calculator based on game_name
scorer = create_behavioral_extractor(
    method="scorer",
    game_name="uno",  # Triggers UnoStateParser + UnoScoreCalculator
)
```

---

### 3. LLM Method (Judge)

Uses LLM to evaluate each step with reasoning.

**Architecture:** Modular and game-agnostic

```
BehavioralLLMJudge (coordinator)
  └── PromptBuilder: Constructs game-specific prompts
```

**Prompt Structure:**

```
State: {observation}
Action: {action}
Context: {previous_actions}

Evaluate personality on 4 dimensions [0-1]:
- risk, aggression, cooperation, deception
```

**Configuration:**

```toml
[llm]
judge_model = "gpt-4o-mini"
judge_temperature = 0.3
```

**Cost Optimization:**
- Batching: Processes multiple steps per API call
- Caching: Avoids repeated evaluations

**Extending to New Games:**

```python
from persona_gap.metrics.behavioral_llm import PromptBuilder

class UnoPromptBuilder(PromptBuilder):
    def build_judge_prompt(self, steps: list[dict]) -> str:
        prompt = "You are analyzing Uno card game behavior...\n"
        prompt += self.get_common_instructions()  # Reuse personality definitions
        
        # Add Uno-specific context
        prompt += """
**Uno specific context:**
- Colors: Red, Blue, Green, Yellow
- Special cards: Skip, Reverse, Draw Two, Wild, Wild Draw Four
- Deception examples:
  * Playing Wild when you have matching color → Hiding hand composition
  * Holding onto Draw Four illegally → Rule exploitation
"""
        # Add steps and format
        for i, step in enumerate(steps, 1):
            prompt += f"Step {i}: {step['observation']}..."
        
        prompt += self.get_output_format()  # Standard JSON output
        return prompt

# Use custom prompt builder
judge = BehavioralLLMJudge(
    model="gpt-4o-mini",
    prompt_builder=UnoPromptBuilder(),
)
```

---

## Configuration

In `configs/experiment.toml`:

```toml
[env]
game_name = "leduc-holdem"  # Used for auto-selecting parser/calculator
behavioral_method = "scorer"  # Options: "annotation" | "scorer" | "llm"
```

## Usage Examples

### Basic Usage (Single Method)

```bash
# Edit configs/experiment.toml to set behavioral_method
python scripts/run_analysis.py outputs/test1
```

### Comparing All Methods

```bash
python examples/compare_behavioral_methods.py outputs/test1
```

**Example Output:**

```
Method: annotation
  risk=1.000  aggression=1.000  cooperation=0.000  deception=0.000

Method: scorer
  risk=0.905  aggression=1.000  cooperation=0.000  deception=0.808 ✓

Method: llm
  risk=0.765  aggression=0.705  cooperation=0.180  deception=0.535 ✓

Key Insights:
  [*] Scorer detected deception that annotation missed
```

---

## Experimental Results

Tested on `outputs/test1` (agent_0, all raise actions):

| Metric | Annotation | Scorer | LLM |
|--------|-----------|--------|-----|
| Risk | 1.000 | 0.905 | 0.765 |
| Aggression | 1.000 | 1.000 | 0.705 |
| Cooperation | 0.000 | 0.000 | 0.180 |
| **Deception** | **0.000** ❌ | **0.808** ✅ | **0.535** ✅ |

**Key Finding:** Annotation method fails to detect deception (bluffing/trapping), while scorer and LLM methods successfully identify it.

---

## Adding Support for a New Game

**Example: Adding Texas Hold'em**

1. **Define state class:**

```python
class TexasHoldemState(GameState):
    def __init__(self, hole_cards, community_cards, pot, ...):
        ...
    
    def equity(self) -> float:
        # Calculate hand strength using evaluator library
        ...
```

2. **Implement parser:**

```python
class TexasStateParser(StateParser):
    def parse(self, observation: str, legal_actions: list[str]):
        # Parse Texas Hold'em observation
        ...
```

3. **Reuse or customize calculator:**

```python
# Option A: Reuse existing PokerScoreCalculator
scorer = BehavioralScorer(
    parser=TexasStateParser(),
    calculator=PokerScoreCalculator(),  # Works for all poker variants
)

# Option B: Create Texas-specific calculator if needed
class TexasScoreCalculator(ScoreCalculator):
    def score_step(self, state, action):
        # Texas-specific scoring logic
        ...
```

4. **Register in factory (optional):**

```python
# In behavioral_factory.py
def _get_parser_for_game(game_name: str) -> StateParser:
    if "texas" in game_name:
        return TexasStateParser()
    ...
```

---

## Design Principles

✅ **Game-Agnostic Core**: `BehavioralScorer` doesn't know about specific games  
✅ **Dependency Injection**: Parser and calculator are injected, not hardcoded  
✅ **Auto-Selection**: Factory can auto-select based on `game_name`  
✅ **Backward Compatible**: Default to Leduc if no game specified  
✅ **Easy Testing**: Each component (parser, calculator) can be tested independently

---

## Troubleshooting

**Q: Deception is always 0.0**  
A: You're using annotation method. Switch to `scorer` or `llm`.

**Q: How to add a new game?**  
A: Implement the appropriate interfaces:
- **For scorer method**: Implement `StateParser` and optionally `ScoreCalculator`
- **For llm method**: Implement `PromptBuilder`
- See "Adding Support for a New Game" sections above for examples.

**Q: Scorer method fails on my custom game**  
A: Provide custom parser/calculator:
```python
scorer = BehavioralScorer(
    parser=MyGameParser(),
    calculator=MyGameCalculator(),
)
```

---

## References

- Code: `src/persona_gap/metrics/behavioral_scorer.py`
- Code: `src/persona_gap/metrics/expressed.py`
- Tests: `tests/test_behavioral_methods.py`
- Examples: `examples/compare_behavioral_methods.py`

