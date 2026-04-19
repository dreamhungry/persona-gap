# Persona-Gap

**Do LLM Agents Act as They Say?**  
A research framework for measuring the gap between expressed and behavioral personality in LLM agents.

---

## 🔍 Overview

Large Language Model (LLM) agents are often prompted with specific personalities (e.g., aggressive, cooperative). However, it remains unclear whether these *expressed traits* are reflected in their actual decision-making behavior.

This project introduces a framework to:

- Quantify **expressed personality** from language (via LLM-as-judge)
- Measure **behavioral personality** from actions (via trajectory analysis)
- Evaluate the **alignment (or gap)** between them
- Analyze **consistency over time** across episodes

---

## 🧠 Key Features

- 🎭 **Personality-conditioned LLM agents** — 4-dimensional personality vector (risk, aggression, cooperation, deception) injected via prompts
- 🎮 **Pluggable environments** — Registry-based adapter system supporting RLCard (Leduc Hold'em, UNO, Blackjack, etc.), TextArena, and custom environments
- 📊 **Dual-channel personality extraction** — Behavioral traits from action trajectories + Expressed traits from LLM text analysis
- 📏 **Alignment metrics** — L1 distance, cosine similarity, per-dimension gap analysis
- 🔁 **Temporal analysis** — Consistency and drift tracking across 50–100+ episodes
- 💾 **Checkpoint & resume** — Experiment interruption recovery with full state persistence
- 📈 **Visualization** — Radar charts, alignment heatmaps, drift line plots, comparison bar charts

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/persona-gap
cd persona-gap
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.11+
- An LLM API key (OpenAI, Anthropic, etc.) set as environment variable

```bash
export OPENAI_API_KEY="sk-..."
# or for Anthropic:
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 🚀 Quick Start

### 1. Run a single experiment

```bash
python scripts/run_experiment.py configs/experiment.toml
```

This will:
- Load the experiment configuration (agents, environment, personality vectors)
- Run 50 episodes of Leduc Hold'em with 2 LLM agents
- Log all interactions to `outputs/default/trajectory.jsonl`
- Save checkpoints for resume capability

### 2. Run batch experiments

```bash
python scripts/run_batch.py configs/batch_example.toml
```

Sweeps over all combinations of personality presets × memory on/off × communication on/off.

### 3. Resume an interrupted experiment

```bash
python scripts/run_experiment.py configs/experiment.toml
# It automatically resumes from the last checkpoint
```

To force a fresh start:
```bash
python scripts/run_experiment.py configs/experiment.toml --no-resume
```

---

## 📁 Project Structure

```
persona-gap/
├── configs/                      # TOML configuration files
│   ├── experiment.toml           # Default experiment config
│   ├── batch_example.toml        # Batch sweep config example
│   ├── presets/                   # Personality presets
│   │   ├── aggressive.toml
│   │   ├── cooperative.toml
│   │   ├── deceptive.toml
│   │   └── conservative.toml
│   └── envs/                     # Environment-specific action annotations
│       └── leduc_holdem.toml
├── src/persona_gap/              # Main package
│   ├── core/                     # Data models & config loading
│   ├── envs/                     # Environment adapters (protocol + registry)
│   ├── agents/                   # LLM agent, memory, prompts
│   ├── llm/                      # LLM backend (litellm wrapper)
│   ├── runner/                   # Experiment engine, logger, checkpoint
│   ├── metrics/                  # Behavioral/expressed extraction, alignment, temporal
│   └── analysis/                 # Visualization tools
├── scripts/                      # Entry-point scripts
│   ├── run_experiment.py         # Single experiment
│   └── run_batch.py              # Batch sweep
└── tests/                        # Unit tests
```

---

## 🔧 Configuration

All experiments are driven by TOML configuration files. Example:

```toml
[experiment]
seed = 42
num_episodes = 50
output_dir = "outputs/my_experiment"

[env]
adapter = "rlcard"
game_name = "leduc-holdem"
action_annotations_path = "configs/envs/leduc_holdem.toml"

[[agents]]
agent_id = "agent_0"
model = "gpt-4o-mini"
memory_enabled = true
communication_enabled = true

[agents.personality]
risk = 0.8
aggression = 0.9
cooperation = 0.2
deception = 0.5
```

---

## 🌐 Adding a New Environment

1. **Write an adapter class** implementing the `BaseEnv` protocol (`envs/protocol.py`)
2. **Register it** using `@register_env("your_env_name")`
3. **Create an action annotation TOML** in `configs/envs/`

See `configs/envs/README.md` for details.

---

## 📊 Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Behavioral Personality** | `b_dim = #tagged_actions / #total_actions` | Personality inferred from action choices |
| **Expressed Personality** | LLM-as-judge on reasoning text | Personality inferred from language |
| **Alignment** | `-‖b - e‖₁` | Negative L1 distance (higher = better) |
| **Temporal Consistency** | `1 - (1/T)Σ‖bₜ - b̄‖²` | Behavioral stability over episodes |
| **Personality Drift** | `‖bₜ - b₁‖` | Distance from initial behavior |

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
