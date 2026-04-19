"""Visualization tools for Persona-Gap experiment results.

Generates:
- Personality radar charts (spider/polar charts)
- Alignment heatmaps
- Drift line charts
- Multi-agent comparison bar charts
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from persona_gap.core.models import PersonalityVector
from persona_gap.metrics.alignment import AlignmentResult
from persona_gap.metrics.temporal import TemporalResult

# Style defaults
sns.set_theme(style="whitegrid")
_DIM_NAMES = PersonalityVector.dim_names()
_DIM_LABELS = ["Risk", "Aggression", "Cooperation", "Deception"]


def _save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Personality Radar Chart
# ---------------------------------------------------------------------------

def plot_personality_radar(
    vectors: dict[str, PersonalityVector],
    title: str = "Personality Profiles",
    output_path: str | Path = "personality_radar.png",
) -> None:
    """Draw a radar (spider) chart comparing multiple personality vectors.

    Args:
        vectors: {label: PersonalityVector} — e.g. {"Agent 0 (behavioral)": pv, ...}
        title: Chart title.
        output_path: Where to save the PNG.
    """
    num_dims = len(_DIM_LABELS)
    angles = np.linspace(0, 2 * math.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(
        [a * 180 / math.pi for a in angles[:-1]], _DIM_LABELS
    )
    ax.set_ylim(0, 1)

    colors = plt.cm.Set2(np.linspace(0, 1, len(vectors)))
    for (label, pv), color in zip(vectors.items(), colors):
        values = pv.to_list() + pv.to_list()[:1]
        ax.plot(angles, values, "o-", label=label, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title(title, pad=20, fontsize=13)

    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Alignment Heatmap
# ---------------------------------------------------------------------------

def plot_alignment_heatmap(
    results: dict[str, AlignmentResult],
    title: str = "Persona Gap: Behavioral vs Expressed",
    output_path: str | Path = "alignment_heatmap.png",
) -> None:
    """Draw a heatmap of per-dimension differences for multiple agents.

    Args:
        results: {agent_label: AlignmentResult}
        title: Chart title.
        output_path: Where to save the PNG.
    """
    agents = list(results.keys())
    data = np.array([
        [
            r.diff_risk,
            r.diff_aggression,
            r.diff_cooperation,
            r.diff_deception,
        ]
        for r in results.values()
    ])

    fig, ax = plt.subplots(figsize=(8, max(3, len(agents) * 0.8)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=_DIM_LABELS,
        yticklabels=agents,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Personality Dimension")
    ax.set_ylabel("Agent")

    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Drift Line Chart
# ---------------------------------------------------------------------------

def plot_drift(
    temporal_results: dict[str, TemporalResult],
    title: str = "Personality Drift Over Episodes",
    output_path: str | Path = "personality_drift.png",
) -> None:
    """Plot personality drift (distance from first episode) over time.

    Args:
        temporal_results: {agent_label: TemporalResult}
        title: Chart title.
        output_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, tr in temporal_results.items():
        episodes = list(range(len(tr.drift_series)))
        ax.plot(episodes, tr.drift_series, "o-", label=label, markersize=4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Drift (L2 distance from Episode 1)")
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.set_ylim(bottom=0)

    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Multi-agent Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_agent_comparison(
    agents: dict[str, dict[str, PersonalityVector]],
    title: str = "Behavioral vs Expressed Personality",
    output_path: str | Path = "agent_comparison.png",
) -> None:
    """Grouped bar chart comparing behavioral vs expressed for each agent.

    Args:
        agents: {agent_id: {"behavioral": PV, "expressed": PV}}
        title: Chart title.
        output_path: Where to save the PNG.
    """
    agent_labels = list(agents.keys())
    n_agents = len(agent_labels)
    n_dims = len(_DIM_LABELS)
    x = np.arange(n_dims)
    width = 0.35

    fig, axes = plt.subplots(
        1, n_agents, figsize=(5 * n_agents, 5), sharey=True
    )
    if n_agents == 1:
        axes = [axes]

    for ax, agent_label in zip(axes, agent_labels):
        beh = agents[agent_label]["behavioral"].to_list()
        exp = agents[agent_label]["expressed"].to_list()

        ax.bar(x - width / 2, beh, width, label="Behavioral", color="#4C72B0")
        ax.bar(x + width / 2, exp, width, label="Expressed", color="#DD8452")

        ax.set_xticks(x)
        ax.set_xticklabels(_DIM_LABELS, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(agent_label)
        ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_path)
