"""Post-hoc analysis script — compute all metrics and generate visualizations.

Reads the trajectory data produced by run_experiment.py and computes:
1. Behavioral personality (from action annotations)
2. Expressed personality (via LLM-as-judge)
3. Alignment / Persona Gap (behavioral vs expressed)
4. Temporal consistency and drift
5. Win rates and reward statistics
6. Visualization charts (radar, heatmap, drift, comparison)

Usage:
    python scripts/run_analysis.py outputs/default
    python scripts/run_analysis.py outputs/default --no-expressed
    python scripts/run_analysis.py outputs/default --config configs/experiment.toml
    python scripts/run_analysis.py outputs/default --debug  # Enable DEBUG logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


def _load_config(output_dir: Path, config_path: str | None) -> Any:
    """Load the experiment config — from explicit path or from saved config.json."""
    from persona_gap.core.config import load_config
    from persona_gap.core.models import ExperimentConfig

    if config_path:
        return load_config(config_path)

    # Fall back to the config saved by ExperimentRunner
    saved = output_dir / "config.json"
    if saved.exists():
        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentConfig.model_validate(data)

    print(
        f"Error: no config.json in {output_dir} and no --config specified.",
        file=sys.stderr,
    )
    sys.exit(1)


class TracingLLMBackend:
    """Wrapper that traces LLM calls and saves to file."""

    def __init__(self, backend: Any, trace_file: Path):
        self._backend = backend
        self._trace_file = trace_file
        self._call_count = 0

    def call(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call backend and save trace."""
        self._call_count += 1
        result = self._backend.call(messages)

        # Check for out-of-range values
        out_of_range = {}
        for key in ["risk", "aggression", "cooperation", "deception"]:
            val = result.get(key)
            if val is not None and (val < 0.0 or val > 1.0):
                out_of_range[key] = val

        trace = {
            "call_id": self._call_count,
            "messages": messages,
            "response": result,
        }
        if out_of_range:
            trace["out_of_range_warning"] = out_of_range

        with open(self._trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Persona-Gap experiment results"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to experiment output directory (e.g. outputs/default)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config (optional; defaults to output_dir/config.json)",
    )
    parser.add_argument(
        "--no-expressed",
        action="store_true",
        help="Skip expressed personality extraction (saves LLM calls)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for LLM intermediate results",
    )
    args = parser.parse_args()

    # Configure logging - only enable DEBUG for persona_gap modules
    logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")
    if args.debug:
        logging.getLogger("persona_gap").setLevel(logging.DEBUG)
    else:
        logging.getLogger("persona_gap").setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Imports (delayed for structlog config) ----
    from persona_gap.core.config import load_action_annotations
    from persona_gap.core.models import PersonalityVector
    from persona_gap.llm.backend import LLMBackend
    from persona_gap.metrics.alignment import AlignmentCalculator
    from persona_gap.metrics.behavioral_factory import create_behavioral_extractor
    from persona_gap.metrics.expressed import (
        ExpressedExtractor,
        create_expressed_prompt_builder,
    )
    from persona_gap.metrics.temporal import TemporalAnalyzer
    from persona_gap.runner.logger import TrajectoryLogger
    from persona_gap.analysis.visualize import (
        plot_agent_comparison,
        plot_alignment_heatmap,
        plot_drift,
        plot_personality_radar,
    )

    # ---- Load data ----
    config = _load_config(output_dir, args.config)

    traj_path = output_dir / "trajectory.jsonl"
    ep_path = output_dir / "episodes.jsonl"

    if not traj_path.exists():
        print(f"Error: trajectory.jsonl not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading trajectory data", path=str(traj_path))
    records = TrajectoryLogger.load_trajectory(traj_path)
    episodes = TrajectoryLogger.load_episodes(ep_path) if ep_path.exists() else []

    logger.info(
        "Data loaded",
        total_steps=len(records),
        total_episodes=len(episodes),
    )

    # Discover agent IDs from trajectory
    agent_ids = sorted(set(r.agent_id for r in records))
    logger.info("Agents found", agents=agent_ids)

    # ---- Action annotations ----
    annotations = load_action_annotations(config.env.action_annotations_path)

    # ---- Analysis output directory ----
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Win Rates & Reward Statistics
    # ==================================================================
    print("\n" + "=" * 60)
    print("  WIN RATES & REWARD STATISTICS")
    print("=" * 60)

    if episodes:
        winner_counts: Counter[str] = Counter()
        draw_count = 0
        total_rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        for ep in episodes:
            if ep.winner:
                winner_counts[ep.winner] += 1
            else:
                draw_count += 1
            for aid, reward in ep.rewards.items():
                total_rewards[aid] = total_rewards.get(aid, 0.0) + reward

        n_episodes = len(episodes)
        print(f"\n  Total episodes: {n_episodes}")
        print(f"  Draws: {draw_count} ({draw_count / n_episodes * 100:.1f}%)")
        for aid in agent_ids:
            wins = winner_counts.get(aid, 0)
            win_rate = wins / n_episodes * 100
            avg_reward = total_rewards.get(aid, 0.0) / n_episodes
            print(
                f"  {aid}: wins={wins} ({win_rate:.1f}%), "
                f"avg_reward={avg_reward:.3f}, "
                f"total_reward={total_rewards.get(aid, 0.0):.3f}"
            )
    else:
        print("  No episode data found.")

    # ==================================================================
    # 2. Behavioral Personality
    # ==================================================================
    print("\n" + "=" * 60)
    
    # Determine which method to use
    method = getattr(config.env, 'behavioral_method', 'annotation')
    print(f"  BEHAVIORAL PERSONALITY (method: {method})")
    print("=" * 60)

    # Create extractor based on method
    beh_extractor = create_behavioral_extractor(
        method=method,
        action_annotations=annotations if method == 'annotation' else None,
        llm_config=config.llm if method == 'llm' else None,
        game_name=config.env.game_name,  # Auto-select parser/calculator for scorer method
    )
    
    behavioral_vectors: dict[str, PersonalityVector] = {}
    per_episode_behavioral: dict[str, dict[int, PersonalityVector]] = {}

    for aid in agent_ids:
        per_ep = beh_extractor.extract_per_episode(records, aid)
        per_episode_behavioral[aid] = per_ep

        # Aggregate: average across episodes
        if per_ep:
            avg_risk = sum(pv.risk for pv in per_ep.values()) / len(per_ep)
            avg_agg = sum(pv.aggression for pv in per_ep.values()) / len(per_ep)
            avg_coop = sum(pv.cooperation for pv in per_ep.values()) / len(per_ep)
            avg_dec = sum(pv.deception for pv in per_ep.values()) / len(per_ep)
            beh_pv = PersonalityVector(
                risk=avg_risk, aggression=avg_agg, cooperation=avg_coop, deception=avg_dec
            )
        else:
            beh_pv = beh_extractor.extract(records, agent_id=aid)

        behavioral_vectors[aid] = beh_pv
        print(f"\n  {aid} ({len(per_ep)} episodes):")
        print(f"    risk={beh_pv.risk:.3f}  aggression={beh_pv.aggression:.3f}  "
              f"cooperation={beh_pv.cooperation:.3f}  deception={beh_pv.deception:.3f}")

    # ==================================================================
    # 3. Expressed Personality (LLM-as-judge)
    # ==================================================================
    expressed_vectors: dict[str, PersonalityVector] = {}
    per_episode_expressed: dict[str, dict[int, PersonalityVector]] = {}
    llm_trace_file = analysis_dir / "llm_traces.jsonl"

    if not args.no_expressed:
        expressed_context = getattr(config.env, "expressed_context", "text-only")
        print("\n" + "=" * 60)
        print(f"  EXPRESSED PERSONALITY (mode: {expressed_context}, via LLM-as-judge)")
        print("=" * 60)

        # Clear previous trace file
        if llm_trace_file.exists():
            llm_trace_file.unlink()

        judge_backend = LLMBackend.judge_from_llm_config(config.llm)
        tracing_backend = TracingLLMBackend(judge_backend, llm_trace_file)
        exp_extractor = ExpressedExtractor(
            tracing_backend,
            prompt_builder=create_expressed_prompt_builder(expressed_context),
        )

        per_episode_expressed: dict[str, dict[int, PersonalityVector]] = {}

        for aid in agent_ids:
            logger.info("Extracting expressed personality per episode", agent_id=aid)
            per_ep = exp_extractor.extract_per_episode(records, agent_id=aid)
            per_episode_expressed[aid] = per_ep

            # Aggregate: average across episodes
            if per_ep:
                avg_risk = sum(pv.risk for pv in per_ep.values()) / len(per_ep)
                avg_agg = sum(pv.aggression for pv in per_ep.values()) / len(per_ep)
                avg_coop = sum(pv.cooperation for pv in per_ep.values()) / len(per_ep)
                avg_dec = sum(pv.deception for pv in per_ep.values()) / len(per_ep)
                exp_pv = PersonalityVector(
                    risk=avg_risk, aggression=avg_agg, cooperation=avg_coop, deception=avg_dec
                )
            else:
                exp_pv = PersonalityVector(risk=0.5, aggression=0.5, cooperation=0.5, deception=0.5)

            expressed_vectors[aid] = exp_pv
            print(f"\n  {aid} ({len(per_ep)} episodes):")
            print(f"    risk={exp_pv.risk:.3f}  aggression={exp_pv.aggression:.3f}  "
                  f"cooperation={exp_pv.cooperation:.3f}  deception={exp_pv.deception:.3f}")

        print(f"\n  LLM traces saved to: {llm_trace_file}")
    else:
        print("\n  [Skipped expressed personality extraction (--no-expressed)]")

    # ==================================================================
    # 4. Alignment / Persona Gap
    # ==================================================================
    if expressed_vectors:
        print("\n" + "=" * 60)
        print("  ALIGNMENT (Persona Gap: behavioral vs expressed)")
        print("=" * 60)

        alignment_results = {}
        for aid in agent_ids:
            if aid in behavioral_vectors and aid in expressed_vectors:
                result = AlignmentCalculator.compute(
                    behavioral_vectors[aid], expressed_vectors[aid]
                )
                alignment_results[aid] = result
                print(f"\n  {aid}:")
                print(f"    L1 distance:       {result.l1_distance:.4f}")
                print(f"    Cosine similarity: {result.cosine_similarity:.4f}")
                print(f"    Alignment score:   {result.alignment_score:.4f}")
                print(f"    Per-dimension diff (behavioral - expressed):")
                print(f"      risk:        {result.diff_risk:+.4f}")
                print(f"      aggression:  {result.diff_aggression:+.4f}")
                print(f"      cooperation: {result.diff_cooperation:+.4f}")
                print(f"      deception:   {result.diff_deception:+.4f}")

    # ==================================================================
    # 5. Temporal Consistency & Drift
    # ==================================================================
    print("\n" + "=" * 60)
    print("  TEMPORAL CONSISTENCY & DRIFT")
    print("=" * 60)

    temporal_results = {}
    for aid in agent_ids:
        if aid in per_episode_behavioral:
            tr = TemporalAnalyzer.analyze(per_episode_behavioral[aid], aid)
            temporal_results[aid] = tr
            print(f"\n  {aid}:")
            print(f"    Consistency: {tr.consistency:.4f}")
            print(f"    Mean behavioral: "
                  f"risk={tr.mean_behavioral.risk:.3f}  "
                  f"aggression={tr.mean_behavioral.aggression:.3f}  "
                  f"cooperation={tr.mean_behavioral.cooperation:.3f}  "
                  f"deception={tr.mean_behavioral.deception:.3f}")
            if tr.drift_series:
                print(f"    Max drift: {max(tr.drift_series):.4f}")
                print(f"    Final drift: {tr.drift_series[-1]:.4f}")

    # ==================================================================
    # 6. Assigned Personality (from config, for reference)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  ASSIGNED PERSONALITY (from config)")
    print("=" * 60)

    assigned_vectors: dict[str, PersonalityVector] = {}
    for agent_cfg in config.agents:
        pv = agent_cfg.personality
        assigned_vectors[agent_cfg.agent_id] = pv
        print(f"\n  {agent_cfg.agent_id}:")
        print(f"    risk={pv.risk:.3f}  aggression={pv.aggression:.3f}  "
              f"cooperation={pv.cooperation:.3f}  deception={pv.deception:.3f}")

    # ==================================================================
    # 7. Visualization
    # ==================================================================
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 7a. Radar chart — assigned vs behavioral (vs expressed if available)
    radar_vectors: dict[str, PersonalityVector] = {}
    for aid in agent_ids:
        if aid in assigned_vectors:
            radar_vectors[f"{aid} (assigned)"] = assigned_vectors[aid]
        if aid in behavioral_vectors:
            radar_vectors[f"{aid} (behavioral)"] = behavioral_vectors[aid]
        if aid in expressed_vectors:
            radar_vectors[f"{aid} (expressed)"] = expressed_vectors[aid]

    if radar_vectors:
        path = analysis_dir / "personality_radar.png"
        plot_personality_radar(radar_vectors, output_path=path)
        print(f"  Saved: {path}")

    # 7b. Alignment heatmap
    if expressed_vectors and alignment_results:
        path = analysis_dir / "alignment_heatmap.png"
        plot_alignment_heatmap(alignment_results, output_path=path)
        print(f"  Saved: {path}")

    # 7c. Drift chart
    if temporal_results:
        path = analysis_dir / "personality_drift.png"
        plot_drift(temporal_results, output_path=path)
        print(f"  Saved: {path}")

    # 7d. Agent comparison bar chart
    if expressed_vectors:
        comparison_data = {}
        for aid in agent_ids:
            if aid in behavioral_vectors and aid in expressed_vectors:
                comparison_data[aid] = {
                    "behavioral": behavioral_vectors[aid],
                    "expressed": expressed_vectors[aid],
                }
        if comparison_data:
            path = analysis_dir / "agent_comparison.png"
            plot_agent_comparison(comparison_data, output_path=path)
            print(f"  Saved: {path}")

    # ==================================================================
    # 8. Save metrics as JSON
    # ==================================================================
    metrics_summary: dict[str, Any] = {"agents": {}}

    for aid in agent_ids:
        agent_metrics: dict[str, Any] = {}

        if aid in assigned_vectors:
            agent_metrics["assigned"] = assigned_vectors[aid].model_dump()
        if aid in behavioral_vectors:
            agent_metrics["behavioral"] = behavioral_vectors[aid].model_dump()
        if aid in per_episode_behavioral:
            agent_metrics["behavioral_per_episode"] = {
                str(ep_id): pv.model_dump()
                for ep_id, pv in per_episode_behavioral[aid].items()
            }
        if aid in expressed_vectors:
            agent_metrics["expressed"] = expressed_vectors[aid].model_dump()
        if aid in per_episode_expressed:
            agent_metrics["expressed_per_episode"] = {
                str(ep_id): pv.model_dump()
                for ep_id, pv in per_episode_expressed[aid].items()
            }
        if expressed_vectors and aid in alignment_results:
            ar = alignment_results[aid]
            agent_metrics["alignment"] = {
                "l1_distance": ar.l1_distance,
                "cosine_similarity": ar.cosine_similarity,
                "alignment_score": ar.alignment_score,
                "diff_risk": ar.diff_risk,
                "diff_aggression": ar.diff_aggression,
                "diff_cooperation": ar.diff_cooperation,
                "diff_deception": ar.diff_deception,
            }
        if aid in temporal_results:
            tr = temporal_results[aid]
            agent_metrics["temporal"] = {
                "consistency": tr.consistency,
                "max_drift": max(tr.drift_series) if tr.drift_series else 0.0,
                "final_drift": tr.drift_series[-1] if tr.drift_series else 0.0,
                "mean_behavioral": tr.mean_behavioral.model_dump(),
            }

        # Win rate
        if episodes:
            wins = sum(1 for ep in episodes if ep.winner == aid)
            total = len(episodes)
            total_reward = sum(ep.rewards.get(aid, 0.0) for ep in episodes)
            agent_metrics["performance"] = {
                "wins": wins,
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                "total_reward": round(total_reward, 4),
                "avg_reward": round(total_reward / total, 4) if total > 0 else 0.0,
            }

        metrics_summary["agents"][aid] = agent_metrics

    metrics_summary["total_episodes"] = len(episodes)
    metrics_summary["total_steps"] = len(records)

    metrics_path = analysis_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Metrics saved: {metrics_path}")
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
