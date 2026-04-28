"""Example: Compare different behavioral extraction methods.

This script demonstrates how to use all three behavioral extraction methods
on the same trajectory and compare their outputs.

Usage:
    python examples/compare_behavioral_methods.py outputs/test1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from persona_gap.core.config import load_action_annotations, load_config
from persona_gap.metrics.behavioral_factory import create_behavioral_extractor
from persona_gap.runner.logger import TrajectoryLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare behavioral extraction methods")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to experiment output directory (e.g. outputs/test1)",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Agent ID to analyze (default: first agent in trajectory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load config and trajectory
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    import json
    from persona_gap.core.models import ExperimentConfig
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config = ExperimentConfig.model_validate(config_data)
    
    traj_path = output_dir / "trajectory.jsonl"
    if not traj_path.exists():
        print(f"Error: trajectory.jsonl not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    records = TrajectoryLogger.load_trajectory(traj_path)
    
    # Get agent ID
    agent_ids = sorted(set(r.agent_id for r in records))
    if not agent_ids:
        print("Error: no agents found in trajectory", file=sys.stderr)
        sys.exit(1)
    
    agent_id = args.agent_id or agent_ids[0]
    if agent_id not in agent_ids:
        print(f"Error: agent {agent_id} not found. Available: {agent_ids}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Comparing Behavioral Extraction Methods")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Agent ID:         {agent_id}")
    print(f"Total steps:      {len([r for r in records if r.agent_id == agent_id])}")
    
    # Load action annotations for method 1
    annotations = load_action_annotations(config.env.action_annotations_path)

    # ==================================================================
    # Method 1: Annotation
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Method 1: Annotation (action labels)")
    print(f"{'='*70}")
    
    ext_ann = create_behavioral_extractor(
        method="annotation",
        action_annotations=annotations,
    )
    pv_ann = ext_ann.extract(records, agent_id=agent_id)
    
    print(f"\nRisk:        {pv_ann.risk:.4f}")
    print(f"Aggression:  {pv_ann.aggression:.4f}")
    print(f"Cooperation: {pv_ann.cooperation:.4f}")
    print(f"Deception:   {pv_ann.deception:.4f}")

    # ==================================================================
    # Method 2: Scorer
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Method 2: Scorer (state-aware)")
    print(f"{'='*70}")
    
    ext_scorer = create_behavioral_extractor(method="scorer")
    pv_scorer = ext_scorer.extract(records, agent_id=agent_id)
    
    print(f"\nRisk:        {pv_scorer.risk:.4f}")
    print(f"Aggression:  {pv_scorer.aggression:.4f}")
    print(f"Cooperation: {pv_scorer.cooperation:.4f}")
    print(f"Deception:   {pv_scorer.deception:.4f}")

    # ==================================================================
    # Method 3: LLM (optional, requires API key)
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Method 3: LLM Judge (optional)")
    print(f"{'='*70}")
    
    if config.llm.api_key:
        print("\nCalling LLM judge (this may take a moment)...")
        ext_llm = create_behavioral_extractor(
            method="llm",
            llm_config=config.llm,
        )
        pv_llm = ext_llm.extract(records, agent_id=agent_id)
        
        print(f"\nRisk:        {pv_llm.risk:.4f}")
        print(f"Aggression:  {pv_llm.aggression:.4f}")
        print(f"Cooperation: {pv_llm.cooperation:.4f}")
        print(f"Deception:   {pv_llm.deception:.4f}")
    else:
        print("\n[Skipped: No LLM API key configured]")
        print("To enable LLM method, set api_key in config or LLM_API_KEY environment variable")
        pv_llm = None

    # ==================================================================
    # Comparison Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Comparison Summary")
    print(f"{'='*70}\n")
    
    # Calculate differences
    diff_scorer = {
        "risk": pv_scorer.risk - pv_ann.risk,
        "aggression": pv_scorer.aggression - pv_ann.aggression,
        "cooperation": pv_scorer.cooperation - pv_ann.cooperation,
        "deception": pv_scorer.deception - pv_ann.deception,
    }
    
    print(f"{'Dimension':<15} {'Annotation':>12} {'Scorer':>12} {'Difference':>12}")
    print("-" * 70)
    for dim in ["risk", "aggression", "cooperation", "deception"]:
        ann_val = getattr(pv_ann, dim)
        scorer_val = getattr(pv_scorer, dim)
        diff = diff_scorer[dim]
        sign = "+" if diff >= 0 else ""
        print(f"{dim.capitalize():<15} {ann_val:>12.4f} {scorer_val:>12.4f} {sign}{diff:>11.4f}")
    
    if pv_llm:
        print("\n" + "-" * 70)
        print(f"{'Dimension':<15} {'Scorer':>12} {'LLM':>12} {'Difference':>12}")
        print("-" * 70)
        for dim in ["risk", "aggression", "cooperation", "deception"]:
            scorer_val = getattr(pv_scorer, dim)
            llm_val = getattr(pv_llm, dim)
            diff = llm_val - scorer_val
            sign = "+" if diff >= 0 else ""
            print(f"{dim.capitalize():<15} {scorer_val:>12.4f} {llm_val:>12.4f} {sign}{diff:>11.4f}")

    # Key insights
    print(f"\n{'='*70}")
    print("  Key Insights")
    print(f"{'='*70}\n")
    
    if diff_scorer["deception"] > 0.1:
        print("[*] Scorer detected deception (bluffing/trapping) that annotation missed")
    else:
        print("[*] Deception scores are similar across methods")
    
    if abs(diff_scorer["risk"]) > 0.1:
        print("[*] Risk assessment differs significantly (scorer considers hand strength)")
    
    if abs(diff_scorer["cooperation"]) > 0.1:
        print("[*] Cooperation/restraint differs (scorer adjusts for trapping behavior)")
    
    print("\nRecommendations:")
    print("- Use 'scorer' method for poker experiments (detects context-dependent behaviors)")
    print("- Use 'annotation' method for quick baselines or non-poker games")
    print("- Use 'llm' method for exploratory analysis or games without scorer support")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
