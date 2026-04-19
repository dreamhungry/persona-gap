"""Single experiment entry point.

Usage:
    python scripts/run_experiment.py configs/experiment.toml
    python scripts/run_experiment.py configs/experiment.toml --no-resume
    python scripts/run_experiment.py configs/experiment.toml --no-analysis
    python scripts/run_experiment.py configs/experiment.toml --no-expressed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single Persona-Gap experiment"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the TOML experiment configuration file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore existing checkpoints)",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip post-experiment metric analysis",
    )
    parser.add_argument(
        "--no-expressed",
        action="store_true",
        help="Skip expressed personality extraction in analysis (saves LLM calls)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Import here to allow structlog to be configured first
    from persona_gap.core.config import load_config
    from persona_gap.runner.experiment import ExperimentRunner

    config = load_config(config_path)
    runner = ExperimentRunner(config)
    results = runner.run(resume=not args.no_resume)

    print(f"\nExperiment complete: {len(results)} episodes")
    for r in results[-5:]:  # Show last 5
        print(f"  Episode {r.episode_id}: rewards={r.rewards}, winner={r.winner}")

    # ---- Post-experiment analysis ----
    if not args.no_analysis and results:
        print("\n--- Running post-experiment analysis ---\n")
        analysis_cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_analysis.py"),
            config.output_dir,
            "--config",
            str(config_path),
        ]
        if args.no_expressed:
            analysis_cmd.append("--no-expressed")
        subprocess.run(analysis_cmd, check=False)


if __name__ == "__main__":
    main()
