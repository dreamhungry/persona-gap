"""Batch experiment entry point — sweep over personality × memory × communication.

Usage:
    python scripts/run_batch.py configs/batch_example.toml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a batch of Persona-Gap experiments"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the TOML batch configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    from persona_gap.core.config import load_batch_config, load_personality_preset
    from persona_gap.core.models import ExperimentConfig
    from persona_gap.runner.experiment import ExperimentRunner

    batch = load_batch_config(config_path)

    # Build all combinations
    personality_files = batch.sweep_personalities or [""]
    memory_opts = batch.sweep_memory
    comm_opts = batch.sweep_communication

    combinations = list(
        itertools.product(personality_files, memory_opts, comm_opts)
    )
    logger.info(
        "Batch experiment",
        total_runs=len(combinations),
        personalities=len(personality_files),
        memory_options=memory_opts,
        communication_options=comm_opts,
    )

    for i, (personality_path, memory, communication) in enumerate(combinations):
        # Deep copy the base config
        config_dict = batch.base.model_dump()

        # Override personality for agent_0 (if preset provided)
        if personality_path:
            preset = load_personality_preset(personality_path)
            if config_dict["agents"]:
                config_dict["agents"][0]["personality"] = preset.model_dump()

        # Override memory and communication for all agents
        for agent in config_dict["agents"]:
            agent["memory_enabled"] = memory
            agent["communication_enabled"] = communication

        # Unique output directory
        personality_name = (
            Path(personality_path).stem if personality_path else "default"
        )
        run_name = f"{personality_name}_mem{memory}_comm{communication}"
        config_dict["output_dir"] = str(
            Path(batch.base.output_dir) / run_name
        )

        config = ExperimentConfig.model_validate(config_dict)

        logger.info(
            "Starting run",
            run=i + 1,
            total=len(combinations),
            name=run_name,
        )

        runner = ExperimentRunner(config)
        results = runner.run(resume=True)

        logger.info(
            "Run complete",
            run=i + 1,
            name=run_name,
            episodes=len(results),
        )

    logger.info("Batch experiment complete", total_runs=len(combinations))


if __name__ == "__main__":
    main()
