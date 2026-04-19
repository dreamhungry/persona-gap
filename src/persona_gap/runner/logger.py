"""TrajectoryLogger — writes StepRecord data as JSONL for post-hoc analysis.

Each experiment run produces a single JSONL file where every line is a JSON-serialized
StepRecord. This makes it trivial to load with pandas:

    import pandas as pd
    df = pd.read_json("trajectory.jsonl", lines=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from persona_gap.core.models import EpisodeResult, StepRecord

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """Append-only JSONL logger for experiment trajectories."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trajectory_path = self.output_dir / "trajectory.jsonl"
        self._episodes_path = self.output_dir / "episodes.jsonl"

        # Open files in append mode
        self._traj_file = open(self._trajectory_path, "a", encoding="utf-8")
        self._ep_file = open(self._episodes_path, "a", encoding="utf-8")

        logger.info("TrajectoryLogger writing to %s", self.output_dir)

    def log_step(self, record: StepRecord) -> None:
        """Write a single step record."""
        line = record.model_dump_json()
        self._traj_file.write(line + "\n")
        self._traj_file.flush()

    def log_episode_end(self, result: EpisodeResult) -> None:
        """Write an episode summary record."""
        line = result.model_dump_json()
        self._ep_file.write(line + "\n")
        self._ep_file.flush()

    def close(self) -> None:
        """Flush and close log files."""
        self._traj_file.close()
        self._ep_file.close()

    # ------------------------------------------------------------------
    # Loading utilities (for metrics computation)
    # ------------------------------------------------------------------

    @staticmethod
    def load_trajectory(path: str | Path) -> list[StepRecord]:
        """Load all step records from a JSONL file."""
        records: list[StepRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(StepRecord.model_validate_json(line))
        return records

    @staticmethod
    def load_episodes(path: str | Path) -> list[EpisodeResult]:
        """Load all episode results from a JSONL file."""
        results: list[EpisodeResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(EpisodeResult.model_validate_json(line))
        return results
