"""CheckpointManager — episode-granularity save/restore for experiment resumption.

Saves the current episode index and each agent's memory state to a JSON file,
allowing experiments to resume from the last completed episode after interruption.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages experiment checkpoints for resume-from-failure."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = self.output_dir / "checkpoint.json"

    def save(
        self,
        episode_id: int,
        agent_memories: dict[str, list[str]],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save a checkpoint after completing an episode.

        Args:
            episode_id: The last completed episode index.
            agent_memories: {agent_id: list of memory summaries}.
            extra: Any additional metadata to persist.
        """
        data = {
            "last_completed_episode": episode_id,
            "agent_memories": agent_memories,
        }
        if extra:
            data["extra"] = extra

        with open(self._ckpt_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("Checkpoint saved: episode %d", episode_id)

    def load(self) -> dict[str, Any] | None:
        """Load the latest checkpoint, or return None if none exists."""
        if not self._ckpt_path.exists():
            return None
        with open(self._ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Checkpoint loaded: resuming from episode %d",
            data.get("last_completed_episode", -1),
        )
        return data

    def exists(self) -> bool:
        return self._ckpt_path.exists()
