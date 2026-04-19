"""Memory module for LLM agents.

Stores episode summaries and provides a sliding-window view
to keep the LLM context within budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    """Simple episodic memory: stores text summaries of past episodes.

    Uses a sliding window so that only the most recent ``max_episodes``
    summaries are included when building the LLM prompt.
    """

    max_episodes: int = 10
    _summaries: list[str] = field(default_factory=list)

    def add(self, summary: str) -> None:
        """Store a new episode summary."""
        self._summaries.append(summary)

    def summarize(self) -> str:
        """Return the most recent summaries as a single text block.

        If there are no summaries yet, returns an empty string.
        """
        if not self._summaries:
            return ""
        recent = self._summaries[-self.max_episodes :]
        parts: list[str] = []
        for i, s in enumerate(recent, 1):
            parts.append(f"Episode {len(self._summaries) - len(recent) + i}: {s}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Wipe all stored summaries."""
        self._summaries.clear()

    @property
    def num_episodes(self) -> int:
        return len(self._summaries)

    def get_all(self) -> list[str]:
        """Return all stored summaries (for checkpointing)."""
        return list(self._summaries)

    def restore(self, summaries: list[str]) -> None:
        """Restore summaries from a checkpoint."""
        self._summaries = list(summaries)
