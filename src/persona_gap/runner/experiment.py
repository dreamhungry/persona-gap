"""ExperimentRunner — orchestrates multi-agent, multi-episode experiments.

The runner:
1. Loads config and creates the environment + agents.
2. Runs N episodes of the game, each consisting of multiple steps.
3. At each step: optionally has agents communicate, then act.
4. Logs every step as a StepRecord, and episode summaries.
5. Manages checkpointing for resume-from-failure.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import structlog

from persona_gap.agents.llm_agent import LLMAgent
from persona_gap.core.config import load_config
from persona_gap.core.models import (
    AgentConfig,
    EpisodeResult,
    ExperimentConfig,
    StepRecord,
)
from persona_gap.envs.registry import create_env
from persona_gap.llm.backend import LLMBackend
from persona_gap.runner.checkpoint import CheckpointManager
from persona_gap.runner.logger import TrajectoryLogger

logger = structlog.get_logger(__name__)


class ExperimentRunner:
    """Runs a full persona-gap experiment from a configuration."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        random.seed(config.seed)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)

        # Create environment
        self.env = create_env(
            adapter=config.env.adapter,
            game_name=config.env.game_name,
            seed=config.seed,
            action_annotations_path=config.env.action_annotations_path,
        )

        # Create agents with shared or per-agent LLM backends
        self.agents: dict[str, LLMAgent] = {}
        for agent_cfg in config.agents:
            backend = LLMBackend.from_llm_config(
                config.llm,
                model_override=agent_cfg.model,
                temperature_override=agent_cfg.temperature,
                max_tokens_override=agent_cfg.max_tokens,
            )
            self.agents[agent_cfg.agent_id] = LLMAgent(agent_cfg, backend)

        # Logger and checkpoint
        self.traj_logger = TrajectoryLogger(self.output_dir)
        self.ckpt_manager = CheckpointManager(self.output_dir)

    def run(self, resume: bool = True) -> list[EpisodeResult]:
        """Run the full experiment.

        Args:
            resume: If True and a checkpoint exists, resume from it.

        Returns:
            List of EpisodeResult for all episodes.
        """
        start_episode = 0
        all_results: list[EpisodeResult] = []

        # Resume from checkpoint if available
        if resume:
            ckpt = self.ckpt_manager.load()
            if ckpt is not None:
                start_episode = ckpt["last_completed_episode"] + 1
                # Restore agent memories
                for aid, summaries in ckpt.get("agent_memories", {}).items():
                    if aid in self.agents:
                        self.agents[aid].memory.restore(summaries)
                logger.info(
                    "Resuming experiment",
                    start_episode=start_episode,
                    total_episodes=self.config.num_episodes,
                )

        for episode_id in range(start_episode, self.config.num_episodes):
            logger.info("Starting episode", episode_id=episode_id)
            t0 = time.monotonic()

            result = self._run_episode(episode_id)
            all_results.append(result)

            elapsed = time.monotonic() - t0
            logger.info(
                "Episode complete",
                episode_id=episode_id,
                rewards=result.rewards,
                steps=result.num_steps,
                elapsed_seconds=round(elapsed, 2),
            )

            # Checkpoint after each episode
            agent_memories = {
                aid: agent.memory.get_all()
                for aid, agent in self.agents.items()
            }
            self.ckpt_manager.save(
                episode_id=episode_id,
                agent_memories=agent_memories,
            )

        # Print usage summary
        self._print_usage_summary()

        self.traj_logger.close()
        return all_results

    def _run_episode(self, episode_id: int) -> EpisodeResult:
        """Run a single episode of the game."""
        observations, current_player = self.env.reset()
        done = False
        step_count = 0
        cumulative_rewards: dict[str, float] = {aid: 0.0 for aid in self.agents}
        agent_actions: dict[str, list[str]] = {aid: [] for aid in self.agents}

        while not done:
            agent = self.agents.get(current_player)
            if agent is None:
                logger.error(
                    "No agent for player", player=current_player
                )
                break

            obs = observations.get(current_player, "")
            legal_actions = self.env.get_legal_actions(current_player)

            # --- Communication phase (optional) ---
            message = ""
            if agent.config.communication_enabled:
                message, _comm_reasoning = agent.communicate(obs)

            # --- Action phase ---
            action, reasoning = agent.act(obs, legal_actions)
            agent_actions[current_player].append(action)

            is_fallback = "[FALLBACK]" in reasoning

            # Step the environment
            observations, rewards, done, info = self.env.step(action)

            # Accumulate rewards
            for aid, r in rewards.items():
                cumulative_rewards[aid] = cumulative_rewards.get(aid, 0.0) + r

            # Log step
            record = StepRecord(
                episode_id=episode_id,
                step=step_count,
                agent_id=current_player,
                observation=obs,
                legal_actions=legal_actions,
                action=action,
                reasoning=reasoning,
                message=message if message else None,
                reward=rewards.get(current_player, 0.0),
                is_fallback=is_fallback,
            )
            self.traj_logger.log_step(record)

            step_count += 1

            # Update current player for next turn
            if not done:
                current_player = info.get(
                    "current_player", self.env.get_current_player()
                )

        # Episode result
        winner = info.get("winner") if done else None
        result = EpisodeResult(
            episode_id=episode_id,
            rewards=cumulative_rewards,
            num_steps=step_count,
            winner=winner,
        )
        self.traj_logger.log_episode_end(result)

        # Update agent memories
        for aid, agent in self.agents.items():
            agent.update_memory(
                episode_id=episode_id,
                total_reward=cumulative_rewards.get(aid, 0.0),
                key_actions=agent_actions.get(aid, []),
            )

        return result

    def _print_usage_summary(self) -> None:
        """Print token usage summary for all agents."""
        logger.info("=" * 50)
        logger.info("LLM Usage Summary")
        logger.info("=" * 50)
        total_tokens = 0
        for aid, agent in self.agents.items():
            stats = agent.backend.usage.summary()
            total_tokens += stats["total_tokens"]
            logger.info(
                "Agent usage",
                agent_id=aid,
                **stats,
            )
        logger.info("Total tokens across all agents", total_tokens=total_tokens)
