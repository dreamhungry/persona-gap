"""Tests for the RLCard adapter.

Note: These tests require the `rlcard` package to be installed.
They are skipped automatically if rlcard is not available.
"""

import pytest

try:
    import rlcard

    HAS_RLCARD = True
except ImportError:
    HAS_RLCARD = False

from pathlib import Path

pytestmark = pytest.mark.skipif(
    not HAS_RLCARD, reason="rlcard not installed"
)


@pytest.fixture
def leduc_annotations(tmp_path: Path) -> Path:
    """Create a temporary action annotations TOML file."""
    content = """
[actions.raise]
is_risky = true
is_aggressive = true

[actions.call]
is_cooperative = true

[actions.fold]

[actions.check]
"""
    path = tmp_path / "leduc.toml"
    path.write_text(content, encoding="utf-8")
    return path


class TestRLCardAdapter:
    def test_creation(self, leduc_annotations: Path):
        # Import triggers auto-registration
        from persona_gap.envs.rlcard_adapter import RLCardAdapter

        adapter = RLCardAdapter(
            game_name="leduc-holdem",
            seed=42,
            action_annotations_path=str(leduc_annotations),
        )
        assert adapter.num_agents == 2
        assert len(adapter.agent_ids) == 2

    def test_reset(self, leduc_annotations: Path):
        from persona_gap.envs.rlcard_adapter import RLCardAdapter

        adapter = RLCardAdapter(
            game_name="leduc-holdem",
            seed=42,
            action_annotations_path=str(leduc_annotations),
        )
        observations, current_player = adapter.reset()

        assert current_player in adapter.agent_ids
        assert isinstance(observations, dict)
        # Current player should have non-empty observation
        assert observations[current_player] != ""

    def test_step_and_game_loop(self, leduc_annotations: Path):
        from persona_gap.envs.rlcard_adapter import RLCardAdapter

        adapter = RLCardAdapter(
            game_name="leduc-holdem",
            seed=42,
            action_annotations_path=str(leduc_annotations),
        )
        observations, current_player = adapter.reset()
        done = False
        steps = 0

        while not done:
            legal = adapter.get_legal_actions(current_player)
            assert len(legal) > 0

            action = legal[0]  # Always pick first legal action
            observations, rewards, done, info = adapter.step(action)
            steps += 1

            if not done:
                current_player = info["current_player"]
                assert current_player in adapter.agent_ids

        # Game should eventually end
        assert done
        assert steps > 0
        # Rewards should be assigned
        assert any(r != 0.0 for r in rewards.values())

    def test_action_annotations(self, leduc_annotations: Path):
        from persona_gap.envs.rlcard_adapter import RLCardAdapter

        adapter = RLCardAdapter(
            game_name="leduc-holdem",
            seed=42,
            action_annotations_path=str(leduc_annotations),
        )
        adapter.reset()
        current = adapter.get_current_player()
        annotations = adapter.get_action_annotations(current)

        assert isinstance(annotations, dict)
        # Each legal action should have an annotation
        legal = adapter.get_legal_actions(current)
        for action in legal:
            assert action in annotations

    def test_registry_integration(self, leduc_annotations: Path):
        """Test that the adapter is properly registered and can be created via registry."""
        # Ensure rlcard_adapter module is imported (triggers registration)
        import persona_gap.envs.rlcard_adapter  # noqa: F401
        from persona_gap.envs.registry import create_env, list_envs

        assert "rlcard" in list_envs()

        env = create_env(
            "rlcard",
            game_name="leduc-holdem",
            seed=42,
            action_annotations_path=str(leduc_annotations),
        )
        assert env.num_agents == 2
