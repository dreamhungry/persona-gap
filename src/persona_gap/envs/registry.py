"""Environment registry — maps adapter names to environment classes.

Usage:
    from persona_gap.envs.registry import register_env, create_env

    @register_env("rlcard")
    class RLCardAdapter:
        ...

    env = create_env("rlcard", game_name="leduc-holdem", seed=42,
                     action_annotations_path="configs/envs/leduc_holdem.toml")
"""

from __future__ import annotations

from typing import Any, Callable, Type

# Global registry: adapter_name -> class
_ENV_REGISTRY: dict[str, Type] = {}


def register_env(name: str) -> Callable[[Type], Type]:
    """Decorator to register an environment adapter class.

    Example:
        @register_env("rlcard")
        class RLCardAdapter:
            ...
    """

    def decorator(cls: Type) -> Type:
        if name in _ENV_REGISTRY:
            raise ValueError(
                f"Environment adapter '{name}' is already registered "
                f"(existing: {_ENV_REGISTRY[name].__name__}, "
                f"new: {cls.__name__})"
            )
        _ENV_REGISTRY[name] = cls
        return cls

    return decorator


def create_env(adapter: str, **kwargs: Any) -> Any:
    """Create an environment instance by adapter name.

    Args:
        adapter: Registered adapter name (e.g. "rlcard").
        **kwargs: Passed to the adapter's __init__.

    Returns:
        An instance satisfying the BaseEnv protocol.

    Raises:
        KeyError: If the adapter name is not registered.
    """
    if adapter not in _ENV_REGISTRY:
        available = ", ".join(sorted(_ENV_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown environment adapter '{adapter}'. "
            f"Available adapters: {available}"
        )
    cls = _ENV_REGISTRY[adapter]
    return cls(**kwargs)


def list_envs() -> list[str]:
    """Return a sorted list of all registered adapter names."""
    return sorted(_ENV_REGISTRY.keys())
