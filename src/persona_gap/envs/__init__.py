"""Environment adapters with unified Gym-like interface."""

# Import adapters so that @register_env decorators execute at import time.
from persona_gap.envs import rlcard_adapter as _rlcard_adapter  # noqa: F401
