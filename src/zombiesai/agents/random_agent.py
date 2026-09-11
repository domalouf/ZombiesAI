"""Uniform random policy over the factored action space: the baseline that can't be blamed for harness bugs."""

import numpy as np

from zombiesai import spec


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._nvec = np.array(spec.ACTION_NVEC)

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.rng.integers(0, self._nvec)
