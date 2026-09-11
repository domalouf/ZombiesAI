"""Adapters over the factored canonical action space."""

import gymnasium as gym

from zombiesai import spec


class CompactActionWrapper(gym.ActionWrapper):
    """Exposes Discrete(N_COMPACT) for value methods that need max_a Q(s, a) over a joint action."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spec.compact_action_space()

    def action(self, action):
        return spec.expand_compact(int(action))
