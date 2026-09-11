"""Actor-critic MLPs over flattened vector observations."""

import math

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from zombiesai.rl.distributions import FactoredCategorical


class ObsFlattener:
    """Turns Box or Dict observations (batched or not) into one float32 row per env, in a fixed key order."""

    def __init__(self, space: spaces.Space, keys: tuple[str, ...] | None = None):
        if isinstance(space, spaces.Dict):
            self.keys = keys or tuple(space.spaces)
            self.dim = sum(int(np.prod(space[k].shape)) for k in self.keys)
        else:
            self.keys = None
            self.dim = int(np.prod(space.shape))

    def __call__(self, obs, device: str | torch.device = "cpu") -> torch.Tensor:
        if self.keys is None:
            arr = np.asarray(obs, dtype=np.float32)
            arr = arr.reshape(len(arr), -1)
        else:
            arr = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(len(obs[k]), -1) for k in self.keys], 1)
        return torch.as_tensor(arr, device=device)

    def stack(self, observations: list):
        """Batch a list of single observations (e.g. vector-env final_obs) into the batched layout."""
        if self.keys is None:
            return np.stack(observations)
        return {k: np.stack([o[k] for o in observations]) for k in self.keys}


def action_nvec(space: spaces.Space) -> tuple[int, ...]:
    if isinstance(space, spaces.Discrete):
        return (int(space.n),)
    if isinstance(space, spaces.MultiDiscrete):
        return tuple(int(n) for n in space.nvec)
    raise TypeError(f"PPO here supports Discrete and MultiDiscrete actions, got {space}")


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_std: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
        d = h
    layers.append(layer_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs. The 0.01-scaled policy head starts near-uniform; the value head at 1."""

    def __init__(self, obs_dim: int, nvec: tuple[int, ...], hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.nvec = tuple(nvec)
        self.actor = mlp(obs_dim, hidden, sum(self.nvec), out_std=0.01)
        self.critic = mlp(obs_dim, hidden, 1, out_std=1.0)

    def dist(self, obs: torch.Tensor) -> FactoredCategorical:
        return FactoredCategorical(self.actor(obs), self.nvec)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
