"""Layer initialisation shared by the networks here (rl/encoders.py)."""

import math

from torch import nn


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weights at gain `std`, constant bias: the PPO convention, so a 0.01-scaled policy head starts
    near-uniform and a value head at 1 starts near zero."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer
