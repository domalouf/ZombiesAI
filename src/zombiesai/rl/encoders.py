"""Convolutional trunk and pixel policy: the network that sees what a person sees.

Everything that learns from real footage shares this encoder -- behavioural cloning, the inverse dynamics
model, and (in M7) the RL run started from BC weights -- so a checkpoint trained on video loads into the
agent that plays the game without reshaping anything.

Layout notes that are decisions, not defaults:

* Frames arrive uint8 and are divided by 255 inside the module. Storing float32 stacks would cost four times
  the memory for no information, and the division is free on the GPU (PLAN.md, "Observation").
* A stack of decision frames is folded into the channel dimension, so 4 frames x RGB is 12 input channels.
* GroupNorm after each convolution: it costs nothing at this size and it is the plasticity insurance the
  high-replay-ratio regime later needs, which is cheaper to have from the start than to retrofit.
"""

import numpy as np
import torch
from torch import nn

from zombiesai import spec
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.networks import layer_init

FRAME_H, FRAME_W = spec.PIXELS_SHAPE[:2]


def stack_to_nchw(pixels: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) or (B, H, W, C) uint8 frames -> (B, T*C, H, W) in the layout convolutions want."""
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(1)
    if pixels.dim() != 5:
        raise ValueError(f"expected (B, T, H, W, C) frames, got {tuple(pixels.shape)}")
    b, t, h, w, c = pixels.shape
    return pixels.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)


class PixelEncoder(nn.Module):
    """Nature-CNN proportions, resized for 128x72 and 16:9."""

    def __init__(self, in_channels: int, out_dim: int = 512, width: int = 1, norm: bool = True):
        super().__init__()
        c1, c2, c3 = 32 * width, 64 * width, 64 * width
        layers: list[nn.Module] = []
        for in_c, out_c, kernel, stride, groups in (
            (in_channels, c1, 8, 4, 8),
            (c1, c2, 4, 2, 8),
            (c2, c3, 3, 1, 8),
        ):
            layers.append(layer_init(nn.Conv2d(in_c, out_c, kernel, stride)))
            if norm:
                layers.append(nn.GroupNorm(groups, out_c))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, FRAME_H, FRAME_W)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(flat, out_dim)), nn.ReLU(inplace=True))
        self.out_dim = out_dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = stack_to_nchw(pixels) if pixels.dim() > 4 or pixels.shape[-1] == 3 else pixels
        return self.fc(self.conv(x.float() / 255.0))


class PixelActorCritic(nn.Module):
    """Pixels (+ any vector observations) -> one categorical per action head, a value, and auxiliary heads.

    The auxiliary heads are the plan's cheapest representation win: forcing the trunk to predict "did I just
    score" and "am I being hit" makes it encode the two things a value function most needs, from data that
    has no rewards in it at all.
    """

    def __init__(
        self,
        nvec: tuple[int, ...] = spec.ACTION_NVEC,
        *,
        frame_stack: int = 4,
        vector_dim: int = 0,
        hidden: int = 512,
        width: int = 1,
        norm: bool = True,
        aux_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.nvec = tuple(int(n) for n in nvec)
        self.frame_stack = frame_stack
        self.vector_dim = vector_dim
        self.encoder = PixelEncoder(3 * frame_stack, out_dim=hidden, width=width, norm=norm)
        self.mixer = (
            nn.Sequential(layer_init(nn.Linear(hidden + vector_dim, hidden)), nn.ReLU(inplace=True))
            if vector_dim
            else nn.Identity()
        )
        self.actor = layer_init(nn.Linear(hidden, sum(self.nvec)), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.aux = nn.ModuleDict(
            {name: layer_init(nn.Linear(hidden, dim), std=0.01) for name, dim in (aux_heads or {}).items()}
        )

    def features(self, pixels: torch.Tensor, vector: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(pixels)
        if self.vector_dim:
            if vector is None:
                raise ValueError(f"this network was built with vector_dim={self.vector_dim} but got none")
            h = self.mixer(torch.cat([h, vector.float()], dim=-1))
        return h

    def forward(self, pixels: torch.Tensor, vector: torch.Tensor | None = None):
        h = self.features(pixels, vector)
        aux = {name: head(h) for name, head in self.aux.items()}
        return self.actor(h), self.critic(h).squeeze(-1), aux

    def dist(self, pixels: torch.Tensor, vector: torch.Tensor | None = None) -> FactoredCategorical:
        return FactoredCategorical(self.actor(self.features(pixels, vector)), self.nvec)

    def value(self, pixels: torch.Tensor, vector: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self.features(pixels, vector)).squeeze(-1)


def vector_dim(obs_keys: tuple[str, ...]) -> int:
    """Width of the non-pixel observations a policy is configured to read."""
    return sum(int(np.prod(spec.OBS_KEYS[k][0])) for k in obs_keys if k != "pixels")


def vector_from_obs(obs: dict, obs_keys: tuple[str, ...]) -> np.ndarray | None:
    """Concatenate the non-pixel observations, batched or not, in the order the network was built with."""
    keys = [k for k in obs_keys if k != "pixels"]
    if not keys:
        return None
    parts = []
    for key in keys:
        value = np.asarray(obs[key], dtype=np.float32)
        parts.append(value.reshape(-1) if value.ndim == 1 else value.reshape(len(value), -1))
    return np.concatenate(parts, axis=-1)
