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
* Training runs it in mixed precision where the GPU has the tensor cores for it (`Precision`): bf16 on Ampere
  and later (the learner PC's RTX 5070), fp16 with a gradient scaler on Turing (the 2080 Ti, whose bf16 is
  emulated and slower than fp32), fp32 on the CPU. Losses and the action distribution stay in fp32.
"""

from dataclasses import dataclass

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
        # Straight from uint8 to the dtype the first convolution will run in: under autocast a float32 copy of
        # the stack would only be made to be cast again, and it is the largest tensor in the forward pass.
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else torch.float32
        return self.fc(self.conv(x.to(dtype).div_(255.0)))


class AudioEncoder(nn.Module):
    """A stereo log-mel window (B, 2, frames, mels) -> (B, out_dim). Left and right are the two input
    channels, so the first convolution can take their difference -- the only direction cue there is."""

    def __init__(self, shape: tuple[int, int, int], out_dim: int = 128, norm: bool = True):
        super().__init__()
        channels = shape[0]
        layers: list[nn.Module] = []
        for in_c, out_c, stride in ((channels, 16, (1, 2)), (16, 32, (2, 2)), (32, 32, (2, 2))):
            layers.append(layer_init(nn.Conv2d(in_c, out_c, 3, stride, padding=1)))
            if norm:
                layers.append(nn.GroupNorm(4, out_c))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, *shape)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(flat, out_dim)), nn.ReLU(inplace=True))
        self.shape, self.out_dim = tuple(shape), out_dim

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(audio.float()))


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
        audio_shape: tuple[int, int, int] | None = None,
        audio_dim: int = 128,
    ):
        super().__init__()
        self.nvec = tuple(int(n) for n in nvec)
        self.frame_stack = frame_stack
        self.vector_dim = vector_dim
        self.encoder = PixelEncoder(3 * frame_stack, out_dim=hidden, width=width, norm=norm)
        # Optional hearing: an audio embedding joins the mixer, gated by a has-audio mask so a clip recorded
        # without sound (or a dead live stream) contributes exactly nothing rather than a silence it never had.
        # Without it the module list is unchanged, so every checkpoint from before audio loads as it was.
        self.audio_encoder = AudioEncoder(audio_shape, audio_dim, norm=norm) if audio_shape else None
        fused = vector_dim + (audio_dim + 1 if audio_shape else 0)
        self.mixer = (
            nn.Sequential(layer_init(nn.Linear(hidden + fused, hidden)), nn.ReLU(inplace=True))
            if fused
            else nn.Identity()
        )
        self.actor = layer_init(nn.Linear(hidden, sum(self.nvec)), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.aux = nn.ModuleDict(
            {name: layer_init(nn.Linear(hidden, dim), std=0.01) for name, dim in (aux_heads or {}).items()}
        )

    def features(
        self,
        pixels: torch.Tensor,
        vector: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(pixels)
        parts = [h]
        if self.vector_dim:
            if vector is None:
                raise ValueError(f"this network was built with vector_dim={self.vector_dim} but got none")
            parts.append(vector.float())
        if self.audio_encoder is not None:
            if audio is None:
                raise ValueError("this network hears: pass audio (and audio_mask=0 where there is none)")
            mask = torch.ones(len(h), device=h.device) if audio_mask is None else audio_mask.float()
            parts += [self.audio_encoder(audio) * mask[:, None], mask[:, None]]
        return self.mixer(torch.cat(parts, dim=-1)) if len(parts) > 1 else h

    def forward(
        self,
        pixels: torch.Tensor,
        vector: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ):
        h = self.features(pixels, vector, audio, audio_mask)
        aux = {name: head(h) for name, head in self.aux.items()}
        return self.actor(h), self.critic(h).squeeze(-1), aux

    def dist(self, pixels: torch.Tensor, vector: torch.Tensor | None = None) -> FactoredCategorical:
        return FactoredCategorical(self.actor(self.features(pixels, vector)), self.nvec)

    def value(self, pixels: torch.Tensor, vector: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self.features(pixels, vector)).squeeze(-1)


@dataclass(frozen=True)
class Precision:
    """How a training loop runs the network: `mode` is "bf16", "fp16" or "off" (fp32).

    Both halves are always used, so the fp32 path is the same code: a disabled autocast is a no-op, and a
    disabled GradScaler scales by nothing and steps the optimizer as `opt.step()` would."""

    mode: str
    device_type: str

    def autocast(self):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.mode, torch.float32)
        return torch.autocast(self.device_type, dtype=dtype, enabled=self.mode != "off")

    def grad_scaler(self) -> torch.amp.GradScaler:
        # fp16 has 5 exponent bits: small gradients underflow to zero unless the loss is scaled up first.
        return torch.amp.GradScaler(self.device_type, enabled=self.mode == "fp16")


def precision(amp: str, device: torch.device | str) -> Precision:
    """`amp` is "auto", "off", "bf16" or "fp16". "auto" is bf16 where the GPU computes it natively, else fp16;
    the CPU always trains in fp32. (`torch.cuda.is_bf16_supported()` says yes on Turing too, where bf16 is
    emulated -- hence `including_emulation=False`.)"""
    device = torch.device(device)
    if amp not in ("auto", "off", "bf16", "fp16"):
        raise ValueError(f"amp must be auto, off, bf16 or fp16, got {amp!r}")
    if device.type != "cuda" or amp == "off":
        return Precision("off", device.type)
    if amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    return Precision(amp, device.type)


# Observations with an encoder of their own rather than a place in the flat vector.
_ENCODED_KEYS = ("pixels", "audio")


def vector_dim(obs_keys: tuple[str, ...]) -> int:
    """Width of the non-pixel observations a policy is configured to read."""
    return sum(int(np.prod(spec.OBS_KEYS[k][0])) for k in obs_keys if k not in _ENCODED_KEYS)


def vector_from_obs(obs: dict, obs_keys: tuple[str, ...]) -> np.ndarray | None:
    """Concatenate the non-pixel observations, batched or not, in the order the network was built with."""
    keys = [k for k in obs_keys if k not in _ENCODED_KEYS]
    if not keys:
        return None
    parts = []
    for key in keys:
        value = np.asarray(obs[key], dtype=np.float32)
        parts.append(value.reshape(-1) if value.ndim == 1 else value.reshape(len(value), -1))
    return np.concatenate(parts, axis=-1)
