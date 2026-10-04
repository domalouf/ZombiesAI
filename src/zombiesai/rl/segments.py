"""What an actor ships and what the learner trains on: a `Segment` of decisions under one policy version, and
the frame history both sides index into the same way (PLAN.md, "Frames stored once, not stacked")."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    """Up to `segment_steps` decisions of one actor under one policy version.

    `frames` holds `context` frames before step 0 (for the stack to reach into), then one frame per step, then
    the observation after the last step (for the bootstrap value): `context + T + 1` frames in all. Step t's
    stack is frames[context + t - back] for each of the policy's offsets."""

    actor: int
    version: int
    context: int
    frames: np.ndarray  # (context + T + 1, H, W, 3) uint8
    actions: np.ndarray  # (T, heads) int64
    logp: np.ndarray  # (T,) behaviour log-prob of the action taken
    rewards: np.ndarray  # (T,) float32
    bad: np.ndarray  # (T,) bool
    terminated: bool
    audio: np.ndarray | None = None  # (T + 1, *feature) float32, one per observation
    audio_mask: np.ndarray | None = None  # (T + 1,) float32
    hud_view: np.ndarray | None = None  # (T + 1, 60, 80, 3) uint8, one per observation (demos/hud_crops.py)

    @property
    def n(self) -> int:
        return len(self.actions)


class FrameHistory:
    """The actor's view of the past: enough frames for the oldest offset, clamped at the last reset exactly as
    `demos.agent.BCAgent` and the training loader clamp (the first frame since reset stands in for anything
    older)."""

    def __init__(self, offsets: tuple[int, ...]):
        self.offsets = tuple(offsets)  # steps back, oldest first
        self.depth = max(self.offsets)
        self.frames: deque[np.ndarray] = deque(maxlen=self.depth + 1)

    def reset(self, frame: np.ndarray) -> None:
        self.frames.clear()
        self.push(frame)

    def push(self, frame: np.ndarray) -> None:
        self.frames.append(np.array(frame, dtype=np.uint8))

    def stack(self) -> np.ndarray:
        last = len(self.frames) - 1
        return np.stack([self.frames[max(last - back, 0)] for back in self.offsets])

    def context(self) -> list[np.ndarray]:
        """The `depth` frames before the newest, oldest first, padded by repeating the oldest held."""
        held = list(self.frames)[:-1]
        pad = [self.frames[0]] * (self.depth - len(held))
        return pad + held


def stack_indices(context: int, n: int, offsets: tuple[int, ...]) -> np.ndarray:
    """(n + 1, len(offsets)) frame indices of each step's stack, and of the final observation's."""
    t = np.arange(n + 1)[:, None]
    return context + t - np.asarray(offsets)[None, :]
