"""Did the player just get hit? Read off the policy's own frame: WaW floods the whole screen dark red.

A zombie's hit (and a grenade's, and the hit that downs) washes the picture red for about 0.4 s -- six or seven
decisions -- and then leaves it grey while health is low. Neither the HUD nor the points say so, which is why
`damage_event` paid nothing on the real game until this.

The cue is the red's *hue*, not how warm the picture is. Over the 61k frames of the five demos and every run's
films (2026-10-26), at the policy's 128x72:

* a hit scores 22-37 on `hit_score` (the red above both other channels, less how far green and blue differ),
  green and blue staying within a few levels of each other;
* Nacht's warm-lit help room scores as much red as a hit on a plain tint, but it is orange -- green 20-25 above
  blue -- and scores 7-10 here; blue fog and night sky score below zero;
* nothing that was not a hit scored above 17 (demos) or 16 (films).

So a hit is the score rising past `on`, and the detector re-arms once it falls below `off`: one flash is one
hit, and two hits close together are two as long as the red lets go between them (0.6 s apart was seen).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DamageConfig:
    on: float = 20.0
    off: float = 10.0


def hit_score(frame: np.ndarray) -> float:
    """How much an RGB frame looks like the hit flash: mean(R - max(G, B)) - |mean(G - B)|."""
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float((r - np.maximum(g, b)).mean() - abs((g - b).mean()))


class DamageDetector:
    """Feed it each step's frame; `step` is True on the step a hit's flash starts."""

    def __init__(self, config: DamageConfig | None = None):
        self.config = config or DamageConfig()
        self.reset()

    def reset(self) -> None:
        self.armed = True
        self.hits = 0

    def step(self, frame: np.ndarray) -> bool:
        score = hit_score(frame)
        if self.armed and score > self.config.on:
            self.armed = False
            self.hits += 1
            return True
        if not self.armed and score < self.config.off:
            self.armed = True
        return False
