"""Did the player just get hit? Read off the policy's own frame: WaW tints the whole screen red at once.

A zombie's hit lays a red tint over the entire picture -- every part of it, at the same moment. How strong and
how long depends: a hard hit floods the screen dark red for ~0.4 s, the hit that downs flashes red for a step,
and a first hit at full health only warms the picture for a step or two and jolts the view. That last one is the
one that matters most (it is the warning) and it is faint: in a blue-lit corner of the spawn room, live on the
fleet (2026-10-26), it took the frame from 9 below neutral to 9 above, never as red as a warm room is. So a hit
is not "the frame is red" but "the frame just got redder, everywhere at once, and only redder":

* **everywhere at once** -- the frame is cut into a 3x4 grid of cells, and the *smallest* rise in red (R above
  both G and B) over the last one or two steps is what counts. A turn, a zombie walking in front, a muzzle
  flash move some cells; only an overlay moves all of them.
* **only redder** -- green and blue have to move together (`gb_shift`). A power-up's green glow fading out
  raises "red above green" everywhere just as a hit does (rise up to 27), but it does it by dropping green
  against blue (shift 26-53); a hit shifts them by at most ~9.

Over the five demos' 61k frames (2026-10-26) every hit seen by eye rose 14.6-42 with a shift under 9; nothing
else with a shift under 12 rose more than 7.

The tint is a blend toward red, so on a scene that is already warm it raises red less (8, on a brown wall in
rl13's best game). That is the case the plain level catches: `hit_score` (red above both other channels, less
how far green and blue differ) rising past `on`. Nacht's orange-lit rooms score 7-10 on it (green 20-25 above
blue), blue fog below zero, and across the demos and every run's film nothing that was not a hit scored above
17; hits 22-37. It re-arms below `off`.

Either one fires; then the detector waits `refractory_steps` before it can fire again, so a flash coming up
over two steps is one hit, and two hits half a second apart are two.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

GRID = (3, 4)  # rows, columns of cells over the 72x128 frame


@dataclass(frozen=True)
class DamageConfig:
    min_rise: float = 11.0
    max_gb_shift: float = 15.0
    lags: tuple[int, ...] = (1, 2)  # a flash can take two steps to come up
    refractory_steps: int = 3
    on: float = 20.0
    off: float = 10.0


def hit_score(frame: np.ndarray) -> float:
    """How red the whole frame is: mean(R - max(G, B)) - |mean(G - B)|."""
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float((r - np.maximum(g, b)).mean() - abs((g - b).mean()))


def frame_cells(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """(per-cell mean of R - max(G, B), the frame's mean G - B) of one RGB frame."""
    f = frame.astype(np.float32)
    red = f[..., 0] - np.maximum(f[..., 1], f[..., 2])
    rows, cols = GRID
    h, w = red.shape
    cells = red[: h - h % rows, : w - w % cols].reshape(rows, h // rows, cols, w // cols).mean((1, 3))
    return cells, float((f[..., 1] - f[..., 2]).mean())


class DamageDetector:
    """Feed it each step's frame; `step` is True on the step a hit's tint comes up."""

    def __init__(self, config: DamageConfig | None = None):
        self.config = config or DamageConfig()
        self.reset()

    def reset(self) -> None:
        self._history: deque[tuple[np.ndarray, float]] = deque(maxlen=max(self.config.lags))
        self._quiet = 0  # steps left before another hit can fire
        self._armed = True  # the level has been below `off` since the last hit
        self.hits = 0
        self.last_rise = self.last_score = 0.0  # this step's numbers, for logging

    def step(self, frame: np.ndarray) -> bool:
        c = self.config
        cells, gb = frame_cells(frame)
        rise = -np.inf
        for lag in c.lags:
            if lag <= len(self._history):
                before, gb_before = self._history[-lag]
                if abs(gb - gb_before) < c.max_gb_shift:
                    rise = max(rise, float((cells - before).min()))
        self._history.append((cells, gb))
        score = hit_score(frame)
        self.last_rise, self.last_score = rise, score
        hit = not self._quiet and (rise > c.min_rise or (self._armed and score > c.on))
        self._quiet = c.refractory_steps if hit else max(0, self._quiet - 1)
        if hit or score > c.on:
            self._armed = False
        elif score < c.off:
            self._armed = True
        self.hits += hit
        return hit
