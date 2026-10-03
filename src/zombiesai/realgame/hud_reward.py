"""The HUD tracker's settled events, turned into the reward shaper's `StepSignals`.

The sim hands `reward.RewardShaper` its signals directly; the real game only has the HUD, read by
`hud/parse.py` and made trustworthy by `hud/track.HudTracker`, which only ever reports a change the game's
rules allow. This is the last step: which of those changes pay, and as what.

* **Gains** are the tracker's settled `gain` events. Settled means the counter has stopped rolling, so one kill
  is one gain, not four partial ones; an `implausible` change pays nothing (and `hud_lost` ends the episode --
  that is the environment's call).
* **A round** is complete when the tracked round goes up. A new game resetting it to 1 is not a round.
* **Death** is the settled `downed` penalty: solo Nacht has no revive, so going down is the game over.
* **Purchases** are settled `spend` events, classified by price -- the HUD says what was paid, not what was
  bought, and the prompt reader that would say so does not exist yet. 1000 is a door or the debris (and also
  the double-barrel, which is counted as a door: both are progress); 200, 600 and 1200 are the Kar98k, the
  M1 Carbine and the Thompson on the wall. The shaper novelty-gates both, so re-buying pays nothing.
* **Repairs are the one thing the HUD cannot tell from a hit**: a plank and a hit both pay +10. The actor knows
  what it did, though. A small gain (at most `max_repair_points`) that settles within `repair_window_steps` of
  a `use` press, with the trigger untouched all that while, is a repair. That keeps the plan's board-farming
  defences -- the 0.2 weight, the per-round cap, the repair-share alarm -- working on the real game.
* **Kills** are counted, not paid for separately (their points already pay): a settled gain of at least
  `KILL_MIN_POINTS` that is not a repair holds a kill, since a hit is 10 and a kill 50-100 (hud/track.py). It is
  a lower bound -- two kills settling in one roll of the counter count once -- and double points' 20-point hits
  can sum past it. The HUD has no kill counter to do better with.

Not here yet: `damage_event` (no damage detector reads the real screen) and ammo rebuys (a rebuy costs half
the gun, which collides with other prices). Both stay zero rather than guessed.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from zombiesai import spec
from zombiesai.hud.track import Tracked
from zombiesai.reward import StepSignals

KILL_MIN_POINTS = 50
DOOR_PRICES = (1000,)
WALL_WEAPON_PRICES = (200, 600, 1200)
USE = spec.BUTTONS.index("use")


@dataclass(frozen=True)
class SignalConfig:
    # How long after a use press a small gain can still be a repair: the tracker settles ~3 steps after the
    # counter stops, and holding F rebuilds a plank about every half second.
    repair_window_steps: int = 20
    max_repair_points: int = 60
    doors_per_game: int = 3  # two doors and the debris


class HudSignals:
    """Feed it each step's `Tracked` and the action that led to it; get the step's `StepSignals`."""

    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.reset()

    def reset(self) -> None:
        self._recent: deque[tuple[bool, bool]] = deque(maxlen=self.config.repair_window_steps)
        self._round = -1
        self._doors = 0
        self.events: dict[str, int] = {"gain": 0, "repair": 0, "spend": 0, "implausible": 0, "rounds": 0,
                                       "kill": 0}
        self.points_gained = 0
        self.died = False

    def step(self, tracked: Tracked, action=None) -> StepSignals:
        if action is not None:
            a = np.asarray(action)
            self._recent.append((int(a[spec.BUTTON]) == USE, bool(a[spec.FIRE])))
        kind, delta = tracked.points_event, int(tracked.points_delta)
        gain = repair = 0
        doors: tuple[str, ...] = ()
        weapons: tuple[str, ...] = ()
        death = False
        if kind == "gain" and delta > 0:
            gain = delta
            self.points_gained += delta
            if self._looks_like_repair(delta):
                repair = delta
                self.events["repair"] += 1
            else:
                self.events["gain"] += 1
                if delta >= KILL_MIN_POINTS:
                    self.events["kill"] += 1
        elif kind == "spend" and delta < 0:
            self.events["spend"] += 1
            price = -delta
            if price in DOOR_PRICES and self._doors < self.config.doors_per_game:
                doors = (f"door{self._doors}",)
                self._doors += 1
            elif price in WALL_WEAPON_PRICES:
                weapons = (f"wall{price}",)
        elif kind in ("downed", "game_over"):
            death = not self.died
            self.died = True
        elif kind == "implausible":
            self.events["implausible"] += 1

        rounds = 0
        if tracked.round >= 0:
            if self._round >= 0 and tracked.round == self._round + 1:
                rounds = 1
                self.events["rounds"] += 1
            self._round = tracked.round
        return StepSignals(delta_points=float(gain), repair_points=float(repair), rounds_completed=rounds,
                           doors_opened=doors, wall_weapons_bought=weapons, death=death)

    def _looks_like_repair(self, delta: int) -> bool:
        if delta > self.config.max_repair_points or delta % 10 or not self._recent:
            return False
        used = any(use for use, _ in self._recent)
        fired = any(fire for _, fire in self._recent)
        return used and not fired

    @property
    def round(self) -> int:
        return self._round
