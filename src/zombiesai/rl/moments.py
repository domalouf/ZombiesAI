"""The agent's firsts, kept as clips: the first kill, door, box, headshot, Ray Gun, round 5, ... of its training.

Each actor watches its game for moments (`MomentSpotter`, on each step's `info`): the events the HUD shows, read
the way the reward reads them (realgame/hud_reward.py). A moment is the first of its kind in that game; whether
it is the first of the agent's training is decided when the game is over, against the moments kept so far
(`MomentBook`), because that is when the game's film is whole: the clip is cut out of it (`cut_clip`) -- the same
film the run's best game is kept from (rl/best_episode.py), with the game's sound when it has a sink of its own.
So a moment costs nothing while the game plays, and a game cut short by the actor stopping keeps none of its
moments: the next game that has one takes its place.

What can be seen, and how sure it is:

* **kill, repair, door, box, wall_buy** -- settled HUD events: a non-repair gain of at least 50 points; a repair
  (hud_reward's rule); a spend of 1000 (a door or the debris; the double-barrel costs the same), 950 (the
  mystery box) and a wall weapon's price.
* **doors_2, doors_3** -- the first game to open two, then all three, of Nacht's doors and debris.
* **headshot** -- the HUD has no headshot counter, so it is a guess checked afterwards: a settled gain of
  exactly a headshot kill's points while the trigger was pulled, kept only when the game's own scoreboard says
  the game had at least one headshot. Two body kills settling together can still pass for one.
* **knife_kill** -- a settled gain of a knife kill's 130 points just after a melee press. Not checked: the
  scoreboard does not count knife kills.
* **weapon_<key>** -- the first time the HUD names a gun other than the Colt as the one held: a wall buy or a
  box gun, the Ray Gun among them.
* **round_<n>** -- the first game to reach round n, for the rounds in `ROUND_MILESTONES`.

The kept moments are the `firsts/` shelf of the video directory (rl/keepsakes.py): `firsts.json` (kind ->
record), and a `<kind>.mp4` with its `<kind>.brain.jsonl` each. The directory is shared by every run on this PC
(`runs/video`), so a run continued from another does not film its firsts again. Two actors with the same first
are settled by when they saw it: the earlier one is kept. `scripts/moments.py` lists them in order.

The spotter also keeps the game's longest stretch without a kill, for the records shelf.
"""

import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.realgame.hud_reward import DOOR_PRICES, KILL_MIN_POINTS, WALL_WEAPON_PRICES
from zombiesai.rl.keepsakes import Shelf

BOX_PRICE = 950
HEADSHOT_POINTS = (100, 110)  # a headshot kill; with the killing hit's 10 if the counter settles on both
KNIFE_POINTS = 130
START_WEAPON = "colt"
ROUND_MILESTONES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50)
RECENT_STEPS = 20  # how far back a gain can look for the trigger or the melee that made it (~1.3 s)

FIRE = spec.FIRE
MELEE = spec.BUTTONS.index("melee")


@dataclass(frozen=True)
class Kind:
    title: str
    before_s: float = 8.0  # how much of the film before the moment the clip keeps
    after_s: float = 5.0  # and after it


KINDS = {
    "kill": Kind("First kill"),
    "headshot": Kind("First headshot"),
    "knife_kill": Kind("First knife kill"),
    "repair": Kind("First barricade repair", after_s=4.0),
    "door": Kind("First door opened", after_s=8.0),
    "doors_2": Kind("Two doors opened in one game", after_s=8.0),
    "doors_3": Kind("Every door opened in one game", after_s=8.0),
    "wall_buy": Kind("First wall weapon bought", after_s=6.0),
    # the box spins ~4 s before it offers a gun, and the gun is only taken with another press
    "box": Kind("First mystery box", after_s=12.0),
}


def kind(name: str) -> Kind:
    """A moment kind's title and clip window, the rounds' and weapons' made up from their names."""
    if name in KINDS:
        return KINDS[name]
    if name.startswith("round_"):
        return Kind(f"First time reaching round {name[6:]}", before_s=10.0, after_s=6.0)
    if name.startswith("weapon_"):
        from zombiesai.hud.weapons import WEAPONS

        key = name[7:]
        gun = next((w.name for w in WEAPONS if w.key == key), key)
        return Kind(f"First time holding the {gun}", before_s=8.0, after_s=6.0)
    return Kind(name.replace("_", " "))


class MomentSpotter:
    """One game's moments: feed it each step's `info` and the action that led to it; it returns the moments that
    step holds, each the first of its kind in this game."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.seen: set[str] = set()
        self._recent: deque[tuple[bool, bool]] = deque(maxlen=RECENT_STEPS)
        self._doors = 0
        self._round = -1
        self.steps = 0
        self._last_kill = 0  # the step of the last kill (the game's start before the first)
        self._drought = 0  # the longest stretch between kills so far, in steps

    def step(self, info: dict, action=None) -> list[dict]:
        self.steps += 1
        if action is not None:
            a = np.asarray(action)
            self._recent.append((bool(a[FIRE]), int(a[spec.BUTTON]) == MELEE))
        found: list[str] = []
        event, delta = info.get("points_event") or "", int(info.get("points_delta") or 0)
        if event == "gain" and delta > 0:
            if info.get("repair"):
                found.append("repair")
            elif delta >= KILL_MIN_POINTS:
                found.append("kill")
                self._drought = max(self._drought, self.steps - self._last_kill)
                self._last_kill = self.steps
                fired = any(f for f, _ in self._recent)
                meleed = any(m for _, m in self._recent)
                if delta in HEADSHOT_POINTS and fired and not meleed:
                    found.append("headshot")
                if delta == KNIFE_POINTS and meleed:
                    found.append("knife_kill")
        elif event == "spend" and delta < 0:
            price = -delta
            if price in DOOR_PRICES:
                self._doors += 1
                found.append({1: "door", 2: "doors_2", 3: "doors_3"}.get(self._doors, ""))
            elif price == BOX_PRICE:
                found.append("box")
            elif price in WALL_WEAPON_PRICES:
                found.append("wall_buy")
        weapon = info.get("weapon")
        if weapon and weapon != START_WEAPON:
            found.append(f"weapon_{weapon}")
        r = info.get("round")
        if r is not None and r >= 0:
            if self._round >= 0 and r > self._round and r in ROUND_MILESTONES:
                found.append(f"round_{r}")
            self._round = r

        moments = []
        for name in found:
            if name and name not in self.seen:
                self.seen.add(name)
                moments.append({"kind": name, "title": kind(name).title, "t_unix": time.time(),
                                "round": info.get("round"), "points": info.get("points")})
        return moments

    def longest_without_kill_s(self) -> float:
        """The game's longest stretch without a kill, the one it is in now included."""
        return max(self._drought, self.steps - self._last_kill) / spec.DECISION_HZ


def counts(moment: dict, summary: dict) -> bool:
    """Whether the finished game bears the moment out: a guessed headshot needs the scoreboard's headshots."""
    if moment["kind"] == "headshot":
        return (summary.get("end_headshots") or 0) >= 1
    return True


class MomentBook(Shelf):
    """The firsts kept so far: a shelf keyed by kind, where the earlier of two firsts wins."""

    def __init__(self, directory: Path):
        super().__init__(directory, "firsts")

    def better(self, new: dict, old: dict) -> bool:
        return new["t_unix"] < old.get("t_unix", float("inf"))

    def wanted(self, moments: list[dict], summary: dict) -> list[dict]:
        """The game's moments worth cutting a clip for: firsts the book has not got, or got later than this."""
        kept = self.read()
        return [m for m in moments if counts(m, summary) and self.wants(m["kind"], m, kept)]

    def offer_clip(self, moment: dict, files: dict[str, Path]) -> bool:
        """Keep the clip (and its sidecar) if the moment is still the first of its kind, else delete them."""
        return self.offer(moment["kind"], moment, files, move=True)


def cut_clip(film: Path, out: Path, start_s: float, seconds: float, encoder: tuple[str, ...]) -> None:
    """`seconds` of `film` from `start_s`, re-encoded so it starts on that frame rather than the keyframe
    before, with the film's sound if it has any."""
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(film),
         "-t", f"{seconds:.3f}", *encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(out)],
        capture_output=True, timeout=300)
    if done.returncode != 0:
        Path(out).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited {done.returncode}: {done.stderr.decode(errors='replace')[-300:]}")
