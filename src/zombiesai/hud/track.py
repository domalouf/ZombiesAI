"""Temporal consistency: which per-step reads agree with how the game can actually change its HUD.

The per-frame reader (`hud/parse.py`) can be confidently wrong; the game's rules are what catch it
(PLAN.md: "the sanity filters are where reliability actually comes from"). What the recordings show:

- **Points roll.** WaW does not jump the counter to its new value: it counts there over a few frames
  (2070 -> 2139 -> 2164 -> 2170). So single-step deltas are arbitrary; what the rules constrain is the
  change between two *settled* values (the same read on `SETTLE` ok steps in a row), and the reads
  between them must run monotonically from one to the other.
- **Settled deltas** are multiples of 10 and one of: a gain (hits 10, kills 50-100, and sums of those
  while the counter rolls; double points doubles them); a spend no larger than what was there (wall
  weapons 200-1500, box 950, doors 1000, ...); the downed penalty, 5% of the points rounded to 10 (solo
  Nacht has no revive, so a down is the death); the game-over count-up to the game's total score; and
  the reset to 500 when a new game starts.
- **Round** only goes up by one, except back to 1 (or blank) for a new game. The tallies flash during a
  round change; round 9 looks like round 8 in this crop, so a round change while 8 strokes are showing
  counts as 9 (and so on until the tenth stroke, which is visible, confirms it).
- **Grenades** go down one at a time and come back at a round start (up to 4). **Magazine** goes down
  while the fire button is held (cross-checked against the recordings' input labels) and jumps up on a
  reload or weapon swap. **Reserve** changes on reloads, ammo buys and weapon swaps.

`check_clip` runs all of it over one clip's parsed arrays and returns per-step consistency flags (the
accuracy estimate) plus the events a summary is built from (`hud/summary.py`).
"""

from dataclasses import dataclass, field

import numpy as np

from zombiesai.hud.parse import OK, TRANSITION

SETTLE = 3  # ok steps with the same read before a value counts as settled
START_POINTS = 500
MAX_GAIN = 2000  # largest believable settled-to-settled gain in normal play (a train of kills, doubled)
DOWNED_FRACTION = 0.05
RESTART_GAP = 30  # steps without a points read (a menu, a load) before a return to 500 counts as a new game
FLASH_JOIN = 60  # flashing reads this close together belong to one round change (it blinks ~1 Hz for ~10 s)


@dataclass
class Run:
    start: int  # first step of the run
    end: int  # last step of the run (inclusive)
    value: int
    n_ok: int  # ok reads inside it


def settled_runs(values: np.ndarray, ok: np.ndarray, min_len: int = SETTLE, max_gap: int | None = None) -> list[Run]:
    """Maximal runs of ok reads holding one value, kept when they hold at least `min_len` reads. Non-ok
    steps inside don't break a run unless more than `max_gap` of them come in a row."""
    idx = np.flatnonzero(ok)
    runs = []
    if not len(idx):
        return runs
    v = values[idx]
    brk = np.diff(v) != 0
    if max_gap is not None:
        brk |= np.diff(idx) > max_gap + 1
    cut = np.flatnonzero(brk) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(v)]):
        if b - a >= min_len:
            runs.append(Run(int(idx[a]), int(idx[b - 1]), int(v[a]), int(b - a)))
    return runs


@dataclass
class PointsEvent:
    step: int  # first step showing the new settled value
    before: int
    after: int
    kind: str  # gain | spend | downed | game_over | new_game | implausible

    @property
    def delta(self) -> int:
        return self.after - self.before


def classify_points(runs: list[Run]) -> list[PointsEvent]:
    events = []
    for i in range(1, len(runs)):
        a, b = runs[i - 1].value, runs[i].value
        d = b - a
        if d == 0:
            continue
        nxt = runs[i + 1].value if i + 1 < len(runs) else None
        after_death = bool(events) and events[-1].kind in ("downed", "game_over")
        gap = runs[i].start - runs[i - 1].end
        if d % 10:
            kind = "implausible"
        elif b == START_POINTS and (d < 0 or after_death or gap >= RESTART_GAP):
            # Back to the starting points: a new game. From below too when the last game ended with fewer
            # than 500 (died in round 1) or the HUD was gone meanwhile (the restart went through a menu).
            kind = "new_game"
        elif d > 0 and (nxt == START_POINTS or (nxt is None and d > MAX_GAIN)):
            kind = "game_over"  # the counter runs up to the game's total score, then a new game starts
        elif d > 0:
            kind = "gain" if d <= MAX_GAIN else "implausible"
        elif -d == max(10, int(round(a * DOWNED_FRACTION / 10.0)) * 10) or abs(-d - a * DOWNED_FRACTION) <= 10:
            kind = "downed"
        elif -d <= a:
            kind = "spend"
        else:
            kind = "implausible"
        events.append(PointsEvent(runs[i].start, a, b, kind))
    return events


def roll_consistent(values: np.ndarray, ok: np.ndarray, runs: list[Run]) -> np.ndarray:
    """(T,) bool: ok reads that are part of a settled run, or lie between two settled values on a
    monotonic roll from one to the other. A misread shows up as a read off that path."""
    good = np.zeros(len(values), bool)
    for r in runs:
        seg = np.arange(r.start, r.end + 1)
        good[seg] = ok[seg] & (values[seg] == r.value)
    for a, b in zip(runs, runs[1:]):
        seg = np.arange(a.end + 1, b.start)
        seg = seg[ok[seg]]
        if not len(seg):
            continue
        lo, hi = sorted((a.value, b.value))
        v = values[seg]
        inside = (v >= lo - 1) & (v <= hi + 1)  # the counter can land one short and step up
        rising = b.value >= a.value
        # Monotone: each read no further back than the furthest the roll had already reached.
        reached = np.maximum.accumulate(v) if rising else np.minimum.accumulate(v)
        mono = np.abs(v - reached) <= 1
        good[seg] = inside & mono
    return good


@dataclass
class RoundEvent:
    step: int
    before: int
    after: int
    kind: str  # next | new_game | inferred_next (a change while 8 strokes show) | implausible


def flash_episodes(status: np.ndarray, join: int = FLASH_JOIN, min_steps: int = 5) -> list[tuple[int, int]]:
    """[first, last] steps of each round-change flash: flashing reads no more than `join` steps apart."""
    idx = np.flatnonzero(status == TRANSITION)
    out = []
    if not len(idx):
        return out
    cut = np.flatnonzero(np.diff(idx) > join) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]):
        if b - a >= min_steps:
            out.append((int(idx[a]), int(idx[b - 1])))
    return out


def round_track(values: np.ndarray, status: np.ndarray, flags: np.ndarray, min_len: int = 5):
    """Settled round reads -> (runs, events, per-step inferred round, per-step consistent).

    Runs are built from solid red reads only, and cut at every round-change flash. During the flash WaW
    blinks the groups separately (the first five strokes can show alone for a second) and blinks back to
    red between the white, so a flashing count is evidence of a change, not a reading to settle on; it is
    checked against the counts either side instead. A flash between two runs of 8 is round 9 (the ninth
    stroke is off the crop)."""
    readable = status == OK
    flashes = flash_episodes(status)
    epoch = np.zeros(len(values), np.int64)
    for _, last in flashes:
        epoch[last + 1 :] += 1
    key = np.where(readable, values.astype(np.int64) + 1000 * epoch, -1)
    runs = [Run(r.start, r.end, r.value % 1000, r.n_ok) for r in settled_runs(key, readable, min_len)]
    run_epoch = [int(epoch[r.start]) for r in runs]
    events: list[RoundEvent] = []
    inferred = np.full(len(values), -1, np.int32)
    good = np.zeros(len(values), bool)
    extra = 0  # rounds past 8 counted from round changes the crop cannot show
    last = None
    for i, r in enumerate(runs):
        if last is not None:
            flashed = run_epoch[i] > run_epoch[i - 1]
            if r.value == last.value + 1:
                events.append(RoundEvent(r.start, last.value, r.value, "next"))
            elif r.value == 1 and last.value > 1 and not flashed:
                events.append(RoundEvent(r.start, last.value + (extra if last.value == 8 else 0), 1, "new_game"))
                extra = 0
            elif r.value == last.value == 8 and flashed:
                extra += 1
                events.append(RoundEvent(r.start, 8 + extra - 1, 8 + extra, "inferred_next"))
            elif r.value != last.value:
                events.append(RoundEvent(r.start, last.value, r.value, "implausible"))
        if r.value != 8:
            extra = 0
        seg = slice(r.start, runs[i + 1].start if i + 1 < len(runs) else len(values))
        inferred[seg] = r.value + (extra if r.value == 8 else 0)
        good[r.start : r.end + 1] = readable[r.start : r.end + 1] & (values[r.start : r.end + 1] == r.value)
        last = r
    # A flashing read is consistent when it shows the count before or after the change, or the first
    # group alone (5) while the second one blinks.
    flash = np.flatnonzero(status == TRANSITION)
    if len(runs) and len(flash):
        starts = np.array([r.start for r in runs])
        k = np.searchsorted(starts, flash)  # the run after each flash step (len(runs) if none)
        before = np.array([runs[j - 1].value if j > 0 else -9 for j in k])
        after = np.array([runs[j].value if j < len(runs) else -9 for j in k])
        v = values[flash]
        good[flash] = (v == before) | (v == after) | ((v == 5) & (np.maximum(before, after) > 5))
    return runs, events, inferred, good


def counter_blips(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(T,) bool: ok reads that differ from both neighbouring ok reads while those two agree -- a one-step
    misread, the signature of a wrong glyph."""
    idx = np.flatnonzero(ok)
    out = np.zeros(len(values), bool)
    if len(idx) < 3:
        return out
    v = values[idx]
    blip = (v[1:-1] != v[:-2]) & (v[:-2] == v[2:])
    out[idx[1:-1][blip]] = True
    return out


def grenade_check(values: np.ndarray, ok: np.ndarray):
    """Settled grenade counts -> (runs, number of changes, number that are not -1/-2 or a refill)."""
    runs = settled_runs(values, ok)
    bad = 0
    changes = 0
    for a, b in zip(runs, runs[1:]):
        d = b.value - a.value
        if d == 0:
            continue
        changes += 1
        if not (d in (-1, -2) or (0 < d <= 4 and b.value <= 4)):
            bad += 1
    return runs, changes, bad


def mag_fire_check(mag: np.ndarray, ok: np.ndarray, fire: np.ndarray, window: int = 3) -> dict:
    """Cross-check the magazine with the recorded fire button: every drop of 1-3 rounds between two
    consecutive ok reads should have fire held within `window` steps before it."""
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return {"drops": 0, "drops_with_fire": 0, "fire_share": float("nan")}
    a, b = idx[:-1], idx[1:]
    d = mag[b] - mag[a]
    near = b - a <= 2
    drops = near & (d < 0) & (d >= -3)
    held = np.convolve(fire.astype(np.int32), np.ones(window + 1, np.int32))[: len(fire)] > 0
    n = int(drops.sum())
    with_fire = int((drops & held[b]).sum())
    return {"drops": n, "drops_with_fire": with_fire, "fire_share": with_fire / n if n else float("nan")}


@dataclass
class ClipCheck:
    """Per-field consistency over one clip. `*_good` are per-step flags; rates are over ok reads."""

    n_steps: int
    points_runs: list = field(default_factory=list)
    points_events: list = field(default_factory=list)
    points_good: np.ndarray | None = None
    round_runs: list = field(default_factory=list)
    round_events: list = field(default_factory=list)
    round_inferred: np.ndarray | None = None
    round_good: np.ndarray | None = None
    stats: dict = field(default_factory=dict)


def check_clip(parsed: dict[str, np.ndarray], fire: np.ndarray | None = None, playing: np.ndarray | None = None) -> ClipCheck:
    """All the consistency checks over one clip's parsed arrays. `fire` (T,) is the recorded fire button,
    `playing` (T,) masks out steps marked as menus/not playing (they still get parsed; they just don't
    count toward the rates)."""
    n = len(parsed["points"])
    playing = np.ones(n, bool) if playing is None else playing.astype(bool)
    out = ClipCheck(n)
    stats = out.stats

    p, pok = parsed["points"], parsed["points_status"] == OK
    out.points_runs = settled_runs(p, pok)
    out.points_events = classify_points(out.points_runs)
    out.points_good = roll_consistent(p, pok, out.points_runs)
    stats["points"] = _rates(pok, out.points_good, playing)
    stats["points"]["settled_changes"] = len(out.points_events)
    stats["points"]["implausible_changes"] = sum(e.kind == "implausible" for e in out.points_events)
    stats["points"]["blips"] = int(counter_blips(p, pok)[playing].sum())

    r, rs = parsed["round"], parsed["round_status"]
    out.round_runs, out.round_events, out.round_inferred, out.round_good = round_track(r, rs, parsed["round_flags"])
    rok = (rs == OK) | (rs == TRANSITION)
    stats["round"] = _rates(rok, out.round_good, playing)
    stats["round"]["changes"] = len(out.round_events)
    stats["round"]["implausible_changes"] = sum(e.kind == "implausible" for e in out.round_events)
    stats["round"]["blips"] = int(counter_blips(r, rok)[playing].sum())

    for name in ("grenades", "reserve", "mag"):
        v, ok = parsed[name], parsed[f"{name}_status"] == OK
        blips = counter_blips(v, ok)
        stats[name] = _rates(ok, ok & ~blips, playing)
        stats[name]["blips"] = int((blips & playing).sum())
    _, changes, bad = grenade_check(parsed["grenades"], parsed["grenades_status"] == OK)
    stats["grenades"]["changes"], stats["grenades"]["implausible_changes"] = changes, bad
    if fire is not None:
        stats["mag"].update(mag_fire_check(parsed["mag"], parsed["mag_status"] == OK, fire))
    return out


def _rates(ok: np.ndarray, good: np.ndarray, playing: np.ndarray) -> dict:
    n = int(playing.sum())
    n_ok = int((ok & playing).sum())
    n_good = int((good & ok & playing).sum())
    return {
        "steps": n,
        "read": n_ok,
        "read_rate": n_ok / n if n else float("nan"),
        "consistent": n_good,
        "consistent_rate": n_good / n_ok if n_ok else float("nan"),
    }


# ------------------------------------------------------------------------------------------------ online


@dataclass
class Tracked:
    """The live view of the HUD after one step: settled values, held through unreadable steps."""

    points: int = -1  # last settled points (-1 until one settles)
    points_event: str = ""  # kind of the settled change this step, if one settled ("" otherwise)
    points_delta: int = 0
    round: int = -1  # the round, counting round changes past 8
    round_changed: bool = False
    fresh: bool = False  # this step's points read agreed with the settled value or a roll toward it
    suspect_streak: int = 0  # settled changes in a row that the rules call implausible
    hud_lost: bool = False  # the reader has lost the HUD: too many implausible changes in a row


class HudTracker:
    """Step-at-a-time version of the checks, for a live loop: feed each step's `HudReading`, get settled
    values that only change when the game's rules allow it (PLAN.md: "a stateful HudTracker that may
    reject readings and hold the previous value"). It decides a change once the new value has held for
    `SETTLE` steps, so it lags the screen by ~0.2 s; a game-over count-up is reported when the reset to
    500 that follows it settles."""

    def __init__(self, lost_after: int = 3):
        self.lost_after = lost_after
        self.reset()

    def reset(self) -> None:
        self._runs: list[Run] = []
        self._cand, self._cand_n, self._step = None, 0, 0
        self._last_ok = -1
        self._round_cand, self._round_n = None, 0
        self._round_settled = -1
        self._extra = 0
        self._flash_steps: list[int] = []
        self._flash_since_settle = False
        self.state = Tracked()

    def step(self, r) -> Tracked:
        s = self.state
        s.points_event, s.points_delta, s.round_changed, s.fresh = "", 0, False, False
        t = self._step
        self._step += 1
        if r.points_status == OK:
            if r.points == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = r.points, 1
            if self._cand_n == SETTLE and (not self._runs or self._runs[-1].value != self._cand):
                start = t - SETTLE + 1
                self._runs.append(Run(start, t, self._cand, SETTLE))
                if len(self._runs) >= 2:
                    # Decide the previous change now that this one is known (a count-up is only a game
                    # over once the reset to 500 follows it).
                    events = classify_points(self._runs[-3:] if len(self._runs) >= 3 else self._runs[-2:])
                    ev = events[-1]
                    if ev.kind == "gain" and ev.delta > MAX_GAIN:
                        ev.kind = "implausible"
                    s.points_event, s.points_delta = ev.kind, ev.delta
                    s.suspect_streak = s.suspect_streak + 1 if ev.kind == "implausible" else 0
                    s.hud_lost = s.suspect_streak >= self.lost_after
                    if ev.kind != "implausible":
                        s.points = self._cand
                else:
                    s.points = self._cand
            elif self._runs and self._runs[-1].value == r.points:
                self._runs[-1].end = t
            if s.points >= 0:
                lo, hi = sorted((s.points, self._cand))
                s.fresh = lo - 1 <= r.points <= hi + 1
        # Round: settle solid red counts; a flash between two settles of 8 is round 9 (and on).
        if r.round_status == TRANSITION:
            self._flash_steps = [x for x in self._flash_steps if t - x <= FLASH_JOIN] + [t]
            if len(self._flash_steps) >= 5:
                self._flash_since_settle = True
        if r.round_status != OK:
            self._round_n = 0  # after a flash or a blank the count has to settle again
        else:
            self._round_n = self._round_n + 1 if r.round == self._round_cand else 1
            self._round_cand = r.round
            if self._round_n == 5:
                prev, cur = self._round_settled, s.round
                # The moves round_track calls plausible: the next round, back to 1, or 8 -> flash -> 8. Any
                # other jump (1 -> 10) is a misread, held like an implausible points change.
                if prev < 0 or r.round in (prev, cur, cur + 1, 1):
                    if r.round != prev:
                        self._extra = 0
                        s.round_changed = prev >= 0 and r.round != cur
                    elif r.round == 8 and self._flash_since_settle:
                        self._extra += 1
                        s.round_changed = True
                    self._round_settled = r.round
                    self._flash_since_settle = False
                    s.round = r.round + (self._extra if r.round == 8 else 0)
        return s
