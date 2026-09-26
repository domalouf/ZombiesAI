"""Raw human input -> spec actions.

A demo is only worth what its labels are worth, and the labels come from what the recorder logged off the
mouse and keyboard. Two decisions here follow PLAN.md ("Demonstrations and BC"):

* **Raw mouse counts, never cursor deltas.** In a mouse-look FPS the cursor is captured and re-centred, so
  position deltas are useless. Counts are also the unit the synthetic mouse action emits, and that symmetry
  is the whole reason a human's label means anything to the policy that will replay it.
* **Keep the pre-quantization degrees.** The bins are a spec decision that may change; the recording must
  not have to. Re-binning is then arithmetic over `yaw_deg`, not another evening of play.

The log itself is newline-delimited JSON on one monotonic clock, shared with whatever captured the frames:

    {"t": 1234.5, "type": "key",    "code": "w",      "down": true}
    {"t": 1234.6, "type": "button", "code": "mouse1", "down": true}
    {"t": 1234.7, "type": "mouse",  "dx": 42, "dy": -3}
    {"t": 1240.0, "type": "marker", "name": "round_start"}

One key is not a control at all: the **mark key** (F8 unless changed). The player taps it on the way into a
menu, the pause screen, a loading screen or the game-over card, and again on the way back, and the steps in
between are flagged as not playing (`not_playing` below; `clips.FLAG_NOT_PLAYING`). It stays in the raw log
like every other key, so the marking is recomputed from the log exactly as the actions are.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zombiesai import spec

CONTROLS = ("forward", "back", "left", "right", "sprint", "fire", "ads", "use", "reload", "melee", "grenade", "swap")
CONTROL_INDEX = {c: i for i, c in enumerate(CONTROLS)}

# Call of Duty: World at War's default PC bindings. Remap in the recorder's session.json if yours differ --
# the recorder stores the map it used, so a clip is always interpretable years later.
DEFAULT_BINDINGS = {
    "w": "forward",
    "s": "back",
    "a": "left",
    "d": "right",
    "shift": "sprint",
    "mouse1": "fire",
    "mouse2": "ads",
    "f": "use",
    "r": "reload",
    "v": "melee",
    "mouse3": "melee",
    "g": "grenade",
    "q": "swap",
}
# One button head, several buttons possible in one 67 ms window. Buying beats everything (it is rare, and
# mislabelling it teaches the policy that the prompt means nothing); a swap loses to every other press.
BUTTON_PRIORITY = ("use", "grenade", "reload", "melee", "swap")
HOLD_FRACTION = 0.5  # a control counts as held for a decision if it was down for at least half of it
# F8 because World at War binds nothing to it by default (F5, F10 and F12 are taken), and it sits where a
# hand can find it without looking. It must never be a control: see InputConfig.__post_init__.
MARK_KEY = "f8"
YAW_LIMIT_DEG = max(spec.YAW_BINS_DEG)
PITCH_LIMIT_DEG = max(spec.PITCH_BINS_DEG)

_YAW_BINS = np.array(spec.YAW_BINS_DEG)
_PITCH_BINS = np.array(spec.PITCH_BINS_DEG)


@dataclass(frozen=True)
class InputConfig:
    """How to read one recording. `counts_per_degree` is spike S4's number: mouse counts per degree of yaw
    at the sensitivity the demo was played at. Get it wrong and every look label is scaled wrong."""

    counts_per_degree: float = 1.0
    bindings: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    hold_fraction: float = HOLD_FRACTION
    # Mouse counts arrive in bursts; a step whose look exceeds the widest bin is a flick the action space
    # cannot express. Clamp it, and flag the step so it can be dropped from training instead of teaching a
    # 30-degree turn where the human turned 90.
    flag_clamped: bool = True
    # The key that toggles "not playing" (see `PlayMarker`). None turns the marking off.
    mark_key: str | None = MARK_KEY

    def __post_init__(self):
        if self.mark_key is not None:
            object.__setattr__(self, "mark_key", self.mark_key.lower())
            if self.mark_key in {str(code).lower() for code in self.bindings}:
                # A mark key that is also bound would pause the dataset every time the player reloaded.
                raise ValueError(f"mark key {self.mark_key!r} is bound to {self.bindings[self.mark_key]!r}")


@dataclass
class Labels:
    """Per-decision actions plus everything needed to re-derive them under different bins."""

    actions: np.ndarray  # (T, 8) uint8
    yaw_deg: np.ndarray  # (T,) float32, summed raw look before binning
    pitch_deg: np.ndarray  # (T,) float32
    held: np.ndarray  # (T, len(CONTROLS)) float32 fraction of the decision each control was down
    presses: np.ndarray  # (T, len(CONTROLS)) int16 down-edges inside the decision
    clamped: np.ndarray  # (T,) bool: look exceeded the widest bin

    def __len__(self) -> int:
        return len(self.actions)


def inverse_bindings(bindings: dict[str, str]) -> dict[str, str]:
    """control -> the code that drives it, first binding wins.

    The map is written code-first because that is how a log reads, but everything that *emits* input -- the
    synthetic log in `synthesize`, the agent's dispatcher in `realgame/dispatch.py` -- needs it the other way
    round. Deriving it here keeps one source of truth for which key means "reload"."""
    out: dict[str, str] = {}
    for code, control in bindings.items():
        out.setdefault(control, code)
    return out


def read_log(path: str | Path) -> list[dict]:
    """Parse inputs.jsonl, tolerating a truncated final line from a recorder that was killed."""
    events = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break
    events.sort(key=lambda e: e["t"])
    return events


def bin_index(degrees, bins: np.ndarray) -> np.ndarray:
    """Nearest bin by angle. Ties go to the smaller magnitude, so noise around zero stays still."""
    d = np.asarray(degrees, dtype=np.float64)[..., None]
    cost = np.abs(d - bins) + 1e-9 * np.abs(bins)
    return cost.argmin(axis=-1).astype(np.int64)


class InputFolder:
    """Streaming fold of raw events into per-decision measurements.

    It exists because a recording is read twice: once offline over a whole log, and once live, one decision
    at a time, while a person is playing. Both go through this, so a demo recorded at 15 Hz and the same log
    re-quantized later cannot disagree.
    """

    def __init__(self, config: InputConfig | None = None):
        self.config = config or InputConfig()
        self._down: list[float | None] = [None] * len(CONTROLS)

    @property
    def held_controls(self) -> tuple[str, ...]:
        return tuple(CONTROLS[i] for i, t in enumerate(self._down) if t is not None)

    def feed(self, events, start: float, end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One window's (held fraction, press counts, summed mouse counts). Events outside it are clamped in."""
        held = np.zeros(len(CONTROLS))
        presses = np.zeros(len(CONTROLS), dtype=np.int16)
        counts = np.zeros(2)
        for event in events:
            kind = event.get("type")
            if kind == "mouse":
                counts += (event.get("dx", 0), event.get("dy", 0))
                continue
            if kind not in ("key", "button"):
                continue
            t = min(max(float(event["t"]), start), end)
            control = self.config.bindings.get(str(event["code"]).lower())
            if control is None:
                continue
            i = CONTROL_INDEX[control]
            if event["down"]:
                if self._down[i] is None:
                    self._down[i] = t
                    presses[i] += 1
            elif self._down[i] is not None:
                held[i] += t - max(self._down[i], start)
                self._down[i] = None
        for i, since in enumerate(self._down):
            if since is not None:
                held[i] += end - max(since, start)
        return held / max(end - start, 1e-9), presses, counts


def actions_from(held: np.ndarray, presses: np.ndarray, counts: np.ndarray, config: InputConfig) -> Labels:
    """Turn per-window measurements into factored actions. The one place bins are applied."""
    held = np.atleast_2d(np.asarray(held, dtype=np.float64))
    presses = np.atleast_2d(np.asarray(presses, dtype=np.int64))
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    yaw_deg = counts[:, 0] / config.counts_per_degree
    pitch_deg = -counts[:, 1] / config.counts_per_degree  # raw dy counts down; positive pitch looks up
    clamped = (np.abs(yaw_deg) > YAW_LIMIT_DEG) | (np.abs(pitch_deg) > PITCH_LIMIT_DEG)

    down = held >= config.hold_fraction
    take = lambda name: down[:, CONTROL_INDEX[name]]  # noqa: E731
    actions = np.zeros((len(held), len(spec.ACTION_NVEC)), dtype=np.uint8)
    actions[:, spec.STRAFE] = take("right").astype(np.int8) - take("left") + 1
    actions[:, spec.FORWARD] = take("forward").astype(np.int8) - take("back") + 1
    actions[:, spec.YAW] = bin_index(np.clip(yaw_deg, -YAW_LIMIT_DEG, YAW_LIMIT_DEG), _YAW_BINS)
    actions[:, spec.PITCH] = bin_index(np.clip(pitch_deg, -PITCH_LIMIT_DEG, PITCH_LIMIT_DEG), _PITCH_BINS)
    actions[:, spec.FIRE] = take("fire")
    actions[:, spec.ADS] = take("ads")
    actions[:, spec.SPRINT] = take("sprint")
    # One button head, several presses possible in one 67 ms window: assign in reverse priority so the
    # press that matters most is written last.
    for name in reversed(BUTTON_PRIORITY):
        actions[presses[:, CONTROL_INDEX[name]] > 0, spec.BUTTON] = spec.BUTTONS.index(name)
    return Labels(
        actions,
        yaw_deg.astype(np.float32),
        pitch_deg.astype(np.float32),
        held.astype(np.float32),
        presses.astype(np.int16),
        clamped,
    )


class PlayMarker:
    """Streaming fold of the mark key into "was this decision play?", one decision at a time.

    Recording starts in "playing"; each press of the mark key flips it. A decision counts as not playing if
    the state was "not playing" at *any* moment of its input window -- so the step whose window holds the
    press that stops play is already excluded, and so is the one whose window holds the press that resumes
    it. Play resumes on the step after that: its frame was captured at the end of the resuming window, after
    the player had said they were back. Both edges round towards exclusion because a menu frame mislabelled
    as play costs more than one real step thrown away.

    Only down-edges count, and a key already down does not count again: Windows Raw Input repeats the make
    code while a key is held (evdev's value-2 repeats are dropped at decode), and a held F8 is one toggle.
    """

    def __init__(self, key: str | None = MARK_KEY):
        self.key = None if key is None else key.lower()
        self.playing = True
        self._down = False

    def feed(self, events) -> bool:
        """One window's events (already in time order) -> True if any part of the window was not play."""
        not_playing = not self.playing
        if self.key is None:
            return not_playing
        for event in events:
            if event.get("type") != "key" or str(event.get("code", "")).lower() != self.key:
                continue
            if event["down"] and not self._down:
                self.playing = not self.playing
                not_playing = True
            self._down = bool(event["down"])
        return not_playing


def _windows(events, t0: float, n_steps: int, dt: float):
    """The log cut into decision windows on deadlines t0 + k*dt: (k, start, end, events in the window).

    Shared by the action quantizer and the not-playing marker, so an event can never count towards one
    step's action and another step's marking."""
    events = sorted((e for e in events if e.get("type") in ("key", "button", "mouse")), key=lambda e: e["t"])
    cursor = 0
    for k in range(n_steps):
        start, end = t0 + k * dt, t0 + (k + 1) * dt
        first = cursor
        while cursor < len(events) and events[cursor]["t"] < end:
            cursor += 1
        yield k, start, end, events[first:cursor]


def not_playing(
    events, t0: float, n_steps: int, config: InputConfig | None = None, dt: float | None = None
) -> np.ndarray:
    """(T,) bool: which decisions of a raw log the player marked as not playing, by `PlayMarker`'s rule.

    The recorder applies the same marker live; this is the offline half, so `requantize` can re-derive the
    marking -- under a different mark key, if the one recorded with was wrong."""
    config = config or InputConfig()
    dt = dt or 1.0 / spec.DECISION_HZ
    marker = PlayMarker(config.mark_key)
    out = np.zeros(n_steps, dtype=bool)
    for k, _start, _end, window in _windows(events, t0, n_steps, dt):
        out[k] = marker.feed(window)
    return out


def quantize(
    events, t0: float, n_steps: int, config: InputConfig | None = None, dt: float | None = None
) -> Labels:
    """Fold a raw input log into one factored action per decision, on deadlines t0 + k*dt."""
    config = config or InputConfig()
    dt = dt or 1.0 / spec.DECISION_HZ
    folder = InputFolder(config)
    held = np.zeros((n_steps, len(CONTROLS)))
    presses = np.zeros((n_steps, len(CONTROLS)), dtype=np.int64)
    counts = np.zeros((n_steps, 2))
    for k, start, end, window in _windows(events, t0, n_steps, dt):
        held[k], presses[k], counts[k] = folder.feed(window, start, end)
    return actions_from(held, presses, counts, config)


def synthesize(action, start: float, dt: float, config: InputConfig | None = None) -> list[dict]:
    """The raw events a person would have produced for one factored action -- the plan's FakeInput.

    It makes the recorder and the quantizer testable without a game or a mouse, and asserts the property
    that matters: what the recorder writes for a demonstration is the action that produced it.
    """
    config = config or InputConfig()
    values = spec.action_tuple(action)
    inverse = inverse_bindings(config.bindings)
    events: list[dict] = []
    mid = start + dt / 2

    def hold(control: str) -> None:
        # Released just inside the window rather than exactly on its edge: a release landing on a deadline
        # is ambiguous about which decision it belongs to, and a synthetic log should not lean on that.
        code = inverse[control]
        kind = "button" if code.startswith("mouse") else "key"
        events.append({"t": start, "type": kind, "code": code, "down": True})
        events.append({"t": start + 0.98 * dt, "type": kind, "code": code, "down": False})

    for control, active in (
        ("forward", spec.FORWARD_VALUES[values[spec.FORWARD]] > 0),
        ("back", spec.FORWARD_VALUES[values[spec.FORWARD]] < 0),
        ("right", spec.STRAFE_VALUES[values[spec.STRAFE]] > 0),
        ("left", spec.STRAFE_VALUES[values[spec.STRAFE]] < 0),
        ("fire", values[spec.FIRE]),
        ("ads", values[spec.ADS]),
        ("sprint", values[spec.SPRINT]),
    ):
        if active:
            hold(control)
    button = spec.BUTTONS[values[spec.BUTTON]]
    if button != "none":
        code = inverse[button]
        kind = "button" if code.startswith("mouse") else "key"
        events.append({"t": mid, "type": kind, "code": code, "down": True})
        events.append({"t": mid + dt / 8, "type": kind, "code": code, "down": False})
    dx = round(spec.YAW_BINS_DEG[values[spec.YAW]] * config.counts_per_degree)
    dy = -round(spec.PITCH_BINS_DEG[values[spec.PITCH]] * config.counts_per_degree)
    if dx or dy:
        events.append({"t": mid, "type": "mouse", "dx": int(dx), "dy": int(dy)})
    return sorted(events, key=lambda e: e["t"])


def label_confidence(labels: Labels, config: InputConfig | None = None) -> np.ndarray:
    """1.0 for a step the log describes unambiguously; lower where a control was held for part of the step
    (so the 0/1 label is a coin-flip) or the look was clamped."""
    config = config or InputConfig()
    ambiguity = np.abs(labels.held - config.hold_fraction).min(axis=1, initial=config.hold_fraction)
    confidence = 0.5 + 0.5 * np.clip(ambiguity / config.hold_fraction, 0.0, 1.0)
    if config.flag_clamped:
        confidence = np.where(labels.clamped, np.minimum(confidence, 0.5), confidence)
    return confidence.astype(np.float32)


def yaw_flow_agreement(
    yaw_deg: np.ndarray, frames: np.ndarray, sample: int = 250, lags=(0, 1, 2, 3), seed: int = 0
) -> dict:
    """Cross-check look labels against the pixels: a turn to the right must drag the image left.

    The action recorded at step k is what the player *commanded* while looking at frame k, and the game
    answers a decision or two later, so the check sweeps a few lags and reports the one that fits best.
    That best lag is not just bookkeeping -- it is the closed-loop delay, measured from the recording
    itself, and it should match what spike S3 measured and what the sim's latency knob is set to.

    A correlation below about 0.9 at every lag means the input log and the capture are misaligned in time,
    or `counts_per_degree` is wrong. Either way it is a bug no amount of training will absorb.
    """
    from zombiesai.demos.frames import estimate_shift

    n = min(len(yaw_deg), len(frames))
    lags = tuple(int(lag) for lag in lags)
    turned = np.flatnonzero(np.abs(np.asarray(yaw_deg[:n])) > 0.5)
    turned = turned[turned + max(lags) + 1 < n]
    if len(turned) < 8:
        return {"n": int(len(turned)), "lag": None, "correlation": float("nan"), "by_lag": {}}
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(turned, size=min(sample, len(turned)), replace=False))

    shifts: dict[int, float] = {}
    for i in idx:
        for lag in lags:
            j = int(i) + lag
            if j not in shifts:
                shifts[j] = float(estimate_shift(frames[j], frames[j + 1])[0])
    turns = np.asarray(yaw_deg, dtype=np.float64)[idx]
    by_lag: dict[int, float] = {}
    for lag in lags:
        moved = np.array([shifts[int(i) + lag] for i in idx])
        if moved.std() < 1e-9 or turns.std() < 1e-9:
            continue
        # Negated: positive yaw (turning right) drags the scene to negative x.
        by_lag[lag] = float(-np.corrcoef(turns, moved)[0, 1])
    if not by_lag:
        return {"n": int(len(idx)), "lag": None, "correlation": float("nan"), "by_lag": {}}
    best = max(by_lag, key=lambda lag: by_lag[lag])
    return {
        "n": int(len(idx)),
        "lag": int(best),
        "correlation": by_lag[best],
        "by_lag": by_lag,
        "mean_abs_shift_px": float(np.mean([abs(shifts[int(i) + best]) for i in idx])),
    }


def fire_ammo_agreement(fire: np.ndarray, mag_ammo: np.ndarray) -> dict:
    """The other cheap cross-check: firing must correlate with the magazine going down."""
    fire = np.asarray(fire, dtype=np.float64)
    spent = -np.diff(np.asarray(mag_ammo, dtype=np.float64), prepend=mag_ammo[0])
    spent = np.clip(spent, 0.0, None)  # a reload refills the magazine; only decrements are shots
    if fire.std() < 1e-9 or spent.std() < 1e-9:
        return {"n": int(len(fire)), "correlation": float("nan")}
    return {
        "n": int(len(fire)),
        "correlation": float(np.corrcoef(fire, spent)[0, 1]),
        "shots_while_not_firing": float(spent[fire < 0.5].sum()),
    }
