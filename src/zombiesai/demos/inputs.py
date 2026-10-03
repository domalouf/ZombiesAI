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

The mouse wheel is logged as the pseudo-buttons `wheelup`/`wheeldown` (and `wheelleft`/`wheelright`), one
press and release at the same instant per notch. That keeps it inside the existing schema: a wheel notch
bound to "swap" is then counted by the fold exactly like a tap of the key bound to it, with no new event
type for every reader of the log to learn.

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
    # The stock config also cycles weapons on the wheel (MWHEELDOWN weapnext, MWHEELUP weapprev). Nacht
    # gives you two guns, so either direction is simply "swap". Left out, every scrolled swap in a demo
    # would go unlabelled and the policy would learn that the weapon changes by itself.
    "wheeldown": "swap",
    "wheelup": "swap",
}
# Codes the recorder logs but nothing can send: the virtual device has no wheel, and the action space has
# one "swap" whichever way you scroll. See inverse_bindings.
WHEEL_CODES = ("wheelup", "wheeldown", "wheelleft", "wheelright")
# The desktop's own modifier. While it is down, keys and the mouse drive the compositor -- Super+1..4 switches
# workspace, Super+drag moves a window -- and the game never sees them, so they must not become labels ("1"
# is weapon swap). "key125"/"key126" are how logs from before the key had a name spell it.
COMPOSITOR_MODIFIERS = ("super", "key125", "key126")
# One button head, several buttons possible in one 67 ms window. Buying beats everything (it is rare, and
# mislabelling it teaches the policy that the prompt means nothing); a swap loses to every other press.
# Presses of one control collapse the same way: three wheel notches inside a window are one "swap" label,
# because the head can say that a swap happened but not how many. At 15 Hz that only bites on a fast
# scroll, and with two weapons an even number of swaps is a no-op the label cannot tell apart from one.
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
    """control -> the code that drives it: first key or mouse button wins, a wheel direction only if the
    control has nothing else bound.

    The map is written code-first because that is how a log reads, but everything that *emits* input -- the
    synthetic log in `synthesize`, the agent's dispatcher in `realgame/dispatch.py` -- needs it the other way
    round. Deriving it here keeps one source of truth for which key means "reload".

    Several codes can drive one control ("q", "wheeldown" and "wheelup" are all "swap"), so the choice has
    to be deterministic and emittable: the virtual device has no wheel, so a wheel code is used only as a
    last resort, and then the dispatcher fails loudly on it rather than sending nothing."""
    out: dict[str, str] = {}
    for code, control in bindings.items():
        if code not in WHEEL_CODES:
            out.setdefault(control, code)
    for code, control in bindings.items():
        out.setdefault(control, code)
    return out


def _event_kind(code: str) -> str:
    return "button" if code.startswith("mouse") or code in WHEEL_CODES else "key"


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
        self._chord: set[str] = set()  # compositor modifiers currently down

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
            code = str(event.get("code", "")).lower()
            if kind == "key" and code in COMPOSITOR_MODIFIERS:
                (self._chord.add if event["down"] else self._chord.discard)(code)
                continue
            if kind == "mouse":
                if not self._chord:
                    counts += (event.get("dx", 0), event.get("dy", 0))
                continue
            if kind not in ("key", "button"):
                continue
            t = min(max(float(event["t"]), start), end)
            if self._chord and event["down"]:
                # A chord the compositor took. Its release still lands below, but finds nothing held: a key
                # that was already down before Super keeps its hold, one pressed under Super never starts one.
                continue
            control = self.config.bindings.get(code)
            if control is None:
                continue
            i = CONTROL_INDEX[control]
            if code in WHEEL_CODES:
                # A notch has no duration: count its press and leave the held state alone, so scrolling
                # while "q" is down cannot release the swap that "q" is holding (or vice versa).
                presses[i] += bool(event["down"])
                continue
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

    Only down-edges count, and a key already down does not count again: a held F8 is one toggle, however many
    downs the log holds for it (evdev's value-2 repeats are dropped at decode, but nothing else promises a log
    without them).
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
        kind = _event_kind(code)
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
        kind = _event_kind(code)
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


@dataclass(frozen=True)
class FlowCheck:
    """Pass/fail criteria for `yaw_flow_agreement`, in one place, each number with the reason it is that number.

    The check exists to catch two bugs, and each leaves its own fingerprint in the pixels:

    * **The log and the frames are out of step in time.** Then yaw agrees with image motion best at some
      *other* lag than the recording's closed-loop delay. A timing bug moves the peak of the lag curve; it
      does not flatten it (on demo_0000, rolling the labels by k steps moves the peak by exactly k and leaves
      its height alone). So timing is judged on *where* the peak is, never on how tall it is.
    * **`counts_per_degree` is wrong.** Then yaw and image motion still agree in rank -- a bigger turn still
      moves the image more -- so no correlation can see it. What changes is the *scale*: pixels per labelled
      degree, which the game's field of view fixes. So scale is judged on the implied horizontal FOV.

    Plain Pearson correlation, the first version of this check, failed both ways on real footage: a handful of
    SAD matches locked onto fog, a zombie or the static gun dragged it to 0.52 on a perfectly good recording,
    and it was blind to a sensitivity error anyway. Everything here is a rank, a median or a bootstrap instead.
    """

    # Steps whose labelled yaw is below this are not sampled: an integer-pixel search cannot see a turn that
    # moves the 128-px frame by under a pixel, so they carry only noise.
    min_turn_deg: float = 0.5
    # Fewer sampled turning steps than this and the verdict is "too_little_turning", not a guess.
    min_samples: int = 8
    # Spearman rank correlation of yaw against image motion at the best lag must reach this. demo_0000 (real,
    # 10 minutes) scores 0.70-0.74 depending on the sample; recordings of the synthetic stand-in 0.73-0.90;
    # labels shuffled against their frames stay within +-0.2 on a few hundred samples but reached 0.34 on a
    # short clip's ~40 turning steps, hence the second condition.
    min_rank_correlation: float = 0.3
    # ...and be this many standard errors (rho * sqrt(n - 1)) clear of zero, so a short clip's lucky 0.34 at
    # one of seven lags is not read as signal (that one is 2.3). Real footage clears it six times over.
    min_z: float = 2.5
    # The best lag is only called wrong when it beats the expected lag with this bootstrap confidence (the
    # share of resamples of the sampled steps in which it still does). A fixed margin will not do: on
    # demo_0000 the gap between the true peak and its neighbour is 0.09-0.20 depending on which 400 steps are
    # drawn, and a shifted log shows the same gap the other way round. The paired bootstrap knows how noisy
    # this particular sample is. A delay that straddles a decision boundary splits the response between two
    # lags, the resamples disagree about which is higher, and nothing is flagged -- which is right.
    lag_confidence: float = 0.95
    bootstrap: int = 200
    # The px/deg fit uses turns in this range. Below 2 degrees the shift is a pixel or two and the ratio is
    # all rounding. Above 8 the shift nears the +-24 px search edge if the FOV is narrow (8 degrees at 2.4
    # px/deg, a 2x sensitivity error on real footage, is 19 px), and big flicks smear.
    slope_turn_deg: tuple[float, float] = (2.0, 8.0)
    # Horizontal FOV implied by the measured px/deg, on the 128-px observation. WaW's cg_fov 65-80 is 81-96
    # degrees at 16:9 (Hor+); the synthetic stand-in draws 80 and the estimator reads 79.9-80.1 there;
    # demo_0000 reads 83-86.
    # A 2x error in counts_per_degree moves an 80-96 degree FOV to 118-132 (labels too big) or 45-59 (too
    # small); the band sits between. A 1.5x error mostly passes -- it is the gross errors this is for. ADS does
    # not bias it: WaW scales sensitivity with zoom (demo_0000: 1.17 px/deg aiming down sights, 1.19 hip).
    fov_deg: tuple[float, float] = (62.0, 110.0)
    # Only for the hint in the "wrong_scale" reason: the FOV to assume when suggesting a corrected
    # counts_per_degree. demo_0000 measures 85; WaW's default is 81.
    typical_fov_deg: float = 85.0


FLOW_CHECK = FlowCheck()


def implied_fov_deg(px_per_deg: float, width: int = spec.PIXELS_SHAPE[1]) -> float:
    """Horizontal field of view of a pinhole camera whose centre pixels move `px_per_deg` per degree of yaw."""
    if not px_per_deg > 0:
        return float("nan")
    focal = px_per_deg * 180.0 / np.pi  # pixels per radian at the centre of the frame
    return float(np.degrees(2.0 * np.arctan(width / 2.0 / focal)))


def _px_per_deg_at(fov_deg: float, width: int = spec.PIXELS_SHAPE[1]) -> float:
    """The inverse of `implied_fov_deg`."""
    return float(width / 2.0 / np.tan(np.radians(fov_deg) / 2.0) * np.pi / 180.0)


def _ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _px_per_deg(turns: np.ndarray, moved: np.ndarray) -> float:
    """Robust slope of image motion against yaw, through the origin; NaN if there is nothing to fit.

    Zero shifts are dropped first. In the fitting range every plausible FOV moves the image at least a pixel,
    so a zero is the SAD search locking onto something that does not move with the view -- the gun, the HUD,
    a stretch of featureless wall -- not a measurement; a short clip can be mostly those. The median of
    per-step ratios then shrugs off the wrong locks that remain, but on a clip of a few discrete turns it is
    stuck on integer-pixel ratios (2 degrees reads 1.0 or 1.5 px/deg, never 1.33). So it only picks the
    inliers -- steps within 1.5 px or 30% of what it predicts -- and a least-squares fit through the origin
    over those gives the number, which averages the rounding away.
    """
    keep = moved != 0
    turns, moved = turns[keep], moved[keep]
    if len(turns) < 3:
        return float("nan")
    rough = float(np.median(moved / turns))
    if not rough > 0:
        return rough
    inlier = np.abs(moved - rough * turns) <= np.maximum(1.5, 0.3 * rough * np.abs(turns))
    if inlier.sum() < 3:
        return rough
    t, m = turns[inlier], moved[inlier]
    return float((t * m).sum() / (t * t).sum())


def yaw_flow_agreement(
    yaw_deg: np.ndarray,
    frames: np.ndarray,
    sample: int = 400,
    lags=(-3, -2, -1, 0, 1, 2, 3),
    expected_lag: int = 0,
    seed: int = 0,
    check: FlowCheck = FLOW_CHECK,
) -> dict:
    """Cross-check look labels against the pixels: a turn to the right must drag the image left, by an amount
    the game's field of view fixes, at the moment the recording's closed-loop delay says.

    Samples up to `sample` turning steps, measures each one's horizontal image shift with `estimate_shift` at
    a sweep of lags, and reports:

    * **lag** -- where yaw and image motion agree best, by Spearman rank correlation (all lags in `by_lag`).
      The action recorded at step k is what the player commanded while looking at frame k, so on a correctly
      paired recording the response shows up between frames k+lag and k+lag+1, where lag is the closed-loop
      delay in decisions: 0 for the recorder on the real game (demo_0000), and a source's own
      `latency_steps` where it declares one. The sweep includes negative lags on purpose -- a peak there means the image moved *before*
      the hand did, which only a misaligned log produces. `lag_confidence` is the bootstrap share of
      resamples in which the best lag beats `expected_lag`.
    * **px_per_deg** and the **fov_deg** it implies -- a robust slope of image shift against labelled yaw at
      the best lag. A wrong `counts_per_degree` leaves every correlation untouched and shows up only here.
    * **verdict** -- "ok", or the first thing that is wrong: "too_little_turning"; "no_signal" (yaw does not
      track the pixels at any lag: labels from another recording, a flipped sign, a broken capture);
      "misaligned" (the best lag is confidently not `expected_lag`); or "wrong_scale" (an implied FOV no
      game would have). `reasons` says why in words. The criteria are `check`'s, documented on `FlowCheck`.

    Cost: one SAD search per sampled step per lag, under a second for 400 steps.
    """
    from zombiesai.demos.frames import estimate_shift

    n = min(len(yaw_deg), len(frames))
    yaw = np.asarray(yaw_deg[:n], dtype=np.float64)
    expected_lag = int(expected_lag)
    lags = tuple(sorted({int(lag) for lag in lags} | {expected_lag}))
    turned = np.flatnonzero(np.abs(yaw) > check.min_turn_deg)
    turned = turned[(turned + min(lags) >= 0) & (turned + max(lags) + 1 < n)]
    report = {
        "n": int(len(turned)),
        "verdict": "too_little_turning",
        "reasons": [f"only {len(turned)} turning steps; need {check.min_samples}"],
        "lag": None,
        "expected_lag": expected_lag,
        "rank_correlation": float("nan"),
        "by_lag": {},
        "px_per_deg": float("nan"),
        "fov_deg": float("nan"),
    }
    if len(turned) < check.min_samples:
        return report
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(turned, size=min(sample, len(turned)), replace=False))
    report["n"] = int(len(idx))

    shifts: dict[int, float] = {}
    for i in idx:
        for lag in lags:
            j = int(i) + lag
            if j not in shifts:
                shifts[j] = float(estimate_shift(frames[j], frames[j + 1])[0])
    turns = yaw[idx]
    # Negated: positive yaw (turning right) drags the scene to negative x.
    moved = {lag: -np.array([shifts[int(i) + lag] for i in idx]) for lag in lags}
    turn_ranks = _ranks(turns)
    moved_ranks = {lag: _ranks(m) for lag, m in moved.items()}
    by_lag = {lag: _pearson(turn_ranks, r) for lag, r in moved_ranks.items()}
    by_lag = {lag: rho for lag, rho in by_lag.items() if not np.isnan(rho)}
    report["by_lag"] = by_lag
    if not by_lag:
        report.update(verdict="no_signal", reasons=["the image did not move at any lag"])
        return report
    best = max(by_lag, key=lambda lag: by_lag[lag])
    rho = by_lag[best]
    expected = by_lag.get(expected_lag, float("-inf"))

    # Paired bootstrap over the sampled steps: in what share of resamples does the best lag still beat the
    # expected one? Ranks are kept from the full sample; for a confidence that only has to separate 0.95
    # from a coin flip, re-ranking every resample is not worth the time.
    confidence = 0.0
    if best != expected_lag and expected_lag in by_lag:
        boot = np.random.default_rng(seed + 1)
        wins = 0
        for _ in range(check.bootstrap):
            pick = boot.integers(0, len(idx), len(idx))
            t = turn_ranks[pick]
            wins += _pearson(t, moved_ranks[best][pick]) > _pearson(t, moved_ranks[expected_lag][pick])
        confidence = wins / check.bootstrap

    lo, hi = check.slope_turn_deg
    fit = (np.abs(turns) >= lo) & (np.abs(turns) <= hi)
    px_per_deg = _px_per_deg(turns[fit], moved[best][fit])
    width = int(np.shape(frames[0])[1])
    fov = implied_fov_deg(px_per_deg, width)
    report.update(
        lag=int(best),
        rank_correlation=rho,
        lag_confidence=confidence,
        px_per_deg=px_per_deg,
        fov_deg=fov,
        sign_agreement=float(np.mean(np.sign(turns) == np.sign(moved[best]))),
        mean_abs_shift_px=float(np.mean(np.abs(moved[best]))),
    )

    z = rho * np.sqrt(len(idx) - 1)
    if rho < check.min_rank_correlation or z < check.min_z:
        worst = min(by_lag, key=lambda lag: by_lag[lag])
        mirrored = by_lag[worst] < -check.min_rank_correlation and -by_lag[worst] > rho
        verdict, reason = "no_signal", (
            f"yaw tracks image motion at rank correlation {rho:.2f} at best ({z:.1f} standard errors); "
            f"need {check.min_rank_correlation} and {check.min_z}"
            + (f" -- but it tracks it backwards ({by_lag[worst]:.2f}): yaw's sign is flipped" if mirrored else "")
        )
    elif best != expected_lag and confidence >= check.lag_confidence:
        off = abs(best - expected_lag)
        steps = f"{off} decision{'s' if off > 1 else ''}"
        verdict, reason = "misaligned", (
            f"yaw fits the pixels best at lag {best} ({rho:.2f}), not the expected {expected_lag} "
            f"({expected:.2f}), in {confidence:.0%} of resamples: "
            + (
                f"the image moves before the labelled turn, so each label is stamped {steps} late"
                if best < expected_lag
                else f"each label is stamped {steps} early, or the closed-loop delay really is {best} "
                "decisions (compare spike S3)"
            )
        )
    elif np.isnan(px_per_deg):
        # Fewer than 3 moving samples in the fitting range (all flicks, or all nudges): the timing half of the
        # check still stands, the scale half has nothing to say.
        verdict, reason = "ok", f"scale not checked: too few sampled {lo:.0f}-{hi:.0f} degree turns moved the view"
    elif px_per_deg <= 0:
        verdict, reason = "wrong_scale", (
            f"the image moves the wrong way or not at all on {lo:.0f}-{hi:.0f} degree turns "
            f"({px_per_deg:.2f} px per labelled degree)"
        )
    elif not check.fov_deg[0] <= fov <= check.fov_deg[1]:
        verdict, reason = "wrong_scale", (
            f"{px_per_deg:.2f} px per labelled degree implies a {fov:.0f} degree horizontal FOV, outside "
            f"{check.fov_deg[0]:.0f}-{check.fov_deg[1]:.0f}; at a typical {check.typical_fov_deg:.0f} degrees, "
            f"counts_per_degree should be about {_px_per_deg_at(check.typical_fov_deg, width) / px_per_deg:.2g}x "
            "what it is"
        )
    else:
        verdict, reason = "ok", None
    report.update(verdict=verdict, reasons=[reason] if reason else [])
    return report


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
