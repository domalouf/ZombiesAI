"""The demo recorder: a person plays, and what they saw and did comes out as a labelled clip.

The loop is the same discipline the real environment will run under, and for the same reasons (PLAN.md,
"Real-time loop"):

* **Absolute monotonic deadlines**, never `sleep(period)`, so a slow step does not push every later step.
* **Skip, hold, drop -- never catch up.** A decision that ran late is flagged `bad_step` and excluded from
  training rather than quietly shortening the next one.
* **Alignment is explicit.** The frame captured at deadline k is paired with the input collected over
  [t_k, t_k+1) -- what the player could see, and what they did about it. Off by one here is the bug that
  looks like "the model is bad at aiming" for a week.
* **A lost window costs steps, not the session.** While the game window cannot be grabbed the source repeats
  its last good frame and says so; those steps are flagged `bad_step`, and the recording carries on until the
  window is back -- or stops cleanly once it has been gone for `max_outage_seconds`.

The raw input log is written alongside the clip, so the labels can be recomputed later under different
bindings, a different `counts_per_degree`, or different bins, without asking anyone to play again.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.capture import CaptureLost, sleep_until
from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, ClipWriter, load_clip
from zombiesai.demos.inputs import (
    InputConfig,
    InputFolder,
    actions_from,
    label_confidence,
    read_log,
    yaw_flow_agreement,
)


@dataclass(frozen=True)
class RecorderConfig:
    max_steps: int = 18_000  # 20 minutes at 15 Hz
    max_seconds: float | None = None
    overrun_factor: float = 1.5  # a decision this much longer than the period is flagged, per the plan
    # How long the capture may keep repeating a stale frame before the recording gives up. Long enough to
    # glance at another workspace; short enough that a closed game does not leave the recorder running on.
    max_outage_seconds: float = 30.0
    # A human plays in real time and the loop must wait for them. A sim source produces its own time, so
    # waiting on the wall clock would only make CI slow.
    realtime: bool = True
    input: InputConfig = None  # type: ignore[assignment]
    notes: str = ""

    def __post_init__(self):
        if self.input is None:
            object.__setattr__(self, "input", InputConfig())

    @property
    def dt(self) -> float:
        return 1.0 / spec.DECISION_HZ


def record(
    frame_source,
    input_source,
    out_dir: str | Path,
    config: RecorderConfig | None = None,
    *,
    stop=None,
    progress_every: int = 150,
) -> Path:
    """Record one demo into `out_dir`. `stop()` may return True to end early (a hotkey, a finished sim)."""
    config = config or RecorderConfig()
    dt = config.dt
    writer = ClipWriter(
        out_dir,
        source={
            **(frame_source.describe() if hasattr(frame_source, "describe") else {"kind": "unknown"}),
            "recorded_unix": time.time(),
        },
        label_source="input_log",
        config={"recorder": asdict(config), "decision_hz": spec.DECISION_HZ},
    )
    log = open(Path(out_dir) / "inputs.jsonl", "a", buffering=1)
    folder = InputFolder(config.input)
    overruns = stale_steps = outages = outage_run = 0
    capture_lost: str | None = None
    t0 = time.monotonic()
    try:
        pending = frame_source.read()
        # A source that cuts full-resolution HUD crops from each grab exposes the latest set as `last_hud`,
        # and one that can lose its window says whether the frame it returned is a repeat (`last_stale`).
        # Both describe the frame, so both travel with `pending` to the step that frame is paired with.
        pending_hud = getattr(frame_source, "last_hud", None)
        pending_stale = bool(getattr(frame_source, "last_stale", False))
        resumed = False  # the pending frame is the first good one after an outage
        for k in range(1, config.max_steps + 1):
            if config.max_seconds and k * dt > config.max_seconds:
                break
            overshoot = sleep_until(t0 + k * dt) if config.realtime else 0.0
            start, end = t0 + (k - 1) * dt, t0 + k * dt
            events = input_source.drain(start, end)
            for event in events:
                log.write(json.dumps(event) + "\n")
            try:
                frame = frame_source.read()
            except StopIteration:
                break
            except CaptureLost as error:
                capture_lost = str(error)
                print(f"  capture lost for good at {k * dt:.0f}s, stopping: {error}", flush=True)
                break
            stale = bool(getattr(frame_source, "last_stale", False))
            if stale and not outage_run:
                outages += 1
                reason = getattr(frame_source, "stale_reason", None) or "no reason given"
                print(f"  WARNING capture lost at {k * dt:.0f}s ({reason}); flagging steps bad until it is back",
                      flush=True)
            elif outage_run and not stale:
                print(f"  capture back at {k * dt:.0f}s after {outage_run * dt:.1f}s", flush=True)
            held, presses, counts = folder.feed(events, start, end)
            labels = actions_from(held, presses, counts, config.input)
            late = overshoot > (config.overrun_factor - 1.0) * dt
            overruns += late
            stale_steps += pending_stale
            writer.add(
                pending,
                labels.actions[0],
                confidence=float(label_confidence(labels, config.input)[0]),
                yaw_deg=float(labels.yaw_deg[0]),
                pitch_deg=float(labels.pitch_deg[0]),
                # The first good frame after an outage has only repeats behind it, so a frame stack must not
                # reach back past it -- the same situation as the first step of a clip.
                flags=(FLAG_BAD_STEP if late or pending_stale else 0) | (FLAG_CLIP_START if resumed else 0),
                hud=pending_hud,
            )
            resumed = outage_run > 0 and not stale
            outage_run = outage_run + 1 if stale else 0
            pending, pending_hud, pending_stale = frame, getattr(frame_source, "last_hud", None), stale
            if progress_every and k % progress_every == 0:
                print(f"  {k} decisions ({k * dt:.0f}s), {overruns} overruns, {stale_steps} stale", flush=True)
            if outage_run and outage_run * dt >= config.max_outage_seconds:
                reason = getattr(frame_source, "stale_reason", None) or "no reason given"
                capture_lost = f"no capture for {outage_run * dt:.0f}s: {reason}"
                print(f"  capture has been lost for {outage_run * dt:.0f}s, stopping the recording ({reason})",
                      flush=True)
                break
            if stop is not None and stop():
                break
    finally:
        log.close()
        frame_source.close()
        input_source.close()
        seconds = writer.n_steps * dt
        path = writer.close(
            summary={
                "seconds": seconds,
                "overruns": int(overruns),
                "overrun_rate": overruns / max(writer.n_steps, 1),
                # Steps whose frame was a repeat because the window could not be grabbed (all flagged bad),
                # how many separate outages they came from, and what ended the recording if capture did.
                "stale_steps": int(stale_steps),
                "stale_rate": stale_steps / max(writer.n_steps, 1),
                "capture_outages": int(outages),
                "capture_lost": capture_lost,
                # The clock the input log is stamped on, so the labels can be rebuilt against the same
                # decision boundaries the recorder used rather than guessed ones.
                "t0_mono": t0,
                "notes": config.notes,
            }
        )
    return path


def wait_to_start(
    probe,
    *,
    timeout: float,
    countdown: float = 3.0,
    poll: float = 0.5,
    discard=None,
    say=print,
    notify=None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> bool:
    """Hold the start of a recording until `probe()` says the game can be captured, then count down.

    Returns True the moment recording should begin, False if `timeout` seconds pass with nothing capturable.
    The game usually lives on another workspace from the terminal that launched the recorder, and grabbing a
    window nobody can see fails -- so rather than make someone race from the terminal to the game, the
    recorder waits for them to get there. The countdown is for the player, whose hands are not on the
    controls yet the instant the window appears; `probe()` is asked again at the end of it, and a window that
    went away in the meantime (a quick alt-tab back) sends it back to waiting rather than recording black.

    `discard()` is called on every tick and once more immediately before returning True. It exists for the
    input source: `drain()` hands back everything since the previous call, and `InputFolder` clamps events
    from before a decision into it -- so without this, a minute of mouse movement spent reaching the game
    would land in the first label as one enormous flick. A key held down across the start is lost with the
    rest (its press was discarded), which is why there is a countdown to let go on.

    `clock` and `sleep` are injectable so the logic is tested without anybody waiting.
    """
    discard = discard or (lambda: None)
    deadline = clock() + timeout
    while True:
        if not probe():
            say(f"waiting for the game window to be on screen -- switch to it (giving up after {timeout:.0f}s)")
            while True:
                discard()
                if clock() >= deadline:
                    return False
                sleep(poll)
                if probe():
                    break
        ticks = math.ceil(countdown)
        if ticks:
            say(f"game window is on screen; recording in {countdown:g}s")
            if notify is not None:
                notify(f"recording in {countdown:g}s")
        for n in range(ticks, 0, -1):
            say(f"  {n}...\a")  # the bell, for anyone who left the terminal where they can hear it
            discard()
            sleep(countdown / ticks)
        if probe():
            discard()
            return True
        say("the game window left the screen during the countdown")


def requantize(clip_dir: str | Path, config: InputConfig, *, dt: float | None = None) -> np.ndarray:
    """Recompute a recorded demo's labels from its raw input log and write them back.

    This is what keeping the log buys: a corrected `counts_per_degree`, a rebound key, or a change to the
    yaw bins costs a second of arithmetic instead of another evening at the game.
    """
    from zombiesai.demos.clips import attach_labels

    clip_dir = Path(clip_dir)
    clip = load_clip(clip_dir)
    events = read_log(clip_dir / "inputs.jsonl")
    if not events:
        raise ValueError(f"{clip_dir} has no inputs.jsonl to re-quantize")
    dt = dt or 1.0 / spec.DECISION_HZ
    # Step k of the clip covers [t0 + k*dt, t0 + (k+1)*dt) on the recorder's own clock. Falling back to the
    # first event's timestamp would shift every label by up to one decision, which is exactly the misalignment
    # the yaw-versus-flow check exists to catch -- so prefer the recorded t0 and say so when it is missing.
    t0 = clip.manifest.get("summary", {}).get("t0_mono")
    if t0 is None:
        t0 = min(e["t"] for e in events)
    from zombiesai.demos.inputs import quantize

    labels = quantize(events, t0, clip.n_steps, config, dt)
    attach_labels(
        clip_dir,
        labels.actions,
        label_confidence(labels, config),
        label_source="input_log",
        flags=clip.flags,
        yaw_deg=labels.yaw_deg,
        pitch_deg=labels.pitch_deg,
        detail={"requantized_unix": time.time(), "counts_per_degree": config.counts_per_degree},
    )
    return labels.actions


def quality_report(clip, sample: int = 400) -> dict:
    """Check a finished recording before trusting it: are the labels aligned with the pixels?

    The cheap version of the plan's cross-validation. Yaw must correlate with the direction the image
    actually moved; below about 0.9 the input log and the capture are out of step in time, and no amount of
    training absorbs that. (The other half, firing against the magazine counter, needs the HUD parser M4
    brings, so it is not here yet.)
    """
    from zombiesai.demos import stats

    if not clip.labelled:
        raise ValueError(f"{clip.path} has no labels to check")
    summary = clip.manifest.get("summary", {})
    flow = yaw_flow_agreement(clip.labels["yaw_deg"], clip.frames, sample=sample)
    return {
        "steps": clip.n_steps,
        "seconds": clip.n_steps / spec.DECISION_HZ,
        "overrun_rate": summary.get("overrun_rate"),
        "stale_steps": summary.get("stale_steps"),
        "capture_outages": summary.get("capture_outages"),
        "capture_lost": summary.get("capture_lost"),
        "mean_confidence": float(clip.confidence.mean()),
        "clamped_looks": float((np.abs(clip.labels["yaw_deg"]) > max(spec.YAW_BINS_DEG)).mean()),
        "yaw_flow": flow,
        "behaviour": stats.behaviour_stats(clip.actions),
    }
