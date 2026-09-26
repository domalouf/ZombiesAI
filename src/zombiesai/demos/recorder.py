"""The demo recorder: a person plays, and what they saw and did comes out as a labelled clip.

The loop is the same discipline the real environment will run under, and for the same reasons (PLAN.md,
"Real-time loop"):

* **Absolute monotonic deadlines**, never `sleep(period)`, so a slow step does not push every later step.
* **Skip, hold, drop -- never catch up.** A decision that ran late is flagged `bad_step` and excluded from
  training rather than quietly shortening the next one.
* **Alignment is explicit.** The frame captured at deadline k is paired with the input collected over
  [t_k, t_k+1) -- what the player could see, and what they did about it. Off by one here is the bug that
  looks like "the model is bad at aiming" for a week.
* **Not every second of a session is play.** The player taps the mark key (F8) going into a menu, the pause
  screen, a loading screen or the game-over card, and again coming back; those steps are kept, flagged
  FLAG_NOT_PLAYING, and left out of training. Which step a press lands on is `inputs.PlayMarker`'s rule.

The raw input log is written alongside the clip, so the labels can be recomputed later under different
bindings, a different `counts_per_degree`, or different bins, without asking anyone to play again.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.capture import sleep_until
from zombiesai.demos.clips import (
    FLAG_BAD_STEP,
    FLAG_CLIP_START,
    FLAG_NOT_PLAYING,
    ClipWriter,
    load_clip,
)
from zombiesai.demos.inputs import (
    InputConfig,
    InputFolder,
    PlayMarker,
    actions_from,
    label_confidence,
    not_playing,
    read_log,
    yaw_flow_agreement,
)


@dataclass(frozen=True)
class RecorderConfig:
    max_steps: int = 18_000  # 20 minutes at 15 Hz
    max_seconds: float | None = None
    overrun_factor: float = 1.5  # a decision this much longer than the period is flagged, per the plan
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
    marker = PlayMarker(config.input.mark_key)
    overruns = 0
    idle = 0  # steps marked not playing
    was_idle = False
    t0 = time.monotonic()
    try:
        pending = frame_source.read()
        # A source that cuts full-resolution HUD crops from each grab exposes the latest set as `last_hud`.
        pending_hud = getattr(frame_source, "last_hud", None)
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
            held, presses, counts = folder.feed(events, start, end)
            labels = actions_from(held, presses, counts, config.input)
            late = overshoot > (config.overrun_factor - 1.0) * dt
            overruns += late
            # The same fold `requantize` runs offline (`inputs.not_playing`), fed the same window, and the
            # same flag rule as `clips.with_play_marks` -- so the marking is recomputable from the log.
            playing_before = marker.playing
            is_idle = marker.feed(events)
            idle += is_idle
            if marker.playing != playing_before:
                state = "playing again" if marker.playing else "NOT PLAYING"
                print(f"  [{k * dt:6.0f}s] {state} ({config.input.mark_key} toggles)", flush=True)
            flags = (
                (FLAG_BAD_STEP if late else 0)
                | (FLAG_NOT_PLAYING if is_idle else 0)
                | (FLAG_CLIP_START if was_idle and not is_idle else 0)
            )
            was_idle = is_idle
            writer.add(
                pending,
                labels.actions[0],
                confidence=float(label_confidence(labels, config.input)[0]),
                yaw_deg=float(labels.yaw_deg[0]),
                pitch_deg=float(labels.pitch_deg[0]),
                flags=flags,
                hud=pending_hud,
            )
            pending, pending_hud = frame, getattr(frame_source, "last_hud", None)
            if progress_every and k % progress_every == 0:
                print(f"  {k} decisions ({k * dt:.0f}s), {overruns} overruns", flush=True)
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
                "not_playing_steps": int(idle),
                "not_playing_seconds": idle * dt,
                "mark_key": config.input.mark_key,
                # The clock the input log is stamped on, so the labels can be rebuilt against the same
                # decision boundaries the recorder used rather than guessed ones.
                "t0_mono": t0,
                "notes": config.notes,
            }
        )
    return path


def requantize(clip_dir: str | Path, config: InputConfig, *, dt: float | None = None) -> np.ndarray:
    """Recompute a recorded demo's labels from its raw input log and write them back.

    This is what keeping the log buys: a corrected `counts_per_degree`, a rebound key, or a change to the
    yaw bins costs a second of arithmetic instead of another evening at the game. The not-playing marking
    is re-derived from the same log under `config.mark_key` -- so a clip recorded with a different mark key
    wants that key passed here, and `mark_key=None` clears the marking. Every other flag is carried over.
    """
    from zombiesai.demos.clips import attach_labels, with_play_marks

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
    flags = with_play_marks(clip.flags, not_playing(events, t0, clip.n_steps, config, dt))
    attach_labels(
        clip_dir,
        labels.actions,
        label_confidence(labels, config),
        label_source="input_log",
        flags=flags,
        yaw_deg=labels.yaw_deg,
        pitch_deg=labels.pitch_deg,
        detail={
            "requantized_unix": time.time(),
            "counts_per_degree": config.counts_per_degree,
            "mark_key": config.mark_key,
        },
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
    idle = (clip.flags & FLAG_NOT_PLAYING) != 0
    # Menu time says nothing about how someone plays; describe the hands over the steps that were play.
    played = clip.actions[~idle] if (~idle).any() else clip.actions
    return {
        "steps": clip.n_steps,
        "seconds": clip.n_steps / spec.DECISION_HZ,
        "not_playing_steps": int(idle.sum()),
        "not_playing_seconds": float(idle.sum()) / spec.DECISION_HZ,
        "overrun_rate": summary.get("overrun_rate"),
        "mean_confidence": float(clip.confidence.mean()),
        "clamped_looks": float((np.abs(clip.labels["yaw_deg"]) > max(spec.YAW_BINS_DEG)).mean()),
        "yaw_flow": flow,
        "behaviour": stats.behaviour_stats(played),
    }
