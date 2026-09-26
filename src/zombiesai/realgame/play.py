"""A policy playing the real game: screen in, model, virtual mouse and keyboard out, fifteen times a second.

This is the recorder turned around, and it runs under the same discipline (absolute monotonic deadlines,
skip-never-catch-up), but the failure that matters here is different. A recorder that goes wrong writes a bad
clip. A player that goes wrong holds W down in your terminal. So most of this module is about *when not to
send input*, and every one of those rules ends in `release_all()`:

* **Only into the game.** Input goes to whatever window has focus, so before every action the focus guard
  asks the compositor which window that is. Anything else -- a workspace switch, a notification that took
  focus, the terminal -- and the player lets go of everything and waits.
* **Only while the picture is live.** A frozen or lost frame (the pause menu, a hidden workspace; see
  `demos.capture.FROZEN_READS`) means the policy would be acting on a picture that is not the game's, and
  clicking through a menu it cannot see.
* **Nothing until asked.** It starts in standby and only takes the controls when the human presses the
  toggle key (F7) in the game; F7 again puts it back in standby, F9 quits. A countdown to a start the player
  cannot see coming was the first design, and the first real run showed why not: the start raced the pause
  menu, and the fullscreen game hid the notification meant to announce it.
* **The human always wins.** Any other input from a real device hands control back: the player releases
  everything and stays out until the human has been idle for `human_idle_s`.
* **Always released on the way out**, whatever the way out was.

A policy that hears gets `hearing` (`demos.hearing.LiveAudio`): the audio thread runs all along, and each
acting tick asks it for the half second that ended when the frame was grabbed -- a copy and ~2 ms of numpy,
never a wait. Nothing about it is reset on a pause; see `LiveAudio` for why.

Every tick outside standby is written to a clip (`label_source="play"`), so a run can be watched back and
its HUD crops parsed later like any recording; standby is left out altogether, so an hour of waiting costs no
disk. Each step records who acted (`clips.ACTOR_*`, labels.npz["actor"]):

* **The policy's steps** carry the action it sent. They are there to watch, never to train on: a policy
  cloned from its own actions learns nothing.
* **The human's steps are corrections, and they are training data** -- the fix, in exactly the state the
  policy drove itself into, which no demo covers (HG-DAgger). The events `HumanWatch` reads to notice the
  takeover go through the recorder's own decoder (`inputs.InputFolder`, the same bindings, counts per degree
  and hold fraction), so a correction is labelled exactly as a demo step would be. The kill, toggle and mark
  keys are taken out first and are never a label.
* **Alignment is the recorder's**: the frame read at a tick is paired with the input over the window from
  that tick to the next, so a step is written one tick late, once its window has closed. The window's edges
  are the loop's own poll instants -- the deadlines whenever the loop keeps time, and still the truth when a
  tick runs late.
* **The first step of a takeover is the human's.** Its window holds their first touch: their reaction to
  what the policy was doing, which is the whole point. The frames behind it are the policy's, and meant to
  be -- that history is the state the correction answers.
* **The idle tail is not.** The policy takes the controls back only after `human_idle_s` of nothing, and
  those last untouched steps are the human waiting for it, not playing: they are rewritten `ACTOR_HUMAN_IDLE`
  when the takeover ends, and never trained on. A pause *inside* a takeover (the human moves again before
  handing back) is play and stays.
* Steps nobody acted on -- the game unfocused, the picture frozen -- are flagged bad, whoever held the
  controls.

The raw events of every written step's window go to the clip's `inputs.jsonl`, and each step's poll instant
to labels.npz["t_mono"], so a correction's label can be rebuilt from the log like a demo's.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.clips import (
    ACTOR_HUMAN,
    ACTOR_HUMAN_IDLE,
    ACTOR_NONE,
    ACTOR_POLICY,
    FLAG_BAD_STEP,
    FLAG_CLIP_START,
    ClipWriter,
)
from zombiesai.demos.inputs import InputConfig, InputFolder, Labels, actions_from, label_confidence

KILL_KEY = "f9"  # WaW binds nothing to it; F8 is the recorder's mark key
TOGGLE_KEY = "f7"  # nor to this one (F5, F10 and F12 are taken)
GAME_WINDOW_CLASS = "steam_app_10090"


@dataclass(frozen=True)
class PlayConfig:
    max_seconds: float = 180.0  # of the policy holding the controls; standby does not count
    kill_key: str = KILL_KEY
    toggle_key: str = TOGGLE_KEY
    # How long the human must leave the controls alone before the policy takes them back.
    human_idle_s: float = 1.5
    # Mouse motion below this many counts in a tick is a hand resting on the mouse, not a human taking over.
    human_motion_counts: int = 3
    overrun_factor: float = 1.5
    # "mean": turn by the probability-weighted mean of the look bins, smoothed over time -- what a person's
    # hand does. "sample": draw a bin every tick like every other head, which jumps between 0, +6 and -2
    # degrees 15 times a second and was, in the first live run, hard to watch.
    look: str = "mean"
    # How far each tick's look moves towards the policy's new one (exponential smoothing). 0.35 is a time
    # constant of about two decisions, 130 ms: gentle enough to take the jitter out, quick enough to aim.
    look_smoothing: float = 0.35

    @property
    def dt(self) -> float:
        return 1.0 / spec.DECISION_HZ


class HyprlandFocus:
    """Which window has focus, asked of Hyprland over its IPC socket every tick (well under a millisecond,
    where spawning `hyprctl` would cost several). The game is recognised by its Proton window class."""

    def __init__(self, window_class: str = GAME_WINDOW_CLASS, title: str = "Call of Duty"):
        signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        if not signature:
            raise RuntimeError("not running under Hyprland: no HYPRLAND_INSTANCE_SIGNATURE")
        self.path = str(Path(runtime) / "hypr" / signature / ".socket.sock")
        self.window_class, self.title = window_class, title

    def active_window(self) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.05)
            sock.connect(self.path)
            sock.sendall(b"j/activewindow")
            chunks = []
            while chunk := sock.recv(65536):
                chunks.append(chunk)
        text = b"".join(chunks).decode(errors="replace").strip()
        return json.loads(text) if text.startswith("{") else {}

    def is_focused(self) -> bool:
        try:
            window = self.active_window()
        except (OSError, ValueError):
            return False  # when unsure, the answer that sends no input
        return window.get("class") == self.window_class or self.title in str(window.get("title", ""))


class LookSmoother:
    """Exponential smoothing of the look command, in degrees per decision."""

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"look smoothing must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.look = np.zeros(2)

    def __call__(self, target) -> tuple[float, float]:
        self.look += self.alpha * (np.asarray(target, dtype=np.float64) - self.look)
        return float(self.look[0]), float(self.look[1])

    def reset(self) -> None:
        self.look[:] = 0.0


def nearest_bins(action: np.ndarray, look: tuple[float, float]) -> np.ndarray:
    """The action as sent: its look heads replaced by the bins nearest the continuous turn that went out, so
    the recording labels what the game was actually told rather than the sample that was thrown away."""
    out = np.array(action, copy=True)
    out[spec.YAW] = int(np.argmin(np.abs(np.asarray(spec.YAW_BINS_DEG) - look[0])))
    out[spec.PITCH] = int(np.argmin(np.abs(np.asarray(spec.PITCH_BINS_DEG) - look[1])))
    return out


class HumanWatch:
    """Reads the real mouse and keyboard (never our own virtual device) for three things: the kill and toggle
    keys, any sign the human has taken the controls back, and -- through the demo recorder's own decoder --
    what they did with them, one window per poll.

    The decoder is fed every window, the policy's and standby's included, so it always knows which keys are
    down: a key pressed before a takeover and still held is a held key in the first correction, as it would be
    in a demo. Holding a control is the human playing too (running down a corridor with W held sends nothing
    after the press), so a held control keeps them in control as surely as moving does."""

    def __init__(self, source, config: PlayConfig, input_config: InputConfig | None = None):
        self.source, self.config = source, config
        self.input = input_config or InputConfig()
        bound = {str(code).lower() for code in self.input.bindings}
        self.kill_key, self.toggle_key = config.kill_key.lower(), config.toggle_key.lower()
        commands = {self.kill_key, self.toggle_key}
        if commands & bound:
            # A command key that is also a control would be the game's input and a label at once.
            raise ValueError(f"the kill/toggle keys {sorted(commands & bound)} are bound to game controls")
        # Never a label and never a takeover: the loop's own commands, and the recorder's mark key, which
        # means nothing here (a stray tap must not become a step of standing still).
        self.reserved = commands | ({self.input.mark_key} if self.input.mark_key else set())
        self.folder = InputFolder(self.input)
        self.last_human = -float("inf")
        self.last_poll: float | None = None
        self.events: list[dict] = []  # everything the last poll drained, raw, for the clip's input log
        self.labels: Labels | None = None  # the human's input over the last poll's window, as an action
        self.touched = False  # that window held a sign of the human: a press, real motion, a held control

    def poll(self, now: float) -> str | None:
        """Drain what the human did since the last poll: "kill" or "toggle" if they pressed one of those
        keys (the kill key wins), otherwise None. The window [last poll, now) is also decoded into `labels`,
        and any other input marks the human as holding the controls."""
        self.events = self.source.drain(0.0, now)
        start = now - self.config.dt if self.last_poll is None else self.last_poll
        self.last_poll = now
        moved = [0, 0]
        toggles, kill, pressed = 0, False, False
        controls = []
        for event in self.events:
            kind = event.get("type")
            code = str(event.get("code", "")).lower()
            if kind in ("key", "button") and code in self.reserved:
                if event.get("down"):
                    kill |= code == self.kill_key
                    toggles += code == self.toggle_key
                continue
            controls.append(event)
            if kind == "mouse":
                moved[0] += abs(event.get("dx", 0))
                moved[1] += abs(event.get("dy", 0))
            elif kind in ("key", "button") and event.get("down"):
                pressed = True
        held, presses, counts = self.folder.feed(controls, start, now)
        self.labels = actions_from(held, presses, counts, self.input)
        self.touched = pressed or max(moved) >= self.config.human_motion_counts or bool(np.any(held > 0))
        if self.touched:
            self.last_human = now
        if kill:
            return "kill"
        return "toggle" if toggles % 2 else None  # two taps inside one tick cancel out

    def in_control(self, now: float) -> bool:
        return now - self.last_human < self.config.human_idle_s


@dataclass
class _Tick:
    """A step whose input window is still open: it is written once the next poll has closed it."""

    frame: np.ndarray
    hud: dict | None
    actor: int
    flags: int
    action: np.ndarray  # the policy's, as sent; replaced by the human's label if they acted
    look: tuple[float, float] | None  # the continuous turn the policy sent, if it sent one
    sent: np.ndarray | None  # the mouse motor's running total when the tick began
    t: float  # the poll instant that opened the window (the frame was read just after)


def play(
    capture,
    agent,
    dispatcher,
    *,
    focus,
    human,
    config: PlayConfig | None = None,
    writer: ClipWriter | None = None,
    clock=time.monotonic,
    say=print,
    on_state=None,
    hearing=None,
) -> dict:
    """Run until the kill key, the policy's time limit, or an interrupt. Starts in standby: nothing is sent
    until the toggle key. `on_state(state)` hears every change -- "standby", "acting", a pause reason, and
    "stopped" -- for cues the player can hear over a fullscreen game, and must not block.

    `writer`, if given, must be a label_source="play" clip: its steps are labelled by whoever acted.
    Returns a summary of the run (also written into the clip)."""
    config = config or PlayConfig()
    if writer is not None and writer.label_source != "play":
        # Any other source would make the policy's own steps look like demonstrations to the trainer.
        raise ValueError(f"a play run is written as label_source='play', not {writer.label_source!r}")
    dt = config.dt
    counts = {"acted": 0, "unfocused": 0, "frozen": 0, "human": 0, "overruns": 0, "toggles": 0,
              "corrections": 0, "human_idle": 0}
    armed = False  # the human has handed over the controls with the toggle key
    state = "standby"
    ended = "time limit"
    fresh = True  # the next written step has no usable history before it
    smooth_look = config.look == "mean" and hasattr(agent, "last_look")
    motor = getattr(dispatcher, "motor", None)
    # A mouse motor smooths in continuous time on its own; smoothing per tick as well would only add lag.
    smoother = LookSmoother(1.0 if motor is not None else config.look_smoothing)
    log = open(writer.path / "inputs.jsonl", "a", buffering=1) if writer is not None else None
    pending: _Tick | None = None
    n_written = 0
    idle_run: list[tuple[int, bool]] = []  # (step, was it usable) for the human's untouched steps so far
    t0 = clock()
    k = 0

    def announce(new: str) -> None:
        nonlocal state
        if new == state:
            return
        if state == "acting":
            # Let go of everything before anything else: a held key must not outlive the reason to hold it.
            # The frame stack is stale too, so the policy starts fresh when it comes back.
            dispatcher.release_all()
            agent.reset()
            smoother.reset()  # a turn in progress does not carry across a pause
        state = new
        say(f"  [{k * dt:7.1f}s] {state_text(new, config)}")
        if on_state is not None:
            on_state(new)

    def hand_back() -> None:
        """The takeover is over: the untouched steps at its end were the human waiting, not playing."""
        for step, was_usable in idle_run:
            counts["corrections"] -= was_usable
            counts["human_idle"] += 1
            if writer is not None:
                writer.amend(step, actor=ACTOR_HUMAN_IDLE)
        idle_run.clear()

    def settle(tick: _Tick, closed: bool) -> None:
        """Label a tick from its now-closed input window and write it. `closed=False` is the last tick of a
        run that ended before its window did: nothing is known of what the human did in it."""
        nonlocal writer, log, n_written
        actor, action, confidence, look = tick.actor, tick.action, 1.0, tick.look or (0.0, 0.0)
        touched = closed and human.touched
        if actor == ACTOR_POLICY and touched:
            actor = ACTOR_HUMAN  # the takeover began inside this window: the human's first touch is here
        if actor == ACTOR_HUMAN:
            if closed:
                labels = human.labels
                action = labels.actions[0]
                confidence = float(label_confidence(labels, human.input)[0])
                look = (float(labels.yaw_deg[0]), float(labels.pitch_deg[0]))
            else:
                action, look = np.asarray(spec.NEUTRAL_ACTION), (0.0, 0.0)
            usable = not tick.flags & FLAG_BAD_STEP
            counts["corrections"] += usable
            if touched:
                idle_run.clear()  # they are still at it: any pause before this was play
            else:
                idle_run.append((n_written, usable))
        else:
            hand_back()
            if actor == ACTOR_POLICY and motor is not None and tick.look is not None:
                # Label what the motor actually sent over this tick's window, not what was asked of it.
                emitted = motor.sent_counts - tick.sent
                look = (float(emitted[0]) / motor.cpd, float(emitted[1]) / motor.cpd)
                action = nearest_bins(action, look)
        n_written += 1
        if writer is None:
            return
        if closed and log is not None:
            for event in human.events:
                log.write(json.dumps(event) + "\n")
        try:
            writer.add(tick.frame, action, confidence=confidence, yaw_deg=look[0], pitch_deg=look[1],
                       flags=tick.flags, hud=tick.hud,
                       extras={"actor": np.uint8(actor), "t_mono": np.float64(tick.t)})
        except ValueError as error:
            # The game came back at another resolution, so its HUD crops no longer fit this clip.
            # Losing the rest of the recording is better than losing the controls mid-game.
            say(f"  recording stopped ({error}); still playing")
            hand_back()
            writer.close(summary={**counts, "ended": "recording stopped: HUD shape changed"})
            writer = None
            log.close()
            log = None

    say(f"  standby: press {config.toggle_key.upper()} in the game to hand it the controls, "
        f"{config.kill_key.upper()} to quit")
    try:
        while True:
            k += 1
            deadline = t0 + k * dt
            now = clock()
            command = human.poll(now)  # closes the pending tick's input window
            if pending is not None:
                settle(pending, closed=True)
                pending = None
            if command == "kill":
                ended = f"kill key ({config.kill_key.upper()})"
                break
            if command == "toggle":
                armed = not armed
                counts["toggles"] += 1
                if not armed:
                    announce("standby")
            if not armed:
                hand_back()
                fresh = True
                dispatcher.pump_until(deadline)
                continue
            t_frame = clock()
            frame = capture.read()
            stale = bool(getattr(capture, "last_stale", False))
            focused = focus.is_focused()
            if human.in_control(now):
                reason = "human"
            elif not focused:
                reason = "unfocused"
            elif stale:
                reason = "frozen"
            else:
                reason = None
            look = None
            if reason is None:
                announce("acting")
                obs = {"pixels": frame}
                if hearing is not None:
                    obs["audio"], obs["audio_mask"] = hearing.observe(t_frame)
                action = np.asarray(agent.act(obs), dtype=np.int64)
                look = smoother(agent.last_look) if smooth_look else None
                dispatcher.apply(action, now, dt, look_deg=look)
                if look is not None:
                    action = nearest_bins(action, look)
                counts["acted"] += 1
                actor, flags = ACTOR_POLICY, FLAG_CLIP_START if fresh else 0
                fresh = False
            else:
                announce(reason)
                counts[reason] += 1
                action = np.asarray(spec.NEUTRAL_ACTION, dtype=np.int64)
                # The human playing a live, focused game is a step like any other, and its history is real;
                # anything else is nobody's play, whoever held the controls.
                live = reason == "human" and focused and not stale
                actor = ACTOR_HUMAN if reason == "human" else ACTOR_NONE
                flags = (FLAG_CLIP_START if fresh else 0) if live else FLAG_BAD_STEP
                fresh = not live
            if getattr(capture, "has_frame", True):
                pending = _Tick(frame, getattr(capture, "last_hud", None), actor, flags, action, look,
                                motor.sent_counts.copy() if motor is not None else None, now)
            if clock() > deadline + (config.overrun_factor - 1.0) * dt:
                counts["overruns"] += 1
            if counts["acted"] * dt >= config.max_seconds:
                break
            dispatcher.pump_until(deadline)
    except KeyboardInterrupt:
        ended = "interrupted"
    finally:
        dispatcher.release_all()
        if pending is not None:
            settle(pending, closed=False)
        hand_back()
        summary = {**counts, "seconds": k * dt, "played_seconds": counts["acted"] * dt,
                   "correction_seconds": counts["corrections"] * dt, "ended": ended, "t0_mono": t0}
        if hearing is not None:
            summary["hearing"] = hearing.stats()
        if log is not None:
            log.close()
        if writer is not None:
            writer.close(summary=summary)
        if on_state is not None:
            on_state("stopped")
    say(f"stopped: {ended}")
    return summary


def state_text(state: str, config: PlayConfig) -> str:
    return {
        "standby": f"standby -- {config.toggle_key.upper()} hands it the controls again",
        "acting": f"AI playing -- {config.toggle_key.upper()} for standby, {config.kill_key.upper()} to quit",
    }.get(state) or "paused: " + reason_text(state, config)


def reason_text(reason: str, config: PlayConfig) -> str:
    return {
        "human": f"you have the controls (idle {config.human_idle_s:g}s to hand back; {config.kill_key.upper()} stops)",
        "unfocused": "the game is not the focused window",
        "frozen": "the picture is frozen or lost (pause menu, hidden workspace)",
    }[reason]
