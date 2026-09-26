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

Every tick outside standby is written to a clip (`label_source="agent"`), so a run can be watched back and
its HUD crops parsed later like any recording; steps the policy did not act on are flagged bad, and standby
is left out altogether, so an hour of waiting costs no disk.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.capture import sleep_until
from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, ClipWriter

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
    """Reads the real mouse and keyboard (never our own virtual device) for two things: the kill key, and
    any sign the human has taken the controls back."""

    def __init__(self, source, config: PlayConfig):
        self.source, self.config = source, config
        self.last_human = -float("inf")

    def poll(self, now: float) -> str | None:
        """Drain what the human did since the last poll: "kill" or "toggle" if they pressed one of those
        keys (the kill key wins), otherwise None. Any other input marks the human as holding the controls."""
        events = self.source.drain(0.0, now)
        moved = [0, 0]
        toggles = 0
        for event in events:
            kind = event.get("type")
            if kind == "mouse":
                moved[0] += abs(event.get("dx", 0))
                moved[1] += abs(event.get("dy", 0))
            elif kind in ("key", "button") and event.get("down"):
                code = str(event.get("code", "")).lower()
                if code == self.config.kill_key:
                    return "kill"
                if code == self.config.toggle_key:
                    toggles += 1
                    continue
                self.last_human = now
        if max(moved) >= self.config.human_motion_counts:
            self.last_human = now
        return "toggle" if toggles % 2 else None  # two taps inside one tick cancel out

    def in_control(self, now: float) -> bool:
        return now - self.last_human < self.config.human_idle_s


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
) -> dict:
    """Run until the kill key, the policy's time limit, or an interrupt. Starts in standby: nothing is sent
    until the toggle key. `on_state(state)` hears every change -- "standby", "acting", a pause reason, and
    "stopped" -- for cues the player can hear over a fullscreen game, and must not block.

    Returns a summary of the run (also written into the clip)."""
    config = config or PlayConfig()
    dt = config.dt
    counts = {"acted": 0, "unfocused": 0, "frozen": 0, "human": 0, "overruns": 0, "toggles": 0}
    armed = False  # the human has handed over the controls with the toggle key
    state = "standby"
    ended = "time limit"
    fresh = True  # the next written step has no usable history before it
    smooth_look = config.look == "mean" and hasattr(agent, "last_look")
    motor = getattr(dispatcher, "motor", None)
    # A mouse motor smooths in continuous time on its own; smoothing per tick as well would only add lag.
    smoother = LookSmoother(1.0 if motor is not None else config.look_smoothing)
    sent_before = motor.sent_counts.copy() if motor is not None else None
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

    say(f"  standby: press {config.toggle_key.upper()} in the game to hand it the controls, "
        f"{config.kill_key.upper()} to quit")
    try:
        while True:
            k += 1
            deadline = t0 + k * dt
            now = clock()
            command = human.poll(now)
            if command == "kill":
                ended = f"kill key ({config.kill_key.upper()})"
                break
            if command == "toggle":
                armed = not armed
                counts["toggles"] += 1
                if not armed:
                    announce("standby")
            if not armed:
                fresh = True
                dispatcher.pump_until(deadline)
                continue
            frame = capture.read()
            stale = bool(getattr(capture, "last_stale", False))
            if human.in_control(now):
                reason = "human"
            elif not focus.is_focused():
                reason = "unfocused"
            elif stale:
                reason = "frozen"
            else:
                reason = None
            if reason is None:
                announce("acting")
                action = np.asarray(agent.act({"pixels": frame}), dtype=np.int64)
                look = smoother(agent.last_look) if smooth_look else None
                dispatcher.apply(action, now, dt, look_deg=look)
                if look is not None:
                    action = nearest_bins(action, look)
                counts["acted"] += 1
                flags = FLAG_CLIP_START if fresh else 0
                fresh = False
            else:
                announce(reason)
                counts[reason] += 1
                action = np.asarray(spec.NEUTRAL_ACTION, dtype=np.int64)
                flags = FLAG_BAD_STEP
                fresh = True
            if motor is not None:
                # Label what the motor actually sent during this tick, not what was asked of it.
                counts = motor.sent_counts - sent_before
                sent_before = motor.sent_counts.copy()
                if reason is None and look is not None:
                    look = (float(counts[0]) / motor.cpd, float(counts[1]) / motor.cpd)
                    action = nearest_bins(action, look)
            if writer is not None:
                sent = look if reason is None and look is not None else (0.0, 0.0)
                writer.add(frame, action, flags=flags, hud=getattr(capture, "last_hud", None),
                           yaw_deg=sent[0], pitch_deg=sent[1])
            if clock() > deadline + (config.overrun_factor - 1.0) * dt:
                counts["overruns"] += 1
            if counts["acted"] * dt >= config.max_seconds:
                break
            dispatcher.pump_until(deadline)
    except KeyboardInterrupt:
        ended = "interrupted"
    finally:
        dispatcher.release_all()
        summary = {**counts, "seconds": k * dt, "played_seconds": counts["acted"] * dt, "ended": ended,
                   "t0_mono": t0}
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
