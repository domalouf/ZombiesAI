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
* **The human always wins.** Any input from a real device hands control back: the player releases
  everything and stays out until the human has been idle for `human_idle_s`. The kill key (F9) ends it.
* **Always released on the way out**, whatever the way out was.

Every tick is written to a clip (`label_source="agent"`), acted or not, so a run can be watched back and its
HUD crops parsed later like any recording; steps the policy did not act on are flagged bad.
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
GAME_WINDOW_CLASS = "steam_app_10090"


@dataclass(frozen=True)
class PlayConfig:
    max_seconds: float = 180.0
    kill_key: str = KILL_KEY
    # How long the human must leave the controls alone before the policy takes them back.
    human_idle_s: float = 1.5
    # Mouse motion below this many counts in a tick is a hand resting on the mouse, not a human taking over.
    human_motion_counts: int = 3
    overrun_factor: float = 1.5

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


class HumanWatch:
    """Reads the real mouse and keyboard (never our own virtual device) for two things: the kill key, and
    any sign the human has taken the controls back."""

    def __init__(self, source, config: PlayConfig):
        self.source, self.config = source, config
        self.last_human = -float("inf")

    def poll(self, now: float) -> bool:
        """Drain what the human did since the last poll; True if they pressed the kill key."""
        events = self.source.drain(0.0, now)
        moved = [0, 0]
        for event in events:
            kind = event.get("type")
            if kind == "mouse":
                moved[0] += abs(event.get("dx", 0))
                moved[1] += abs(event.get("dy", 0))
            elif kind in ("key", "button") and event.get("down"):
                if str(event.get("code", "")).lower() == self.config.kill_key:
                    return True
                self.last_human = now
        if max(moved) >= self.config.human_motion_counts:
            self.last_human = now
        return False

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
) -> dict:
    """Run the policy until the time limit, the kill key, or a capture that is gone for good.

    Returns a summary of the run (also written into the clip)."""
    config = config or PlayConfig()
    dt = config.dt
    counts = {"acted": 0, "unfocused": 0, "frozen": 0, "human": 0, "overruns": 0}
    state = "acting"
    ended = "time limit"
    t0 = clock()
    k = 0
    try:
        while (k := k + 1) * dt <= config.max_seconds:
            deadline = t0 + k * dt
            now = clock()
            if human.poll(now):
                ended = f"kill key ({config.kill_key.upper()})"
                break
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
                action = np.asarray(agent.act({"pixels": frame}), dtype=np.int64)
                dispatcher.apply(action, now, dt)
                counts["acted"] += 1
                flags = FLAG_CLIP_START if state != "acting" else 0
                if state != "acting":
                    say(f"  [{k * dt:6.1f}s] playing")
                state = "acting"
            else:
                if state == "acting":
                    # Let go of everything before anything else: a held key must not outlive the reason to
                    # hold it. The frame stack is stale too, so the policy starts fresh when it comes back.
                    dispatcher.release_all()
                    agent.reset()
                    say(f"  [{k * dt:6.1f}s] paused: {reason_text(reason, config)}")
                state = reason
                counts[reason] += 1
                action = np.asarray(spec.NEUTRAL_ACTION, dtype=np.int64)
                flags = FLAG_BAD_STEP
            if writer is not None:
                writer.add(frame, action, flags=flags, hud=getattr(capture, "last_hud", None))
            if clock() > deadline + (config.overrun_factor - 1.0) * dt:
                counts["overruns"] += 1
            dispatcher.pump_until(deadline)
    except KeyboardInterrupt:
        ended = "interrupted"
    finally:
        dispatcher.release_all()
        summary = {**counts, "steps": k - 1, "seconds": (k - 1) * dt, "ended": ended, "t0_mono": t0}
        if writer is not None:
            writer.close(summary=summary)
    say(f"stopped: {ended}")
    return summary


def reason_text(reason: str, config: PlayConfig) -> str:
    return {
        "human": f"you have the controls (idle {config.human_idle_s:g}s to hand back; {config.kill_key.upper()} stops)",
        "unfocused": "the game is not the focused window",
        "frozen": "the picture is frozen or lost (pause menu, hidden workspace)",
    }[reason]
