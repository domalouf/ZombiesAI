"""Frame sources for the demo recorder, real and fake.

The plan insists that the real env loop be runnable end to end on Linux with fake adapters, and the same
insistence applies here: the recorder's timing, alignment and writing are the parts most likely to be
subtly wrong, and they are all testable without a game. `ClipPlayback` is this module's FakeCapture and
`SimSource` its FakeCapture-with-a-player -- between them the whole recording path runs in CI.

A source answers `read()` with one policy-sized frame. Cropping and downsampling happen here rather than
later so that nothing downstream ever sees a frame whose geometry it has to guess at.
"""

import time
from pathlib import Path

import numpy as np

from zombiesai.demos import frames as fr
from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import InputConfig, synthesize


class CaptureLost(RuntimeError):
    """The thing being captured is gone for good (the game window was destroyed), so there is nothing left to
    wait for. A recorder should stop cleanly on this rather than hold a stale frame forever."""


# Consecutive identical policy frames after which the picture counts as frozen. On this desktop a window on a
# hidden Hyprland workspace can still be grabbed -- XWayland keeps handing back its last buffer -- so a failed
# grab is not the only sign the game is out of sight. A live WaW scene never area-averages to the same 128x72
# twice (fog, lighting, the viewmodel's idle sway; the capture spike saw zero unchanged frames in 150), and a
# hidden window repeats exactly. Three reads is 200 ms: long enough that a coincidence is not an outage.
FROZEN_READS = 3


class ScreenCapture:
    """Desktop capture behind one `grab()`: X11/XWayland on Linux, DXGI Desktop Duplication on Windows.

    Borderless windowed is the assumption on Windows, as the plan requires -- DirectX 9 exclusive fullscreen
    generally cannot be captured by Desktop Duplication at all. On Linux the game is an XWayland client, so
    capture targets its window by title or id and the compositor never enters the picture.
    """

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,  # (left, top, width, height)
        *,
        backend: str = "auto",
        fit: str = "crop",
        monitor: int = 1,
        window: int | str | None = None,
        display: str | None = None,
        hud_regions: dict | None = None,
        hud_scale: float = 1.0,
    ):
        self.region, self.fit, self.monitor = region, fit, monitor
        self.window, self.display = window, display
        # Full-resolution HUD crops, cut from the same grab as each policy frame (see demos/hud_crops.py).
        self.hud_regions, self.hud_scale = hud_regions, hud_scale
        self.last_hud: dict[str, np.ndarray] | None = None
        self._frozen = 0  # consecutive reads whose policy frame repeated the one before
        self._probe: np.ndarray | None = None  # is_capturable's previous sample, to tell live from frozen
        # Whether the frame `read()` last returned is a repeat of an earlier one because the window could not
        # be grabbed, and why. A repeat is not what the player saw, so a recorder must not train on it.
        self.last_stale = False
        self.stale_reason: str | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._last: np.ndarray | None = None
        self._shape: tuple[int, ...] | None = None  # full-resolution shape of the first good grab
        self._policy: np.ndarray | None = None  # the last good policy frame, returned again while stale
        self.backend = backend if backend != "auto" else self._pick()
        self._open()

    def _pick(self) -> str:
        import importlib.util
        import os
        import sys

        if sys.platform == "win32" and importlib.util.find_spec("dxcam") is not None:
            return "dxcam"
        # An explicit display or window is a statement of intent; otherwise DISPLAY says whether there is an
        # X server (which on a Wayland session means XWayland, which is where the game will be).
        if sys.platform.startswith("linux") and (self.display or self.window is not None or os.environ.get("DISPLAY")):
            return "x11"
        if importlib.util.find_spec("mss") is not None:
            return "mss"
        raise RuntimeError(
            "no capture backend: on Linux set DISPLAY (the game runs under XWayland), "
            "on Windows pip install dxcam, or install mss"
        )

    def _open(self) -> None:
        if self.backend == "dxcam":
            import dxcam

            self._camera = dxcam.create(output_color="RGB")
        elif self.backend == "x11":
            from zombiesai.demos.x11_capture import X11Grabber

            # The region goes to the server, which sends back only those pixels; cropping here instead would
            # copy the whole screen across the socket every tick to throw most of it away.
            self._grabber = X11Grabber(window=self.window, display=self.display, region=self.region)
        else:
            import mss

            self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """One full-resolution RGB frame, repeating the last one if the compositor had no new frame."""
        if self.backend == "x11":
            self._last = self._grabber.grab()
            return self._last
        if self.backend == "dxcam":
            frame = self._camera.grab(region=self.region)
            if frame is None:
                if self._last is None:
                    raise RuntimeError("capture produced no frame at all; is the game running?")
                return self._last
            self._last = np.asarray(frame, dtype=np.uint8)
            return self._last
        box = (
            {"left": self.region[0], "top": self.region[1], "width": self.region[2], "height": self.region[3]}
            if self.region
            else self._sct.monitors[self.monitor]
        )
        shot = self._sct.grab(box)
        self._last = np.asarray(shot, dtype=np.uint8)[:, :, 2::-1]  # BGRA -> RGB
        return self._last

    def is_capturable(self) -> bool:
        """Whether a grab would give a live picture now. Only X11 can say no: its window can be on another
        workspace, whereas the other backends grab a monitor that is always there.

        A grab that succeeds is not enough on its own -- a window on a hidden workspace can still be grabbed,
        and returns its last frame forever (see FROZEN_READS). So the picture must also have changed since
        the previous question, which is why the first question after opening always answers no."""
        if self.backend != "x11":
            return True
        if not self._grabber.is_capturable():
            self._probe = None
            return False
        sample = np.ascontiguousarray(self.grab()[::16, ::16])
        live = self._probe is not None and not np.array_equal(sample, self._probe)
        self._probe = sample
        return live

    def read(self) -> np.ndarray:
        """The next policy frame -- or, while the window cannot be grabbed, the last good one again, with
        `last_stale` set and `last_hud` left at the crops that went with it.

        Losing the window for a moment is ordinary on a desktop (a workspace switch, a notification that
        steals the surface) and should cost the recording those steps, not the whole session. With no good
        frame to fall back on there is nothing honest to return, so the error propagates; and a window that
        has been destroyed raises `CaptureLost`, since no amount of waiting brings that XID back.
        """
        try:
            frame = self.grab()
        except Exception as error:
            if self._policy is None or not self._transient(error):
                raise
            return self._stale(str(error))
        if self._shape is not None and frame.shape != self._shape:
            # The window came back at another size. The policy frame would survive a rescale, but the crop box
            # was frozen on the first frame and the HUD crops' shapes are fixed for the life of a clip (the
            # writer refuses a change, rightly: the crops would no longer mean the same pixels). So a resized
            # window is an outage like any other until it is put back, and the recorder gives up on it after
            # its outage limit rather than splicing two geometries into one clip.
            height, width = self._shape[:2]
            return self._stale(
                f"the window is now {frame.shape[1]}x{frame.shape[0]}; the recording started at {width}x{height}"
            )
        if self._box is None:
            # Detected once, on the first frame, and then frozen: a crop that drifts mid-recording would
            # change what the pixels mean halfway through the clip.
            self._box = fr.crop_box(*frame.shape[:2], fr.detect_bars(frame), self.fit)
        if self.hud_regions:
            from zombiesai.demos.hud_crops import crop_regions

            self.last_hud = crop_regions(frame, self.hud_regions, self.hud_scale)
        self._shape = frame.shape
        policy = fr.to_policy_frame(frame, self._box, self.fit)
        self._frozen = self._frozen + 1 if self._policy is not None and np.array_equal(policy, self._policy) else 0
        self._policy = policy
        if self._frozen >= FROZEN_READS - 1:
            return self._stale(
                f"the picture has not changed for {self._frozen + 1} frames (window on a hidden workspace?)"
            )
        self.last_stale, self.stale_reason = False, None
        return self._policy

    def _transient(self, error: Exception) -> bool:
        """Whether a failed grab is worth waiting out. Only the X11 path can tell the difference; anything
        it cannot vouch for is treated as the fault it looks like."""
        if self.backend != "x11":
            return False
        from zombiesai.demos.x11_capture import WindowGone, WindowUnavailable

        if isinstance(error, WindowGone):
            raise CaptureLost(str(error)) from error
        return isinstance(error, WindowUnavailable)

    def _stale(self, reason: str) -> np.ndarray:
        self.last_stale, self.stale_reason = True, reason
        return self._policy

    def describe(self) -> dict:
        out = {"kind": "screen", "backend": self.backend, "region": self.region, "fit": self.fit, "box": self._box}
        if self.hud_regions:
            out["hud"] = {"regions": {k: list(v) for k, v in self.hud_regions.items()}, "scale": self.hud_scale}
        if self.backend == "x11":
            out["source"] = self._grabber.describe()
        return out

    def close(self) -> None:
        if self.backend == "dxcam":
            self._camera.release()
        elif self.backend == "x11":
            self._grabber.close()
        else:
            self._sct.close()


class FollowWindow:
    """A screen capture that follows the game's window by title instead of holding one window id.

    WaW under Proton opens a window for its intro, destroys it and opens the real one, and a restart or a
    video mode change does the same mid-session. A capture opened on the first window then fails for good
    (a live player crashed on exactly that, the moment it was handed the controls). This one closes the dead
    capture, looks the title up again on the next read, and in between answers with its last good frame
    marked stale -- which a recorder flags bad and a player treats as a pause.

    `has_frame` stays False until a real frame has been captured, so a caller writing a clip can skip the
    placeholder steps before there is anything (or any HUD crop shape) to write.
    """

    def __init__(self, open_capture, *, reopen_every_s: float = 0.5, clock=time.monotonic):
        self._open, self._capture = open_capture, None
        self.reopen_every_s, self.clock = reopen_every_s, clock
        self._next_try = 0.0
        self._policy = np.zeros(spec_pixels_shape(), dtype=np.uint8)
        self.last_hud = None
        self.last_stale, self.stale_reason = True, "no game window yet"
        self.has_frame = False

    def _lost(self, reason: str) -> np.ndarray:
        self.last_stale, self.stale_reason = True, reason
        return self._policy

    def read(self) -> np.ndarray:
        from zombiesai.demos.x11_capture import WindowNotFound, X11Error

        if self._capture is None:
            now = self.clock()
            if now < self._next_try:
                return self._lost(self.stale_reason or "looking for the game window")
            self._next_try = now + self.reopen_every_s
            try:
                self._capture = self._open()
            except WindowNotFound:
                return self._lost("no game window")
        try:
            frame = self._capture.read()
        except (CaptureLost, X11Error) as error:
            # The window this capture held is gone (or cannot be read at all): drop it and find the title again.
            self.close()
            return self._lost(f"the game window went away ({error}); looking for it again")
        self._policy = frame
        self.last_hud = self._capture.last_hud
        self.last_stale = bool(getattr(self._capture, "last_stale", False))
        self.stale_reason = getattr(self._capture, "stale_reason", None)
        self.has_frame = self.has_frame or not self.last_stale
        return frame

    def is_capturable(self) -> bool:
        return self._capture is not None and self._capture.is_capturable()

    def describe(self) -> dict:
        return {"kind": "follow-window", **(self._capture.describe() if self._capture is not None else {})}

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None


def spec_pixels_shape() -> tuple[int, ...]:
    from zombiesai import spec

    return tuple(spec.PIXELS_SHAPE)


class ClipPlayback:
    """Replays a recorded clip as if it were a live capture -- the plan's FakeCapture, for CI."""

    def __init__(self, path: str | Path, loop: bool = False):
        self.clip = load_clip(path)
        self.loop = loop
        self.i = 0

    def read(self) -> np.ndarray:
        if self.i >= self.clip.n_steps:
            if not self.loop:
                raise StopIteration(f"{self.clip.path} has only {self.clip.n_steps} frames")
            self.i = 0
        frame = np.asarray(self.clip.frames[self.i], dtype=np.uint8)
        self.i += 1
        return frame

    def describe(self) -> dict:
        return {"kind": "clip_playback", "path": str(self.clip.path)}

    def close(self) -> None:
        pass


class SimSource:
    """NachtSim in render mode with an agent at the controls, emitting both frames and the raw input events
    that agent would have produced.

    It is how the recording path is exercised without Windows, and how labelled clips for training the
    inverse dynamics model are generated before anyone has recorded a real demo. What it is *not* is a
    substitute for real footage: the raycast view is a crude stand-in and visual sim-to-real transfer is a
    non-goal (docs/sim_lies.md).
    """

    def __init__(self, agent, *, seed: int = 0, hardness: float = 0.5, max_steps: int = 18_000,
                 input_config: InputConfig | None = None):
        from zombiesai.sim.nacht_sim import NachtSim, SimConfig

        self.input_config = input_config or InputConfig(counts_per_degree=10.0)
        self.env = NachtSim(SimConfig(hardness=hardness, max_steps=max_steps, obs_profile="render"))
        self.agent = agent
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.agent.reset()
        self.done = False
        self.info: dict = {}
        self.steps = 0

    def read(self) -> np.ndarray:
        return np.asarray(self.obs["pixels"], dtype=np.uint8)

    def agent_obs(self) -> dict:
        """What the *player* sees. A scripted agent reads the state vector, a BC policy reads the pixels;
        the clip stores only the pixels either way, because that is all a human had."""
        return {**self.obs, "state": self.env.state()}

    def drain(self, start: float, end: float) -> list[dict]:
        """Advance the sim by one decision and return the events the agent's action is made of."""
        if self.done:
            return []
        action = np.asarray(self.agent.act(self.agent_obs()))
        events = synthesize(action, start, end - start, self.input_config)
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated
        self.steps += 1
        return events

    def describe(self) -> dict:
        # The sim's input latency is drawn per episode; the yaw-versus-flow check needs it to know which lag a
        # correctly paired recording should peak at.
        return {"kind": "sim", "seed": self.seed, "agent": type(self.agent).__name__,
                "latency_steps": int(self.env.timing.latency_steps)}

    def close(self) -> None:
        self.env.close()


class ReplayInput:
    """Plays a recorded inputs.jsonl back in step with a ClipPlayback, for testing the recorder offline.

    The log was stamped on the clock of whichever session produced it, and the recorder runs on this one, so
    the events are rebased onto the first window the recorder asks for. Without that every event lands in the
    first decision and the labels come out as a single mash of the whole session.
    """

    def __init__(self, events: list[dict], origin: float | None = None):
        self.events = sorted(events, key=lambda e: e["t"])
        self.origin = origin if origin is not None else (self.events[0]["t"] if self.events else 0.0)
        self.shift: float | None = None
        self.i = 0

    def drain(self, start: float, end: float) -> list[dict]:
        if self.shift is None:
            self.shift = start - self.origin
        out = []
        while self.i < len(self.events) and self.events[self.i]["t"] + self.shift < end:
            event = self.events[self.i]
            out.append({**event, "t": event["t"] + self.shift})
            self.i += 1
        return out

    def close(self) -> None:
        pass


def sleep_until(deadline: float) -> float:
    """Sleep to an absolute monotonic deadline, never `sleep(period)`: periods accumulate drift, deadlines
    do not. Returns the overshoot in seconds."""
    remaining = deadline - time.monotonic()
    if remaining > 0.002:
        time.sleep(remaining - 0.001)
    while time.monotonic() < deadline:  # the last millisecond, spun: the OS timer is not that precise
        pass
    return time.monotonic() - deadline
