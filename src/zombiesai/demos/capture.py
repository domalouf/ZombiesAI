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
    ):
        self.region, self.fit, self.monitor = region, fit, monitor
        self.window, self.display = window, display
        self._box: tuple[int, int, int, int] | None = None
        self._last: np.ndarray | None = None
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

    def read(self) -> np.ndarray:
        frame = self.grab()
        if self._box is None:
            # Detected once, on the first frame, and then frozen: a crop that drifts mid-recording would
            # change what the pixels mean halfway through the clip.
            self._box = fr.crop_box(*frame.shape[:2], fr.detect_bars(frame), self.fit)
        return fr.to_policy_frame(frame, self._box, self.fit)

    def describe(self) -> dict:
        out = {"kind": "screen", "backend": self.backend, "region": self.region, "fit": self.fit, "box": self._box}
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
        return {"kind": "sim", "seed": self.seed, "agent": type(self.agent).__name__}

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
