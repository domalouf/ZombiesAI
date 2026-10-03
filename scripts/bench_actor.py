"""What one actor's own work costs per decision, stage by stage, on synthetic frames: no game, no X server.

An actor has 66.7 ms per decision (15 Hz on absolute deadlines) and a step that runs over is wasted, so every
millisecond spent here is one less for the X server, the game and the other actors on the same PC. This times
exactly the code a `RealGameEnv` step runs, single-threaded as the actors run it (OMP/OPENBLAS/MKL threads = 1,
set before numpy loads), on frames shaped like what MIT-SHM hands back: BGRX, padded to bytes_per_line.

What is faked, and why it does not flatter the numbers:

* The X server. `XShmGetImage` itself (the server copying the window into the shared segment; ~1.4 ms at 1440p
  under Weston, docs/rl.md) is the server's work and is not timed. The segment is rewritten before every step,
  outside the timer, so the actor reads a frame that has just been written -- as it does live.
* The HUD. Each frame carries real HUD crops (tests/fixtures/hud/crops.npz) pasted where the game draws them,
  so the parser takes the branches it takes on real digits and tallies, not on noise.
* The audio. `LiveAudio` is fed 10 ms chunks of noise on a clock that keeps pace with the frames, as the
  stream thread does; only `observe` is timed.

    uv run python scripts/bench_actor.py                      # 2560x1440, 1920x1080 and 1280x720
    uv run python scripts/bench_actor.py --sizes 2560x1440 --steps 1000
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # before numpy loads, as train() does for its actors

import argparse
import time
import types
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos import frames as fr
from zombiesai.demos import x11_capture
from zombiesai.demos.hud_crops import HUD_REGIONS, crop_regions, region_box
from zombiesai.realgame.console import CONSOLE_REGION, console_open
from zombiesai.realgame.scoreboard import SCOREBOARD_REGION, scoreboard_shown

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hud" / "crops.npz"
SIZES = ((2560, 1440), (1920, 1080), (1280, 720))
N_SCENES = 4
RATE, CHUNK = 48_000, 480  # what parec delivers: 10 ms chunks


def actor_regions(height: int) -> tuple[dict, float]:
    """The crops and scale `Instance.capture()` asks for: 1440p HUD at half scale, i.e. 720p's own pixels."""
    regions = {**{k: HUD_REGIONS[k] for k in ("points_ammo", "round")}, "console": CONSOLE_REGION,
               "scores": SCOREBOARD_REGION}
    return regions, min(1.0, 720.0 / height)


# ------------------------------------------------------------------------------------------------ the fakes


def nearest(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize: at 1440p, an exact 2x of a half-scale crop, which crops back to the original."""
    ys = (np.arange(h) * image.shape[0]) // h
    xs = (np.arange(w) * image.shape[1]) // w
    return image[ys][:, xs]


class FakeServer:
    """The X server's half of MIT-SHM: one shared segment, rewritten with the next frame before each step.

    Frames are a few textured scenes (bright enough that no edge reads as a letterbox bar, different enough
    that the freeze check never fires) with real HUD crops pasted in at their boxes."""

    def __init__(self, width: int, height: int, *, pad: int = 0, seed: int = 0):
        self.width, self.height = width, height
        self.stride = width * 4 + pad
        self.flat = np.zeros(self.stride * height, np.uint8)
        self.rows = self.flat.reshape(height, self.stride)
        self.view = self.rows[:, : width * 4].reshape(height, width, 4)  # BGRX
        rng = np.random.default_rng(seed)
        self.scenes = []
        for _ in range(N_SCENES):
            coarse = rng.integers(40, 200, (height // 16 + 1, width // 16 + 1, 3), dtype=np.uint8)
            scene = np.repeat(np.repeat(coarse, 16, 0), 16, 1)[:height, :width]
            scene = (scene + rng.integers(0, 24, scene.shape, dtype=np.uint8)).astype(np.uint8)
            bgrx = np.empty((height, width, 4), np.uint8)
            bgrx[..., :3] = scene[..., ::-1]
            bgrx[..., 3] = 255
            self.scenes.append(bgrx)
        with np.load(FIXTURES) as z:
            fixtures = {k: z[k] for k in ("points_ammo", "round")}
        self.hud = {}
        for name, crops in fixtures.items():
            left, top, w, h = region_box(height, width, HUD_REGIONS[name])
            self.hud[name] = (top, left, [np.concatenate([nearest(c, h, w)[..., ::-1],
                                                          np.full((h, w, 1), 255, np.uint8)], -1) for c in crops])

    def render(self, k: int, hud: dict[str, int]) -> None:
        """Frame k into the segment, with HUD fixture `hud[name]` pasted into each region."""
        np.copyto(self.view, self.scenes[k % N_SCENES])
        for name, index in hud.items():
            top, left, crops = self.hud[name]
            crop = crops[index % len(crops)]
            self.view[top : top + crop.shape[0], left : left + crop.shape[1]] = crop

    def grabber(self):
        """An X11Grabber whose `XShmGetImage` is this segment: everything after the server's copy is real."""
        grabber = x11_capture.X11Grabber.__new__(x11_capture.X11Grabber)
        grabber.x, grabber.display, grabber.window, grabber.screen, grabber.root = None, None, 0x4800006, 0, 0x100
        grabber.depth, grabber.backend, grabber.shm_info = 24, "xshm", None
        grabber.image, grabber._buffer = object(), self.flat
        grabber.region = (0, 0, self.width, self.height)
        grabber._follows_window, grabber._lost, grabber._shm = True, False, True
        grabber._shm_read = lambda: (self.flat, self.stride)
        grabber.close = lambda: None
        return grabber


class AudioFeed:
    """Feeds a LiveAudio the way its stream thread would: 10 ms chunks of noise, stamped on arrival, on a
    clock that keeps pace with the frames."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t0 = 1000.0
        self.sent = 0
        self.noise = self.rng.integers(-6000, 6000, (RATE * 2, 2)).astype(np.int16)

    def until(self, live, t: float) -> None:
        while self.t0 + (self.sent + CHUNK) / RATE <= t:
            start = self.sent % (len(self.noise) - CHUNK)
            self.sent += CHUNK
            arrival = self.t0 + self.sent / RATE + 0.008 + float(self.rng.uniform(0, 0.003))
            live.feed(self.noise[start : start + CHUNK], arrival)


class FakeDispatcher:
    """Sends nothing, and makes the env's wait for its deadline instant: the clock jumps to it."""

    def __init__(self, clock):
        self.clock = clock

    def apply(self, action, now=None, dt=None, look_deg=None):
        pass

    def pump_until(self, deadline, poll_s=0.002):
        self.clock.t = max(self.clock.t, deadline)

    def release_all(self):
        pass

    def close(self):
        pass


class VirtualClock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------------------------------------ timing


class Timer:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def __call__(self, name: str, fn, *args):
        start = time.perf_counter_ns()
        out = fn(*args)
        self.samples.setdefault(name, []).append((time.perf_counter_ns() - start) / 1e6)
        return out

    def row(self, name: str) -> tuple[float, float]:
        x = np.asarray(self.samples[name])
        return float(np.median(x)), float(np.percentile(x, 99))


def bench_stages(server: FakeServer, steps: int, warmup: int) -> Timer:
    """Each stage on its own, fed by the one before, as `ScreenCapture.read` and `RealGameEnv.step` chain them."""
    from zombiesai.demos.hearing import LiveAudio
    from zombiesai.hud.parse import HudParser
    from zombiesai.hud.track import HudTracker
    from zombiesai.realgame.hud_reward import HudSignals
    from zombiesai.reward import RewardShaper

    regions, scale = actor_regions(server.height)
    grabber = server.grabber()
    parser, tracker, signals, shaper = HudParser(), HudTracker(), HudSignals(), RewardShaper()
    clock = VirtualClock(1000.0)
    live = LiveAudio(types.SimpleNamespace(rate=RATE, channels=2), clock=clock)
    feed = AudioFeed()
    action = np.asarray(spec.make_action(forward=1, fire=1), np.int64)
    box = None
    timer = Timer()
    for k in range(warmup + steps):
        if k == warmup:
            timer = Timer()
        server.render(k, {"points_ammo": k // 3, "round": k // 7})
        clock.t += 1.0 / spec.DECISION_HZ
        feed.until(live, clock.t)
        frame = timer("grab_bgrx (after the server's copy)", grabber.grab_bgrx)
        if box is None:
            box = fr.crop_box(*frame.shape[:2], fr.detect_bars(frame[..., fr.BGRX]), "crop")
        policy = timer("policy frame", lambda: fr.to_policy_frame(frame, box, "crop", channels=fr.BGRX))
        crops = timer("HUD crops", lambda: crop_regions(frame, regions, scale, channels=fr.BGRX))
        reading = timer("HUD parse", parser.parse, crops)
        timer("console_open", console_open, crops["console"])
        timer("scoreboard_shown", scoreboard_shown, crops["scores"])

        def bookkeeping():
            return shaper(signals.step(tracker.step(reading), action))

        timer("tracker + signals + reward", bookkeeping)
        timer("LiveAudio.observe", live.observe, clock.t)
        assert policy.shape == spec.PIXELS_SHAPE
    return timer


def bench_env(server: FakeServer, steps: int, warmup: int) -> Timer:
    """`RealGameEnv.step` end to end -- capture, HUD, console, scoreboard, reward, audio -- with the wait for the
    deadline made instant. What is left is the actor's own work per decision, less the policy's forward."""
    from zombiesai.demos.capture import ScreenCapture
    from zombiesai.demos.hearing import LiveAudio
    from zombiesai.realgame.env import RealGameEnv

    regions, scale = actor_regions(server.height)
    grabber = server.grabber()
    real = x11_capture.X11Grabber
    x11_capture.X11Grabber = lambda window=None, display=None, region=None: grabber
    try:
        capture = ScreenCapture(window="bench", hud_regions=regions, hud_scale=scale)
    finally:
        x11_capture.X11Grabber = real
    clock = VirtualClock(1000.0)
    live = LiveAudio(types.SimpleNamespace(rate=RATE, channels=2), clock=clock)
    feed = AudioFeed()
    env = RealGameEnv(capture, FakeDispatcher(clock), focus=lambda: True, press=lambda key: None,
                      console_open=lambda: console_open((capture.last_hud or {}).get("console")),
                      downed=lambda: scoreboard_shown((capture.last_hud or {}).get("scores")),
                      hearing=live, clock=clock, sleep=lambda s: None, say=lambda m: None)
    action = np.asarray(spec.make_action(forward=1, yaw=2.0), np.int64)
    hud = {"points_ammo": 4, "round": 1}  # 3710 points, round 3: a settled HUD, as most steps are
    timer = Timer()
    for k in range(warmup + steps):
        if k == warmup:
            timer = Timer()
        server.render(k, hud)
        feed.until(live, clock.t + 1.0 / spec.DECISION_HZ)
        timer("capture.read", capture.read)  # the same work env.step does first, timed alone
        server.render(k + 1, hud)
        timer("RealGameEnv.step", env.step, action)
    return timer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", nargs="*", default=[f"{w}x{h}" for w, h in SIZES])
    parser.add_argument("--steps", type=int, default=450, help="timed steps per size (450 is 30 s of play)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--pad", type=int, default=0, help="bytes of padding per row, past width * 4")
    args = parser.parse_args()

    print(f"single-threaded (OMP/OPENBLAS/MKL threads = {os.environ['OPENBLAS_NUM_THREADS']}), numpy {np.__version__},"
          f" {args.steps} steps after {args.warmup} of warm-up; median / p99 in ms")
    for size in args.sizes:
        width, height = (int(v) for v in size.lower().split("x"))
        server = FakeServer(width, height, pad=args.pad)
        stages = bench_stages(server, args.steps, args.warmup)
        whole = bench_env(server, args.steps, args.warmup)
        print(f"\n{width}x{height} (stride {server.stride} bytes)")
        print(f"  {'stage':<32} {'median':>8} {'p99':>8}")
        total = 0.0
        for name in stages.samples:
            median, p99 = stages.row(name)
            total += median
            print(f"  {name:<32} {median:8.2f} {p99:8.2f}")
        print(f"  {'(sum of the medians)':<32} {total:8.2f}")
        for name in whole.samples:
            median, p99 = whole.row(name)
            print(f"  {name:<32} {median:8.2f} {p99:8.2f}")


if __name__ == "__main__":
    main()
