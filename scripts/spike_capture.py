"""Spikes S2 and S3: how fast can we read the screen, and how long does the game take to answer an input?

    uv run python scripts/spike_capture.py --window "World at War" --latency

Three measurements, each with the plan's own pass mark:

* **S2 capture.** Grab rate and the p99 grab time, plus a check that the frames actually change -- a capture
  that returns the same buffer forever looks fast and teaches the agent nothing. PASS is >= 60 fps with a p99
  under 10 ms, in a mode the game still runs in.
* **The tick.** The same capture on 15 Hz deadlines with the downsample included, which is what the real loop
  will do. PASS is an overrun rate under 2%.
* **S3 latency.** With `--latency`, it turns the view with the virtual mouse and times how long until the
  pixels change. This is the closed-loop delay the whole delayed-MDP design is built around: it does not
  have to be small, it has to be known and stable. Set the sim's latency knob to the number it prints.

Nothing here needs the agent, a policy, or a trained anything: it is a measurement of your machine.
"""

import argparse
import statistics
import subprocess
import sys
import time

import numpy as np

from zombiesai import spec
from zombiesai.demos import frames as fr
from zombiesai.demos.capture import sleep_until
from zombiesai.demos.x11_capture import X11Grabber, find_window

TICK = 1.0 / spec.DECISION_HZ


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def throughput(grabber, frames: int) -> dict:
    times, previous, identical, dark = [], None, 0, 0
    for _ in range(frames):
        start = time.perf_counter()
        frame = grabber.grab()
        times.append((time.perf_counter() - start) * 1000.0)
        if previous is not None and np.array_equal(frame, previous):
            identical += 1
        dark += fr.luma(frame).mean() < 6.0
        previous = frame
    return {
        "frames": frames,
        "fps": frames / (sum(times) / 1000.0),
        "p50_ms": percentile(times, 50),
        "p99_ms": percentile(times, 99),
        "max_ms": max(times),
        "identical": identical,
        "black": int(dark),
    }


def tick_loop(grabber, box, fit: str, seconds: float) -> tuple[dict, list]:
    """Capture and downsample on absolute 15 Hz deadlines, the way the real loop will."""
    deadline = time.monotonic()
    overruns, costs, policy_frames = 0, [], []
    steps = int(seconds * spec.DECISION_HZ)
    for _ in range(steps):
        deadline += TICK
        sleep_until(deadline)
        start = time.perf_counter()
        policy_frames.append(fr.to_policy_frame(grabber.grab(), box, fit))
        cost = (time.perf_counter() - start) * 1000.0
        costs.append(cost)
        overruns += time.monotonic() > deadline + 0.5 * TICK
    deltas = fr.frame_delta(np.stack(policy_frames))
    return {
        "steps": steps,
        "overruns": overruns,
        "overrun_rate": overruns / max(steps, 1),
        "capture_p50_ms": percentile(costs, 50),
        "capture_p99_ms": percentile(costs, 99),
        "still_frames": int((deltas[1:] < 0.01).sum()),
        "mean_delta": float(deltas[1:].mean()) if len(deltas) > 1 else 0.0,
    }, policy_frames


def idle_noise(grabber, samples: int = 30) -> float:
    """How much the screen changes on its own. Zombies shamble and the HUD ticks, so "the view moved" has to
    clear what the scene does by itself or the latency number is measuring animation."""
    previous, diffs = fr.luma(grabber.grab()), []
    for _ in range(samples):
        time.sleep(0.02)
        current = fr.luma(grabber.grab())
        diffs.append(float(np.abs(current - previous).mean()))
        previous = current
    return percentile(diffs, 95)


def latency(grabber, dispatcher, counts: int, trials: int, timeout_s: float) -> dict:
    """Turn the view and time how long until the pixels say so."""
    threshold = max(3.0 * idle_noise(grabber), 1.5)
    delays = []
    for _ in range(trials):
        reference = fr.luma(grabber.grab())
        start = time.perf_counter()
        dispatcher.sink.move(counts, 0)
        dispatcher.sink.sync()
        while time.perf_counter() - start < timeout_s:
            if float(np.abs(fr.luma(grabber.grab()) - reference).mean()) > threshold:
                delays.append((time.perf_counter() - start) * 1000.0)
                break
        else:
            delays.append(float("nan"))
        dispatcher.sink.move(-counts, 0)
        dispatcher.sink.sync()
        time.sleep(0.3)
    good = [d for d in delays if d == d]
    return {
        "trials": trials,
        "answered": len(good),
        "threshold": threshold,
        "p50_ms": percentile(good, 50),
        "p99_ms": percentile(good, 99),
        "spread_ms": (max(good) - min(good)) if good else float("nan"),
        "stdev_ms": statistics.stdev(good) if len(good) > 1 else 0.0,
    }


def write_video(path, policy_frames, scale: int = 4) -> None:
    height, width = spec.PIXELS_SHAPE[:2]
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width * scale}x{height * scale}", "-framerate", str(spec.DECISION_HZ), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    for frame in policy_frames:
        encoder.stdin.write(np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8)).tobytes())
    encoder.stdin.close()
    encoder.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default="World at War", help="window title substring, or 0x id")
    parser.add_argument("--display", help="X display, if not $DISPLAY")
    parser.add_argument("--list-windows", action="store_true", help="print every window X can see and stop")
    parser.add_argument("--frames", type=int, default=300, help="frames for the throughput measurement")
    parser.add_argument("--seconds", type=float, default=10.0, help="seconds of the 15 Hz tick loop")
    parser.add_argument("--fit", choices=fr.FITS, default="crop")
    parser.add_argument("--latency", action="store_true", help="also measure input-to-pixels delay (S3)")
    parser.add_argument("--latency-counts", type=int, default=600, help="counts to turn for each latency trial")
    parser.add_argument("--latency-trials", type=int, default=20)
    parser.add_argument("--video", help="write what the agent would see to this mp4 (needs ffmpeg)")
    args = parser.parse_args()

    if not sys.platform.startswith("linux"):
        raise SystemExit("this spike drives the X11 capture path; on Windows measure the dxcam path instead")
    if args.list_windows:
        from zombiesai.demos.x11_capture import list_windows

        for info in sorted(list_windows(args.display), key=lambda w: -w.width * w.height):
            print(f"  0x{info.id:08x}  {info.width:>5}x{info.height:<5}  {info.title or '(untitled)'}")
        return
    window = int(args.window, 16) if args.window.startswith("0x") else find_window(args.window, args.display).id
    grabber = X11Grabber(window=window, display=args.display)
    width, height = grabber.size
    print(f"{grabber.describe()}  {width}x{height}")

    fast = throughput(grabber, args.frames)
    verdict = "PASS" if fast["fps"] >= 60 and fast["p99_ms"] < 10 else "FAIL"
    print(f"\nS2 capture {verdict}: {fast['fps']:.0f} fps, p50 {fast['p50_ms']:.2f} ms, "
          f"p99 {fast['p99_ms']:.2f} ms, max {fast['max_ms']:.2f} ms")
    print(f"  {fast['identical']} of {fast['frames']} frames identical to the one before, {fast['black']} black")
    if fast["identical"] > fast["frames"] * 0.5:
        print("  most frames never changed: the game may be paused, or this is the wrong window")

    box = fr.crop_box(height, width, fr.detect_bars(grabber.grab()), args.fit)
    print(f"\n15 Hz tick loop for {args.seconds:g}s (capture + crop + downsample to {spec.PIXELS_SHAPE[1]}x"
          f"{spec.PIXELS_SHAPE[0]})...")
    tick, policy_frames = tick_loop(grabber, box, args.fit, args.seconds)
    verdict = "PASS" if tick["overrun_rate"] < 0.02 else "FAIL"
    print(f"  {verdict}: {tick['overruns']}/{tick['steps']} overruns ({tick['overrun_rate']:.1%}), "
          f"capture p50 {tick['capture_p50_ms']:.2f} ms, p99 {tick['capture_p99_ms']:.2f} ms")
    print(f"  mean frame delta {tick['mean_delta']:.2f}, {tick['still_frames']} frames with no change at all")

    if args.latency:
        from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
        from zombiesai.realgame.uinput import UinputDevice

        config = DispatchConfig()
        device = UinputDevice(config.codes.values())
        dispatcher = ActionDispatcher(device, config)
        try:
            print("\nS3 latency: keep the game focused and your hands off the mouse...")
            delay = latency(grabber, dispatcher, args.latency_counts, args.latency_trials, timeout_s=0.5)
            if not delay["answered"]:
                print("  FAIL: the view never moved. Run scripts/calibrate_mouse.py -- this is spike S1.")
            else:
                verdict = "PASS" if delay["p99_ms"] < 120 and delay["stdev_ms"] < 25 else "CHECK"
                print(f"  {verdict}: p50 {delay['p50_ms']:.0f} ms, p99 {delay['p99_ms']:.0f} ms, "
                      f"stdev {delay['stdev_ms']:.0f} ms over {delay['answered']}/{delay['trials']} trials")
                print(f"  that is {delay['p50_ms'] / (TICK * 1000):.1f} decisions of delay; set the sim's "
                      f"latency_steps to {round(delay['p50_ms'] / (TICK * 1000))} and check the agent can "
                      "still learn to aim there")
        finally:
            dispatcher.close()

    if args.video:
        write_video(args.video, policy_frames)
        print(f"\nwhat the agent sees: {args.video}")
    grabber.close()


if __name__ == "__main__":
    main()
