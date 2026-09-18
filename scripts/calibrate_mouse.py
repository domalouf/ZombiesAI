"""Spikes S1 and S4 in one run: does synthetic input reach the game, and how many mouse counts is a degree?

    uv run python scripts/calibrate_mouse.py --window "World at War"

Stand somewhere with the room in view (not facing a blank wall), leave the game focused, and don't touch the
mouse. The script turns the view with the virtual mouse and watches the pixels move:

* **S1 (synthetic input).** If the view never moves, the engine is ignoring the virtual device and nothing
  else in the project can proceed. That is the answer, and it is worth knowing in the first ten seconds.
* **S4 (mouse linearity).** It sweeps a range of count deltas and fits pixel shift against counts. R^2 > 0.98
  is the pass mark; below it, in-game mouse smoothing or acceleration is still on, or libinput is
  accelerating the virtual device (see docs/linux.md).
* **counts per degree.** Measured two ways: from an assumed field of view, and -- with `--full-turn` -- by
  turning all the way around until the view comes back to where it started, which needs no field of view at
  all and also tells you what the real one is.

The number it prints is what `record_demo.py --counts-per-degree` and `DispatchConfig` want.
"""

import argparse
import math
import sys
import time

import numpy as np

from zombiesai.demos.frames import estimate_shift, luma
from zombiesai.demos.x11_capture import X11Grabber, find_window
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
from zombiesai.realgame.uinput import UinputDevice

SETTLE_S = 0.25  # the game needs a few frames to finish the turn before the pixels are worth reading


def turn(dispatcher, grabber, counts: int, settle: float = SETTLE_S) -> tuple[np.ndarray, np.ndarray]:
    before = grabber.grab()
    dispatcher.sink.move(counts, 0)
    dispatcher.sink.sync()
    time.sleep(settle)
    return before, grabber.grab()


def linearity(dispatcher, grabber, steps, repeats: int, settle: float) -> dict:
    """Sweep count deltas both ways and fit pixel shift against counts through the origin."""
    counts, shifts = [], []
    for magnitude in steps:
        for sign in (1, -1):
            for _ in range(repeats):
                before, after = turn(dispatcher, grabber, sign * magnitude, settle)
                shift, confidence = estimate_shift(before, after, max_shift=before.shape[1] // 3)
                if confidence < 0.02:
                    continue  # nothing distinctive in view; that sample says nothing
                counts.append(sign * magnitude)
                shifts.append(-shift)  # turning right drags the scene left
                turn(dispatcher, grabber, -sign * magnitude, settle)  # put the view back
    if len(counts) < 4:
        return {"samples": len(counts), "r2": float("nan"), "pixels_per_count": float("nan")}
    x, y = np.array(counts, dtype=float), np.array(shifts, dtype=float)
    slope = float(x @ y / (x @ x))  # through the origin: zero counts must mean zero rotation
    residual = y - slope * x
    r2 = 1.0 - float(residual @ residual) / float(((y - y.mean()) ** 2).sum() or 1.0)
    return {"samples": len(x), "r2": r2, "pixels_per_count": slope, "counts": x.tolist(), "shifts": y.tolist()}


def full_turn(dispatcher, grabber, step_counts: int, max_steps: int, settle: float) -> dict:
    """Turn until the view comes back to where it started. Total counts / 360 is counts per degree, with no
    assumption about the field of view anywhere in it."""
    reference = grabber.grab()
    reference_luma = luma(reference)
    total_counts, total_shift = 0, 0.0
    trace = []
    previous = reference
    for step in range(1, max_steps + 1):
        _, frame = turn(dispatcher, grabber, step_counts, settle)
        shift, _ = estimate_shift(previous, frame, max_shift=frame.shape[1] // 3)
        previous = frame
        total_counts += step_counts
        total_shift += -shift
        difference = float(np.abs(luma(frame) - reference_luma).mean())
        trace.append({"step": step, "counts": total_counts, "shift_px": total_shift, "difference": difference})
        # Only start looking for the return once the view has genuinely left, or step one wins trivially.
        if step > 8 and difference <= min(t["difference"] for t in trace[:8]) * 1.1:
            degrees_per_count = 360.0 / total_counts
            return {
                "found": True,
                "counts_per_360": total_counts,
                "counts_per_degree": 1.0 / degrees_per_count,
                "pixels_per_degree": abs(total_shift) / 360.0,
                "steps": step,
            }
    return {"found": False, "counts": total_counts, "trace": trace[-5:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default="World at War", help="window title substring, or 0x id")
    parser.add_argument("--display", help="X display, if not $DISPLAY")
    parser.add_argument("--fov", type=float, default=80.0,
                        help="horizontal field of view in degrees (WaW's cg_fov 65 is about 80 at 16:9)")
    parser.add_argument("--steps", type=int, nargs="*", default=[20, 50, 100, 200, 400],
                        help="mouse count deltas to sweep")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle", type=float, default=SETTLE_S, help="seconds to wait after each turn")
    parser.add_argument("--full-turn", action="store_true", help="also measure by turning all the way around")
    parser.add_argument("--turn-step", type=int, default=200, help="counts per step of the full turn")
    parser.add_argument("--countdown", type=int, default=5, help="seconds to click into the game first")
    args = parser.parse_args()

    if not sys.platform.startswith("linux"):
        raise SystemExit("this script drives the Linux virtual device; on Windows use the SendInput path")
    window = int(args.window, 16) if args.window.startswith("0x") else find_window(args.window, args.display).id
    grabber = X11Grabber(window=window, display=args.display)
    print(f"capturing {grabber.describe()}")

    config = DispatchConfig(counts_per_degree=1.0)
    device = UinputDevice(config.codes.values())
    print(f"virtual device: {device.describe()}")
    dispatcher = ActionDispatcher(device, config)
    try:
        for remaining in range(args.countdown, 0, -1):
            print(f"  focus the game and keep your hands off the mouse... {remaining}", end="\r", flush=True)
            time.sleep(1.0)
        print(" " * 60, end="\r")

        fit = linearity(dispatcher, grabber, args.steps, args.repeats, args.settle)
        if not fit["samples"] or not np.isfinite(fit["pixels_per_count"]) or abs(fit["pixels_per_count"]) < 1e-4:
            print("\nS1 FAIL: the view never moved. The engine is not seeing the virtual device.")
            print("  - is the game focused, and is this the window you captured?")
            print("  - is the device listed in `libinput list-devices` / your compositor's device list?")
            print("  - some engines ignore devices that appear after they start: relaunch the game with the")
            print("    virtual device already created, or create it before launching.")
            raise SystemExit(1)

        width = grabber.size[0]
        focal = width / 2 / math.tan(math.radians(args.fov) / 2)
        pixels_per_degree = focal * math.tan(math.radians(1.0))
        counts_per_degree = pixels_per_degree / abs(fit["pixels_per_count"])
        print(f"\nS1 PASS: the view moves with the virtual mouse ({fit['samples']} usable samples)")
        print(f"S4 {'PASS' if fit['r2'] > 0.98 else 'FAIL'}: linearity R^2 = {fit['r2']:.4f} (needs > 0.98)")
        if fit["r2"] <= 0.98:
            print("  turn off in-game mouse smoothing and acceleration, and set the virtual device to a flat")
            print("  acceleration profile in your compositor (docs/linux.md).")
        print(f"  {abs(fit['pixels_per_count']):.4f} px per count at {width} px wide")
        print(f"  counts per degree = {counts_per_degree:.3f}  (assuming a {args.fov:g} degree field of view)")

        if args.full_turn:
            print("\nturning all the way around...")
            spin = full_turn(dispatcher, grabber, args.turn_step, 400, args.settle)
            if spin["found"]:
                # pixels per degree fixes the focal length, and the focal length fixes the field of view --
                # so turning all the way around measures the FOV the renderer's 80-degree guess stands in for.
                focal_px = spin["pixels_per_degree"] / math.tan(math.radians(1.0))
                measured_fov = 2 * math.degrees(math.atan(width / 2 / focal_px))
                print(f"  full turn took {spin['counts_per_360']:,} counts over {spin['steps']} steps")
                print(f"  counts per degree = {spin['counts_per_degree']:.3f}  (no field of view assumed)")
                print(f"  implied field of view = {measured_fov:.1f} degrees (you assumed {args.fov:g})")
                counts_per_degree = spin["counts_per_degree"]
            else:
                print("  never came back to the starting view; try a smaller --turn-step, or stand somewhere")
                print("  with more to look at.")

        print(f"\nuse: --counts-per-degree {counts_per_degree:.3f}")
        print("record it in docs/spikes.md -- every look label and every turn the agent makes depends on it.")
    finally:
        dispatcher.close()
        grabber.close()


if __name__ == "__main__":
    main()
