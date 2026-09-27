"""Before a long parallel run: does every instance take input, alone, and can it be read?

    uv run python scripts/instances.py up --n 2
    uv run python scripts/spike_instances.py --counts-per-degree 9.09

For each instance of the fleet, with all of them running, it checks -- and each is a pass/fail, because each
one can end the parallel plan on its own:

1. **window**  the game has a window on its own display.
2. **capture** the picture is live (not frozen), how long a grab takes, and whether the HUD reads (points).
3. **focus**   the game window holds its X server's focus -- what Wine needs before DirectInput delivers.
4. **mouse**   an XTEST turn moves *this* game's view (image shift, the S1 test) and nobody else's.
5. **keys**    an XTEST key reaches the game: the console key visibly drops the console down.

Stand each game somewhere with detail in view (the start room, not a wall) before running it; the mouse test
reads horizontal image motion. Then, for counts per degree on an instance:

    uv run python scripts/calibrate_mouse.py --display :60 --window "Call of Duty" --xtest --full-turn
"""

import argparse
import time

import numpy as np

from zombiesai.demos.frames import estimate_shift, luma
from zombiesai.hud.parse import OK, STATUS_NAMES, HudParser
from zombiesai.realgame.instances import fleet, load_fleet


def grab_policy(capture, n: int = 1, dt: float = 1 / 15):
    frames, stale, costs = [], 0, []
    for _ in range(n):
        t = time.perf_counter()
        frames.append(capture.read().copy())
        costs.append(time.perf_counter() - t)
        stale += bool(capture.last_stale)
        time.sleep(dt)
    return frames, stale, costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs/instances")
    parser.add_argument("--counts-per-degree", type=float, default=9.09)
    parser.add_argument("--turn-deg", type=float, default=20.0)
    parser.add_argument("--fov", type=float, default=80.0, help="horizontal field of view, for the expected shift")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for each game window")
    args = parser.parse_args()

    config = load_fleet(args.root)
    instances = fleet(config, say=lambda m: None)
    reader = HudParser()
    results = []
    captures, sinks, windows = {}, {}, {}
    for inst in instances:
        r = {"instance": inst.spec.index, "display": inst.spec.display}
        results.append(r)
        deadline = time.monotonic() + args.wait
        while True:
            try:
                windows[inst.spec.index] = inst.window()
                r["window"] = True
                break
            except Exception as error:  # noqa: BLE001
                if time.monotonic() > deadline:
                    r["window"] = False
                    r["why"] = str(error)[:120]
                    break
                time.sleep(1.0)
        if not r["window"]:
            continue
        sinks[inst.spec.index] = sink = inst.sink()
        sink.focus(windows[inst.spec.index].id)
        r["focus"] = sink.focused_window() == windows[inst.spec.index].id
        captures[inst.spec.index] = cap = inst.capture()
        frames, stale, costs = grab_policy(cap, 30)
        r["capture_live"] = stale < 5
        r["grab_ms_median"] = round(1000 * float(np.median(costs)), 2)
        if cap.last_hud is not None:
            reading = reader.parse(cap.last_hud)
            r["hud_points"] = reading.points if reading.points_status == OK else STATUS_NAMES[reading.points_status]
        r["hud"] = isinstance(r.get("hud_points"), int)

    ready = [i for i in captures]
    counts = int(round(args.turn_deg * args.counts_per_degree))
    expected_px = args.turn_deg * 128 / args.fov
    for i in ready:
        r = results[i]
        before = {j: grab_policy(captures[j])[0][0] for j in ready}
        sink = sinks[i]
        steps = 8
        for k in range(steps):  # spread over ~0.25 s, as the dispatcher would
            sink.move(counts // steps + (1 if k < counts % steps else 0), 0)
            sink.sync()
            time.sleep(0.03)
        time.sleep(0.35)
        after = {j: grab_policy(captures[j])[0][0] for j in ready}
        shift, confidence = estimate_shift(before[i], after[i], max_shift=40)
        r["turn_px"] = -shift
        r["turn_expected_px"] = round(expected_px, 1)
        r["turn_confidence"] = round(confidence, 3)
        r["mouse"] = -shift >= max(3, expected_px / 3) and confidence > 0.02
        r["others_moved_px"] = {j: -estimate_shift(before[j], after[j], max_shift=40)[0] for j in ready if j != i}
        for k in range(steps):  # and back
            sink.move(-(counts // steps + (1 if k < counts % steps else 0)), 0)
            sink.sync()
            time.sleep(0.03)
        time.sleep(0.35)

        top = slice(0, 24)
        closed = luma(grab_policy(captures[i])[0][0][None])[0][top]
        for down in (True, False):
            sink.key("grave", down)
            sink.sync()
        time.sleep(0.5)
        opened = luma(grab_policy(captures[i])[0][0][None])[0][top]
        for down in (True, False):
            sink.key("grave", down)
            sink.sync()
        time.sleep(0.3)
        r["console_change"] = round(float(np.abs(opened - closed).mean()), 2)
        r["keys"] = r["console_change"] > 8.0

    checks = ("window", "focus", "capture_live", "hud", "mouse", "keys")
    print()
    print(f"{'inst':<5}{'display':<9}" + "".join(f"{c:<14}" for c in checks) + "notes")
    for r in results:
        cells = "".join(f"{('PASS' if r.get(c) else 'FAIL' if c in r else '-'):<14}" for c in checks)
        notes = []
        if "grab_ms_median" in r:
            notes.append(f"grab {r['grab_ms_median']} ms")
        if "hud_points" in r:
            notes.append(f"points {r['hud_points']}")
        if "turn_px" in r:
            notes.append(f"turned {r['turn_px']} px (expected ~{r['turn_expected_px']}), "
                         f"others {r['others_moved_px']}")
        if "console_change" in r:
            notes.append(f"console change {r['console_change']}")
        if "why" in r:
            notes.append(r["why"])
        print(f"{r['instance']:<5}{r['display']:<9}{cells}{'; '.join(notes)}")
    bad = [r["instance"] for r in results if not all(r.get(c) for c in checks)]
    others = [abs(v) for r in results for v in r.get("others_moved_px", {}).values()]
    if others and max(others) > max(3, expected_px / 3):
        print("\nWARNING: another instance's view moved during a turn -- input is not isolated (or a zombie "
              "walked through the frame; rerun to tell)")
    if bad:
        print(f"\nnot ready: instances {bad}. See docs/rl.md, 'When the spike fails'.")
        raise SystemExit(1)
    print(f"\nall {len(results)} instances take input alone and can be read. counts per degree for the run: "
          f"`calibrate_mouse.py --display {results[0]['display']} --xtest --full-turn`")
    for cap in captures.values():
        cap.close()
    for sink in sinks.values():
        sink.close()


if __name__ == "__main__":
    main()
