"""Sample the points_ammo HUD crop off running game instances, for harvesting templates (weapon names).

Read-only: it grabs each instance's window every `--every` seconds and saves the crop the env's own capture
cuts (demos/hud_crops.py), so it can run beside training without touching the games. The agents' games are
where the guns they actually use show up; a recording (record_demo.py) has the same crops for a human's.

    uv run python scripts/sample_hud.py --minutes 30 --out data/hud_samples/live.npz
    uv run --with pillow python scripts/build_hud_atlas.py harvest-weapons data/hud_samples/live.npz --out /tmp/weapons
"""

import argparse
import time
from pathlib import Path

import numpy as np

from zombiesai.demos.hud_crops import HUD_REGIONS, crop_regions
from zombiesai.hud.parse import HudParser
from zombiesai.realgame.instances import FleetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help=".npz to write")
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--every", type=float, default=1.5, help="seconds between rounds of grabs")
    parser.add_argument("--instances", type=int, default=0, help="how many (default: the fleet config's n)")
    parser.add_argument("--display-base", type=int, default=None)
    args = parser.parse_args()

    from zombiesai.demos.x11_capture import X11Grabber

    config = FleetConfig()
    n = args.instances or config.n
    base = config.display_base if args.display_base is None else args.display_base
    grabbers: dict[int, X11Grabber | None] = {i: None for i in range(n)}
    region = {"points_ammo": HUD_REGIONS["points_ammo"]}
    reader = HudParser()
    crops, sources = [], []
    end = time.monotonic() + args.minutes * 60
    while time.monotonic() < end:
        for i, grabber in list(grabbers.items()):
            try:
                if grabber is None:
                    grabber = grabbers[i] = X11Grabber(window=config.window_title, display=f":{base + i}")
                frame = grabber.grab()
            except Exception:  # noqa: BLE001 -- no game there yet, or it is restarting: try again next round
                grabbers[i] = None
                continue
            scale = min(1.0, 720.0 / frame.shape[0])  # the env's own capture scale (realgame/instances.py)
            crop = crop_regions(frame, region, scale)["points_ammo"]
            crops.append(crop)
            sources.append(f"i{i}@{time.time():.1f}")
        if crops and len(crops) % (20 * n) < n:
            r = reader.parse({"points_ammo": crops[-1]})
            print(f"{len(crops)} crops; last: points {r.points} weapon {r.weapon} mag {r.mag} reserve {r.reserve}")
        time.sleep(args.every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, points_ammo=np.stack(crops), source=np.array(sources))
    print(f"wrote {len(crops)} crops to {args.out}")


if __name__ == "__main__":
    main()
