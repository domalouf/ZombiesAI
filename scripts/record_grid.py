"""Film every running game at once, tiled in one grid (viz/grid.py): the "it plays them all at the same time"
shot for the video about training.

    uv run python scripts/record_grid.py --minutes 2            # -> runs/video/grid/grid-<date>.mp4
    uv run python scripts/record_grid.py 60 61 62 63 --seconds 30 --out grid.mp4

It only reads the games' pictures, as the viewers do; training goes on undisturbed. Ctrl-C stops it early and
keeps what was filmed.
"""

import argparse
import time
from pathlib import Path

from zombiesai.viz.grid import record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("displays", nargs="*", type=int, help="display numbers to film (default: every instance)")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--seconds", type=float)
    length.add_argument("--minutes", type=float)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-width", type=int, default=1920, help="the grid's widest (default: 1920)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    seconds = args.seconds or 60 * (args.minutes or 1.0)
    out = args.out or Path("runs/video/grid") / time.strftime("grid-%Y%m%d-%H%M%S.mp4")
    displays = [f":{d}" for d in args.displays] or None
    print(record(out, seconds, fps=args.fps, max_width=args.max_width, displays=displays))


if __name__ == "__main__":
    main()
