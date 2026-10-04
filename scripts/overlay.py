"""Draw what the policy was thinking over a kept film (viz/overlay.py): its value estimate, how sure it was,
what it pressed, and each points change.

    uv run python scripts/overlay.py runs/video/firsts/box.mp4
    uv run python scripts/overlay.py runs/rl9/best/best.mp4 --from 120 --to 180 --out best-2min.mp4

The film's sidecar (`<film>.brain.jsonl`, written beside every film the actors keep) is read from beside it.
The result is a new file, `<film>.overlay.mp4` unless --out says otherwise; the film itself is left as it is.
"""

import argparse
from pathlib import Path

from zombiesai.viz.overlay import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("films", nargs="+", type=Path)
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="start this many seconds in")
    parser.add_argument("--to", dest="end", type=float, default=None, help="stop this many seconds in")
    parser.add_argument("--out", type=Path, default=None, help="where to write it (one film only)")
    args = parser.parse_args()
    if args.out and len(args.films) > 1:
        parser.error("--out takes one film")
    for film in args.films:
        if not film.with_suffix(".brain.jsonl").exists():
            print(f"{film}: no sidecar beside it ({film.with_suffix('.brain.jsonl').name}); skipped")
            continue
        out = args.out or film.with_name(film.stem + ".overlay.mp4")
        print(f"{film} -> {render(film, out, start=args.start, end=args.end)}")


if __name__ == "__main__":
    main()
