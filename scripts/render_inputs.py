"""Render a recorded demo with your inputs drawn on it: keys, mouse buttons, wheel and look (viz/input_video.py).

    uv run python scripts/render_inputs.py data/demos/demo_0005
    uv run python scripts/render_inputs.py data/demos/demo_0005 --from 120 --to 180 --out clip.mp4
    uv run python scripts/render_inputs.py data/demos/demo_0005 data/demos/demo_0006   # several at once

The picture is the clip's 640x360 scene crop (or, for clips from before it, the 128x72 policy frames), at
1280x720, with the game's audio if it was recorded. The keys light at the input log's own timestamps. The
result is a new file, `<clip>/inputs.mp4` unless --out says otherwise; the clip itself is not touched.
"""

import argparse
from pathlib import Path

from zombiesai.viz.input_video import describe, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="start this many seconds in")
    parser.add_argument("--to", dest="end", type=float, default=None, help="stop this many seconds in")
    parser.add_argument("--out", type=Path, default=None, help="where to write it (one clip only)")
    parser.add_argument("--no-audio", action="store_true", help="leave the game's audio out")
    args = parser.parse_args()
    if args.out and len(args.clips) > 1:
        parser.error("--out takes one clip")
    for clip in args.clips:
        try:
            out = render(clip, args.out, start=args.start, end=args.end, audio=not args.no_audio)
        except ValueError as error:
            print(f"{clip}: {error}; skipped")
            continue
        info = describe(out)
        print(f"  -> {out} ({info['width']}x{info['height']}, {info['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
