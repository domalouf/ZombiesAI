"""Pack recordings' raw HUD crops as verified video, ~17x smaller (see demos/hud_video.py).

    uv run python scripts/pack_hud.py data/demos            # every clip under it
    uv run python scripts/pack_hud.py data/demos/demo_0002  # one clip

record_demo.py and play_real.py already do this when a session ends; this is for recordings made before
they did, or one whose packing was interrupted. Each region's raw file is deleted only after every chunk
of its video has been decoded back and compared with it. Already-packed regions are skipped.
"""

import argparse
import sys
from pathlib import Path

from zombiesai.demos.hud_video import CRF, PackError, pack_hud


def clip_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "clip.json").exists():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.rglob("clip.json"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="clip directories, or directories holding clips")
    parser.add_argument("--crf", type=int, default=CRF, help="H.264 quality: lower is bigger and closer")
    parser.add_argument("--keep-raw", action="store_true", help="write the video but keep the raw crops too")
    args = parser.parse_args()

    failed = 0
    for clip_dir in clip_dirs(args.paths):
        before = sum(p.stat().st_size for p in clip_dir.rglob("*") if p.is_file())
        try:
            written = pack_hud(clip_dir, crf=args.crf, keep_raw=args.keep_raw)
        except PackError as e:
            print(f"{clip_dir}: {e}", file=sys.stderr)
            failed += 1
            continue
        after = sum(p.stat().st_size for p in clip_dir.rglob("*") if p.is_file())
        if written:
            print(f"{clip_dir}: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")
        else:
            print(f"{clip_dir}: nothing to pack")
    if failed:
        raise SystemExit(f"{failed} clip(s) kept raw crops; see above")


if __name__ == "__main__":
    main()
