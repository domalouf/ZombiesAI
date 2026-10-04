"""List the films kept for the video about training (rl/keepsakes.py), each shelf in the order training
reached them: the firsts, the hourly check-ins and the records.

    uv run python scripts/video_films.py                 # runs/video
    uv run python scripts/video_films.py runs/rl3/video  # a rehearsal's own
"""

import argparse
import time
from pathlib import Path

from zombiesai.rl.clock import TrainingClock
from zombiesai.rl.keepsakes import Keepsakes


def when(record: dict) -> str:
    clock = record.get("clock") or record.get("start_clock")
    if clock:
        return TrainingClock.from_dict(clock).label()
    t = record.get("t_unix") or record.get("recorded_unix")
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, nargs="?", default=Path("runs/video"))
    args = parser.parse_args()

    shelves = Keepsakes(args.dir).shelves()
    any_kept = False
    for name, shelf in shelves.items():
        kept = shelf.read()
        if not kept:
            continue
        any_kept = True
        order = sorted(kept.items(), key=lambda kv: ((kv[1].get("clock") or kv[1].get("start_clock") or {})
                                                    .get("train_s", 0), kv[1].get("t_unix") or 0))
        print(f"\n{name} ({len(kept)})")
        for key, record in order:
            title = record.get("title") or f"check-in {key}"
            value = f"  {record['value']:g} {record.get('unit', '')}".rstrip() if "value" in record else ""
            sidecar = "" if record.get("brain") else "  (no sidecar)"
            print(f"  {when(record):<24} {title + value:<48} {shelf.dir / (record.get('clip') or '?')}{sidecar}")
    if not any_kept:
        raise SystemExit(f"nothing kept in {args.dir} yet")


if __name__ == "__main__":
    main()
