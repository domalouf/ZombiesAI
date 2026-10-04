"""Read the HUD of recorded clips -- demos or AI play runs -- and summarise each one.

    uv run python scripts/parse_hud.py data/demos/demo_0002          # writes data/demos/demo_0002/hud.npz
    uv run python scripts/parse_hud.py data/demos runs/play           # every clip with HUD crops under them
    uv run python scripts/parse_hud.py runs/play --out /tmp/hud       # write to /tmp/hud/<clip>/ instead
    uv run python scripts/parse_hud.py data/demos --dry-run           # summaries only, write nothing
    uv run python scripts/parse_hud.py data/demos --lowres --dry-run  # clips without crops: round only

Per clip: hud.npz (per-step points, round, grenades, reserve, magazine and weapon with confidences and statuses,
plus the temporally checked tracks) and hud_summary.json (games, highest round, score, points per minute,
game overs, read and consistency rates). See docs/hud.md for what the numbers mean. The table at the end
is the comparison across clips.
"""

import argparse
import json
import sys
from pathlib import Path

from zombiesai.hud.clip_hud import parse_clip, write_hud
from zombiesai.hud.parse import HudParser


def clip_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "clip.json").exists():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.rglob("clip.json"))
    return found


def row(name: str, s: dict) -> str:
    rates = s["read"]
    return (f"{name:<14} {s['seconds'] / 60:6.1f} {len(s['games']):5d} {s['game_overs']:5d} {s['highest_round']:5d} "
            f"{s['best_score']:7d} {s['peak_points']:7d} {s['points_per_minute']:7.0f} "
            f"{rates['points']['read_rate']:6.1%} {rates['points']['consistent_rate']:6.1%} "
            f"{rates['round']['read_rate']:6.1%} {s['parse_ms_per_step']:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="clip directories, or directories holding clips")
    ap.add_argument("--out", type=Path, help="write <out>/<clip name>/hud.npz instead of into each clip")
    ap.add_argument("--dry-run", action="store_true", help="parse and summarise, write nothing")
    ap.add_argument("--json", action="store_true", help="print each summary as JSON")
    ap.add_argument("--lowres", action="store_true",
                    help="for clips without HUD crops, read the round alone from the 128x72 frames")
    args = ap.parse_args()

    parser = HudParser()
    rows = []
    for clip_dir in clip_dirs(args.paths):
        try:
            parsed, check, summary = parse_clip(clip_dir, parser, lowres=args.lowres)
        except ValueError as e:
            print(f"{clip_dir}: skipped: {e}", file=sys.stderr)
            continue
        if not args.dry_run:
            dest = (args.out / clip_dir.name) if args.out else clip_dir
            print(f"{clip_dir}: wrote {write_hud(dest, parsed, check, summary)}")
        if args.json:
            print(json.dumps(summary, indent=2))
        rows.append(row(clip_dir.name, summary))
        if summary["hud_source"] != "hud_crops":
            print(f"  from {summary['hud_source']}: round read on {summary['read']['round']['read_rate']:.0%} of steps")
        for i, g in enumerate(summary["games"]):
            end = "game over" if g["game_over"] else "still alive at the end"
            score = f", score {g['score']}" if g["score"] >= 0 else ""
            print(f"  game {i + 1}: round {g['rounds_reached']}, {g['seconds'] / 60:.1f} min, peak {g['peak_points']} "
                  f"points, gained {g['points_gained']}, spent {g['points_spent']}, downs {g['downs']}{score} ({end})")
    if rows:
        print(f"\n{'clip':<14} {'min':>6} {'games':>5} {'overs':>5} {'round':>5} {'score':>7} {'peak':>7} {'pts/min':>7} "
              f"{'pts rd':>6} {'pts ok':>6} {'rnd rd':>6} {'ms/st':>6}")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
