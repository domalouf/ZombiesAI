"""Pseudo-label ingested video with a trained inverse dynamics model.

    uv run python scripts/label_clips.py runs/idm1/idm.pt data/clips/session1

Every clip gets a labels.npz with one action per step and the model's confidence in it. Confidence is what
behavioural cloning uses to weight the step, and `--min-confidence` marks the worst steps so they can be
dropped entirely. Check the reported behaviour statistics against a human's before training on the result:
a model that has labelled an hour of footage as "never fires" has told you something is wrong.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.clips import iter_clips
from zombiesai.demos.idm import label_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="flag steps below this so training skips them")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--relabel", action="store_true", help="also overwrite clips that already have labels")
    args = parser.parse_args()

    clips = [c for root in args.clips for c in iter_clips(root)]
    if not args.relabel:
        clips = [c for c in clips if not c.labelled or c.label_source == "idm"]
    if not clips:
        raise SystemExit("nothing to label (use --relabel to overwrite human-labelled clips)")

    summaries = label_clips(
        args.checkpoint, clips, device=args.device, batch_size=args.batch_size, min_confidence=args.min_confidence
    )
    steps = sum(s["steps"] for s in summaries)
    kept = sum(s["kept"] for s in summaries)
    for s in summaries:
        b = s["behaviour"]
        print(
            f"{Path(s['clip']).name}: {s['steps']:>6} steps, confidence {s['mean_confidence']:.3f}, "
            f"fire {b['fire_duty']:.2f}, |yaw| {b['abs_yaw_deg_per_s']:.0f} deg/s, "
            f"reloads {b['reload_per_min']:.1f}/min"
        )
    print(f"\n{kept:,}/{steps:,} steps kept ({kept / max(steps, 1):.1%}) across {len(summaries)} clips")
    Path(args.clips[0], "labels_report.json").write_text(json.dumps(summaries, indent=2))
    print("next: scripts/train_bc.py")


if __name__ == "__main__":
    main()
