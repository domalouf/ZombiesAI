"""Train the inverse dynamics model: given the frames around a decision, what did the player press?

    uv run python scripts/train_idm.py data/demos --out runs/idm1

It trains on clips that already have actions -- demos recorded with input logging, or sim recordings from
`record_demo.py --source sim`. The model may see the future as well as the past, which is what makes the
problem tractable and what makes its labels worth more than a policy's guess.

Judge it on balanced per-head accuracy, not raw accuracy: 'no button' and 'no turn' are the majority labels
by a wide margin, and a model that always says so scores well and is useless.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.clips import clip_from_episode, iter_clips
from zombiesai.demos.idm import IDMConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", type=Path, help="directories of labelled clips")
    parser.add_argument("--episodes", nargs="*", type=Path, default=[], help="episode dirs to include as clips")
    parser.add_argument("--out", type=Path, default=Path("runs/idm"))
    parser.add_argument("--epochs", type=int, default=IDMConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=IDMConfig.batch_size)
    parser.add_argument("--lr", type=float, default=IDMConfig.lr)
    parser.add_argument("--before", type=int, default=IDMConfig.before)
    parser.add_argument("--after", type=int, default=IDMConfig.after)
    parser.add_argument("--hidden", type=int, default=IDMConfig.hidden)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips = [c for root in args.clips for c in iter_clips(root) if c.labelled]
    clips += [clip_from_episode(path) for path in args.episodes]
    if not clips:
        raise SystemExit("no labelled clips found; record some with scripts/record_demo.py")
    steps = sum(c.n_steps for c in clips)
    print(f"{len(clips)} clips, {steps:,} labelled decisions ({steps / 15 / 60:.1f} min of play)")

    config = IDMConfig(
        before=args.before, after=args.after, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, hidden=args.hidden, device=args.device, seed=args.seed,
    )
    checkpoint = train(clips, config, args.out)
    report = json.loads((args.out / "metrics.jsonl").read_text().splitlines()[-1])
    print(f"\ncheckpoint: {checkpoint}")
    balanced = report.get("val", {}).get("balanced", {})
    baseline = report.get("val", {}).get("majority_baseline", {})
    for head, value in balanced.items():
        print(f"  {head:>8}: balanced {value:.3f}  (always-majority baseline {baseline.get(head, float('nan')):.3f})")
    print("\nnext: scripts/label_clips.py to put these labels on unlabelled video")


if __name__ == "__main__":
    main()
