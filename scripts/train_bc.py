"""Behavioural cloning: learn to play from what a person did, seen only through the screen.

    uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1

Takes any mix of clips -- demos with logged input, video labelled by the inverse dynamics model, sim
recordings -- and fits one policy over pixels alone. The validation report is the one that matters:
per-frame accuracy against the majority baseline, whether the policy's own behaviour statistics land within
2x of the human's, and whether it has learned to copy its last action instead of looking at the screen.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.bc import BCConfig, train
from zombiesai.demos.clips import clip_from_episode, iter_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--episodes", nargs="*", type=Path, default=[], help="episode dirs to include as clips")
    parser.add_argument("--out", type=Path, default=Path("runs/bc"))
    parser.add_argument("--epochs", type=int, default=BCConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BCConfig.batch_size)
    parser.add_argument("--lr", type=float, default=BCConfig.lr)
    parser.add_argument("--frame-stack", type=int, default=BCConfig.frame_stack)
    parser.add_argument("--hidden", type=int, default=BCConfig.hidden)
    parser.add_argument("--min-confidence", type=float, default=BCConfig.min_confidence,
                        help="skip steps whose label is worth less than this")
    parser.add_argument("--prev-actions", action="store_true",
                        help="condition on the last two actions (watch the copy rate if you do)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips = [c for root in args.clips for c in iter_clips(root) if c.labelled]
    clips += [clip_from_episode(path) for path in args.episodes]
    if not clips:
        raise SystemExit("no labelled clips found; record demos or label video with the IDM first")
    steps = sum(int(c.usable(args.min_confidence).sum()) for c in clips)
    sources = sorted({c.label_source for c in clips})
    print(f"{len(clips)} clips, {steps:,} usable decisions ({steps / 15 / 3600:.2f} h), labels from {sources}")

    config = BCConfig(
        frame_stack=args.frame_stack, use_prev_actions=args.prev_actions, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, min_confidence=args.min_confidence,
        augment=not args.no_augment, device=args.device, seed=args.seed,
    )
    checkpoint = train(clips, config, args.out)
    report = json.loads((args.out / "report.json").read_text())
    final = report.get("final", {})
    print(f"\ncheckpoint: {checkpoint}")
    if final:
        accuracy = final["accuracy"]
        for head in accuracy["balanced"]:
            print(
                f"  {head:>8}: balanced {accuracy['balanced'][head]:.3f}  "
                f"raw {accuracy['per_head'][head]:.3f}  baseline {accuracy['majority_baseline'][head]:.3f}"
            )
        divergence = final["divergence"]
        verdict = "within 2x of the human" if divergence["passed"] else f"off on {sorted(divergence['failed'])}"
        print(f"  rollout statistics: {verdict}")
        inertia = final["inertia"]
        print(f"  action inertia: {'ok' if inertia['passed'] else 'copying on ' + str(inertia['inert_heads'])}")
    print("\nnext: scripts/eval_bc.py to play it, scripts/watch.py --checkpoint to see it")


if __name__ == "__main__":
    main()
