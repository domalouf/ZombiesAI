"""Behavioural cloning: learn to play from what a person did, seen only through the screen.

    uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1

Takes any mix of clips -- demos with logged input, video labelled by the inverse dynamics model, sim
recordings, and a policy's live runs (runs/play/play_NNNN, or all of runs/play) -- and fits one policy over
pixels alone. From a play run only the human's corrections are used: the steps where you took the controls
back, labelled from your input exactly like a demo (the policy's own steps, and the idle wait before you
handed back, are left out). They weigh --correction-weight times a demo step.

    uv run python scripts/train_bc.py data/demos runs/play --out runs/bc2

The validation report is the one that matters: per-frame accuracy against the majority baseline, whether the
policy's own behaviour statistics land within 2x of the human's, and whether it has learned to copy its last
action instead of looking at the screen.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.bc import BCConfig, train
from zombiesai.demos.clips import iter_clips
from zombiesai.demos.dataset import training_clips
from zombiesai.demos.hearing import DEFAULT_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/bc"))
    parser.add_argument("--epochs", type=int, default=BCConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BCConfig.batch_size)
    parser.add_argument("--lr", type=float, default=BCConfig.lr)
    parser.add_argument("--frame-stack", type=int, default=BCConfig.frame_stack)
    parser.add_argument("--frame-offsets", type=int, nargs="+", default=BCConfig.frame_offsets,
                        help="steps back of each input frame, e.g. 0 1 2 4 8 16 30 for two seconds of memory "
                             "(overrides --frame-stack)")
    parser.add_argument("--val-clips", nargs="*", type=Path, default=None,
                        help="validate on exactly these clip roots and train on all the others (default: "
                             "--val-fraction of every long recording, held out as ~2-minute blocks)")
    parser.add_argument("--val-fraction", type=float, default=BCConfig.val_fraction,
                        help="share of each recording held out when --val-clips isn't given")
    parser.add_argument("--focal-gamma", type=float, default=BCConfig.focal_gamma,
                        help="focal loss exponent: down-weights steps the policy already gets right (0 is "
                             "plain cross-entropy)")
    parser.add_argument("--class-balance", type=float, default=BCConfig.class_balance,
                        help="effective-number beta for per-class loss weights (0 turns class weighting off)")
    parser.add_argument("--class-balance-power", type=float, default=BCConfig.class_balance_power,
                        help="how hard class weighting corrects: 1 is full inverse frequency, 0.5 square "
                             "root; lower it if the policy reloads and swaps far more than you do")
    parser.add_argument("--hidden", type=int, default=BCConfig.hidden)
    parser.add_argument("--min-confidence", type=float, default=BCConfig.min_confidence,
                        help="skip steps whose label is worth less than this")
    parser.add_argument("--correction-weight", type=float, default=BCConfig.correction_weight,
                        help="loss weight of a human correction from a play run, against a demo step's 1")
    parser.add_argument("--prev-actions", action="store_true",
                        help="condition on the last two actions (watch the copy rate if you do)")
    parser.add_argument("--audio", action="store_true",
                        help="hear the game too: a stereo log-mel of the 0.5 s before each frame (clips "
                             "recorded without audio still train, with their audio masked out)")
    parser.add_argument("--audio-cache", type=Path, default=DEFAULT_CACHE,
                        help="where per-clip audio features are cached (never inside the clips)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips, empty = training_clips(args.clips, args.min_confidence)
    val_clips = None
    if args.val_clips is not None:
        val_clips = [c for root in args.val_clips for c in iter_clips(root) if c.labelled]
        held = {c.path.resolve() for c in val_clips}
        clips = [c for c in clips if c.path.resolve() not in held]
    if not clips:
        raise SystemExit("no labelled clips found; record demos or label video with the IDM first")
    steps = sum(int(c.usable(args.min_confidence).sum()) for c in clips)
    sources = sorted({c.label_source for c in clips})
    print(f"{len(clips)} clips, {steps:,} usable decisions ({steps / 15 / 3600:.2f} h), labels from {sources}")
    plays = [c for c in clips if c.is_play_run]
    if plays or empty:
        corrections = sum(int(c.usable(args.min_confidence).sum()) for c in plays)
        print(f"  of which {corrections:,} human corrections ({corrections / 15:.0f} s) from {len(plays)} play "
              f"runs, weighted {args.correction_weight:g}x; {len(empty)} clips with nothing usable skipped")

    config = BCConfig(
        frame_stack=args.frame_stack, frame_offsets=args.frame_offsets,
        use_prev_actions=args.prev_actions, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, min_confidence=args.min_confidence,
        correction_weight=args.correction_weight, augment=not args.no_augment, device=args.device, seed=args.seed,
        use_audio=args.audio, val_fraction=args.val_fraction, focal_gamma=args.focal_gamma,
        class_balance=args.class_balance, class_balance_power=args.class_balance_power,
    )
    if args.audio:
        heard = sum(c.audio() is not None for c in clips)
        print(f"hearing: {heard}/{len(clips)} clips have audio; the rest train with it masked out")
    checkpoint = train(clips, config, args.out, val_clips=val_clips, audio_cache=args.audio_cache)
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
    print("\nnext: scripts/eval_bc.py to score it, scripts/play_real.py to watch it play the game")


if __name__ == "__main__":
    main()
