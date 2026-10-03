"""Evaluate a cloned policy on held-out recordings, the three ways per-frame accuracy cannot.

    uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos-heldout

1. Held-out accuracy, balanced per head, against the majority baseline.
2. What it would do -- fire duty cycle, mean |yaw|/s, reload rate -- from actions *sampled* on the held-out
   frames, against the human's own, which the plan asks to be within a factor of 2.
3. Action inertia: its copy rate against the human's action autocorrelation.

The fourth level, rounds survived, is the real game's to answer: `scripts/play_real.py` plays a checkpoint
against it, and an RL run's `episodes.jsonl` (scripts/train_rl.py) records every game's round.
"""

import argparse
import json
from pathlib import Path

import torch

from zombiesai.demos.agent import BCAgent
from zombiesai.demos.bc import evaluate
from zombiesai.demos.clips import iter_clips
from zombiesai.demos.dataset import ClipDataset, DataConfig
from zombiesai.demos.idm import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--clips", nargs="+", type=Path, required=True, help="held-out labelled clips")
    parser.add_argument("--seed", type=int, default=20_000, help="seeds the sampling of the policy's actions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="write the full report here as JSON")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent = BCAgent(args.checkpoint, device=args.device)
    report: dict = {"checkpoint": str(args.checkpoint), "epoch": agent.step}

    held_out = [c for root in args.clips for c in iter_clips(root) if c.labelled]
    if not held_out:
        raise SystemExit(f"no labelled clips under {', '.join(map(str, args.clips))}")
    data = ClipDataset(held_out, DataConfig(), offsets=agent.config.data_offsets)
    report["held_out"] = evaluate(agent.net, data, agent.config, resolve_device(args.device))
    accuracy = report["held_out"]["accuracy"]
    print(f"held-out clips: {len(data):,} steps")
    for head in accuracy["balanced"]:
        print(
            f"  {head:>8}: balanced {accuracy['balanced'][head]:.3f}  raw {accuracy['per_head'][head]:.3f}  "
            f"baseline {accuracy['majority_baseline'][head]:.3f}"
        )

    behaviour = report["held_out"]["behaviour"]
    policy, human = behaviour["policy"], behaviour["human"]
    divergence = report["held_out"]["divergence"]
    print("\nbehaviour vs the human it learned from (sampled on the same frames):")
    for key, ratio in divergence["ratios"].items():
        mark = "ok " if ratio <= divergence["factor"] else "OFF"
        print(f"  {mark} {key:>20}: policy {policy.get(key, float('nan')):.3f} "
              f"vs human {human.get(key, float('nan')):.3f}  ({ratio:.1f}x)")
    print(f"  -> {'within 2x on every statistic' if divergence['passed'] else 'diverged'}")
    inertia = report["held_out"]["inertia"]
    print(f"  copy rate {inertia['policy']['copy_joint']:.2f} joint vs human {inertia['human']['copy_joint']:.2f}"
          + ("" if inertia["passed"] else f"; inert heads: {', '.join(inertia['inert_heads'])}"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
