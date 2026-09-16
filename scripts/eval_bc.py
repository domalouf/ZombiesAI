"""Evaluate a cloned policy the four ways per-frame accuracy cannot.

    uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos --episodes 20

1. Held-out accuracy, balanced per head, against the majority baseline.
2. What it does over a rollout -- fire duty cycle, mean |yaw|/s, reload rate -- against the human's own,
   which the plan asks to be within a factor of 2.
3. Action inertia: its copy rate against the human's action autocorrelation.
4. Rounds survived in NachtSim, which is a smoke test rather than a score: the sim's raycast view is a crude
   stand-in for the real screen and visual transfer is a non-goal (docs/sim_lies.md).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zombiesai.demos import stats
from zombiesai.demos.agent import BCAgent
from zombiesai.demos.bc import evaluate
from zombiesai.demos.clips import iter_clips
from zombiesai.demos.dataset import ClipDataset, DataConfig
from zombiesai.demos.idm import resolve_device
from zombiesai.sim.nacht_sim import NachtSim, SimConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--clips", nargs="*", type=Path, default=[], help="held-out labelled clips")
    parser.add_argument("--episodes", type=int, default=10, help="NachtSim games to play")
    parser.add_argument("--max-steps", type=int, default=9_000)
    parser.add_argument("--hardness", type=float, default=0.5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=20_000, help="first episode seed, kept apart from training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="write the full report here as JSON")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent = BCAgent(args.checkpoint, deterministic=args.deterministic, device=args.device)
    report: dict = {"checkpoint": str(args.checkpoint), "epoch": agent.step}
    human = agent.meta.get("human_behaviour", {})

    held_out = [c for root in args.clips for c in iter_clips(root) if c.labelled]
    if held_out:
        data = ClipDataset(held_out, DataConfig(), before=agent.config.frame_stack - 1)
        report["held_out"] = evaluate(agent.net, data, agent.config, resolve_device(args.device))
        accuracy = report["held_out"]["accuracy"]
        print(f"held-out clips: {len(data):,} steps")
        for head in accuracy["balanced"]:
            print(
                f"  {head:>8}: balanced {accuracy['balanced'][head]:.3f}  raw {accuracy['per_head'][head]:.3f}  "
                f"baseline {accuracy['majority_baseline'][head]:.3f}"
            )
        human = report["held_out"]["behaviour"]["human"]

    rounds, actions, returns = [], [], []
    for episode in range(args.episodes):
        env = NachtSim(SimConfig(hardness=args.hardness, max_steps=args.max_steps, obs_profile="render"))
        obs, _ = env.reset(seed=args.seed + episode)
        agent.reset()
        total, done, taken = 0.0, False, []
        while not done:
            action = agent.act(obs)
            taken.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rounds.append(info["round_reached"])
        returns.append(total)
        actions.append(np.stack(taken))
        print(f"  episode {episode}: round {info['round_reached']}, return {total:.1f}, {len(taken)} steps")

    played = np.concatenate(actions)
    policy_stats = stats.behaviour_stats(played)
    report["sim"] = {
        "rounds": rounds,
        "round_mean": float(np.mean(rounds)),
        "return_mean": float(np.mean(returns)),
        "behaviour": policy_stats,
    }
    r = np.array(rounds)
    sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    print(f"\nNachtSim: round {r.mean():.2f} +/- {sem:.2f} over {len(r)} episodes, return {np.mean(returns):.1f}")

    if human:
        divergence = stats.divergence_report(policy_stats, human)
        report["divergence"] = divergence
        print("\nbehaviour vs the human it learned from:")
        for key, ratio in divergence["ratios"].items():
            mark = "ok " if ratio <= divergence["factor"] else "OFF"
            print(f"  {mark} {key:>20}: policy {policy_stats.get(key, float('nan')):.3f} "
                  f"vs human {human.get(key, float('nan')):.3f}  ({ratio:.1f}x)")
        print(f"  -> {'within 2x on every statistic' if divergence['passed'] else 'diverged'}")
        copy = stats.copy_rates(played)
        print(f"  copy rate {copy['copy_joint']:.2f} joint vs human {human.get('copy_joint', float('nan')):.2f}")
        report["copy_rates"] = copy

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
