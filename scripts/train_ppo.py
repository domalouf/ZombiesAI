"""Train the from-scratch PPO on a preset: cartpole or lunarlander (correctness checks), or nacht-state."""

import argparse
import time
from pathlib import Path

from zombiesai.rl.envs import PRESETS
from zombiesai.rl.ppo import PPOConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=sorted(PRESETS))
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--hardness", type=float, help="NachtSim sim_hardness (default 0.5)")
    parser.add_argument("--run-name", help="default: ppo-<env>-s<seed>-<timestamp>")
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in {
            "total_steps": args.total_steps,
            "num_envs": args.num_envs,
            "lr": args.lr,
            "ent_coef": args.ent_coef,
            "gamma": args.gamma,
        }.items()
        if v is not None
    }
    if args.hardness is not None:
        overrides["sim"] = {"hardness": args.hardness}
    cfg = PPOConfig.for_preset(args.env, seed=args.seed, **overrides)
    name = args.run_name or f"ppo-{args.env}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint = train(cfg, Path("runs") / name)
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
