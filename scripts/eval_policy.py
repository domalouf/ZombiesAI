"""Evaluate a trained PPO checkpoint: mean return over N episodes, plus rounds reached for NachtSim."""

import argparse

import numpy as np
import torch

from zombiesai.rl.agent import PolicyAgent
from zombiesai.rl.envs import PRESETS, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true", help="take each head's most likely action")
    parser.add_argument("--seed", type=int, default=10_000, help="first episode seed (kept apart from training)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent = PolicyAgent(args.checkpoint, deterministic=args.deterministic)
    preset = PRESETS[agent.env]
    env = make_env(agent.env)
    returns, rounds = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
        if "round_reached" in info:
            rounds.append(info["round_reached"])

    r = np.array(returns)
    sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    print(f"{agent.env} after {agent.step:,} training steps, {args.episodes} episodes:")
    print(f"  return {r.mean():.2f} +/- {sem:.2f} (sem), min {r.min():.1f}, max {r.max():.1f}")
    if rounds:
        rr = np.array(rounds)
        sem = rr.std(ddof=1) / np.sqrt(len(rr))
        print(f"  round reached {rr.mean():.2f} +/- {sem:.2f}, histogram {np.bincount(rr)[1:].tolist()}")
    if preset.solved_at is not None:
        print(f"  solved at {preset.solved_at}: {'SOLVED' if r.mean() >= preset.solved_at else 'not solved'}")


if __name__ == "__main__":
    main()
