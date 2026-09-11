"""M1 exit check: mean rounds reached by the scripted vs random agents over N episodes each."""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp

import numpy as np

AGENTS = ("random", "scripted")


def _episode(task: tuple[str, int, float, int]) -> tuple[str, int, bool]:
    from zombiesai.agents.random_agent import RandomAgent
    from zombiesai.agents.scripted import ScriptedAgent
    from zombiesai.rollout import run_episode
    from zombiesai.sim.nacht_sim import NachtSim, SimConfig

    name, seed, hardness, max_steps = task
    agent = RandomAgent(seed) if name == "random" else ScriptedAgent()
    summary = run_episode(NachtSim(SimConfig(hardness=hardness, max_steps=max_steps)), agent, seed=seed)
    return name, summary["round_reached"], summary["terminated"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--hardness", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=500_000)
    args = parser.parse_args()

    tasks = [(name, seed, args.hardness, args.max_steps) for name in AGENTS for seed in range(args.episodes)]
    rounds: dict[str, list[int]] = {name: [] for name in AGENTS}
    censored = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, reached, terminated in pool.imap_unordered(_episode, tasks):
            rounds[name].append(reached)
            censored += not terminated

    for name in AGENTS:
        r = np.array(rounds[name])
        print(
            f"{name:>9}: mean round {r.mean():.2f} +/- {r.std(ddof=1) / np.sqrt(len(r)):.2f} (sem)  "
            f"median {np.median(r):.0f}  max {r.max()}  histogram {np.bincount(r)[1:].tolist()}"
        )
    gap = np.mean(rounds["scripted"]) - np.mean(rounds["random"])
    if censored:
        print(f"note: {censored} episodes hit --max-steps, so their round counts are lower bounds")
    print(f"\nM1 target: scripted beats random by >=2 rounds -> {'PASS' if gap >= 2 else 'FAIL'} (gap {gap:.2f})")


if __name__ == "__main__":
    main()
