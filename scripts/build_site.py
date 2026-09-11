"""Build the public replay page for domalouf.com/zombies/: one game by a trained checkpoint, zero external requests."""

import argparse
from pathlib import Path

import torch

from zombiesai.rl.agent import PolicyAgent
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.viz.replay import record_replay, write_site_page

REPO_URL = "https://github.com/domalouf/ZombiesAI"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ppo-nacht-state-s1/checkpoint.pt"))
    parser.add_argument("--seed", type=int, default=10030, help="game seed; 10030 is one of its round-4 games")
    parser.add_argument("--out", type=Path, default=Path("site/zombies"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent = PolicyAgent(args.checkpoint)
    replay = record_replay(NachtSim(SimConfig(max_steps=100_000)), agent, seed=args.seed, agent_name="ppo")
    millions = round(agent.step / 1e6)
    intro = (
        "A reinforcement-learning agent learning to survive Nacht der Untoten, the first Call of Duty zombies map. "
        f"Its policy is PPO, written from scratch and trained for {millions} million steps in NachtSim, a simulator "
        "built from the map's own script. This is one of its better games: it reaches round 4 about once in 16. "
        "The agent never sees this map, only a 50-number state vector and the HUD."
    )
    index = write_site_page(
        replay,
        args.out,
        intro=intro,
        links=[("← domalouf.com", "/"), ("Code on GitHub", REPO_URL)],
        description="Watch a from-scratch PPO agent play Nacht der Untoten in simulation, move by move.",
    )
    s = replay["summary"]
    print(f"seed {args.seed}: survived {s['rounds_survived']} rounds in {s['game_time_s']:.0f}s -> {index}")


if __name__ == "__main__":
    main()
