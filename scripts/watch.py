"""Play one NachtSim episode and write a self-contained HTML replay you can watch in any browser."""

import argparse
import webbrowser
from pathlib import Path

from zombiesai.agents.random_agent import RandomAgent
from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.viz.replay import record_replay, write_replay_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardness", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--out", type=Path, help="default: runs/replays/<agent>-seed<seed>.html")
    parser.add_argument("--no-open", action="store_true", help="don't open the replay in a browser")
    args = parser.parse_args()

    agent = ScriptedAgent() if args.agent == "scripted" else RandomAgent(args.seed)
    env = NachtSim(SimConfig(hardness=args.hardness, max_steps=args.max_steps))
    replay = record_replay(env, agent, seed=args.seed, agent_name=args.agent)
    out = write_replay_html(replay, args.out or Path("runs/replays") / f"{args.agent}-seed{args.seed}.html")

    s = replay["summary"]
    ending = f"survived {s['rounds_survived']} rounds" if s["ended"] == "game_over" else "hit the step cap"
    print(f"{args.agent} agent, seed {args.seed}: {ending} in {s['game_time_s']:.0f}s of game time")
    print(f"replay: {out.resolve()}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
