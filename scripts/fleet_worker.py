"""Play this PC's games for a learner on another PC: one PPO run, several machines (docs/rl.md, "Several PCs").

On the learner PC (the one with the GPU that trains):

    ZOMBIES_FLEET_TOKEN=<secret> uv run python scripts/train_rl.py runs/bc_real3/bc.pt --actors 4 --listen :47860 --out runs/rl5

On every other gaming PC, same commit, same token, the games already up:

    uv run python scripts/instances.py up --n 4
    ZOMBIES_FLEET_TOKEN=<secret> uv run python scripts/fleet_worker.py --learner gamingpc.lan --actors 4

The worker says hello (and is refused if its commit, spec or game settings differ from the learner's), fetches
the run's starting checkpoint and settings, starts one actor per game exactly as train_rl.py does, and from then
on sends its segments and pulls each new version of the weights. When the learner stops, the worker stops its
actors (every key released) and waits for the next run, so it can be left running (deploy/zombiesai-worker.service).
"""

import argparse
import json
import os
import socket
import sys

from zombiesai.rl.fleet import (DEFAULT_PORT, TOKEN_ENV, FleetClient, FleetError, FleetWorker, WorkerOptions,
                                describe, play_settings, token_from_env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learner", default=os.environ.get("ZOMBIES_FLEET_LEARNER"),
                        help=f"the learner's host[:port] (port {DEFAULT_PORT} by default; $ZOMBIES_FLEET_LEARNER)")
    parser.add_argument("--actors", type=int, default=4, help="one per game instance on this machine")
    parser.add_argument("--name", default=socket.gethostname().split(".")[0],
                        help="how the learner tells this machine apart (default: its host name)")
    parser.add_argument("--fleet", default="runs/instances", help="this machine's fleet root (scripts/instances.py)")
    parser.add_argument("--out", default="runs/fleet", help="where each run's starting checkpoint and weights land")
    parser.add_argument("--counts-per-degree", type=float, help="this machine's own, if it calibrated differently")
    parser.add_argument("--record-every", type=int, help="keep every k-th episode of each actor as a clip, here")
    parser.add_argument("--lost", type=float, default=60.0,
                        help="seconds without the learner before this machine's games stop (default: 60)")
    parser.add_argument("--once", action="store_true", help="play one run, then exit")
    parser.add_argument("--describe", action="store_true",
                        help="print this machine's commit, game settings and running games as JSON, and exit "
                             "(what scripts/fleet.py checks)")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.fleet)))
        return
    if not args.learner:
        parser.error("--learner (or ZOMBIES_FLEET_LEARNER): the PC running train_rl.py --listen")
    token = token_from_env()
    if not token:
        parser.error(f"{TOKEN_ENV} must be set, to the same value as on the learner")

    overrides = {}
    if args.counts_per_degree is not None:
        overrides["counts_per_degree"] = args.counts_per_degree
    if args.record_every is not None:
        overrides["record_every"] = args.record_every
    settings = play_settings(args.fleet)
    if settings is None:
        print("no World at War config.cfg here: a learner training on the real game will refuse this machine",
              file=sys.stderr)
    worker = FleetWorker(FleetClient(args.learner, token, args.name),
                         WorkerOptions(actors=args.actors, fleet_root=args.fleet, out_root=args.out,
                                       overrides=overrides, lost_s=args.lost),
                         settings=settings, say=lambda m: print(m, flush=True))
    try:
        worker.run(once=args.once)
    except FleetError as error:
        raise SystemExit(f"the learner refused this machine: {error}")
    except FileNotFoundError as error:  # no fleet here yet
        raise SystemExit(str(error))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
