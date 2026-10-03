"""Play this PC's games for a learner on another PC: one PPO run, several machines (docs/rl.md, "Several PCs").

On the learner PC (the one with the GPU that trains):

    ZOMBIES_FLEET_TOKEN=<secret> uv run python scripts/train_rl.py runs/bc_real3/bc.pt --actors 4 --listen :47860 --out runs/rl5

On every other gaming PC, same commit, same token -- nothing else:

    ZOMBIES_FLEET_TOKEN=<secret> uv run python scripts/fleet_worker.py

The worker listens for the learner's beacon (UDP 47861) and goes where it points -- `--learner host` instead
names it -- says hello (and is refused if its commit, spec or game settings differ from the learner's), fetches
the run's starting checkpoint and settings, starts as many games as this PC can carry (`--actors auto`; or a
number) if they are not up yet, one actor per game exactly as train_rl.py does, and from then on sends its
segments and pulls each new version of the weights. When the learner stops, the worker stops its actors (every
key released) and listens for the next run, so it can be left running (deploy/zombiesai-worker.service).

With --yield-to-games it steps aside while you play: a Steam game (or a Windows game through umu-run) running, or
~/.config/zombiesai/pause existing, stops its actors and its games until the PC has been free for a minute.

    touch ~/.config/zombiesai/pause      # the PC is yours until you remove it
"""

import argparse
import json
import os
import socket
import sys

from zombiesai.rl.discovery import BEACON_PORT
from zombiesai.rl.fleet import (DEFAULT_PORT, TOKEN_ENV, FleetClient, FleetError, FleetWorker, WorkerOptions,
                                describe, play_settings, token_from_env)


def actors_arg(value: str):
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: a number of games, or auto") from None
    if n < 1:
        raise argparse.ArgumentTypeError("at least one game")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learner", default=os.environ.get("ZOMBIES_FLEET_LEARNER"),
                        help=f"the learner's host[:port] (port {DEFAULT_PORT} by default; $ZOMBIES_FLEET_LEARNER). "
                             "Default: found by its beacon, and found again whenever it is lost")
    parser.add_argument("--run", help="with several learners announcing, join only this run's (default: the newest)")
    parser.add_argument("--beacon-port", type=int, default=BEACON_PORT, help=f"UDP (default: {BEACON_PORT})")
    parser.add_argument("--actors", type=actors_arg, default="auto",
                        help="games on this machine, one actor each; auto (the default): as many as fit now, "
                             "judged before every run (rl/capacity.py)")
    parser.add_argument("--name", default=socket.gethostname().split(".")[0],
                        help="how the learner tells this machine apart (default: its host name)")
    parser.add_argument("--fleet", default="runs/instances", help="this machine's fleet root (scripts/instances.py)")
    parser.add_argument("--out", default="runs/fleet", help="where each run's starting checkpoint and weights land")
    parser.add_argument("--manage-games", action=argparse.BooleanOptionalAction, default=True,
                        help="start the games a run needs if they are not up (default: on)")
    parser.add_argument("--down-after", type=float, metavar="MIN",
                        help="take the games down after this many minutes without a run (default: leave them up)")
    parser.add_argument("--yield-to-games", action="store_true",
                        help="leave the run while this PC's owner plays a game, or ~/.config/zombiesai/pause exists")
    parser.add_argument("--yield-keeps-games", action="store_true",
                        help="when yielding, stop only the actors; leave the games up")
    parser.add_argument("--pause-file", help="the file that pauses this worker (default ~/.config/zombiesai/pause)")
    parser.add_argument("--compress", action="store_true",
                        help="deflate segments before sending: for a PC on Wi-Fi (costs CPU here and on the learner)")
    parser.add_argument("--counts-per-degree", type=float, help="this machine's own, if it calibrated differently")
    parser.add_argument("--record-every", type=int, help="keep every k-th episode of each actor as a clip, here")
    parser.add_argument("--lost", type=float, default=60.0,
                        help="seconds without the learner before this machine's games stop (default: 60)")
    parser.add_argument("--once", action="store_true", help="play one run, then exit")
    parser.add_argument("--describe", action="store_true",
                        help="print this machine's commit, game settings, running games and room for games as "
                             "JSON, and exit (what scripts/fleet.py checks)")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.fleet)))
        return
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
    auto = args.actors == "auto"
    options = WorkerOptions(actors=1 if auto else args.actors, auto_actors=auto, fleet_root=args.fleet,
                            out_root=args.out, overrides=overrides, lost_s=args.lost, beacon_port=args.beacon_port,
                            run=args.run, manage_games=args.manage_games,
                            down_after_s=None if args.down_after is None else 60.0 * args.down_after,
                            yield_to_games=args.yield_to_games, yield_stops_games=not args.yield_keeps_games,
                            pause_file=args.pause_file, compress=args.compress)
    worker = FleetWorker(FleetClient(args.learner, token, args.name), options, settings=settings,
                         say=lambda m: print(m, flush=True))
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
