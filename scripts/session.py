"""Start training on this PC, see what is running, and stop all of it with everything saved (zombiesai/session.py).
`./zai` at the top of the checkout runs this.

    ./zai start                      # games up, then continue the newest real-game run as runs/rl<next>
    ./zai start --games 6 --watch    # six games, and open the viewers on workspace 9
    ./zai start --from runs/bc1/bc.pt --name rl-bc1
    ./zai start --from fresh         # a new pixel+audio policy, no BC prior
    ./zai start -- --lr 1e-4         # anything after -- goes to scripts/train_rl.py as it is
    ./zai status                     # the trainer's progress, the games, what else is open
    ./zai stop                       # trainer saves its checkpoint and stops; viewers, dashboard, games closed
    ./zai stop --keep-games          # the trainer only: games stay up for the next start

`start` refuses while a run is already playing the games. `stop` waits for each trainer to write its checkpoint
and for its actors to finish (a few seconds; up to 150 s while one encodes a best game's film), kills it only after
--timeout, then takes down the games, their X servers and sound sinks, the viewers and the dashboard, and anything
of ours left over. It ends by listing what each stopped run saved. Safe to run when nothing is running.
"""

import argparse
import os
import sys
from pathlib import Path

from zombiesai import session

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("start", help="bring the games up and start training")
    up.add_argument("--from", dest="init", default=None,
                    help="checkpoint to start from (default: the newest real-game run's checkpoint.pt, else the "
                         "newest bc.pt, else fresh), or 'fresh'")
    up.add_argument("--games", type=int, default=None,
                    help="games to play (default: the fleet's size, or what this PC can carry)")
    up.add_argument("--name", default=None, help="the run's name under runs/ (default: rl<next number>)")
    up.add_argument("--watch", action="store_true", help="open the game viewers once training starts")
    up.add_argument("train_args", nargs=argparse.REMAINDER, help="after --: more scripts/train_rl.py flags")
    down = sub.add_parser("stop", help="stop training gracefully, save everything, close everything")
    down.add_argument("--timeout", type=float, default=session.STOP_TIMEOUT_S,
                      help=f"seconds to wait for a trainer before killing it (default {session.STOP_TIMEOUT_S:.0f})")
    down.add_argument("--keep-games", action="store_true", help="stop the trainer only; leave the games up")
    sub.add_parser("status", help="what is running")
    args = parser.parse_args()

    os.chdir(REPO)  # runs/ and the fleet's root are relative to the checkout
    if args.command == "start":
        extra = [a for a in args.train_args if a != "--"]
        return session.start(REPO, init=args.init, games=args.games, name=args.name, extra=" ".join(extra),
                             watch=args.watch)
    if args.command == "stop":
        return session.stop(REPO, timeout_s=args.timeout, keep_games=args.keep_games)
    return session.status(REPO)


if __name__ == "__main__":
    sys.exit(main())
