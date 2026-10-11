"""Keep a run's best game on the site for good (viz/saved_games.py), before the run does better and replaces it.

    uv run python scripts/save_game.py runs/rl10 rl10-grenade --title "Took two with him" \\
        --caption "Its grenade kills two zombies, and the agent with them."
    uv run python scripts/save_game.py --list

It goes up under "Saved games" with the next deploy/deploy.sh.
"""

import argparse
import time
from pathlib import Path

from zombiesai.viz.saved_games import SAVED_DIR, read_index, save_game

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="the run directory whose best game to keep")
    parser.add_argument("name", nargs="?", help="its name on the site (saved/<name>.mp4): lower case, digits, dashes")
    parser.add_argument("--title", help="what the site calls it")
    parser.add_argument("--caption", default="", help="a line about what happens in it")
    parser.add_argument("--saved", type=Path, default=REPO / "runs" / SAVED_DIR,
                        help="where saved games live (default: runs/saved)")
    parser.add_argument("--list", action="store_true", help="list the saved games and stop")
    args = parser.parse_args()

    if args.list:
        for name, entry in read_index(args.saved).items():
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.get("saved") or 0))
            print(f"{name:24} {when}  {entry.get('run')}: round {entry.get('round')}, {entry.get('kills')} kills  "
                  f"{entry.get('title')}")
        return
    if not (args.run and args.name and args.title):
        parser.error("give the run, a name and --title (or --list)")
    entry = save_game(args.run, args.saved, args.name, args.title, args.caption)
    print(f"saved {args.run.name}'s best game as {args.saved / entry['name']}.mp4: round {entry['round']}, "
          f"{entry['kills']} kills, {entry['points']} points. deploy/deploy.sh puts it on the site.")


if __name__ == "__main__":
    main()
