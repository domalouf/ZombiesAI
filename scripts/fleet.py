"""Get the other gaming PCs ready to play for this one, over SSH (rl/fleet_admin.py; docs/rl.md, "Several PCs").

Run on the training PC, before train_rl.py --listen:

    uv run python scripts/fleet.py prep rig2.lan                 # one PC, 4 games
    uv run python scripts/fleet.py prep rig2.lan=2 rig3.lan      # 2 games on rig2, 4 on rig3
    uv run python scripts/fleet.py check                         # read-only: would the learner take them?

For each PC: check out this machine's commit (it must be pushed; a PC with uncommitted changes is left alone),
uv sync, install this machine's config.cfg for its games (its Steam profile is not touched), start its games --
restarting them if their settings changed -- restart its zombiesai-worker if the code or settings changed, then
check it as the learner's hello will. Hosts are anything ssh takes; $ZOMBIES_FLEET_HOSTS holds the usual list.
"""

import argparse
import os
import sys

from zombiesai.rl.fleet_admin import Learner, format_report, parse_targets, run_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prep", "check"))
    parser.add_argument("hosts", nargs="*", help="host or host=N (N games there); default: $ZOMBIES_FLEET_HOSTS")
    parser.add_argument("--actors", type=int, default=4, help="games per PC unless host=N says (default: 4)")
    parser.add_argument("--repo-dir", default="~/Projects/ZombiesAI", help="the checkout's path on the PCs")
    parser.add_argument("--fleet", default="runs/instances", help="the fleet's root on the PCs, and here")
    parser.add_argument("--no-games", action="store_true", help="leave the games alone (code and settings only)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="prep even though this checkout has uncommitted changes (the PCs get the last commit)")
    args = parser.parse_args()

    specs = args.hosts or os.environ.get("ZOMBIES_FLEET_HOSTS", "").split()
    if not specs:
        parser.error("which PCs? name them, or set ZOMBIES_FLEET_HOSTS")
    try:
        targets = parse_targets(specs, actors=args.actors, repo=args.repo_dir, fleet=args.fleet)
    except ValueError as error:
        parser.error(str(error))

    learner = Learner.here(args.fleet)
    if learner.sha == "unknown":
        raise SystemExit("this checkout has no git commit to hand out")
    if learner.dirty and args.command == "check":
        print("note: this checkout has uncommitted changes, so a PC on the same commit still runs different code")
    elif learner.dirty and not args.allow_dirty:
        raise SystemExit("this checkout has uncommitted changes: the PCs can only get the last commit, so they "
                         "would train on different code under the same commit. Commit and push, or --allow-dirty.")
    if learner.config is None:
        print("no config.cfg here (no Steam profile, nothing installed in the fleet's root): each PC keeps its own")
    print(f"{args.command}: {', '.join(t.host for t in targets)} -> commit {learner.sha[:8]}", flush=True)
    reports = run_all(targets, learner, mode=args.command, games=not args.no_games,
                      say=lambda m: print(m, flush=True))
    print()
    for report in reports:
        print(format_report(report))
    if not all(r.ok for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
