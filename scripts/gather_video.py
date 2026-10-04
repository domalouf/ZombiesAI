"""Bring the video's films home from the other PCs that played for the learner (rl/keepsakes.py `merge`).

    uv run python scripts/gather_video.py                    # every PC in $ZOMBIES_FLEET_HOSTS
    uv run python scripts/gather_video.py gamer2 gamer3      # these (anything ssh takes)

A fleet worker keeps its games' firsts, check-ins and records in its own `runs/fleet/video`. This copies each
PC's over SSH (rsync, into runs/video/.remote/<host>/, so a second gather only fetches what is new) and merges
them into runs/video here: a first another PC reached earlier replaces ours, a record it beat replaces ours,
and so on. Nothing on the other PCs is changed.
"""

import argparse
import os
import subprocess
from pathlib import Path

from zombiesai.rl.keepsakes import merge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hosts", nargs="*", help="default: $ZOMBIES_FLEET_HOSTS (host or host=N, as scripts/fleet.py)")
    parser.add_argument("--repo-dir", default="~/Projects/ZombiesAI", help="the checkout's path on the PCs")
    parser.add_argument("--remote-dir", default="runs/fleet/video", help="the workers' video directory, in the repo")
    parser.add_argument("--into", type=Path, default=Path("runs/video"))
    args = parser.parse_args()
    hosts = [h.partition("=")[0] for h in (args.hosts or os.environ.get("ZOMBIES_FLEET_HOSTS", "").split())]
    if not hosts:
        parser.error("no hosts: name them, or set ZOMBIES_FLEET_HOSTS")
    failed = 0
    for host in hosts:
        staging = args.into / ".remote" / host
        staging.mkdir(parents=True, exist_ok=True)
        source = f"{host}:{args.repo_dir.rstrip('/')}/{args.remote_dir.strip('/')}/"
        done = subprocess.run(["rsync", "-a", "--exclude", ".lock", "--exclude", "*.tmp", source, f"{staging}/"],
                              capture_output=True, text=True)
        if done.returncode != 0:
            failed += 1
            print(f"{host}: could not copy ({done.stderr.strip()[-300:] or done.returncode})")
            continue
        taken = merge(args.into, staging)
        print(f"{host}: " + ", ".join(f"{n} {name}" for name, n in taken.items()) + " taken")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
