"""Supervise training live: every run's curves, what the running trainers print, and the game instances, with
the switch that shows or hides their viewer windows on a Hyprland workspace (viz/supervise.py).

    uv run python scripts/supervise.py                 # http://127.0.0.1:8765, opened in your browser
    uv run python scripts/supervise.py --port 8800 --no-open

Run it again while it serves and it only opens the page, so one key can do both (a Hyprland binding runs
~/.local/bin/zombiesai-dashboard, which runs this).

It reads runs/ in the main checkout and every worktree, the trainers' processes and output, and the games' X
servers. It changes things only when you press a button: the viewer windows; a new run (scripts/train_rl.py in
the checkout you pick, as a process of its own that outlives the dashboard); and stopping one (SIGINT, as
Ctrl-C: its actors stop and it writes its checkpoint).
"""

import argparse
import errno
import json
import os
import time
import urllib.request
import webbrowser
from pathlib import Path

from zombiesai.viz.supervise import serve

REPO = Path(__file__).resolve().parents[1]


def already_serving(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2) as response:
            return json.load(response).get("app") == "zombiesai-supervise"
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    parser.add_argument("--idle", type=float, default=20.0,
                        help="stop the game monitors this many seconds after the page stops looking (default: 20)")
    args = parser.parse_args()
    state = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "zombiesai-supervise"
    if already_serving(args.port):
        url = f"http://127.0.0.1:{args.port}/"
        print(f"already supervising at {url}", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        return

    def ready(url: str) -> None:
        print(f"supervising at {url}  (Ctrl-C to stop)", flush=True)
        if not args.no_open:
            webbrowser.open(url)

    try:
        serve(REPO, args.port, state, idle_s=args.idle, ready=ready)
    except KeyboardInterrupt:
        print("\nstopped")
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        # Two launches at once (a key pressed twice): the other one won the port. Show its page, if it is ours.
        time.sleep(1.0)
        if not already_serving(args.port):
            raise SystemExit(f"port {args.port} is taken by something else; try --port")
        print(f"already supervising at http://127.0.0.1:{args.port}/", flush=True)
        if not args.no_open:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")


if __name__ == "__main__":
    main()
