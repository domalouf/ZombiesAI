"""Supervise training live: every run's curves, what the running trainers print, and the game instances, with
the switch that shows or hides their viewer windows on a Hyprland workspace (viz/supervise.py).

    uv run python scripts/supervise.py                 # http://127.0.0.1:8765, opened in your browser
    uv run python scripts/supervise.py --port 8800 --no-open

It only reads: runs/ in the main checkout and every worktree, the trainers' processes and output, and the
games' X servers. The one thing it changes is the viewer windows, when you press the switch.
"""

import argparse
import os
import webbrowser
from pathlib import Path

from zombiesai.viz.supervise import serve

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    parser.add_argument("--idle", type=float, default=20.0,
                        help="stop the game monitors this many seconds after the page stops looking (default: 20)")
    args = parser.parse_args()
    state = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "zombiesai-supervise"

    def ready(url: str) -> None:
        print(f"supervising at {url}  (Ctrl-C to stop)", flush=True)
        if not args.no_open:
            webbrowser.open(url)

    try:
        serve(REPO, args.port, state, idle_s=args.idle, ready=ready)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
