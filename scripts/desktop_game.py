"""Play the fleet's game on your own desktop: Plutonium T4 in LAN mode, offline (realgame/desktop.py).

    uv run python scripts/desktop_game.py up        # launch it (a no-op if it is already running)
    uv run python scripts/desktop_game.py status
    uv run python scripts/desktop_game.py down      # quit it and remove its sound sink

At the main menu, press F9 to start Nacht (and again after each game over); the console (~) takes
`map nazi_zombie_prototype` too. To record a demo on it, use scripts/record_lan.sh, which runs `up` itself.

It uses the fleet's own settings (runs/instances/fleet.json: game, Proton, Plutonium) and its config.cfg, so
what you record is the game the agent plays. Its copies of Plutonium and the prefix live in runs/desktop/.
"""

import argparse
import json
import os
import sys
import time

from zombiesai.realgame.desktop import (
    DESKTOP_ROOT, SINK, START_KEY, DesktopGame, desktop_config, desktop_spec, profile_config,
)
from zombiesai.realgame.instances import wait_for_window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("up", "status", "down"))
    parser.add_argument("--fleet-root", default="runs/instances", help="the fleet whose settings to use")
    parser.add_argument("--root", default=DESKTOP_ROOT)
    parser.add_argument("--display", default=os.environ.get("DISPLAY"),
                        help="the X display to play on (default: yours)")
    parser.add_argument("--width", type=int, help="default: the fleet's (2560; the HUD reader is built for 1440p)")
    parser.add_argument("--height", type=int)
    parser.add_argument("--start-key", default=START_KEY, help="the key bound to starting Nacht")
    parser.add_argument("--no-sound-sink", action="store_true",
                        help="play through the default output (a recording would then hear the whole desktop)")
    parser.add_argument("--wait", type=float, default=0.0,
                        help="up: wait this many seconds for the game's window, and focus it (for a display "
                             "with no window manager)")
    args = parser.parse_args()
    if not args.display:
        sys.exit("no X display: pass --display")

    sink = None if args.no_sound_sink else SINK
    config = desktop_config(args.fleet_root, root=args.root, width=args.width, height=args.height, sink=sink)
    game = DesktopGame(config, desktop_spec(config, args.display), fleet_root=args.fleet_root,
                       key=args.start_key, sink=sink)

    if args.command == "up":
        running = game.game_running()
        game.up()
        if running:
            print(f"already running (pid {game.game_pid()})")
        if args.wait and not running:
            start = time.monotonic()
            if wait_for_window(game, args.wait):
                print(f"  window up after {time.monotonic() - start:.0f} s")
        print(f"press {args.start_key} at the main menu to start Nacht, and again after each game over")
        if sink:
            print(f"its sound: {sink}.monitor")
        print(f"its config.cfg: {profile_config(config)}")
        return 0
    if args.command == "down":
        game.down()
        print("stopped")
        return 0
    status = {"display": args.display, "game_pid": game.game_pid(), "sink": sink,
              "config": str(profile_config(config))}
    try:
        window = game.window()
        status["window"] = {"title": window.title, "size": [window.width, window.height]}
    except Exception as exc:  # noqa: BLE001 -- no window is a status, not an error
        status["window"] = None
        status["window_error"] = str(exc).splitlines()[0][:200]
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
