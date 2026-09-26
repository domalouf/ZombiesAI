"""Run several World at War instances side by side, each in an X server of its own (realgame/instances.py).

    uv run python scripts/instances.py up --n 4          # prefixes (copied once), X servers, sinks, games
    uv run python scripts/instances.py status            # what is up, and does each game have a window
    uv run python scripts/instances.py show              # toggle the hidden workspace to watch them
    uv run python scripts/instances.py restart 2         # relaunch one game
    uv run python scripts/instances.py down              # stop the games, the X servers and the sinks

`up` remembers the fleet in runs/instances/fleet.json; everything else, and the RL actors, read it from there.
The games start straight into Nacht (`+map nazi_zombie_prototype`) at the fleet's resolution, filling their own
X server, with the console enabled for resets. Your own config.cfg is copied, not edited.

Before a long run, check the instances really take input: scripts/spike_instances.py.
"""

import argparse
import json
import time
from dataclasses import replace

from zombiesai.realgame.instances import (
    SPECIAL_WORKSPACE,
    FleetConfig,
    fleet,
    load_fleet,
    save_fleet,
    toggle_shown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("up", "status", "show", "restart", "down"))
    parser.add_argument("which", nargs="*", type=int, help="instance indices (default: all)")
    parser.add_argument("--root", default="runs/instances")
    parser.add_argument("--n", type=int, default=None, help="how many instances (up)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--display-base", type=int, default=None, help="instance i gets DISPLAY :base+i")
    parser.add_argument("--visible", action="store_true", help="ordinary windows, not a hidden workspace")
    parser.add_argument("--game-dir")
    parser.add_argument("--proton", help="a Proton directory (default: Steam's Proton - Experimental)")
    parser.add_argument("--template-prefix", help="a Steam compatdata dir to copy (default: WaW's own)")
    parser.add_argument("--no-audio-sinks", action="store_true")
    parser.add_argument("--stagger", type=float, default=8.0,
                        help="seconds between game launches: Proton start-up is heavy, and Steam is asked once each")
    args = parser.parse_args()

    if args.command == "up":
        try:
            config = load_fleet(args.root)
        except FileNotFoundError:
            config = FleetConfig(root=args.root)
        overrides = {k: v for k, v in {
            "n": args.n, "width": args.width, "height": args.height, "display_base": args.display_base,
            "game_dir": args.game_dir, "proton": args.proton, "template_prefix": args.template_prefix,
        }.items() if v is not None}
        if args.visible:
            overrides["hidden"] = False
        if args.no_audio_sinks:
            overrides["audio_sinks"] = False
        config = replace(config, **overrides)
        save_fleet(config)
    else:
        config = load_fleet(args.root)
    instances = fleet(config)
    chosen = [instances[i] for i in args.which] if args.which else instances

    if args.command == "up":
        print(f"{len(chosen)} instances at {config.width}x{config.height}, displays "
              f":{config.display_base}-:{config.display_base + config.n - 1}")
        for k, instance in enumerate(chosen):
            if k:
                time.sleep(args.stagger if not instance.game_running() else 0)
            instance.up()
        print(f"up. `scripts/instances.py status` to check; `scripts/instances.py show` to watch "
              f"(the special workspace {SPECIAL_WORKSPACE!r})")
    elif args.command == "status":
        for instance in chosen:
            print(json.dumps(instance.status()))
    elif args.command == "show":
        if not toggle_shown():
            print(f"Hyprland did not toggle the special workspace {SPECIAL_WORKSPACE!r}")
    elif args.command == "restart":
        for instance in chosen:
            instance.restart_game()
    elif args.command == "down":
        for instance in chosen:
            instance.down()
            print(f"  instance {instance.spec.index}: down")


if __name__ == "__main__":
    main()
