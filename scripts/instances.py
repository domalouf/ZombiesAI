"""Run several World at War instances side by side, each in an X server of its own (realgame/instances.py).

    uv run python scripts/instances.py up --n 4          # prefixes (copied once), X servers, sinks, games
    uv run python scripts/instances.py status            # what is up, and does each game have a window
    uv run python scripts/instances.py show              # toggle the hidden workspace to watch them
    uv run python scripts/instances.py restart 2         # relaunch one game
    uv run python scripts/instances.py down              # stop the games, the X servers and the sinks

`up` remembers the fleet in runs/instances/fleet.json; everything else, and the RL actors, read it from there.
The client is Plutonium's T4 in LAN mode unless `--client steam` (see realgame/instances.py for why). The games
start straight into Nacht (`+map nazi_zombie_prototype`) at the fleet's resolution, filling their own
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


def wait_for_window(instance, timeout_s: float) -> bool:
    """Wait for the game's window and give it its X server's focus. There is no window manager to do it, and
    the game will not get past loading its renderer until its window is the focused one."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            window = instance.window()
            sink = instance.sink()
            sink.focus(window.id)
            sink.close()
            return True
        except Exception:  # noqa: BLE001 -- not there yet
            time.sleep(1.0)
    print(f"  instance {instance.spec.index}: no game window after {timeout_s:.0f} s; carrying on")
    return False


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
    parser.add_argument("--host", choices=("auto", "weston", "hyprland"), default=None,
                        help="who hosts the X servers: a headless weston each (default when installed), or your "
                             "Hyprland session on a hidden workspace")
    parser.add_argument("--client", choices=("plutonium", "steam"), default=None,
                        help="plutonium (default: T4 in LAN mode) or steam (CoDWaW.exe, needs the Steam client)")
    parser.add_argument("--plutonium-dir", help="where plutonium-updater installed it (default ~/.local/share/plutonium)")
    parser.add_argument("--game-dir")
    parser.add_argument("--proton", help="a Proton directory (default: Steam's Proton - Experimental)")
    parser.add_argument("--template-prefix", help="a Steam compatdata dir to copy (default: WaW's own)")
    parser.add_argument("--no-audio-sinks", action="store_true")
    parser.add_argument("--stagger", type=float, default=5.0,
                        help="seconds to wait after a game's window appears before launching the next")
    args = parser.parse_args()

    if args.command == "up":
        try:
            config = load_fleet(args.root)
        except FileNotFoundError:
            config = FleetConfig(root=args.root)
        overrides = {k: v for k, v in {
            "n": args.n, "width": args.width, "height": args.height, "display_base": args.display_base,
            "host": args.host, "client": args.client, "plutonium_dir": args.plutonium_dir,
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
        for instance in chosen:
            was_running = instance.game_running()
            instance.up()
            if not was_running:
                wait_for_window(instance, 120.0)
                if instance is not chosen[-1]:
                    time.sleep(args.stagger)
        if config.resolved_host() == "weston":
            print("up, each in a headless weston: nothing shows on your desktop. `scripts/instances.py status` "
                  "to check; `DISPLAY=:60 import -window root shot.png` to look at one")
        else:
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
            wait_for_window(instance, 120.0)
            if instance is not chosen[-1]:
                time.sleep(args.stagger)
    elif args.command == "down":
        for instance in chosen:
            instance.down()
            print(f"  instance {instance.spec.index}: down")


if __name__ == "__main__":
    main()
