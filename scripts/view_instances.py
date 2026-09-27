"""Watch every running World at War instance live, in a grid on one Hyprland workspace (realgame/viewer.py).

    uv run python scripts/view_instances.py              # a viewer per running instance on workspace 9, and go there
    uv run python scripts/view_instances.py 60 61        # only these displays
    uv run python scripts/view_instances.py --workspace 7 --fps 30
    uv run python scripts/view_instances.py --close      # close every viewer (Super+W closes one)

The viewers only read each instance's picture; closing them, or never opening them, changes nothing for the
games or the agents playing them. Run it again after `instances.py up` to lay out the new ones too. A viewer
whose game restarts keeps showing it -- the X server outlives the game.
"""

import argparse

from zombiesai.realgame.viewer import close, running_screens, show


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("displays", nargs="*", type=int, help="display numbers to watch (default: every instance)")
    parser.add_argument("--workspace", default="9", help="the Hyprland workspace to put them on (default: 9)")
    parser.add_argument("--fps", type=int, default=15, help="frames per second each viewer grabs (default: 15)")
    parser.add_argument("--stay", action="store_true", help="open them without switching to the workspace")
    parser.add_argument("--close", action="store_true", help="close every viewer and exit")
    args = parser.parse_args()

    if args.close:
        close()
        return
    screens = running_screens()
    if args.displays:
        screens = [s for s in screens if s.number in args.displays]
    if not screens:
        print("no running instances found (`scripts/instances.py status`)")
        return
    opened = show(screens, workspace=args.workspace, fps=args.fps, focus=not args.stay)
    print(f"watching {opened} of {', '.join(s.display for s in screens)} on workspace {args.workspace}")


if __name__ == "__main__":
    main()
