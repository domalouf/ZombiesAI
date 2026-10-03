"""Watch every running World at War instance live, in a grid on one Hyprland workspace (realgame/viewer.py).

    uv run python scripts/view_instances.py              # a viewer per running instance on workspace 9, and go there
    uv run python scripts/view_instances.py 60 61        # only these displays
    uv run python scripts/view_instances.py --workspace 7 --fps 15
    uv run python scripts/view_instances.py --close      # close every viewer (Super+W closes one)
    uv run python scripts/view_instances.py 62 --no-audio --run runs/rl9

Over each picture: what the agent sees (its 128x72 view), the Twitch overlay's numbers for the run training now
(the newest `runs/*/episodes.jsonl`, or --run), and the HUD regions the reward reads with what it reads there.
Watching one game plays its sound too (--audio/--no-audio to choose; a grid is silent unless asked).

The viewers only read each instance's picture; closing them, or never opening them, changes nothing for the
games or the agents playing them. Each shows only whole frames, up to --fps a second or as fast as the game
delivers them, whichever is lower (a `-shm` instance manages ~11; realgame/viewer.py explains). Run it again after `instances.py up` to lay out the new ones
too. A viewer whose game restarts keeps showing it -- the X server outlives the game.
"""

import argparse
from pathlib import Path

from zombiesai.realgame.viewer import close, running_screens, show


def newest_run(root: Path = Path("runs")) -> Path | None:
    """The run whose games were written last: the one training now, if one is."""
    found = [p for p in root.glob("*/episodes.jsonl")]
    return max(found, key=lambda p: p.stat().st_mtime).parent if found else None


def fleet_sinks(root: str = "runs/instances") -> dict[str, str]:
    """Each instance's display and its audio sink, from the fleet that started them; none without one."""
    from zombiesai.realgame.instances import load_fleet, specs

    try:
        fleet = load_fleet(root)
    except FileNotFoundError:
        return {}
    return {s.display: s.sink for s in specs(fleet)} if fleet.audio_sinks else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("displays", nargs="*", type=int, help="display numbers to watch (default: every instance)")
    parser.add_argument("--workspace", default="9", help="the Hyprland workspace to put them on (default: 9)")
    parser.add_argument("--fps", type=float, default=30,
                        help="at most this many frames a second per viewer (default: 30; 0: every frame)")
    parser.add_argument("--agent-view", action=argparse.BooleanOptionalAction, default=True,
                        help="inset what the agent sees -- the policy's 128x72 view -- in each (default: on)")
    parser.add_argument("--run", type=Path, default=None,
                        help="the run whose numbers to show (default: the newest runs/*/episodes.jsonl)")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=None,
                        help="play the game's sound (default: only when watching one)")
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
    audio = args.audio if args.audio is not None else len(screens) == 1
    run_dir = args.run or newest_run()
    opened = show(screens, workspace=args.workspace, fps=args.fps, agent_view=args.agent_view,
                  run_dir=str(run_dir) if run_dir else None, sinks=fleet_sinks() if audio else None,
                  focus=not args.stay)
    print(f"watching {opened} of {', '.join(s.display for s in screens)} on workspace {args.workspace}")


if __name__ == "__main__":
    main()
