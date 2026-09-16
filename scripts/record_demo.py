"""Record a demonstration: frames the player saw, paired with the input they actually gave.

    # Windows, with the game running in borderless windowed:
    uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20

    # Anywhere, to generate labelled clips for the inverse dynamics model without touching the game:
    uv run python scripts/record_demo.py --source sim --episodes 20 --agent scripted

Demos recorded this way are the only labels good enough to train the IDM, which is what then labels hours of
ordinary gameplay video. `--counts-per-degree` is spike S4's number for the sensitivity you play at; the raw
input log is stored beside the clip, so getting it wrong costs a re-quantization rather than a re-recording.

Play deliberately varied games -- camping, trains, bad positioning, early deaths. A clone of expert-only play
has no idea what to do the moment it drifts off-distribution, and there is no DAgger loop here to save it.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import DEFAULT_BINDINGS, InputConfig
from zombiesai.demos.recorder import RecorderConfig, quality_report, record


def check(path: Path) -> None:
    """Say immediately whether the recording is worth keeping, while the game is still open."""
    report = quality_report(load_clip(path))
    flow, behaviour = report["yaw_flow"], report["behaviour"]
    print(
        f"  {report['steps']} steps ({report['seconds'] / 60:.1f} min), "
        f"overruns {report['overrun_rate']:.1%}, label confidence {report['mean_confidence']:.2f}"
    )
    print(
        f"  fire {behaviour['fire_duty']:.2f} duty, |yaw| {behaviour['abs_yaw_deg_per_s']:.0f} deg/s, "
        f"reloads {behaviour['reload_per_min']:.1f}/min, clamped looks {report['clamped_looks']:.1%}"
    )
    correlation = flow["correlation"]
    rough = f" (only {flow['n']} turning steps sampled, so this is rough)" if flow["n"] < 60 else ""
    if correlation != correlation:  # NaN: the player barely turned, so there is nothing to check against
        print("  yaw vs image motion: not enough turning to check")
    elif correlation < 0.9:
        print(f"  WARNING yaw vs image motion is only {correlation:.2f}{rough} -- the input log and the")
        print("          capture look misaligned in time, or counts-per-degree is wrong. Fix it before")
        print("          recording more. (On --source sim it runs lower: the raycast view is flat-shaded.)")
    else:
        print(
            f"  yaw vs image motion: {correlation:.2f}{rough} at a lag of {flow['lag']} decisions "
            f"({flow['lag'] * 1000 / 15:.0f} ms of closed-loop delay)"
        )


def next_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in root.glob(f"{prefix}_*")]
    return root / f"{prefix}_{len(existing):04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("screen", "sim"), default="screen")
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--minutes", type=float, default=20.0, help="stop after this long (screen source)")
    parser.add_argument("--counts-per-degree", type=float, default=1.0,
                        help="mouse counts per degree of yaw at your sensitivity, from spike S4")
    parser.add_argument("--bindings", type=Path, help="JSON map of key -> control, if yours aren't the defaults")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
                        help="capture this rectangle instead of the whole monitor")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--notes", default="", help="what you were trying to do -- camping, training, dying early")
    # Sim source only.
    parser.add_argument("--episodes", type=int, default=1, help="how many sim games to record")
    parser.add_argument("--agent", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--checkpoint", type=Path, help="record a trained policy instead of a baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardness", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=4_000)
    args = parser.parse_args()

    bindings = json.loads(args.bindings.read_text()) if args.bindings else dict(DEFAULT_BINDINGS)
    input_config = InputConfig(counts_per_degree=args.counts_per_degree, bindings=bindings)

    if args.source == "screen":
        from zombiesai.demos.capture import ScreenCapture
        from zombiesai.demos.win32_input import RawInputRecorder

        if args.counts_per_degree == 1.0:
            print("warning: --counts-per-degree is still 1.0, so every look label is scaled wrong. Run spike S4.")
        capture = ScreenCapture(tuple(args.region) if args.region else None, monitor=args.monitor)
        inputs = RawInputRecorder()
        inputs.start()
        out = next_dir(args.out, "demo")
        print(f"recording to {out} -- play; ctrl-c to stop early")
        config = RecorderConfig(
            max_seconds=args.minutes * 60, max_steps=int(args.minutes * 60 * 15) + 10,
            input=input_config, notes=args.notes,
        )
        try:
            path = record(capture, inputs, out, config)
        except KeyboardInterrupt:
            raise SystemExit("\nstopped") from None
        print(f"wrote {path}")
        check(path)
        return

    from zombiesai.demos.capture import SimSource

    if args.checkpoint:
        from zombiesai.rl.agent import load_agent

        make_agent = lambda: load_agent(args.checkpoint)  # noqa: E731
        label = "policy"
    else:
        from zombiesai.agents.random_agent import RandomAgent
        from zombiesai.agents.scripted import ScriptedAgent

        make_agent = lambda: ScriptedAgent() if args.agent == "scripted" else RandomAgent(args.seed)  # noqa: E731
        label = args.agent

    config = RecorderConfig(max_steps=args.max_steps, realtime=False, input=input_config, notes=args.notes)
    total = 0
    for episode in range(args.episodes):
        source = SimSource(
            make_agent(), seed=args.seed + episode, hardness=args.hardness, max_steps=args.max_steps,
            input_config=input_config,
        )
        out = next_dir(args.out, f"sim-{label}")
        path = record(source, source, out, config, stop=lambda s=source: s.done, progress_every=0)
        steps = source.steps
        total += steps
        print(f"{path.name}: {steps} steps, round {source.env.round}")
        if episode == 0:
            check(path)
    print(f"\n{total:,} labelled decisions under {args.out}")


if __name__ == "__main__":
    main()
