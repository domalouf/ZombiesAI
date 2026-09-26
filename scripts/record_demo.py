"""Record a demonstration: frames the player saw, paired with the input they actually gave.

    # Windows, with the game running in borderless windowed:
    uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20

    # Linux, launched from a terminal on another workspace: start once the game is on screen, after 3 s.
    uv run python scripts/record_demo.py --source screen --window "Call of Duty" --wait --counts-per-degree 6.4
    # Linux, also keeping the game's audio (the whole default output -- other desktop sounds too):
    uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20 --audio

    # Anywhere, to generate labelled clips for the inverse dynamics model without touching the game:
    uv run python scripts/record_demo.py --source sim --episodes 20 --agent scripted

Demos recorded this way are the only labels good enough to train the IDM, which is what then labels hours of
ordinary gameplay video. `--counts-per-degree` is spike S4's number for the sensitivity you play at; the raw
input log is stored beside the clip, so getting it wrong costs a re-quantization rather than a re-recording.

Tap F8 (`--mark-key`) going into a menu, the pause screen, a loading screen or the game-over card, and again
coming back. Those steps stay in the recording but are flagged and never trained on.

Play deliberately varied games -- camping, trains, bad positioning, early deaths. A clone of expert-only play
has no idea what to do the moment it drifts off-distribution, and there is no DAgger loop here to save it.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import DEFAULT_BINDINGS, MARK_KEY, InputConfig
from zombiesai.demos.recorder import RecorderConfig, quality_report, record, wait_to_start


def check(path: Path) -> None:
    """Say immediately whether the recording is worth keeping, while the game is still open."""
    report = quality_report(load_clip(path))
    flow, behaviour = report["yaw_flow"], report["behaviour"]
    print(
        f"  {report['steps']} steps ({report['seconds'] / 60:.1f} min), "
        f"overruns {report['overrun_rate']:.1%}, label confidence {report['mean_confidence']:.2f}"
    )
    if report["stale_steps"]:  # None on recordings from before capture could survive losing the window
        print(
            f"  capture lost for {report['stale_steps']} steps over {report['capture_outages']} outage(s), "
            f"all flagged bad"
        )
    if report["capture_lost"]:
        print(f"  WARNING the recording stopped early because capture was lost: {report['capture_lost']}")
    if report["not_playing_steps"]:
        share = report["not_playing_steps"] / max(report["steps"], 1)
        minutes = report["not_playing_seconds"] / 60
        print(f"  not playing (marked): {minutes:.1f} min ({share:.0%}), left out of training")
    else:
        print("  not playing (marked): none -- every step counts as play")
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


def notify(message: str, seconds: float = 3.0) -> None:
    """A desktop notification, when there is a notifier -- by the time the countdown runs, the player is
    looking at the game and not at this terminal. Best effort: no notify-send, or a notifier that hangs, is
    not a reason to lose the recording."""
    if shutil.which("notify-send") is None:
        return
    try:
        subprocess.run(
            ["notify-send", "--app-name", "ZombiesAI", "--expire-time", str(int(seconds * 1000)),
             "record_demo", message],
            timeout=2, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def next_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in root.glob(f"{prefix}_*")]
    return root / f"{prefix}_{len(existing):04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("screen", "sim"), default="screen")
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="stop after this many minutes of recording (screen source; --wait time not counted)")
    parser.add_argument("--wait", action="store_true",
                        help="don't start until the game window can be captured (it is on the visible "
                             "workspace), then count down -- for launching from a terminal elsewhere")
    parser.add_argument("--countdown", type=float, default=3.0, help="with --wait: seconds between the game "
                        "appearing and recording starting")
    parser.add_argument("--wait-timeout", type=float, default=300.0,
                        help="with --wait: give up after this many seconds without a capturable window")
    parser.add_argument("--counts-per-degree", type=float, default=1.0,
                        help="mouse counts per degree of yaw at your sensitivity, from spike S4")
    parser.add_argument("--bindings", type=Path, help="JSON map of key -> control, if yours aren't the defaults")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
                        help="capture this rectangle instead of the whole window or monitor")
    parser.add_argument("--window", default="World at War",
                        help="Linux: the game window to capture, by title or 0x id (it runs under XWayland)")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--no-hud", action="store_true",
                        help="don't save full-resolution HUD crops (they cost ~2.6 GB per 20 minutes at 1440p)")
    parser.add_argument("--mark-key", default=MARK_KEY,
                        help="key that toggles 'not playing' (menus, pause, loading, game over); 'none' disables")
    parser.add_argument("--audio", action="store_true",
                        help="Linux: also record the default output's monitor (never a microphone), aligned to "
                             "the frames; raw 11.5 MB/min while recording, lossless FLAC after a clean stop")
    parser.add_argument("--audio-device", help="a specific sink monitor to record, e.g. <sink name>.monitor "
                                               "(pactl list short sources); default: the default sink's")
    parser.add_argument("--audio-raw", action="store_true", help="keep audio as raw PCM, skip FLAC compression")
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
    mark_key = None if args.mark_key.lower() == "none" else args.mark_key
    try:
        input_config = InputConfig(counts_per_degree=args.counts_per_degree, bindings=bindings, mark_key=mark_key)
    except ValueError as err:
        raise SystemExit(f"--mark-key: {err}; pick a key the game does not use") from None

    if args.source == "screen":
        import sys

        from zombiesai.demos.capture import ScreenCapture

        if args.counts_per_degree == 1.0:
            print("warning: --counts-per-degree is still 1.0, so every look label is scaled wrong.")
            print("         Run scripts/calibrate_mouse.py first (spike S4).")
        window = args.window
        if isinstance(window, str) and window.startswith("0x"):
            window = int(window, 16)
        from zombiesai.demos.hud_crops import HUD_REGIONS, HUD_SCALE

        def open_capture() -> ScreenCapture:
            return ScreenCapture(
                tuple(args.region) if args.region else None, monitor=args.monitor,
                window=window if sys.platform.startswith("linux") else None,
                hud_regions=None if args.no_hud else HUD_REGIONS, hud_scale=HUD_SCALE,
            )

        not_found: tuple[type[Exception], ...] = ()
        if args.wait and sys.platform.startswith("linux"):
            from zombiesai.demos.x11_capture import WindowNotFound

            # Only a window that isn't there yet is worth waiting for. No X display, or the wrong depth, stays
            # an immediate error rather than a five-minute timeout.
            not_found = (WindowNotFound,)
        try:
            capture = open_capture()
        except not_found:
            capture = None
        # Raw device counts either way: Raw Input on Windows, the event devices on Linux. Cursor deltas
        # would be useless in both -- the game captures and re-centres the pointer.
        if sys.platform == "win32":
            from zombiesai.demos.win32_input import RawInputRecorder

            inputs = RawInputRecorder()
            inputs.start()
        else:
            from zombiesai.demos.evdev_input import EvdevInput

            inputs = EvdevInput()
            print(f"reading {', '.join(d.name for d in inputs.devices)}")
            if not inputs.monotonic:
                print("warning: this kernel would not switch the devices to CLOCK_MONOTONIC; timestamps are")
                print("         converted from wall clock, so a clock step mid-recording would shift labels.")
        audio = None
        if args.audio:
            if not sys.platform.startswith("linux"):
                # WASAPI loopback would slot in as another stream behind AudioRecorder; see demos/audio.py.
                raise SystemExit("--audio is Linux-only for now (PipeWire/PulseAudio monitor via parec)")
            from zombiesai.demos.audio import AudioRecorder, PulseMonitorStream

            stream = PulseMonitorStream(args.audio_device)
            if stream.device is None:
                stream.device = stream.default_monitor()
            stream.command()  # refuses anything that is not a .monitor before a second is recorded
            audio = AudioRecorder(stream, compress=not args.audio_raw)
            print(f"recording audio from {stream.device} (everything that plays through it, not only the game)")

        if args.wait:
            def capturable() -> bool:
                nonlocal capture
                if capture is None:
                    try:
                        capture = open_capture()
                    except not_found:
                        return False
                return capture.is_capturable()

            try:
                ready = wait_to_start(
                    capturable, timeout=args.wait_timeout, countdown=args.countdown,
                    discard=inputs.drain, notify=lambda message: notify(message, args.countdown),
                )
            except KeyboardInterrupt:
                ready = None
            if not ready:
                # Nothing was recorded, so there is nothing to write -- just let go of the devices and say why.
                inputs.close()
                if capture is not None:
                    capture.close()
                if ready is None:
                    raise SystemExit("\nstopped before recording started")
                raise SystemExit(
                    f"gave up after {args.wait_timeout:.0f}s: no capturable window titled {args.window!r}. "
                    "It has to be on the visible workspace and wholly on screen."
                )
        out = next_dir(args.out, "demo")
        print(f"recording to {out} -- play; ctrl-c to stop early")
        if mark_key:
            print(f"tap {mark_key} going into a menu, pause, loading or game-over screen, and again coming back")
        config = RecorderConfig(
            max_seconds=args.minutes * 60, max_steps=int(args.minutes * 60 * 15) + 10,
            input=input_config, notes=args.notes,
        )
        try:
            path = record(capture, inputs, out, config, audio=audio)
        except KeyboardInterrupt:
            raise SystemExit("\nstopped") from None
        print(f"wrote {path}")
        clip = load_clip(path)
        for name in clip.hud_regions:
            print(f"  HUD crops {name}: {clip.hud(name).shape[1:]} per step")
        if audio is not None:
            from zombiesai.demos.audio import describe_audio

            print(f"  {describe_audio(path)}")
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
