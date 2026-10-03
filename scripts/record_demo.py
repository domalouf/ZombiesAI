"""Record a demonstration: frames the player saw, paired with the input they actually gave.

    uv run python scripts/record_demo.py --counts-per-degree 6.4 --minutes 20

    # Launched from a terminal on another workspace: start once the game is on screen, after 3 s.
    uv run python scripts/record_demo.py --window "Call of Duty" --wait --counts-per-degree 6.4
    # Also keeping the game's audio (the whole default output -- other desktop sounds too):
    uv run python scripts/record_demo.py --counts-per-degree 6.4 --minutes 20 --audio

The frames come from the game's XWayland window (demos/x11_capture.py) and the input from the kernel's event
devices (demos/evdev_input.py); docs/linux.md has the setup both need. Demos recorded this way are the only labels good enough to train the IDM, which is what then labels hours of
ordinary gameplay video. `--counts-per-degree` is spike S4's number for the sensitivity you play at; the raw
input log is stored beside the clip, so getting it wrong costs a re-quantization rather than a re-recording.

Tap F8 (`--mark-key`) going into a menu, the pause screen, a loading screen or the game-over card, and again
coming back. Those steps stay in the recording but are flagged and never trained on.

Play deliberately varied games -- camping, trains, bad positioning, early deaths. A clone of expert-only play
has no idea what to do the moment it drifts off-distribution. Taking the controls back while a policy plays
(scripts/play_real.py) records corrections for exactly those moments.
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import DEFAULT_BINDINGS, MARK_KEY, InputConfig
from zombiesai.demos.recorder import RecorderConfig, quality_report, record, wait_to_start


def pack(path: Path) -> None:
    """Swap the raw HUD crops for verified video, after the session so it never costs the recording a step."""
    from zombiesai.demos.hud_video import PackError, pack_hud

    try:
        pack_hud(path, say=lambda message: print(f"  {message}"))
    except (PackError, KeyboardInterrupt) as e:
        print(f"  HUD crops left raw ({e or 'stopped'}); pack them later with scripts/pack_hud.py {path}")


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
    rough = f", only {flow['n']} turning steps sampled so this is rough" if flow["n"] < 60 else ""
    if flow["verdict"] == "too_little_turning":
        print("  yaw vs image motion: not enough turning to check")
        return
    if flow["lag"] is not None:
        print(
            f"  yaw vs image motion: rank correlation {flow['rank_correlation']:.2f} at a lag of {flow['lag']} "
            f"decisions ({flow['lag'] * 1000 / 15:.0f} ms), expected {flow['expected_lag']}{rough}"
        )
    if not math.isnan(flow["fov_deg"]):  # NaN when too few 2-8 degree turns moved the image
        print(
            f"  scale: {flow['px_per_deg']:.2f} px per labelled degree = a {flow['fov_deg']:.0f} degree "
            "horizontal FOV on the 128-px frame (WaW is 81-96)"
        )
    if flow["verdict"] == "ok":
        for reason in flow["reasons"]:  # e.g. the scale could not be checked
            print(f"  note: {reason}")
        return
    what = {
        "no_signal": "yaw does not track the image at any lag -- wrong log for these frames, flipped sign, or a"
        " dead capture",
        "misaligned": "the input log and the capture are out of step in time",
        "wrong_scale": "counts-per-degree is wrong; fix it with requantize (docs/demos.md), no need to re-record",
    }[flow["verdict"]]
    print(f"  WARNING {what}.")
    for reason in flow["reasons"]:
        print(f"          {reason}")
    if flow["verdict"] != "wrong_scale":
        print("          Fix it before recording more: no amount of training absorbs a timing bug.")


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


def notify_now(message: str, seconds: float = 3.0) -> None:
    """`notify` without waiting for it, for use inside the recording loop: a notifier that takes a second to
    answer must cost nothing but the notification, never a decision."""
    if shutil.which("notify-send") is None:
        return
    try:
        subprocess.Popen(
            ["notify-send", "--app-name", "ZombiesAI", "--expire-time", str(int(seconds * 1000)),
             "record_demo", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        pass


def announce_mark(mark_key: str):
    """What the player sees when they tap the mark key: a missed second tap would silently mark the rest of
    the session as menus (it did, in the first real test), so each toggle says which state it left them in."""
    key = mark_key.upper()

    def on_mark(playing: bool) -> None:
        notify_now("Recording play" if playing else f"NOT PLAYING -- press {key} when you're back in the game",
                   seconds=2.0 if playing else 6.0)

    return on_mark


def next_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in root.glob(f"{prefix}_*")]
    return root / f"{prefix}_{len(existing):04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="stop after this many minutes of recording (--wait time not counted)")
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
                        help="capture this rectangle of the game window instead of all of it")
    parser.add_argument("--window", default="World at War",
                        help="the game window to capture, by title or 0x id (it runs under XWayland)")
    parser.add_argument("--no-hud", action="store_true",
                        help="don't save full-resolution HUD crops (~7.4 GB per 20 minutes at 1440p while recording, ~0.4 GB once packed as video)")
    parser.add_argument("--keep-raw-hud", action="store_true",
                        help="don't pack the HUD crops as video when the session ends (~17x smaller, verified)")
    parser.add_argument("--mark-key", default=MARK_KEY,
                        help="key that toggles 'not playing' (menus, pause, loading, game over); 'none' disables")
    parser.add_argument("--game-config", type=Path,
                        help="WaW config.cfg to stamp into the recording; default: the newest profile found")
    parser.add_argument("--audio", action="store_true",
                        help="also record the default output's monitor (never a microphone), aligned to "
                             "the frames; raw 11.5 MB/min while recording, lossless FLAC after a clean stop")
    parser.add_argument("--audio-device", help="a specific sink monitor to record, e.g. <sink name>.monitor "
                                               "(pactl list short sources); default: the default sink's")
    parser.add_argument("--audio-raw", action="store_true", help="keep audio as raw PCM, skip FLAC compression")
    parser.add_argument("--notes", default="", help="what you were trying to do -- camping, training, dying early")
    args = parser.parse_args()

    bindings = json.loads(args.bindings.read_text()) if args.bindings else dict(DEFAULT_BINDINGS)
    mark_key = None if args.mark_key.lower() == "none" else args.mark_key
    try:
        input_config = InputConfig(counts_per_degree=args.counts_per_degree, bindings=bindings, mark_key=mark_key)
    except ValueError as err:
        raise SystemExit(f"--mark-key: {err}; pick a key the game does not use") from None

    from zombiesai.demos.capture import CaptureLost, ScreenCapture
    from zombiesai.demos.evdev_input import EvdevInput
    from zombiesai.demos.x11_capture import WindowGone, WindowNotFound

    if args.counts_per_degree == 1.0:
        print("warning: --counts-per-degree is still 1.0, so every look label is scaled wrong.")
        print("         Run scripts/calibrate_mouse.py first (spike S4).")
    from zombiesai.demos.game_settings import read_settings, settings_warnings

    # Read before anyone plays: a wrong sensitivity or toggle-ADS is worth hearing about now, not after.
    game_settings = read_settings(args.game_config)
    if game_settings is not None:
        d = game_settings["dvars"]
        age_h = game_settings["config_age_s"] / 3600
        print(f"game settings: sensitivity {d.get('sensitivity')}, m_yaw {d.get('m_yaw')}, "
              f"{d.get('r_mode')}, fov {d.get('cg_fov', 'default')} (config.cfg written {age_h:.1f} h ago; "
              "the game only saves it on exit)")
    for warning in settings_warnings(game_settings, args.counts_per_degree):
        print(f"warning: {warning}")
    window = args.window
    if isinstance(window, str) and window.startswith("0x"):
        window = int(window, 16)
    from zombiesai.demos.hud_crops import HUD_REGIONS, HUD_SCALE

    def open_capture() -> ScreenCapture:
        return ScreenCapture(
            tuple(args.region) if args.region else None, window=window,
            hud_regions=None if args.no_hud else HUD_REGIONS, hud_scale=HUD_SCALE,
        )

    # Only a window that isn't there yet is worth waiting for. No X display, or the wrong depth, stays an
    # immediate error rather than a five-minute timeout.
    not_found: tuple[type[Exception], ...] = (WindowNotFound,) if args.wait else ()
    gone: tuple[type[Exception], ...] = (WindowGone, CaptureLost) if args.wait else ()
    try:
        capture = open_capture()
    except not_found:
        capture = None
    # Raw device counts from the event devices. Cursor deltas would be useless -- the game captures and
    # re-centres the pointer.
    inputs = EvdevInput()
    print(f"reading {', '.join(d.name for d in inputs.devices)}")
    if not inputs.monotonic:
        print("warning: this kernel would not switch the devices to CLOCK_MONOTONIC; timestamps are")
        print("         converted from wall clock, so a clock step mid-recording would shift labels.")
    audio = None
    if args.audio:
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
            try:
                return capture.is_capturable()
            except gone:
                # The window found was the game's start-up window, since destroyed (WaW opens one for its
                # intro, then the real one): let it go and look for the title again next time.
                capture.close()
                capture = None
                return False

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
    # Closing the terminal (SIGHUP) or a kill (SIGTERM) would end Python without running record()'s cleanup,
    # so the labels, the summary and the audio's compression would never be written. Treat both as Ctrl-C,
    # which record() already turns into a clean stop -- and only the first: `uv run` forwards what it gets
    # to this process and a closing terminal signals the whole session, so a second signal typically lands
    # mid-cleanup and would abort exactly the writes this is for. Everything after the first is ignored.
    import signal

    stops = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def stop_cleanly(signum, _frame):
        for sig in stops:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for sig in stops:
        signal.signal(sig, stop_cleanly)
    try:
        path = record(
            capture, inputs, out, config, audio=audio,
            annotations={"game_settings": game_settings} if game_settings is not None else None,
            on_mark=announce_mark(config.input.mark_key) if config.input.mark_key else None,
        )
    except KeyboardInterrupt:
        # record() has already closed the clip; say where it is and check it, as for a full-length session.
        path = out
        print("\nstopped early")
    print(f"wrote {path}")
    clip = load_clip(path)
    for name in clip.hud_regions:
        print(f"  HUD crops {name}: {clip.hud(name).shape[1:]} per step")
    if audio is not None:
        from zombiesai.demos.audio import describe_audio

        print(f"  {describe_audio(path)}")
    check(path)
    if clip.hud_regions and not args.keep_raw_hud:
        pack(path)


if __name__ == "__main__":
    main()
