"""Let a trained policy play World at War: the screen in, the model, a virtual mouse and keyboard out.

    # First, with no input sent at all: everything runs, the actions are only printed and recorded.
    uv run python scripts/play_real.py runs/bc_real1/bc.pt --dry-run --minutes 1

    # For real (Linux, Hyprland): start it, go to the game, press F7 when you want the AI to take over.
    uv run python scripts/play_real.py runs/bc_real1/bc.pt --minutes 3

It starts in standby. In the game, F7 hands the AI the controls and F7 again takes them back; F9 quits. A
chime says which (the fullscreen game hides notifications). It only ever sends input while the game is the
focused window and its picture is live, and touching your own mouse or keyboard takes the controls back
until you have been idle for 1.5 s. Every run is recorded to runs/play/ (frames, HUD crops, the actions
chosen) -- watch it back with the same tools as a demo.

Taking over is also how you teach it. Whenever you grab the controls, what you do is decoded exactly like a
demo (same --bindings, same --counts-per-degree) and kept as a correction: your fix, in the very state the
policy got itself into. The summary says how much was captured; train on it alongside the demos with

    uv run python scripts/train_bc.py data/demos runs/play --out runs/bc2
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.inputs import DEFAULT_BINDINGS, InputConfig
from zombiesai.realgame.play import KILL_KEY, TOGGLE_KEY, PlayConfig

SOUNDS = Path("/usr/share/sounds/freedesktop/stereo")
# What the player hears for each state change: they are looking at a fullscreen game, which hides
# notifications, and the terminal is on another workspace.
CUES = {"acting": "device-added.oga", "standby": "device-removed.oga", "stopped": "service-logout.oga"}
PAUSE_CUE = "dialog-warning.oga"  # focus lost, picture frozen, or the human took over


def cue(state: str) -> None:
    sound = SOUNDS / CUES.get(state, PAUSE_CUE)
    if not sound.exists():
        return
    import subprocess

    try:  # never waits: this runs inside the decision loop
        subprocess.Popen(["paplay", str(sound)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def next_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{prefix}_{len(list(root.glob(f'{prefix}_*'))):04d}"


def describe_actions(actions: np.ndarray) -> str:
    if not len(actions):
        return "no actions"
    heads = {"forward": spec.FORWARD, "strafe": spec.STRAFE, "fire": spec.FIRE, "ads": spec.ADS,
             "sprint": spec.SPRINT}
    parts = []
    for name, head in heads.items():
        neutral = spec.NEUTRAL_ACTION[head]
        parts.append(f"{name} {np.mean(actions[:, head] != neutral):.0%}")
    turning = np.mean(actions[:, spec.YAW] != spec.NEUTRAL_ACTION[spec.YAW])
    buttons = Counter(spec.BUTTONS[b] for b in actions[:, spec.BUTTON] if spec.BUTTONS[b] != "none")
    return ", ".join(parts) + f", turning {turning:.0%}, buttons {dict(buttons) or 'none'}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--window", default="Call of Duty")
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--counts-per-degree", type=float, default=9.09)
    parser.add_argument("--bindings", type=Path, default=Path("configs/waw_bindings.json"))
    parser.add_argument("--dry-run", action="store_true", help="run everything but send no input")
    parser.add_argument("--deterministic", action="store_true", help="take each head's most likely action")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--look", choices=("mean", "sample"), default="mean",
                        help="turn by the smoothed mean look (default) or a fresh sample every tick")
    parser.add_argument("--smoothness", type=float, default=0.08,
                        help="mouse motor time constant in seconds: higher is smoother and laggier "
                             "(lag is about twice this)")
    parser.add_argument("--private-device", action="store_true",
                        help="create and destroy a virtual device for this run instead of borrowing the "
                             "session's -- destroying one while the game is focused can unlock its pointer")
    parser.add_argument("--kill-key", default=KILL_KEY)
    parser.add_argument("--toggle-key", default=TOGGLE_KEY)
    parser.add_argument("--quiet", action="store_true", help="no sound cues")
    parser.add_argument("--deaf", action="store_true",
                        help="a policy trained with audio plays without it (as on the clips recorded silent)")
    parser.add_argument("--audio-device", help="sink monitor the policy listens to (default: the default "
                        "sink's .monitor; only a .monitor is ever opened)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/play"))
    args = parser.parse_args()

    import json
    from dataclasses import asdict

    import torch

    from zombiesai.demos.agent import BCAgent
    from zombiesai.demos.capture import FollowWindow, ScreenCapture
    from zombiesai.demos.clips import ClipWriter
    from zombiesai.demos.evdev_input import EvdevInput
    from zombiesai.demos.hud_crops import HUD_REGIONS, HUD_SCALE
    from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig, FakeSink
    from zombiesai.realgame.play import HumanWatch, HyprlandFocus, play

    if not sys.platform.startswith("linux"):
        raise SystemExit("play_real.py is Linux-only for now (uinput, XWayland capture, Hyprland focus)")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    agent = BCAgent(args.checkpoint, deterministic=args.deterministic, device=device,
                    temperature=args.temperature)
    bindings = json.loads(args.bindings.read_text()) if args.bindings.exists() else dict(DEFAULT_BINDINGS)
    dispatch_config = DispatchConfig(counts_per_degree=args.counts_per_degree, bindings=bindings)
    config = PlayConfig(max_seconds=args.minutes * 60, kill_key=args.kill_key.lower(),
                        toggle_key=args.toggle_key.lower(), look=args.look)
    focus = HyprlandFocus()
    # Your corrections are labelled exactly as a demo would be: the same bindings and counts per degree.
    input_config = InputConfig(counts_per_degree=args.counts_per_degree, bindings=bindings)
    human = HumanWatch(EvdevInput(), config, input_config)  # excludes our own virtual device by name

    print(f"policy {args.checkpoint} on {device}; {'DRY RUN: no input will be sent' if args.dry_run else 'LIVE'}")
    # Follows the window by title: WaW's intro window dies and the real one replaces it, and a player that held
    # the first one crashed the moment it was handed the controls.
    capture = FollowWindow(lambda: ScreenCapture(window=args.window, hud_regions=HUD_REGIONS, hud_scale=HUD_SCALE))
    hearing = None
    if agent.audio_features is not None and not args.deaf:
        from zombiesai.demos.audio import PulseMonitorStream
        from zombiesai.demos.hearing import LiveAudio

        # The game's sound as the policy learned it: the output sink's monitor, never a microphone.
        hearing = LiveAudio(PulseMonitorStream(args.audio_device, stream_name="policy hearing"),
                            agent.audio_features).start()
        print(f"  listening to {hearing.stream.device} ({agent.audio_features.window_s:.2f} s log-mel)")

    if args.dry_run:
        sink = FakeSink()
    elif args.private_device:
        from zombiesai.realgame.uinput import UinputDevice

        sink = UinputDevice(dispatch_config.codes.values())
    else:
        from zombiesai.realgame.input_service import RemoteSink, ensure_running

        # One virtual device for the whole session, started on first use and left running: unplugging a
        # mouse under a focused fullscreen game is what made the human's clicks jump after quitting.
        sink = RemoteSink(ensure_running())
    # The motor sends mouse motion every 4 ms at a smoothly varying rate -- a hand, not a burst per decision.
    dispatcher = ActionDispatcher(sink, dispatch_config, motor=args.look == "mean",
                                  motor_time_constant_s=args.smoothness)
    out = next_dir(args.out, "dry" if args.dry_run else "play")
    writer = ClipWriter(
        out,
        source={**capture.describe(), "policy": str(args.checkpoint), "dry_run": args.dry_run,
                "temperature": args.temperature, "deterministic": args.deterministic},
        label_source="play",
        config={"play": asdict(config), "input": asdict(input_config), "decision_hz": spec.DECISION_HZ,
                "counts_per_degree": args.counts_per_degree},
    )
    import signal

    stops = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]

    def stop_cleanly(signum, _frame):
        for sig in stops:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for sig in stops:
        signal.signal(sig, stop_cleanly)
    human.source.drain()  # keys pressed while the model loaded are not commands
    try:
        summary = play(capture, agent, dispatcher, focus=focus, human=human, config=config, writer=writer,
                       on_state=None if args.quiet else cue, hearing=hearing)
    finally:
        if hearing is not None:
            hearing.close()
        dispatcher.close()  # releases everything again; hangs up on the service (the device stays)
        capture.close()
        human.source.close()

    from zombiesai.demos.clips import ACTOR_POLICY, FLAG_BAD_STEP, load_clip

    clip = load_clip(out)
    if not clip.n_steps:
        print(f"never handed the controls ({args.toggle_key.upper()}), so nothing was played or written")
        return
    actor = clip.extra("actor")
    live = (clip.flags & FLAG_BAD_STEP) == 0
    print(f"wrote {out}: played {summary['played_seconds']:.0f}s ({summary['acted']} steps), "
          f"paused unfocused {summary['unfocused']} / frozen {summary['frozen']} / you {summary['human']}, "
          f"overruns {summary['overruns']}")
    print(f"  what it did: {describe_actions(clip.actions[live & (actor == ACTOR_POLICY)])}")
    corrections = clip.usable()
    if corrections.any() or summary["human_idle"]:
        print(f"  your corrections: {int(corrections.sum())} steps ({corrections.sum() / spec.DECISION_HZ:.1f}s) "
              f"kept for training, {summary['human_idle']} idle steps before handing back left out; "
              f"{describe_actions(clip.actions[corrections])}")
    else:
        print("  no corrections: you never took the controls over while it played")
    if "hearing" in summary:
        h = summary["hearing"]
        print(f"  hearing: {h['observed'] - h['no_audio']}/{h['observed']} ticks heard, window ends "
              f"{h['lag_ms_median']:.0f} ms (p95 {h['lag_ms_p95']:.0f}) before the frame, "
              f"{h['cost_ms_median']:.1f} ms a tick{', ' + h['error'] if h['error'] else ''}")
    if clip.hud_regions:
        from zombiesai.demos.hud_video import PackError, pack_hud

        try:  # the raw crops are ~6 MB a second of play; as verified video, ~17x less
            pack_hud(out, say=lambda message: print(f"  {message}"))
        except (PackError, KeyboardInterrupt) as e:
            print(f"  HUD crops left raw ({e or 'stopped'}); pack them later with scripts/pack_hud.py {out}")


if __name__ == "__main__":
    main()
