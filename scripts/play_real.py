"""Let a trained policy play World at War: the screen in, the model, a virtual mouse and keyboard out.

    # First, with no input sent at all: everything runs, the actions are only printed and recorded.
    uv run python scripts/play_real.py runs/bc_real1/bc.pt --dry-run --minutes 1

    # For real (Linux, Hyprland): start from any workspace, then switch to the game.
    uv run python scripts/play_real.py runs/bc_real1/bc.pt --minutes 3

It only ever sends input while the game is the focused window and its picture is live; touching your own
mouse or keyboard takes the controls back until you have been idle for 1.5 s; F9 stops it. Every run is
recorded to runs/play/ (frames, HUD crops, the actions chosen) -- watch it back with the same tools as a demo.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.demos.inputs import DEFAULT_BINDINGS
from zombiesai.realgame.play import KILL_KEY, PlayConfig


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
    parser.add_argument("--kill-key", default=KILL_KEY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/play"))
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    args = parser.parse_args()

    import json
    import subprocess
    from dataclasses import asdict

    import torch

    from zombiesai.demos.agent import BCAgent
    from zombiesai.demos.capture import CaptureLost, ScreenCapture
    from zombiesai.demos.clips import ClipWriter
    from zombiesai.demos.evdev_input import EvdevInput
    from zombiesai.demos.hud_crops import HUD_REGIONS, HUD_SCALE
    from zombiesai.demos.recorder import wait_to_start
    from zombiesai.demos.x11_capture import WindowGone, WindowNotFound
    from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig, FakeSink
    from zombiesai.realgame.play import HumanWatch, HyprlandFocus, play

    if not sys.platform.startswith("linux"):
        raise SystemExit("play_real.py is Linux-only for now (uinput, XWayland capture, Hyprland focus)")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    agent = BCAgent(args.checkpoint, deterministic=args.deterministic, device=device,
                    temperature=args.temperature)
    bindings = json.loads(args.bindings.read_text()) if args.bindings.exists() else dict(DEFAULT_BINDINGS)
    dispatch_config = DispatchConfig(counts_per_degree=args.counts_per_degree, bindings=bindings)
    config = PlayConfig(max_seconds=args.minutes * 60, kill_key=args.kill_key.lower())
    focus = HyprlandFocus()
    human = HumanWatch(EvdevInput(), config)  # excludes our own virtual device by name

    def open_capture():
        return ScreenCapture(window=args.window, hud_regions=HUD_REGIONS, hud_scale=HUD_SCALE)

    capture = None

    def ready() -> bool:
        nonlocal capture
        if capture is None:
            try:
                capture = open_capture()
            except WindowNotFound:
                return False
        try:
            return capture.is_capturable() and focus.is_focused()
        except (WindowGone, CaptureLost):
            capture.close()
            capture = None
            return False

    print(f"policy {args.checkpoint} on {device}; {'DRY RUN: no input will be sent' if args.dry_run else 'LIVE'}")
    print(f"{args.kill_key.upper()} stops it; touching your mouse or keyboard takes over until you're idle")
    def notify(message: str) -> None:
        try:  # the player is looking at the game by now, not at this terminal
            subprocess.Popen(["notify-send", "--app-name", "ZombiesAI", "--expire-time",
                              str(int(args.countdown * 1000)), "play_real", message.replace("recording", "AI playing")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass

    if not wait_to_start(ready, timeout=args.wait_timeout, countdown=args.countdown, discard=human.source.drain,
                         notify=notify):
        raise SystemExit("gave up waiting for the game window to be on screen and focused")

    if args.dry_run:
        sink = FakeSink()
    else:
        from zombiesai.realgame.uinput import UinputDevice

        sink = UinputDevice(dispatch_config.codes.values())
    dispatcher = ActionDispatcher(sink, dispatch_config)
    out = next_dir(args.out, "dry" if args.dry_run else "play")
    writer = ClipWriter(
        out,
        source={**capture.describe(), "policy": str(args.checkpoint), "dry_run": args.dry_run,
                "temperature": args.temperature, "deterministic": args.deterministic},
        label_source="agent",
        config={"play": asdict(config),
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
    try:
        summary = play(capture, agent, dispatcher, focus=focus, human=human, config=config, writer=writer)
    finally:
        dispatcher.close()  # releases everything again and destroys the virtual device
        capture.close()
        human.source.close()

    from zombiesai.demos.clips import load_clip

    clip = load_clip(out)
    acted = clip.usable()
    print(f"wrote {out}: {summary['steps']} steps ({summary['seconds'] / 60:.1f} min), played {summary['acted']}, "
          f"paused unfocused {summary['unfocused']} / frozen {summary['frozen']} / you {summary['human']}, "
          f"overruns {summary['overruns']}")
    print(f"  what it did: {describe_actions(clip.actions[acted])}")


if __name__ == "__main__":
    main()
