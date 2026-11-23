"""The fleet's game on your own desktop, for recording demos: Plutonium T4 in LAN mode, offline, played by you.

The agent trains on Plutonium's co-op Nacht (last stand and all) with the fleet's config.cfg; the Steam game
the first demos were recorded on is solo, with no last stand. Recording on the same client keeps the demos
and the fleet's games the same game. This starts it the way `instances.py` starts a fleet game:

* **Its own copies** of Plutonium and the Steam prefix under `runs/desktop/` (Plutonium keeps the profile and
  console.log beside its executable, so sharing an install with the fleet would make both hang).
* **The fleet's config.cfg** (`instances.game_config`: the one installed for the fleet, else your Steam
  profile's), copied over the profile each launch, so your bindings and sensitivity are the fleet's.
* **Offline**: a network namespace with loopback only (`instances.offline_wrapper`). The X socket and the
  sound server are files, which a network namespace does not hide.
* **Its own sound**: a null sink with a loopback to your speakers, so a recording's audio is the game alone
  (`zai_desktop.monitor`) and not the whole desktop. Not named `zombiesai_*`: a fleet shutdown removes those.
* **A key that starts Nacht** (F9): in LAN mode the menu cannot start a co-op game and a `+map` on the command
  line is thrown back to it, so the launcher binds `map nazi_zombie_prototype` to a key. Press it at the main
  menu and after every game over. The console (`~`) also works.

Nothing it passes changes the game's look or feel beyond what the demos need: fullscreen at the given size,
mouse acceleration and smoothing off, the console on. Your own frame rate and graphics settings stand.
"""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

from zombiesai.realgame.instances import (
    NACHT, PLUTONIUM_PROFILE, FleetConfig, Instance, InstanceSpec, game_command, game_config, game_env,
    load_fleet, offline_wrapper, prepare_plutonium, prepare_prefix, sync_plutonium_profile,
)

DESKTOP_ROOT = "runs/desktop"
SINK = "zai_desktop"
START_KEY = "F9"
# The fleet's dvars without its frame cap and vsync choice: those are for 7 games sharing a PC, not for you.
# Aim assist off: Plutonium's LAN game has it on for the mouse too (Steam's does not), and aiming down sights
# then pulls the view onto the nearest zombie -- a turn no mouse count records, so the labels would not match
# the picture. sv_allowAimAssist is the server's half: the LAN game is its own server.
DESKTOP_DVARS = {"r_fullscreen": "1", "com_introplayed": "1", "monkeytoy": "0", "cl_mouseAccel": "0",
                 "m_filter": "0", "cg_aimAssistEnabled": "0", "sv_allowAimAssist": "0"}


def desktop_config(fleet_root: str | Path = "runs/instances", *, root: str | Path = DESKTOP_ROOT,
                   width: int | None = None, height: int | None = None, sink: str | None = SINK) -> FleetConfig:
    """The fleet's own settings (game, Proton, Plutonium), for one game on the desktop. Without a fleet yet,
    the defaults a fleet would have."""
    try:
        base = load_fleet(fleet_root)
    except FileNotFoundError:
        base = FleetConfig()
    env = dict(base.extra_env)
    if sink:
        env["PULSE_SINK"] = sink
    return replace(base, n=1, root=str(root), width=width or base.width, height=height or base.height,
                   client="plutonium", offline=True, audio_sinks=False, dvars=dict(DESKTOP_DVARS),
                   extra_env=env)


def desktop_spec(config: FleetConfig, display: str) -> InstanceSpec:
    return InstanceSpec(0, display, config.width, config.height, Path(config.root))


def start_args(key: str = START_KEY) -> list[str]:
    """Binds `key` to loading Nacht. `bind` takes the rest of its line as the command, so no quotes, which
    would have to survive Wine's command-line parsing."""
    return ["+bind", key, "map", NACHT]


def profile_config(config: FleetConfig) -> Path:
    """The config.cfg the desktop game plays with: what a recording should stamp (`record_demo --game-config`)."""
    return desktop_spec(config, ":0").plutonium / PLUTONIUM_PROFILE / "config.cfg"


def desktop_command(config: FleetConfig, spec: InstanceSpec, key: str = START_KEY) -> tuple[list[str], Path]:
    command, cwd = game_command(config, spec)
    return offline_wrapper() + command + start_args(key), cwd


# ------------------------------------------------------------------------------------------------ audio


def _pactl(*args: str, run=subprocess.run) -> subprocess.CompletedProcess | None:
    try:
        return run(["pactl", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _modules(sink: str, run=subprocess.run) -> list[tuple[str, str]]:
    """(id, line) of every module that is this sink or feeds it to the speakers."""
    listed = _pactl("list", "short", "modules", run=run)
    if listed is None:
        return []
    out = []
    for line in listed.stdout.splitlines():
        if f"sink_name={sink}" in line or f"source={sink}.monitor" in line:
            out.append((line.split("\t")[0], line))
    return out


def ensure_sound(sink: str = SINK, *, run=subprocess.run) -> bool:
    """The game's own null sink, and a loopback from its monitor to the default output so you hear it. False
    when there is no pactl (the game then plays through the default output, and so would a recording)."""
    have = _modules(sink, run=run)
    if not any("module-null-sink" in line for _, line in have):
        made = _pactl("load-module", "module-null-sink", f"sink_name={sink}",
                      "sink_properties=device.description=ZombiesAI-desktop-game", run=run)
        if made is None or made.returncode:
            return False
    if not any("module-loopback" in line for _, line in have):
        # No sink given: the loopback follows the default output, wherever you move it.
        made = _pactl("load-module", "module-loopback", f"source={sink}.monitor", "latency_msec=20",
                      "source_dont_move=true", run=run)
        if made is None or made.returncode:
            return False
    return True


def remove_sound(sink: str = SINK, *, run=subprocess.run) -> int:
    # The loopback first: unloading its source under it leaves a stream with nowhere to go.
    found = sorted(_modules(sink, run=run), key=lambda m: "module-loopback" not in m[1])
    for module, _ in found:
        _pactl("unload-module", module, run=run)
    return len(found)


# ------------------------------------------------------------------------------------------------ the game


class DesktopGame(Instance):
    """One game on an existing X display (yours). Up launches it; down quits it and its sound."""

    def __init__(self, config: FleetConfig, spec: InstanceSpec, *, fleet_root: str | Path = "runs/instances",
                 key: str = START_KEY, sink: str | None = SINK, say=print):
        super().__init__(config, spec, say=say)
        self.fleet_root, self.key, self.sound = Path(fleet_root), key, sink

    def up(self) -> None:
        prepare_prefix(self.config, self.spec, say=self.say)
        if self.sound and not ensure_sound(self.sound):
            self.say("  no pactl: the game plays through the default output, and a recording would hear "
                     "everything on it")
        if not self.game_running():
            self.launch_game()

    def launch_game(self) -> None:
        prepare_plutonium(self.config, self.spec, say=self.say)
        source = game_config(self.fleet_root)
        copied = sync_plutonium_profile(self.spec.plutonium, source=source, say=self.say)
        if copied is not None:
            self.say(f"  config.cfg: {source}")
        command, cwd = desktop_command(self.config, self.spec, self.key)
        self.spec.dir.mkdir(parents=True, exist_ok=True)
        with open(self.spec.dir / "game.log", "ab") as log:
            log.write(f"\n=== launching on {self.spec.display}: {' '.join(command)}\n".encode())
            log.flush()
            self.proc = subprocess.Popen(command, cwd=cwd, env=game_env(self.config, self.spec, base=dict(os.environ)),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.pid_file.write_text(str(self.proc.pid))
        self.launches += 1
        self.say(f"  launched Plutonium T4 (LAN, offline) on {self.spec.display}, pid {self.proc.pid}")

    def down(self) -> None:
        self.stop_game()
        if self.sound:
            remove_sound(self.sound)
