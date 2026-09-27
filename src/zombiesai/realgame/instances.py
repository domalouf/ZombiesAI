"""Several World at War instances at once, each in an X server of its own.

PLAN.md's first "fact that drives every decision" is that the real game cannot be parallelized: WaW's
DirectInput only listens to the foreground window, and a desktop has one. That was checked on Windows, where a
desktop session is the unit and multiplying sessions needs a GPU per seat. On Linux the unit is the *X server*,
and X servers are cheap. So each instance here gets:

* **A rootful Xwayland** (`:60`, `:61`, ...) -- a complete X server of its own, GPU-accelerated through the
  compositor, sized to the game's resolution. By default it is opened on a hidden Hyprland special workspace
  (`special:zombiesai`), floating, at a fixed size, so the instances neither take your focus nor retile your
  windows. `hyprctl dispatch togglespecialworkspace zombiesai` shows them all when you want to watch.
  Checked on this machine: Vulkan-over-X11 (DXVK's path) renders at 60 fps in such a server while it is
  hidden, and MIT-SHM capture reads every frame (Xwayland paces a client's copy-presents with its own timer,
  not the compositor's frame callbacks).
* **Its own game process**, launched with `umu-run` into a private copy of the Steam prefix (Wine keeps its
  named objects, and so any single-instance guard, per prefix), with the settings the agent needs passed as
  `+set` arguments on the command line rather than edited into your config: the resolution (fullscreen at the
  X server's own size), the console (for resets), mouse acceleration and smoothing off, and `+map` straight
  into Nacht.
* **Its own input** (`xtest.XTestSink`): injected into that X server only.
* **Its own audio sink** -- a PulseAudio/PipeWire null sink -- so a policy that hears (`demos/hearing.py`)
  hears its own game and not the other five.

Nothing here touches the desktop's own X server, your mouse, or your keyboard.
"""

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from zombiesai.demos.game_settings import WAW_STEAM_APP_ID
from zombiesai.realgame.xtest import display_env

STEAM_ROOTS = (Path.home() / ".local" / "share" / "Steam", Path.home() / ".steam" / "steam")
GAME_DIR_NAME = "Call of Duty World at War"
PROTON_NAME = "Proton - Experimental"
SPECIAL_WORKSPACE = "zombiesai"
WINDOW_TITLE = "Call of Duty"
NACHT = "nazi_zombie_prototype"


def _steam_path(*parts: str) -> Path | None:
    for root in STEAM_ROOTS:
        path = root.joinpath(*parts)
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class FleetConfig:
    """How to run N instances. Every path defaults to what Steam installed; override any of them."""

    n: int = 4
    width: int = 1280
    height: int = 720
    display_base: int = 60  # instance i lives on DISPLAY=:{display_base + i}
    root: str = "runs/instances"
    hidden: bool = True  # a hidden Hyprland special workspace; False opens ordinary windows
    game_dir: str | None = None
    exe: str = "CoDWaW.exe"
    template_prefix: str | None = None  # a Steam compatdata dir to copy (default: WaW's own)
    proton: str | None = None
    launcher: str = "umu-run"
    # Passed to the game after the exe. The engine takes `+set dvar value` and `+command` on its command line,
    # which leaves your config.cfg alone. monkeytoy 0 enables the console the reset FSM types into.
    dvars: dict = field(default_factory=lambda: {
        # Fullscreen at exactly the X server's size: with no window manager, a window is wherever Wine puts it,
        # and any offset leaves it partly off a root its own size, which X refuses to capture (BadMatch).
        "r_fullscreen": "1",
        "com_introplayed": "1",
        "monkeytoy": "0",
        "cl_mouseAccel": "0",
        "m_filter": "0",
        "r_vsync": "0",
        "com_maxfps": "60",
    })
    start_command: str = f"map {NACHT}"
    extra_env: dict = field(default_factory=dict)
    audio_sinks: bool = True

    def resolved_game_dir(self) -> Path:
        path = Path(self.game_dir) if self.game_dir else _steam_path("steamapps", "common", GAME_DIR_NAME)
        if path is None or not (path / self.exe).exists():
            raise FileNotFoundError(f"no {self.exe} under {path}; pass --game-dir")
        return path

    def resolved_template(self) -> Path:
        path = Path(self.template_prefix) if self.template_prefix else _steam_path(
            "steamapps", "compatdata", str(WAW_STEAM_APP_ID))
        if path is None or not (path / "pfx").is_dir():
            raise FileNotFoundError(f"no Proton prefix at {path}; run the game once from Steam, or pass --template-prefix")
        return path

    def resolved_proton(self) -> Path:
        path = Path(self.proton) if self.proton else _steam_path("steamapps", "common", PROTON_NAME)
        if path is None or not (path / "proton").exists():
            raise FileNotFoundError(f"no Proton at {path}; pass --proton")
        return path


@dataclass(frozen=True)
class InstanceSpec:
    index: int
    display: str
    width: int
    height: int
    dir: Path

    @property
    def number(self) -> int:
        return int(self.display.lstrip(":"))

    @property
    def prefix(self) -> Path:
        return self.dir / "compat"

    @property
    def sink(self) -> str:
        return f"zombiesai_{self.index}"

    @property
    def monitor(self) -> str:
        return f"{self.sink}.monitor"

    @property
    def socket(self) -> Path:
        return Path("/tmp/.X11-unix") / f"X{self.number}"


def specs(config: FleetConfig) -> list[InstanceSpec]:
    root = Path(config.root)
    return [
        InstanceSpec(i, f":{config.display_base + i}", config.width, config.height, root / f"i{i}")
        for i in range(config.n)
    ]


def game_args(config: FleetConfig, spec: InstanceSpec) -> list[str]:
    """The exe's arguments: the resolution, the dvars, then the command that loads Nacht."""
    args = ["+set", "r_mode", f"{spec.width}x{spec.height}"]
    for name, value in config.dvars.items():
        args += ["+set", name, str(value)]
    if config.start_command:
        args += ["+" + config.start_command.split()[0], *config.start_command.split()[1:]]
    return args


def game_env(config: FleetConfig, spec: InstanceSpec, base: dict | None = None) -> dict[str, str]:
    """The game's environment: its own display and nothing else, its own prefix, its own sink."""
    env = display_env(spec.display) if base is None else {k: v for k, v in base.items() if k != "WAYLAND_DISPLAY"}
    env["DISPLAY"] = spec.display
    env.update({
        "WINEPREFIX": str(spec.prefix.resolve()),
        "PROTONPATH": str(config.resolved_proton()),
        "GAMEID": f"umu-{WAW_STEAM_APP_ID}",
        "STORE": "steam",
        # Lets steam_api find the running Steam client, as a Steam launch would.
        "SteamAppId": str(WAW_STEAM_APP_ID),
        "SteamGameId": str(WAW_STEAM_APP_ID),
        # Proton would otherwise prefer its Wayland driver when it finds a compositor.
        "PROTON_ENABLE_WAYLAND": "0",
    })
    if config.audio_sinks:
        env["PULSE_SINK"] = spec.sink
    env.update({k: str(v) for k, v in config.extra_env.items()})
    return env


# ------------------------------------------------------------------------------------------------ prefixes


def prepare_prefix(config: FleetConfig, spec: InstanceSpec, *, say=print) -> Path:
    """A private copy of the Steam prefix for this instance, made once. It carries your config.cfg (bindings,
    sensitivity) -- the same units the demos were recorded in -- and Proton's own version stamp, so Proton
    does not rebuild it."""
    if (spec.prefix / "pfx").is_dir():
        return spec.prefix
    template = config.resolved_template()
    spec.dir.mkdir(parents=True, exist_ok=True)
    say(f"  instance {spec.index}: copying the prefix {template} -> {spec.prefix} (once)")
    tmp = spec.dir / "compat.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    # --reflink shares blocks on btrfs/xfs and falls back to a plain copy elsewhere.
    result = subprocess.run(["cp", "-a", "--reflink=auto", str(template), str(tmp)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"copying the prefix failed: {result.stderr.strip()}")
    tmp.rename(spec.prefix)
    return spec.prefix


# ------------------------------------------------------------------------------------------------ displays


def hyprland_dispatch(lua: str, classic: list[str], *, run=subprocess.run) -> bool:
    """Run a Hyprland dispatcher in whichever config dialect this Hyprland speaks: Lua (0.55+, what Omarchy
    ships) first, then the classic one. Stops at the first that answers "ok", so nothing runs twice."""
    for args in (["hyprctl", "dispatch", lua], ["hyprctl", "dispatch", *classic]):
        try:
            result = run(args, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode == 0 and result.stdout.strip() == "ok":
            return True
    return False


def hyprland_exec(command: str, rules: str, *, run=subprocess.run) -> bool:
    """Ask Hyprland to launch `command` under window rules."""
    return hyprland_dispatch(f'hl.dsp.exec_cmd("[{rules}] {command}")', ["exec", f"[{rules}] {command}"], run=run)


def toggle_shown(*, run=subprocess.run) -> bool:
    """Show or hide the instances' special workspace."""
    return hyprland_dispatch(f'hl.dsp.workspace.toggle_special("{SPECIAL_WORKSPACE}")',
                             ["togglespecialworkspace", SPECIAL_WORKSPACE], run=run)


def xwayland_command(spec: InstanceSpec) -> str:
    return f"Xwayland {spec.display} -geometry {spec.width}x{spec.height}"


def display_rules(spec: InstanceSpec) -> str:
    return f"workspace special:{SPECIAL_WORKSPACE} silent; float; size {spec.width} {spec.height}"


def server_pids(spec: InstanceSpec) -> list[int]:
    result = subprocess.run(["pgrep", "-f", f"^Xwayland {spec.display}( |$)"], capture_output=True, text=True)
    return [int(p) for p in result.stdout.split()]


def display_alive(spec: InstanceSpec) -> bool:
    return spec.socket.exists() and bool(server_pids(spec))


def start_display(config: FleetConfig, spec: InstanceSpec, *, timeout_s: float = 10.0, say=print) -> None:
    if display_alive(spec):
        return
    if spec.socket.exists():
        raise RuntimeError(f"{spec.socket} exists but no Xwayland {spec.display} is running; another X server "
                           f"owns display {spec.display} -- pick another --display-base")
    command = xwayland_command(spec)
    launched = config.hidden and os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and hyprland_exec(
        command, display_rules(spec))
    if not launched:
        if config.hidden:
            say("  (not under Hyprland, or it refused the rule: the X servers open as ordinary windows)")
        env = {k: v for k, v in os.environ.items()}
        subprocess.Popen(command.split(), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if display_alive(spec):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Xwayland {spec.display} did not come up within {timeout_s:.0f} s")


def stop_display(spec: InstanceSpec) -> None:
    for pid in server_pids(spec):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


# ------------------------------------------------------------------------------------------------ audio


def ensure_sink(spec: InstanceSpec, *, run=subprocess.run) -> bool:
    """A null sink for this instance's sound, made if missing. False when there is no pactl."""
    try:
        listed = run(["pactl", "list", "short", "sinks"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if any(line.split("\t")[1:2] == [spec.sink] for line in listed.stdout.splitlines()):
        return True
    made = run(["pactl", "load-module", "module-null-sink", f"sink_name={spec.sink}",
                f"sink_properties=device.description=ZombiesAI-instance-{spec.index}"],
               capture_output=True, text=True, timeout=5)
    return made.returncode == 0


def remove_sink(spec: InstanceSpec) -> None:
    try:
        modules = subprocess.run(["pactl", "list", "short", "modules"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return
    for line in modules.stdout.splitlines():
        if f"sink_name={spec.sink}" in line:
            subprocess.run(["pactl", "unload-module", line.split("\t")[0]], capture_output=True, timeout=5)


# ------------------------------------------------------------------------------------------------ the game


class Instance:
    """One game in one X server: bring it up, find its window, give it focus, restart it when it dies."""

    def __init__(self, config: FleetConfig, spec: InstanceSpec, *, say=print):
        self.config, self.spec, self.say = config, spec, say
        self.proc: subprocess.Popen | None = None
        self.launches = 0

    @property
    def pid_file(self) -> Path:
        return self.spec.dir / "game.pid"

    def up(self) -> None:
        prepare_prefix(self.config, self.spec, say=self.say)
        start_display(self.config, self.spec, say=self.say)
        if self.config.audio_sinks and not ensure_sink(self.spec):
            self.say(f"  instance {self.spec.index}: no pactl, so its sound goes to the default sink")
        if not self.game_running():
            self.launch_game()

    def launch_game(self) -> None:
        game_dir = self.config.resolved_game_dir()
        self.spec.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.spec.dir / "game.log", "ab")
        command = [self.config.launcher, str(game_dir / self.config.exe), *game_args(self.config, self.spec)]
        log.write(f"\n=== {time.ctime()} launching: {' '.join(command)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(command, cwd=game_dir, env=game_env(self.config, self.spec),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log.close()
        self.pid_file.write_text(str(self.proc.pid))
        self.launches += 1
        self.say(f"  instance {self.spec.index}: launched the game on {self.spec.display} (pid {self.proc.pid})")

    def game_pid(self) -> int | None:
        if self.proc is not None:
            return self.proc.pid if self.proc.poll() is None else None
        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError):
            return None

    def game_running(self) -> bool:
        return self.game_pid() is not None

    def stop_game(self, timeout_s: float = 10.0) -> None:
        """Kill the launcher's whole session, then make sure the prefix's wineserver is gone too."""
        pid = self.game_pid()
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            wineserver = self.config.resolved_proton() / "files" / "bin" / "wineserver"
            subprocess.run([str(wineserver), "-k"], env={**os.environ, "WINEPREFIX": str((self.spec.prefix / "pfx").resolve())},
                           capture_output=True, timeout=timeout_s)
        except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if pid is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    os.killpg(pid, 0)
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.2)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self.proc = None
        self.pid_file.unlink(missing_ok=True)

    def restart_game(self) -> None:
        self.say(f"  instance {self.spec.index}: restarting the game")
        self.stop_game()
        if not display_alive(self.spec):
            start_display(self.config, self.spec, say=self.say)
        self.launch_game()

    def down(self) -> None:
        self.stop_game()
        stop_display(self.spec)
        if self.config.audio_sinks:
            remove_sink(self.spec)

    def window(self):
        """The game's window on this instance's display, or raise WindowNotFound."""
        from zombiesai.demos.x11_capture import find_window

        return find_window(WINDOW_TITLE, self.spec.display)

    def capture(self, hud_regions=None):
        """A capture that follows the game window on this display across the game's own window churn."""
        from zombiesai.demos.capture import FollowWindow, ScreenCapture
        from zombiesai.demos.hud_crops import HUD_REGIONS

        regions = hud_regions if hud_regions is not None else {k: HUD_REGIONS[k] for k in ("points_ammo", "round")}
        # The crops' reference size is 1440p at half scale, i.e. 720p's own pixels.
        scale = min(1.0, 720.0 / self.spec.height)
        return FollowWindow(lambda: ScreenCapture(window=WINDOW_TITLE, display=self.spec.display,
                                                  hud_regions=regions, hud_scale=scale))

    def sink(self):
        from zombiesai.realgame.xtest import XTestSink

        return XTestSink(self.spec.display)

    def focuser(self, sink, *, refind_s: float = 2.0, clock=time.monotonic):
        """A `focus()` for RealGameEnv: keep the game window holding this server's input focus, and say
        whether there is a game window at all. The window is looked up again at most every `refind_s` --
        WaW replaces its window during start-up and on a video restart."""
        state = {"window": None, "next": 0.0}

        def focus() -> bool:
            from zombiesai.demos.x11_capture import X11Error

            now = clock()
            if state["window"] is None and now < state["next"]:
                return False
            try:
                if state["window"] is None:
                    state["next"] = now + refind_s
                    state["window"] = self.window().id
                sink.focus(state["window"])
                return True
            except (X11Error, OSError):
                state["window"] = None
                return False

        return focus

    def status(self) -> dict:
        out = {"index": self.spec.index, "display": self.spec.display, "display_alive": display_alive(self.spec),
               "game_pid": self.game_pid(), "prefix": str(self.spec.prefix), "sink": self.spec.sink}
        if out["display_alive"]:
            try:
                w = self.window()
                out["window"] = {"id": hex(w.id), "title": w.title, "size": [w.width, w.height]}
            except Exception as error:  # noqa: BLE001 -- a status line, never a crash
                out["window"] = f"none ({str(error)[:80]})"
        return out


def fleet(config: FleetConfig, *, say=print) -> list[Instance]:
    return [Instance(config, spec, say=say) for spec in specs(config)]


def save_fleet(config: FleetConfig) -> Path:
    path = Path(config.root) / "fleet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2))
    return path


def load_fleet(root: str | Path = "runs/instances") -> FleetConfig:
    path = Path(root) / "fleet.json"
    if not path.exists():
        raise FileNotFoundError(f"no fleet at {path}; start one with scripts/instances.py up")
    return FleetConfig(**json.loads(path.read_text()))
