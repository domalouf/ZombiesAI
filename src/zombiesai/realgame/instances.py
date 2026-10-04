"""Several World at War instances at once, each in an X server of its own.

PLAN.md's first "fact that drives every decision" is that the real game cannot be parallelized: WaW's
DirectInput only listens to the foreground window, and a desktop has one. That was checked on Windows, where a
desktop session is the unit and multiplying sessions needs a GPU per seat. On Linux the unit is the *X server*,
and X servers are cheap. So each instance here gets:

* **A rootful Xwayland** (`:60`, `:61`, ...) -- a complete X server of its own, sized to the game's resolution,
  hosted by a headless Weston of exactly that size (`host="weston"`), which nothing on the desktop can resize or
  hide. Under Weston it runs glamor, so the game's frames arrive as GPU buffers, whole, at 60 fps (see
  XWAYLAND_FLAGS). Without weston it opens on a hidden Hyprland special workspace instead, in `-shm` mode --
  fine until the monitor sleeps (see `_start_under_hyprland`). MIT-SHM capture reads every frame.
* **Its own game process**, launched with `umu-run` into a private copy of the Steam prefix (Wine keeps its
  named objects, and so any single-instance guard, per prefix). The client is Plutonium's T4 in LAN mode by
  default (`client="plutonium"`): Steam's `CoDWaW.exe` is wrapped in SteamStub DRM and, started outside the
  Steam client, stops at "Application load error P:0000065432"; Plutonium runs the same game files through its
  own executable and never asks Steam. By default (`offline=True`) the game runs in a network namespace with
  loopback only. Either way, the settings the agent needs passed as
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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from zombiesai.demos.game_settings import WAW_STEAM_APP_ID
from zombiesai.realgame.xtest import display_env

STEAM_ROOTS = (Path.home() / ".local" / "share" / "Steam", Path.home() / ".steam" / "steam")
GAME_DIR_NAME = "Call of Duty World at War"
PROTON_NAME = "Proton - Experimental"
SPECIAL_WORKSPACE = "zombiesai"
WINDOW_TITLE = "Call of Duty"  # the Steam client's window
PLUTONIUM_TITLE = "Plutonium T4"  # "Plutonium T4 Co-Op (r5354) (LAN)"; it also opens a small "Call of Duty®" window
PLUTONIUM_DIR = Path.home() / ".local" / "share" / "plutonium"  # where plutonium-updater -d puts it
PLUTONIUM_BOOTSTRAPPER = Path("bin") / "plutonium-bootstrapper-win32.exe"
# Plutonium keeps its profiles in its own storage, not the prefix. `$$$` is the profile it makes on first run.
PLUTONIUM_PROFILE = Path("storage") / "t4" / "players" / "profiles" / "$$$"
GAME_CONFIG = "config.cfg"  # in the fleet's root: the settings every game plays with, if installed there
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
    width: int = 2560  # the HUD glyph atlas is 1440p read at half scale; native 720p text does not match it
    height: int = 1440
    display_base: int = 60  # instance i lives on DISPLAY=:{display_base + i}
    root: str = "runs/instances"
    # Who hosts the X servers: "weston" -- a headless Weston per instance, nothing on your desktop (the default
    # when weston is installed) -- or "hyprland", your own session, on a hidden special workspace. "auto" picks.
    host: str = "auto"
    hidden: bool = True  # hyprland host: a hidden special workspace; False opens ordinary windows
    game_dir: str | None = None
    exe: str = "CoDWaW.exe"
    template_prefix: str | None = None  # a Steam compatdata dir to copy (default: WaW's own)
    proton: str | None = None
    launcher: str = "umu-run"
    client: str = "plutonium"  # "plutonium" (T4, LAN mode) or "steam" (CoDWaW.exe; needs the Steam client)
    plutonium_dir: str | None = None
    # Run each game with no network at all: a network namespace with only loopback (the solo game still talks
    # to its own local server over 127.0.0.1). Plutonium's -lan mode already never logs in or contacts its
    # servers; this makes that a guarantee rather than a promise. Needs unprivileged user namespaces.
    offline: bool = True
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
        # Twice the policy's 15 Hz is enough to see, and each game costs about two thirds of the CPU of 60 fps.
        "com_maxfps": "30",
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

    def resolved_host(self) -> str:
        if self.host == "auto":
            return "weston" if shutil.which("weston") else "hyprland"
        if self.host not in ("weston", "hyprland"):
            raise ValueError(f"unknown host {self.host!r}: weston, hyprland or auto")
        return self.host

    def resolved_plutonium(self) -> Path:
        path = Path(self.plutonium_dir) if self.plutonium_dir else PLUTONIUM_DIR
        if not (path / PLUTONIUM_BOOTSTRAPPER).exists():
            raise FileNotFoundError(
                f"no Plutonium at {path}; install it with mxve/plutonium-updater.rs "
                f"(`plutonium-updater -d {path}`), or pass --plutonium-dir")
        return path

    @property
    def window_title(self) -> str:
        return PLUTONIUM_TITLE if self.client == "plutonium" else WINDOW_TITLE

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
    def plutonium(self) -> Path:
        return self.dir / "plutonium"

    @property
    def sink(self) -> str:
        return f"zombiesai_{self.index}"

    @property
    def monitor(self) -> str:
        return f"{self.sink}.monitor"

    @property
    def socket(self) -> Path:
        return Path("/tmp/.X11-unix") / f"X{self.number}"

    @property
    def wayland_socket(self) -> str:
        """The name of this instance's headless Weston socket, under $XDG_RUNTIME_DIR (weston host)."""
        return f"zombiesai-{self.number}"


def specs(config: FleetConfig) -> list[InstanceSpec]:
    root = Path(config.root)
    return [
        InstanceSpec(i, f":{config.display_base + i}", config.width, config.height, root / f"i{i}")
        for i in range(config.n)
    ]


def game_args(config: FleetConfig, spec: InstanceSpec, *, start: bool = True) -> list[str]:
    """The exe's arguments: the resolution, the dvars, then (if `start`) the command that loads Nacht."""
    args = ["+set", "r_mode", f"{spec.width}x{spec.height}"]
    for name, value in config.dvars.items():
        args += ["+set", name, str(value)]
    if start and config.start_command:
        args += ["+" + config.start_command.split()[0], *config.start_command.split()[1:]]
    return args


def windows_path(path: Path) -> str:
    """A Linux path as Wine sees it: the whole filesystem is drive Z:."""
    return "Z:" + str(Path(path).resolve()).replace("/", "\\")


def game_command(config: FleetConfig, spec: InstanceSpec) -> tuple[list[str], Path]:
    """The command that starts this instance's game, and the directory to start it in."""
    game_dir = config.resolved_game_dir()
    if config.client == "plutonium":
        pluto = spec.plutonium.resolve()
        # -lan: no Plutonium login, no servers, no anti-cheat; the game is offline and solo.
        command = [config.launcher, str(pluto / PLUTONIUM_BOOTSTRAPPER), "t4sp", windows_path(game_dir),
                   "-lan", "+name", f"zombiesai{spec.index}"]
        # No `+map`: in LAN mode the co-op menu asks for an "online profile" it cannot make, and a map started
        # from the command line is thrown back to that dialog. The same `map` typed into the console once the
        # menu is up loads and stays -- which is what RealGameEnv's reset types.
        return command + game_args(config, spec, start=False), pluto
    if config.client != "steam":
        raise ValueError(f"unknown client {config.client!r}: plutonium or steam")
    return [config.launcher, str(game_dir / config.exe), *game_args(config, spec)], game_dir


def prepare_plutonium(config: FleetConfig, spec: InstanceSpec, *, say=print) -> Path:
    """A private copy of Plutonium for this instance, remade whenever the installed revision changes.

    Plutonium keeps everything the running game writes -- the profile, console.log, mpdata -- in `storage/`
    beside its executable, so two games started from one install share it, and the second hangs on a black
    screen. A copy each is the fix; on btrfs/xfs `--reflink` makes it share the blocks (~430 MB otherwise)."""
    source = config.resolved_plutonium()
    stamp = (source / "cdn_info.json").read_bytes() if (source / "cdn_info.json").exists() else b""
    target = spec.plutonium
    if target.is_dir() and (target / "cdn_info.json").exists() and (target / "cdn_info.json").read_bytes() == stamp:
        return target
    say(f"  instance {spec.index}: copying Plutonium {source} -> {target}")
    spec.dir.mkdir(parents=True, exist_ok=True)
    tmp = spec.dir / "plutonium.partial"
    for old in (tmp, target):
        if old.exists():
            shutil.rmtree(old)
    result = subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(tmp)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"copying Plutonium failed: {result.stderr.strip()}")
    tmp.rename(target)
    return target


def offline_wrapper() -> list[str]:
    """A command prefix that runs the rest with loopback and nothing else. Two user namespaces: the outer maps
    us to root just long enough to bring `lo` up, the inner maps us back to our own uid, so Wine and the
    prefix see the same owner as always."""
    return ["unshare", "--user", "--map-root-user", "--net", "sh", "-c",
            f'ip link set lo up && exec unshare --user --map-user={os.getuid()} --map-group={os.getgid()} -- "$@"',
            "sh"]


def game_config(fleet_root: str | Path) -> Path | None:
    """The config.cfg the fleet's games play with: one installed in the fleet's root (the training PC's, which
    `scripts/fleet.py prep` puts on every PC that plays for it -- rl/fleet.py), else your Steam profile's.

    Only Plutonium reads an installed one: `launch_game` copies it into each instance's Plutonium profile.
    Steam's CoDWaW.exe reads the profile inside the instance's own copy of the Steam prefix, which nothing here
    writes to, so for a fleet saved with `client="steam"` this is that profile, whatever is installed. Without
    a fleet.json (`instances.py up` has not run yet) the fleet will be Plutonium's, the default."""
    from zombiesai.demos.game_settings import candidate_configs

    root = Path(fleet_root)
    try:
        config = load_fleet(root)
    except FileNotFoundError:
        config = None
    if config is not None and config.client == "steam":
        return prefix_config(config)
    installed = root / GAME_CONFIG
    if installed.is_file():
        return installed
    found = candidate_configs()
    return found[0] if found else None


# Where the game keeps its profiles inside a Proton prefix (as demos/game_settings.candidate_configs looks).
PREFIX_PROFILES = Path("pfx/drive_c/users/steamuser/AppData/Local/Activision/CoDWaW/players/profiles")


def prefix_config(config: FleetConfig) -> Path | None:
    """The config.cfg a steam-client game reads: the newest profile in the first instance's copy of the prefix
    (every copy is made from the same template, once), or, before there is one, in the template it will be
    copied from. None if neither has a profile."""
    prefixes = [spec.prefix for spec in specs(config)[:1]]
    try:
        prefixes.append(config.resolved_template())
    except FileNotFoundError:
        pass
    for prefix in prefixes:
        found = sorted((prefix / PREFIX_PROFILES).glob("*/config.cfg"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        if found:
            return found[0]
    return None


def sync_plutonium_profile(root: Path, *, source: Path | None = None, say=print) -> Path | None:
    """Copy a config.cfg -- `source`, or your Steam profile's -- over Plutonium's before a launch, so the
    instances play with the bindings and sensitivity the demos were recorded with. Plutonium's own defaults
    differ where it matters: right mouse is `+toggleads_throw` there, `+speed_throw` (hold) in the Steam game."""
    from zombiesai.demos.game_settings import candidate_configs

    if source is None:
        found = candidate_configs()
        source = found[0] if found else None
    if source is None:
        say("  no Steam config.cfg to copy: Plutonium plays with its own bindings -- check them")
        return None
    profile = root / PLUTONIUM_PROFILE
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, profile / "config.cfg")
    # Without active.txt naming a profile the game opens on "Create Online Profile", and `+map` never runs.
    (profile.parent / "active.txt").write_text(profile.name)
    return profile / "config.cfg"


def game_env(config: FleetConfig, spec: InstanceSpec, base: dict | None = None) -> dict[str, str]:
    """The game's environment: its own display and nothing else, its own prefix, its own sink."""
    env = display_env(spec.display) if base is None else {k: v for k, v in base.items() if k != "WAYLAND_DISPLAY"}
    env["DISPLAY"] = spec.display
    env.update({
        "WINEPREFIX": str(spec.prefix.resolve()),
        "PROTONPATH": str(config.resolved_proton()),
        "GAMEID": f"umu-{WAW_STEAM_APP_ID}",
        "STORE": "steam",
        # Offline means offline: no runtime update check either (the runtime is already installed).
        "UMU_RUNTIME_UPDATE": "0",
        # Proton would otherwise prefer its Wayland driver when it finds a compositor.
        "PROTON_ENABLE_WAYLAND": "0",
    })
    if config.client == "steam":
        # Lets steam_api find the running Steam client, as a Steam launch would.
        env["SteamAppId"] = env["SteamGameId"] = str(WAW_STEAM_APP_ID)
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


# How the game's frames reach the X server decides what the game and every capture of it cost.
#
# Under a headless Weston: glamor. Weston's GL renderer offers linux-dmabuf, so Xwayland has DRI3 and a frame is
# one GPU copy into the root. Measured 2026-09-27 at 1440p in Nacht: 60 fps delivered, each frame one whole
# damage rectangle; the game ~0.6 of a core, Xwayland + Weston ~4%; a full XShmGetImage 1.4 ms.
# `-shm` instead has no DRI3, so NVIDIA's Vulkan reads every frame back and sends it as PutImage requests: a
# 1440p frame is four 4 MB requests (409 rows each) arriving over ~65 ms. That caps the game at ~11 fps at
# ~1.2 cores (it waits on its own presents) and Xwayland at ~15%, and a grab on a clock lands mid-frame: most
# captured frames are torn into four bands of two different frames.
#
# Under Hyprland: `-shm`, because there Xwayland's glamor path (with Hyprland's explicit sync) aborts in
# xwl_glamor_gbm_dispose_syncpts as soon as WaW starts, and hidden only draws ~2 fps besides.
#
# `-noreset` on both: without it Xwayland resets when its last X client disconnects (a launcher's short probe
# connection does, before the game connects) and re-creates its window. Weston 15.0.1's kiosk shell answers a
# re-created window with a use-after-free (kiosk_shell_output_set_active_surface_tree) and segfaults.
XWAYLAND_FLAGS = {"weston": "-glamor gl -noreset", "hyprland": "-shm -noreset"}


def xwayland_command(spec: InstanceSpec, host: str = "weston") -> str:
    return f"Xwayland {spec.display} -geometry {spec.width}x{spec.height} {XWAYLAND_FLAGS[host]}"


def weston_command(spec: InstanceSpec) -> list[str]:
    """A headless compositor exactly the game's size. The kiosk shell makes the rootful Xwayland fullscreen,
    i.e. exactly the output, so the X root -- which follows its Wayland window's size -- is the game's size and
    nothing can change it. The GL renderer, although nothing is ever shown: it is what offers linux-dmabuf, and
    so DRI3 to Xwayland (XWAYLAND_FLAGS). Needs gl-renderer.so in the weston wrapper's WESTON_MODULE_MAP."""
    return ["weston", "--backend=headless", "--renderer=gl", "--shell=kiosk", f"--width={spec.width}",
            f"--height={spec.height}", f"--socket={spec.wayland_socket}", "--idle-time=0"]


def weston_pids(spec: InstanceSpec) -> list[int]:
    result = subprocess.run(["pgrep", "-f", f"weston .*--socket={spec.wayland_socket}( |$)"],
                            capture_output=True, text=True)
    return [int(p) for p in result.stdout.split()]


def runtime_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")


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
    host = config.resolved_host()
    command = xwayland_command(spec, host)
    if host == "weston":
        _start_under_weston(spec, command, timeout_s=timeout_s, say=say)
    else:
        _start_under_hyprland(config, spec, command, say=say)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if display_alive(spec):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Xwayland {spec.display} did not come up within {timeout_s:.0f} s")


def _quiet_env(**extra) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("WAYLAND_DISPLAY", "DISPLAY")}
    env.update(extra)
    return env


def _start_under_weston(spec: InstanceSpec, command: str, *, timeout_s: float, say=print) -> None:
    socket = runtime_dir() / spec.wayland_socket
    if not weston_pids(spec):
        socket.unlink(missing_ok=True)
        spec.dir.mkdir(parents=True, exist_ok=True)
        log = open(spec.dir / "weston.log", "ab")
        subprocess.Popen(weston_command(spec), env=_quiet_env(), stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        deadline = time.monotonic() + timeout_s
        while not socket.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f"headless weston for {spec.display} did not come up; see {spec.dir / 'weston.log'}")
            time.sleep(0.1)
    subprocess.Popen(command.split(), env=_quiet_env(WAYLAND_DISPLAY=spec.wayland_socket),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def _start_under_hyprland(config: FleetConfig, spec: InstanceSpec, command: str, *, say=print) -> None:
    """Your own session hosts the X server, on a hidden special workspace. Works, with one catch for long runs:
    when the last physical monitor disconnects (some sleep that way), Hyprland folds special workspaces into
    normal ones and tiles the windows -- and a rootful Xwayland resizes its root, and the game, to the tile."""
    launched = config.hidden and os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and hyprland_exec(
        command, display_rules(spec))
    if not launched:
        if config.hidden:
            say("  (not under Hyprland, or it refused the rule: the X servers open as ordinary windows)")
        env = {k: v for k, v in os.environ.items()}
        subprocess.Popen(command.split(), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


def stop_display(spec: InstanceSpec) -> None:
    for pid in server_pids(spec) + weston_pids(spec):
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
        if self.config.client == "plutonium":
            prepare_plutonium(self.config, self.spec, say=self.say)
            sync_plutonium_profile(self.spec.plutonium, source=game_config(self.config.root), say=self.say)
        command, cwd = game_command(self.config, self.spec)
        if self.config.offline:
            command = offline_wrapper() + command
        self.spec.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.spec.dir / "game.log", "ab")
        log.write(f"\n=== {time.ctime()} launching: {' '.join(command)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(command, cwd=cwd, env=game_env(self.config, self.spec),
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

        return find_window(self.config.window_title, self.spec.display)

    def capture(self, hud_regions=None, video_height: int | None = None):
        """A capture that follows the game window on this display across the game's own window churn; with
        `video_height`, also keeping a watchable copy of each grab (`ScreenCapture.last_video`)."""
        from zombiesai.demos.capture import FollowWindow, ScreenCapture
        from zombiesai.demos.frames import NO_BARS
        from zombiesai.demos.hud_crops import HUD_REGIONS

        from zombiesai.realgame.console import CONSOLE_REGION
        from zombiesai.realgame.scoreboard import SCOREBOARD_REGION

        regions = hud_regions if hud_regions is not None else {
            **{k: HUD_REGIONS[k] for k in ("points_ammo", "round")}, "console": CONSOLE_REGION,
            "scores": SCOREBOARD_REGION}
        # The crops' reference size is 1440p at half scale, i.e. 720p's own pixels.
        scale = min(1.0, 720.0 / self.spec.height)
        # The game runs fullscreen in an X server of its own size, so the picture fills the window: no bars to
        # find, and none to imagine in a dark first frame, which would crop the policy's view for good.
        return FollowWindow(lambda: ScreenCapture(window=self.config.window_title, display=self.spec.display,
                                                  hud_regions=regions, hud_scale=scale, bars=NO_BARS,
                                                  video_height=video_height))

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


# ------------------------------------------------------------------------------------------------ up and down


def wait_for_window(instance: Instance, timeout_s: float, *, say=print, should_stop=None, sleep=time.sleep,
                    clock=time.monotonic) -> bool:
    """Wait for the game's window and give it its X server's focus. There is no window manager to do it, and
    the game will not get past loading its renderer until its window is the focused one."""
    deadline = clock() + timeout_s
    while clock() < deadline:
        if should_stop is not None and should_stop():
            return False
        try:
            window = instance.window()
            sink = instance.sink()
            sink.focus(window.id)
            sink.close()
            return True
        except Exception:  # noqa: BLE001 -- not there yet
            sleep(1.0)
    say(f"  instance {instance.spec.index}: no game window after {timeout_s:.0f} s; carrying on")
    return False


def bring_up(instances: list[Instance], *, window_timeout_s: float = 120.0, stagger_s: float = 5.0, say=print,
             should_stop=None, sleep=time.sleep) -> int:
    """Up every one of `instances` (`Instance.up` is a no-op for what is already running), one game at a time:
    each new game gets its window focused, and the next waits `stagger_s` after it -- several games loading at
    once fight over the disk and the GPU. `should_stop()` is asked between games, so a long start-up can be
    abandoned (the owner sat down to play); a game already launching is left to finish. Returns how many games
    were launched."""
    launched = 0
    for k, instance in enumerate(instances):
        if should_stop is not None and should_stop():
            break
        was_running = instance.game_running()
        instance.up()
        if was_running:
            continue
        launched += 1
        wait_for_window(instance, window_timeout_s, say=say, should_stop=should_stop, sleep=sleep)
        if k < len(instances) - 1 and stagger_s > 0:
            deadline = time.monotonic() + stagger_s
            while time.monotonic() < deadline and not (should_stop is not None and should_stop()):
                sleep(min(0.5, stagger_s))
    return launched


def ensure_fleet(root: str | Path, n: int, *, say=print, should_stop=None, **up_kwargs) -> FleetConfig:
    """At least `n` games of the fleet at `root` running: what `scripts/instances.py up --n n` does, for a fleet
    worker joining a run (rl/fleet.py). The fleet's saved settings are kept, its size only ever grows (a bigger
    fleet.json costs nothing until its games start), and a PC without a fleet gets the defaults `up` would give
    it. Only the first `n` games are started; any beyond are left as they are."""
    try:
        config = load_fleet(root)
    except FileNotFoundError:
        config = FleetConfig(root=str(root), n=n)
        save_fleet(config)
    if config.n < n:
        config = replace(config, n=n)
        save_fleet(config)
    bring_up(fleet(config, say=say)[:n], say=say, should_stop=should_stop, **up_kwargs)
    return config


def take_down(root: str | Path, *, say=print) -> int:
    """Stop every game of the fleet at `root`, with its X server and sink: the GPU and RAM back to the PC's
    owner. Returns how many games were running. The prefixes and fleet.json stay, so `ensure_fleet` brings the
    same fleet back."""
    try:
        config = load_fleet(root)
    except FileNotFoundError:
        return 0
    running = 0
    for instance in fleet(config, say=say):
        running += instance.game_running()
        instance.down()
    return running


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
