"""How much of this PC the fleet may use: how many games fit (`--actors auto`), and whether its owner wants it
back right now (`--yield-to-games`).

The PCs that play for the learner are gaming PCs people use. Two things follow, and both live here:

* **Right-sizing.** A worker should not need to be told how many games its PC can carry. `games_for()` is a
  pure function from a `Snapshot` of the hardware -- logical CPUs, MemAvailable, the GPU's VRAM, how many of our
  games are already up -- to a number of games, using the per-game costs below and leaving the desktop its
  share. `probe()` takes the snapshot of the real machine. The learner's own `train_rl.py --actors auto` can use
  the same function with `learner=True`, which also sets aside what the PPO update needs.
* **Yielding.** While the owner is playing something else -- a Steam game, which Steam starts under its
  `reaper SteamLaunch AppId=<id>` process, or a Windows game through umu-run (Heroic, Lutris) -- or has asked
  for a pause by creating PAUSE_FILE, the worker leaves the run (`owner_busy()`, `YieldWatch`). Our own games
  are umu-run too, and carry Steam's app id in their environment (umu sets SteamAppId from GAMEID), so they are
  told apart by what they always have and an owner's game never does: a Wine prefix inside the fleet's root.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

GB = 2**30

# ---- What one game slot costs. Estimates from what was seen on the learner PC (docs/rl.md), not a benchmark:
# watch `bad_step_frac` per machine (the dashboard's "Each machine") and lower MAX_GAMES or raise these if a PC
# sized by them runs late.
# The game under Proton, its Xwayland and headless Weston: docs/rl.md has "roughly 1-1.5 GB of RAM" per game.
GAME_RAM_GB = 1.3
# One actor process: Python, torch on the CPU and the policy -- 0.67 GB measured for the BC net with hearing --
# plus its capture and HUD buffers.
ACTOR_RAM_GB = 0.7
# The game ~0.4 of a core at com_maxfps 30 (0.6 at 60 fps under Weston), its actor ~0.4 (a step's grab, resize
# and HUD read take ~25 ms single-threaded at 15 Hz). One logical CPU each, SMT siblings counted as CPUs.
CPU_PER_GAME = 1.0
# WaW's textures and a 2560x1440 swapchain under DXVK. The fleet runs at 1440p because the HUD reader needs it.
VRAM_PER_GAME_GB = 1.0

# ---- What the PC keeps for itself: the desktop, a browser, a chat client -- the owner is still using it.
DESKTOP_RAM_GB = 4.0
DESKTOP_CPUS = 2.0  # also covers the worker's own forwarder, and the X servers' share
DESKTOP_VRAM_GB = 1.5
# ---- And, on the learner PC, what the learner itself takes: the batch in RAM, the PPO update on the GPU.
LEARNER_RAM_GB = 4.0
LEARNER_CPUS = 2.0
LEARNER_VRAM_GB = 2.5

# More games than this on one PC has never been tried; past it the GPU's frame pacing, not memory, decides.
MAX_GAMES = 8


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_total_gb: float
    vram_free_gb: float


@dataclass(frozen=True)
class Snapshot:
    cpus: int  # logical CPUs this process may run on
    mem_available_gb: float  # /proc/meminfo's MemAvailable: what can be had without swapping
    gpus: tuple[Gpu, ...] = ()  # the games render on the first
    games_running: int = 0  # our games already up: their RAM and VRAM are already out of the numbers above
    mem_total_gb: float | None = None


@dataclass(frozen=True)
class Verdict:
    games: int
    limits: dict = field(default_factory=dict)  # what each resource alone would allow
    why: str = ""

    def __str__(self) -> str:
        return f"{self.games} games ({self.why})"


def games_for(snap: Snapshot, *, learner: bool = False, cap: int = MAX_GAMES) -> Verdict:
    """How many games this PC can run with an actor each, after the desktop's share (and the learner's, on the
    learner PC). Every resource gives its own limit and the smallest wins.

    A game already running has already taken its RAM and VRAM -- they are missing from MemAvailable and the free
    VRAM -- so only the games still to start are charged for those; every game's actor is charged, since the
    actors start with the run. CPUs are charged in full: a running game's CPU time is not "used up" the way its
    memory is. With no GPU visible (none, or not NVIDIA/amdgpu), VRAM sets no limit and the verdict says so."""
    keep_ram = DESKTOP_RAM_GB + (LEARNER_RAM_GB if learner else 0.0)
    keep_cpus = DESKTOP_CPUS + (LEARNER_CPUS if learner else 0.0)
    keep_vram = DESKTOP_VRAM_GB + (LEARNER_VRAM_GB if learner else 0.0)
    running = max(int(snap.games_running), 0)
    limits = {"cap": cap, "cpu": int((snap.cpus - keep_cpus) // CPU_PER_GAME)}

    def fits(n: int) -> bool:
        return (max(n - running, 0) * GAME_RAM_GB + n * ACTOR_RAM_GB) <= snap.mem_available_gb - keep_ram

    ram = 0
    while ram < 4 * cap and fits(ram + 1):
        ram += 1
    limits["ram"] = ram
    gpu = snap.gpus[0] if snap.gpus else None
    if gpu is not None:
        limits["vram"] = running + int(max(gpu.vram_free_gb - keep_vram, 0.0) // VRAM_PER_GAME_GB)
    games = max(0, min(limits.values()))
    binding = min(limits, key=lambda k: (limits[k], k != "cap"))
    detail = {"cap": f"capped at {cap}",
              "cpu": f"{snap.cpus} CPUs, {keep_cpus:g} kept",
              "ram": f"{snap.mem_available_gb:.1f} GB RAM available, {keep_ram:g} kept",
              "vram": f"{gpu.vram_free_gb:.1f} GB VRAM free on the {gpu.name}, {keep_vram:g} kept" if gpu else ""}
    why = (f"{detail[binding]}; " if games == limits[binding] else "") + ", ".join(
        f"{k} {v}" for k, v in limits.items() if k != "cap")
    if gpu is None:
        why += "; no GPU seen, so VRAM was not counted"
    if running:
        why += f"; {running} already running"
    return Verdict(games=games, limits=limits, why=why)


# ------------------------------------------------------------------------------------------------ probing


def parse_meminfo(text: str) -> dict[str, float]:
    """/proc/meminfo's kB lines as GB."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * 1024 / GB
    return out


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Rows of `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits` (MiB)."""
    out = []
    for line in text.strip().splitlines():
        rest = line.rsplit(",", 2)  # from the right: the name is the one field that could hold a comma
        try:
            gpu_name, total, free = rest[0].strip(), float(rest[1]), float(rest[2])
        except (ValueError, IndexError):
            continue
        out.append(Gpu(gpu_name, total / 1024, free / 1024))
    return out


def amd_gpus(sys_root: str | Path = "/sys") -> list[Gpu]:
    """amdgpu's VRAM counters in sysfs, for a PC without nvidia-smi."""
    out = []
    for card in sorted(Path(sys_root).glob("class/drm/card[0-9]*/device")):
        try:
            total = int((card / "mem_info_vram_total").read_text())
            used = int((card / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        out.append(Gpu(f"AMD GPU ({card.parent.name})", total / GB, (total - used) / GB))
    return out


def probe_gpus(*, run=subprocess.run, sys_root: str | Path = "/sys") -> list[Gpu]:
    try:
        result = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            found = parse_nvidia_smi(result.stdout)
            if found:
                return found
    except (OSError, subprocess.TimeoutExpired):
        pass
    return amd_gpus(sys_root)


def games_running(fleet_root: str | Path) -> int:
    """How many of the fleet's games are up, by their pid files (0 without a fleet)."""
    from zombiesai.realgame.instances import fleet, load_fleet

    try:
        config = load_fleet(fleet_root)
    except (FileNotFoundError, ValueError, TypeError):
        return 0
    return sum(i.game_running() for i in fleet(config, say=lambda m: None))


def probe(fleet_root: str | Path = "runs/instances", *, proc_root: str | Path = "/proc", run=subprocess.run,
          sys_root: str | Path = "/sys") -> Snapshot:
    """This machine, now. Cheap enough to take before every run: one nvidia-smi call."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    try:
        mem = parse_meminfo((Path(proc_root) / "meminfo").read_text())
    except OSError:
        mem = {}
    return Snapshot(cpus=cpus, mem_available_gb=mem.get("MemAvailable", 0.0), mem_total_gb=mem.get("MemTotal"),
                    gpus=tuple(probe_gpus(run=run, sys_root=sys_root)), games_running=games_running(fleet_root))


# ------------------------------------------------------------------------------------------------ the owner

PAUSE_FILE = Path.home() / ".config" / "zombiesai" / "pause"
_APP_ID = re.compile(r"AppId=(\d+)")


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _environ(proc: Path) -> dict[str, str] | None:
    """A process's environment, or None when it is not ours to read (another user's)."""
    raw = _read(proc / "environ")
    if raw is None:
        return None
    out = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            out[key.decode(errors="replace")] = value.decode(errors="replace")
    return out


def _ours(env: dict[str, str] | None, fleet_root: Path | None) -> bool:
    """One of the fleet's own games: its Wine prefix is an instance's copy, inside the fleet's root
    (realgame/instances.py: game_env). Nothing else on the PC uses a prefix there."""
    if env is None or fleet_root is None:
        return False
    prefix = env.get("WINEPREFIX")
    if not prefix:
        return False
    try:
        return Path(prefix).resolve().is_relative_to(fleet_root)
    except (OSError, ValueError):
        return False


def owner_games(proc_root: str | Path = "/proc", *, fleet_root: str | Path | None = None) -> list[str]:
    """What the owner is playing, as words for a log: one entry per game found. A /proc scan of every process's
    command line -- a few hundred small reads -- and the environment only of the few that look like games.

    * Steam starts every game, native or Proton, under `reaper SteamLaunch AppId=<id> -- ...`. Our games never
      go through Steam's reaper; one that somehow did would still be ours by its prefix.
    * umu-run starts Windows games outside Steam (Heroic, Lutris, ...): an owner's game unless its prefix is in
      the fleet's root, which is how every one of ours runs."""
    root = Path(fleet_root).resolve() if fleet_root is not None else None
    found = []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():
            continue
        raw = _read(proc / "cmdline")
        if not raw:
            continue
        argv = [a.decode(errors="replace") for a in raw.rstrip(b"\0").split(b"\0")]
        program = os.path.basename(argv[0])
        if program == "reaper" and "SteamLaunch" in argv:
            if _ours(_environ(proc), root):
                continue
            app = next((m[1] for a in argv if (m := _APP_ID.fullmatch(a))), "?")
            found.append(f"a Steam game (app {app})")
        elif program == "umu-run" or (len(argv) > 1 and program.startswith("python")
                                      and os.path.basename(argv[1]) == "umu-run"):
            env = _environ(proc)
            if _ours(env, root):
                continue
            found.append(f"a game through umu-run ({os.path.basename(argv[-1] if len(argv) > 1 else argv[0])[:60]})")
    return found


def owner_busy(proc_root: str | Path = "/proc", *, fleet_root: str | Path | None = None,
               pause_file: str | Path | None = PAUSE_FILE) -> str | None:
    """Why the owner wants the PC now, or None."""
    if pause_file is not None and Path(pause_file).exists():
        return f"paused by {pause_file}"
    games = owner_games(proc_root, fleet_root=fleet_root)
    if games:
        return "the owner is playing " + ", ".join(sorted(set(games)))
    return None


class YieldWatch:
    """`owner_busy()` with a little hysteresis: busy as soon as one scan says so -- the owner comes first --
    and free only once every scan for `resume_after_s` has said so, so a launcher that restarts its game, or a
    loading screen between two games, does not have the fleet's games starting up under it."""

    def __init__(self, *, fleet_root: str | Path | None, proc_root: str | Path = "/proc",
                 pause_file: str | Path | None = PAUSE_FILE, resume_after_s: float = 60.0, clock=time.monotonic,
                 scan=None):
        self.resume_after_s, self.clock = resume_after_s, clock
        self.scan = scan or (lambda: owner_busy(proc_root, fleet_root=fleet_root, pause_file=pause_file))
        self.reason: str | None = None  # why we are yielding, while we are
        self._free_since: float | None = None

    def check(self) -> str | None:
        """Why the fleet should stay out of this PC now, or None when it may play."""
        now, busy = self.clock(), self.scan()
        if busy:
            self.reason, self._free_since = busy, None
            return busy
        if self.reason is None:
            return None
        if self._free_since is None:
            self._free_since = now
        if now - self._free_since >= self.resume_after_s:
            self.reason = self._free_since = None
            return None
        return self.reason

    def resuming_in(self) -> float | None:
        if self.reason is None or self._free_since is None:
            return None
        return max(0.0, self.resume_after_s - (self.clock() - self._free_since))
