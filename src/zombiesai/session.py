"""One command to start training on this PC, one to stop it and everything it started (scripts/session.py, `./zai`).

`start` brings the games up (realgame/instances.py), then starts scripts/train_rl.py as a process of its own, the
way the dashboard starts one (viz/supervise.py `start_run`): `-u` so its log is live, printing to runs/<run>.log,
in a session of its own so closing the terminal does not take it along. Either can stop what the other started.
It continues the newest real-game run unless told what to start from.

`stop` asks every trainer and fleet worker on the PC to stop as Ctrl-C asks -- the learner writes its checkpoint
first, then its actors finish (keys released, a finished game's film kept as the best if it is; rl/parallel_ppo.py
`StopRequest`) -- and waits for them. Then the viewers, the dashboard, the games with their X servers and sound
sinks, and, last, a sweep for anything of ours still running. It ends with what each stopped run saved. The site's
reporter (zombiesai-live.service) is left alone: it is meant to run always, and says the PC is idle.

What counts as ours is decided by `classify`, from each process's command line, environment and parent -- never
from a name alone, so the desktop's own Xwayland, an unrelated Python, or a BC training are never touched.
"""

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from zombiesai.viz.supervise import checkpoints, descendants, next_run_name, start_run, trees

FLEET_ROOT = "runs/instances"
STOP_TIMEOUT_S = 300.0  # the learner's own wait for its actors is 150 s (parallel_ppo.ACTOR_STOP_S)
GRACE_S = 5.0  # SIGTERM to SIGKILL, for what is left after the trainers are gone

_XSERVER = re.compile(r"^Xwayland (:\d+) -geometry \d+x\d+ .*-noreset")  # realgame/instances.py xwayland_command


@dataclass
class Proc:
    pid: int
    ppid: int
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def script(self) -> str | None:
        """The scripts/*.py it runs, if it is one of ours."""
        return next((Path(a).name for a in self.argv if re.search(r"(^|/)scripts/\w+\.py$", a)), None)

    def flag(self, name: str) -> str | None:
        return self.argv[self.argv.index(name) + 1] if name in self.argv[:-1] else None


def processes(proc: Path = Path("/proc")) -> dict[int, Proc]:
    """Every process this user can read, by pid."""
    found = {}
    for entry in proc.iterdir() if proc.is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            argv = [a.decode(errors="replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
            ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if not argv:  # a kernel thread, or a zombie
            continue
        p = Proc(int(entry.name), ppid, argv)
        try:
            p.cwd = os.readlink(entry / "cwd")
            raw = (entry / "environ").read_bytes().split(b"\0")
            p.env = dict(kv.decode(errors="replace").split("=", 1) for kv in raw if b"=" in kv)
        except OSError:
            pass  # someone else's: its command line is enough to leave it alone
        found[p.pid] = p
    return found


def classify(p: Proc, procs: dict[int, Proc], trees_: list[Path]) -> str | None:
    """What this process is to a training session -- "trainer", "worker", "actor", "game", "game display",
    "viewer", "dashboard" -- or None when it is not ours."""
    cmd = " ".join(p.argv)
    script = p.script
    if script == "train_rl.py":
        return "trainer"
    if script == "fleet_worker.py":
        return "worker"
    if script == "supervise.py" or "-m zombiesai.viz.supervise" in cmd:
        return "dashboard"
    if "-m zombiesai.realgame.viewer" in cmd:
        return "viewer"
    if _XSERVER.match(cmd) or (Path(p.argv[0]).name == "weston" and "--socket=zombiesai-" in cmd):
        return "game display"
    if p.env.get("PULSE_SINK", "").startswith("zombiesai_") or "/runs/instances/" in p.env.get("WINEPREFIX", ""):
        return "game"
    if "multiprocessing" in cmd and p.cwd and any(p.cwd == str(t) for t in trees_):
        # A trainer's actor, or one a killed trainer left behind (re-parented to init or the user's systemd).
        # Anything else's children -- a BC training's data loaders, a test run's -- have a Python parent of their
        # own and are left alone; so is a live trainer's resource tracker, which goes when its trainer does.
        parent = procs.get(p.ppid)
        orphan = parent is None or "python" not in Path(parent.argv[0]).name
        if orphan or (parent.script in ("train_rl.py", "fleet_worker.py") and "resource_tracker" not in cmd):
            return "actor"
    return None


def ours(trees_: list[Path], *, procs: dict[int, Proc] | None = None) -> dict[int, tuple[Proc, str]]:
    procs = processes() if procs is None else procs
    me = {os.getpid(), os.getppid()}
    found = {}
    for pid, p in procs.items():
        kind = classify(p, procs, trees_)
        if kind is not None and pid not in me:
            found[pid] = (p, kind)
    return found


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def signal_all(pids, sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate(pids: list[int], grace_s: float = GRACE_S) -> list[int]:
    """SIGTERM, then SIGKILL for whatever is still there after `grace_s`. Returns the ones that needed it."""
    signal_all(pids, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(alive(p) for p in pids):
        time.sleep(0.2)
    stubborn = [p for p in pids if alive(p)]
    signal_all(stubborn, signal.SIGKILL)
    return stubborn


# ------------------------------------------------------------------------------------------------ runs


def run_dir_of(p: Proc) -> Path | None:
    out = p.flag("--out")
    if out is None or p.cwd is None:
        return None
    return (Path(p.cwd) / out).resolve()


def default_init(tree: Path) -> str:
    """What `start` continues: the newest real-game PPO checkpoint, else the newest BC policy, else a fresh one."""
    found = checkpoints(tree)
    for c in found:
        config = c["config"] or {}
        if c["kind"] == "RL" and config.get("algorithm") == "ppo-finetune" and config.get("env") == "real-waw":
            return c["path"]
    return next((c["path"] for c in found if c["kind"] == "BC"), "fresh")


def default_games(root: str = FLEET_ROOT) -> int:
    """The fleet's size, or as many games as this PC carries beside the learner when it has no fleet yet."""
    from zombiesai.realgame.instances import load_fleet

    try:
        return load_fleet(root).n
    except FileNotFoundError:
        from zombiesai.rl.capacity import games_for, probe

        return games_for(probe(root), learner=True).games


def last_line(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            lines = f.read().decode(errors="replace").splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def saved(run: Path, *, now: float | None = None) -> str:
    """One line on what a run has on disk: its checkpoint and how fresh it is, its updates, games and best game."""
    now = time.time() if now is None else now
    checkpoint = run / "checkpoint.pt"
    if not checkpoint.exists():
        return f"{run.name}: no checkpoint.pt"
    parts = [f"{run.name}: checkpoint.pt written {max(0, now - checkpoint.stat().st_mtime):.0f} s ago"]
    row = last_line(run / "metrics.jsonl")
    if row:
        parts.append(f"{row.get('update', '?')} updates, {row.get('step', 0):,} steps")
    try:
        games = sum(1 for _ in open(run / "episodes.jsonl"))
        parts.append(f"{games} games")
    except OSError:
        pass
    try:
        best = json.loads((run / "best" / "best.json").read_text())
        parts.append(f"best game round {best.get('round', '?')}, {best.get('points', '?')} points (best/best.mp4)")
    except (OSError, ValueError):
        pass
    return ", ".join(parts)


# ------------------------------------------------------------------------------------------------ start


def start(repo: Path, *, init: str | None = None, games: int | None = None, name: str | None = None,
          extra: str = "", watch: bool = False, say=print) -> int:
    """Games up, then a trainer on them. Returns a process exit code."""
    from dataclasses import replace

    from zombiesai.realgame.instances import FleetConfig, bring_up, fleet, load_fleet, save_fleet

    all_trees = trees(repo)
    label = next((lbl for lbl, path in all_trees if path.resolve() == repo.resolve()), "main")
    running = [(p, k) for p, k in ours([t for _, t in all_trees]).values() if k == "trainer"]
    busy = [p for p, _ in running if p.flag("--env") != "synthetic"]
    if busy:
        run = run_dir_of(busy[0])
        say(f"already training: {run.name if run else 'a run'} (pid {busy[0].pid}). `./zai stop` first.")
        return 1
    init = init or default_init(repo)
    games = games or default_games()
    name = name or next_run_name(repo)
    if re.search(r"--env[= ]synthetic\b", extra):  # a rehearsal of the plumbing: no games to bring up
        say(f"starting runs/{name}: a rehearsal with {games} synthetic games, from {init}")
    else:
        try:
            config = load_fleet(FLEET_ROOT)
        except FileNotFoundError:
            config = FleetConfig(root=FLEET_ROOT, n=games)
        if config.n < games:
            config = replace(config, n=games)
        save_fleet(config)
        say(f"starting runs/{name}: {games} games, from {init}")
        say("bringing the games up (a minute or so each; ones already running are kept)")
        bring_up(fleet(config, say=say)[:games], say=say)
    (repo / "runs").mkdir(exist_ok=True)
    result = start_run(repo, label, init, games, name, extra, [], origin="./zai start")
    if "error" in result:
        say(f"the trainer did not start: {result['error']}")
        for line in result.get("tail", []):
            say(f"  | {line}")
        say("the games are still up; `./zai stop` takes them down")
        return 1
    say(f"training: runs/{name} (pid {result['pid']}), log runs/{name}.log")
    if watch:
        from zombiesai.realgame.viewer import running_screens, show

        show(running_screens(), run_dir=str(repo / "runs" / name), say=say)
    say("")
    say(f"  follow it:   tail -f runs/{name}.log      (or ./zai status)")
    say("  watch it:    uv run python scripts/view_instances.py")
    say("  stop it:     ./zai stop                 (saves everything, closes everything)")
    return 0


# ------------------------------------------------------------------------------------------------ stop


def stop(repo: Path, *, timeout_s: float = STOP_TIMEOUT_S, keep_games: bool = False, say=print) -> int:
    """Everything `start` started, and anything else of ours, stopped -- trainers first, gracefully."""
    from zombiesai.realgame.instances import take_down

    tree_paths = [t for _, t in trees(repo)]
    found = ours(tree_paths)
    trainers = [p for p, k in found.values() if k in ("trainer", "worker")]
    runs = [r for r in (run_dir_of(p) for p in trainers) if r is not None]
    below = {p.pid: descendants(p.pid) for p in trainers}
    if trainers:
        for p in trainers:
            run = run_dir_of(p)
            say(f"stopping {p.script} {run.name if run else ''} (pid {p.pid}): checkpoint first, then its actors")
        signal_all([p.pid for p in trainers], signal.SIGINT)
        start_t = time.monotonic()
        next_word = start_t + 15
        waiting = [p for p in trainers if alive(p.pid)]
        while waiting and time.monotonic() - start_t < timeout_s:
            time.sleep(0.5)
            waiting = [p for p in waiting if alive(p.pid)]
            if waiting and time.monotonic() > next_word:
                next_word += 15
                say(f"  still stopping ({time.monotonic() - start_t:.0f} s): actors finishing a game's film, "
                    f"or releasing their games")
        for p in waiting:
            say(f"  pid {p.pid} did not stop within {timeout_s:.0f} s: killing it. Its checkpoint is the last "
                f"one it wrote.")
            signal_all([*below[p.pid], p.pid], signal.SIGKILL)
        left = [pid for pids in below.values() for pid in pids if alive(pid)]
        if left:
            terminate(left)
        say(f"trainers stopped ({time.monotonic() - start_t:.0f} s)")
    else:
        say("no trainer running")

    if keep_games:
        say("games left running (--keep-games)")
    else:
        n = take_down(FLEET_ROOT, say=say)
        say(f"games down ({n} were running), with their X servers and sound sinks")

    # The sweep: viewers and the dashboard, and whatever outlived all of the above (games of another checkout's
    # fleet, a game whose launcher left its session, an actor orphaned by a crash).
    leftovers = {pid: (p, k) for pid, (p, k) in ours(tree_paths).items()
                 if not (keep_games and k in ("game", "game display"))}
    if leftovers:
        kinds = sorted({k for _, k in leftovers.values()})
        killed = terminate(list(leftovers))
        say(f"closed {len(leftovers)} more: {', '.join(kinds)}" + (f" ({len(killed)} needed SIGKILL)" if killed else ""))
    if not keep_games:
        sinks = remove_sinks()
        if sinks:
            say(f"removed {sinks} leftover sound sinks")

    still = ours(tree_paths)
    if keep_games:
        still = {pid: v for pid, v in still.items() if v[1] in ("trainer", "worker", "actor")}
    for p, k in still.values():
        say(f"  STILL RUNNING: {k} pid {p.pid}: {' '.join(p.argv)[:120]}")
    if runs:
        say("saved:")
        for run in runs:
            say(f"  {saved(run)}")
    say("all stopped" if not still else "some processes would not stop (above)")
    return 0 if not still else 1


def remove_sinks(*, run=subprocess.run) -> int:
    """Unload any `zombiesai_*` null sink still loaded (realgame/instances.py ensure_sink makes them)."""
    try:
        listed = run(["pactl", "list", "short", "modules"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    removed = 0
    for line in listed.stdout.splitlines():
        if "module-null-sink" in line and "sink_name=zombiesai_" in line:
            run(["pactl", "unload-module", line.split("\t")[0]], capture_output=True, timeout=5)
            removed += 1
    return removed


# ------------------------------------------------------------------------------------------------ status


def status(repo: Path, *, say=print) -> int:
    from zombiesai.realgame.instances import fleet, load_fleet

    tree_paths = [t for _, t in trees(repo)]
    found = ours(tree_paths)
    trainers = [p for p, k in found.values() if k in ("trainer", "worker")]
    if not trainers:
        say("training: nothing running")
    for p in trainers:
        run = run_dir_of(p)
        row = last_line(run / "metrics.jsonl") if run else None
        line = f"training: {p.script} {run.name if run else ''} (pid {p.pid})"
        if row:
            line += (f" -- update {row.get('update')}, step {row.get('step', 0):,}, {row.get('sps', '?')} steps/s"
                     + (f", round {row['round_reached_mean']:.2f}" if "round_reached_mean" in row else "")
                     + (f", return {row['return_mean']:.1f}" if "return_mean" in row else ""))
        say(line)
        if run is not None:
            say(f"  log: {run.parent / (run.name + '.log')}")
    try:
        config = load_fleet(FLEET_ROOT)
        instances = fleet(config, say=lambda m: None)
        up = sum(i.game_running() for i in instances)
        say(f"games: {up} of {config.n} running")
    except FileNotFoundError:
        say("games: no fleet yet (`./zai start` makes one)")
    counts: dict[str, int] = {}
    for _, k in found.values():
        counts[k] = counts.get(k, 0) + 1
    other = {k: v for k, v in counts.items() if k in ("viewer", "dashboard", "actor", "game", "game display")}
    if other:
        say("processes: " + ", ".join(f"{v} {k}" for k, v in sorted(other.items())))
    return 0
