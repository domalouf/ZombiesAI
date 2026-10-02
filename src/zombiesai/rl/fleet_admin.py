"""Get every PC that plays for the learner ready for a run, from the learner, over SSH (`scripts/fleet.py`).

The learner turns a worker away at hello for two things a person has to fix by hand on that PC: a different
commit, or different game settings (rl/fleet.py). Both are the training PC's to hand out, so `prep` does it for
each machine in turn, in the order a person would:

    reach     ssh in; the repo is there and has nothing uncommitted (never touched if it has)
    commit    git fetch, and check out the learner's commit -- detached, so no branch of theirs moves
    sync      uv sync, for whatever that commit's lock file says
    settings  the learner's config.cfg installed in the fleet's root, which the games play with from then on
              (realgame/instances.py: game_config) -- the PC's own Steam profile is left alone. Only Plutonium
              games read it: a fleet on the steam client is warned about and left as it is, for `check` to judge
    games     instances.py up; restarted if their settings just changed (a game reads config.cfg at launch)
    worker    zombiesai-worker restarted if it runs and the code or settings changed under it
    check     fleet_worker.py --describe, judged as the learner's hello judges it

`check` alone is read-only. Nothing here needs the fleet's token: it is the same SSH access a person would use.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from zombiesai.rl.fleet import describe, settings_differences


@dataclass
class Target:
    host: str  # anything ssh takes: an alias from ~/.ssh/config, user@host, host
    actors: int = 4
    repo: str = "~/Projects/ZombiesAI"  # where the checkout is on that PC (the systemd units assume this)
    fleet: str = "runs/instances"  # the fleet's root, relative to the repo


@dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""
    warn: bool = False  # ok, but not done: a person should read why


@dataclass
class Report:
    host: str
    steps: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def add(self, name: str, ok: bool, detail: str = "", *, warn: bool = False) -> bool:
        self.steps.append(Step(name, ok, detail, warn))
        return ok


@dataclass
class Learner:
    """What the PCs have to match: this machine's commit, and the config.cfg its own games play with."""

    sha: str
    dirty: bool
    spec_version: str
    settings: dict | None
    config: bytes | None

    @classmethod
    def here(cls, fleet_root: str = "runs/instances") -> "Learner":
        from zombiesai.realgame.instances import game_config

        me = describe(fleet_root)
        path = game_config(fleet_root)
        return cls(sha=me["sha"], dirty=bool(me["dirty"]), spec_version=me["spec_version"],
                   settings=me["settings"], config=path.read_bytes() if path else None)


def _cd(repo: str) -> str:
    """`cd` to a path that may start with ~/, which must stay unquoted for the remote shell to expand it."""
    if repo == "~" or repo.startswith("~/"):
        rest = repo[2:]
        return "cd ~" + (f"/{shlex.quote(rest)}" if rest else "")
    return f"cd {shlex.quote(repo)}"


def _last_line(text: str) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class Remote:
    """One PC over SSH: a login bash (so ~/.local/bin and uv are on PATH) in the repo, one connection reused."""

    def __init__(self, target: Target, *, run=subprocess.run, ssh: str = "ssh", control_dir: str = "/tmp"):
        self.target, self.run = target, run
        control = str(Path(control_dir) / "zombiesai-fleet-%C")
        self.ssh = [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ControlMaster=auto",
                    "-o", f"ControlPath={control}", "-o", "ControlPersist=120"]

    def __call__(self, script: str, *, data: bytes | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
        command = f"{_cd(self.target.repo)} && {script}"
        argv = [*self.ssh, self.target.host, "bash -lc " + shlex.quote(command)]
        try:
            return self.run(argv, input=data, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(argv, 124, b"", f"timed out after {timeout:.0f}s".encode())
        except FileNotFoundError:
            return subprocess.CompletedProcess(argv, 255, b"", f"no {self.ssh[0]} on this machine".encode())
        except OSError as error:
            return subprocess.CompletedProcess(argv, 255, b"", str(error).encode())


def _text(b) -> str:
    return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")


def check(target: Target, learner: Learner, remote: Remote, report: Report, *, games: bool = True) -> bool:
    """The PC as the learner's hello will see it: same commit and spec, same settings, its games running."""
    q = shlex.quote
    result = remote(f"uv run --no-sync python scripts/fleet_worker.py --describe --fleet {q(target.fleet)}", timeout=120)
    if result.returncode != 0:
        return report.add("check", False, f"fleet_worker.py --describe failed: {_last_line(_text(result.stderr))}")
    try:
        them = json.loads(_last_line(_text(result.stdout)))
    except ValueError:
        return report.add("check", False, "fleet_worker.py --describe printed no JSON")
    problems = []
    if them.get("sha") != learner.sha:
        problems.append(f"on commit {str(them.get('sha'))[:8]}, the learner on {learner.sha[:8]}")
    elif them.get("dirty"):
        problems.append("uncommitted changes there: the same commit, but not the same code")
    if them.get("spec_version") != learner.spec_version:
        problems.append(f"spec {them.get('spec_version')}, the learner's {learner.spec_version}")
    if learner.settings is not None:
        diffs = settings_differences(learner.settings, them.get("settings"))
        if diffs:
            problems.append("game settings differ: " + "; ".join(diffs[:4]) + (" ..." if len(diffs) > 4 else ""))
    running = int(them.get("games_running") or 0)
    if games and running < target.actors:
        problems.append(f"{running} of {target.actors} games running")
    if problems:
        return report.add("check", False, "; ".join(problems))
    return report.add("check", True, f"on {learner.sha[:8]}, settings match"
                      + (f", {running} of {target.actors} games running" if games else ""))


def prep(target: Target, learner: Learner, remote: Remote, *, games: bool = True, say=print) -> Report:
    """Bring one PC to the learner's commit and settings, start its games, and check it. Stops at the first
    step that fails: every later step depends on it."""
    report, q = Report(target.host), shlex.quote
    sha, fleet = learner.sha, target.fleet

    result = remote("git rev-parse HEAD && git status --porcelain", timeout=60)
    if result.returncode != 0:
        report.add("reach", False, _last_line(_text(result.stderr)) or f"ssh exited {result.returncode}")
        return report
    lines = _text(result.stdout).strip().splitlines()
    before, dirty = (lines[0].strip() if lines else ""), any(ln.strip() for ln in lines[1:])
    if dirty:
        report.add("reach", False, "uncommitted changes in the checkout there; not touching it")
        return report
    report.add("reach", True, f"on {before[:8]}")

    if before == sha:
        report.add("commit", True, "already on the learner's commit")
    else:
        say(f"[{target.host}] checking out {sha[:8]}")
        result = remote(f"git fetch --quiet origin || exit 10; git cat-file -e {q(sha + '^{commit}')} 2>/dev/null "
                        f"|| exit 11; git checkout --quiet --detach {q(sha)}", timeout=300)
        if result.returncode != 0:
            why = {10: "git fetch failed: " + _last_line(_text(result.stderr)),
                   11: f"the learner's commit {sha[:8]} is not on origin: push it from the training PC first"}
            report.add("commit", False, why.get(result.returncode, _last_line(_text(result.stderr))))
            return report
        report.add("commit", True, f"{before[:8]} -> {sha[:8]} (detached; `git switch <branch>` to go back)")
    code_changed = before != sha

    say(f"[{target.host}] uv sync")
    result = remote("uv sync --quiet", timeout=1200)
    if not report.add("sync", result.returncode == 0,
                      "" if result.returncode == 0 else _last_line(_text(result.stderr))):
        return report

    settings_changed = False
    if learner.config is None:
        report.add("settings", True, "the training PC has no config.cfg to hand out; its own Steam one stays")
    elif _fleet_client(remote, fleet) == "steam":
        # Steam's CoDWaW.exe reads the profile inside each instance's copy of the prefix, never the fleet root's
        # config.cfg (instances.game_config), so installing one would change nothing those games play with.
        # Not a failure by itself: if that profile already matches, the PC is fine, and `check` says which.
        report.add("settings", True, "NOT installed: this PC's fleet runs the steam client, whose games read "
                   "the profile in their own prefixes, not an installed config.cfg -- `instances.py up --client "
                   "plutonium` there to use it, or edit that profile in its instances' prefixes by hand", warn=True)
    else:
        want = hashlib.sha256(learner.config).hexdigest()
        path = f"{fleet}/config.cfg"
        result = remote(f"mkdir -p {q(fleet)} && (sha256sum {q(path)} 2>/dev/null || true)", timeout=60)
        have = (_text(result.stdout).split() or [""])[0]
        if have == want:
            report.add("settings", True, "the learner's config.cfg is already installed")
        else:
            result = remote(f"cat > {q(path + '.part')} && mv {q(path + '.part')} {q(path)}", data=learner.config,
                            timeout=60)
            if not report.add("settings", result.returncode == 0,
                              "installed the learner's config.cfg" if result.returncode == 0
                              else _last_line(_text(result.stderr))):
                return report
            settings_changed = True

    if games:
        say(f"[{target.host}] starting {target.actors} games" + (" (restarting: new settings)" if settings_changed else ""))
        script = f"uv run --no-sync python scripts/instances.py up --root {q(fleet)} --n {target.actors}"
        if settings_changed:  # a running game read its config.cfg at launch; `up` leaves running games alone
            script = (f"if [ -f {q(fleet + '/fleet.json')} ]; then uv run --no-sync python scripts/instances.py "
                      f"restart --root {q(fleet)} || exit 1; fi; " + script)
        result = remote(script, timeout=1800)
        if not report.add("games", result.returncode == 0,
                          "up" + (", restarted with the new settings" if settings_changed else "")
                          if result.returncode == 0 else _last_line(_text(result.stderr) or _text(result.stdout))):
            return report

    if code_changed or settings_changed:
        result = remote("if systemctl --user is-active --quiet zombiesai-worker; then "
                        "systemctl --user restart zombiesai-worker && echo restarted; else echo idle; fi", timeout=60)
        out = _last_line(_text(result.stdout))
        report.add("worker", result.returncode == 0,
                   {"restarted": "zombiesai-worker restarted on the new code and settings",
                    "idle": "no zombiesai-worker running; start fleet_worker.py there"}.get(out, out)
                   if result.returncode == 0 else _last_line(_text(result.stderr)))

    check(target, learner, remote, report, games=games)
    return report


def _fleet_client(remote: Remote, fleet: str) -> str | None:
    """The client the PC's fleet.json names, or None when it has none yet (`instances.py up` makes a Plutonium
    one by default) or it cannot be read."""
    result = remote(f"cat {shlex.quote(fleet + '/fleet.json')} 2>/dev/null || true", timeout=60)
    try:
        saved = json.loads(_text(result.stdout) or "{}")
    except ValueError:
        return None
    return saved.get("client") if isinstance(saved, dict) else None


def run_all(targets: list[Target], learner: Learner, *, mode: str = "prep", games: bool = True, run=subprocess.run,
            ssh: str = "ssh", say=print) -> list[Report]:
    """Every PC at once: a game launch takes minutes, and the PCs do not wait on each other."""
    from concurrent.futures import ThreadPoolExecutor

    def one(target: Target) -> Report:
        remote = Remote(target, run=run, ssh=ssh)
        if mode == "check":
            report = Report(target.host)
            check(target, learner, remote, report, games=games)
            return report
        return prep(target, learner, remote, games=games, say=say)

    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        return list(pool.map(one, targets))


def parse_targets(specs: list[str], *, actors: int, repo: str, fleet: str) -> list[Target]:
    """"host" or "host=N" (N games on that PC; --actors otherwise)."""
    out = []
    for spec in specs:
        host, sep, n = spec.partition("=")
        if not host:
            raise ValueError(f"no host in {spec!r}")
        out.append(Target(host=host, actors=int(n) if sep else actors, repo=repo, fleet=fleet))
    return out


def format_report(report: Report) -> str:
    lines = [f"{report.host}: {'ready' if report.ok else 'NOT READY'}"]
    for s in report.steps:
        lines.append(f"  {('WARN' if s.warn else 'ok  ') if s.ok else 'FAIL'} {s.name:<8} {s.detail}")
    return "\n".join(lines)
