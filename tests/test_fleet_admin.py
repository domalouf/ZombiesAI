import hashlib
import json
import subprocess

import pytest

from zombiesai.rl import fleet_admin
from zombiesai.rl.fleet_admin import Learner, Remote, Target, check, parse_targets, prep

SHA = "c3d4" + "0" * 36
OLD = "a1b2" + "0" * 36
SETTINGS = {"dvars": {"sensitivity": "2"}, "binds": {"mouse1": "+attack"}}
CONFIG = b'seta sensitivity "2"\nbind MOUSE1 "+attack"\n'


def learner(**kw) -> Learner:
    return Learner(**{"sha": SHA, "dirty": False, "spec_version": "spec1", "settings": SETTINGS, "config": CONFIG, **kw})


class FakePC:
    """A PC over ssh: answers each remote script by what it asks for, and remembers what it was told."""

    def __init__(self, head=OLD, dirty=False, config=None, fetch_ok=True, has_commit=True, worker_active=True,
                 games=4, describe=None, client=None):
        self.head, self.dirty, self.config, self.client = head, dirty, config, client
        self.fetch_ok, self.has_commit, self.worker_active, self.games = fetch_ok, has_commit, worker_active, games
        self.describe = describe
        self.scripts: list[str] = []

    def __call__(self, argv, input=None, capture_output=True, timeout=None):
        script = argv[-1]
        assert script.startswith("bash -lc ") and argv[-2] == "rig2.lan" and "BatchMode=yes" in argv
        self.scripts.append(script)
        out, code, err = "", 0, ""
        if "git rev-parse HEAD" in script:
            out = self.head + "\n" + (" M src/x.py\n" if self.dirty else "")
        elif "git fetch" in script:
            if not self.fetch_ok:
                code, err = 10, "fatal: unable to access origin"
            elif not self.has_commit:
                code = 11
            else:
                self.head = SHA
        elif "sha256sum" in script:
            out = f"{hashlib.sha256(self.config).hexdigest()}  runs/instances/config.cfg\n" if self.config else ""
        elif "cat " in script and "fleet.json" in script and "cat >" not in script:
            out = json.dumps({"n": 4, "client": self.client}) if self.client else ""
        elif "cat >" in script:
            self.config = input
        elif "systemctl" in script:
            out = "restarted\n" if self.worker_active else "idle\n"
        elif "--describe" in script:
            out = json.dumps(self.describe or {
                "sha": self.head, "dirty": False, "spec_version": "spec1",
                "settings": SETTINGS if self.config == CONFIG else {"dvars": {"sensitivity": "9"}, "binds": {}},
                "games_running": self.games}) + "\n"
        return subprocess.CompletedProcess(argv, code, out.encode(), err.encode())

    def ran(self, needle: str) -> list[str]:
        return [s for s in self.scripts if needle in s]


def run_prep(pc: FakePC, games=True, who=None):
    target = Target("rig2.lan", actors=4)
    return prep(target, who or learner(), Remote(target, run=pc), games=games, say=lambda m: None)


def test_a_pc_behind_is_brought_to_the_learners_commit_settings_and_games():
    pc = FakePC()
    report = run_prep(pc)
    assert report.ok, [(s.name, s.detail) for s in report.steps]
    assert [s.name for s in report.steps] == ["reach", "commit", "sync", "settings", "games", "worker", "check"]
    assert pc.head == SHA and pc.config == CONFIG
    assert pc.ran(f"git checkout --quiet --detach {SHA}")  # no branch of theirs moves
    assert pc.ran("instances.py restart") and pc.ran("instances.py up --root runs/instances --n 4")
    assert all(s.startswith("bash -lc 'cd ~/Projects/ZombiesAI && ") for s in pc.scripts)  # ~ left to the shell


def test_a_pc_already_ready_is_only_checked_and_its_games_are_not_restarted():
    pc = FakePC(head=SHA, config=CONFIG)
    report = run_prep(pc)
    assert report.ok and [s.name for s in report.steps] == ["reach", "commit", "sync", "settings", "games", "check"]
    assert not pc.ran("git fetch") and not pc.ran("cat >") and not pc.ran("restart")


def test_uncommitted_work_on_a_pc_is_never_touched():
    pc = FakePC(dirty=True)
    report = run_prep(pc)
    assert not report.ok and report.steps[-1].name == "reach" and "uncommitted" in report.steps[-1].detail
    assert len(pc.scripts) == 1


def test_a_commit_the_pc_cannot_fetch_says_to_push_it():
    pc = FakePC(has_commit=False)
    report = run_prep(pc)
    assert not report.ok and report.steps[-1].name == "commit" and "push it" in report.steps[-1].detail
    assert not pc.ran("uv sync")


def test_check_reports_what_the_learners_hello_would_refuse():
    pc = FakePC(describe={"sha": OLD, "dirty": False, "spec_version": "spec1",
                          "settings": {"dvars": {"sensitivity": "9"}, "binds": {"mouse1": "+attack"}},
                          "games_running": 2})
    target = Target("rig2.lan", actors=4)
    report = fleet_admin.Report("rig2.lan")
    assert not check(target, learner(), Remote(target, run=pc), report)
    detail = report.steps[0].detail
    assert "on commit a1b2" in detail and "sensitivity" in detail and "2 of 4 games" in detail
    assert len(pc.scripts) == 1  # read-only


def test_a_training_pc_without_a_config_leaves_each_pcs_own():
    pc = FakePC(head=SHA, config=b"theirs")
    report = run_prep(pc, games=False, who=learner(config=None, settings=None))
    assert report.ok and pc.config == b"theirs" and not pc.ran("instances.py")


def test_hosts_take_their_own_game_counts():
    targets = parse_targets(["rig2.lan", "me@rig3=2"], actors=4, repo="~/z", fleet="runs/instances")
    assert [(t.host, t.actors) for t in targets] == [("rig2.lan", 4), ("me@rig3", 2)]
    with pytest.raises(ValueError):
        parse_targets(["=3"], actors=4, repo="~", fleet="f")


def test_a_steam_client_fleet_is_warned_about_not_given_a_config_nothing_reads():
    pc = FakePC(head=SHA, config=b"theirs", client="steam")
    report = run_prep(pc)
    settings = next(s for s in report.steps if s.name == "settings")
    assert settings.ok and settings.warn and "steam client" in settings.detail
    assert pc.config == b"theirs" and not pc.ran("cat >") and not pc.ran("instances.py restart")
    # What its games do play with is for check to judge: here, not the learner's settings.
    assert not report.ok and report.steps[-1].name == "check" and "settings differ" in report.steps[-1].detail
    assert "WARN settings" in fleet_admin.format_report(report)


def test_a_plutonium_client_fleet_still_gets_the_learners_config():
    pc = FakePC(head=SHA, config=b"theirs", client="plutonium")
    report = run_prep(pc)
    assert report.ok and pc.config == CONFIG and pc.ran("instances.py restart")
    assert not any(s.warn for s in report.steps)
