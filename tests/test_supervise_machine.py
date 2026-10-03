import json
import subprocess
import time
from pathlib import Path

import pytest

from zombiesai.viz import supervise
from zombiesai.viz.supervise import Supervisor, gpu_status, hwmon_temps, level

# Stands in for scripts/instances.py: says what it was asked and takes a moment, as bringing games up does.
FAKE_INSTANCES = """\
import sys, time
print("instances.py", *sys.argv[1:], flush=True)
time.sleep(float(__import__("os").environ.get("FAKE_SECONDS", "1")))
print("done", flush=True)
"""


def completed(stdout: str):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_a_reading_is_marked_by_this_pcs_limits():
    assert level("cpu_temp", 60) == "good" and level("cpu_temp", 86) == "warning"
    assert level("cpu_temp", 91) == "serious" and level("cpu_temp", 95) == "critical"
    assert level("gpu_temp", 79) == "good" and level("gpu_temp", 88) == "serious"
    assert level("cpu_busy", 84) == "good" and level("cpu_busy", 90) == "warning"
    assert level("ssd_temp", None) == "info"


def test_the_cpu_ssd_and_ram_temperatures_are_found_among_the_sensors(tmp_path):
    def sensor(hw, name, readings):
        d = tmp_path / hw
        d.mkdir()
        (d / "name").write_text(name + "\n")
        for i, (label, millideg) in enumerate(readings, 1):
            (d / f"temp{i}_input").write_text(f"{millideg}\n")
            if label:
                (d / f"temp{i}_label").write_text(label + "\n")

    sensor("hwmon0", "nvme", [("Composite", 40900), ("Sensor 1", 41000)])
    sensor("hwmon1", "amdgpu", [("edge", 51000)])
    sensor("hwmon2", "k10temp", [("Tctl", 60400), ("Tccd1", 55100)])
    sensor("hwmon3", "spd5118", [(None, 36250)])
    sensor("hwmon4", "r8169_0_900:00", [(None, 51500)])
    assert hwmon_temps(tmp_path) == {"ssd": 40.9, "cpu": 60.4, "cpu_die": 55.1, "ram": 36.2}
    assert hwmon_temps(tmp_path / "missing") == {}


def test_the_gpu_reading_says_when_it_is_slowing_itself_down():
    line = "NVIDIA GeForce RTX 5070, 41, 8, 4398, 12227, 34.13, 250.00, 30, Not Active, Not Active\n"
    gpu = gpu_status(run=lambda *a, **k: completed(line))
    assert gpu == {"name": "NVIDIA GeForce RTX 5070", "temp": 41.0, "busy": 8.0, "vram_used": 4398.0,
                   "vram_total": 12227.0, "power": 34.13, "power_limit": 250.0, "fan": 30.0, "throttling": False}
    hot = line.replace("Not Active, Not Active", "Not Active, Active")
    assert gpu_status(run=lambda *a, **k: completed(hot))["throttling"] is True
    assert gpu_status(run=lambda *a, **k: completed("")) is None  # no NVIDIA GPU


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "instances.py").write_text(FAKE_INSTANCES)
    (tmp_path / "runs" / "instances").mkdir(parents=True)
    (tmp_path / "runs" / "instances" / "fleet.json").write_text(json.dumps({"n": 6, "display_base": 190}))
    return tmp_path


def test_the_fleet_starts_and_stops_through_its_own_script_one_operation_at_a_time(repo, tmp_path_factory, monkeypatch):
    supervisor = Supervisor(repo, tmp_path_factory.mktemp("state"))
    [fleet] = supervisor.fleets()["fleets"]
    assert fleet["label"] == "main" and fleet["n"] == 6 and fleet["running"] == 0 and fleet["op"] is None
    assert "between 1 and 16" in supervisor.fleet_action("main", "up", 0)["error"]
    assert "no fleet" in supervisor.fleet_action("elsewhere", "down")["error"]
    assert "up or down" in supervisor.fleet_action("main", "restart")["error"]

    monkeypatch.setenv("FAKE_SECONDS", "2")
    assert supervisor.fleet_action("main", "up", 6) == {"ok": True}
    assert "already starting" in supervisor.fleet_action("main", "down")["error"]
    op = supervisor.fleets()["fleets"][0]["op"]
    assert op["action"] == "up" and op["running"]
    deadline = time.time() + 10
    while supervisor.fleets()["fleets"][0]["op"]["running"] and time.time() < deadline:
        time.sleep(0.1)
    op = supervisor.fleets()["fleets"][0]["op"]
    assert not op["running"] and op["code"] == 0 and op["tail"][-1] == "done"
    log = (repo / "runs" / "instances" / "up.log").read_text()
    assert "from the dashboard: scripts/instances.py up --n 6" in log and "instances.py up --n 6" in log


def test_games_a_run_is_playing_are_not_stopped_under_it(repo, tmp_path_factory, monkeypatch):
    supervisor = Supervisor(repo, tmp_path_factory.mktemp("state"))
    playing = [{"pid": 7, "run": "rl9", "cwd": str(repo), "rehearsal": False, "tree": repo.name, "stopping_s": None,
                "can_force": False}]
    monkeypatch.setattr(supervisor, "live", lambda: {"trainers": playing})
    assert supervisor.fleet_action("main", "down") == {"error": "rl9 is playing these games: stop it first"}
    assert not (repo / "runs" / "instances" / "down.log").exists()
    playing[0]["rehearsal"] = True  # a rehearsal on the synthetic stand-in plays no game
    assert supervisor.fleet_action("main", "down") == {"ok": True}


def test_the_machine_reports_now_and_the_peak_of_the_last_ten_minutes(tmp_path, monkeypatch):
    supervisor = Supervisor(tmp_path, tmp_path / "state")
    temps = iter([{"cpu": 70.0, "ssd": 40.0}, {"cpu": 62.0, "ssd": 41.0}])
    monkeypatch.setattr(supervise, "hwmon_temps", lambda: next(temps))
    monkeypatch.setattr(supervise, "gpu_status", lambda: None)
    supervisor.machine()
    time.sleep(0.05)  # CPU load is the time between two readings: none within one 10 ms tick
    m = supervisor.machine()
    assert m["now"]["cpu_temp"] == 62.0 and m["peak"]["cpu_temp"] == 70.0 and m["peak"]["ssd_temp"] == 41.0
    assert m["levels"]["cpu_temp"] == "good" and m["now"]["gpu_temp"] is None and 0 <= m["now"]["cpu_busy"] <= 100
