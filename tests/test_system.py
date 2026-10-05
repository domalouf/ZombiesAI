import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from zombiesai.viz import supervise
from zombiesai.viz.supervise import Handler
from zombiesai.viz.system import SystemSampler, disk_counters, parse_gpu_line, sensors

GPU_ROW = ("0, NVIDIA GeForce RTX 5070, 610.57.04, 97, 41, 8368, 12227, 71, 231.40, 250.00, 64, 2610, 14001, P0, "
           "0x0000000000000024\n")


class FakeGpu:
    def __init__(self, rows):
        self.rows = rows
        self.closed = 0

    def read(self):
        return self.rows

    def close(self):
        self.closed += 1


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def machine(root: Path, *, idle: int, busy: int, core_busy: int, read_sectors: int, rx: int, ticks: int) -> None:
    """A two-core machine's /proc and /sys, with the counters that move between samples as arguments."""
    proc, sys = root / "proc", root / "sys"
    write(proc / "stat", f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 {core_busy} 0 0 {idle // 2} 0 0 0 0 0 0\n"
                         f"cpu1 {busy - core_busy} 0 0 {idle // 2} 0 0 0 0 0 0\nintr 1 2 3\n")
    write(proc / "meminfo", "MemTotal:  8000000 kB\nMemFree:  1000000 kB\nMemAvailable:  6000000 kB\n"
                            "Buffers: 0 kB\nCached:  2000000 kB\nSwapTotal:  1000000 kB\nSwapFree:  750000 kB\n")
    write(proc / "diskstats", f" 259 0 nvme0n1 10 0 {read_sectors} 0 5 0 8 0 0 {read_sectors} 0\n"
                              f" 259 1 nvme0n1p1 10 0 {read_sectors} 0 5 0 8 0 0 0 0\n"
                              " 7 0 loop0 1 0 999999 0 0 0 0 0 0 0 0\n 252 0 zram0 1 0 999999 0 0 0 0 0 0 0 0\n")
    write(proc / "net" / "dev", "Inter-|   Receive\n face |bytes packets\n"
                                f"    lo: 999 1 0 0 0 0 0 0 999 1 0 0 0 0 0 0\n"
                                f"wlp8s0: {rx} 1 0 0 0 0 0 0 500 1 0 0 0 0 0 0\n"
                                f"enp9s0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    write(proc / "loadavg", "1.50 1.00 0.50 2/300 4242\n")
    write(proc / "uptime", "3600.0 7000.0\n")
    write(proc / "mounts", "")
    pid = proc / "4242"
    write(pid / "stat", f"4242 (python (trainer)) R 1 1 1 0 -1 0 0 0 0 0 {ticks} 0 0 0 20 0 7 0 100 1000 256\n")
    write(pid / "cmdline", "python\0-u\0scripts/train_rl.py\0")
    hw = sys / "class" / "hwmon" / "hwmon2"
    write(hw / "name", "k10temp\n")
    write(hw / "temp1_input", "68500\n")
    write(hw / "temp1_label", "Tctl\n")
    write(sys / "class" / "hwmon" / "hwmon0" / "name", "nvme\n")
    write(sys / "class" / "hwmon" / "hwmon0" / "temp1_input", "41800\n")
    write(sys / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_cur_freq", "5100000\n")


def test_two_samples_give_the_rates_btop_shows(tmp_path):
    hz = os.sysconf("SC_CLK_TCK")
    sampler = SystemSampler(proc=tmp_path / "proc", sys=tmp_path / "sys", gpu=FakeGpu([parse_gpu_line(GPU_ROW)]))
    machine(tmp_path, idle=1000, busy=1000, core_busy=500, read_sectors=0, rx=0, ticks=0)
    first = sampler.sample(now=100.0)
    assert first["cpu"]["pct"] is None and sampler.history() == []  # nothing to take a rate over yet
    # Two seconds later: 300 of 400 jiffies busy (75%), core 0 all busy, 1 MiB read, 2 MiB in, 1.5 cores of CPU.
    machine(tmp_path, idle=1100, busy=1300, core_busy=750, read_sectors=2048, rx=2 * 2**20, ticks=3 * hz)
    snap = sampler.sample(now=102.0)

    cpu = snap["cpu"]
    assert cpu["pct"] == 75.0 and cpu["cores"] == [83.3, 50.0]  # core 0: 250 of 300; core 1: 50 of 100
    assert cpu["freqs"] == [5.1, None] and cpu["load"] == [1.5, 1.0, 0.5]
    assert cpu["temp"] == 68.5  # Tctl, the reading the CPU throttles on
    mem = snap["memory"]
    assert mem["pct"] == 25.0 and mem["swap_pct"] == 25.0 and mem["used"] == 2000000 * 1024
    assert snap["disks"] == [{"name": "nvme0n1", "read": 2**20 / 2, "write": 0.0, "busy": 100.0}]  # no partitions
    assert [n["name"] for n in snap["net"]] == ["wlp8s0"] and snap["net"][0]["rx"] == 2**20  # lo and idle ports out
    [p] = snap["procs"]
    assert p["name"] == "python (trainer)" and p["cpu"] == 150.0 and p["threads"] == 7
    assert p["command"] == "python -u scripts/train_rl.py"
    assert snap["gpus"][0]["util"] == 97 and snap["uptime_s"] == 3600.0

    [point] = sampler.history()
    assert point["cpu"] == 75.0 and point["gpu_temp"] == 71 and round(point["vram"]) == 68 and point["net_rx"] == 2**20
    assert sampler.history(since=102.0) == []  # the page asks only for what it has not got
    assert sampler.payload()["specs"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 5070"


def test_a_paused_sampler_stops_nvidia_smi_and_takes_no_rate_across_the_pause(tmp_path):
    gpu = FakeGpu([])
    sampler = SystemSampler(proc=tmp_path / "proc", sys=tmp_path / "sys", gpu=gpu)
    machine(tmp_path, idle=1000, busy=1000, core_busy=500, read_sectors=0, rx=0, ticks=0)
    sampler.sample(now=100.0)
    sampler.pause()
    assert sampler.paused
    threading.Thread(target=sampler._rest, daemon=True).start()  # what the sampling thread does when paused
    deadline = time.monotonic() + 5
    while not gpu.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gpu.closed == 1
    sampler.resume()
    assert not sampler.paused
    while sampler._prev is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    machine(tmp_path, idle=1100, busy=1300, core_busy=750, read_sectors=0, rx=0, ticks=0)
    assert sampler.sample(now=400.0)["cpu"]["pct"] is None and sampler.history() == []
    machine(tmp_path, idle=1200, busy=1400, core_busy=800, read_sectors=0, rx=0, ticks=0)
    assert sampler.sample(now=402.0)["cpu"]["pct"] == 50.0


def test_a_gpu_row_reads_as_numbers_with_its_reasons_to_slow_down():
    g = parse_gpu_line(GPU_ROW)
    assert g["name"] == "NVIDIA GeForce RTX 5070" and g["temp"] == 71 and g["power"] == 231.4
    assert g["vram_total"] == 12227 * 2**20 and g["slowdown"] == ["power cap", "thermal (driver)"]
    assert parse_gpu_line("0, x, 1, [N/A]" + ", 1" * 11)["util"] is None
    assert parse_gpu_line("nvidia-smi has failed") is None


def test_whole_disks_only_and_every_sensor_labelled(tmp_path):
    machine(tmp_path, idle=0, busy=0, core_busy=0, read_sectors=4, rx=0, ticks=0)
    assert list(disk_counters(tmp_path / "proc")) == ["nvme0n1"]
    found = {(s["chip"], s["label"]): s for s in sensors(tmp_path / "sys")}
    assert found[("k10temp", "Tctl")]["kind"] == "cpu" and found[("nvme", "temp1")]["value"] == 41.8


def test_the_machine_is_served_with_only_the_history_asked_for(tmp_path):
    sampler = SystemSampler(proc=tmp_path / "proc", sys=tmp_path / "sys", gpu=FakeGpu([]))
    machine(tmp_path, idle=10, busy=10, core_busy=5, read_sectors=0, rx=0, ticks=0)
    sampler.sample(now=100.0)
    machine(tmp_path, idle=30, busy=10, core_busy=5, read_sectors=0, rx=0, ticks=0)
    sampler.sample(now=102.0)
    fake = SimpleNamespace(system_payload=lambda since: sampler.payload(since), state_dir=tmp_path)
    server = supervise.ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"supervisor": fake, "port": 0}))
    server.RequestHandlerClass.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        full = json.load(urllib.request.urlopen(base + "/api/system"))
        assert full["now"]["cpu"]["pct"] == 0.0 and len(full["history"]) == 1 and full["specs"]["threads"]
        assert json.load(urllib.request.urlopen(base + "/api/system?since=102"))["history"] == []
    finally:
        server.shutdown()
