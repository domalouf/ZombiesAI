"""The machine the training runs on, as btop shows it: CPU (total, per core, clocks, load), memory and swap, the
GPU, every temperature sensor, disks, network, and the processes using the most CPU -- for the Training Room.

Stdlib only, like the rest of viz/: it reads /proc and /sys, and the GPU from one long-lived `nvidia-smi -lms`
stream rather than a process per sample. `SystemSampler` samples on a thread every `interval_s` and keeps
`history_s` of the headline numbers, so a page opened after a spike still shows it.
"""

import os
import pwd
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

GPU_FIELDS = ("index", "name", "driver_version", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "temperature.gpu", "power.draw", "power.limit", "fan.speed", "clocks.gr",
              "clocks.mem", "pstate", "clocks_event_reasons.active")
# Why the GPU is running below its clocks, from the clocks_event_reasons bitmask (nvml.h). Idle is not news.
GPU_SLOWDOWNS = {0x4: "power cap", 0x8: "hardware slowdown", 0x20: "thermal (driver)", 0x40: "thermal (hardware)",
                 0x80: "power brake"}
# (warning, critical) in °C. The 7600X throttles at 95 (Tctl); an RTX 50 starts slowing down near 90; an NVMe
# drive throttles around 80. Anything unrecognised is shown without a verdict.
TEMP_LIMITS = {"cpu": (85, 95), "gpu": (80, 88), "nvme": (70, 80)}
CPU_SENSORS = (("k10temp", "Tctl"), ("zenpower", "Tdie"), ("coretemp", "Package id 0"))
HISTORY_KEYS = ("cpu", "mem", "swap", "gpu", "vram", "cpu_temp", "gpu_temp", "gpu_power", "net_rx", "net_tx",
                "disk_read", "disk_write")
# Block devices that are not disks: loop mounts, ramdisks, device-mapper views of a disk counted already, zram swap.
NOT_DISKS = re.compile(r"^(loop|ram|dm-|zram|md|sr)\d*")
SECTOR = 512


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _num(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 1) if whole > 0 else None


# ------------------------------------------------------------------------------------------------ readers


def cpu_times(proc: Path) -> dict[str, tuple[int, int]]:
    """(busy, total) jiffies for "cpu" and each "cpuN". Guest time is already inside user, so it is not added."""
    out = {}
    for line in (_read(proc / "stat") or "").splitlines():
        if not line.startswith("cpu"):
            break
        name, *fields = line.split()
        v = [int(x) for x in fields[:8]] + [0] * max(0, 8 - len(fields))
        total = sum(v)
        out[name] = (total - v[3] - v[4], total)  # idle and iowait are the not-busy parts
    return out


def meminfo(proc: Path) -> dict[str, int]:
    """/proc/meminfo in bytes."""
    out = {}
    for line in (_read(proc / "meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    return out


def disk_counters(proc: Path) -> dict[str, tuple[int, int, int]]:
    """Whole disks only: (bytes read, bytes written, ms spent doing I/O). Partitions would count twice."""
    out = {}
    for line in (_read(proc / "diskstats") or "").splitlines():
        f = line.split()
        if len(f) < 13 or NOT_DISKS.match(f[2]) or re.search(r"(\d+p\d+|[a-z]d[a-z]+\d+)$", f[2]):
            continue
        out[f[2]] = (int(f[5]) * SECTOR, int(f[9]) * SECTOR, int(f[12]))
    return out


def net_counters(proc: Path) -> dict[str, tuple[int, int]]:
    """(bytes received, bytes sent) per interface, loopback left out."""
    out = {}
    for line in (_read(proc / "net" / "dev") or "").splitlines()[2:]:
        name, _, rest = line.partition(":")
        f = rest.split()
        if name.strip() != "lo" and len(f) >= 9:
            out[name.strip()] = (int(f[0]), int(f[8]))
    return out


def mounts(proc: Path, statvfs=os.statvfs) -> list[dict]:
    """Every filesystem on a real device, once: a btrfs volume mounted at five subvolumes is one disk's space."""
    seen, out = set(), []
    for line in (_read(proc / "mounts") or "").splitlines():
        f = line.split()
        if len(f) < 3 or not f[0].startswith("/dev/") or f[0] in seen:
            continue
        seen.add(f[0])
        mount = f[1].replace("\\040", " ")
        try:
            st = statvfs(mount)
        except OSError:
            continue
        total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
        used = total - st.f_bfree * st.f_frsize
        out.append({"mount": mount, "device": f[0], "fs": f[2], "total": total, "used": used, "free": free,
                    "pct": _pct(used, used + free)})
    return out


def sensors(sys: Path) -> list[dict]:
    """Every hwmon temperature and fan: chip, label, value, and what kind of part it is ("cpu", "nvme", ...)."""
    out = []
    root = sys / "class" / "hwmon"
    for hw in sorted(root.iterdir(), key=lambda p: p.name) if root.is_dir() else []:
        chip = (_read(hw / "name") or hw.name).strip()
        for item in sorted(hw.glob("temp*_input")) + sorted(hw.glob("fan*_input")):
            value = _num(_read(item))
            if value is None:
                continue
            base = item.name[: -len("_input")]
            label = (_read(hw / f"{base}_label") or "").strip() or base
            is_fan = base.startswith("fan")
            kind = ("cpu" if chip in {c for c, _ in CPU_SENSORS} else "nvme" if chip == "nvme" else
                    "gpu" if chip in ("amdgpu", "nouveau") else "other")
            out.append({"chip": chip, "label": label, "kind": kind, "fan": is_fan,
                        "value": value if is_fan else round(value / 1000.0, 1)})
    return out


def cpu_temp(found: list[dict]) -> float | None:
    for chip, label in CPU_SENSORS:
        for s in found:
            if s["chip"] == chip and s["label"] == label and not s["fan"]:
                return s["value"]
    return next((s["value"] for s in found if s["kind"] == "cpu" and not s["fan"]), None)


def core_freqs(sys: Path, n: int) -> list[float | None]:
    """Each logical CPU's current clock in GHz."""
    base = sys / "devices" / "system" / "cpu"
    out = []
    for i in range(n):
        khz = _num(_read(base / f"cpu{i}" / "cpufreq" / "scaling_cur_freq"))
        out.append(None if khz is None else round(khz / 1e6, 2))
    return out


class _Users:
    def __init__(self):
        self.names: dict[int, str] = {}

    def __call__(self, uid: int) -> str:
        if uid not in self.names:
            try:
                self.names[uid] = pwd.getpwuid(uid).pw_name
            except KeyError:
                self.names[uid] = str(uid)
        return self.names[uid]


def process_table(proc: Path) -> dict[int, dict]:
    """Per process: its name, state, CPU jiffies so far, resident bytes, threads, and owner."""
    page = os.sysconf("SC_PAGE_SIZE")
    out = {}
    for entry in proc.iterdir() if proc.is_dir() else []:
        if not entry.name.isdigit():
            continue
        stat = _read(entry / "stat")
        if not stat or ")" not in stat:
            continue
        head, _, tail = stat.rpartition(")")
        f = tail.split()
        if len(f) < 22:
            continue
        try:
            uid = (entry / "stat").stat().st_uid
        except OSError:
            continue
        out[int(entry.name)] = {"name": head.partition("(")[2], "state": f[0], "ticks": int(f[11]) + int(f[12]),
                                "threads": int(f[17]), "rss": int(f[21]) * page, "uid": uid}
    return out


def cmdline(proc: Path, pid: int, limit: int = 160) -> str:
    raw = (proc / str(pid) / "cmdline")
    try:
        text = raw.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""
    return text[:limit]


# ------------------------------------------------------------------------------------------------ the GPU


def parse_gpu_line(line: str) -> dict | None:
    """One nvidia-smi CSV row (GPU_FIELDS, noheader, nounits) as numbers, "[N/A]" as None."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(GPU_FIELDS) or not parts[0].isdigit():
        return None
    raw = dict(zip(GPU_FIELDS, parts))
    num = {k: _num(v) for k, v in raw.items()}
    try:
        reasons = int(raw["clocks_event_reasons.active"], 16)
    except ValueError:
        reasons = 0
    return {
        "index": int(raw["index"]), "name": raw["name"], "driver": raw["driver_version"],
        "util": num["utilization.gpu"], "mem_util": num["utilization.memory"],
        "vram_used": None if num["memory.used"] is None else num["memory.used"] * 2**20,
        "vram_total": None if num["memory.total"] is None else num["memory.total"] * 2**20,
        "temp": num["temperature.gpu"], "power": num["power.draw"], "power_limit": num["power.limit"],
        "fan": num["fan.speed"], "clock": num["clocks.gr"], "mem_clock": num["clocks.mem"], "pstate": raw["pstate"],
        "slowdown": [word for bit, word in GPU_SLOWDOWNS.items() if reasons & bit],
    }


class GpuStream:
    """`nvidia-smi --query-gpu ... -lms`, one process for the life of the server, its latest row per GPU kept.
    A machine without nvidia-smi has no GPUs here; a stream that dies is started again, at most every 30 s."""

    def __init__(self, interval_s: float = 2.0, command: str = "nvidia-smi"):
        self.interval_s, self.command = interval_s, command
        self.latest: dict[int, dict] = {}
        self.proc: subprocess.Popen | None = None
        self.next_try = 0.0
        self._lock = threading.Lock()

    def _start(self) -> None:
        if time.monotonic() < self.next_try:
            return
        self.next_try = time.monotonic() + 30.0
        try:
            self.proc = subprocess.Popen(
                [self.command, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
                 f"-lms={int(self.interval_s * 1000)}"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError:
            self.proc = None
            return
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            row = parse_gpu_line(line)
            if row is not None:
                row["at"] = time.time()
                with self._lock:
                    self.latest[row["index"]] = row

    def read(self) -> list[dict]:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        fresh = time.time() - 4 * self.interval_s - 2
        with self._lock:
            return [dict(g) for _, g in sorted(self.latest.items()) if g["at"] >= fresh]

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)  # reaped, not left a zombie until the next start
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.next_try = 0.0  # closed on purpose, not dying: the next read() may start it again straight away


# ------------------------------------------------------------------------------------------------ the sampler


def specs(proc: Path = Path("/proc"), sys: Path = Path("/sys"), gpus: list[dict] | None = None) -> dict:
    """What the machine is: the numbers that do not change while it runs."""
    info = _read(proc / "cpuinfo") or ""
    model = next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("model name")), None)
    cores = {(m[0], m[1]) for m in re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", info, re.S)}
    max_khz = _num(_read(sys / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "cpuinfo_max_freq"))
    release = _read(Path("/etc/os-release")) or ""
    distro = re.search(r'^PRETTY_NAME="?([^"\n]*)', release, re.M)
    uname = os.uname()
    return {
        "host": uname.nodename, "kernel": uname.release, "os": distro[1] if distro else uname.sysname,
        "cpu": model, "cores": len(cores) or None, "threads": os.cpu_count(),
        "cpu_max_ghz": None if max_khz is None else round(max_khz / 1e6, 2),
        "ram": meminfo(proc).get("MemTotal"),
        "gpus": [{"name": g["name"], "vram": g["vram_total"], "driver": g["driver"]} for g in gpus or []],
    }


class SystemSampler:
    """Samples every `interval_s` on a thread of its own and keeps `history_s` of the headline numbers.
    `snapshot()` is the latest whole sample; `history(since)` the headline points after `since`.
    `pause()` stops the sampling (and nvidia-smi) until `resume()`; the history keeps the gap as a gap."""

    def __init__(self, interval_s: float = 2.0, history_s: float = 1800.0, top: int = 15,
                 proc: Path = Path("/proc"), sys: Path = Path("/sys"), gpu: GpuStream | None = None,
                 statvfs=os.statvfs):
        self.interval_s, self.top, self.proc, self.sys, self.statvfs = interval_s, top, proc, sys, statvfs
        self.gpu = GpuStream(interval_s) if gpu is None else gpu
        self.points: deque[dict] = deque(maxlen=max(2, int(history_s / interval_s)))
        self.latest: dict | None = None
        self._prev: dict | None = None
        self._users = _Users()
        self._specs: dict | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._awake = threading.Event()
        self._awake.set()

    def start(self) -> "SystemSampler":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while True:
            self._awake.wait()
            try:
                self.sample()
            except Exception:  # one bad read (a sensor that vanished) must not end the sampling for good
                pass
            time.sleep(self.interval_s)

    @property
    def paused(self) -> bool:
        return not self._awake.is_set()

    def pause(self) -> None:
        if self.paused:
            return
        self._awake.clear()
        self.gpu.close()
        with self._lock:
            self._prev = None  # a rate over the whole pause would be an average, not what it is doing now

    def resume(self) -> None:
        self._awake.set()

    def specs(self) -> dict:
        if self._specs is None or (not self._specs["gpus"] and self.latest and self.latest["gpus"]):
            self._specs = specs(self.proc, self.sys, self.latest["gpus"] if self.latest else [])
        return self._specs

    def sample(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        cur = {"t": now, "cpu": cpu_times(self.proc), "disk": disk_counters(self.proc),
               "net": net_counters(self.proc), "procs": process_table(self.proc)}
        prev = self._prev
        dt = now - prev["t"] if prev else 0.0

        def rate(a: int, b: int) -> float | None:
            return round((a - b) / dt, 1) if dt > 0 and a >= b else None

        cpus = []
        for name, (busy, total) in cur["cpu"].items():
            b0, t0 = prev["cpu"].get(name, (busy, total)) if prev else (busy, total)
            cpus.append(_pct(busy - b0, total - t0) if prev else None)
        total_cpu, per_core = (cpus[0], cpus[1:]) if cpus else (None, [])

        mem = meminfo(self.proc)
        mem_total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", mem.get("MemFree", 0))
        swap_total, swap_free = mem.get("SwapTotal", 0), mem.get("SwapFree", 0)
        memory = {"total": mem_total, "used": mem_total - avail, "available": avail, "free": mem.get("MemFree"),
                  "cached": mem.get("Cached", 0) + mem.get("SReclaimable", 0) + mem.get("Buffers", 0),
                  "pct": _pct(mem_total - avail, mem_total),
                  "swap_total": swap_total, "swap_used": swap_total - swap_free,
                  "swap_pct": _pct(swap_total - swap_free, swap_total)}

        disks = []
        for name, (r, w, busy_ms) in cur["disk"].items():
            r0, w0, ms0 = prev["disk"].get(name, (r, w, busy_ms)) if prev else (r, w, busy_ms)
            disks.append({"name": name, "read": rate(r, r0), "write": rate(w, w0),
                          "busy": None if not dt else min(100.0, round((busy_ms - ms0) / (dt * 10), 1))})
        nets = []
        for name, (rx, tx) in cur["net"].items():
            rx0, tx0 = prev["net"].get(name, (rx, tx)) if prev else (rx, tx)
            if rx or tx:  # an interface that has never moved a byte (a down Ethernet port) is not worth a row
                nets.append({"name": name, "rx": rate(rx, rx0), "tx": rate(tx, tx0), "rx_total": rx, "tx_total": tx})

        hz, ncpu = os.sysconf("SC_CLK_TCK"), max(1, len(per_core))
        rows = []
        for pid, p in cur["procs"].items():
            before = prev["procs"].get(pid) if prev else None
            # As btop and top count it: 100% is one core, so a busy trainer can read several hundred.
            p_cpu = (round(100.0 * (p["ticks"] - before["ticks"]) / (hz * dt), 1)
                     if before and dt > 0 and p["ticks"] >= before["ticks"] else None)
            rows.append((p_cpu or 0.0, p["rss"], pid, p, p_cpu))
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        procs = [{"pid": pid, "name": p["name"], "user": self._users(p["uid"]), "state": p["state"],
                  "threads": p["threads"], "rss": p["rss"], "cpu": p_cpu, "mem_pct": _pct(p["rss"], mem_total),
                  "command": cmdline(self.proc, pid)}
                 for _, _, pid, p, p_cpu in rows[: self.top]]

        found = sensors(self.sys)
        gpus = self.gpu.read()
        load = (_read(self.proc / "loadavg") or "").split()
        uptime = _num((_read(self.proc / "uptime") or "").split(" ")[0])
        freqs = core_freqs(self.sys, len(per_core))
        known = [f for f in freqs if f is not None]
        snap = {
            "t": now, "interval_s": self.interval_s,
            "cpu": {"pct": total_cpu, "cores": per_core, "freqs": freqs,
                    "freq": round(sum(known) / len(known), 2) if known else None,
                    "load": [_num(x) for x in load[:3]], "temp": cpu_temp(found),
                    "tasks": len(cur["procs"]), "threads": sum(p["threads"] for p in cur["procs"].values())},
            "memory": memory, "gpus": gpus, "sensors": found, "disks": disks, "net": nets,
            "mounts": mounts(self.proc, self.statvfs), "procs": procs, "uptime_s": uptime, "limits": TEMP_LIMITS,
        }
        gpu0 = gpus[0] if gpus else {}
        point = {
            "t": round(now, 1), "cpu": total_cpu, "mem": memory["pct"], "swap": memory["swap_pct"],
            "gpu": gpu0.get("util"), "cpu_temp": snap["cpu"]["temp"], "gpu_temp": gpu0.get("temp"),
            "gpu_power": gpu0.get("power"),
            "vram": _pct(gpu0["vram_used"], gpu0["vram_total"]) if gpu0.get("vram_total") else None,
            "net_rx": sum(n["rx"] or 0 for n in nets) if prev else None,
            "net_tx": sum(n["tx"] or 0 for n in nets) if prev else None,
            "disk_read": sum(d["read"] or 0 for d in disks) if prev else None,
            "disk_write": sum(d["write"] or 0 for d in disks) if prev else None,
        }
        with self._lock:
            self._prev, self.latest = cur, snap
            if prev:  # the first sample has no rates to plot
                self.points.append(point)
        return snap

    def history(self, since: float = 0.0) -> list[dict]:
        with self._lock:
            return [p for p in self.points if p["t"] > since]

    def payload(self, since: float = 0.0) -> dict:
        """What /api/system answers: the specs, the latest sample, and the history after `since`."""
        with self._lock:
            latest = self.latest
        return {"specs": self.specs(), "now": latest, "history": self.history(since),
                "history_s": self.points.maxlen * self.interval_s, "keys": HISTORY_KEYS}

    def close(self) -> None:
        self.gpu.close()
