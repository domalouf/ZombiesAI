import subprocess
from pathlib import Path

from zombiesai.rl import capacity
from zombiesai.rl.capacity import Gpu, Snapshot, YieldWatch, games_for, owner_busy, owner_games

RTX_2080_TI = Gpu("NVIDIA GeForce RTX 2080 Ti", 11.0, 10.2)
RTX_5070 = Gpu("NVIDIA GeForce RTX 5070", 12.0, 10.8)


def test_a_roomy_gaming_pc_is_capped():
    v = games_for(Snapshot(cpus=16, mem_available_gb=26.0, gpus=(RTX_2080_TI,)))
    assert v.games == capacity.MAX_GAMES and v.limits["cpu"] == 14 and v.limits["vram"] == 8
    assert "capped at 8" in v.why


def test_the_learner_pc_keeps_room_for_the_learner():
    alone = games_for(Snapshot(cpus=12, mem_available_gb=24.0, gpus=(RTX_5070,)))
    learning = games_for(Snapshot(cpus=12, mem_available_gb=24.0, gpus=(RTX_5070,)), learner=True)
    assert learning.games < alone.games
    assert learning.games == 6 and learning.limits["cpu"] == 8 and learning.limits["vram"] == 6


def test_the_scarcest_resource_decides_and_says_so():
    v = games_for(Snapshot(cpus=16, mem_available_gb=9.0, gpus=(RTX_2080_TI,)))
    # 5 GB after the desktop's 4: two slots of 1.3 + 0.7 fit, three do not
    assert v.games == 2 and v.limits["ram"] == 2 and "9.0 GB RAM available" in v.why
    v = games_for(Snapshot(cpus=4, mem_available_gb=26.0, gpus=(RTX_2080_TI,)))
    assert v.games == 2 and "4 CPUs" in v.why
    v = games_for(Snapshot(cpus=16, mem_available_gb=26.0, gpus=(Gpu("small", 4.0, 3.0),)))
    assert v.games == 1 and "VRAM free" in v.why


def test_games_already_running_are_not_charged_twice():
    """Their RAM and VRAM are already missing from the snapshot; only their actors still have to fit."""
    tight = Snapshot(cpus=16, mem_available_gb=7.0, gpus=(Gpu("g", 11.0, 2.0),), games_running=4)
    v = games_for(tight)
    assert v.games == 4 and v.limits["vram"] == 4 and v.limits["ram"] == 4  # 4 actors x 0.7 in 3 GB
    assert "4 already running" in v.why
    cold = Snapshot(cpus=16, mem_available_gb=7.0, gpus=(Gpu("g", 11.0, 2.0),))
    assert games_for(cold).games == 0  # the same numbers with nothing up: no room to start one


def test_no_gpu_seen_and_no_room_at_all():
    v = games_for(Snapshot(cpus=8, mem_available_gb=12.0))
    assert "vram" not in v.limits and "no GPU seen" in v.why and v.games == 4
    assert games_for(Snapshot(cpus=2, mem_available_gb=3.0)).games == 0


def test_probe_reads_meminfo_nvidia_smi_and_the_fleet(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       32781240 kB\nMemFree: 1000 kB\nMemAvailable:   25435300 kB\n")

    def run(argv, **kwargs):
        assert argv[0] == "nvidia-smi" and "--format=csv,noheader,nounits" in argv
        return subprocess.CompletedProcess(argv, 0, "NVIDIA GeForce RTX 2080 Ti, 11264, 10411\n", "")

    snap = capacity.probe(tmp_path / "no-fleet", proc_root=proc, run=run)
    assert snap.cpus >= 1 and abs(snap.mem_available_gb - 24.26) < 0.01 and snap.games_running == 0
    assert snap.gpus == (Gpu("NVIDIA GeForce RTX 2080 Ti", 11.0, 10411 / 1024),)


def test_without_nvidia_smi_amdgpu_sysfs_is_read(tmp_path):
    device = tmp_path / "sys" / "class" / "drm" / "card1" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text(str(16 * 2**30))
    (device / "mem_info_vram_used").write_text(str(4 * 2**30))

    def run(argv, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert capacity.probe_gpus(run=run, sys_root=tmp_path / "sys") == [Gpu("AMD GPU (card1)", 16.0, 12.0)]
    assert capacity.parse_nvidia_smi("garbage\n\n") == []


# ------------------------------------------------------------------------------------------------ the owner


def fake_proc(root: Path, pid: int, argv: list[str], env: dict | None = None) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    if env is not None:
        (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0")


def a_pc(tmp_path: Path, fleet_root: Path) -> Path:
    """A /proc with Steam itself, our two games, and nothing of the owner's."""
    proc = tmp_path / "proc"
    fake_proc(proc, 1, ["/sbin/init"])
    fake_proc(proc, 2000, [str(Path.home() / ".local/share/Steam/ubuntu12_32/steam"), "-srt-logger-opened"])
    (proc / "self").mkdir()  # not a pid
    for i in range(2):
        # How instances.py launches a game: umu-run in a namespace, Steam's app id in its environment.
        ours = {"WINEPREFIX": str(fleet_root / f"i{i}" / "compat"), "SteamAppId": "10090", "UMU_ID": "umu-10090"}
        fake_proc(proc, 3000 + i, ["/usr/bin/python3", "/usr/bin/umu-run",
                                   str(fleet_root / f"i{i}" / "plutonium/bin/plutonium-bootstrapper-win32.exe"),
                                   "t4sp", "Z:\\games\\waw", "-lan"], ours)
        fake_proc(proc, 3100 + i, ["C:\\windows\\system32\\wineserver"], ours)
    return proc


def test_our_own_games_are_not_the_owner_playing(tmp_path):
    fleet_root = tmp_path / "runs" / "instances"
    proc = a_pc(tmp_path, fleet_root)
    assert owner_games(proc, fleet_root=fleet_root) == []
    assert owner_busy(proc, fleet_root=fleet_root, pause_file=tmp_path / "pause") is None


def test_a_steam_game_is_the_owner_playing(tmp_path):
    fleet_root = tmp_path / "runs" / "instances"
    proc = a_pc(tmp_path, fleet_root)
    steam = str(Path.home() / ".local/share/Steam/ubuntu12_32")
    fake_proc(proc, 5000, [f"{steam}/reaper", "SteamLaunch", "AppId=1091500", "--", f"{steam}/steam-launch-wrapper",
                           "--", "proton", "waitforexitandrun", "Cyberpunk2077.exe"],
              {"SteamAppId": "1091500", "WINEPREFIX": "/home/x/.steam/steam/steamapps/compatdata/1091500/pfx"})
    assert owner_games(proc, fleet_root=fleet_root) == ["a Steam game (app 1091500)"]
    assert "app 1091500" in owner_busy(proc, fleet_root=fleet_root, pause_file=None)


def test_a_steam_launch_of_our_own_prefix_is_still_ours(tmp_path):
    fleet_root = tmp_path / "runs" / "instances"
    proc = a_pc(tmp_path, fleet_root)
    fake_proc(proc, 5001, ["reaper", "SteamLaunch", "AppId=10090", "--", "x"],
              {"WINEPREFIX": str(fleet_root / "i3" / "compat")})
    assert owner_games(proc, fleet_root=fleet_root) == []


def test_an_owners_game_through_umu_run_counts_and_an_unreadable_one_is_assumed_theirs(tmp_path):
    fleet_root = tmp_path / "runs" / "instances"
    proc = a_pc(tmp_path, fleet_root)
    fake_proc(proc, 6000, ["umu-run", "/games/heroic/Hades/Hades.exe"], {"WINEPREFIX": "/games/heroic/prefixes/Hades"})
    fake_proc(proc, 6001, ["/usr/bin/python3", "/usr/bin/umu-run", "Other.exe"])  # another user's: no environ
    found = owner_games(proc, fleet_root=fleet_root)
    assert sorted(found) == ["a game through umu-run (Hades.exe)", "a game through umu-run (Other.exe)"]


def test_the_pause_file_pauses(tmp_path):
    pause = tmp_path / "pause"
    proc = a_pc(tmp_path, tmp_path / "fleet")
    assert owner_busy(proc, fleet_root=tmp_path / "fleet", pause_file=pause) is None
    pause.touch()
    assert owner_busy(proc, fleet_root=tmp_path / "fleet", pause_file=pause) == f"paused by {pause}"


def test_the_real_proc_scan_is_cheap_and_finds_no_game_of_ours(tmp_path):
    if not Path("/proc/self/cmdline").exists():
        return
    import time

    start = time.perf_counter()
    owner_games("/proc", fleet_root=tmp_path)
    assert time.perf_counter() - start < 1.0


def test_yielding_is_at_once_and_rejoining_waits_for_a_quiet_minute():
    now = [0.0]
    busy = [None]
    watch = YieldWatch(fleet_root=None, resume_after_s=60.0, clock=lambda: now[0], scan=lambda: busy[0])
    assert watch.check() is None
    busy[0] = "the owner is playing a Steam game (app 1)"
    assert watch.check() == busy[0]
    busy[0] = None
    now[0] = 10.0
    assert watch.check() is not None and watch.resuming_in() == 60.0  # free, but not for long enough
    now[0] = 40.0
    busy[0] = "the owner is playing a Steam game (app 2)"  # the next game: the minute starts over
    assert watch.check().endswith("(app 2)")
    busy[0] = None
    now[0] = 50.0
    assert watch.check() is not None
    now[0] = 109.0
    assert watch.check() is not None and 0 < watch.resuming_in() < 2
    now[0] = 110.0
    assert watch.check() is None and watch.resuming_in() is None
