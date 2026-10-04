import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from zombiesai.realgame import instances as inst
from zombiesai.realgame.instances import FleetConfig, Instance, game_args, game_env, prepare_prefix, specs


def fake_install(tmp_path: Path, client: str = "steam") -> FleetConfig:
    game = tmp_path / "game"
    game.mkdir()
    (game / "CoDWaW.exe").write_bytes(b"MZ")
    proton = tmp_path / "proton"
    proton.mkdir()
    (proton / "proton").write_text("#!/bin/sh\n")
    template = tmp_path / "compatdata"
    (template / "pfx" / "drive_c").mkdir(parents=True)
    (template / "version").write_text("9.0-100")
    pluto = tmp_path / "plutonium"
    (pluto / "bin").mkdir(parents=True)
    (pluto / "bin" / "plutonium-bootstrapper-win32.exe").write_bytes(b"MZ")
    (pluto / "cdn_info.json").write_text('{"revision": 1}')
    return FleetConfig(n=3, display_base=70, root=str(tmp_path / "fleet"), game_dir=str(game), proton=str(proton),
                       template_prefix=str(template), client=client, plutonium_dir=str(pluto))


def test_an_instances_capture_takes_the_whole_picture_however_dark_its_first_frame(tmp_path, monkeypatch):
    """The game fills its X server, so the policy sees all of it: a dark first frame must not be read as
    letterbox bars and crop the view for the rest of the run."""
    import numpy as np

    from zombiesai.demos import frames as fr
    from zombiesai.demos import x11_capture

    config = fake_install(tmp_path)
    spec = specs(config)[0]
    frame = np.full((spec.height, spec.width, 4), 10, np.uint8)
    frame[spec.height // 3:spec.height * 2 // 3, spec.width // 2:spec.width * 9 // 10, :3] = 120  # one lit doorway

    class Grabber:
        def __init__(self, window=None, display=None, region=None):
            pass

        def grab_bgrx(self):
            return frame

        def describe(self):
            return {"kind": "x11"}

        def close(self):
            pass

    monkeypatch.setattr(x11_capture, "X11Grabber", Grabber)
    capture = Instance(config, spec, say=lambda m: None).capture()
    policy = capture.read()
    assert capture.describe()["box"] == (0, 0, spec.height, spec.width)
    np.testing.assert_array_equal(policy, fr.to_policy_frame(frame, (0, 0, spec.height, spec.width),
                                                             channels=fr.BGRX))


def test_each_instance_gets_its_own_display_prefix_and_sink(tmp_path):
    config = fake_install(tmp_path)
    s = specs(config)
    assert [x.display for x in s] == [":70", ":71", ":72"]
    assert len({x.prefix for x in s}) == 3 and len({x.sink for x in s}) == 3
    assert s[1].socket == Path("/tmp/.X11-unix/X71") and s[1].monitor == "zombiesai_1.monitor"


def test_game_args_set_the_resolution_console_and_start_in_nacht(tmp_path):
    config = fake_install(tmp_path)
    args = game_args(config, specs(config)[0])
    pairs = {args[i + 1]: args[i + 2] for i in range(len(args) - 2) if args[i] == "+set"}
    assert pairs["r_mode"] == "2560x1440" and pairs["monkeytoy"] == "0" and pairs["r_fullscreen"] == "1"
    assert args[-2:] == ["+map", "nazi_zombie_prototype"]


def test_game_env_lives_on_its_display_and_nowhere_else(tmp_path):
    config = fake_install(tmp_path)
    spec = specs(config)[2]
    env = game_env(config, spec, base={"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0", "HOME": "/h"})
    assert "WAYLAND_DISPLAY" not in env and env["DISPLAY"] == ":72"
    assert env["WINEPREFIX"] == str(spec.prefix.resolve()) and env["PULSE_SINK"] == "zombiesai_2"
    assert env["PROTONPATH"] == config.proton and env["SteamAppId"] == "10090" and env["HOME"] == "/h"


def test_the_prefix_is_copied_once(tmp_path):
    config = fake_install(tmp_path)
    spec = specs(config)[0]
    said = []
    prepare_prefix(config, spec, say=said.append)
    assert (spec.prefix / "pfx" / "drive_c").is_dir() and (spec.prefix / "version").read_text() == "9.0-100"
    (spec.prefix / "pfx" / "mine").write_text("kept")
    prepare_prefix(config, spec, say=said.append)
    assert (spec.prefix / "pfx" / "mine").read_text() == "kept" and len(said) == 1


def test_missing_installs_say_what_to_pass(tmp_path):
    config = FleetConfig(game_dir=str(tmp_path), template_prefix=str(tmp_path), proton=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="--game-dir"):
        config.resolved_game_dir()
    with pytest.raises(FileNotFoundError, match="--template-prefix"):
        config.resolved_template()
    with pytest.raises(FileNotFoundError, match="--proton"):
        config.resolved_proton()


def completed(stdout: str, code: int = 0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


def test_hyprland_dispatch_speaks_either_dialect_and_never_twice():
    calls = []

    def lua_ok(args, **kwargs):
        calls.append(args)
        return completed("ok\n")

    assert inst.hyprland_exec("Xwayland :70", "float", run=lua_ok)
    assert len(calls) == 1 and calls[0][2].startswith("hl.dsp.exec_cmd(")

    calls.clear()

    def classic_only(args, **kwargs):
        calls.append(args)
        return completed("ok" if args[2] == "exec" else "error: lua")

    assert inst.hyprland_exec("Xwayland :70", "float", run=classic_only)
    assert calls[-1] == ["hyprctl", "dispatch", "exec", "[float] Xwayland :70"]
    assert not inst.toggle_shown(run=lambda args, **kw: completed("error"))


def test_sinks_are_made_only_when_missing(tmp_path):
    spec = specs(fake_install(tmp_path))[0]
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["pactl", "list", "short"]:
            return completed("0\talsa_output\tmodule\n" + ("5\tzombiesai_0\tmodule-null-sink\n" if made else ""))
        return completed("")

    made = False
    assert inst.ensure_sink(spec, run=run)
    assert any("load-module" in a for a in calls[-1])
    made, calls[:] = True, []
    assert inst.ensure_sink(spec, run=run) and len(calls) == 1


def test_focuser_finds_the_window_again_after_losing_it(tmp_path):
    from zombiesai.demos.x11_capture import WindowNotFound

    class Win:
        id = 0x400001

    class Sink:
        def __init__(self):
            self.focused = []

        def focus(self, window):
            self.focused.append(window)
            return True

    now = [0.0]
    config = fake_install(tmp_path)
    instance = Instance(config, specs(config)[0], say=lambda m: None)
    windows = [WindowNotFound("not yet"), Win()]

    def window():
        w = windows.pop(0) if len(windows) > 1 else windows[0]
        if isinstance(w, Exception):
            raise w
        return w

    instance.window = window
    sink = Sink()
    focus = instance.focuser(sink, refind_s=2.0, clock=lambda: now[0])
    assert focus() is False
    now[0] = 1.0
    assert focus() is False  # not looked up again inside refind_s
    now[0] = 2.5
    assert focus() is True and sink.focused == [0x400001]
    assert focus() is True and sink.focused == [0x400001, 0x400001]


def test_plutonium_starts_t4_offline_in_lan_mode_without_map(tmp_path):
    config = fake_install(tmp_path, client="plutonium")
    spec = specs(config)[1]
    command, cwd = inst.game_command(config, spec)
    assert command[0] == "umu-run" and command[1] == str((spec.plutonium / "bin" / "plutonium-bootstrapper-win32.exe").resolve())
    assert command[2] == "t4sp" and command[3] == inst.windows_path(tmp_path / "game") and command[3].startswith("Z:\\")
    assert "-lan" in command and command[command.index("+name") + 1] == "zombiesai1"
    assert "+map" not in command  # thrown back to the profile dialog in LAN mode; the reset types it instead
    assert cwd == spec.plutonium.resolve()
    assert config.window_title == "Plutonium T4" and FleetConfig(client="steam").window_title == "Call of Duty"
    env = game_env(config, spec, base={"HOME": "/h"})
    assert "SteamAppId" not in env and env["UMU_RUNTIME_UPDATE"] == "0"


def test_each_instance_gets_its_own_plutonium_recopied_when_the_revision_changes(tmp_path):
    config = fake_install(tmp_path, client="plutonium")
    spec = specs(config)[0]
    said = []
    inst.prepare_plutonium(config, spec, say=said.append)
    (spec.plutonium / "storage").mkdir()
    (spec.plutonium / "storage" / "console.log").write_text("mine")
    inst.prepare_plutonium(config, spec, say=said.append)
    assert (spec.plutonium / "storage" / "console.log").exists() and len(said) == 1
    (Path(config.plutonium_dir) / "cdn_info.json").write_text('{"revision": 2}')
    inst.prepare_plutonium(config, spec, say=said.append)
    assert not (spec.plutonium / "storage").exists() and len(said) == 2


def test_the_steam_profile_is_copied_into_plutonium_with_active_txt(tmp_path, monkeypatch):
    steam_cfg = tmp_path / "steam" / "config.cfg"
    steam_cfg.parent.mkdir()
    steam_cfg.write_text('bind MOUSE2 "+speed_throw"\n')
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [steam_cfg])
    written = inst.sync_plutonium_profile(tmp_path / "pluto", say=lambda m: None)
    assert written.read_text() == steam_cfg.read_text()
    assert (written.parent.parent / "active.txt").read_text() == "$$$"


def test_offline_runs_the_game_with_loopback_only_as_ourselves():
    wrapper = inst.offline_wrapper()
    assert wrapper[:4] == ["unshare", "--user", "--map-root-user", "--net"]
    assert "ip link set lo up" in wrapper[6] and f"--map-user={os.getuid()}" in wrapper[6]
    assert wrapper[-1] == "sh"  # $0 for sh -c; the game's command follows as "$@"


def test_a_headless_weston_is_exactly_the_games_size(tmp_path, monkeypatch):
    config = fake_install(tmp_path)
    spec = specs(config)[2]
    command = inst.weston_command(spec)
    assert "--backend=headless" in command and "--shell=kiosk" in command
    assert "--width=2560" in command and "--height=1440" in command and f"--socket={spec.wayland_socket}" in command
    assert "--renderer=gl" in command  # linux-dmabuf, so Xwayland gets DRI3 and glamor
    assert inst.xwayland_command(spec, "weston").endswith("-glamor gl -noreset")
    assert inst.xwayland_command(spec, "hyprland").endswith("-shm -noreset")
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)
    assert config.resolved_host() == "hyprland"
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/weston")
    assert config.resolved_host() == "weston"


def test_a_config_installed_in_the_fleet_root_wins_over_the_steam_profile(tmp_path, monkeypatch):
    steam_cfg = tmp_path / "steam" / "config.cfg"
    steam_cfg.parent.mkdir()
    steam_cfg.write_text('seta sensitivity "9"\n')
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [steam_cfg])
    root = tmp_path / "fleet"
    assert inst.game_config(root) == steam_cfg  # nothing installed: the PC's own
    root.mkdir()
    (root / "config.cfg").write_text('seta sensitivity "2"\n')
    assert inst.game_config(root) == root / "config.cfg"
    written = inst.sync_plutonium_profile(tmp_path / "pluto", source=inst.game_config(root), say=lambda m: None)
    assert written.read_text() == 'seta sensitivity "2"\n' and steam_cfg.read_text() == 'seta sensitivity "9"\n'


def profile(prefix: Path, text: str) -> Path:
    path = prefix / inst.PREFIX_PROFILES / "$$$" / "config.cfg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_steam_client_fleet_plays_with_its_prefixes_profile_not_an_installed_config(tmp_path, monkeypatch):
    steam_cfg = tmp_path / "steam" / "config.cfg"
    steam_cfg.parent.mkdir()
    steam_cfg.write_text('seta sensitivity "9"\n')
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [steam_cfg])
    config = fake_install(tmp_path, client="steam")
    root = Path(config.root)
    root.mkdir()
    (root / "config.cfg").write_text('seta sensitivity "2"\n')  # what `fleet.py prep` would install
    assert inst.game_config(root) == root / "config.cfg"  # no fleet.json yet: it will be Plutonium's
    inst.save_fleet(config)
    template = profile(Path(config.template_prefix), 'seta sensitivity "5"\n')
    assert inst.game_config(root) == template  # no instance prefix yet: the one it will be copied from
    prepare_prefix(config, specs(config)[0], say=lambda m: None)
    copied = inst.game_config(root)
    assert copied.is_relative_to(specs(config)[0].prefix) and copied.read_text() == 'seta sensitivity "5"\n'

    from zombiesai.rl.fleet import describe, play_settings

    assert play_settings(root)["dvars"]["sensitivity"] == "5"
    me = describe(root)
    assert me["client"] == "steam" and me["settings_from"] == "prefix" and me["instances"] == 3


def test_a_plutonium_fleet_plays_with_the_installed_config(tmp_path, monkeypatch):
    monkeypatch.setattr("zombiesai.demos.game_settings.candidate_configs", lambda: [])
    config = fake_install(tmp_path, client="plutonium")
    inst.save_fleet(config)
    root = Path(config.root)
    profile(Path(config.template_prefix), 'seta sensitivity "5"\n')
    assert inst.game_config(root) is None  # Plutonium never reads the prefix's profile
    (root / "config.cfg").write_text('seta sensitivity "2"\n')
    assert inst.game_config(root) == root / "config.cfg"


def test_a_steam_client_fleet_without_any_profile_has_no_config(tmp_path):
    config = fake_install(tmp_path, client="steam")
    inst.save_fleet(config)
    (Path(config.root) / "config.cfg").write_text('seta sensitivity "2"\n')
    assert inst.game_config(config.root) is None


class FakeInstance:
    """An instance that never launches anything: up() marks its game running, its window shows at once."""

    def __init__(self, index: int, running: bool = False, log: list | None = None):
        self.spec = type("Spec", (), {"index": index})()
        self.running, self.log = running, log if log is not None else []

    def game_running(self) -> bool:
        return self.running

    def up(self) -> None:
        self.log.append(("up", self.spec.index))
        self.running = True

    def down(self) -> None:
        self.log.append(("down", self.spec.index))
        self.running = False

    def window(self):
        return type("Window", (), {"id": 0x400000 + self.spec.index})()

    def sink(self):
        log = self.log

        class Sink:
            def focus(self, window):
                log.append(("focus", window))

            def close(self):
                pass

        return Sink()


def test_bring_up_starts_what_is_missing_one_game_at_a_time():
    log: list = []
    instances = [FakeInstance(0, running=True, log=log), FakeInstance(1, log=log), FakeInstance(2, log=log)]
    slept: list = []
    assert inst.bring_up(instances, stagger_s=5.0, say=lambda m: None, sleep=slept.append) == 2
    # The running game is only checked (up() is idempotent); each new one has its window focused.
    assert log == [("up", 0), ("up", 1), ("focus", 0x400001), ("up", 2), ("focus", 0x400002)]
    assert slept  # the stagger after game 1, none after the last


def test_bring_up_gives_up_between_games_when_asked():
    log: list = []
    instances = [FakeInstance(i, log=log) for i in range(3)]
    asked = iter([False, False, True, True, True, True])
    assert inst.bring_up(instances, stagger_s=0.0, say=lambda m: None, should_stop=lambda: next(asked),
                         sleep=lambda s: None) == 1
    assert [e for e in log if e[0] == "up"] == [("up", 0)]


def test_a_window_that_never_comes_is_waited_for_then_left(tmp_path):
    from zombiesai.demos.x11_capture import WindowNotFound

    instance = FakeInstance(0)

    def no_window():
        raise WindowNotFound("not yet")

    instance.window = no_window
    now = [0.0]

    def sleep(s):
        now[0] += s

    said: list = []
    assert not inst.wait_for_window(instance, 5.0, say=said.append, sleep=sleep, clock=lambda: now[0])
    assert now[0] >= 5.0 and "no game window after 5 s" in said[0]


def test_ensure_fleet_grows_the_saved_fleet_and_starts_only_what_the_run_needs(tmp_path, monkeypatch):
    config = fake_install(tmp_path, client="plutonium")
    inst.save_fleet(replace(config, n=2, width=1920, height=1080))
    started: list = []
    log: list = []

    def fake_fleet(cfg, *, say=print):
        started.append(cfg)
        return [FakeInstance(i, running=i == 0, log=log) for i in range(cfg.n)]

    monkeypatch.setattr(inst, "fleet", fake_fleet)
    got = inst.ensure_fleet(config.root, 3, say=lambda m: None, stagger_s=0.0)
    assert got.n == 3 and inst.load_fleet(config.root).n == 3
    assert (got.width, got.height, got.client) == (1920, 1080, "plutonium")  # its own settings kept
    assert [e for e in log if e[0] == "up"] == [("up", 0), ("up", 1), ("up", 2)]
    log.clear()
    assert inst.ensure_fleet(config.root, 1, say=lambda m: None).n == 3  # never shrinks
    assert [e for e in log if e[0] == "up"] == [("up", 0)]


def test_ensure_fleet_on_a_pc_without_one_makes_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "fleet", lambda cfg, say=print: [FakeInstance(i) for i in range(cfg.n)])
    root = tmp_path / "fleet"
    got = inst.ensure_fleet(root, 2, say=lambda m: None, stagger_s=0.0)
    assert got == FleetConfig(n=2, root=str(root)) and inst.load_fleet(root) == got


def test_take_down_stops_every_game_and_counts_the_running_ones(tmp_path, monkeypatch):
    assert inst.take_down(tmp_path / "none") == 0
    config = fake_install(tmp_path)
    inst.save_fleet(config)
    log: list = []
    monkeypatch.setattr(inst, "fleet", lambda cfg, say=print: [FakeInstance(i, running=i < 2, log=log)
                                                              for i in range(cfg.n)])
    assert inst.take_down(config.root, say=lambda m: None) == 2
    assert log == [("down", 0), ("down", 1), ("down", 2)]
