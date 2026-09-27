import subprocess
from pathlib import Path

import pytest

from zombiesai.realgame import instances as inst
from zombiesai.realgame.instances import FleetConfig, Instance, game_args, game_env, prepare_prefix, specs


def fake_install(tmp_path: Path) -> FleetConfig:
    game = tmp_path / "game"
    game.mkdir()
    (game / "CoDWaW.exe").write_bytes(b"MZ")
    proton = tmp_path / "proton"
    proton.mkdir()
    (proton / "proton").write_text("#!/bin/sh\n")
    template = tmp_path / "compatdata"
    (template / "pfx" / "drive_c").mkdir(parents=True)
    (template / "version").write_text("9.0-100")
    return FleetConfig(n=3, display_base=70, root=str(tmp_path / "fleet"), game_dir=str(game), proton=str(proton),
                       template_prefix=str(template))


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
    assert pairs["r_mode"] == "1280x720" and pairs["monkeytoy"] == "0" and pairs["r_fullscreen"] == "1"
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
