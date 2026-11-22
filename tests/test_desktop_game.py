import subprocess
from pathlib import Path

from test_instances import fake_install

from zombiesai.realgame import desktop
from zombiesai.realgame.instances import NACHT, save_fleet


def a_fleet(tmp_path: Path):
    fleet = fake_install(tmp_path, client="plutonium")
    save_fleet(fleet)
    return fleet


def test_the_desktop_game_is_the_fleets_client_and_settings_without_its_frame_cap(tmp_path):
    fleet = a_fleet(tmp_path)
    config = desktop.desktop_config(fleet.root, root=tmp_path / "desktop")
    assert (config.client, config.offline, config.n) == ("plutonium", True, 1)
    assert (config.game_dir, config.proton, config.plutonium_dir) == (fleet.game_dir, fleet.proton, fleet.plutonium_dir)
    assert (config.width, config.height) == (2560, 1440)
    assert "com_maxfps" not in config.dvars and config.dvars["cl_mouseAccel"] == "0"
    # its own sink, which a fleet shutdown (session.remove_sinks: zombiesai_*) leaves alone
    assert config.extra_env["PULSE_SINK"] == "zai_desktop" and not config.audio_sinks
    assert desktop.desktop_config(tmp_path / "no-fleet", root=tmp_path / "d", sink=None).extra_env == {}


def test_it_launches_offline_on_your_display_with_a_key_that_starts_nacht(tmp_path):
    config = desktop.desktop_config(a_fleet(tmp_path).root, root=tmp_path / "desktop")
    spec = desktop.desktop_spec(config, ":0")
    command, cwd = desktop.desktop_command(config, spec, "F9")
    assert command[0] == "unshare" and "--net" in command  # loopback only
    assert "-lan" in command and command[-4:] == ["+bind", "F9", "map", NACHT]
    assert ["+set", "r_mode", "2560x1440"] == command[command.index("r_mode") - 1:command.index("r_mode") + 2]
    assert cwd == spec.plutonium.resolve()
    env = desktop.game_env(config, spec, base={"DISPLAY": ":5", "WAYLAND_DISPLAY": "wayland-1", "HOME": "/h"})
    assert env["DISPLAY"] == ":0" and "WAYLAND_DISPLAY" not in env and env["PULSE_SINK"] == "zai_desktop"
    assert desktop.profile_config(config) == spec.plutonium / desktop.PLUTONIUM_PROFILE / "config.cfg"


class FakePactl:
    def __init__(self, modules=()):
        self.modules = dict(modules)
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append(args[1:])
        if args[1:4] == ["list", "short", "modules"]:
            out = "".join(f"{i}\t{line}\n" for i, line in self.modules.items())
            return subprocess.CompletedProcess(args, 0, out, "")
        if args[1] == "load-module":
            i = str(100 + len(self.modules))
            self.modules[i] = "\t".join([args[2], " ".join(args[3:])])
            return subprocess.CompletedProcess(args, 0, i + "\n", "")
        if args[1] == "unload-module":
            self.modules.pop(args[2])
        return subprocess.CompletedProcess(args, 0, "", "")


def test_its_sound_is_a_sink_of_its_own_looped_to_your_speakers_and_made_once():
    pactl = FakePactl({"7": "module-null-sink\tsink_name=zombiesai_0"})
    assert desktop.ensure_sound("zai_desktop", run=pactl)
    loads = [c for c in pactl.calls if c[0] == "load-module"]
    assert [c[1] for c in loads] == ["module-null-sink", "module-loopback"]
    assert "sink_name=zai_desktop" in loads[0] and "source=zai_desktop.monitor" in loads[1]
    assert desktop.ensure_sound("zai_desktop", run=pactl)
    assert len([c for c in pactl.calls if c[0] == "load-module"]) == 2
    # down removes the loopback before its source, and never a fleet's sink
    assert desktop.remove_sound("zai_desktop", run=pactl) == 2
    unloaded = [c[1] for c in pactl.calls if c[0] == "unload-module"]
    assert unloaded == ["102", "101"] and list(pactl.modules) == ["7"]


def test_no_pactl_is_no_sound_sink_not_a_crash():
    def missing(*a, **k):
        raise FileNotFoundError("pactl")
    assert not desktop.ensure_sound(run=missing) and desktop.remove_sound(run=missing) == 0
