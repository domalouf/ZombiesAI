import json
import re
import subprocess
import sys

from zombiesai.realgame import viewer
from zombiesai.realgame.viewer import (Box, FrameGate, Screen, grid, mpv_command, running_screens, usable_area,
                                      viewer_command)


def completed(stdout: str, code: int = 0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


PGREP_XWAYLAND = """\
1201 /usr/bin/Xwayland :0 -rootless -core -listenfd 55 -wm 62
41442 Xwayland :61 -geometry 2560x1440 -shm
41440 Xwayland :60 -geometry 1280x720 -shm
"""
MONITORS = [
    {"name": "DP-1", "width": 2560, "height": 1440, "scale": 1.0, "transform": 0, "reserved": [0, 26, 0, 0],
     "focused": True},
    {"name": "HDMI-A-1", "width": 3840, "height": 2160, "scale": 2.0, "transform": 1, "reserved": [0, 0, 0, 0],
     "focused": False},
]


def hyprctl(workspaces):
    def run(args, **kwargs):
        if args[:2] == ["hyprctl", "monitors"]:
            return completed(json.dumps(MONITORS))
        if args[:2] == ["hyprctl", "workspaces"]:
            return completed(json.dumps(workspaces))
        return completed("ok")
    return run


def test_only_the_instances_rootful_servers_are_found_in_display_order():
    screens = running_screens(run=lambda args, **kw: completed(PGREP_XWAYLAND))
    assert screens == [Screen(":60", 1280, 720), Screen(":61", 2560, 1440)]


def test_the_area_is_the_workspaces_own_monitor_less_its_bars():
    assert usable_area("9", run=hyprctl([])) == Box(0, 26, 2560, 1414)  # not made yet: the focused monitor
    rotated_hidpi = usable_area("9", run=hyprctl([{"name": "9", "monitor": "HDMI-A-1"}]))
    assert rotated_hidpi == Box(0, 0, 1080, 1920)


def test_four_make_a_centred_two_by_two_grid_in_order_and_three_leave_the_last_cell_empty():
    area = Box(0, 26, 2560, 1414)
    boxes = grid(4, area, 16 / 9, gap=10)
    assert len({(b.w, b.h) for b in boxes}) == 1 and abs(boxes[0].w / boxes[0].h - 16 / 9) < 0.01
    assert boxes[0].y == boxes[1].y < boxes[2].y == boxes[3].y and boxes[0].x == boxes[2].x < boxes[1].x
    assert boxes[3].y + boxes[3].h <= area.y + area.h - 10 and boxes[1].x + boxes[1].w <= area.w - 10
    assert boxes[0].x - area.x == area.x + area.w - (boxes[1].x + boxes[1].w)  # centred
    assert grid(3, area, 16 / 9, gap=10)[:2] == boxes[:2]
    assert len(grid(5, area, 16 / 9)) == 5 and grid(1, area, 16 / 9)[0].w > boxes[0].w


def test_the_viewer_runs_this_module_with_this_python_and_nothing_a_lua_string_cannot_hold():
    command = viewer_command(Screen(":61", 2560, 1440))
    assert command.split()[0] == sys.executable
    assert command.endswith("-m zombiesai.realgame.viewer --watch :61 --title zombiesai-view:61 --fps 30")
    assert '"' not in command and re.match(viewer._PROCESS, command)


def test_a_frame_is_grabbed_once_its_bottom_strip_lands_and_a_partial_redraw_after_it_settles():
    gate = FrameGate(1440, settle_s=0.25)
    gate.damaged(0, 409, now=0.0)
    gate.damaged(409, 409, now=0.02)
    assert not gate.due(0.03)
    gate.damaged(1227, 213, now=0.06)
    assert gate.due(0.06)
    gate.shown(0.06)
    assert not gate.due(1.0)  # nothing new since
    gate.damaged(100, 50, now=1.0)
    assert not gate.due(1.2) and gate.due(1.25) and abs(gate.wait_s(1.1) - 0.15) < 1e-9


def test_whole_frames_faster_than_the_cap_wait_for_their_turn():
    gate = FrameGate(1440, min_interval_s=1 / 30)
    gate.damaged(0, 1440, now=0.0)  # glamor: one rectangle, the whole frame
    assert gate.due(0.0)
    gate.shown(0.0)
    gate.damaged(0, 1440, now=0.016)
    assert not gate.due(0.016) and abs(gate.wait_s(0.016) - (1 / 30 - 0.016)) < 1e-9
    assert gate.due(1 / 30)


def test_a_whole_frame_waiting_for_its_turn_is_not_grabbed_once_the_next_one_starts_landing():
    gate = FrameGate(1440, min_interval_s=0.1)
    gate.damaged(1227, 213, now=0.0)
    gate.shown(0.0)
    gate.damaged(0, 409, now=0.01)
    gate.damaged(1227, 213, now=0.05)  # whole, but too soon
    gate.damaged(0, 409, now=0.08)  # the next frame's top strip: grabbing now would tear
    assert not gate.due(0.1)
    gate.damaged(1227, 213, now=0.12)
    assert gate.due(0.12)


def test_mpv_reads_raw_frames_of_the_screens_size_and_shows_them_untimed():
    command = mpv_command(2560, 1440, "zombiesai-view:60")
    assert "--untimed" in command and "--demuxer-rawvideo-w=2560" in command and "--demuxer-rawvideo-h=1440" in command
    assert "--title=zombiesai-view:60" in command and command[-1] == "-"


def test_show_replaces_the_old_viewers_places_each_on_the_workspace_then_goes_there():
    calls = []
    base = hyprctl([])

    def run(args, **kwargs):
        calls.append(args)
        return base(args, **kwargs)

    screens = [Screen(":60", 2560, 1440), Screen(":61", 2560, 1440)]
    assert viewer.show(screens, workspace="7", run=run, say=lambda *a: None) == 2
    assert calls[0] == ["pkill", "-f", viewer._PROCESS]
    dispatched = [c[2] for c in calls if c[:2] == ["hyprctl", "dispatch"]]
    assert len(dispatched) == 3
    assert "[workspace 7 silent; float; size " in dispatched[0] and "--watch :60 " in dispatched[0]
    assert "--watch :61 " in dispatched[1]
    assert dispatched[2] == 'hl.dsp.focus({ workspace = "7" })'


def test_the_agent_view_is_the_policy_frame_in_whole_pixels_in_the_top_right_corner():
    import numpy as np

    from zombiesai.demos import frames as fr
    from zombiesai.realgame.viewer import AgentInset

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (1440, 2560, 4), dtype=np.uint8)
    seen = fr.to_policy_frame(frame, channels=fr.BGRX)
    inset = AgentInset(1440, 2560)
    out = inset.draw(frame.copy())
    assert inset.scale == 5 and (inset.h, inset.w) == (360, 640)
    assert inset.x + inset.w + inset.BORDER + inset.MARGIN == 2560
    drawn = out[inset.y:inset.y + inset.h, inset.x:inset.x + inset.w, 2::-1]  # BGRX back to RGB
    assert np.array_equal(drawn[::inset.scale, ::inset.scale], seen)
    assert np.array_equal(drawn, seen.repeat(5, axis=0).repeat(5, axis=1))
    assert (out[inset.y - 1, inset.x:inset.x + inset.w, :3] == 255).all()  # the border
    assert np.array_equal(out[inset.bottom:], frame[inset.bottom:])  # nothing below it touched


def test_a_read_only_grab_is_drawn_on_a_copy():
    import numpy as np

    from zombiesai.realgame.viewer import AgentInset

    frame = np.zeros((720, 1280, 4), dtype=np.uint8)
    frame.flags.writeable = False
    out = AgentInset(720, 1280).draw(frame)
    assert out is not frame and not frame.any() and out.any()


def test_with_a_socket_mpv_listens_for_the_overlay_and_without_one_draws_nothing():
    assert "--input-ipc-server=/run/v.sock" in mpv_command(2560, 1440, "t", "/run/v.sock")
    assert "--osd-level=0" in mpv_command(2560, 1440, "t")


def test_the_stream_panel_shows_the_overlays_four_numbers_as_it_formats_them():
    from zombiesai.realgame.viewer import stats_panel

    stats = {"games": 1234, "window": 100, "best_round": 9, "avg_round": 4.6, "accuracy": 0.0318,
             "avg_survival_s": 432, "avg_points": 1234.5, "avg_kills": 7.25}
    events = "\n".join(stats_panel(stats, "rl9", True, 1440))
    for text in ("ZOMBIES AI \u00b7 PPO \u00b7 rl9", "TRAINING LIVE", "}9", "}4.6", "}3.2%", "}7:12", "}1,234", "}7.2", "Avg kills",
                 "last 100 of 1,234 games"):
        assert text in events
    assert "TRAINING LIVE" not in "\n".join(stats_panel(stats, "rl9", False, 1440))
    assert "Waiting for the first game" in "\n".join(stats_panel(None, None, False, 1440))


def test_the_hud_marks_say_what_the_parser_read_or_why_not():
    from zombiesai.hud.parse import HudReading
    from zombiesai.realgame.viewer import hud_marks

    boxes = {"points_ammo": (2180, 1170, 380, 270), "round": (0, 1160, 320, 280)}
    events = "\n".join(hud_marks(boxes, HudReading(points=500, points_status=0, round=3, round_status=0), 1440))
    assert "reward reads: 500 pts" in events and "round 3" in events
    events = "\n".join(hud_marks(boxes, HudReading(points_status=2, round_status=1), 1440))
    assert "reward reads: ?" in events and "round \u2013" in events


def test_the_gun_label_says_what_is_held_and_how_much_is_left():
    from zombiesai.hud.parse import MAG_AT_LEAST, HudReading
    from zombiesai.hud.weapons import WEAPON_INDEX
    from zombiesai.realgame.viewer import gun_text

    colt = HudReading(weapon=WEAPON_INDEX["colt"], weapon_status=0, mag=6, mag_status=0, reserve=32,
                      reserve_status=0, grenades=0, grenades_status=0)
    assert gun_text(colt) == "Colt M1911 \u00b7 6/32 \u00b7 0 grenades"
    kar = HudReading(weapon=WEAPON_INDEX["kar98k"], weapon_status=0, mag=4, mag_status=0, mag_flags=MAG_AT_LEAST,
                     reserve=50, reserve_status=0, grenades_status=2)
    assert gun_text(kar) == "Kar98k \u00b7 4+/50 \u00b7 ? grenades"
    assert gun_text(HudReading()) == "\u2013 \u00b7 \u2013/\u2013 \u00b7 \u2013 grenades"


def test_text_from_outside_cannot_open_an_ass_tag():
    from zombiesai.realgame.viewer import _ass

    assert "{" not in _ass("{\\an5}x") and "\\" not in _ass("a\\b")


def test_the_sound_is_the_instances_sink_monitor_and_dies_with_the_viewer():
    from zombiesai.realgame.viewer import loopback_command

    command = loopback_command("zombiesai_2", "zombiesai-view:62")
    assert command[0] == "pw-loopback" and "target.object=zombiesai_2 stream.capture.sink=true" in " ".join(command)
    assert command[-1] == 'media.name="zombiesai-view:62"'
    command = viewer_command(Screen(":62", 2560, 1440), run_dir="/r/rl9", sink="zombiesai_2")
    assert command.endswith("--fps 30 --run /r/rl9 --sink zombiesai_2") and '"' not in command


def test_the_viewer_command_says_so_only_when_the_agent_view_is_off():
    assert viewer_command(Screen(":61", 2560, 1440), agent_view=False).endswith("--fps 30 --no-agent-view")
