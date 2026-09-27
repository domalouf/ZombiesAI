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
