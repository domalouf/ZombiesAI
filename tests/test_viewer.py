import json
import subprocess

from zombiesai.realgame import viewer
from zombiesai.realgame.viewer import Box, Screen, grid, running_screens, usable_area, viewer_command


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


def test_the_viewer_grabs_its_display_at_its_size_and_scales_to_its_window():
    command = viewer_command(Screen(":61", 2560, 1440), Box(0, 0, 1200, 675), fps=20)
    assert "-video_size 2560x1440" in command and "-framerate 20" in command and "scale=1200:675" in command
    assert command.endswith("-window_title zombiesai-view:61 -i :61")
    assert '"' not in command  # it travels inside a Lua string literal
    assert "scale=" not in viewer_command(Screen(":60", 1280, 720), Box(0, 0, 1280, 720))


def test_show_replaces_the_old_viewers_places_each_on_the_workspace_then_goes_there():
    calls = []
    base = hyprctl([])

    def run(args, **kwargs):
        calls.append(args)
        return base(args, **kwargs)

    screens = [Screen(":60", 2560, 1440), Screen(":61", 2560, 1440)]
    assert viewer.show(screens, workspace="7", run=run, say=lambda *a: None) == 2
    assert calls[0][0] == "pkill"
    dispatched = [c[2] for c in calls if c[:2] == ["hyprctl", "dispatch"]]
    assert len(dispatched) == 3
    assert "[workspace 7 silent; float; size " in dispatched[0] and dispatched[0].endswith('-i :60")')
    assert dispatched[1].endswith('-i :61")')
    assert dispatched[2] == 'hl.dsp.focus({ workspace = "7" })'
