import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from zombiesai.demos.clips import load_clip
from zombiesai.demos.inputs import InputConfig
from zombiesai.demos.recorder import RecorderConfig, record
from zombiesai.synthetic import SyntheticSource
from zombiesai.viz import input_layout as L
from zombiesai.viz.input_video import build_ass, key_intervals, merge, render, runs, step_motion

ROOT = Path(__file__).resolve().parents[1]
DT = 1 / 15


def load_overlay_script():
    spec = importlib.util.spec_from_file_location("record_overlay", ROOT / "scripts" / "record_overlay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------ the layout


def test_no_two_caps_overlap_and_every_bound_control_has_a_cap():
    caps = L.ALL_CAPS
    for i, a in enumerate(caps):
        for b in caps[i + 1:]:
            apart = a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
            assert apart, (a, b)
    codes = {c.code for c in caps}
    bindings = json.loads((ROOT / "configs" / "waw_bindings.json").read_text())
    assert set(bindings) <= codes
    assert set(json.loads((ROOT / "configs" / "pad_bindings.json").read_text())["buttons"].values()) <= codes | {
        "escape", "f8"}  # the pad's menu keys are not drawn on the keyboard block


def test_controls_are_named_short_enough_to_fit_under_a_cap():
    named = L.control_of({"mouse2": "ads", "g": "grenade", "x": "made-up"})
    assert named == {"mouse2": "aim", "g": "nade", "x": "made-up"}


def test_the_look_meter_points_where_the_mouse_went_and_saturates_at_the_rim():
    x, y, speed = L.look_vector(dx=90, dy=0, seconds=0.1, counts_per_degree=9.0)  # 100 deg/s to the right
    assert speed == pytest.approx(100)
    assert x > 0 and y == 0 and x < 1
    x, y, _ = L.look_vector(dx=0, dy=-10_000, seconds=0.1, counts_per_degree=9.0)
    assert (x, y) == pytest.approx((0, -1))
    assert L.look_vector(0, 0, 0.1, 9.0) == (0.0, 0.0, 0.0)


def test_the_live_state_folds_presses_releases_wheel_notches_and_motion():
    now = [100.0]
    state = L.InputState(clock=lambda: now[0])
    assert state.feed([{"type": "key", "code": "w", "down": True},
                       {"type": "button", "code": "wheelup", "down": True},
                       {"type": "button", "code": "wheelup", "down": False},
                       {"type": "mouse", "dx": 45, "dy": 0}])
    assert state.lit() == {"w", "wheelup"}
    assert state.look(9.0)[2] == pytest.approx(50)
    now[0] += 0.2  # the notch's flash and the motion window have both passed
    assert state.lit() == {"w"}
    assert state.look(9.0) == (0.0, 0.0, 0.0)
    assert state.feed([{"type": "key", "code": "w", "down": False}])
    assert not state.feed([{"type": "key", "code": "w", "down": False}])  # nothing new to draw
    assert state.lit() == set()


# ------------------------------------------------------------------------------------------------ the video


def test_key_intervals_are_in_video_seconds_and_survive_a_missing_release():
    t0 = 1000.0
    events = [
        {"t": 1000.5, "type": "key", "code": "w", "down": True},
        {"t": 1001.0, "type": "key", "code": "w", "down": False},
        {"t": 1002.0, "type": "button", "code": "mouse1", "down": True},  # never released
        {"t": 1001.5, "type": "button", "code": "wheeldown", "down": True},
        {"t": 999.0, "type": "key", "code": "r", "down": True},  # held from before the recording began
        {"t": 1000.2, "type": "key", "code": "r", "down": False},
    ]
    spans = key_intervals(sorted(events, key=lambda e: e["t"]), t0, end_s=4.0)
    assert spans["w"] == [(0.5, 1.0)]
    assert spans["mouse1"] == [(2.0, 4.0)]
    assert spans["wheeldown"] == [(1.5, 1.5 + L.WHEEL_FLASH_S)]
    assert spans["r"] == [(0.0, pytest.approx(0.2))]


def test_spans_merge_and_runs_find_stretches():
    assert merge([(2, 3), (0, 1), (0.5, 1.5)]) == [(0, 1.5), (2, 3)]
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_mouse_motion_is_summed_over_the_recorders_own_decision_windows():
    events = [{"t": 10.0 + DT * 0.5, "type": "mouse", "dx": 3, "dy": 1},
              {"t": 10.0 + DT * 0.9, "type": "mouse", "dx": 2, "dy": 0},
              {"t": 10.0 + DT * 1.1, "type": "mouse", "dx": -7, "dy": 0},
              {"t": 9.0, "type": "mouse", "dx": 99, "dy": 99}]
    motion = step_motion(events, 10.0, 3, DT)
    assert motion.tolist() == [[5, 1], [-7, 0], [0, 0]]


def test_the_script_lights_each_key_exactly_while_it_was_down():
    events = [{"t": 5.5, "type": "key", "code": "w", "down": True},
              {"t": 6.25, "type": "key", "code": "w", "down": False}]
    ass = build_ass(events, 5.0, 45, bindings={"w": "forward"}, counts_per_degree=9.0,
                    not_playing=np.r_[np.zeros(30, bool), np.ones(15, bool)], title="demo_0009 · test")
    amber = "&H4DB0DD&"
    lit = [line for line in ass.splitlines() if line.startswith("Dialogue: 4,") and amber in line]
    assert any(",0:00:00.50,0:00:01.25," in line for line in lit)
    assert "NOT PLAYING (F8)" in ass
    banner = next(line for line in ass.splitlines() if "NOT PLAYING" in line)
    assert ",0:00:02.00,0:00:03.00," in banner
    assert "demo_0009" in ass


def test_a_cut_is_timed_from_its_own_start():
    events = [{"t": 12.0, "type": "key", "code": "w", "down": True},
              {"t": 13.0, "type": "key", "code": "w", "down": False}]
    ass = build_ass(events, 0.0, 300, bindings={"w": "forward"}, counts_per_degree=9.0, start=10.0, end=15.0)
    amber = "&H4DB0DD&"
    lit = [line for line in ass.splitlines() if line.startswith("Dialogue: 4,") and amber in line]
    assert any(",0:00:02.00,0:00:03.00," in line for line in lit)
    ends = [re.match(r"Dialogue: \d+,[^,]+,([^,]+),", line)[1] for line in ass.splitlines()
            if line.startswith("Dialogue")]
    assert max(ends) <= "0:00:05.00"


def test_a_recorded_demo_renders_with_its_inputs_burned_in(tmp_path):
    source = SyntheticSource(seed=0, max_steps=45)
    config = RecorderConfig(max_steps=45, realtime=False, input=InputConfig(counts_per_degree=10.0), notes="smoke")
    path = record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0)
    out = render(path, tmp_path / "inputs.mp4", say=lambda _m: None)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames",
                            "-of", "json", str(out)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (1280, 720)
    assert int(stream["nb_frames"]) == load_clip(path).n_steps
    assert not (tmp_path / "inputs.ass").exists()


# ------------------------------------------------------------------------------------------------ the live overlay


def test_the_status_panel_says_what_the_recorder_is_doing():
    overlay = load_overlay_script()
    tone, parts, _ = overlay.status_lines(None, 0.0)
    assert tone == "gone"
    tone, parts, sub = overlay.status_lines({"state": "recording", "elapsed_s": 75, "limit_s": 1200,
                                             "playing": True, "mark_key": "f8"}, 0.0)
    text = "".join(p for p, _ in parts)
    assert tone == "rec" and "01:15 / 20:00" in text and "PLAYING" in text and "F8" in sub
    tone, parts, _ = overlay.status_lines({"state": "recording", "elapsed_s": 75, "playing": False,
                                           "mark_key": "f8"}, 0.0)
    assert tone == "idle" and "press F8" in "".join(p for p, _ in parts)
    tone, parts, _ = overlay.status_lines({"state": "recording", "elapsed_s": 75, "stale": True}, 0.0)
    assert tone == "warn" and "CAPTURE LOST" in "".join(p for p, _ in parts)
    assert overlay.status_lines({"state": "waiting"}, 0.0)[0] == "wait"
    assert overlay.clock_text(3725) == "1:02:05"


def test_the_recorder_reports_every_step_for_the_overlay(tmp_path):
    source = SyntheticSource(seed=0, max_steps=20)
    config = RecorderConfig(max_steps=20, realtime=False, input=InputConfig(counts_per_degree=10.0))
    seen = []
    record(source, source, tmp_path / "demo", config, stop=lambda: source.done, progress_every=0, on_step=seen.append)
    assert [s["step"] for s in seen] == list(range(1, len(seen) + 1))
    assert seen[-1]["elapsed_s"] == pytest.approx(len(seen) / 15)
    assert {"playing", "stale", "overruns", "stale_steps"} <= set(seen[0])
