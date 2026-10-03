import os
from pathlib import Path

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import inputs

DT = 1.0 / spec.DECISION_HZ
CONFIG = inputs.InputConfig(counts_per_degree=10.0)


def log_for(actions, config=CONFIG, dt=DT):
    return [e for k, a in enumerate(actions) for e in inputs.synthesize(a, k * dt, dt, config)]


def test_synthesize_and_quantize_round_trip_every_action():
    space = spec.factored_action_space()
    space.seed(7)
    actions = [space.sample() for _ in range(400)]
    labels = inputs.quantize(log_for(actions), 0.0, len(actions), CONFIG, DT)
    np.testing.assert_array_equal(labels.actions, np.array(actions, dtype=np.uint8))


@pytest.mark.parametrize("counts_per_degree", [1.0, 6.4, 400.0])
def test_look_labels_survive_any_sensitivity(counts_per_degree):
    config = inputs.InputConfig(counts_per_degree=counts_per_degree)
    actions = [spec.make_action(yaw=y, pitch=p) for y in spec.YAW_BINS_DEG for p in spec.PITCH_BINS_DEG]
    labels = inputs.quantize(log_for(actions, config), 0.0, len(actions), config, DT)
    np.testing.assert_array_equal(labels.actions, np.array(actions, dtype=np.uint8))
    # The raw degrees are kept so re-binning later is arithmetic, not another evening of play.
    np.testing.assert_allclose(labels.yaw_deg, [spec.YAW_BINS_DEG[a[spec.YAW]] for a in actions], atol=0.6)


def test_a_key_held_for_less_than_half_a_decision_is_not_held():
    events = [
        {"t": 0.0, "type": "key", "code": "w", "down": True},
        {"t": 0.4 * DT, "type": "key", "code": "w", "down": False},
        {"t": DT, "type": "key", "code": "w", "down": True},
        {"t": 1.9 * DT, "type": "key", "code": "w", "down": False},
    ]
    labels = inputs.quantize(events, 0.0, 2, CONFIG, DT)
    assert labels.actions[0][spec.FORWARD] == spec.FORWARD_VALUES.index(0)
    assert labels.actions[1][spec.FORWARD] == spec.FORWARD_VALUES.index(1)
    assert inputs.label_confidence(labels, CONFIG)[0] < inputs.label_confidence(labels, CONFIG)[1]


def test_a_key_held_across_several_decisions_stays_held():
    events = [
        {"t": -5.0, "type": "key", "code": "w", "down": True},
        {"t": 10.0, "type": "key", "code": "w", "down": False},
    ]
    labels = inputs.quantize(events, 0.0, 5, CONFIG, DT)
    assert (labels.actions[:, spec.FORWARD] == spec.FORWARD_VALUES.index(1)).all()


def test_the_most_important_button_in_a_window_wins():
    events = [
        {"t": 0.1 * DT, "type": "key", "code": "r", "down": True},
        {"t": 0.2 * DT, "type": "key", "code": "r", "down": False},
        {"t": 0.3 * DT, "type": "key", "code": "f", "down": True},
        {"t": 0.4 * DT, "type": "key", "code": "f", "down": False},
    ]
    labels = inputs.quantize(events, 0.0, 1, CONFIG, DT)
    assert spec.BUTTONS[labels.actions[0][spec.BUTTON]] == "use"


def wheel(t, code="wheeldown"):
    return [{"t": t, "type": "button", "code": code, "down": d} for d in (True, False)]


def test_a_wheel_notch_is_a_swap_press_like_the_swap_key():
    labels = inputs.quantize(wheel(0.5 * DT) + wheel(1.5 * DT, "wheelup"), 0.0, 3, CONFIG, DT)
    assert [spec.BUTTONS[a[spec.BUTTON]] for a in labels.actions] == ["swap", "swap", "none"]
    assert labels.presses[0][inputs.CONTROL_INDEX["swap"]] == 1


def test_several_notches_in_one_window_are_one_swap_label():
    """The head can say a swap happened, not how many; the raw count is kept in `presses`."""
    labels = inputs.quantize(wheel(0.2 * DT) + wheel(0.5 * DT) + wheel(0.8 * DT), 0.0, 1, CONFIG, DT)
    assert spec.BUTTONS[labels.actions[0][spec.BUTTON]] == "swap"
    assert labels.presses[0][inputs.CONTROL_INDEX["swap"]] == 3


def test_a_wheel_swap_still_loses_to_a_more_important_press():
    events = wheel(0.1 * DT) + [
        {"t": 0.3 * DT, "type": "key", "code": "r", "down": True},
        {"t": 0.4 * DT, "type": "key", "code": "r", "down": False},
    ]
    assert spec.BUTTONS[inputs.quantize(events, 0.0, 1, CONFIG, DT).actions[0][spec.BUTTON]] == "reload"


def test_scrolling_while_the_swap_key_is_held_does_not_release_it():
    """Two codes share one control; a notch has no duration and must not end the key's hold."""
    events = [{"t": 0.0, "type": "key", "code": "q", "down": True}] + wheel(0.2 * DT)
    labels = inputs.quantize(events, 0.0, 2, CONFIG, DT)
    assert labels.held[1][inputs.CONTROL_INDEX["swap"]] == pytest.approx(1.0)
    assert labels.presses[0][inputs.CONTROL_INDEX["swap"]] == 2


def test_the_inverse_map_emits_the_swap_key_not_the_wheel_whatever_the_order():
    """The virtual device has no wheel, so the key must win even when a config lists the wheel first."""
    assert inputs.inverse_bindings(inputs.DEFAULT_BINDINGS)["swap"] == "q"
    assert inputs.inverse_bindings({"wheeldown": "swap", "wheelup": "swap", "1": "swap"})["swap"] == "1"
    # With nothing else bound the wheel is still the answer, rather than no answer at all.
    assert inputs.inverse_bindings({"wheelup": "swap"})["swap"] == "wheelup"


def test_a_wheel_only_swap_still_round_trips_through_synthesize():
    bindings = {c: b for c, b in inputs.DEFAULT_BINDINGS.items() if c != "q"}
    config = inputs.InputConfig(counts_per_degree=10.0, bindings=bindings)
    events = inputs.synthesize(spec.make_action(button="swap"), 0.0, DT, config)
    assert [(e["type"], e["code"]) for e in events][0] == ("button", "wheeldown")
    labels = inputs.quantize(events, 0.0, 1, config, DT)
    assert spec.BUTTONS[labels.actions[0][spec.BUTTON]] == "swap"


def test_the_shipped_waw_bindings_cycle_weapons_on_the_wheel():
    import json
    from pathlib import Path

    bindings = json.loads((Path(__file__).parents[1] / "configs" / "waw_bindings.json").read_text())
    assert bindings["wheeldown"] == bindings["wheelup"] == bindings["1"] == "swap"
    assert inputs.inverse_bindings(bindings)["swap"] == "1"


def test_a_flick_wider_than_the_widest_bin_is_clamped_and_flagged():
    events = [{"t": 0.5 * DT, "type": "mouse", "dx": int(180 * CONFIG.counts_per_degree), "dy": 0}]
    labels = inputs.quantize(events, 0.0, 1, CONFIG, DT)
    assert labels.actions[0][spec.YAW] == len(spec.YAW_BINS_DEG) - 1
    assert labels.clamped[0] and labels.yaw_deg[0] == pytest.approx(180.0)
    assert inputs.label_confidence(labels, CONFIG)[0] <= 0.5


def test_raw_mouse_dy_counts_down_but_pitch_looks_up():
    labels = inputs.quantize([{"t": 0.0, "type": "mouse", "dx": 0, "dy": -60}], 0.0, 1, CONFIG, DT)
    assert spec.PITCH_BINS_DEG[labels.actions[0][spec.PITCH]] > 0


def test_unbound_keys_are_ignored():
    labels = inputs.quantize([{"t": 0.0, "type": "key", "code": "F5", "down": True}], 0.0, 1, CONFIG, DT)
    np.testing.assert_array_equal(labels.actions[0], np.array(spec.NEUTRAL_ACTION, dtype=np.uint8))


def test_read_log_survives_a_truncated_last_line(tmp_path):
    path = tmp_path / "inputs.jsonl"
    path.write_text('{"t": 1.0, "type": "key", "code": "w", "down": true}\n{"t": 2.0, "type": "ke')
    assert len(inputs.read_log(path)) == 1


def turning_video(turns, lag=0, seed=1):
    """Frames of a scene panning by `turns`, with the response delayed by `lag` decisions."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, size=(72, 1024, 3), dtype=np.uint8)
    position, video = 300.0, []
    for step in range(len(turns) + lag + 2):
        video.append(scene[:, int(position) : int(position) + 128])
        position += turns[step - lag] if lag <= step < len(turns) + lag else 0.0
    return np.stack(video)


def random_turns(seed, size=120, choices=(-6.0, -3.0, 3.0, 6.0)):
    return np.random.default_rng(seed).choice(choices, size=size)


def test_yaw_labels_are_checked_against_the_pixels():
    turns = random_turns(2)
    report = inputs.yaw_flow_agreement(turns, turning_video(turns))
    assert report["verdict"] == "ok" and report["reasons"] == []
    assert report["rank_correlation"] > 0.9 and report["lag"] == 0
    # The synthetic pan moves one pixel per "degree", which on a 128-px frame is a 96-degree lens.
    assert report["px_per_deg"] == pytest.approx(1.0, abs=0.05)
    assert report["fov_deg"] == pytest.approx(96.3, abs=2.0)


def test_the_check_reports_the_delay_between_command_and_response():
    """The best-fitting lag is the closed-loop delay, read straight off the recording."""
    turns = random_turns(3)
    video = turning_video(turns, lag=2)
    report = inputs.yaw_flow_agreement(turns, video, expected_lag=2)
    assert report["verdict"] == "ok" and report["lag"] == 2 and report["rank_correlation"] > 0.9
    assert report["by_lag"][0] < report["by_lag"][2]
    # The same pixels, when the recording should have answered at once: that is a timing bug.
    assert inputs.yaw_flow_agreement(turns, video)["verdict"] == "misaligned"


@pytest.mark.parametrize("steps", [-3, -2, -1, 1, 2, 3])
def test_labels_slipped_against_their_frames_are_called_misaligned(steps):
    turns = random_turns(5)
    video = turning_video(turns)
    slipped = np.roll(turns, steps)
    report = inputs.yaw_flow_agreement(slipped, video)
    assert report["verdict"] == "misaligned" and report["lag"] == -steps
    assert report["lag_confidence"] >= inputs.FLOW_CHECK.lag_confidence
    # A slip moves the peak of the lag curve; it does not lower it -- which is why a correlation threshold
    # alone could not tell a slipped log from a noisy one.
    assert report["rank_correlation"] > 0.9


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_a_wrong_counts_per_degree_is_caught_by_the_implied_field_of_view(scale):
    """Every turn scaled by the same factor ranks exactly as before, so only the pixels-per-degree see it."""
    turns = random_turns(6)
    video = turning_video(turns)
    report = inputs.yaw_flow_agreement(turns * scale, video)
    assert report["rank_correlation"] > 0.9 and report["lag"] == 0
    assert report["verdict"] == "wrong_scale"
    assert report["px_per_deg"] == pytest.approx(1.0 / scale, rel=0.1)
    assert "counts_per_degree" in report["reasons"][0]


def test_a_handful_of_wrong_matches_does_not_sink_a_good_recording():
    """Real footage: fog, zombies and the gun make the SAD search lock onto nonsense on some steps. A rank
    and a median shrug that off where Pearson on the raw shifts collapsed to 0.5 (demo_0000)."""
    turns = random_turns(7, size=200)
    video = turning_video(turns).copy()
    rng = np.random.default_rng(8)
    for step in rng.choice(len(turns), size=30, replace=False):  # 15% of steps: the next frame is noise
        video[step + 1] = rng.integers(0, 255, size=video[step + 1].shape, dtype=np.uint8)
    report = inputs.yaw_flow_agreement(turns, video)
    assert report["verdict"] == "ok" and report["lag"] == 0
    assert report["fov_deg"] == pytest.approx(96.3, abs=3.0)


def test_labels_that_belong_to_another_recording_do_not_correlate():
    turns = random_turns(4)
    video = turning_video(turns)
    unrelated = np.random.default_rng(4).permutation(turns)
    report = inputs.yaw_flow_agreement(unrelated, video)
    assert report["verdict"] == "no_signal" and report["rank_correlation"] < 0.5


def test_a_flipped_yaw_sign_is_named_as_such():
    turns = random_turns(9)
    report = inputs.yaw_flow_agreement(-turns, turning_video(turns))
    assert report["verdict"] == "no_signal" and "sign is flipped" in report["reasons"][0]


def test_a_recording_with_no_turning_in_it_says_so_rather_than_inventing_a_number():
    still = np.zeros(40)
    report = inputs.yaw_flow_agreement(still, turning_video(still))
    assert report["verdict"] == "too_little_turning"
    assert report["lag"] is None and np.isnan(report["rank_correlation"])


def test_the_implied_field_of_view_inverts_the_pinhole_model():
    for fov in (60.0, 80.0, 96.0):
        assert inputs.implied_fov_deg(inputs._px_per_deg_at(fov)) == pytest.approx(fov)
    assert np.isnan(inputs.implied_fov_deg(0.0))


def real_demo():
    """The first real recording, if this checkout has it (data/ is not in git): the numbers in `FlowCheck`
    were chosen on it. Point ZOMBIESAI_REAL_DEMO at a clip to run against it from a worktree."""
    path = Path(os.environ.get("ZOMBIESAI_REAL_DEMO", Path(__file__).parents[1] / "data" / "demos" / "demo_0000"))
    if not (path / "clip.json").exists():
        pytest.skip(f"no real recording at {path}")
    from zombiesai.demos.clips import load_clip

    clip = load_clip(path)
    if not clip.labelled:
        pytest.skip(f"{path} has no labels")
    return clip


def test_a_real_recording_passes_and_its_corruptions_do_not():
    """Read-only: every corruption is made in memory."""
    clip = real_demo()
    yaw = np.asarray(clip.labels["yaw_deg"], dtype=np.float64)
    report = inputs.yaw_flow_agreement(yaw, clip.frames)
    assert report["verdict"] == "ok" and report["lag"] == 0
    assert 75.0 < report["fov_deg"] < 100.0
    for steps in (-2, 1, 3):
        assert inputs.yaw_flow_agreement(np.roll(yaw, steps), clip.frames)["verdict"] == "misaligned"
    for scale in (0.5, 2.0):
        assert inputs.yaw_flow_agreement(yaw * scale, clip.frames)["verdict"] == "wrong_scale"


def test_firing_is_checked_against_the_magazine():
    fire = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    mag = np.array([32, 31, 30, 29, 29, 29, 28, 27])
    assert inputs.fire_ammo_agreement(fire, mag)["correlation"] > 0.9
    assert inputs.fire_ammo_agreement(fire, mag)["shots_while_not_firing"] == 0
    assert inputs.fire_ammo_agreement(fire[::-1], mag)["correlation"] < 0.5


def test_a_mark_key_that_is_also_a_control_is_refused():
    with pytest.raises(ValueError, match="mark key"):
        inputs.InputConfig(bindings={**inputs.DEFAULT_BINDINGS, "f8": "use"})
    assert inputs.InputConfig(mark_key="F7").mark_key == "f7"


def test_the_default_mark_key_is_free_in_every_shipped_binding_map():
    import json
    from pathlib import Path

    shipped = json.loads((Path(__file__).parents[1] / "configs" / "waw_bindings.json").read_text())
    for bindings in (inputs.DEFAULT_BINDINGS, shipped):
        inputs.InputConfig(bindings=bindings)  # raises if the mark key were bound


def test_the_recorder_and_the_agents_hands_name_the_mark_key_the_same_way():
    from zombiesai.demos import evdev_input
    from zombiesai.realgame.uinput import code_for

    # KEY_F8 is 66 in linux/input-event-codes.h: the physical F8 is what the recorder hears as its mark key,
    # and the key the virtual device would press by that name.
    assert evdev_input.key_name(66) == inputs.MARK_KEY and code_for(inputs.MARK_KEY) == 66


def test_not_playing_marks_from_the_window_of_one_press_to_the_window_of_the_next():
    events = log_for([spec.NEUTRAL_ACTION] * 10) + [
        {"t": 2.5 * DT, "type": "key", "code": "f8", "down": True},
        {"t": 2.6 * DT, "type": "key", "code": "f8", "down": False},
        {"t": 5.0 * DT, "type": "key", "code": "f8", "down": True},  # exactly on a deadline: belongs to step 5
        {"t": 5.1 * DT, "type": "key", "code": "f8", "down": False},
        {"t": 8.2 * DT, "type": "key", "code": "f8", "down": True},  # a double tap inside one window
        {"t": 8.3 * DT, "type": "key", "code": "f8", "down": False},
        {"t": 8.5 * DT, "type": "key", "code": "f8", "down": True},
        {"t": 8.6 * DT, "type": "key", "code": "f8", "down": False},
    ]
    marked = inputs.not_playing(events, 0.0, 10, CONFIG, DT)
    np.testing.assert_array_equal(np.flatnonzero(marked), [2, 3, 4, 5, 8])
    off = inputs.not_playing(events, 0.0, 10, inputs.InputConfig(counts_per_degree=10.0, mark_key=None), DT)
    assert not off.any()


def test_keys_and_mouse_under_the_compositors_super_key_are_not_labels():
    """Super+1 switches Hyprland workspace; the game never sees the 1, which is bound to weapon swap."""
    config = inputs.InputConfig(counts_per_degree=10.0, bindings={"1": "swap", "w": "forward"})
    folder = inputs.InputFolder(config)
    events = [
        {"t": 0.00, "type": "key", "code": "w", "down": True},  # held before Super: keeps its hold
        {"t": 0.01, "type": "key", "code": "super", "down": True},
        {"t": 0.02, "type": "key", "code": "1", "down": True},
        {"t": 0.03, "type": "mouse", "dx": 500, "dy": 0},  # Super+drag moves a window
        {"t": 0.04, "type": "key", "code": "1", "down": False},
        {"t": 0.05, "type": "key", "code": "super", "down": False},
    ]
    held, presses, counts = folder.feed(events, 0.0, 0.066)
    assert presses[inputs.CONTROL_INDEX["swap"]] == 0
    assert held[inputs.CONTROL_INDEX["forward"]] == 1.0
    assert counts[0] == 0
    # ...and once Super is up, the same key is a swap again.
    later = [{"t": 0.07, "type": "key", "code": "1", "down": True}, {"t": 0.08, "type": "key", "code": "1", "down": False}]
    _, presses, _ = folder.feed(later, 0.066, 0.133)
    assert presses[inputs.CONTROL_INDEX["swap"]] == 1


def test_logs_from_before_super_had_a_name_are_read_the_same():
    config = inputs.InputConfig(counts_per_degree=10.0, bindings={"1": "swap"})
    folder = inputs.InputFolder(config)
    events = [
        {"t": 0.00, "type": "key", "code": "key125", "down": True},
        {"t": 0.01, "type": "key", "code": "1", "down": True},
        {"t": 0.02, "type": "key", "code": "1", "down": False},
        {"t": 0.03, "type": "key", "code": "key125", "down": False},
    ]
    _, presses, _ = folder.feed(events, 0.0, 0.066)
    assert presses.sum() == 0
