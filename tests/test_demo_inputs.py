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


def test_yaw_labels_are_checked_against_the_pixels():
    rng = np.random.default_rng(2)
    turns = rng.choice([-8.0, -3.0, 3.0, 8.0], size=60)
    video = turning_video(turns)
    report = inputs.yaw_flow_agreement(turns, video)
    assert report["correlation"] > 0.9 and report["lag"] == 0


def test_the_check_reports_the_delay_between_command_and_response():
    """The best-fitting lag is the closed-loop delay, read straight off the recording."""
    rng = np.random.default_rng(3)
    turns = rng.choice([-8.0, -3.0, 3.0, 8.0], size=60)
    report = inputs.yaw_flow_agreement(turns, turning_video(turns, lag=2))
    assert report["lag"] == 2 and report["correlation"] > 0.9
    assert report["by_lag"][0] < report["by_lag"][2]


def test_labels_that_belong_to_another_recording_do_not_correlate():
    rng = np.random.default_rng(4)
    turns = rng.choice([-8.0, -3.0, 3.0, 8.0], size=60)
    video = turning_video(turns)
    unrelated = rng.permutation(turns)
    assert inputs.yaw_flow_agreement(unrelated, video)["correlation"] < 0.5


def test_a_recording_with_no_turning_in_it_says_so_rather_than_inventing_a_number():
    still = np.zeros(40)
    report = inputs.yaw_flow_agreement(still, turning_video(still))
    assert report["lag"] is None and np.isnan(report["correlation"])


def test_firing_is_checked_against_the_magazine():
    fire = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    mag = np.array([32, 31, 30, 29, 29, 29, 28, 27])
    assert inputs.fire_ammo_agreement(fire, mag)["correlation"] > 0.9
    assert inputs.fire_ammo_agreement(fire, mag)["shots_while_not_firing"] == 0
    assert inputs.fire_ammo_agreement(fire[::-1], mag)["correlation"] < 0.5
