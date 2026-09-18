"""The agent's hands: what a factored action turns into, and whether a recording reads it back unchanged."""

import os

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos import evdev_input as ev
from zombiesai.demos.inputs import InputConfig, quantize
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig, FakeSink
from zombiesai.realgame.uinput import UinputDevice, code_for

DT = 1.0 / spec.DECISION_HZ


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def dispatcher(counts_per_degree=10.0, **kwargs):
    clock = Clock()
    sink = FakeSink()
    return ActionDispatcher(sink, DispatchConfig(counts_per_degree=counts_per_degree, **kwargs), clock=clock), sink, clock


def codes(sink, kind="key"):
    return [(e["code"], e["down"]) for e in sink.events if e["type"] in ("key", "button")]


def test_a_held_control_is_pressed_once_and_released_once():
    """fire, ads and sprint are hold states: the dispatcher diffs against what is already held."""
    d, sink, clock = dispatcher()
    for step in range(3):
        clock.t = step * DT
        d.apply(spec.make_action(forward=1, fire=1), now=clock.t, dt=DT)
    assert sorted(codes(sink)) == [("mouse1", True), ("w", True)]  # pressed once each, not once per tick
    clock.t = 3 * DT
    d.apply(spec.make_action(), now=clock.t, dt=DT)
    assert sorted(codes(sink)[2:]) == [("mouse1", False), ("w", False)]
    assert d.held == set()


def test_movement_reverses_without_pressing_both_keys():
    d, sink, clock = dispatcher()
    d.apply(spec.make_action(forward=1), now=0.0, dt=DT)
    clock.t = DT
    d.apply(spec.make_action(forward=-1), now=DT, dt=DT)
    assert codes(sink) == [("w", True), ("w", False), ("s", True)]


def test_a_button_is_a_tap_that_gets_released_by_pumping():
    d, sink, clock = dispatcher()
    d.apply(spec.make_action(button="reload"), now=0.0, dt=DT)
    assert codes(sink) == [("r", True)]
    clock.t = 0.02
    d.pump(clock.t)
    assert codes(sink) == [("r", True)], "released before the engine could see the press"
    clock.t = d.config.tap_hold_s + 0.001
    d.pump(clock.t)
    assert codes(sink) == [("r", True), ("r", False)]


def test_the_same_button_twice_running_is_two_edges():
    d, sink, clock = dispatcher()
    d.apply(spec.make_action(button="use"), now=0.0, dt=DT)
    clock.t = DT
    d.apply(spec.make_action(button="use"), now=DT, dt=DT)  # still held: it has to be released first
    assert codes(sink) == [("f", True), ("f", False), ("f", True)]


def test_a_turn_goes_out_as_several_sub_moves_across_the_tick():
    """One large delta is what old DirectX 9 raw-input paths clamp or drop."""
    d, sink, clock = dispatcher(counts_per_degree=10.0, submoves=3)
    d.apply(spec.make_action(yaw=30.0), now=0.0, dt=DT)
    for step in range(1, 4):
        clock.t = step * DT / 3
        d.pump(clock.t)
    moves = [e for e in sink.events if e["type"] == "mouse"]
    assert len(moves) == 3
    assert sum(m["dx"] for m in moves) == 300  # 30 degrees at 10 counts each
    assert {m["t"] for m in moves} == {0.0, DT / 3, 2 * DT / 3}


def test_rounding_residue_is_carried_instead_of_dropped():
    """At a low sensitivity a 2-degree turn is a fraction of a count; truncating every tick would lose a
    degree a second, and the policy would never learn why its aim drifts."""
    d, sink, clock = dispatcher(counts_per_degree=1.6, submoves=3)
    for step in range(15):
        clock.t = step * DT
        d.apply(spec.make_action(yaw=2.0), now=clock.t, dt=DT)
        d.pump(clock.t + DT * 0.99)
    total = sum(e["dx"] for e in sink.events if e["type"] == "mouse")
    assert total == pytest.approx(15 * 2.0 * 1.6, abs=1)


def test_pitch_is_inverted_because_mice_count_downwards():
    d, sink, _ = dispatcher()
    d.apply(spec.make_action(pitch=6.0), now=0.0, dt=DT)
    d.flush()
    assert sum(e["dy"] for e in sink.events if e["type"] == "mouse") == -60


def test_release_all_lets_go_of_everything():
    """A key left held when the process dies is a keyboard nobody can use."""
    d, sink, _ = dispatcher()
    d.apply(spec.make_action(forward=1, strafe=1, fire=1, ads=1, sprint=1), now=0.0, dt=DT)
    d.release_all()
    assert d.held == set()
    assert not [e for e in sink.events if e["type"] in ("key", "button") and e["down"] and e["t"] > 0]
    down = {e["code"] for e in sink.events if e["down"]}
    up = {e["code"] for e in sink.events if not e["down"]}
    assert down == up


@pytest.mark.parametrize("counts_per_degree", [1.0, 6.4, 400.0])
def test_what_the_agent_sends_is_what_a_recording_reads_back(counts_per_degree):
    """The symmetry the whole project rests on: the agent's output and a human's input are the same units,
    so a demonstration and a rollout are the same kind of data."""
    d, sink, clock = dispatcher(counts_per_degree=counts_per_degree)
    space = spec.factored_action_space()
    space.seed(11)
    actions = [space.sample() for _ in range(120)]
    for step, action in enumerate(actions):
        clock.t = step * DT
        d.apply(action, now=clock.t, dt=DT)
        for slice_index in range(1, 7):  # the env loop pumping in the slack it has anyway
            clock.t = step * DT + slice_index * DT / 7
            d.pump(clock.t)
    labels = quantize(sink.events, 0.0, len(actions), InputConfig(counts_per_degree=counts_per_degree), DT)
    np.testing.assert_array_equal(labels.actions, np.array(actions, dtype=np.uint8))


def test_the_kernel_encoding_round_trips_through_the_decoder(tmp_path):
    """UinputDevice writes the same struct demos/evdev_input.py reads, so what we send to the kernel and
    what we read back off a device are provably one format."""
    path = tmp_path / "uinput.bin"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT)
    device = UinputDevice(("w", "r", "mouse1"), fd=fd, create=False)
    dispatch = ActionDispatcher(device, DispatchConfig(counts_per_degree=10.0), clock=Clock())
    dispatch.apply(spec.make_action(forward=1, fire=1, yaw=6.0, button="reload"), now=0.0, dt=DT)
    dispatch.flush()
    device.close()
    os.close(fd)

    events = ev.decode_events(path.read_bytes())
    assert ("w", True) in [(e.get("code"), e.get("down")) for e in events]
    assert ("mouse1", True) in [(e.get("code"), e.get("down")) for e in events]
    assert ("r", True) in [(e.get("code"), e.get("down")) for e in events]
    assert sum(e["dx"] for e in events if e["type"] == "mouse") == 60


def test_every_default_binding_has_a_kernel_code():
    from zombiesai.demos.inputs import DEFAULT_BINDINGS

    for code in DEFAULT_BINDINGS:
        assert code_for(code) > 0
    with pytest.raises(KeyError):
        code_for("no-such-key")


def test_an_unbound_control_fails_loudly_rather_than_silently_doing_nothing():
    sink = FakeSink()
    config = DispatchConfig(bindings={"w": "forward"})  # no binding for fire
    dispatch = ActionDispatcher(sink, config, clock=Clock())
    with pytest.raises(KeyError):
        dispatch.apply(spec.make_action(fire=1), now=0.0, dt=DT)
