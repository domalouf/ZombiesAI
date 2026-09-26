import math

import numpy as np

from zombiesai import spec
from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_CLIP_START, ClipWriter, load_clip
from zombiesai.demos.inputs import DEFAULT_BINDINGS
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig, FakeSink
from zombiesai.realgame.play import HumanWatch, PlayConfig, play

DT = 1.0 / spec.DECISION_HZ
FORWARD_AND_FIRE = spec.make_action(forward=1, fire=1)


class Clock:
    """Time moves only when the loop waits for its deadline (via the dispatcher's pump_until)."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Screen:
    """Frame k is filled with k; `stale` lists the ticks (1-based) whose picture is frozen."""

    def __init__(self, stale=()):
        self.k, self.stale = 0, set(stale)
        self.last_stale, self.last_hud = False, None

    def read(self):
        self.k += 1
        self.last_stale = self.k in self.stale
        return np.full(spec.PIXELS_SHAPE, self.k % 256, np.uint8)


class Agent:
    def __init__(self):
        self.calls, self.resets = 0, 0

    def act(self, obs):
        self.calls += 1
        return np.asarray(FORWARD_AND_FIRE)

    def reset(self):
        self.resets += 1


class Focus:
    def __init__(self, unfocused=()):
        self.tick, self.unfocused = 0, set(unfocused)

    def is_focused(self):
        self.tick += 1
        return self.tick not in self.unfocused


class Hands:
    """The human's real devices: events to hand back at given ticks."""

    def __init__(self, at=None):
        self.at, self.tick = at or {}, 0

    def drain(self, start, end):
        self.tick += 1
        return self.at.get(self.tick, [])


class TimedDispatcher(ActionDispatcher):
    def __init__(self, clock):
        super().__init__(FakeSink(), DispatchConfig(counts_per_degree=9.09, bindings=dict(DEFAULT_BINDINGS)),
                         clock=clock)

    def pump_until(self, deadline, poll_s=0.002):
        self.clock.now = deadline
        self.pump(deadline)


def run(tmp_path=None, *, seconds=1.0, screen=None, focus=None, hands=None):
    clock = Clock()
    dispatcher = TimedDispatcher(clock)
    agent = Agent()
    config = PlayConfig(max_seconds=seconds)
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent") if tmp_path else None
    said = []
    summary = play(
        screen or Screen(), agent, dispatcher, focus=focus or Focus(), human=HumanWatch(hands or Hands(), config),
        config=config, writer=writer, clock=clock, say=said.append,
    )
    return summary, dispatcher, agent, said


def held_after(sink):
    held = set()
    for e in sink.events:
        if e["type"] in ("key", "button"):
            (held.add if e["down"] else held.discard)(e["code"])
    return held


def test_the_policy_drives_the_game_every_tick_while_all_is_well():
    summary, dispatcher, agent, _ = run(seconds=1.0)
    assert agent.calls == summary["acted"] == 15
    assert {"w", "mouse1"} <= {e["code"] for e in dispatcher.sink.events if e["type"] != "mouse"}
    assert summary["ended"] == "time limit"


def test_everything_is_released_on_the_way_out():
    _, dispatcher, _, _ = run(seconds=0.5)
    assert held_after(dispatcher.sink) == set()


def test_losing_focus_lets_go_at_once_and_sends_nothing_until_it_is_back():
    summary, dispatcher, agent, said = run(seconds=1.0, focus=Focus(unfocused={5, 6, 7}))
    assert summary["unfocused"] == 3 and agent.calls == 12
    assert agent.resets == 1  # a stale frame stack is not carried across the gap
    # Between the release at tick 5 and the re-press at tick 8, nothing at all was sent.
    t5, t8 = 4 * DT, 7 * DT
    during = [e for e in dispatcher.sink.events if t5 < e["t"] < t8]
    assert during == []
    released = {e["code"] for e in dispatcher.sink.events if e["t"] == t5 and not e.get("down", True)}
    assert {"w", "mouse1"} <= released
    assert any("not the focused window" in line for line in said)


def test_a_frozen_picture_pauses_the_policy():
    summary, _, agent, said = run(seconds=1.0, screen=Screen(stale={3, 4}))
    assert summary["frozen"] == 2 and agent.calls == 13
    assert any("frozen" in line for line in said)


def test_the_kill_key_on_a_real_keyboard_stops_everything():
    kill = {4: [{"t": 0, "type": "key", "code": "f9", "down": True}]}
    summary, dispatcher, agent, _ = run(seconds=5.0, hands=Hands(kill))
    assert summary["ended"].startswith("kill key") and agent.calls == 3
    assert held_after(dispatcher.sink) == set()


def test_touching_the_controls_hands_them_to_the_human_until_they_let_go():
    grab = {3: [{"t": 0, "type": "mouse", "dx": 40, "dy": 0}]}
    summary, _, agent, said = run(seconds=3.0, hands=Hands(grab))
    idle_ticks = math.ceil(PlayConfig().human_idle_s / DT)  # the partial tick is the human's too
    assert summary["human"] == idle_ticks
    assert agent.calls == 45 - idle_ticks
    assert any("you have the controls" in line for line in said)


def test_a_hand_resting_on_the_mouse_is_not_a_takeover():
    jitter = {3: [{"t": 0, "type": "mouse", "dx": 1, "dy": -1}]}
    summary, _, _, _ = run(seconds=1.0, hands=Hands(jitter))
    assert summary["human"] == 0


def test_every_tick_is_recorded_and_the_ones_not_played_are_flagged(tmp_path):
    summary, _, _, _ = run(tmp_path, seconds=1.0, focus=Focus(unfocused={5, 6}))
    clip = load_clip(tmp_path / "run")
    assert clip.n_steps == 15 and clip.label_source == "agent"
    bad = (clip.flags & FLAG_BAD_STEP) != 0
    np.testing.assert_array_equal(np.flatnonzero(bad), [4, 5])
    assert clip.flags[6] & FLAG_CLIP_START  # the policy's first step back has no usable history
    np.testing.assert_array_equal(clip.actions[0], FORWARD_AND_FIRE)
    assert clip.manifest["summary"]["unfocused"] == 2
