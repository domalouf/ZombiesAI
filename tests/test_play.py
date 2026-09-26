import math

import numpy as np
import pytest

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


TOGGLE = {"t": 0, "type": "key", "code": "f7", "down": True}


class Hands:
    """The human's real devices: events to hand back at given ticks. By default the human presses the toggle
    key on the first tick, handing the policy the controls; `start=False` leaves it in standby."""

    def __init__(self, at=None, start=True):
        self.at, self.tick = dict(at or {}), 0
        if start:
            self.at[1] = [TOGGLE] + self.at.get(1, [])

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


def run(tmp_path=None, *, seconds=1.0, screen=None, focus=None, hands=None, states=None):
    clock = Clock()
    dispatcher = TimedDispatcher(clock)
    agent = Agent()
    config = PlayConfig(max_seconds=seconds)
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent") if tmp_path else None
    said = []
    summary = play(
        screen or Screen(), agent, dispatcher, focus=focus or Focus(), human=HumanWatch(hands or Hands(), config),
        config=config, writer=writer, clock=clock, say=said.append, on_state=states.append if states is not None else None,
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
    assert summary["unfocused"] == 3 and agent.calls == 15  # the limit is on time played
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
    assert summary["frozen"] == 2 and agent.calls == 15
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
    assert agent.calls == 45
    assert any("you have the controls" in line for line in said)


def test_a_hand_resting_on_the_mouse_is_not_a_takeover():
    jitter = {3: [{"t": 0, "type": "mouse", "dx": 1, "dy": -1}]}
    summary, _, _, _ = run(seconds=1.0, hands=Hands(jitter))
    assert summary["human"] == 0


def test_every_tick_is_recorded_and_the_ones_not_played_are_flagged(tmp_path):
    summary, _, _, _ = run(tmp_path, seconds=1.0, focus=Focus(unfocused={5, 6}))
    clip = load_clip(tmp_path / "run")
    assert clip.n_steps == 17 and clip.label_source == "agent"
    bad = (clip.flags & FLAG_BAD_STEP) != 0
    np.testing.assert_array_equal(np.flatnonzero(bad), [4, 5])
    assert clip.flags[6] & FLAG_CLIP_START  # the policy's first step back has no usable history
    np.testing.assert_array_equal(clip.actions[0], FORWARD_AND_FIRE)
    assert clip.manifest["summary"]["unfocused"] == 2


def test_it_does_nothing_at_all_until_handed_the_controls(tmp_path):
    """Standby: no frames grabbed, no input sent, nothing written -- until the toggle key."""
    quit_later = {20: [{"t": 0, "type": "key", "code": "f9", "down": True}]}
    screen = Screen()
    summary, dispatcher, agent, said = run(tmp_path, screen=screen, hands=Hands(quit_later, start=False))
    assert agent.calls == 0 and screen.k == 0 and dispatcher.sink.events == []
    assert load_clip(tmp_path / "run").n_steps == 0
    assert "F7" in said[0] and summary["ended"].startswith("kill key")


def test_the_toggle_key_puts_it_back_in_standby_and_lets_go(tmp_path):
    back_off = {6: [TOGGLE], 30: [{"t": 0, "type": "key", "code": "f9", "down": True}]}
    states = []
    summary, dispatcher, agent, _ = run(tmp_path, seconds=10.0, hands=Hands(back_off), states=states)
    assert agent.calls == 5 and summary["toggles"] == 2
    assert held_after(dispatcher.sink) == set()
    assert states == ["acting", "standby", "stopped"]
    assert load_clip(tmp_path / "run").n_steps == 5  # standby is not written


def test_the_toggle_key_itself_is_not_the_human_taking_over():
    summary, _, _, _ = run(seconds=1.0)
    assert summary["human"] == 0


class JitteryAgent(Agent):
    """A policy whose mean look swings every tick -- 0, +6, -2 degrees -- the way sampled bins did live."""

    def __init__(self, looks):
        super().__init__()
        self.looks, self.last_look = list(looks), (0.0, 0.0)

    def act(self, obs):
        self.last_look = (self.looks[self.calls % len(self.looks)], 0.0)
        return super().act(obs)


def turn_per_tick(sink):
    """Mouse counts sent in each decision window."""
    per = {}
    for e in sink.events:
        if e["type"] == "mouse":
            per[round(e["t"] / DT)] = per.get(round(e["t"] / DT), 0) + e["dx"]
    return np.array([per.get(k, 0) for k in range(max(per) + 1)]) if per else np.zeros(1)


def play_with(agent, **config):
    clock = Clock()
    dispatcher = TimedDispatcher(clock)
    play(Screen(), agent, dispatcher, focus=Focus(), human=HumanWatch(Hands(), PlayConfig(max_seconds=2.0)),
         config=PlayConfig(max_seconds=2.0, **config), clock=clock, say=lambda _: None)
    return dispatcher


def test_the_smoothed_look_turns_far_less_jerkily_than_the_policy_swings():
    swings = [0.0, 6.0, -2.0, 6.0, 0.0, 2.0]
    smooth = turn_per_tick(play_with(JitteryAgent(swings)).sink)
    raw = turn_per_tick(play_with(JitteryAgent(swings), look_smoothing=1.0).sink)
    assert np.abs(np.diff(smooth)).mean() < 0.4 * np.abs(np.diff(raw)).mean()
    # ...while still turning the way the policy means to on average (+2 degrees a tick, 9.09 counts each).
    assert np.mean(smooth[5:]) == pytest.approx(2.0 * 9.09, rel=0.2)


def test_a_steady_look_is_reached_within_a_few_ticks():
    sent = turn_per_tick(play_with(JitteryAgent([6.0])).sink)
    assert sent[1] < sent[4] and sent[8] == pytest.approx(6.0 * 9.09, rel=0.05)


def test_the_look_is_sent_as_a_continuous_turn_not_a_bin():
    sent = turn_per_tick(play_with(JitteryAgent([3.3])).sink)
    assert sent[-1] == pytest.approx(3.3 * 9.09, abs=2)  # 3.3 degrees is between bins, and still sent as asked


def test_the_recording_labels_the_turn_that_was_sent(tmp_path):
    clock = Clock()
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent")
    play(Screen(), JitteryAgent([6.0]), TimedDispatcher(clock), focus=Focus(),
         human=HumanWatch(Hands(), PlayConfig()), config=PlayConfig(max_seconds=1.0), writer=writer,
         clock=clock, say=lambda _: None)
    clip = load_clip(tmp_path / "run")
    yaw = np.asarray(clip.labels["yaw_deg"])
    assert yaw[0] == pytest.approx(0.35 * 6.0) and yaw[-1] == pytest.approx(6.0, rel=0.02)
    assert spec.YAW_BINS_DEG[clip.actions[-1][spec.YAW]] == 6.0


def test_a_pause_does_not_carry_a_turn_across_it():
    agent = JitteryAgent([6.0])
    focus = Focus(unfocused=set(range(10, 13)))
    clock = Clock()
    dispatcher = TimedDispatcher(clock)
    play(Screen(), agent, dispatcher, focus=focus, human=HumanWatch(Hands(), PlayConfig()),
         config=PlayConfig(max_seconds=2.0), clock=clock, say=lambda _: None)
    sent = turn_per_tick(dispatcher.sink)
    assert sent[12] < sent[9]  # it eases back in after the gap instead of resuming at full speed


def test_a_game_window_that_dies_under_the_player_is_a_pause_not_a_crash(tmp_path):
    """Handed the controls while holding WaW's intro window, the first live player crashed on its first read."""
    from zombiesai.demos.capture import CaptureLost, FollowWindow

    class Window:
        def __init__(self, lives):
            self.lives, self.last_stale, self.stale_reason = lives, False, None
            self.last_hud = {"points_ammo": np.zeros((2, 2, 3), np.uint8)}

        def read(self):
            if self.lives <= 0:
                raise CaptureLost("window 0x220010e no longer exists")
            self.lives -= 1
            return np.full(spec.PIXELS_SHAPE, self.lives % 256, np.uint8)

        def close(self):
            pass

    windows = iter([Window(0), Window(1000)])  # the intro window is already dead; the real one is fine
    capture = FollowWindow(lambda: next(windows), reopen_every_s=0.0)
    clock = Clock()
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent")
    agent = Agent()
    summary = play(capture, agent, TimedDispatcher(clock), focus=Focus(), human=HumanWatch(Hands(), PlayConfig()),
                   config=PlayConfig(max_seconds=1.0), writer=writer, clock=clock, say=lambda _: None)
    assert summary["frozen"] == 1 and agent.calls == 15
    clip = load_clip(tmp_path / "run")
    assert clip.n_steps == 15  # the placeholder step before any real frame is not written


def test_the_loop_runs_with_the_mouse_motor_as_the_live_script_uses_it(tmp_path):
    """Every other test here drives looks as per-tick sub-moves; the live script uses the motor thread, and the
    first run with it died on a name the motor code shadowed. So: the real motor, a pause, and a recording."""
    import time

    clock = Clock()
    dispatcher = ActionDispatcher(FakeSink(), DispatchConfig(counts_per_degree=9.09), motor=True)
    dispatcher.pump_until = lambda deadline, poll_s=0.002: (setattr(clock, "now", deadline), time.sleep(0.004))
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent")
    try:
        summary = play(Screen(), JitteryAgent([6.0]), dispatcher, focus=Focus(unfocused={4, 5}),
                       human=HumanWatch(Hands(), PlayConfig()), config=PlayConfig(max_seconds=1.0),
                       writer=writer, clock=clock, say=lambda _: None)
    finally:
        dispatcher.close()
    assert summary["acted"] == 15 and summary["unfocused"] == 2
    clip = load_clip(tmp_path / "run")
    assert clip.n_steps == 17 and np.isfinite(clip.labels["yaw_deg"]).all()
