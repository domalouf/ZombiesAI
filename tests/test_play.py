import json
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
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="play") if tmp_path else None
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
    assert clip.n_steps == 17 and clip.label_source == "play"
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
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="play")
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
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="play")
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
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="play")
    try:
        summary = play(Screen(), JitteryAgent([6.0]), dispatcher, focus=Focus(unfocused={4, 5}),
                       human=HumanWatch(Hands(), PlayConfig()), config=PlayConfig(max_seconds=1.0),
                       writer=writer, clock=clock, say=lambda _: None)
    finally:
        dispatcher.close()
    assert summary["acted"] == 15 and summary["unfocused"] == 2
    clip = load_clip(tmp_path / "run")
    assert clip.n_steps == 17 and np.isfinite(clip.labels["yaw_deg"]).all()


# --- The human's corrections (HG-DAgger): taking over is also teaching ------------------------------------

from zombiesai.demos.clips import ACTOR_HUMAN, ACTOR_HUMAN_IDLE, ACTOR_POLICY  # noqa: E402
from zombiesai.demos.inputs import InputConfig, label_confidence, quantize, synthesize  # noqa: E402

CPD = 9.09
INPUT = InputConfig(counts_per_degree=CPD, bindings=dict(DEFAULT_BINDINGS))
# Back off to the left while turning right and reloading: nothing the stub policy (forward and fire) ever does.
CORRECTION = spec.make_action(forward=-1, strafe=-1, yaw=6.0, button="reload")
IDLE_TICKS = math.ceil(PlayConfig().human_idle_s / DT)  # untouched ticks before the policy takes back over


def window(tick):
    """Poll `tick` closes the input window of the step read at the previous poll: [(tick-2)*dt, (tick-1)*dt)
    on the test clock (the first poll is at 0, and every later one at the deadline before it)."""
    return (tick - 2) * DT


def correcting(ticks, action=CORRECTION):
    """What a person's hands produce for `action` in each of `ticks`' windows, exactly as a demo would log it."""
    return {tick: synthesize(action, window(tick), DT, INPUT) for tick in ticks}


def play_run(tmp_path, hands, *, seconds=3.0, focus=None, input_config=INPUT):
    clock = Clock()
    config = PlayConfig(max_seconds=seconds)
    writer = ClipWriter(tmp_path / "play_0000", source={"kind": "test"}, label_source="play",
                        config={"play": {"max_seconds": seconds}})
    summary = play(Screen(), Agent(), TimedDispatcher(clock), focus=focus or Focus(),
                   human=HumanWatch(hands, config, input_config), config=config, writer=writer, clock=clock,
                   say=lambda _: None)
    return summary, load_clip(tmp_path / "play_0000")


def test_a_takeover_is_labelled_with_the_humans_own_input_decoded_like_a_demo(tmp_path):
    summary, clip = play_run(tmp_path, Hands(correcting(range(4, 9))))
    actor = clip.extra("actor")
    # Poll 4 closes step 2's window, the policy's step: the human's first touch is in it, so it is theirs.
    # Polls 5-8 close steps 3-6, all theirs. Then nothing, until the policy takes back over.
    human = np.flatnonzero(actor == ACTOR_HUMAN)
    np.testing.assert_array_equal(human, [2, 3, 4, 5, 6])
    for step in human:
        np.testing.assert_array_equal(clip.actions[step], CORRECTION)
    assert clip.labels["yaw_deg"][3] == pytest.approx(6.0, abs=0.1)  # raw degrees, at the demos' 9.09 counts
    # Confidence too is the demo's own: the reload tap is held for part of the window, as it would be there.
    demo = quantize(synthesize(CORRECTION, 0.0, DT, INPUT), 0.0, 1, INPUT)
    np.testing.assert_array_equal(clip.confidence[human], label_confidence(demo, INPUT)[0])
    assert summary["corrections"] == 5 and summary["correction_seconds"] == pytest.approx(5 * DT)
    np.testing.assert_array_equal(np.flatnonzero(clip.usable()), human)
    np.testing.assert_array_equal(np.flatnonzero(clip.corrections), human)


def test_the_policys_own_steps_are_never_training_labels(tmp_path):
    _, clip = play_run(tmp_path, Hands(correcting(range(4, 9))))
    policy = clip.extra("actor") == ACTOR_POLICY
    assert policy.sum() > 30 and (clip.flags[policy] & FLAG_BAD_STEP == 0).all()  # good steps, just not labels
    np.testing.assert_array_equal(clip.actions[policy], np.tile(FORWARD_AND_FIRE, (policy.sum(), 1)))
    assert not clip.usable()[policy].any()


def test_the_idle_wait_before_handing_back_is_not_a_label(tmp_path):
    summary, clip = play_run(tmp_path, Hands(correcting(range(4, 9))))
    actor = clip.extra("actor")
    idle = np.flatnonzero(actor == ACTOR_HUMAN_IDLE)
    # The last touch closes at poll 8 (step 6); steps 7 onwards are the human's untouched ticks until the
    # policy resumes.
    np.testing.assert_array_equal(idle, np.arange(7, 7 + IDLE_TICKS))
    assert actor[7 + IDLE_TICKS] == ACTOR_POLICY
    assert summary["human_idle"] == IDLE_TICKS
    assert not clip.usable()[idle].any()
    assert (clip.flags[idle] & FLAG_BAD_STEP == 0).all()  # real frames of the game, kept for history


def test_a_pause_inside_a_takeover_is_play_and_stays(tmp_path):
    pause = range(6, 11)  # five untouched windows, well under the 1.5 s that hands back
    hands = Hands({**correcting(range(4, 6)), **correcting(range(11, 13))})
    _, clip = play_run(tmp_path, hands)
    actor = clip.extra("actor")
    np.testing.assert_array_equal(np.flatnonzero(actor == ACTOR_HUMAN), np.arange(2, 11))
    assert (clip.actions[[t - 2 for t in pause]] == spec.NEUTRAL_ACTION).all()  # standing still, deliberately
    np.testing.assert_array_equal(np.flatnonzero(actor == ACTOR_HUMAN_IDLE), np.arange(11, 11 + IDLE_TICKS))


def test_a_held_key_is_the_human_playing_not_idling(tmp_path):
    """Running down a corridor sends one key press and then nothing: the policy must not take over mid-run,
    and the steps must be labelled forward, not idle."""
    hold = {4: [{"t": window(4), "type": "key", "code": "w", "down": True}],
            40: [{"t": window(40) + DT / 2, "type": "key", "code": "w", "down": False}]}
    summary, clip = play_run(tmp_path, Hands(hold), seconds=2.0)
    actor = clip.extra("actor")
    forward = spec.make_action(forward=1)
    np.testing.assert_array_equal(np.flatnonzero(actor == ACTOR_HUMAN), np.arange(2, 39))
    assert (clip.actions[2:38] == forward).all()
    assert (actor[39:39 + IDLE_TICKS] == ACTOR_HUMAN_IDLE).all()
    assert summary["corrections"] == 37


def test_the_command_keys_are_never_labels_nor_a_takeover(tmp_path):
    def tap(code, t):
        return [{"t": t, "type": "key", "code": code, "down": True},
                {"t": t + 0.01, "type": "key", "code": code, "down": False}]

    at = correcting(range(4, 7))
    # Mid-correction, a double tap of the toggle key (which cancels out) and the mark key.
    at[5] = at[5] + tap("f7", window(5)) + tap("f7", window(5) + 0.02) + tap("f8", window(5) + 0.03)
    at[20] = tap("f8", window(20))  # while the human idles before handing back: does not extend it
    at[40] = tap("f8", window(40))  # while the policy is back in control: not a takeover
    at[60] = tap("f9", window(60))
    summary, clip = play_run(tmp_path, Hands(at), seconds=10.0)
    actor = clip.extra("actor")
    np.testing.assert_array_equal(clip.actions[3], CORRECTION)  # the step whose window held the taps
    np.testing.assert_array_equal(np.flatnonzero(actor == ACTOR_HUMAN), [2, 3, 4])
    np.testing.assert_array_equal(np.flatnonzero(actor == ACTOR_HUMAN_IDLE), np.arange(5, 5 + IDLE_TICKS))
    assert actor[38] == ACTOR_POLICY
    assert summary["ended"].startswith("kill key") and summary["toggles"] == 1  # the double tap cancelled
    events = [json.loads(line) for line in (clip.path / "inputs.jsonl").read_text().splitlines()]
    assert any(e.get("code") == "f9" for e in events)  # the raw log keeps everything, commands included


def test_a_command_key_bound_to_a_control_is_refused():
    with pytest.raises(ValueError, match="bound to game controls"):
        HumanWatch(Hands(), PlayConfig(), InputConfig(bindings={**DEFAULT_BINDINGS, "f7": "use"}))


def test_the_humans_steps_on_a_dead_picture_are_nobodys_labels(tmp_path):
    _, clip = play_run(tmp_path, Hands(correcting(range(4, 12))), focus=Focus(unfocused={6}))
    # Iteration 6 reads step 5: the human still has the controls, but the game was not focused.
    assert clip.extra("actor")[5] == ACTOR_HUMAN and clip.flags[5] & FLAG_BAD_STEP
    assert not clip.usable()[5] and clip.usable()[4] and clip.usable()[6]
    assert clip.flags[6] & FLAG_CLIP_START  # its history does not reach back across the dead frame


def test_each_step_keeps_the_instant_its_window_opened(tmp_path):
    _, clip = play_run(tmp_path, Hands(correcting(range(4, 9))), seconds=1.0)
    t = clip.extra("t_mono")
    assert t.dtype == np.float64
    np.testing.assert_allclose(t[:5], np.arange(5) * DT)


def test_a_play_run_is_refused_as_any_other_label_source(tmp_path):
    writer = ClipWriter(tmp_path / "run", source={"kind": "test"}, label_source="agent")
    with pytest.raises(ValueError, match="label_source='play'"):
        play(Screen(), Agent(), TimedDispatcher(Clock()), focus=Focus(), human=HumanWatch(Hands(), PlayConfig()),
             writer=writer, say=lambda _: None)


def test_a_play_run_cannot_be_requantized_over_the_policys_steps(tmp_path):
    from zombiesai.demos.recorder import requantize

    _, clip = play_run(tmp_path, Hands(correcting(range(4, 9))), seconds=1.0)
    with pytest.raises(ValueError, match="play run"):
        requantize(clip.path, INPUT)


def demo_clip(path, n=40, seed=0):
    rng = np.random.default_rng(seed)
    writer = ClipWriter(path, source={"kind": "test"}, label_source="input_log")
    for i in range(n):
        writer.add(np.full(spec.PIXELS_SHAPE, i, np.uint8), rng.integers(0, spec.ACTION_NVEC))
    writer.close()
    return load_clip(path)


def test_training_takes_only_the_corrections_from_a_play_run_and_weighs_them(tmp_path):
    from zombiesai.demos.dataset import ClipDataset, DataConfig, training_clips

    _, played = play_run(tmp_path / "runs", Hands(correcting(range(4, 9))))
    idle, _ = play_run(tmp_path / "idle", Hands())  # nobody took over: nothing to learn from
    demo = demo_clip(tmp_path / "demos" / "demo_0000")
    clips, skipped = training_clips([tmp_path / "demos", tmp_path / "runs", tmp_path / "idle"])
    assert {c.path for c in clips} == {demo.path, played.path}
    assert [c.path for c in skipped] == [tmp_path / "idle" / "play_0000"]

    data = ClipDataset(clips, DataConfig(correction_weight=3.0), before=3)
    play_index = next(i for i, c in enumerate(data.clips) if c.is_play_run)
    steps = data.index[data.index[:, 0] == play_index, 1]
    np.testing.assert_array_equal(steps, [2, 3, 4, 5, 6])
    assert len(data) == demo.n_steps + 5 and data.n_corrections == 5
    batch = data.batch(data.index)
    np.testing.assert_array_equal(batch["action"][data.index[:, 0] == play_index], np.tile(CORRECTION, (5, 1)))
    corrections = data.index[:, 0] == play_index
    np.testing.assert_allclose(batch["weight"][corrections], 3.0 * played.confidence[steps])
    np.testing.assert_array_equal(batch["weight"][~corrections], 1.0)
    # The correction's frame history is the policy's steps before it, which is the point.
    first = data.frames(np.array([[play_index, 2]]))[0]
    assert [int(f[0, 0, 0]) for f in first] == [1, 1, 2, 3]


def test_a_play_run_from_before_corrections_offers_nothing(tmp_path):
    """Old runs were label_source "agent" with every policy step unflagged; they must not train as demos."""
    writer = ClipWriter(tmp_path / "play_0000", source={"kind": "test"}, label_source="agent",
                        config={"play": {"max_seconds": 1.0}})
    for i in range(5):
        writer.add(np.zeros(spec.PIXELS_SHAPE, np.uint8), FORWARD_AND_FIRE)
    writer.close()
    clip = load_clip(tmp_path / "play_0000")
    assert clip.is_play_run and not clip.usable().any()


def test_train_bc_accepts_play_runs_beside_demos(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    for seed in range(3):
        demo_clip(tmp_path / "demos" / f"demo_{seed:04d}", n=30, seed=seed)
    _, played = play_run(tmp_path / "runs" / "play", Hands(correcting(range(4, 9))))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_bc.py"
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "demos"), str(tmp_path / "runs" / "play"), "--out",
         str(tmp_path / "bc"), "--epochs", "1", "--batch-size", "8", "--hidden", "16", "--device", "cpu",
         "--correction-weight", "2.5"],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "5 human corrections" in out.stdout
    config = json.loads((tmp_path / "bc" / "config.json").read_text())
    assert config["correction_weight"] == 2.5 and "play" in config["label_sources"]
    assert config["train_steps"] + config["val_steps"] == 3 * 30 + 5
    assert str(played.path) in config["train_clips"] + config["val_clips"]
