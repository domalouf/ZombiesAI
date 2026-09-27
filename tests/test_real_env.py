"""RealGameEnv end to end on a fake game: scripted HUD reads, a fake clock, a dispatcher that only records."""

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.hud.parse import ABSENT, OK, HudReading
from zombiesai.realgame.env import EnvConfig, RealGameEnv, ResetFailed

IDLE = spec.make_action()


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeGame:
    """What the capture and the HUD reader see: a points value per read, or None for no HUD at all."""

    def __init__(self, clock: Clock, points=None):
        self.clock = clock
        self.points = list(points or [])  # consumed one per read; the last one repeats
        self.stale = False
        self.last_hud = None
        self.last_stale = False
        self.reads = 0

    def read(self):
        self.reads += 1
        value = self.points.pop(0) if len(self.points) > 1 else (self.points[0] if self.points else None)
        self.last_hud = {"points": value}
        self.last_stale = self.stale
        frame = np.full(spec.PIXELS_SHAPE, self.reads % 255, dtype=np.uint8)
        return frame

    def close(self):
        pass


def reader(crops):
    value = crops["points"]
    if value is None:
        return HudReading()
    return HudReading(points=value, points_conf=1.0, points_status=OK, round=1, round_status=OK, round_conf=1.0)


class FakeDispatcher:
    def __init__(self, clock: Clock):
        self.clock = clock
        self.applied, self.releases = [], 0

    def apply(self, action, now=None, dt=None, look_deg=None):
        self.applied.append(np.asarray(action))

    def pump_until(self, deadline, poll_s=0.002):
        self.clock.t = max(self.clock.t, deadline)

    def release_all(self):
        self.releases += 1

    def close(self):
        pass


def make_env(points, **kwargs):
    clock = Clock()
    game = FakeGame(clock, points)
    dispatcher = FakeDispatcher(clock)
    typed = []
    config = kwargs.pop("config", EnvConfig())
    env = RealGameEnv(game, dispatcher, reader=reader, console=typed.append, config=config, clock=clock,
                      say=lambda m: None, **kwargs)
    return env, game, dispatcher, typed, clock


def test_the_first_reset_waits_for_a_settled_500_without_typing():
    env, game, _, typed, _ = make_env([None] * 10 + [500])
    obs, info = env.reset()
    assert obs["pixels"].shape == spec.PIXELS_SHAPE
    assert typed == [] and game.reads >= 13  # ten blank reads, then 500 has to hold for three


def test_a_gain_pays_once_it_settles_and_a_death_terminates():
    env, game, dispatcher, _, _ = make_env([500] * 4)
    env.reset()
    game.points = [550] * 5 + [520] * 6  # a kill, then the downed penalty (5% of 550, to the nearest 10)
    rewards, done = [], None
    for _ in range(20):
        _, r, terminated, truncated, info = env.step(IDLE)
        rewards.append(r)
        if terminated or truncated:
            done = (terminated, truncated, info)
            break
    assert done is not None and done[0] and not done[1]
    assert done[2]["episode"]["reason"] == "death"
    assert rewards[2] == pytest.approx(0.5)  # +50 points, settled on the third read
    assert rewards[-1] == pytest.approx(-15.0)  # the death term
    assert dispatcher.releases >= 2  # released at reset and again at the end
    assert done[2]["episode"]["points_gained"] == 50


def test_three_implausible_changes_in_a_row_truncate_as_hud_lost():
    env, game, _, _, _ = make_env([500] * 4)
    env.reset()
    game.points = [503] * 3 + [517] * 3 + [529] * 3 + [529]  # not multiples of 10: misreads, not points
    for _ in range(12):
        _, reward, terminated, truncated, info = env.step(IDLE)
        assert reward == 0.0
        if truncated:
            break
    assert truncated and not terminated and info["episode"]["reason"] == "hud_lost"


def test_a_frozen_picture_is_bad_steps_then_a_truncation():
    env, game, _, _, _ = make_env([500] * 4, config=EnvConfig(stale_s=1.0))
    env.reset()
    game.stale = True
    bads, truncated = [], False
    for _ in range(30):
        _, _, _, truncated, info = env.step(IDLE)
        bads.append(info["bad"])
        if truncated:
            break
    assert truncated and info["episode"]["reason"] == "picture_lost" and all(bads)


def test_later_resets_type_the_start_command_and_escalate_to_a_relaunch():
    relaunched = []
    config = EnvConfig(start_timeout_s=1.0, launch_timeout_s=2.0, after_death_s=0.0)
    env, game, _, typed, _ = make_env([500] * 4, config=config, restart=lambda: (relaunched.append(1),
                                                                                  game.points.__setitem__(slice(None), [500])))
    env.reset()
    game.points = [None]  # the game never comes back by itself
    env._ended_by_death = True
    env.reset()
    assert typed == [config.start_command] * config.reset_attempts
    assert relaunched == [1] and env.resets == {"console": 3, "relaunch": 1}


def test_no_way_to_relaunch_means_the_reset_fails_loudly():
    env, game, _, _, _ = make_env([None], config=EnvConfig(start_timeout_s=0.5, first_look_s=0.5))
    with pytest.raises(ResetFailed):
        env.reset()


def test_an_overrun_is_a_bad_step_and_the_schedule_does_not_catch_up():
    env, game, _, _, clock = make_env([500] * 4)
    env.reset()
    _, _, _, _, info = env.step(IDLE)
    assert not info["bad"]
    clock.t += 0.5  # the policy took half a second
    _, _, _, _, info = env.step(IDLE)
    assert info["bad"]
    t = clock.t
    _, _, _, _, info = env.step(IDLE)
    assert not info["bad"] and clock.t == pytest.approx(t + env.config.dt)


def test_the_hud_vanishing_mid_game_truncates():
    env, game, _, _, _ = make_env([500] * 4, config=EnvConfig(hud_absent_s=1.0))
    env.reset()
    game.points = [None]
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(IDLE)
        if truncated or terminated:
            break
    assert truncated and not terminated and info["episode"]["reason"] == "hud_gone"
    assert HudReading().points_status == ABSENT
