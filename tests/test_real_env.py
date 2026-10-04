"""RealGameEnv end to end on a fake game: scripted HUD reads, a fake clock, a dispatcher that only records."""

from dataclasses import replace

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


def test_info_carries_the_held_gun_and_this_steps_ammo():
    from zombiesai.hud.parse import MAG_AT_LEAST, UNREADABLE
    from zombiesai.hud.weapons import WEAPON_INDEX

    env, game, _, _, _ = make_env([500] * 4)
    env.reset()
    reads = iter([dict(weapon=WEAPON_INDEX["kar98k"], weapon_status=OK, mag=4, mag_status=OK, mag_flags=MAG_AT_LEAST,
                       reserve=50, reserve_status=OK, grenades=0, grenades_status=OK)] * 3
                 + [dict(grenades_status=UNREADABLE)])
    env.reader = lambda crops: replace(reader(crops), **next(reads))
    infos = [env.step(IDLE)[4] for _ in range(4)]
    assert infos[0]["weapon"] is None and infos[2]["weapon"] == "kar98k"  # settles on the third read
    assert (infos[2]["mag"], infos[2]["mag_at_least"], infos[2]["reserve"], infos[2]["grenades"]) == (4, True, 50, 0)
    assert infos[3]["weapon"] == "kar98k"  # held while the name is gone
    assert (infos[3]["mag"], infos[3]["mag_at_least"], infos[3]["reserve"], infos[3]["grenades"]) == (None, False, None, None)


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
                                                                                  game.points.__setitem__(slice(None), [None, None, 500])))
    env.reset()
    game.points = [7]  # a stuck screen: the command never takes
    env._ended_by_death = True
    env.reset()
    # the quick restart first; then every try types the map, and so does the wait after the relaunch
    # (Plutonium cannot start a map from its command line)
    assert typed == [config.quick_start_command] + [config.start_command] * (config.reset_attempts + 1)
    assert relaunched == [1] and env.resets == {"console": 5, "quick": 0, "relaunch": 1}


def test_a_game_over_screen_still_showing_500_is_not_a_fresh_game():
    config = EnvConfig(start_timeout_s=5.0, after_death_s=0.0)
    env, game, _, typed, clock = make_env([500] * 4, config=config)
    env.reset()
    game.points = [500] * 20 + [None] * 5 + [500]  # the dead game's 500, then the map load, then the new game
    env._ended_by_death = True
    t0 = game.reads
    env.reset()
    assert typed == [config.quick_start_command] and game.reads - t0 >= 25 and env.resets["quick"] == 1


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


def test_a_reset_taps_the_start_key_while_it_waits():
    pressed = []
    env, game, _, typed, clock = make_env([None] * 200 + [500], press=pressed.append)
    env.reset()
    enters = [k for k in pressed if k == "enter"]
    assert enters and set(pressed) == {"enter", "tab"} and pressed[-1] == "tab"  # tab once, when it is up
    assert len(enters) <= clock.t / env.config.start_key_every_s + 1


def test_an_open_console_is_closed_and_the_step_is_bad():
    pressed = []
    shown = {"open": False}
    env, game, dispatcher, _, _ = make_env([500], press=pressed.append, console_open=lambda: shown["open"])
    env.reset()
    pressed.clear()
    idle = np.zeros(len(spec.ACTION_NVEC), dtype=np.int64)
    idle[spec.YAW], idle[spec.PITCH] = spec.YAW_BINS_DEG.index(0.0), spec.PITCH_BINS_DEG.index(0.0)
    _, _, _, _, info = env.step(idle)
    assert not info["bad"] and pressed == []
    shown["open"] = True
    _, _, _, _, info = env.step(idle)
    assert info["bad"] and pressed == ["grave"] and dispatcher.releases >= 1


def test_the_pitch_springs_back_to_level():
    env, _, _, _, _ = make_env([500])
    env.reset()
    action = np.zeros(len(spec.ACTION_NVEC), dtype=np.int64)
    action[spec.YAW] = spec.YAW_BINS_DEG.index(0.0)
    action[spec.PITCH] = spec.PITCH_BINS_DEG.index(-6.0)
    sent = [env._look(action)[1] for _ in range(40)]
    assert env.pitch == sum(sent) or abs(env.pitch - sum(sent)) < 1e-9
    assert -85.0 <= env.pitch < -60.0  # held down, it goes down -- the spring only resists
    held = env.pitch
    action[spec.PITCH] = spec.PITCH_BINS_DEG.index(0.0)
    for _ in range(75):  # five seconds of not touching it: e^(-5/2) of the way left
        env._look(action)
    assert abs(env.pitch) < 0.1 * abs(held)


def test_a_fresh_game_s_scoreboard_is_cleared_and_its_return_is_a_down():
    board = {"up": True}
    pressed = []

    def press(key):
        pressed.append(key)
        if key == "tab":
            board["up"] = False

    env, _, _, _, clock = make_env([500] * 4 + [620], press=press, downed=lambda: board["up"])
    env.reset()
    assert pressed.count("tab") == 1 and not board["up"]
    board["up"] = True  # drawn a moment late, in the first seconds: cleared again, not a down
    _, reward, terminated, _, info = env.step(IDLE)
    assert not terminated and not board["up"] and pressed.count("tab") == 2
    clock.t += env.config.down_grace_s
    board["up"] = True  # co-op draws it when the player goes down
    for k in range(env.config.down_steps):
        _, reward, terminated, truncated, info = env.step(IDLE)
    assert terminated and not truncated and info["episode"]["reason"] == "down"
    assert reward < 0  # the death penalty, the same as solo's game over


def test_shots_are_magazine_marks_gone_while_the_trigger_is_held():
    env, game, _, _, _ = make_env([500] * 4)
    env.reset()
    fire = spec.make_action(fire=1)
    # Held: 30 -> 28 -> 27 is three shots. Idle: 27 -> 30 is a reload, 30 -> 20 a swap, not ten shots. Held
    # again: 20 -> 19 -> 12 is eight more.
    steps = [(30, fire), (28, fire), (27, fire), (30, IDLE), (20, IDLE), (19, fire), (12, fire), (12, IDLE)]
    mags = iter(m for m, _ in steps)

    def with_mag(crops):
        r = reader(crops)
        r.mag, r.mag_status = next(mags), OK
        return r

    env.reader = with_mag
    for _, action in steps:
        env.step(action)
    assert env.summary()["shots"] == 11 and env.summary()["hits"] == 0


def _die(env, game):
    """Play from a fresh game to a settled death; the terminal step's info."""
    game.points = [550] * 5 + [520] * 6
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(IDLE)
        if terminated or truncated:
            return info
    raise AssertionError("no death")


def test_a_death_carries_the_game_over_scoreboard_once_two_looks_agree():
    from zombiesai.realgame.end_screen import EndScreen

    looks = iter([None, EndScreen(550, 3, 1, 0.9), EndScreen(550, 4, 1, 0.9), EndScreen(550, 4, 1, 0.8)])
    env, game, _, _, clock = make_env([500] * 4, end_screen=lambda: next(looks))
    env.reset()
    t = clock.t
    episode = _die(env, game)["episode"]
    assert (episode["end_points"], episode["end_kills"], episode["end_headshots"]) == (550, 4, 1)
    # four looks, a tick apart: what they took comes out of the wait before the restart
    assert env._end_screen_s == pytest.approx(3 * env.config.end_screen_every_s, abs=0.05)
    assert clock.t > t


def test_a_scoreboard_that_never_reads_twice_alike_gives_no_numbers():
    from zombiesai.realgame.end_screen import EndScreen

    n = {"looks": 0}

    def flicker():
        n["looks"] += 1
        return EndScreen(550, n["looks"], 0, 0.9)  # never the same twice

    env, game, _, _, _ = make_env([500] * 4, end_screen=flicker)
    env.reset()
    episode = _die(env, game)["episode"]
    assert episode["end_points"] is None and episode["end_kills"] is None and episode["end_headshots"] is None
    assert env._end_screen_s >= env.config.end_screen_s
    assert n["looks"] <= env.config.end_screen_s / env.config.end_screen_every_s + 2


def test_an_end_that_is_not_a_death_does_not_look_for_the_scoreboard():
    def never():
        raise AssertionError("looked")

    env, game, _, _, _ = make_env([500] * 4, end_screen=never)
    env.reset()
    game.points = [503] * 3 + [517] * 3 + [529] * 3 + [529]  # hud_lost: truncated
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(IDLE)
        if truncated:
            break
    assert truncated and not terminated and info["episode"]["end_kills"] is None
