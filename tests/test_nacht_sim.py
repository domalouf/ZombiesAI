import math
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from zombiesai import spec
from zombiesai.reward import REWARD_TERMS
from zombiesai.sim import mechanics
from zombiesai.sim.nacht_sim import INSIDE, NachtSim, SimConfig
from zombiesai.sim.params import SimParams

H = spec.HUD_INDEX
T = {name: i for i, name in enumerate(REWARD_TERMS)}
NOOP = spec.make_action()


def deterministic_env(**kw) -> NachtSim:
    """hardness 0: nominal params, zero latency, 4 frames/step, no dropout, no HUD noise."""
    return NachtSim(SimConfig(hardness=0.0, **kw))


def hud(obs) -> dict[str, float]:
    raw = spec.decode_hud(obs["hud"])
    return {f.name: float(raw[i]) for i, f in enumerate(spec.HUD_FIELDS)}


def teleport(env: NachtSim, x: float, y: float) -> dict:
    env.px, env.py = x, y
    env._refresh_distance_field()
    return env._observation()


def test_gymnasium_env_checker():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(NachtSim(), skip_render_check=True)


def test_same_seed_same_trajectory():
    actions = np.random.default_rng(0).integers(0, spec.ACTION_NVEC, size=(400, len(spec.ACTION_NVEC)))
    runs = []
    for _ in range(2):
        env = NachtSim()
        obs, _ = env.reset(seed=123)
        trace = [obs["state"].copy()]
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            trace.append(np.append(obs["state"], [r, obs["hud"].sum()]))
            if term or trunc:
                break
        runs.append(np.concatenate(trace))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_observations_stay_in_space():
    env = NachtSim()
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(1500):
        assert env.observation_space.contains(obs)
        obs, _, term, trunc, _ = env.step(rng.integers(0, spec.ACTION_NVEC))
        if term or trunc:
            obs, _ = env.reset()


def test_initial_hud():
    obs, _ = deterministic_env().reset(seed=0)
    h = hud(obs)
    assert h["round"] == pytest.approx(1)
    assert h["points"] == pytest.approx(mechanics.STARTING_POINTS)
    assert (h["mag_ammo"], h["reserve_ammo"]) == (pytest.approx(8), pytest.approx(32))
    assert h["hud_confidence"] == 1.0


def test_rounds_follow_the_ground_truth_table():
    env = deterministic_env(max_steps=10**9)
    env.reset(seed=0)
    spawned, alive_before = [], np.zeros_like(env.z_alive)
    while len(spawned) < 7:
        _, _, term, _, info = env.step(NOOP)
        assert not term
        new = env.z_alive & ~alive_before
        assert np.all(env.z_hp[new] == mechanics.zombie_health(env.round))
        env.z_alive[:] = False  # silently exterminate so the round can end
        alive_before = env.z_alive.copy()
        spawned += [e["spawned"] for e in info["events"] if e["type"] == "round_complete"]
    assert spawned == [mechanics.zombies_in_round_solo(r) for r in range(1, 8)]


def test_start_round_option():
    env = deterministic_env()
    obs, _ = env.reset(seed=0, options={"start_round": 12})
    assert hud(obs)["round"] == pytest.approx(12)
    assert env.zombie_max_hp == mechanics.zombie_health(12)


def test_buying_a_door():
    env = deterministic_env()
    env.reset(seed=0, options={"start_points": 1500})
    h = hud(teleport(env, 13.4, 4.0))
    assert h["prompt_door"] == 1.0 and h["prompt_price"] == pytest.approx(1000)
    _, reward, _, _, info = env.step(spec.make_action(button="use"))
    assert env.open_mask == 1 << spec.DOORS.index("help_door")
    assert env.points == 500
    assert info["reward_terms"][T["door"]] == pytest.approx(3.0)
    assert any(e["type"] == "purchase" for e in info["events"])


def test_cannot_buy_without_points():
    env = deterministic_env()
    env.reset(seed=0, options={"start_points": 999})
    teleport(env, 13.4, 4.0)
    env.step(spec.make_action(button="use"))
    assert env.open_mask == 0 and env.points == 999


def test_buying_a_wall_weapon_then_ammo():
    env = deterministic_env()
    env.reset(seed=0, options={"start_points": 2000})
    teleport(env, 13.0, 1.5)
    _, _, _, _, info = env.step(spec.make_action(button="use"))
    assert [hw.w.name for hw in env.weapons] == ["m1911", "m1a1_carbine"]
    assert info["reward_terms"][T["wall_weapon"]] == pytest.approx(2.0)
    env.weapons[env.slot].reserve = 10
    env.step(NOOP)
    _, _, _, _, info = env.step(spec.make_action(button="use"))
    assert env.points == 2000 - 600 - 300
    assert env.weapons[env.slot].reserve == env.weapons[env.slot].w.max_reserve
    assert info["reward_terms"][T["ammo"]] == pytest.approx(0.5)


def test_holding_use_repairs_a_window():
    env = deterministic_env()
    env.reset(seed=0)
    env.planks[0] = 3
    assert hud(teleport(env, 4.0, 0.8))["prompt_repair"] == 1.0
    repair_reward = 0.0
    for _ in range(60):
        _, _, _, _, info = env.step(spec.make_action(button="use"))
        repair_reward += info["reward_terms"][T["repair"]]
    assert env.planks[0] == spec.MAX_PLANKS
    assert env.points == mechanics.STARTING_POINTS + 3 * mechanics.SCORE_BARRIER_PLANK
    assert repair_reward == pytest.approx(3 * 10 * 0.2 / 100)


def test_semi_auto_fires_only_on_the_press():
    env = deterministic_env()
    env.reset(seed=0)
    for _ in range(10):
        env.step(spec.make_action(fire=1))
    assert env.rs.shots == 1
    for i in range(20):
        env.step(spec.make_action(fire=i % 2))
    assert env.rs.shots >= 5


def test_automatic_fires_while_held():
    env = deterministic_env()
    env.reset(seed=0, options={"loadout": ("thompson",)})
    for _ in range(15):  # one second
        env.step(spec.make_action(fire=1))
    assert 10 <= env.rs.shots <= 13


def test_death_terminates_with_rounds_survived():
    env = deterministic_env(params=SimParams(player_max_hp=1.0))
    env.reset(seed=0)
    env.z_alive[0], env.z_phase[0], env.z_hp[0], env.z_speed[0] = True, INSIDE, 1e9, 0.0
    env.z_pos[0] = complex(env.px + 0.5, env.py)
    for _ in range(60):
        obs, reward, term, trunc, info = env.step(NOOP)
        if term:
            break
    assert term and not trunc
    assert info["rounds_survived"] == 1
    assert hud(obs)["downed"] == 1.0
    assert info["reward_terms"][T["death"]] == pytest.approx(-15.0)


def test_step_cap_truncates_without_a_game_over():
    env = NachtSim(SimConfig(max_steps=5))
    env.reset(seed=0)
    for _ in range(5):
        _, _, term, trunc, info = env.step(NOOP)
    assert trunc and not term
    assert info["round_reached"] == 1 and "rounds_survived" not in info


def test_latency_delays_actions():
    env = deterministic_env(latency_steps=2)
    env.reset(seed=0)
    yaw0 = env.yaw
    env.step(spec.make_action(yaw=30))
    env.step(NOOP)
    assert env.yaw == yaw0
    env.step(NOOP)
    assert math.isclose((yaw0 - env.yaw) % (2 * math.pi), math.radians(30), abs_tol=1e-9)


@pytest.mark.parametrize("through_window", [True, False])
def test_knife_reaches_through_windows_but_not_walls(through_window):
    env = deterministic_env()
    env.reset(seed=0)
    # Player 0.4 m inside the south wall, facing south; a zombie 0.6 m outside it.
    x = 4.0 if through_window else 7.0
    teleport(env, x, 0.4)
    env.yaw = -math.pi / 2
    env.z_alive[0], env.z_phase[0], env.z_hp[0], env.z_speed[0] = True, 1, 1e9, 0.0
    env.z_pos[0] = complex(x, -0.6)
    env.step(spec.make_action(button="melee"))
    assert (env.z_hp[0] < 1e9) == through_window


@pytest.mark.parametrize("door_open", [True, False])
def test_zombies_path_through_open_doors_only(door_open):
    env = deterministic_env(max_steps=10**9)
    env.reset(seed=0, options={"open_doors": ("help_door",) if door_open else ()})
    teleport(env, 19.0, 4.0)  # help room
    env.z_alive[0], env.z_phase[0], env.z_hp[0], env.z_speed[0] = True, INSIDE, 1e9, 3.0
    env.z_pos[0] = complex(3.0, 5.0)  # far side of the start room
    for _ in range(150):
        env.step(NOOP)
    reached = abs(env.z_pos[0] - complex(env.px, env.py)) < 2.0
    assert reached == door_open
