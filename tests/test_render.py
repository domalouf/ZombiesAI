import math

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.sim.nacht_sim import INSIDE, NachtSim, SimConfig
from zombiesai.sim.render import Renderer


def standing(x: float, y: float, yaw: float) -> NachtSim:
    """A deterministic sim with the player placed by hand, before any zombie has spawned."""
    env = NachtSim(SimConfig(hardness=0.0), render_mode="rgb_array")
    env.reset(seed=0)
    env.px, env.py, env.yaw, env.pitch = x, y, yaw, 0.0
    env._refresh_distance_field()
    env._observation()
    return env


def add_zombie(env: NachtSim, x: float, y: float) -> None:
    env.z_alive[0], env.z_phase[0], env.z_hp[0] = True, INSIDE, 1e9
    env.z_pos[0] = complex(x, y)


def test_render_mode_returns_agent_sized_frames():
    env = NachtSim(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == spec.PIXELS_SHAPE and frame.dtype == np.uint8
    assert NachtSim().render() is None
    with pytest.raises(ValueError):
        NachtSim(render_mode="human")


def test_rendering_is_deterministic_at_any_size():
    env = standing(7.0, 7.0, 0.0)
    renderer = Renderer(env.geo, 320, 180)
    first = renderer.render(env)
    assert first.shape == (180, 320, 3)
    np.testing.assert_array_equal(first, renderer.render(env))


def test_axis_aligned_and_diagonal_views_render():
    env = standing(7.0, 5.0, 0.0)
    for k in range(8):
        env.yaw = k * math.pi / 4
        assert env.render().shape == spec.PIXELS_SHAPE


@pytest.mark.parametrize("behind_wall", [False, True])
def test_walls_hide_zombies(behind_wall):
    # Facing east across the start room; the help room lies beyond the solid wall at x = 14..15.
    env = standing(7.0, 7.0, 0.0)
    empty = env.render()
    add_zombie(env, 17.0 if behind_wall else 11.0, 7.0)
    assert np.array_equal(env.render(), empty) == behind_wall


def test_damage_reddens_the_screen_edges():
    env = standing(7.0, 7.0, 0.0)
    calm = env.render().astype(int)
    env.flash_obs = 1.0
    change = env.render().astype(int)[:10, :20] - calm[:10, :20]
    assert change[..., 0].mean() > 15 and change[..., 1].mean() <= 0


def test_torn_planks_open_the_window():
    env = standing(4.0, 2.5, -math.pi / 2)  # facing the start room's south-west window from inside
    boarded = env.render()
    env.planks[:] = 0
    assert not np.array_equal(env.render(), boarded)


def test_muzzle_flash_shows_only_on_the_step_a_shot_is_fired():
    env = standing(7.0, 7.0, 0.0)
    env.step(spec.make_action(fire=1))
    assert env.shots_this_step == 1
    firing = env.render()
    env.step(spec.make_action(fire=1))  # holding a semi-auto trigger doesn't fire again
    assert env.shots_this_step == 0
    assert not np.array_equal(env.render(), firing)
