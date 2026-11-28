"""The hit detector on real frames: tests/fixtures/damage/frames.npz, all at the policy's 128x72.

* `hits` -- the first frame of each of the 16 hits in the five demos (hard hits: the screen floods dark red);
* `lookalikes` -- the frames that come closest to that without being one: fire and smoke, dark red dirt,
  Nacht's orange help room, blue fog;
* `two_hits` -- 30 demo frames in which the player is hit twice half a second apart;
* `live_faint_then_down` -- the last 24 steps of an idle player on the fleet (2026-10-26): a first hit that only
  warms a blue-lit picture for two steps (index 8), then the hit that downs (index 20);
* `glow_fades` -- a power-up's green glow fading out, twice: "red above green" rises everywhere, as in a hit.
"""

from pathlib import Path

import numpy as np
import pytest

from zombiesai.realgame.damage import DamageConfig, DamageDetector, hit_score

FIXTURE = Path(__file__).parent / "fixtures" / "damage" / "frames.npz"


@pytest.fixture(scope="module")
def frames():
    return dict(np.load(FIXTURE))


def hits_in(sequence) -> list[int]:
    detector = DamageDetector()
    return [i for i, frame in enumerate(sequence) if detector.step(frame)]


def test_every_hard_hit_in_the_demos_is_red_enough_and_no_lookalike_is(frames):
    on = DamageConfig().on
    assert all(hit_score(f) > on for f in frames["hits"])
    for frame, name in zip(frames["lookalikes"], frames["lookalike_names"]):
        assert hit_score(frame) < on - 5, name


def test_a_faint_first_hit_is_caught_by_its_rise_and_the_down_after_it(frames):
    assert hits_in(frames["live_faint_then_down"]) == [8, 20]
    assert hit_score(frames["live_faint_then_down"][8]) < DamageConfig().off  # never red by level alone


def test_a_green_glow_fading_out_is_not_a_hit(frames):
    assert hits_in(frames["glow_fades"]) == []


def test_two_hits_half_a_second_apart_are_two(frames):
    assert hits_in(frames["two_hits"]) == [10, 18]


def flat(r, g, b):
    return np.full((72, 128, 3), (r, g, b), np.uint8)


def test_a_rise_everywhere_fires_once_then_waits_out_the_refractory():
    cool, tinted = flat(40, 50, 50), flat(54, 45, 45)  # red above the others: -10, then +9 (a rise of 19)
    assert hit_score(tinted) < DamageConfig().off
    assert [DamageDetector().step(f) for f in (cool,)] == [False]
    detector = DamageDetector()
    steps = [detector.step(f) for f in (cool, cool, tinted, cool, tinted, cool, cool, cool, tinted)]
    assert steps == [False, False, True, False, False, False, False, False, True]  # the second inside 3 steps
    assert detector.hits == 2


def test_a_rise_in_only_part_of_the_picture_is_not_a_hit():
    cool = flat(40, 50, 50)
    half = cool.copy()
    half[:, :64] = (90, 40, 40)  # something red walks into the left half
    assert [DamageDetector().step(f) for f in (cool,)] == [False]
    detector = DamageDetector()
    assert [detector.step(f) for f in (cool, half)] == [False, False]


def test_a_red_level_held_counts_once_and_rearms_only_below_off():
    red, fading, calm = flat(60, 30, 30), flat(45, 30, 30), flat(30, 30, 30)  # scores 30, 15, 0
    assert (hit_score(red), hit_score(fading), hit_score(calm)) == (30.0, 15.0, 0.0)
    detector = DamageDetector()
    assert [detector.step(f) for f in (red,) * 5 + (fading,) * 4] == [True] + [False] * 8
    # Creeping back past `on` a few levels a step is no rise, and the level never fell below `off`: no new hit.
    creep = [flat(45 + k, 30, 30) for k in (3, 6, 9, 12)]
    assert [detector.step(f) for f in creep] == [False] * 4
    assert [detector.step(f) for f in (calm,) * 4 + (red,)] == [False] * 4 + [True] and detector.hits == 2
    detector.reset()
    assert detector.hits == 0


def test_a_dimmed_flash_flaring_up_everywhere_again_is_a_second_hit():
    red, fading = flat(60, 30, 30), flat(45, 30, 30)
    detector = DamageDetector()
    assert [detector.step(f) for f in (red,) + (fading,) * 4 + (red,)] == [True] + [False] * 4 + [True]


def test_orange_is_not_red():
    assert hit_score(flat(150, 110, 60)) < 0  # warm light: as much red above blue as a hit, but green with it
