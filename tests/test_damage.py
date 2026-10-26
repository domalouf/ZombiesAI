"""The hit detector on real frames: tests/fixtures/damage/frames.npz is cut from the demos at the policy's 128x72.

`hits` is the first frame of each of the 16 hits in the five demos, `lookalikes` the frames that come closest
without being one (fire and smoke, dark red dirt, Nacht's orange help room, blue fog), and `two_hits` 30 frames
in which the player is hit twice half a second apart, the picture going grey between the flashes.
"""

from pathlib import Path

import numpy as np
import pytest

from zombiesai.realgame.damage import DamageConfig, DamageDetector, hit_score

FIXTURE = Path(__file__).parent / "fixtures" / "damage" / "frames.npz"


@pytest.fixture(scope="module")
def frames():
    return dict(np.load(FIXTURE))


def test_every_hit_in_the_demos_scores_above_on_and_no_lookalike_does(frames):
    on = DamageConfig().on
    assert all(hit_score(f) > on for f in frames["hits"])
    for frame, name in zip(frames["lookalikes"], frames["lookalike_names"]):
        assert hit_score(frame) < on - 5, name


def test_a_flash_is_one_hit_and_two_flashes_are_two(frames):
    detector = DamageDetector()
    starts = [i for i, frame in enumerate(frames["two_hits"]) if detector.step(frame)]
    assert starts == [10, 18] and detector.hits == 2


def flat(r, g, b):
    return np.full((72, 128, 3), (r, g, b), np.uint8)


def test_a_held_flash_counts_once_and_the_detector_rearms_only_below_off():
    red, fading, calm = flat(60, 30, 30), flat(45, 30, 30), flat(30, 30, 30)  # scores 30, 15, 0
    assert (hit_score(red), hit_score(fading), hit_score(calm)) == (30.0, 15.0, 0.0)
    detector = DamageDetector()
    assert [detector.step(f) for f in (red, red, fading, red)] == [True, False, False, False]  # one flash
    assert [detector.step(f) for f in (calm, red)] == [False, True] and detector.hits == 2
    detector.reset()
    assert detector.hits == 0 and detector.armed


def test_orange_is_not_red():
    assert hit_score(flat(150, 110, 60)) < 0  # warm light: as much red above blue as a hit, but green with it
