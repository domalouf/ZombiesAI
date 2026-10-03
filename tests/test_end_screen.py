"""The game-over scoreboard reader, on real screens.

tests/fixtures/end_screen/rows.npz holds the scoreboard (header and the player's row) cut from three 1440p
game-over screens of rl9's actors: 510 / 0 / 0 over the map, 500 / 4 / 0 (kills that earned no points), and
500 / 0 / 0 on the black the screen fades to. `box` is where they were cut (top, bottom, left, right).
"""

from pathlib import Path

import numpy as np
import pytest

from zombiesai.realgame.end_screen import EndScreen, read_end_screen

FIXTURE = Path(__file__).parent / "fixtures" / "end_screen" / "rows.npz"
EXPECTED = {"over_scene_510_0_0": (510, 0, 0), "kills_500_4_0": (500, 4, 0), "black_500_0_0": (500, 0, 0)}


def frames():
    with np.load(FIXTURE) as z:
        top, bottom, left, right = z["box"].tolist()
        h, w = z["frame"].tolist()
        for name in EXPECTED:
            frame = np.zeros((h, w, 3), np.uint8)
            frame[top:bottom, left:right] = z[name]
            yield name, frame


@pytest.mark.parametrize("name", EXPECTED)
def test_each_real_screen_reads_as_drawn(name):
    frame = dict(frames())[name]
    read = read_end_screen(frame)
    assert read is not None and read.values == EXPECTED[name]
    assert read.confidence >= 0.5


def test_the_same_screen_reads_at_1080p():
    from zombiesai.demos.frames import area_resize

    frame = dict(frames())["kills_500_4_0"]
    read = read_end_screen(area_resize(frame, 1080, 1920))
    assert read is None or read.values == (500, 4, 0)  # refused or right, never another number


def test_no_scoreboard_no_read():
    assert read_end_screen(None) is None
    assert read_end_screen(np.zeros((1440, 2560, 3), np.uint8)) is None
    frame = dict(frames())["over_scene_510_0_0"]
    with np.load(FIXTURE) as z:
        top = int(z["box"][0])
    frame[top : top + 50] = 0  # the header gone: the row alone is not the scoreboard
    assert read_end_screen(frame) is None


def test_a_column_that_does_not_read_refuses_the_whole_row():
    frame = dict(frames())["kills_500_4_0"].copy()
    # A smear of white over the Kills number (1833-1845 at 1440p): not a digit, so no numbers at all.
    frame[334:352, 1828:1860] = 255
    assert read_end_screen(frame) is None


def test_values_are_the_three_numbers():
    assert EndScreen(1230, 14, 3, 0.9).values == (1230, 14, 3)
