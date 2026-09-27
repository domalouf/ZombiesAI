import numpy as np

from zombiesai.realgame.console import FILL_RGB, console_open
from zombiesai.realgame.xtest import console_command


def bar(h=22, w=400):
    crop = np.full((h, w, 3), 40, np.uint8)
    crop[3:20] = FILL_RGB.astype(np.uint8)
    crop[3] = (32, 32, 26)
    crop[11:15, 5:200] = 220  # the input line's text
    return crop


def test_the_console_bar_is_seen_and_a_scene_is_not():
    rng = np.random.default_rng(0)
    assert console_open(bar())
    assert not console_open(np.full((22, 400, 3), 3, np.uint8))  # a loading screen
    assert not console_open(rng.integers(0, 255, (22, 400, 3), dtype=np.uint8))  # any busy scene
    assert not console_open(np.full((22, 400, 3), (80, 64, 51), np.uint8))  # the right flatness, wrong colour
    assert not console_open(None)


class Sink:
    def __init__(self, console):
        self.console, self.keys = console, []

    def key(self, name, down, t=0.0):
        if down:
            self.keys.append(name)
            if name == "grave":
                self.console["open"] = not self.console["open"]

    def sync(self):
        pass


def test_console_command_looks_before_it_toggles():
    for start_open in (False, True):
        console = {"open": start_open}
        sink = Sink(console)
        assert console_command(sink, "map x", is_open=lambda: console["open"], sleep=lambda s: None)
        typed = [k for k in sink.keys if k != "grave"]
        assert typed[:3] == ["m", "a", "p"] and typed[-1] == "enter"
        assert sink.keys.count("grave") == (2 if not start_open else 1) and not console["open"]


def test_nothing_is_typed_into_the_game_when_the_console_will_not_open():
    sink = Sink({"open": False})
    assert not console_command(sink, "map x", is_open=lambda: False, sleep=lambda s: None)
    assert sink.keys == ["grave"]
