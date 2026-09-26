import numpy as np
import pytest

from zombiesai import spec
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
from zombiesai.realgame.xtest import XTestSink, console_command, display_env, type_text, x_keycode


class FakeLib:
    """libX11/libXtst as far as XTestSink uses them, recording what would have reached the server."""

    def __init__(self, focused: int = 0):
        self.calls: list[tuple] = []
        self.focused = focused

    def open_display(self, name):
        self.calls.append(("open", name))
        return 1

    def close_display(self, dpy):
        self.calls.append(("close",))

    def query_extension(self, dpy, *codes):
        return 1

    def fake_key(self, dpy, keycode, down, delay):
        self.calls.append(("key", keycode, down))

    def fake_button(self, dpy, button, down, delay):
        self.calls.append(("button", button, down))

    def fake_relative_motion(self, dpy, dx, dy, delay):
        self.calls.append(("motion", dx, dy))

    def flush(self, dpy):
        self.calls.append(("flush",))

    def sync(self, dpy, discard):
        pass

    def raise_window(self, dpy, window):
        pass

    def set_input_focus(self, dpy, window, revert, time):
        self.calls.append(("focus", window))
        self.focused = window

    def get_input_focus(self, dpy, window, revert):
        window._obj.value = self.focused


def sent(lib, kinds=("key", "button", "motion")):
    return [c for c in lib.calls if c[0] in kinds]


def test_keycodes_are_evdev_plus_eight():
    assert x_keycode("w") == 17 + 8
    assert x_keycode("grave") == 41 + 8
    assert x_keycode("shift") == 42 + 8  # the left one, as a player presses it


def test_keys_buttons_wheel_and_motion():
    lib = FakeLib()
    sink = XTestSink(":61", lib=lib)
    sink.key("w", True)
    sink.key("mouse1", True)
    sink.key("mouse2", True)  # right button is X button 3
    sink.key("wheeldown", True)  # a notch: press and release together
    sink.key("wheeldown", False)  # nothing more
    sink.move(3, -2)
    sink.move(0, 0)  # nothing
    assert sent(lib) == [("key", 25, 1), ("button", 1, 1), ("button", 3, 1), ("button", 5, 1), ("button", 5, 0),
                         ("motion", 3, -2)]
    assert lib.calls[0] == ("open", b":61")


def test_focus_only_moves_when_needed():
    lib = FakeLib(focused=7)
    sink = XTestSink(":60", lib=lib)
    assert sink.focus(7) is False
    assert sink.focus(9) is True
    assert sink.focused_window() == 9


def test_a_dispatcher_drives_the_sink_like_the_uinput_one():
    lib = FakeLib()
    dispatcher = ActionDispatcher(XTestSink(":60", lib=lib), DispatchConfig(counts_per_degree=10.0))
    action = spec.make_action(forward=1, fire=1, yaw=6.0)
    dispatcher.apply(action, now=0.0)
    dispatcher.flush()
    presses = sent(lib, ("key", "button"))
    assert ("key", x_keycode("w"), 1) in presses and ("button", 1, 1) in presses
    assert sum(c[1] for c in sent(lib, ("motion",))) == 60
    dispatcher.release_all()
    assert ("key", x_keycode("w"), 0) in sent(lib) and ("button", 1, 0) in sent(lib)


def test_typing_shifts_where_it_must():
    lib = FakeLib()
    sink = XTestSink(":60", lib=lib)
    type_text(sink, "a_B", sleep=lambda s: None)
    keys = [(c[1], c[2]) for c in sent(lib, ("key",))]
    shift, minus = x_keycode("shift"), x_keycode("minus")
    assert keys == [(x_keycode("a"), 1), (x_keycode("a"), 0), (shift, 1), (minus, 1), (minus, 0), (shift, 0),
                    (shift, 1), (x_keycode("b"), 1), (x_keycode("b"), 0), (shift, 0)]
    with pytest.raises(ValueError):
        type_text(sink, "é", sleep=lambda s: None)


def test_console_command_opens_types_runs_and_closes():
    lib = FakeLib()
    sink = XTestSink(":60", lib=lib)
    console_command(sink, "map x", sleep=lambda s: None)
    keys = [c[1:] for c in sent(lib, ("key",))]
    grave, enter = x_keycode("grave"), x_keycode("enter")
    assert keys[:2] == [(grave, 1), (grave, 0)] and keys[-2:] == [(grave, 1), (grave, 0)]
    assert (enter, 1) in keys and keys.index((enter, 1)) > keys.index((x_keycode("x"), 1))


def test_display_env_keeps_children_off_the_compositor(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    env = display_env(":63")
    assert env["DISPLAY"] == ":63" and "WAYLAND_DISPLAY" not in env
    assert np.all([k != "WAYLAND_DISPLAY" for k in env])
