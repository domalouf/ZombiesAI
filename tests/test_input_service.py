import socket
import threading

import pytest

from zombiesai import spec

from zombiesai.realgame import input_service
from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
from zombiesai.realgame.input_service import RemoteSink, handle
from zombiesai.realgame.uinput import code_for


class FakeDevice:
    def __init__(self):
        self.events = []

    def key_code(self, code, down):
        self.events.append(("key", code, down))

    def move(self, dx, dy):
        self.events.append(("move", dx, dy))

    def sync(self):
        self.events.append(("sync",))


def connected():
    """A service handler on one end of a socket pair and a RemoteSink on the other."""
    server_end, client_end = socket.socketpair()
    device = FakeDevice()
    thread = threading.Thread(target=handle, args=(server_end, device), daemon=True)
    thread.start()
    sink = RemoteSink.__new__(RemoteSink)
    sink.path, sink.sock = "socketpair", client_end
    return sink, device, thread


def test_events_reach_the_device_in_order():
    sink, device, thread = connected()
    sink.key("w", True)
    sink.move(12, -3)
    sink.sync()
    sink.close()
    thread.join(timeout=2)
    assert device.events[:3] == [("key", code_for("w"), True), ("move", 12, -3), ("sync",)]


def test_a_client_that_hangs_up_mid_press_has_its_keys_released():
    """A player killed with W and the trigger down must not leave them held in the game."""
    sink, device, thread = connected()
    sink.key("w", True)
    sink.key("mouse1", True)
    sink.key("a", True)
    sink.key("a", False)
    sink.close()
    thread.join(timeout=2)
    released = {e[1] for e in device.events[4:] if e[0] == "key" and e[2] is False}
    assert released == {code_for("w"), code_for("mouse1")}
    assert device.events[-1] == ("sync",)


def test_a_dispatcher_drives_the_service_like_any_sink():
    sink, device, thread = connected()
    dispatcher = ActionDispatcher(sink, DispatchConfig(counts_per_degree=9.09))
    dispatcher.apply(spec.make_action(forward=1, fire=1), now=0.0)
    dispatcher.close()
    thread.join(timeout=2)
    pressed = {e[1] for e in device.events if e[0] == "key" and e[2]}
    assert {code_for("w"), code_for("mouse1")} <= pressed


def test_the_service_is_found_running_only_when_something_answers(tmp_path):
    path = tmp_path / "input.sock"
    assert not input_service.running(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    try:
        assert input_service.running(path)
    finally:
        server.close()
    assert not input_service.running(path)  # a stale socket file is not a service
