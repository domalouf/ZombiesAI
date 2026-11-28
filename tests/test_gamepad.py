"""The gamepad bridge, tested without a pad: fake devices and a fake sink, plus one real uinput pad if allowed."""

import fcntl
import json
import os
import struct
import time
from pathlib import Path

import pytest

from zombiesai.demos import gamepad as gp
from zombiesai.demos.evdev_input import _EVENT, EV_KEY, EV_SYN
from zombiesai.demos.inputs import InputConfig, InputFolder, actions_from
from zombiesai.viz.live_bus import Publisher, Subscriber

ROOT = Path(__file__).resolve().parents[1]
CPD = 9.09

# An Xbox pad as xpad lists it, the ASRock LED controller on this machine (a js handler, no gamepad), and a
# DirectInput pad with its right stick on Z/RZ and buttons from BTN_TRIGGER.
PROC = """\
I: Bus=0003 Vendor=045e Product=02ea Version=0301
N: Name="Microsoft X-Box One S pad"
H: Handlers=event21 js1
B: PROP=0
B: EV=20000b
B: KEY=7cdb000000000000 0 0 0 0
B: ABS=3003f

I: Bus=0003 Vendor=26ce Product=01a2 Version=0110
N: Name="ASRock LED Controller"
H: Handlers=kbd event2 js0
B: PROP=0
B: EV=1b
B: KEY=c000 0 0 40000001000000 1200000000 c00000000000000 100000800000000 40000010cc00 10168000000000 0
B: ABS=100000301ff

I: Bus=0003 Vendor=0079 Product=0006 Version=0110
N: Name="DragonRise Inc. Generic USB Joystick"
H: Handlers=event22 js2
B: EV=1b
B: KEY=fff00000000 0 0 0 0
B: ABS=30027

I: Bus=0003 Vendor=1209 Product=0001 Version=0001
N: Name="zombiesai-pad-bridge"
H: Handlers=sysrq kbd mouse3 event23
B: EV=17
"""


def test_only_real_gamepads_are_found():
    pads = gp.parse_pads(PROC)
    assert [(p.path, p.name) for p in pads] == [
        ("/dev/input/event21", "Microsoft X-Box One S pad"),
        ("/dev/input/event22", "DragonRise Inc. Generic USB Joystick"),
    ]
    xbox, generic = pads
    assert gp.BTN_SOUTH in xbox.key_codes and gp.BTN_TRIGGER in generic.key_codes
    assert gp.resolve_axes("auto", xbox.abs_codes) == {
        "left_x": 0, "left_y": 1, "right_x": 3, "right_y": 4, "left_trigger": 2, "right_trigger": 5}
    assert gp.resolve_axes("auto", generic.abs_codes) == {"left_x": 0, "left_y": 1, "right_x": 2, "right_y": 5}


def test_bitmaps_read_most_significant_word_first():
    assert gp.parse_bitmap("1 0") == frozenset({64})
    assert gp.parse_bitmap("3003f") == frozenset({0, 1, 2, 3, 4, 5, 16, 17})


def test_choosing_a_path_that_is_not_a_pad_says_what_is():
    with pytest.raises(gp.PadError, match="event21"):
        gp.choose_pad("/dev/input/event2", gp.parse_pads(PROC))
    with pytest.raises(gp.PadError, match="no gamepad"):
        gp.choose_pad(None, [])


def test_axes_normalise_about_their_own_centre():
    assert gp.normalize_stick(0, -32768, 32767) == pytest.approx(0.0, abs=1e-4)
    assert gp.normalize_stick(32767, -32768, 32767) == 1.0
    assert gp.normalize_stick(-32768, -32768, 32767) == -1.0
    assert gp.normalize_stick(128, 0, 255) == pytest.approx(0.0039, abs=1e-3)  # DualSense: 0..255
    assert gp.normalize_trigger(1023, 0, 1023) == 1.0 and gp.normalize_trigger(0, 0, 1023) == 0.0


def test_the_deadzone_is_radial_and_the_curve_keeps_direction():
    assert gp.shape_stick(0.1, 0.05, 0.12) == (0.0, 0.0)
    x, y = gp.shape_stick(0.6, 0.6, 0.12)  # a diagonal stays a diagonal: no snap onto the axes
    assert x == pytest.approx(y)
    assert gp.shape_stick(1.0, 0.0, 0.12) == (1.0, 0.0)
    half, _ = gp.shape_stick(0.56, 0.0, 0.12, 2.0)
    assert half == pytest.approx(0.25)  # (0.56 - 0.12) / 0.88 = 0.5, squared


def test_slow_aim_is_carried_not_rounded_away():
    look = gp.LookConfig(deadzone=0.0, exponent=1.0, max_yaw_deg_s=10.0, max_pitch_deg_s=10.0)
    integ = gp.LookIntegrator(look, CPD)
    dt = 1 / 250
    per_tick = 10.0 * dt * CPD  # 0.36 counts a tick
    total = sum(integ.step(1.0, 0.0, dt)[0] for _ in range(250))
    assert per_tick < 1 and total == int(10.0 * CPD)  # one second at full tilt: 10 degrees, to the count


def test_full_tilt_turns_the_configured_rate_in_degrees_and_up_is_negative_dy():
    look = gp.LookConfig(deadzone=0.0, exponent=1.0, max_yaw_deg_s=240.0, max_pitch_deg_s=80.0)
    integ = gp.LookIntegrator(look, CPD)
    dx = sum(integ.step(-1.0, 0.0, 1 / 250)[0] for _ in range(250))
    dy = sum(integ.step(0.0, 1.0, 1 / 250)[1] for _ in range(250))
    assert dx / CPD == pytest.approx(-240.0, abs=0.2)
    assert dy < 0 and -dy / CPD == pytest.approx(80.0, abs=0.2)  # stick up looks up: mouse dy negative
    inverted = gp.LookIntegrator(gp.LookConfig(deadzone=0.0, exponent=1.0, invert_pitch=True), CPD)
    assert inverted.step(0.0, 1.0, 0.1)[1] > 0


def test_releasing_the_stick_drops_the_leftover_fraction():
    integ = gp.LookIntegrator(gp.LookConfig(deadzone=0.1, exponent=1.0, max_yaw_deg_s=10.0), CPD)
    integ.step(0.5, 0.0, 1 / 250)
    assert integ.carry[0] != 0
    integ.step(0.0, 0.0, 1 / 250)
    assert integ.carry == [0.0, 0.0]


def test_the_default_config_loads_and_fits_the_action_space():
    config = gp.PadConfig.load(ROOT / "configs" / "pad_bindings.json")
    assert config.spec_warnings() == []
    bindings = json.loads((ROOT / "configs" / "waw_bindings.json").read_text())
    controls = {bindings[c] for c in config.buttons.values() if c in bindings}
    assert {"forward", "back", "left", "right", "fire", "ads", "sprint", "use", "reload", "melee", "grenade",
            "swap"} <= controls  # every control the policy has can be played from the pad
    assert config.buttons["dpad_down"] == "f8"  # the mark key, reachable without the keyboard
    fast = gp.PadConfig(look=gp.LookConfig(max_pitch_deg_s=200))
    assert any("max_pitch_deg_s" in w for w in fast.spec_warnings())


def test_configs_that_cannot_be_sent_are_refused():
    with pytest.raises(gp.PadError, match="wheel"):
        gp.PadConfig.from_dict({"buttons": {"north": "wheelup"}})
    with pytest.raises(gp.PadError, match="no evdev code"):
        gp.PadConfig.from_dict({"buttons": {"north": "notakey"}})
    with pytest.raises(gp.PadError, match="not bound"):
        gp.PadConfig.from_dict({"buttons": {}, "toggles": {"l3": "left_stick_up"}})


# ------------------------------------------------------------------------------------------------ the bridge

ABS = {0: gp.AbsInfo(0, -32768, 32767, 16, 128), 1: gp.AbsInfo(0, -32768, 32767, 16, 128),
       3: gp.AbsInfo(0, -32768, 32767, 16, 128), 4: gp.AbsInfo(0, -32768, 32767, 16, 128),
       2: gp.AbsInfo(0, 0, 1023), 5: gp.AbsInfo(0, 0, 1023), 0x10: gp.AbsInfo(0, -1, 1), 0x11: gp.AbsInfo(0, -1, 1)}


class FakePad:
    name, path = "Fake pad", "/dev/input/event99"

    def __init__(self):
        self.absinfo = dict(ABS)
        self.pending: list[tuple[int, int, int]] = []
        self.gone = False
        self.closed = False

    def push(self, *events):
        self.pending.extend(events)

    def read(self):
        if self.gone:
            raise gp.PadLost(19, "gone")
        out, self.pending = self.pending, []
        return out

    def close(self):
        self.closed = True


class FakeSink:
    def __init__(self):
        self.sent = []

    def key(self, code, down, t=0.0):
        self.sent.append(("key", code, down))

    def move(self, dx, dy, t=0.0):
        self.sent.append(("move", dx, dy))

    def sync(self):
        self.sent.append(("sync",))


def bridge(pad=None, **kwargs):
    config = gp.PadConfig.load(ROOT / "configs" / "pad_bindings.json")
    return gp.PadBridge(pad or FakePad(), config, CPD, FakeSink(), **kwargs)


def test_buttons_triggers_and_the_left_stick_press_their_keys():
    b = bridge()
    b.pad.push((EV_KEY, 0x134, 1), (gp.EV_ABS, 5, 1000), (gp.EV_ABS, 1, -30000), (gp.EV_ABS, 0, 30000))
    b.tick(1.0)
    assert b.emitting == {"r", "mouse1", "w", "d"}
    events = b.take()
    assert {(e["type"], e["code"], e["down"]) for e in events} == {
        ("key", "r", True), ("button", "mouse1", True), ("key", "w", True), ("key", "d", True)}
    assert all(e["t"] == 1.0 for e in events)
    b.pad.push((EV_KEY, 0x134, 0), (gp.EV_ABS, 0, 0))
    b.tick(1.004)
    assert b.emitting == {"mouse1", "w"}
    assert {(e["code"], e["down"]) for e in b.take()} == {("r", False), ("d", False)}
    assert b.take() == []  # taken once


def test_the_stick_and_triggers_have_hysteresis():
    b = bridge()
    b.pad.push((gp.EV_ABS, 5, 400))  # 0.39: over press (0.35)
    b.tick(1.0)
    assert "mouse1" in b.emitting
    b.pad.push((gp.EV_ABS, 5, 300))  # 0.29: between release and press, stays held
    b.tick(1.004)
    assert "mouse1" in b.emitting
    b.pad.push((gp.EV_ABS, 5, 200))  # 0.2: released
    b.tick(1.008)
    assert "mouse1" not in b.emitting


def test_l3_toggles_sprint_until_the_stick_stops_pushing_forward():
    b = bridge()
    b.pad.push((gp.EV_ABS, 1, -32000), (EV_KEY, 0x13D, 1))
    b.tick(1.0)
    b.pad.push((EV_KEY, 0x13D, 0))
    b.tick(1.004)
    assert "shift" in b.emitting  # still sprinting after the click is let go
    b.pad.push((gp.EV_ABS, 1, 0))
    b.tick(1.008)
    assert "shift" not in b.emitting and "w" not in b.emitting


def test_the_right_stick_becomes_logged_mouse_counts():
    b = bridge()
    b.pad.push((gp.EV_ABS, 3, 32767))
    for k in range(250):
        b.tick(1.0 + k / 250)
    moves = [e for e in b.take() if e["type"] == "mouse"]
    assert sum(e["dx"] for e in moves) / CPD == pytest.approx(240.0, rel=0.01)
    assert sum(1 for s in b.sink.sent if s[0] == "move") == len(moves)  # every count sent is logged


def test_the_dpad_hat_and_digital_triggers_are_buttons():
    b = bridge()
    b.pad.push((gp.EV_ABS, 0x11, 1), (EV_KEY, 0x138, 1))  # hat down, BTN_TL2
    b.tick(1.0)
    assert {"f8", "mouse2"} <= b.emitting


def test_a_lost_pad_releases_everything_and_comes_back():
    pad = FakePad()
    again = FakePad()
    b = bridge(pad, reopen=lambda: again)
    pad.push((EV_KEY, 0x134, 1), (gp.EV_ABS, 5, 1000))
    b.tick(1.0)
    b.take()
    pad.gone = True
    b.tick(1.004)
    assert b.emitting == set() and pad.closed
    assert {(e["code"], e["down"]) for e in b.take()} == {("r", False), ("mouse1", False)}
    b.tick(1.5)  # under a second: not tried yet
    assert b.pad is None
    b.tick(2.1)
    assert b.pad is again
    again.push((EV_KEY, 0x133, 1))
    b.tick(2.2)
    assert b.emitting == {"1"}


def test_the_pad_message_has_the_overlays_schema(tmp_path):
    sub = Subscriber(tmp_path / "record.sock")
    b = bridge(publisher=Publisher(tmp_path / "record.sock"))
    b.pad.push((gp.EV_ABS, 3, 32767), (gp.EV_ABS, 1, -32767), (gp.EV_ABS, 2, 1023), (EV_KEY, 0x130, 1))
    b.tick(5.0)
    b.tick(5.001)  # inside 1/60 s: not published again
    assert sub.poll()
    message = sub.get("pad")
    assert set(message) == {"kind", "t", "device", "sticks", "triggers", "buttons", "emitting"}
    assert message["device"] == "Fake pad" and message["t"] == 5.0
    assert message["sticks"]["right"] == [1.0, 0.0] and message["sticks"]["left"][1] == pytest.approx(1.0, abs=1e-3)
    assert message["triggers"] == {"left": 1.0, "right": 0.0}
    assert "south" in message["buttons"] and "lt" in message["buttons"]
    assert message["emitting"] == sorted(b.emitting)


class FakeInputs:
    devices, monotonic = [], True

    def __init__(self, events):
        self.events = events
        self.closed = False

    def drain(self, start=0.0, end=0.0):
        out, self.events = self.events, []
        return out

    def describe(self):
        return {"kind": "fake"}

    def close(self):
        self.closed = True


def test_the_merged_source_logs_every_event_once_in_time_order():
    b = bridge()
    inputs = FakeInputs([{"t": 1.002, "type": "key", "code": "e", "down": True}])
    source = gp.PadInput(inputs, b)
    b.pad.push((EV_KEY, 0x134, 1))
    b.tick(1.0)
    b.pad.push((EV_KEY, 0x134, 0))
    b.tick(1.004)
    events = source.drain()
    assert [e["t"] for e in events] == [1.0, 1.002, 1.004]
    assert source.drain() == []
    source.close()
    assert inputs.closed


def test_the_bridge_log_labels_exactly_like_a_keyboard_and_mouse():
    b = bridge()
    b.pad.push((gp.EV_ABS, 1, -32767), (gp.EV_ABS, 5, 1023), (gp.EV_ABS, 3, 16384))
    for k in range(17):
        b.tick(10.0 + k / 250)
    config = InputConfig(counts_per_degree=CPD,
                         bindings=json.loads((ROOT / "configs" / "waw_bindings.json").read_text()))
    held, presses, counts = InputFolder(config).feed(b.take(), 10.0, 10.0 + 1 / 15)
    labels = actions_from(held, presses, counts, config)
    from zombiesai import spec

    a = labels.actions[0]
    assert spec.FORWARD_VALUES[a[spec.FORWARD]] == 1 and a[spec.FIRE] == 1
    assert labels.yaw_deg[0] > 0  # right stick right: a right turn


def test_closing_releases_held_keys_and_the_devices():
    pad = FakePad()
    b = bridge(pad)
    pad.push((EV_KEY, 0x134, 1))
    b.tick(1.0)
    b.close()
    assert ("key", "r", False) in b.sink.sent and pad.closed


def test_the_thread_ticks_and_stops_with_nothing_held():
    pad = FakePad()
    b = bridge(pad)
    pad.push((EV_KEY, 0x133, 1))
    b.start()
    time.sleep(0.05)
    assert b.emitting == {"1"}
    b.close()
    assert not b.thread.is_alive() and b.emitting == set()


# ------------------------------------------------------------------------------------------------ a real pad

UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_ABSBIT = 0x40045564, 0x40045565, 0x40045567
UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502


def fake_gamepad():
    """A uinput gamepad: two sticks, analog triggers and the face buttons. No keys or mouse, so nothing on the
    desktop reacts to it."""
    fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
    fcntl.ioctl(fd, UI_SET_EVBIT, gp.EV_ABS)
    for code in range(0x130, 0x13F):
        fcntl.ioctl(fd, UI_SET_KEYBIT, code)
    absmax, absmin = [0] * 64, [0] * 64
    for axis in (0, 1, 3, 4):
        fcntl.ioctl(fd, UI_SET_ABSBIT, axis)
        absmin[axis], absmax[axis] = -32768, 32767
    for axis in (2, 5):
        fcntl.ioctl(fd, UI_SET_ABSBIT, axis)
        absmin[axis], absmax[axis] = 0, 1023
    name = b"zombiesai-test-gamepad"  # a zombiesai name: never mistaken for the player's own
    os.write(fd, struct.pack("=80sHHHHI" + "i" * 256, name, 3, 0x1209, 0x0002, 1, 0,
                             *absmax, *absmin, *([0] * 64), *([0] * 64)))
    fcntl.ioctl(fd, UI_DEV_CREATE)
    return fd


def write(fd, kind, code, value):
    os.write(fd, _EVENT.pack(0, 0, kind, code, value))


@pytest.mark.skipif(not os.access("/dev/uinput", os.W_OK), reason="/dev/uinput is not writable here")
def test_a_real_uinput_pad_is_grabbed_read_and_bridged():
    fd = fake_gamepad()
    try:
        info = None
        for _ in range(50):
            text = Path("/proc/bus/input/devices").read_text().replace("zombiesai-test-gamepad", "test-gamepad")
            info = next((p for p in gp.parse_pads(text) if p.name == "test-gamepad"), None)
            if info and os.access(info.path, os.R_OK):
                break
            time.sleep(0.05)
        assert info is not None, "the fake pad never showed up"
        pad = gp.PadDevice(info)
        assert pad.grabbed and pad.absinfo[3].maximum == 32767 and pad.absinfo[5].maximum == 1023
        b = bridge(pad)
        write(fd, gp.EV_ABS, 3, 32767)
        write(fd, EV_KEY, 0x134, 1)
        write(fd, EV_SYN, 0, 0)
        time.sleep(0.05)
        b.tick(1.0)
        b.tick(1.1)
        assert "r" in b.emitting
        assert any(e["type"] == "mouse" and e["dx"] > 0 for e in b.take())
        os.close(fd)  # unplugged
        fd = None
        time.sleep(0.05)
        b.tick(1.2)
        assert b.pad is None and b.emitting == set()
    finally:
        if fd is not None:
            fcntl.ioctl(fd, UI_DEV_DESTROY)
            os.close(fd)
