"""The Linux raw-input decoder, tested without touching /dev/input."""

import struct

import pytest

from zombiesai.demos import evdev_input as ev
from zombiesai.demos.inputs import DEFAULT_BINDINGS

PROC_DEVICES = """\
I: Bus=0019 Vendor=0000 Product=0001 Version=0000
N: Name="Power Button"
H: Handlers=kbd event0
B: EV=3

I: Bus=0003 Vendor=046d Product=c08b Version=0111
N: Name="Logitech G502 HERO Gaming Mouse"
H: Handlers=mouse0 event4
B: EV=17

I: Bus=0003 Vendor=1532 Product=0226 Version=0111
N: Name="Razer BlackWidow"
H: Handlers=sysrq kbd event5 leds
B: EV=120013

I: Bus=0003 Vendor=1209 Product=0001 Version=0001
N: Name="zombiesai-virtual-input"
H: Handlers=mouse2 event9
B: EV=17
"""


def event(kind, code, value, seconds=1.0):
    micros = int(round((seconds % 1) * 1e6))
    return ev._EVENT.pack(int(seconds), micros, kind, code, value)


def test_mouse_motion_decodes_as_counts_per_axis():
    buffer = event(ev.EV_REL, ev.REL_X, 40) + event(ev.EV_REL, ev.REL_Y, -7) + event(ev.EV_SYN, 0, 0)
    events = ev.decode_events(buffer)
    assert events == [
        {"t": 1.0, "type": "mouse", "dx": 40, "dy": 0},
        {"t": 1.0, "type": "mouse", "dx": 0, "dy": -7},
    ]


def test_the_quantizer_sums_the_two_axes_back_together():
    from zombiesai.demos.inputs import InputConfig, quantize
    from zombiesai import spec

    buffer = event(ev.EV_REL, ev.REL_X, 60, 0.01) + event(ev.EV_REL, ev.REL_Y, -60, 0.01)
    labels = quantize(ev.decode_events(buffer), 0.0, 1, InputConfig(counts_per_degree=10.0))
    assert spec.YAW_BINS_DEG[labels.actions[0][spec.YAW]] == 6.0
    assert spec.PITCH_BINS_DEG[labels.actions[0][spec.PITCH]] == 6.0  # dy counts down, pitch looks up


def test_keys_decode_to_the_names_the_bindings_use():
    assert ev.key_name(17) == "w" and ev.key_name(30) == "a" and ev.key_name(31) == "s" and ev.key_name(32) == "d"
    assert ev.key_name(42) == "shift" and ev.key_name(19) == "r" and ev.key_name(33) == "f"
    assert ev.key_name(0x110) == "mouse1" and ev.key_name(0x111) == "mouse2"
    assert set(DEFAULT_BINDINGS) <= set(ev.KEY_NAMES.values()) | set(ev.BUTTON_NAMES.values())


def test_key_presses_and_releases_decode_but_auto_repeat_does_not():
    """Holding W streams repeats; counting them as presses would read as thirty reloads a second."""
    buffer = (
        event(ev.EV_KEY, 17, ev.KEY_DOWN)
        + event(ev.EV_KEY, 17, ev.KEY_REPEAT)
        + event(ev.EV_KEY, 17, ev.KEY_REPEAT)
        + event(ev.EV_KEY, 17, ev.KEY_UP)
    )
    events = ev.decode_events(buffer)
    assert [(e["code"], e["down"]) for e in events] == [("w", True), ("w", False)]


def test_buttons_are_buttons_and_keys_are_keys():
    events = ev.decode_events(event(ev.EV_KEY, 0x110, 1) + event(ev.EV_KEY, 33, 1))
    assert [e["type"] for e in events] == ["button", "key"]


def test_a_clock_offset_is_applied_when_the_kernel_reports_wall_time():
    (decoded,) = ev.decode_events(event(ev.EV_REL, ev.REL_X, 1, 100.5), offset=-50.0)
    assert decoded["t"] == pytest.approx(50.5)


def test_a_partial_record_at_the_end_of_a_read_is_ignored():
    buffer = event(ev.EV_REL, ev.REL_X, 5) + b"\\x00\\x01\\x02"
    assert len(ev.decode_events(buffer)) == 1


def test_unknown_codes_keep_their_number_instead_of_vanishing():
    assert ev.key_name(700) == "key700"


def test_devices_are_classified_from_proc():
    devices = {d.name: d for d in ev.parse_devices(PROC_DEVICES)}
    assert devices["Logitech G502 HERO Gaming Mouse"].path == "/dev/input/event4"
    assert devices["Logitech G502 HERO Gaming Mouse"].is_mouse
    assert not devices["Logitech G502 HERO Gaming Mouse"].is_keyboard
    assert devices["Razer BlackWidow"].is_keyboard and devices["Razer BlackWidow"].path == "/dev/input/event5"
    # A power button reports keys but no repeat and no motion: not something a player types on.
    assert not devices["Power Button"].is_keyboard and not devices["Power Button"].is_mouse


def test_our_own_virtual_device_can_be_excluded(monkeypatch, tmp_path):
    """Once the agent is playing, its virtual mouse is an input device too -- and a recorder that reads it
    back would log the agent's own actions as if a human had made them. Discovery stays neutral, because
    the dispatcher looks itself up through the same function; the recorder is what excludes."""
    path = tmp_path / "devices"
    path.write_text(PROC_DEVICES)
    monkeypatch.setattr(ev, "Path", lambda _: path)
    assert "zombiesai-virtual-input" in {d.name for d in ev.find_devices()}
    kept = {d.name for d in ev.find_devices(exclude=("zombiesai",))}
    assert "zombiesai-virtual-input" not in kept
    assert {"Logitech G502 HERO Gaming Mouse", "Razer BlackWidow"} == kept


def test_the_event_struct_matches_the_kernels():
    assert ev.EVENT_SIZE == 24 == struct.calcsize("=qqHHi")
