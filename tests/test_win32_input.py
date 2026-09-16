"""The Raw Input decoder, tested off Windows: the struct parsing is pure, only the message loop is not."""

import ctypes
import struct

from zombiesai.demos import win32_input as raw

SIXTY_FOUR_BIT = ctypes.sizeof(ctypes.c_void_p) == 8


def header(kind: int) -> bytes:
    return struct.pack("=IIQQ" if SIXTY_FOUR_BIT else "=IIII", kind, 0, 0, 0)


def mouse_report(dx=0, dy=0, button_flags=0, flags=0) -> bytes:
    return header(raw.RIM_TYPEMOUSE) + struct.pack("=HxxHHIiiI", flags, button_flags, 0, 0, dx, dy, 0)


def key_report(vkey: int, up: bool = False, extended: bool = False) -> bytes:
    key_flags = (raw.RI_KEY_BREAK if up else 0) | (raw.RI_KEY_E0 if extended else 0)
    return header(raw.RIM_TYPEKEYBOARD) + struct.pack("=HHHHIL", 0, key_flags, 0, vkey, 0, 0)


def test_mouse_motion_is_reported_as_raw_counts():
    (event,) = raw.decode_raw_input(mouse_report(dx=40, dy=-7), t=1.0)
    assert event == {"t": 1.0, "type": "mouse", "dx": 40, "dy": -7, "absolute": False}


def test_negative_counts_survive_the_unpack():
    (event,) = raw.decode_raw_input(mouse_report(dx=-400, dy=0), t=0.0)
    assert event["dx"] == -400


def test_an_absolute_device_is_flagged_rather_than_treated_as_a_turn():
    (event,) = raw.decode_raw_input(mouse_report(dx=10, dy=0, flags=1), t=0.0)
    assert event["absolute"] is True


def test_motion_and_a_click_can_arrive_in_one_report():
    events = raw.decode_raw_input(mouse_report(dx=3, dy=0, button_flags=0x0001), t=2.0)
    assert [e["type"] for e in events] == ["mouse", "button"]
    assert events[1] == {"t": 2.0, "type": "button", "code": "mouse1", "down": True}


def test_every_mouse_button_edge_decodes():
    for bit, (code, down) in raw.MOUSE_BUTTONS.items():
        (event,) = raw.decode_raw_input(mouse_report(button_flags=bit), t=0.0)
        assert (event["code"], event["down"]) == (code, down)


def test_keys_decode_to_the_names_the_bindings_use():
    (event,) = raw.decode_raw_input(key_report(ord("W")), t=0.5)
    assert event == {"t": 0.5, "type": "key", "code": "w", "down": True}
    (up,) = raw.decode_raw_input(key_report(ord("W"), up=True), t=0.6)
    assert up["down"] is False
    (shift,) = raw.decode_raw_input(key_report(0xA0), t=0.0)
    assert shift["code"] == "shift"


def test_an_unknown_key_keeps_its_virtual_key_code_instead_of_vanishing():
    (event,) = raw.decode_raw_input(key_report(0xDB), t=0.0)
    assert event["code"] == "vk db".replace(" ", "")


def test_the_extended_flag_separates_the_right_hand_modifiers():
    assert raw.vk_name(0x11, extended=True) == "rctrl"
    assert raw.vk_name(0x11) == "ctrl"


def test_a_truncated_buffer_is_ignored_rather_than_raising():
    assert raw.decode_raw_input(b"", t=0.0) == []
    assert raw.decode_raw_input(header(raw.RIM_TYPEMOUSE)[:8], t=0.0) == []
    assert raw.decode_raw_input(header(raw.RIM_TYPEMOUSE), t=0.0) == []
