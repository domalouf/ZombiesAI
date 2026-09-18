"""Synthetic input on Linux: a virtual mouse and keyboard the kernel creates for us.

PLAN.md ranks "synthetic input ignored by the engine" as risk one, with a ladder that climbs from scancode
`SendInput` through a kernel virtual HID driver to a microcontroller pretending to be a USB mouse. On Linux
the middle rung is already in the kernel: `/dev/uinput` creates a real input device, and libinput, the
compositor, XWayland, Wine and the game cannot tell it from the mouse on the desk -- because at the level
they read it, it is one. `REL_X`/`REL_Y` carry mouse counts, which is what the action space is denominated
in and what `demos/evdev_input.py` reads back off a human.

Two things still have to be set up outside this file, and both are in docs/linux.md:

* **Permission.** `/dev/uinput` is root-only by default; a udev rule hands it to the `input` group.
* **Flat pointer acceleration for this device.** libinput accelerates pointers by default, which would make
  counts-to-degrees a curve instead of a line and fail spike S4 for a reason that has nothing to do with the
  game. Hyprland takes a per-device block keyed on the device name below.

Events are encoded with the same struct `demos/evdev_input.py` decodes, so what the agent sends and what a
recording reads are provably the same format -- there is a test that sends an action through here and reads
the human's action back out.
"""

import fcntl
import os
import struct
import time

from zombiesai.demos.evdev_input import (
    BUTTON_NAMES,
    EV_KEY,
    EV_REL,
    EV_SYN,
    KEY_NAMES,
    REL_X,
    REL_Y,
    _EVENT,
    find_devices,
)

SYN_REPORT = 0
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566
BUS_USB = 0x03
DEVICE_NAME = "zombiesai-virtual-input"
# struct uinput_user_dev: name, input_id, ff_effects_max, and four ABS_CNT-long axis tables we leave zeroed.
_USER_DEV = struct.Struct("=80sHHHHI" + "i" * 256)

# Name -> code, the inverse of the decoder's table. Lower codes win, so "shift" is the left one and "ctrl"
# the left one, matching what a keyboard sends when you press the key a player actually presses.
KEY_CODES = {name: code for code, name in sorted(KEY_NAMES.items(), reverse=True)}
KEY_CODES.update({name: code for code, name in BUTTON_NAMES.items()})


def code_for(name: str) -> int:
    code = KEY_CODES.get(name.lower())
    if code is None:
        raise KeyError(f"no evdev code for {name!r}; bind it to a key this keyboard has")
    return code


class UinputDevice:
    """A virtual mouse and keyboard. Implements the dispatcher's sink interface.

    Pass `fd` to write the event stream somewhere other than the kernel -- a pipe or a file -- which is how
    the encoding is tested without needing /dev/uinput or root.
    """

    def __init__(
        self,
        codes=(),
        *,
        name: str = DEVICE_NAME,
        fd: int | None = None,
        create: bool | None = None,
        settle_s: float = 0.35,
        path: str = "/dev/uinput",
    ):
        self.name = name
        self.codes = sorted({code_for(c) for c in codes} | set(BUTTON_NAMES))
        self._owned = fd is None
        if fd is None:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except FileNotFoundError as error:
                raise RuntimeError(f"{path} does not exist; run `sudo modprobe uinput` (see docs/linux.md)") from error
            except PermissionError as error:
                raise PermissionError(
                    f"cannot open {path}: install the udev rule in docs/linux.md, or run as root"
                ) from error
        self.fd = fd
        self.created = False
        if create if create is not None else self._owned:
            self._create(settle_s)

    def _create(self, settle_s: float) -> None:
        for kind in (EV_KEY, EV_REL):
            fcntl.ioctl(self.fd, UI_SET_EVBIT, struct.pack("=i", kind))
        for code in self.codes:
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, struct.pack("=i", code))
        for axis in (REL_X, REL_Y):
            fcntl.ioctl(self.fd, UI_SET_RELBIT, struct.pack("=i", axis))
        os.write(self.fd, _USER_DEV.pack(self.name.encode()[:79], BUS_USB, 0x1209, 0x0001, 1, 0, *([0] * 256)))
        fcntl.ioctl(self.fd, UI_DEV_CREATE)
        self.created = True
        # udev, libinput and the compositor all have to notice the new device; events sent before they do
        # are delivered nowhere and look exactly like "synthetic input doesn't work".
        time.sleep(settle_s)

    def _write(self, kind: int, code: int, value: int) -> None:
        os.write(self.fd, _EVENT.pack(0, 0, kind, code, value))  # the kernel stamps uinput events itself

    def key(self, code: str, down: bool, t: float = 0.0) -> None:
        self._write(EV_KEY, code_for(code), 1 if down else 0)

    def move(self, dx: int, dy: int, t: float = 0.0) -> None:
        if dx:
            self._write(EV_REL, REL_X, int(dx))
        if dy:
            self._write(EV_REL, REL_Y, int(dy))

    def sync(self) -> None:
        self._write(EV_SYN, SYN_REPORT, 0)

    def device_path(self) -> str | None:
        """Where the kernel put us, by looking ourselves up the same way the recorder finds a real mouse."""
        try:
            return next((d.path for d in find_devices() if d.name == self.name), None)
        except OSError:
            return None

    def describe(self) -> dict:
        return {"kind": "uinput", "name": self.name, "path": self.device_path(), "codes": len(self.codes)}

    def close(self) -> None:
        if getattr(self, "fd", None) is None:
            return
        try:
            if self.created:
                fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        finally:
            if self._owned:
                os.close(self.fd)
            self.fd = None


def open_dispatcher(config=None, **kwargs):
    """A dispatcher wired to a real virtual device, registering exactly the keys the bindings use."""
    from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig

    config = config or DispatchConfig()
    device = UinputDevice(config.codes.values(), **kwargs)
    return ActionDispatcher(device, config)
