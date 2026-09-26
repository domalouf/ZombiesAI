"""Synthetic input into one X server, through the XTEST extension: the hands of one of several game instances.

`uinput.py` makes a device the *compositor* reads, so its input goes to whichever window Hyprland has focused
-- one game at a time, and never while you are using the desktop. Running several games in parallel needs
input that goes to one game and nowhere else. Each instance gets an X server of its own (`instances.py`: a
rootful Xwayland), and XTEST injects straight into that server's input stream: the X server, not the
compositor, decides where it goes, and in a server that holds nothing but the game, it goes to the game.

What was checked on the target machine before this was written (a hidden, never-focused rootful Xwayland on
an Omarchy/Hyprland desktop):

* `XTestFakeRelativeMotionEvent` arrives as XInput2 **raw motion** with exactly the counts sent, on the
  master pointer -- which is what Wine's winex11 reads for a game's mouse-look (DirectInput, raw input) --
  and raw values bypass the server's pointer acceleration, so counts per degree stays a line.
* Key and button events arrive as raw events and as core events to the window holding the X focus, whatever
  Hyprland has focused.

What only the game can answer, and `scripts/spike_instances.py` asks it: whether WaW under Proton accepts
that stream (acquires its DirectInput devices, turns by the counts it is sent).

The names are the recorder's (`demos/evdev_input.py`) and so are the numbers: under the evdev keymap every X
server uses on Linux, an X keycode is the evdev code plus 8. A dispatcher driving this sink therefore presses
exactly the physical keys a uinput sink would -- the same bindings, the same labels.
"""

import ctypes
import ctypes.util
import os

from zombiesai.realgame.uinput import code_for

X_KEYCODE_OFFSET = 8  # X keycode = evdev code + 8 under the evdev keymap
# X core button numbers for the recorder's mouse names; 4-7 are the wheel, as presses.
BUTTONS = {"mouse1": 1, "mouse3": 2, "mouse2": 3, "mouse4": 8, "mouse5": 9}
WHEEL = {"wheelup": 4, "wheeldown": 5, "wheelleft": 6, "wheelright": 7}
CURRENT_TIME = 0
REVERT_TO_PARENT = 2


class XTestError(RuntimeError):
    pass


class _Lib:
    """libX11 + libXtst entry points with prototypes declared (see x11_capture._Xlib for why that matters)."""

    def __init__(self):
        x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        path = ctypes.util.find_library("Xtst") or "libXtst.so.6"
        try:
            xtst = ctypes.CDLL(path)
        except OSError as error:
            raise XTestError("libXtst is not installed (pacman -S libxtst)") from error
        v, ul, i, u = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_uint

        def bind(lib, name, restype, argtypes):
            f = getattr(lib, name)
            f.restype, f.argtypes = restype, argtypes
            return f

        self.open_display = bind(x11, "XOpenDisplay", v, [ctypes.c_char_p])
        self.close_display = bind(x11, "XCloseDisplay", i, [v])
        self.flush = bind(x11, "XFlush", i, [v])
        self.sync = bind(x11, "XSync", i, [v, i])
        self.set_input_focus = bind(x11, "XSetInputFocus", i, [v, ul, i, ul])
        self.get_input_focus = bind(x11, "XGetInputFocus", i, [v, ctypes.POINTER(ul), ctypes.POINTER(i)])
        self.raise_window = bind(x11, "XRaiseWindow", i, [v, ul])
        self.query_extension = bind(xtst, "XTestQueryExtension", i, [v] + [ctypes.POINTER(i)] * 4)
        self.fake_key = bind(xtst, "XTestFakeKeyEvent", i, [v, u, i, ul])
        self.fake_button = bind(xtst, "XTestFakeButtonEvent", i, [v, u, i, ul])
        self.fake_relative_motion = bind(xtst, "XTestFakeRelativeMotionEvent", i, [v, i, i, ul])


_LIB: _Lib | None = None


def _lib() -> _Lib:
    global _LIB
    if _LIB is None:
        _LIB = _Lib()
    return _LIB


def x_keycode(name: str) -> int:
    """The X keycode of a key the recorder names ("w", "shift", "f7")."""
    return code_for(name) + X_KEYCODE_OFFSET


class XTestSink:
    """A virtual mouse and keyboard for one X server. Implements the dispatcher's sink interface
    (`key`, `move`, `sync`, `close`), so `ActionDispatcher(XTestSink(":61"), config)` plays instance 1.

    Unlike the uinput device this has a wheel: "wheelup"/"wheeldown" go out as a press and release of buttons
    4 and 5, the way every X server reports a notch.

    `focus(window)` hands a window the server's input focus. Nothing else in a private server would, and Wine
    only treats a window as the foreground -- the one DirectInput delivers to -- once the server says it has
    focus.
    """

    def __init__(self, display: str, *, lib=None):
        self.display_name = display
        self.lib = lib or _lib()
        self.dpy = self.lib.open_display(display.encode())
        if not self.dpy:
            raise XTestError(f"cannot open X display {display}")
        codes = [ctypes.c_int() for _ in range(4)]
        if not self.lib.query_extension(self.dpy, *(ctypes.byref(c) for c in codes)):
            self.lib.close_display(self.dpy)
            self.dpy = None
            raise XTestError(f"X display {display} has no XTEST extension")
        self.events = 0

    def key(self, code: str, down: bool, t: float = 0.0) -> None:
        name = code.lower()
        if name in BUTTONS:
            self.lib.fake_button(self.dpy, BUTTONS[name], 1 if down else 0, CURRENT_TIME)
        elif name in WHEEL:
            if down:  # a notch is a click, not a hold: the release half is sent with it
                self.lib.fake_button(self.dpy, WHEEL[name], 1, CURRENT_TIME)
                self.lib.fake_button(self.dpy, WHEEL[name], 0, CURRENT_TIME)
        else:
            self.lib.fake_key(self.dpy, x_keycode(name), 1 if down else 0, CURRENT_TIME)
        self.events += 1

    def move(self, dx: int, dy: int, t: float = 0.0) -> None:
        if dx or dy:
            self.lib.fake_relative_motion(self.dpy, int(dx), int(dy), CURRENT_TIME)
            self.events += 1

    def sync(self) -> None:
        self.lib.flush(self.dpy)

    def focused_window(self) -> int:
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        self.lib.get_input_focus(self.dpy, ctypes.byref(window), ctypes.byref(revert))
        return int(window.value)

    def focus(self, window: int) -> bool:
        """Give `window` the input focus if it does not have it. True when it had to be moved."""
        if self.focused_window() == window:
            return False
        self.lib.raise_window(self.dpy, window)
        self.lib.set_input_focus(self.dpy, window, REVERT_TO_PARENT, CURRENT_TIME)
        self.lib.sync(self.dpy, 0)
        return True

    def describe(self) -> dict:
        return {"kind": "xtest", "display": self.display_name}

    def close(self) -> None:
        if getattr(self, "dpy", None):
            self.lib.close_display(self.dpy)
            self.dpy = None


def type_text(sink, text: str, *, hold_s: float = 0.02, sleep=None) -> None:
    """Type `text` key by key (US layout): what the console needs, lower-case letters, digits, space and a
    little punctuation. Upper case and "_" go out with shift held."""
    import time

    sleep = sleep or time.sleep
    for char in text:
        name, shifted = CHAR_KEYS.get(char, (None, False))
        if name is None:
            raise ValueError(f"cannot type {char!r}")
        if shifted:
            sink.key("shift", True)
        sink.key(name, True)
        sink.sync()
        sleep(hold_s)
        sink.key(name, False)
        if shifted:
            sink.key("shift", False)
        sink.sync()
        sleep(hold_s)


def console_command(sink, command: str, *, console_key: str = "grave", settle_s: float = 0.3, sleep=None) -> None:
    """Open WaW's console, type `command`, run it, close the console. Needs `monkeytoy 0` (instances.py passes
    it at launch); without it the key does nothing and the reset falls through to relaunching the game."""
    import time

    sleep = sleep or time.sleep
    for down in (True, False):
        sink.key(console_key, down)
        sink.sync()
    sleep(settle_s)
    type_text(sink, command + "\n", sleep=sleep)
    sleep(settle_s)
    for down in (True, False):
        sink.key(console_key, down)
        sink.sync()


CHAR_KEYS: dict[str, tuple[str, bool]] = {c: (c, False) for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
CHAR_KEYS.update({c.upper(): (c, True) for c in "abcdefghijklmnopqrstuvwxyz"})
CHAR_KEYS.update({
    " ": ("space", False), "_": ("minus", True), "-": ("minus", False), ".": ("dot", False),
    "/": ("slash", False), "\n": ("enter", False), '"': ("apostrophe", True), "=": ("equal", False),
})


def display_env(display: str) -> dict[str, str]:
    """The environment a child process needs to live on `display` and nowhere else: no Wayland socket, so
    neither Wine nor SDL nor Vulkan's WSI can wander off onto the desktop compositor."""
    env = {k: v for k, v in os.environ.items() if k != "WAYLAND_DISPLAY"}
    env["DISPLAY"] = display
    return env
