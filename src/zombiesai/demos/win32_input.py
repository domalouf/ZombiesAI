"""Windows Raw Input logger: what the human's hands did, in the units the agent's hands will speak.

Why this and not a convenience library: in a mouse-look FPS the cursor is captured and re-centred every
frame, so cursor positions and their deltas carry no information about how far the player turned. Raw Input
(`WM_INPUT`) reports the device's own relative counts, which is the same unit `SendInput` consumes -- so a
human's 400-count flick and the agent's 400-count flick are the same action. Any recorder built on cursor
deltas produces labels that look fine and mean nothing (PLAN.md, "Demonstrations and BC").

The decoder below is a pure function over the bytes Windows hands back, so it is unit-tested on Linux like
everything else; only `RawInputRecorder` needs Windows, and its message loop and device registration are the
part that still wants verifying against spike S1 on the real machine.
"""

import ctypes
import struct
import sys
import threading
import time
from collections import deque

RIM_TYPEMOUSE, RIM_TYPEKEYBOARD = 0, 1
RIDEV_INPUTSINK = 0x00000100  # keep receiving input while the game, not us, has focus
RID_INPUT = 0x10000003
WM_INPUT, WM_QUIT = 0x00FF, 0x0012
HWND_MESSAGE = -3
USAGE_PAGE_GENERIC = 0x01
USAGE_MOUSE, USAGE_KEYBOARD = 0x02, 0x06

RI_KEY_BREAK = 0x01  # key up rather than down
RI_KEY_E0 = 0x02  # extended key: right-hand ctrl/alt, arrows, numpad enter
MOUSE_BUTTONS = {
    0x0001: ("mouse1", True),
    0x0002: ("mouse1", False),
    0x0004: ("mouse2", True),
    0x0008: ("mouse2", False),
    0x0010: ("mouse3", True),
    0x0020: ("mouse3", False),
    0x0040: ("mouse4", True),
    0x0080: ("mouse4", False),
    0x0100: ("mouse5", True),
    0x0200: ("mouse5", False),
}
# Only the keys a Nacht player uses; anything else is logged by its virtual-key number so a rebind can be
# recovered later rather than silently dropped.
VK_NAMES = {
    0x08: "backspace", 0x09: "tab", 0x0D: "enter", 0x10: "shift", 0x11: "ctrl", 0x12: "alt", 0x14: "capslock",
    0x1B: "escape", 0x20: "space", 0xA0: "shift", 0xA1: "shift", 0xA2: "ctrl", 0xA3: "ctrl", 0xA4: "alt",
    0xA5: "alt", 0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down",
}
VK_NAMES.update({code: chr(code).lower() for code in range(0x30, 0x5B)})  # 0-9 and A-Z
VK_NAMES.update({0x70 + i: f"f{i + 1}" for i in range(12)})

# RAWINPUTHEADER is 16 bytes on 32-bit and 24 on 64-bit (HANDLE and WPARAM both widen).
_HEADER = struct.Struct("=IIQQ" if ctypes.sizeof(ctypes.c_void_p) == 8 else "=IIII")
_MOUSE = struct.Struct("=HxxHHIiiI")  # usFlags, (pad), usButtonFlags, usButtonData, ulRawButtons, lLastX, lLastY, extra
_KEYBOARD = struct.Struct("=HHHHIL")  # MakeCode, Flags, Reserved, VKey, Message, ExtraInformation


def vk_name(vkey: int, extended: bool = False) -> str:
    name = VK_NAMES.get(vkey)
    if name is None:
        return f"vk{vkey:02x}"
    return f"r{name}" if extended and name in ("ctrl", "alt") else name


def decode_raw_input(buffer: bytes, t: float) -> list[dict]:
    """One RAWINPUT buffer -> zero or more log events. Mouse motion and button edges arrive in one report."""
    if len(buffer) < _HEADER.size:
        return []
    kind, _size, _device, _wparam = _HEADER.unpack_from(buffer, 0)
    body = buffer[_HEADER.size :]
    events: list[dict] = []
    if kind == RIM_TYPEMOUSE and len(body) >= _MOUSE.size:
        flags, button_flags, _data, _raw, dx, dy, _extra = _MOUSE.unpack_from(body, 0)
        if dx or dy:
            # usFlags bit 0 set means the device reports absolute coordinates (a tablet or an RDP session).
            # Those are not counts and must not be treated as a turn.
            events.append({"t": t, "type": "mouse", "dx": int(dx), "dy": int(dy), "absolute": bool(flags & 1)})
        for bit, (code, down) in MOUSE_BUTTONS.items():
            if button_flags & bit:
                events.append({"t": t, "type": "button", "code": code, "down": down})
    elif kind == RIM_TYPEKEYBOARD and len(body) >= _KEYBOARD.size:
        _make, flags, _reserved, vkey, _message, _extra = _KEYBOARD.unpack_from(body, 0)
        if vkey != 0xFF:  # 0xFF is the "fake key" half of a pause/print-screen sequence
            code = vk_name(vkey, bool(flags & RI_KEY_E0))
            events.append({"t": t, "type": "key", "code": code, "down": not flags & RI_KEY_BREAK})
    return events


# Pointer-width types. Without explicit prototypes ctypes assumes 32-bit ints, which silently truncates
# window handles and the LPARAM carrying the raw-input packet on 64-bit Windows -- the classic way a
# hand-rolled Win32 binding "works" everywhere except where it matters.
LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t
HANDLE = ctypes.c_void_p


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.c_ushort),
        ("usUsage", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("hwndTarget", HANDLE),
    ]


def _prototypes(user32, kernel32) -> None:
    """Declare argument and return types before calling anything: see the note on LRESULT above."""
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [HANDLE, ctypes.c_uint, WPARAM, LPARAM]
    user32.RegisterClassW.restype = ctypes.c_ushort
    user32.CreateWindowExW.restype = HANDLE
    user32.CreateWindowExW.argtypes = [
        ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        HANDLE, HANDLE, HANDLE, ctypes.c_void_p,
    ]
    user32.RegisterRawInputDevices.restype = ctypes.c_bool
    user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), ctypes.c_uint, ctypes.c_uint]
    user32.GetRawInputData.restype = ctypes.c_uint
    user32.GetRawInputData.argtypes = [
        HANDLE, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint
    ]
    user32.GetMessageW.argtypes = [ctypes.c_void_p, HANDLE, ctypes.c_uint, ctypes.c_uint]
    user32.PostMessageW.argtypes = [HANDLE, ctypes.c_uint, WPARAM, LPARAM]
    kernel32.GetModuleHandleW.restype = HANDLE
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]


class RawInputRecorder:
    """Background thread with a message-only window collecting raw mouse and keyboard reports.

    `drain(start, end)` returns everything logged since the last call, timestamped on `time.monotonic()` --
    the same clock the capture thread stamps frames with, which is the only reason the two can be aligned.
    """

    def __init__(self, keep: int = 1 << 16):
        if sys.platform != "win32":
            raise RuntimeError(
                "Raw Input is a Windows API; on Linux record with a video file and label it with the IDM"
            )
        self.events: deque[dict] = deque(maxlen=keep)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._hwnd = None
        self.dropped = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rawinput", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("raw input thread did not register its devices") from self._error
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            self._pump()
        except BaseException as error:  # the thread is where registration fails; surface it to start()
            self._error = error
            self._ready.set()

    def _pump(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        wndproc_type = ctypes.WINFUNCTYPE(LRESULT, HANDLE, ctypes.c_uint, WPARAM, LPARAM)  # type: ignore[attr-defined]
        _prototypes(user32, kernel32)

        def wndproc(hwnd, message, wparam, lparam):
            if message == WM_INPUT:
                self._collect(user32, lparam)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        self._wndproc = wndproc_type(wndproc)  # kept alive for as long as the window exists

        class WNDCLASS(ctypes.Structure):
            _fields_ = [
                ("style", ctypes.c_uint), ("lpfnWndProc", wndproc_type), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", HANDLE), ("hIcon", HANDLE),
                ("hCursor", HANDLE), ("hbrBackground", HANDLE),
                ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p),
            ]

        cls = WNDCLASS()
        cls.lpfnWndProc = self._wndproc
        cls.lpszClassName = f"ZombiesAIRawInput{threading.get_ident()}"
        cls.hInstance = kernel32.GetModuleHandleW(None)
        if not user32.RegisterClassW(ctypes.byref(cls)):
            raise ctypes.WinError()  # type: ignore[attr-defined]
        self._hwnd = user32.CreateWindowExW(
            0, cls.lpszClassName, cls.lpszClassName, 0, 0, 0, 0, 0,
            ctypes.c_void_p(HWND_MESSAGE), None, cls.hInstance, None,
        )
        if not self._hwnd:
            raise ctypes.WinError()  # type: ignore[attr-defined]

        devices = (RAWINPUTDEVICE * 2)(
            RAWINPUTDEVICE(USAGE_PAGE_GENERIC, USAGE_MOUSE, RIDEV_INPUTSINK, self._hwnd),
            RAWINPUTDEVICE(USAGE_PAGE_GENERIC, USAGE_KEYBOARD, RIDEV_INPUTSINK, self._hwnd),
        )
        if not user32.RegisterRawInputDevices(devices, 2, ctypes.sizeof(RAWINPUTDEVICE)):
            raise ctypes.WinError()  # type: ignore[attr-defined]
        self._ready.set()

        message = ctypes.create_string_buffer(64)  # MSG is 48 bytes on x64; the slack costs nothing
        while user32.GetMessageW(message, None, 0, 0) > 0:
            user32.TranslateMessage(message)
            user32.DispatchMessageW(message)

    def _collect(self, user32, lparam) -> None:
        size = ctypes.c_uint(0)
        handle = ctypes.c_void_p(lparam)
        user32.GetRawInputData(handle, RID_INPUT, None, ctypes.byref(size), _HEADER.size)
        buffer = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputData(handle, RID_INPUT, buffer, ctypes.byref(size), _HEADER.size) != size.value:
            self.dropped += 1
            return
        self.events.extend(decode_raw_input(buffer.raw, time.monotonic()))

    def drain(self, start: float = 0.0, end: float = 0.0) -> list[dict]:
        """Everything logged since the last call. The window arguments are ignored: these events carry real
        timestamps, and the recorder clamps them into whichever decision they fall in."""
        out = list(self.events)
        self.events.clear()
        return out

    def close(self) -> None:
        if self._hwnd is not None:
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_QUIT, 0, 0)  # type: ignore[attr-defined]
            self._hwnd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
