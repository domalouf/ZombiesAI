"""Screen capture on Linux: MIT-SHM grabs of an X11 window, which under a Wayland session means XWayland.

Proton runs the game as an XWayland client, so on Omarchy (Hyprland) the game is still a real X11 window with
a real XID -- and an X11 window can be read synchronously, in-process, with no portal, no permission dialog
and no async frame callbacks. That is worth a lot at 15 Hz with a 66 ms budget: `grab()` returns the pixels
by the time it returns, so `capture -> downsample -> HUD parse -> infer -> dispatch` stays a straight line.

The fast path is `XShmGetImage` into a shared-memory segment the X server writes directly -- one copy instead
of a round trip through the socket, which is the difference between ~2 ms and ~20 ms at 1080p. If the
extension is missing (a remote display, a locked-down server) it falls back to `XGetImage`, which is correct
but slow enough to fail spike S2. Which one you got is in `describe()`, so a slow capture is never a mystery.

Grabbing the whole root window does not work on a Wayland compositor -- XWayland's root is not the desktop.
Grab the game's window instead: `X11Grabber(window="World at War")`.
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass

import numpy as np

ZPIXMAP = 2
ALL_PLANES = 0xFFFFFFFF
IPC_CREAT, IPC_RMID = 0o1000, 0
ANY_PROPERTY_TYPE = 0


class X11Error(RuntimeError):
    pass


class XErrorEvent(ctypes.Structure):
    # Field order is Xlib's, and it is not the order the documentation lists them in: the resource id comes
    # before the serial. Getting it wrong reads the error code out of the middle of a pointer.
    _fields_ = [
        ("type", ctypes.c_int), ("display", ctypes.c_void_p), ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong), ("error_code", ctypes.c_ubyte), ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


X_ERROR_NAMES = {
    1: "BadRequest", 2: "BadValue", 3: "BadWindow", 4: "BadPixmap", 5: "BadAtom", 6: "BadCursor",
    7: "BadFont", 8: "BadMatch", 9: "BadDrawable", 10: "BadAccess", 11: "BadAlloc", 12: "BadColor",
    13: "BadGC", 14: "BadIDChoice", 15: "BadName", 16: "BadLength", 17: "BadImplementation",
}
_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(XErrorEvent))
_errors: list[str] = []


@_ERROR_HANDLER
def _record_error(display, event) -> int:
    """Xlib's default error handler prints and calls exit(). That is fatal here in the literal sense: the
    game window closing mid-run would kill the agent process instead of raising something the watchdog can
    catch. X errors are asynchronous, so they are recorded and raised at the next sync point."""
    error = event.contents
    name = X_ERROR_NAMES.get(error.error_code, str(error.error_code))
    _errors.append(
        f"{name} on request {error.request_code}.{error.minor_code} for resource 0x{error.resourceid:x}"
    )
    return 0


class XImageFuncs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("create_image", "destroy_image", "get_pixel", "put_pixel", "sub_image", "add_pixel")]


class XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
        ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong),
        ("obdata", ctypes.c_void_p), ("f", XImageFuncs),
    ]


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("border_width", ctypes.c_int), ("depth", ctypes.c_int), ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong), ("class", ctypes.c_int), ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int), ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p),
    ]


class XShmSegmentInfo(ctypes.Structure):
    _fields_ = [
        ("shmseg", ctypes.c_ulong), ("shmid", ctypes.c_int),
        ("shmaddr", ctypes.c_void_p), ("readOnly", ctypes.c_int),
    ]


def _load(name: str, *candidates: str):
    path = ctypes.util.find_library(name)
    for option in ([path] if path else []) + list(candidates):
        try:
            return ctypes.CDLL(option)
        except OSError:
            continue
    raise X11Error(f"lib{name} is not installed; capture needs the X11 client libraries")


def _bind(lib, name, restype, argtypes):
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


class _Xlib:
    """The handful of Xlib entry points this needs, with prototypes declared.

    ctypes defaults every return value to `int`, which truncates the 64-bit Display and XImage pointers to
    garbage -- the failure looks like a segfault far from its cause, so nothing here is called undeclared.
    """

    def __init__(self):
        x11 = _load("X11", "libX11.so.6")
        xext = _load("Xext", "libXext.so.6")
        libc = ctypes.CDLL(None)
        ulong, uint, cint, void, char_p = ctypes.c_ulong, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p
        image_p = ctypes.POINTER(XImage)

        self.open_display = _bind(x11, "XOpenDisplay", void, [char_p])
        self.close_display = _bind(x11, "XCloseDisplay", cint, [void])
        self.default_screen = _bind(x11, "XDefaultScreen", cint, [void])
        self.root_window = _bind(x11, "XRootWindow", ulong, [void, cint])
        self.default_visual = _bind(x11, "XDefaultVisual", void, [void, cint])
        self.default_depth = _bind(x11, "XDefaultDepth", cint, [void, cint])
        self.get_attributes = _bind(x11, "XGetWindowAttributes", cint, [void, ulong, ctypes.POINTER(XWindowAttributes)])
        self.get_image = _bind(x11, "XGetImage", image_p, [void, ulong, cint, cint, uint, uint, ulong, cint])
        self.sync = _bind(x11, "XSync", cint, [void, cint])
        self.set_error_handler = _bind(x11, "XSetErrorHandler", void, [void])
        self.free = _bind(x11, "XFree", cint, [void])
        self.query_tree = _bind(
            x11, "XQueryTree", cint,
            [void, ulong, ctypes.POINTER(ulong), ctypes.POINTER(ulong), ctypes.POINTER(ctypes.POINTER(ulong)),
             ctypes.POINTER(uint)],
        )
        self.fetch_name = _bind(x11, "XFetchName", cint, [void, ulong, ctypes.POINTER(char_p)])
        self.intern_atom = _bind(x11, "XInternAtom", ulong, [void, char_p, cint])
        self.get_property = _bind(
            x11, "XGetWindowProperty", cint,
            [void, ulong, ulong, ctypes.c_long, ctypes.c_long, cint, ulong, ctypes.POINTER(ulong),
             ctypes.POINTER(cint), ctypes.POINTER(ulong), ctypes.POINTER(ulong), ctypes.POINTER(void)],
        )
        self.shm_query = _bind(xext, "XShmQueryExtension", cint, [void])
        self.shm_create_image = _bind(
            xext, "XShmCreateImage", image_p,
            [void, void, uint, cint, void, ctypes.POINTER(XShmSegmentInfo), uint, uint],
        )
        self.shm_attach = _bind(xext, "XShmAttach", cint, [void, ctypes.POINTER(XShmSegmentInfo)])
        self.shm_detach = _bind(xext, "XShmDetach", cint, [void, ctypes.POINTER(XShmSegmentInfo)])
        self.shm_get_image = _bind(xext, "XShmGetImage", cint, [void, ulong, image_p, cint, cint, ulong])
        self.shmget = _bind(libc, "shmget", cint, [cint, ctypes.c_size_t, cint])
        self.shmat = _bind(libc, "shmat", void, [cint, void, cint])
        self.shmdt = _bind(libc, "shmdt", cint, [void])
        self.shmctl = _bind(libc, "shmctl", cint, [cint, cint, void])


_XLIB: _Xlib | None = None


def xlib() -> _Xlib:
    global _XLIB
    if _XLIB is None:
        _XLIB = _Xlib()
        _XLIB.set_error_handler(ctypes.cast(_record_error, ctypes.c_void_p))
    return _XLIB


def check_errors(x: "_Xlib", display, what: str) -> None:
    """Flush the connection and raise anything the server complained about since the last check."""
    x.sync(display, 0)
    if _errors:
        reported, _errors[:] = list(_errors), []
        raise X11Error(f"{what}: {'; '.join(reported)}")


def clear_errors() -> None:
    _errors.clear()


@dataclass(frozen=True)
class WindowInfo:
    id: int
    title: str
    width: int
    height: int


def _window_title(x, display, window: int) -> str:
    """_NET_WM_NAME first (Wine sets it, and it is UTF-8), then the older WM_NAME."""
    net_wm_name = x.intern_atom(display, b"_NET_WM_NAME", 0)
    kind, fmt, count, after, data = (
        ctypes.c_ulong(), ctypes.c_int(), ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p()
    )
    status = x.get_property(
        display, window, net_wm_name, 0, 1024, 0, ANY_PROPERTY_TYPE,
        ctypes.byref(kind), ctypes.byref(fmt), ctypes.byref(count), ctypes.byref(after), ctypes.byref(data),
    )
    if status == 0 and data:
        title = ctypes.string_at(data).decode("utf-8", "replace")
        x.free(data)
        if title:
            return title
    name = ctypes.c_char_p()
    if x.fetch_name(display, window, ctypes.byref(name)) and name.value:
        title = name.value.decode("utf-8", "replace")
        x.free(ctypes.cast(name, ctypes.c_void_p))
        return title
    return ""


def list_windows(display: str | None = None, min_size: int = 200) -> list[WindowInfo]:
    """Every mapped window at least `min_size` on both sides, deepest first.

    Wine nests the game in a frame window or two, so the useful match is usually the largest window whose
    title contains what you searched for -- which is what `find_window` returns.
    """
    x = xlib()
    name = display
    display = x.open_display(_display_name(name))
    if not display:
        raise X11Error(f"cannot open X display {name or os.environ.get('DISPLAY')!r}")
    try:
        found: list[WindowInfo] = []
        stack = [x.root_window(display, x.default_screen(display))]
        while stack:
            window = stack.pop()
            root, parent = ctypes.c_ulong(), ctypes.c_ulong()
            children, count = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
            if x.query_tree(
                display, window, ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children), ctypes.byref(count)
            ):
                stack.extend(children[i] for i in range(count.value))
                if children:
                    x.free(children)
            attributes = XWindowAttributes()
            if not x.get_attributes(display, window, ctypes.byref(attributes)):
                continue
            if attributes.map_state != 2 or attributes.width < min_size or attributes.height < min_size:
                continue  # IsViewable only: unmapped windows have no pixels to read
            found.append(WindowInfo(int(window), _window_title(x, display, window), attributes.width, attributes.height))
        return found
    finally:
        x.close_display(display)


def find_window(title: str, display: str | None = None) -> WindowInfo:
    """The largest mapped window whose title contains `title`, case-insensitively."""
    matches = [w for w in list_windows(display) if title.lower() in w.title.lower()]
    if not matches:
        titles = sorted({w.title for w in list_windows(display) if w.title})
        raise X11Error(f"no window matching {title!r}; visible windows: {titles or '(none titled)'}")
    return max(matches, key=lambda w: w.width * w.height)


def _display_name(display: str | None) -> bytes | None:
    name = display or os.environ.get("DISPLAY")
    return name.encode() if name else None


class X11Grabber:
    """Reads pixels out of one X11 drawable, as fast as the server will hand them over."""

    def __init__(
        self,
        window: int | str | None = None,
        *,
        display: str | None = None,
        region: tuple[int, int, int, int] | None = None,
        shm: bool = True,
    ):
        x = self.x = xlib()
        self.shm_info = None
        self.image = None
        self._buffer = None
        self.display = x.open_display(_display_name(display))
        if not self.display:
            raise X11Error(f"cannot open X display {display or os.environ.get('DISPLAY')!r}")
        self.screen = x.default_screen(self.display)
        if isinstance(window, str):
            window = find_window(window, display).id
        self.window = int(window) if window is not None else int(x.root_window(self.display, self.screen))

        clear_errors()
        attributes = XWindowAttributes()
        ok = x.get_attributes(self.display, self.window, ctypes.byref(attributes))
        check_errors(x, self.display, f"window 0x{self.window:x}")
        if not ok:
            raise X11Error(f"window 0x{self.window:x} does not exist")
        self.depth = attributes.depth
        if self.depth not in (24, 32):
            raise X11Error(f"window 0x{self.window:x} has depth {self.depth}; only 24- and 32-bit are supported")
        left, top, width, height = region or (0, 0, attributes.width, attributes.height)
        self.region = (int(left), int(top), int(width), int(height))

        if shm and x.shm_query(self.display):
            self._attach_shm()
        self.backend = "xshm" if self.image else "xgetimage"

    @property
    def size(self) -> tuple[int, int]:
        return self.region[2], self.region[3]

    def _attach_shm(self) -> None:
        x, (_, _, width, height) = self.x, self.region
        info = XShmSegmentInfo()
        visual = x.default_visual(self.display, self.screen)
        image = x.shm_create_image(self.display, visual, self.depth, ZPIXMAP, None, ctypes.byref(info), width, height)
        if not image:
            return
        size = image.contents.bytes_per_line * image.contents.height
        info.shmid = x.shmget(0, size, IPC_CREAT | 0o600)  # IPC_PRIVATE
        if info.shmid < 0:
            self._destroy_image(image)
            return
        address = x.shmat(info.shmid, None, 0)
        # Marked for deletion immediately: the segment lives until both we and the server detach, so a crash
        # cannot leave shared memory stranded on the machine.
        x.shmctl(info.shmid, IPC_RMID, None)
        if not address or address == ctypes.c_void_p(-1).value:
            self._destroy_image(image)
            return
        info.shmaddr = image.contents.data = address
        info.readOnly = 0
        if not x.shm_attach(self.display, ctypes.byref(info)):
            x.shmdt(address)
            self._destroy_image(image)
            return
        x.sync(self.display, 0)
        self.shm_info, self.image = info, image
        self._buffer = (ctypes.c_uint8 * size).from_address(address)

    def _destroy_image(self, image, shared: bool = False) -> None:
        """XDestroyImage frees `image->data` with free(). For a shared image that pointer came from shmat,
        not malloc, so it is cleared first and the segment is detached separately."""
        if not image:
            return
        if shared:
            image.contents.data = None
        destroy = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(XImage))(image.contents.f.destroy_image)
        destroy(image)

    def grab(self) -> np.ndarray:
        """The region as an (H, W, 3) uint8 RGB array. The returned array owns its memory: the shared buffer
        is overwritten by the next grab, and a view into it would silently change under the caller."""
        x, (left, top, width, height) = self.x, self.region
        clear_errors()
        if self.image is not None:
            ok = x.shm_get_image(self.display, self.window, self.image, left, top, ALL_PLANES)
            # The window can go away between ticks -- the game crashed, or someone closed it. That has to
            # arrive as an exception the env loop can turn into a watchdog incident, not as process death.
            check_errors(x, self.display, f"grabbing window 0x{self.window:x}")
            if not ok:
                raise X11Error("XShmGetImage failed; is the window still mapped?")
            stride = self.image.contents.bytes_per_line
            raw = np.frombuffer(self._buffer, dtype=np.uint8, count=stride * height)
        else:
            image = x.get_image(self.display, self.window, left, top, width, height, ALL_PLANES, ZPIXMAP)
            check_errors(x, self.display, f"grabbing window 0x{self.window:x}")
            if not image:
                raise X11Error("XGetImage failed; is the window still mapped?")
            stride = image.contents.bytes_per_line
            raw = np.ctypeslib.as_array(
                ctypes.cast(image.contents.data, ctypes.POINTER(ctypes.c_uint8)), shape=(stride * height,)
            ).copy()
            self._destroy_image(image)
        # X hands back BGRX on a little-endian TrueColor display, padded to bytes_per_line.
        pixels = raw.reshape(height, stride // 4, 4)[:, :width, 2::-1]
        return np.ascontiguousarray(pixels)

    def describe(self) -> dict:
        return {
            "kind": "x11",
            "backend": self.backend,
            "window": f"0x{self.window:x}",
            "region": list(self.region),
            "depth": self.depth,
        }

    def close(self) -> None:
        x = getattr(self, "x", None)
        if x is None or not getattr(self, "display", None):
            return
        shared = self.shm_info is not None
        if shared:
            x.shm_detach(self.display, ctypes.byref(self.shm_info))
        if self.image is not None:
            self._destroy_image(self.image, shared=shared)
            self.image = None
        if shared:
            x.shmdt(self.shm_info.shmaddr)
            self.shm_info = None
        self._buffer = None
        x.close_display(self.display)
        self.display = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter teardown: the libraries may already be gone
            pass
