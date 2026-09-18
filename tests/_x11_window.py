"""Helper process for the capture tests: paints a known window on an X display and holds it open.

    python tests/_x11_window.py <title> <width> <height>

X11 windows belong to the connection that created them, so this has to outlive the call that makes it --
hence a subprocess rather than a fixture that draws and returns.
"""

import ctypes
import os
import sys
import time

QUADRANT_COLOURS = (0xFF0000, 0x00FF00, 0x0000FF, 0x808080)  # TL, TR, BL, BR


def main() -> None:
    title, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    x11 = ctypes.CDLL("libX11.so.6")
    ulong, cint, uint, void, char_p = (
        ctypes.c_ulong, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_char_p
    )
    for name, restype, argtypes in (
        ("XOpenDisplay", void, [char_p]), ("XDefaultScreen", cint, [void]), ("XRootWindow", ulong, [void, cint]),
        ("XDefaultDepth", cint, [void, cint]),
        ("XCreateSimpleWindow", ulong, [void, ulong, cint, cint, uint, uint, uint, ulong, ulong]),
        ("XMapRaised", cint, [void, ulong]), ("XStoreName", cint, [void, ulong, char_p]),
        ("XCreateGC", void, [void, ulong, ulong, void]), ("XSetForeground", cint, [void, void, ulong]),
        ("XFillRectangle", cint, [void, ulong, void, cint, cint, uint, uint]), ("XSync", cint, [void, cint]),
        ("XCreatePixmap", ulong, [void, ulong, uint, uint, uint]),
        ("XSetWindowBackgroundPixmap", cint, [void, ulong, ulong]), ("XClearWindow", cint, [void, ulong]),
    ):
        function = getattr(x11, name)
        function.restype, function.argtypes = restype, argtypes

    display = x11.XOpenDisplay(os.environ["DISPLAY"].encode())
    if not display:
        raise SystemExit("cannot open display")
    screen = x11.XDefaultScreen(display)
    root = x11.XRootWindow(display, screen)
    window = x11.XCreateSimpleWindow(display, root, 40, 30, width, height, 0, 0, 0)
    x11.XStoreName(display, window, title.encode())

    # Painted into a pixmap used as the window's background rather than drawn once into the window itself:
    # X does not preserve the contents of a window that gets covered, so a test that raises another window
    # over this one would otherwise leave it blank -- and the capture under test would be right to see black.
    pixmap = x11.XCreatePixmap(display, root, width, height, x11.XDefaultDepth(display, screen))
    gc = x11.XCreateGC(display, pixmap, 0, None)
    half_w, half_h = width // 2, height // 2
    for (offset_x, offset_y), colour in zip(((0, 0), (half_w, 0), (0, half_h), (half_w, half_h)), QUADRANT_COLOURS):
        x11.XSetForeground(display, gc, colour)
        x11.XFillRectangle(display, pixmap, gc, offset_x, offset_y, half_w, half_h)
    x11.XSetWindowBackgroundPixmap(display, window, pixmap)
    x11.XMapRaised(display, window)
    x11.XClearWindow(display, window)
    x11.XSync(display, 0)
    print(window, flush=True)
    while True:  # the parent kills this once the test is done
        time.sleep(3600)


if __name__ == "__main__":
    main()
