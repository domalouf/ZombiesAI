#!/usr/bin/env python3
"""The live overlay for a recording session: what the recorder is doing, and what your hands are doing.

    scripts/record_waw.sh 20 "notes"                 # starts it for you (ZOMBIESAI_OVERLAY=0 turns it off)
    /usr/bin/python3 scripts/record_overlay.py --demo            # try it out with a fake recording
    /usr/bin/python3 scripts/record_overlay.py --snapshot out.png --demo   # draw one frame to a PNG, no screen

Two panels, drawn on the compositor's overlay layer -- above a fullscreen game, never focused, and click-
through, so the game keeps every key and click:

* **top centre: the recorder's state.** Waiting for the game window, the countdown, then REC with the time
  recorded and the time left -- and, when F8 has marked the session as not playing, a wide amber banner that
  stays up until F8 is pressed again (a missed second tap once marked the rest of a session as menus). A
  capture outage (the game's window not visible) shows in red.
* **left: the input display.** The keys around WASD, the mouse buttons and wheel, and a look meter for which
  way and how fast the mouse is turning. A key bound to a control lights amber and says which control; one
  that is not a control (space, ctrl, c) lights grey, because the game sees it and the labels do not. With a
  gamepad bridge running (record_demo.py --pad), the sticks, triggers and buttons are drawn under it.

**It never reaches the training frames.** The recorder grabs the game's own XWayland window
(demos/x11_capture.py), and this is a Wayland surface the X server never sees. A screen recorder (OBS,
gpu-screen-recorder) does see it, which is what you want for a video.

It reads the recorder's status from the live bus (viz/live_bus.py, `record.sock`) and the keys straight off
the event devices, exactly as the recorder does (demos/evdev_input.py), so the display lags your hand by one
frame, not by a decision. It needs GTK 4 and gtk4-layer-shell, which the system Python has and the project's
venv does not -- hence /usr/bin/python3; it re-executes itself with the library preloaded, which
gtk4-layer-shell requires from Python.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zombiesai.viz import input_layout as L  # noqa: E402  (stdlib only)
from zombiesai.viz.live_bus import Subscriber, bus_path  # noqa: E402

LAYER_SHELL = "libgtk4-layer-shell.so"
FONT = "Liberation Sans"
STATUS_W, STATUS_H = 1100, 120  # the status window, at scale 1 (1440p)
STALE_S = 2.0  # a recorder silent this long is shown as gone
PAD_STALE_S = 1.0
DONE_LINGER_S = 5.0


def rgb(hex_colour: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (int(hex_colour[0:2], 16) / 255, int(hex_colour[2:4], 16) / 255, int(hex_colour[4:6], 16) / 255, alpha)


def clock_text(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def status_lines(message: dict | None, now: float) -> tuple[str, list[tuple[str, str]], str | None]:
    """The status panel's content for the recorder's latest message: (tone, [(text, colour)...], sub-line).
    `tone` is the panel's style: "rec", "idle" (not playing), "warn", "wait" or "gone"."""
    if message is None:
        return "gone", [("RECORDER NOT RUNNING", L.CHALK_DIM)], None
    state = message.get("state")
    key = str(message.get("mark_key") or "F8").upper()
    if state == "waiting":
        return "wait", [("● ", L.AMBER), ("WAITING FOR THE GAME WINDOW", L.CHALK)], \
            "switch to the game: recording starts after a countdown"
    if state == "countdown":
        left = max(0.0, float(message.get("until", now)) - time.time())
        return "wait", [("RECORDING IN ", L.CHALK), (f"{math.ceil(left)}", L.AMBER)], "hands on the controls"
    if state == "done":
        return "wait", [("■ SAVED ", L.GOOD), (str(message.get("name", "")), L.CHALK)], \
            message.get("note") or "checking the recording -- see the terminal"
    if state != "recording":
        return "gone", [(str(state).upper(), L.CHALK_DIM)], None
    elapsed, limit = message.get("elapsed_s"), message.get("limit_s")
    timer = clock_text(elapsed) + (f" / {clock_text(limit)}" if limit else "")
    if message.get("stale"):
        return "warn", [("● REC ", L.SANGUINE), (timer, L.CHALK), ("   CAPTURE LOST", L.SANGUINE)], \
            "the game window is not on screen: these steps are flagged and not trained on"
    if not message.get("playing", True):
        return "idle", [("NOT PLAYING", L.INK), (f"   press {key} when you're back in the game", L.INK)], \
            f"REC {timer} -- menus, pauses, deaths and loading are left out of training"
    blink = int(now * 2) % 2 == 0
    return "rec", [("● " if blink else "○ ", L.SANGUINE), ("REC  ", L.CHALK), (timer, L.CHALK),
                   ("   PLAYING", L.GOOD)], f"{key}: going into a menu, pause, loading or game over"


def rounded(cr, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def text(cr, x, y, size, colour, value, bold=True, align="left", alpha=1.0):
    """`value` with its baseline-centre at y: left/centre/right aligned on x. Returns its advance."""
    import cairo

    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(size)
    ext = cr.text_extents(value)
    dx = {"left": 0, "center": -ext.x_advance / 2, "right": -ext.x_advance}[align]
    cr.set_source_rgba(*rgb(colour, alpha))
    cr.move_to(x + dx, y + size * 0.35)
    cr.show_text(value)
    return ext.x_advance


def draw_status(cr, width, height, message, now, scale=1.0):
    tone, parts, sub = status_lines(message, now)
    size = 34 * scale
    import cairo

    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(size)
    line_w = sum(cr.text_extents(p).x_advance for p, _ in parts)
    pad = 26 * scale
    w = min(width, line_w + 2 * pad)
    if sub:
        cr.set_font_size(17 * scale)
        w = min(width, max(w, cr.text_extents(sub).x_advance + 2 * pad))
    h = (92 if sub else 62) * scale
    x = (width - w) / 2
    back = {"idle": (L.AMBER, 0.94), "warn": (L.INK, 0.78), "rec": (L.INK, 0.62), "wait": (L.INK, 0.7),
            "gone": (L.INK, 0.45)}[tone]
    rounded(cr, x, 4 * scale, w, h, 12 * scale)
    cr.set_source_rgba(*rgb(*back))
    cr.fill()
    if tone in ("warn", "rec", "idle"):  # the accent stripe, as on the stream overlay
        cr.rectangle(x, 4 * scale + 10 * scale, 4 * scale, h - 20 * scale)
        cr.set_source_rgba(*rgb(L.SANGUINE if tone != "idle" else L.INK, 0.9))
        cr.fill()
    cx = (width - line_w) / 2
    for part, colour in parts:
        cx += text(cr, cx, 4 * scale + 32 * scale, size, colour, part)
    if sub:
        text(cr, width / 2, 4 * scale + 70 * scale, 17 * scale, L.INK if tone == "idle" else L.CHALK_DIM, sub,
             bold=False, align="center", alpha=0.85)


def draw_cap(cr, cap, u, ox, oy, lit, control, fire=False):
    x, y, w, h = ox + cap.x * u, oy + cap.y * u, cap.w * u, cap.h * u
    rounded(cr, x, y, w, h, 0.16 * u)
    if lit:
        cr.set_source_rgba(*rgb(L.SANGUINE if fire else (L.AMBER if control else L.CHALK_DIM), 0.95))
    else:
        cr.set_source_rgba(*rgb(L.INK, 0.55))
    cr.fill_preserve()
    cr.set_line_width(max(1.0, 0.04 * u))
    cr.set_source_rgba(*rgb(L.CHALK if control else L.CHALK_DIM, 0.55 if control else 0.25))
    cr.stroke()
    ink = L.INK if lit else (L.CHALK if control else L.CHALK_DIM)
    alpha = 1.0 if lit or control else 0.55
    label_size = 0.42 * u if len(cap.label) <= 2 else 0.3 * u
    if control and cap.w >= 0.8:  # the wheel and the middle button are too narrow for a name
        text(cr, x + w / 2, y + h * 0.4, label_size, ink, cap.label, align="center", alpha=alpha)
        text(cr, x + w / 2, y + h * 0.78, 0.22 * u, ink, control, bold=False, align="center", alpha=alpha)
    else:
        text(cr, x + w / 2, y + h / 2, label_size, ink, cap.label, align="center", alpha=alpha)


def draw_look(cr, u, ox, oy, look):
    cx, cy, r = ox + L.LOOK_CX * u, oy + L.LOOK_CY * u, L.LOOK_R * u
    cr.new_path()
    cr.arc(cx, cy, r, 0, 2 * math.pi)
    cr.set_source_rgba(*rgb(L.INK, 0.55))
    cr.fill_preserve()
    cr.set_line_width(max(1.0, 0.04 * u))
    cr.set_source_rgba(*rgb(L.CHALK_DIM, 0.4))
    cr.stroke()
    for dx, dy in ((1, 0), (0, 1)):
        cr.move_to(cx - dx * r * 0.85, cy - dy * r * 0.85)
        cr.line_to(cx + dx * r * 0.85, cy + dy * r * 0.85)
    cr.set_source_rgba(*rgb(L.CHALK_DIM, 0.2))
    cr.stroke()
    x, y, speed = look
    if speed > 0:
        cr.set_line_width(0.1 * u)
        cr.set_source_rgba(*rgb(L.AMBER, 0.9))
        cr.move_to(cx, cy)
        cr.line_to(cx + x * r, cy + y * r)
        cr.stroke()
        cr.new_path()
        cr.arc(cx + x * r, cy + y * r, 0.14 * u, 0, 2 * math.pi)
        cr.fill()
    text(cr, cx, cy + r + 0.32 * u, 0.26 * u, L.CHALK_DIM, f"look {speed:.0f}°/s" if speed else "look",
         bold=False, align="center")


def draw_pad(cr, u, ox, oy, pad):
    """The gamepad under the keys: sticks, triggers and the buttons held (the bridge's `pad` message)."""
    held = set(pad.get("buttons") or [])
    sticks, triggers = pad.get("sticks") or {}, pad.get("triggers") or {}
    rounded(cr, ox - 0.6 * u, oy - 0.3 * u, L.PANEL_W * u + 1.2 * u, 3.0 * u, 0.35 * u)
    cr.set_source_rgba(*rgb(L.INK, 0.5))
    cr.fill()
    text(cr, ox, oy + 0.05 * u, 0.26 * u, L.CHALK_DIM, (pad.get("device") or "gamepad")[:40], bold=False)

    def stick(cx, cy, xy, pressed):
        r = 0.8 * u
        cr.new_path()
        cr.arc(cx, cy, r, 0, 2 * math.pi)
        cr.set_source_rgba(*rgb(L.INK, 0.6))
        cr.fill_preserve()
        cr.set_line_width(max(1.0, 0.04 * u))
        cr.set_source_rgba(*rgb(L.CHALK_DIM, 0.4))
        cr.stroke()
        x, y = (xy or [0, 0])[:2]
        cr.new_path()
        cr.arc(cx + x * r * 0.75, cy - y * r * 0.75, 0.25 * u, 0, 2 * math.pi)
        cr.set_source_rgba(*rgb(L.AMBER if pressed else L.CHALK, 0.95))
        cr.fill()

    def trigger(x, y, value, label):
        h = 1.6 * u
        cr.rectangle(x, y, 0.35 * u, h)
        cr.set_source_rgba(*rgb(L.INK, 0.6))
        cr.fill()
        v = max(0.0, min(1.0, float(value or 0)))
        cr.rectangle(x, y + h * (1 - v), 0.35 * u, h * v)
        cr.set_source_rgba(*rgb(L.SANGUINE if label == "RT" else L.AMBER, 0.95))
        cr.fill()
        text(cr, x + 0.175 * u, y + h + 0.25 * u, 0.22 * u, L.CHALK_DIM, label, align="center")

    def button(cx, cy, name, label, r=0.28):
        cr.new_path()
        cr.arc(cx, cy, r * u, 0, 2 * math.pi)
        cr.set_source_rgba(*rgb(L.AMBER if name in held else L.INK, 0.95 if name in held else 0.6))
        cr.fill()
        text(cr, cx, cy, 0.22 * u, L.INK if name in held else L.CHALK_DIM, label, align="center")

    top = oy + 0.5 * u
    trigger(ox, top, triggers.get("left"), "LT")
    stick(ox + 1.5 * u, top + 0.9 * u, sticks.get("left"), "l3" in held)
    for dx, dy, name in ((0, -1, "dpad_up"), (0, 1, "dpad_down"), (-1, 0, "dpad_left"), (1, 0, "dpad_right")):
        button(ox + 3.3 * u + dx * 0.45 * u, top + 0.9 * u + dy * 0.45 * u, name, "", r=0.2)
    button(ox + 4.5 * u, top + 0.3 * u, "lb", "LB")
    button(ox + 5.2 * u, top + 0.3 * u, "select", "≡", r=0.2)
    button(ox + 5.8 * u, top + 0.3 * u, "start", "☰", r=0.2)
    button(ox + 6.5 * u, top + 0.3 * u, "rb", "RB")
    stick(ox + 5.5 * u, top + 1.45 * u, sticks.get("right"), "r3" in held)
    for dx, dy, name, label in ((0, -1, "north", "Y"), (1, 0, "east", "B"), (0, 1, "south", "A"), (-1, 0, "west", "X")):
        button(ox + 7.6 * u + dx * 0.5 * u, top + 0.9 * u + dy * 0.5 * u, name, label)
    trigger(ox + 8.6 * u, top, triggers.get("right"), "RT")


def keys_size(u: float) -> tuple[int, int]:
    """The input window: the keys panel, then room for the pad under it."""
    return int(L.PANEL_W * u + 1.2 * u), int((L.PANEL_H + 0.75 + 1.2) * u + 0.9 * u + 3.0 * u + 4)


def draw_keys(cr, u, lit, controls, look, pad=None):
    pad_x = 0.6 * u
    oy = 0.6 * u + 0.75 * u  # room above the mouse for the wheel's up notch
    rounded(cr, 0, 0, L.PANEL_W * u + 2 * pad_x, (L.PANEL_H + 0.75) * u + 1.2 * u, 0.35 * u)
    cr.set_source_rgba(*rgb(L.INK, 0.42))
    cr.fill()
    for cap in L.ALL_CAPS:
        control = controls.get(cap.code)
        draw_cap(cr, cap, u, pad_x, oy, cap.code in lit, control, fire=control == "fire")
    draw_look(cr, u, pad_x, oy, look)
    if pad is not None:
        draw_pad(cr, u, pad_x, oy + (L.PANEL_H + 0.75) * u + 0.9 * u, pad)


# ------------------------------------------------------------------------------------------------ the sources


class Sources:
    """The recorder's status and the pad's state off the live bus, and the keys off the event devices."""

    def __init__(self, counts_per_degree: float, read_devices: bool = True):
        self.counts_per_degree = counts_per_degree
        self.bus = Subscriber(bus_path("record"))
        self.state = L.InputState()
        self.inputs = None
        if read_devices:
            from zombiesai.demos.evdev_input import EvdevInput

            try:
                self.inputs = EvdevInput()
            except (RuntimeError, PermissionError) as error:
                print(f"record_overlay: no key display ({error})", file=sys.stderr)

    def poll(self) -> bool:
        changed = self.bus.poll()
        if self.inputs is not None:
            changed |= self.state.feed(self.inputs.drain())
        return changed

    def status(self) -> dict | None:
        message = self.bus.get("recorder", STALE_S)
        if message is None:
            done = self.bus.get("recorder")
            if done is not None and done.get("state") == "done":
                return done
        return message

    def pad(self) -> dict | None:
        return self.bus.get("pad", PAD_STALE_S)

    def lit(self) -> set[str]:
        lit = self.state.lit()
        pad = self.pad()
        if pad:
            lit |= {str(code).lower() for code in pad.get("emitting") or []}
        return lit

    def look(self):
        return self.state.look(self.counts_per_degree)

    def close(self):
        self.bus.close()
        if self.inputs is not None:
            self.inputs.close()


class Demo:
    """A fake recording for --demo: a status that walks through every state, and keys being pressed."""

    def __init__(self, pad: bool = True):
        self.t0 = time.monotonic()
        self.with_pad = pad
        self.counts_per_degree = 9.09

    def _t(self):
        return time.monotonic() - self.t0

    def poll(self):
        return True

    def status(self):
        t = self._t() % 24
        if t < 3:
            return {"state": "waiting", "mark_key": "f8"}
        if t < 6:
            return {"state": "countdown", "until": time.time() + 6 - t, "mark_key": "f8"}
        state = {"state": "recording", "elapsed_s": t - 6, "limit_s": 1200, "mark_key": "f8", "playing": True}
        if 14 < t < 18:
            state["playing"] = False
        if 20 < t < 22:
            state["stale"] = True
        return state

    def lit(self):
        t = self._t()
        seq = ["w", "w", "a", "shift", "mouse1", "mouse2", "r", "space", "f", "1", "wheelup"]
        return {"w"} | {seq[int(t * 2) % len(seq)]}

    def look(self):
        t = self._t()
        return L.look_vector(math.cos(t) * 300, math.sin(t * 0.7) * 60, 0.1, self.counts_per_degree)

    def pad(self):
        if not self.with_pad:
            return None
        t = self._t()
        return {"kind": "pad", "device": "Demo Controller", "sticks": {"left": [0.0, 0.9], "right": [math.cos(t), 0.3 * math.sin(t)]},
                "triggers": {"left": 0.6, "right": (math.sin(t * 3) + 1) / 2}, "buttons": ["rb"] if int(t) % 2 else ["west"],
                "emitting": ["w"]}

    def close(self):
        pass


# ------------------------------------------------------------------------------------------------ the windows


def snapshot(path: Path, sources, u: float, scale: float) -> None:
    """Both panels side by side on a 1440p-ish grey, as a PNG: for checking the look without a compositor."""
    import cairo

    kw, kh = keys_size(u)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(STATUS_W, kw) + 40, int(STATUS_H * scale) + kh + 60)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0.32, 0.3, 0.27)
    cr.paint()
    cr.save()
    cr.translate(20, 20)
    draw_status(cr, STATUS_W * scale, STATUS_H * scale, sources.status(), time.monotonic(), scale)
    cr.restore()
    cr.translate(20, STATUS_H * scale + 40)
    draw_keys(cr, u, sources.lit(), CONTROLS, sources.look(), sources.pad())
    surface.write_to_png(str(path))


CONTROLS: dict[str, str] = {}


def run_gtk(sources, args) -> None:
    import cairo
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gdk, Gio, GLib, Gtk
    from gi.repository import Gtk4LayerShell as Layer

    if not Layer.is_supported():
        raise SystemExit("record_overlay: this compositor has no layer-shell (Hyprland does)")
    u, scale = args.key_size, args.scale
    app = Gtk.Application(application_id="ai.zombies.RecordOverlay", flags=Gio.ApplicationFlags.NON_UNIQUE)
    done_at: list[float] = []

    def layer_window(width, height, anchors, margins, namespace, draw):
        win = Gtk.Window(application=app)
        win.set_decorated(False)
        win.set_default_size(width, height)
        Layer.init_for_window(win)
        Layer.set_namespace(win, namespace)
        Layer.set_layer(win, Layer.Layer.OVERLAY)
        Layer.set_keyboard_mode(win, Layer.KeyboardMode.NONE)
        Layer.set_exclusive_zone(win, -1)
        for edge, margin in zip(anchors, margins):
            Layer.set_anchor(win, edge, True)
            Layer.set_margin(win, edge, margin)
        area = Gtk.DrawingArea()
        area.set_content_width(width)
        area.set_content_height(height)
        area.set_draw_func(lambda _a, cr, w, h: draw(cr, w, h))
        win.set_child(area)
        win.set_can_target(False)

        def click_through(*_):
            surface = win.get_surface()
            if surface is not None:
                surface.set_input_region(cairo.Region())  # empty: every click goes to the game

        win.connect("map", click_through)
        win.present()
        return win, area

    def activate(_app):
        css = Gtk.CssProvider()
        css.load_from_string("window, window.background { background: transparent; }")
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        areas = []
        if args.status:
            _, status_area = layer_window(int(STATUS_W * scale), int(STATUS_H * scale), [Layer.Edge.TOP],
                                          [int(18 * scale)], "zombiesai-record-status",
                                          lambda cr, w, h: draw_status(cr, w, h, sources.status(),
                                                                       time.monotonic(), scale))
            areas.append(("status", status_area))
        if args.keys:
            kw, kh = keys_size(u)
            _, keys_area = layer_window(kw, kh, [Layer.Edge.LEFT, Layer.Edge.TOP],
                                        [int(args.margin), int(args.top * 1440 * scale)], "zombiesai-record-keys",
                                        lambda cr, w, h: draw_keys(cr, u, sources.lit(), CONTROLS, sources.look(),
                                                                   sources.pad()))
            areas.append(("keys", keys_area))
        last = {"status": None, "keys": None}

        def tick():
            if args.parent_pid and not alive(args.parent_pid):
                app.quit()
                return False
            sources.poll()
            status = sources.status()
            if status is not None and status.get("state") == "done":
                if not done_at:
                    done_at.append(time.monotonic())
                elif time.monotonic() - done_at[0] > DONE_LINGER_S:
                    app.quit()
                    return False
            now = time.monotonic()
            for name, area in areas:
                if name == "status":
                    key = (json.dumps(status_lines(status, now)), int(now * 2))
                else:
                    look = sources.look()
                    key = (tuple(sorted(sources.lit())), tuple(round(v, 2) for v in look),
                           json.dumps(sources.pad(), sort_keys=True))
                if key != last[name]:
                    last[name] = key
                    area.queue_draw()
            return True

        GLib.timeout_add(16, tick)

    app.connect("activate", activate)
    try:
        app.run([])
    finally:
        sources.close()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def preload_layer_shell() -> None:
    """gtk4-layer-shell must be linked before libwayland-client, which from Python means LD_PRELOAD: re-exec
    once with it set."""
    if LAYER_SHELL in os.environ.get("LD_PRELOAD", ""):
        return
    os.environ["LD_PRELOAD"] = " ".join(filter(None, [LAYER_SHELL, os.environ.get("LD_PRELOAD")]))
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bindings", type=Path, default=ROOT / "configs" / "waw_bindings.json",
                        help="key -> control, to name each cap's control (the recording's own bindings)")
    parser.add_argument("--counts-per-degree", type=float, default=9.09, help="for the look meter's deg/s")
    parser.add_argument("--keys", action=argparse.BooleanOptionalAction, default=True, help="the input display")
    parser.add_argument("--status", action=argparse.BooleanOptionalAction, default=True, help="the REC panel")
    parser.add_argument("--key-size", type=float, default=40.0, help="a key cap's size in pixels (1440p: 40)")
    parser.add_argument("--scale", type=float, default=1.0, help="the status panel's scale")
    parser.add_argument("--margin", type=float, default=24.0, help="the input display's distance from the left")
    parser.add_argument("--top", type=float, default=0.30,
                        help="the input display's top, as a share of a 1440-row screen (WaW's HUD is clear "
                             "between ~0.25 and ~0.75 on the left)")
    parser.add_argument("--parent-pid", type=int, default=0, help="quit when this process exits")
    parser.add_argument("--demo", action="store_true", help="a fake recording, to see what it looks like")
    parser.add_argument("--snapshot", type=Path, help="draw one frame to this PNG and exit (no compositor needed)")
    parser.add_argument("--no-pad", action="store_true", help="with --demo: leave the gamepad out")
    parser.add_argument("--demo-at", type=float, default=7.5,
                        help="with --demo --snapshot: this many seconds into the fake session (24 s cycle)")
    args = parser.parse_args()
    if args.bindings and Path(args.bindings).exists():
        CONTROLS.update(L.control_of(json.loads(Path(args.bindings).read_text())))
    if args.snapshot:
        sources = Demo(pad=not args.no_pad) if args.demo else Sources(args.counts_per_degree)
        if args.demo:
            sources.t0 -= args.demo_at  # into the recording, where there is something to see
        sources.poll()
        snapshot(args.snapshot, sources, args.key_size, args.scale)
        sources.close()
        print(args.snapshot)
        return
    preload_layer_shell()
    sources = Demo(pad=not args.no_pad) if args.demo else Sources(args.counts_per_degree)
    run_gtk(sources, args)


if __name__ == "__main__":
    main()
