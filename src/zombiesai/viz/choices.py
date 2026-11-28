"""What the policy could have done, and how much it wanted each: its whole action distribution, drawn in ASS.

The policy's action is eight independent choices (spec.ACTION_HEADS), and what it pressed is one sample from
each. The sample alone hides the interesting part -- a turn chosen at 20% against a spread of others looks
exactly like one chosen at 95% -- so this draws every head's probabilities, with the option it actually took
picked out in each:

* **TURN** -- the yaw bins as a row of bars, left turns to the left, "no turn" in the middle.
* **PITCH** -- the pitch bins as a column, looking up at the top.
* **MOVE** -- strafe x forward as a 3x3 grid, each cell shaded by its probability. The heads are independent,
  so a cell's probability is the product of the two marginals.
* **FIRE / AIM / SPRINT** -- the probability of holding each, as a bar.
* **BUTTON** -- the one-press head's six options (nothing, use, reload, melee, grenade, swap).

Bars are drawn to absolute probability (a full bar is 1), not scaled to the largest: a flat, unsure head
should look flat. Each colour's shapes are drawn as one ASS drawing, so a frame of the panel is a dozen
events, not a hundred -- a film's overlay holds one panel per frame.

One renderer for both places the panel appears: the kept films' overlay (viz/overlay.py, from the sidecar's
`p`) and the live fleet viewer (realgame/viewer.py, from the actor's brain feed, viz/live_bus.py).
"""

from zombiesai import spec
from zombiesai.realgame.viewer import AMBER, CHALK, CHALK_DIM, SANGUINE, _text

INK = "&H101010&"
PANEL_ALPHA = "50"  # a shade more opaque than the other panels: its text is smaller
TRACK_ALPHA = "70"  # the empty part of each bar
PANEL_W, PANEL_H = 360.0, 230.0  # in units of u (a 720-row frame's pixels)
BUTTON_NAMES = {"none": "nothing", "use": "use", "reload": "reload", "melee": "melee", "grenade": "grenade",
                "swap": "swap"}


def valid_probs(probs) -> bool:
    """Whether `probs` is a distribution this can draw: one list per head, each that head's width."""
    return (isinstance(probs, list) and len(probs) == len(spec.ACTION_NVEC)
            and all(isinstance(p, list) and len(p) == n for p, n in zip(probs, spec.ACTION_NVEC)))


def _shapes(rects: list[tuple[float, float, float, float]], colour: str, alpha: str = "00") -> str | None:
    """Several rectangles (x, y, w, h, in frame pixels) as one ASS drawing in one colour. The drawing is placed
    at the frame's origin, so its coordinates are the frame's own."""
    rects = [r for r in rects if r[2] > 0 and r[3] > 0]
    if not rects:
        return None
    path = " ".join(f"m {x:.1f} {y:.1f} l {x + w:.1f} {y:.1f} {x + w:.1f} {y + h:.1f} {x:.1f} {y + h:.1f}"
                    for x, y, w, h in rects)
    return f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{colour}\\1a&H{alpha}&}}{path}{{\\p0}}"


def _frames(boxes: list[tuple[float, float, float, float]], colour: str, width: float) -> str | None:
    """Outlines of `boxes` as one drawing: each as four thin rectangles, so no border style is needed."""
    rects = []
    for x, y, w, h in boxes:
        rects += [(x - width, y - width, w + 2 * width, width), (x - width, y + h, w + 2 * width, width),
                  (x - width, y, width, h), (x + w, y, width, h)]
    return _shapes(rects, colour)


def _pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def _deg(value: float) -> str:
    return f"{abs(value):g}°"


class _Painter:
    """Collects rectangles by role, then emits one drawing per role."""

    def __init__(self):
        self.track: list = []
        self.dim: list = []
        self.chosen: list = []
        self.fire: list = []
        self.cells: dict[str, list] = {}  # alpha -> cells of the move grid at that shade
        self.picked: list = []  # outlines round the chosen option
        self.text: list[str] = []

    def bar(self, x, y, w, h, p, chosen, colour_key="chosen", vertical=False):
        """A bar of `p` in a track of (x, y, w, h): filling upward when `vertical`, else rightward."""
        p = min(1.0, max(0.0, float(p)))
        self.track.append((x, y, w, h))
        if vertical:
            fill = (x, y + h * (1 - p), w, h * p)
        else:
            fill = (x, y, w * p, h)
        if chosen:
            (self.fire if colour_key == "fire" else self.chosen).append(fill)
        else:
            self.dim.append(fill)

    def events(self) -> list[str]:
        out = [s for s in (_shapes(self.track, INK, TRACK_ALPHA),) if s]
        for alpha, cells in sorted(self.cells.items(), reverse=True):
            s = _shapes(cells, AMBER, alpha)
            if s:
                out.append(s)
        for rects, colour in ((self.dim, CHALK_DIM), (self.chosen, AMBER), (self.fire, SANGUINE)):
            s = _shapes(rects, colour)
            if s:
                out.append(s)
        s = _frames(self.picked, CHALK, 1.5)
        if s:
            out.append(s)
        return out + self.text


def choices_panel(probs, action, x: float, y: float, u: float, look_bins: dict | None = None,
                  title: str = "CHOICES") -> list[str]:
    """The panel for one step, its top left at (x, y), `u` frame pixels to the layout's unit (height / 720).
    `probs` is a sidecar line's `p` (a list per head), `action` the option taken in each head; `look_bins` the
    film's (its sidecar header's), else today's. Anything malformed draws nothing."""
    if not valid_probs(probs):
        return []
    chose = (list(action) if isinstance(action, (list, tuple)) and len(action) == len(spec.ACTION_HEADS)
             else [max(range(len(p)), key=p.__getitem__) for p in probs])
    bins = look_bins or {}
    yaw_bins = list(bins.get("yaw") or spec.YAW_BINS_DEG)
    pitch_bins = list(bins.get("pitch") or spec.PITCH_BINS_DEG)
    if len(yaw_bins) != len(probs[spec.YAW]):
        yaw_bins = list(spec.YAW_BINS_DEG)
    if len(pitch_bins) != len(probs[spec.PITCH]):
        pitch_bins = list(spec.PITCH_BINS_DEG)

    pt = _Painter()
    dim, label, small = 14 * u, 13 * u, 12 * u

    def X(v):
        return x + v * u

    def Y(v):
        return y + v * u

    def text(vx, vy, align, size, colour, words, bold=False):
        pt.text.append(_text(X(vx), Y(vy), align, size, colour, words, bold))

    panel = [f"{{\\an7\\pos({x:.0f},{y:.0f})\\p1\\bord0\\shad0\\1c{INK}\\1a&H{PANEL_ALPHA}&}}"
             f"m 0 0 l {PANEL_W * u:.0f} 0 {PANEL_W * u:.0f} {PANEL_H * u:.0f} 0 {PANEL_H * u:.0f}{{\\p0}}"]
    text(12, 10, 7, 15 * u, CHALK, title, bold=True)

    # TURN: a row of bars over the yaw bins.
    n = len(yaw_bins)
    bx0, bw, gap, top, height = 12.0, 16.0, 4.0, 46.0, 48.0
    text(12, 30, 7, dim, CHALK_DIM, "TURN")
    for i, p in enumerate(probs[spec.YAW]):
        bx = bx0 + i * (bw + gap)
        pt.bar(X(bx), Y(top), bw * u, height * u, p, i == chose[spec.YAW], vertical=True)
        if i == chose[spec.YAW]:
            pt.picked.append((X(bx), Y(top), bw * u, height * u))
    row_w = n * (bw + gap) - gap
    zero = yaw_bins.index(0.0) if 0.0 in yaw_bins else n // 2
    text(bx0, top + height + 4, 7, small, CHALK_DIM, "←" + _deg(yaw_bins[0]))
    text(bx0 + zero * (bw + gap) + bw / 2, top + height + 4, 8, small, CHALK_DIM, "0")
    text(bx0 + row_w, top + height + 4, 9, small, CHALK_DIM, _deg(yaw_bins[-1]) + "→")
    turn = yaw_bins[chose[spec.YAW]]
    text(bx0 + row_w, 30, 9, dim, CHALK, ("· steady" if not turn else
                                          ("→ " if turn > 0 else "← ") + _deg(turn))
         + f"  {_pct(probs[spec.YAW][chose[spec.YAW]])}")

    # PITCH: a column of bars, looking up at the top.
    m = len(pitch_bins)
    px0, pw, ptop, rh, rgap = 296.0, 52.0, 46.0, 4.6, 1.0
    text(px0 + pw, 30, 9, dim, CHALK_DIM, "PITCH")
    order = sorted(range(m), key=lambda j: -pitch_bins[j])  # +6 first
    for row, j in enumerate(order):
        ry = ptop + row * (rh + rgap)
        pt.bar(X(px0), Y(ry), pw * u, rh * u, probs[spec.PITCH][j], j == chose[spec.PITCH])
        if j == chose[spec.PITCH]:
            pt.picked.append((X(px0), Y(ry), pw * u, rh * u))
    column_h = m * (rh + rgap) - rgap
    text(px0 - 4, ptop, 9, small, CHALK_DIM, "↑" + _deg(max(pitch_bins)))
    text(px0 - 4, ptop + column_h, 3, small, CHALK_DIM, "↓" + _deg(min(pitch_bins)))

    # MOVE: strafe x forward, each cell shaded by the joint probability.
    gy = 134.0
    text(12, gy - 18, 7, dim, CHALK_DIM, "MOVE")
    cell, cgap = 26.0, 3.0
    strafe, forward = probs[spec.STRAFE], probs[spec.FORWARD]
    arrows = {(-1, 1): "↖", (0, 1): "W", (1, 1): "↗", (-1, 0): "A", (0, 0): "·", (1, 0): "D",
              (-1, -1): "↙", (0, -1): "S", (1, -1): "↘"}
    for r, fv in enumerate((1, 0, -1)):  # forward on top
        for c, sv in enumerate((-1, 0, 1)):
            si, fi = spec.STRAFE_VALUES.index(sv), spec.FORWARD_VALUES.index(fv)
            p = float(strafe[si]) * float(forward[fi])
            cx, cy = X(12 + c * (cell + cgap)), Y(gy + r * (cell + cgap))
            pt.track.append((cx, cy, cell * u, cell * u))
            # Alpha in sixteen steps, so cells of nearly the same shade share one drawing.
            level = round(min(1.0, max(0.0, p)) * 15)
            if level:
                pt.cells.setdefault(f"{(15 - level) * 17:02X}", []).append((cx, cy, cell * u, cell * u))
            picked = si == chose[spec.STRAFE] and fi == chose[spec.FORWARD]
            if picked:
                pt.picked.append((cx, cy, cell * u, cell * u))
            ink = INK if p > 0.55 else (CHALK if picked else CHALK_DIM)
            pt.text.append(_text(cx + cell * u / 2, cy + cell * u / 2, 5, (17 if (sv and fv) else 14) * u, ink,
                                 arrows[sv, fv], bold=picked))
    grid = 3 * cell + 2 * cgap

    # FIRE / AIM / SPRINT: the probability of holding each.
    hx, hw = 12 + grid + 16, 74.0
    for row, (name, head) in enumerate((("FIRE", spec.FIRE), ("AIM", spec.ADS), ("SPRINT", spec.SPRINT))):
        ry = gy + 4 + row * 28
        p_on = float(probs[head][1])
        on = chose[head] == 1
        text(hx, ry - 2, 7, label, CHALK if on else CHALK_DIM, name, bold=on)
        pt.bar(X(hx), Y(ry + 14), hw * u, 6 * u, p_on, on, colour_key="fire" if head == spec.FIRE else "chosen")
        text(hx + hw, ry - 2, 9, label, CHALK if on else CHALK_DIM, _pct(p_on))

    # BUTTON: the one-press head.
    bx = hx + hw + 18
    bwid = PANEL_W - 12 - bx - 44 - 32  # name, bar, percentage
    text(bx, gy - 18, 7, dim, CHALK_DIM, "BUTTON")
    for row, name in enumerate(spec.BUTTONS):
        ry = gy - 2 + row * 15.0
        p = float(probs[spec.BUTTON][row])
        picked = row == chose[spec.BUTTON]
        text(bx, ry, 7, small, CHALK if picked else CHALK_DIM, BUTTON_NAMES.get(name, name), bold=picked)
        pt.bar(X(bx + 44), Y(ry + 3), bwid * u, 7 * u, p, picked)
        text(PANEL_W - 12, ry, 9, small, CHALK if picked else CHALK_DIM, _pct(p))
    return panel + pt.events()


def panel_size(u: float) -> tuple[float, float]:
    """The panel's width and height in frame pixels at unit `u`."""
    return PANEL_W * u, PANEL_H * u


def caption(text: str, x: float, y: float, u: float, colour: str = CHALK_DIM) -> str:
    """A line of small text, its bottom left at (x, y): what the panel is of ("policy v312 · game 41")."""
    return _text(x, y, 1, 14 * u, colour, text)
