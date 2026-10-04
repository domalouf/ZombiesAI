"""What the policy was thinking, drawn over a kept film: the overlay for the video about training.

Every film the actors keep (rl/keepsakes.py) has a sidecar, `<film>.brain.jsonl`: a header, then a line per
frame with what the policy made of it (rl/actors.py `brain_line`). This turns that into an ASS subtitle track
and burns it into a copy of the film with ffmpeg's libass, so nothing here touches a pixel itself:

* **top left** -- when in training this was ("Hour 6 · game 812", rl/clock.py) and, for a first, what it is.
* **bottom left** -- the critic's value estimate over the last `HISTORY_S` seconds, as a filled curve: how much
  more reward the policy expects from here. It falls before a death the policy sees coming. Under it, how sure
  the policy was of its action, and how much it wanted the trigger.
* **bottom right** -- what it pressed: the movement keys lit, the turn it chose, and fire / aim / sprint and
  the one button.
* **centre** -- each settled points change, as the reward read it, rising and fading: "+60", "-950".

The value's scale is the film's own (its 2nd to 98th percentile), so the curve uses the panel's height; the
number beside it is the critic's raw output, in the run's normalised reward units.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.realgame.viewer import AMBER, CHALK, CHALK_DIM, GOOD, SANGUINE, _ass, _rect, _text

HISTORY_S = 6.0  # how much of the value curve the panel shows
POPUP_S = 1.2  # how long a points change stays up
KEYS_W = 9.0  # the keys panel's width, in key sizes: room for the longest row of held buttons
PANEL_ALPHA = "60"  # the panels' backing, mostly opaque
INK = "&H101010&"


def read_brain(path: Path) -> tuple[dict, list[dict]]:
    lines = Path(path).read_text().splitlines()
    header = json.loads(lines[0]).get("header", {}) if lines else {}
    return header, [json.loads(line) for line in lines[1:] if line.strip()]


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _event(start: float, end: float, text: str, layer: int = 0) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},Default,,0,0,0,,{text}"


def _title(header: dict) -> str | None:
    from zombiesai.rl.clock import TrainingClock

    clock = header.get("start_clock")
    parts = [TrainingClock.from_dict(clock).label()] if clock else []
    if header.get("moment"):
        from zombiesai.rl.moments import kind

        parts.append(kind(header["moment"]).title)
    return "  —  ".join(parts) or None


def value_scale(lines: list[dict]) -> tuple[float, float]:
    values = np.array([line["v"] for line in lines if isinstance(line.get("v"), (int, float))], float)
    if not len(values):
        return 0.0, 1.0
    lo, hi = np.percentile(values, [2, 98])
    return (float(lo), float(hi)) if hi > lo else (float(lo) - 0.5, float(lo) + 0.5)


def _curve(points: list[tuple[float, float]], x: float, y: float, w: float, h: float, span: float,
           lo: float, hi: float, now: float) -> str | None:
    """The value curve of `points` (time, value) in the last `span` seconds before `now`, filled down to the
    panel's bottom, as one ASS drawing at (x, y) of size (w, h)."""
    shown = [(t, v) for t, v in points if now - span <= t <= now]
    if len(shown) < 2:
        return None
    def px(t):
        return w * (1 - (now - t) / span)
    def py(v):
        return h * (1 - min(1.0, max(0.0, (v - lo) / (hi - lo))))
    path = f"m {px(shown[0][0]):.0f} {h:.0f} " + " ".join(f"l {px(t):.0f} {py(v):.0f}" for t, v in shown)
    path += f" l {px(shown[-1][0]):.0f} {h:.0f}"
    return f"{{\\an7\\pos({x:.0f},{y:.0f})\\p1\\bord0\\shad0\\1c{AMBER}\\1a&H30&}}{path}{{\\p0}}"


def _bar(x: float, y: float, w: float, h: float, share: float, colour: str) -> list[str]:
    share = min(1.0, max(0.0, share))
    return [_rect(x, y, w, h, INK, "40"), _rect(x, y, max(1.0, w * share), h, colour)]


def _keys(action: list[int], x: float, y: float, u: float) -> list[str]:
    """What it pressed, on a panel at (x, y) `KEYS_W` key sizes wide: the movement keys as a WASD cluster lit
    where pressed; to their right the turn it chose and, under it, the held buttons."""
    strafe = spec.STRAFE_VALUES[action[spec.STRAFE]]
    forward = spec.FORWARD_VALUES[action[spec.FORWARD]]
    pad = 0.3 * u
    out = [_rect(x - pad, y - pad, KEYS_W * u + 2 * pad, 2.1 * u + 2 * pad, INK, PANEL_ALPHA)]
    for label, kx, ky, lit in (("W", 1, 0, forward > 0), ("A", 0, 1, strafe < 0), ("S", 1, 1, forward < 0),
                               ("D", 2, 1, strafe > 0)):
        bx, by = x + kx * 1.1 * u, y + ky * 1.1 * u
        out.append(_rect(bx, by, u, u, AMBER if lit else INK, "00" if lit else "50"))
        out.append(_text(bx + u / 2, by + u / 2, 5, u * 0.6, INK if lit else CHALK_DIM, label, bold=True))
    yaw = spec.YAW_BINS_DEG[action[spec.YAW]]
    pitch = spec.PITCH_BINS_DEG[action[spec.PITCH]]
    look = []
    if yaw:
        look.append(("\u2192" if yaw > 0 else "\u2190") + f" {abs(yaw):g}\u00b0")
    if pitch:
        look.append(("\u2191" if pitch > 0 else "\u2193") + f" {abs(pitch):g}\u00b0")
    right = x + 3.6 * u
    out.append(_text(right, y + 0.5 * u, 4, u * 0.55, CHALK, "  ".join(look) or "\u00b7 steady"))
    chips = [name.upper() for name, head in (("fire", spec.FIRE), ("aim", spec.ADS), ("sprint", spec.SPRINT))
             if action[head]]
    button = spec.BUTTONS[action[spec.BUTTON]]
    if button != "none":
        chips.append(button.upper())
    size, cx = u * 0.42, right
    for chip in chips:
        colour = SANGUINE if chip == "FIRE" else AMBER
        out.append(_text(cx, y + 1.6 * u, 4, size, colour, chip, bold=True))
        cx += size * (0.66 * len(chip) + 0.9)  # bold capitals run ~0.65 em
    return out


def build_ass(header: dict, lines: list[dict], width: int, height: int, start: float = 0.0,
              end: float | None = None) -> str:
    """The overlay of `lines` (a sidecar's frames) as an ASS script for a `width` x `height` film, for the
    stretch of it from `start` to `end` seconds, timed from `start`."""
    fps = float(header.get("fps") or spec.DECISION_HZ)
    frames = [line for line in lines if isinstance(line.get("s"), (int, float))]
    end = end if end is not None else (frames[-1]["s"] + 1 / fps if frames else start)
    u = height / 720  # the layout is drawn for 720 rows and scaled
    lo, hi = value_scale(frames)
    curve = [(line["s"], line["v"]) for line in frames if isinstance(line.get("v"), (int, float))]
    events = []

    title = _title(header)
    if title:
        events.append(_event(0, end - start, _rect(24 * u, 20 * u, 30 * u * 0.6 * len(title) + 32 * u, 46 * u,
                                                    INK, PANEL_ALPHA), 0))
        events.append(_event(0, end - start, _text(40 * u, 43 * u, 4, 30 * u, CHALK, title, bold=True), 1))

    px, py, pw, ph = 24 * u, height - 190 * u, 330 * u, 166 * u  # the thinking panel
    key = 40 * u
    kx, ky = width - 24 * u - 0.3 * key - KEYS_W * key, height - 24 * u - 0.3 * key - 2.1 * key  # the keys
    for i, line in enumerate(frames):
        t0 = line["s"]
        t1 = frames[i + 1]["s"] if i + 1 < len(frames) else t0 + 1 / fps
        if t1 <= start or t0 >= end:
            continue
        a, b = max(t0, start) - start, min(t1, end) - start
        parts = [_rect(px, py, pw, ph, INK, PANEL_ALPHA),
                 _text(px + 14 * u, py + 18 * u, 4, 20 * u, CHALK_DIM, "EXPECTED REWARD"),
                 _text(px + pw - 14 * u, py + 18 * u, 6, 22 * u, CHALK,
                       f"{line['v']:+.2f}" if isinstance(line.get("v"), (int, float)) else "–", bold=True)]
        drawn = _curve(curve, px + 14 * u, py + 34 * u, pw - 28 * u, 70 * u, HISTORY_S, lo, hi, t0)
        if drawn:
            parts.append(drawn)
        for row, (label, share, colour) in enumerate((("SURE", line.get("sure"), GOOD),
                                                      ("TRIGGER", line.get("fire"), SANGUINE))):
            ry = py + 116 * u + row * 24 * u
            parts.append(_text(px + 14 * u, ry + 7 * u, 4, 18 * u, CHALK_DIM, label))
            parts += _bar(px + 110 * u, ry, pw - 124 * u, 14 * u, share or 0.0, colour)
        if isinstance(line.get("a"), list) and len(line["a"]) == len(spec.ACTION_HEADS):
            parts += _keys(line["a"], kx, ky, key)
        for layer, part in enumerate(parts):
            events.append(_event(a, b, part, 2 + min(layer, 3)))

        delta = line.get("d")
        if line.get("ev") in ("gain", "spend") and isinstance(delta, (int, float)) and delta:
            p0 = t0 - start
            if 0 <= p0 < end - start:
                colour = GOOD if delta > 0 else SANGUINE
                text = (f"{{\\an5\\move({width / 2:.0f},{height * 0.42:.0f},{width / 2:.0f},{height * 0.30:.0f})"
                        f"\\fad(80,400)\\fs{52 * u:.0f}\\b1\\bord3\\3c{INK}\\shad0\\1c{colour}}}"
                        f"{_ass(f'{int(delta):+d}')}")
                events.append(_event(p0, min(p0 + POPUP_S, end - start), text, 9))

    head = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,DejaVu Sans,{24 * u:.0f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
            "1,0,0,7,0,0,0,1", "", "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    return "\n".join(head + events) + "\n"


def film_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def render(film: Path, out: Path, brain: Path | None = None, start: float = 0.0, end: float | None = None) -> Path:
    """Burn the overlay of `brain` (default: the film's sidecar beside it) into a copy of `film` at `out`."""
    film = Path(film)
    brain = Path(brain) if brain else film.with_suffix(".brain.jsonl")
    header, lines = read_brain(brain)
    width, height = film_size(film)
    ass = Path(out).with_suffix(".ass")
    ass.write_text(build_ass(header, lines, width, height, start, end))
    from zombiesai.rl.best_episode import encoder_args

    cut = ["-ss", f"{start:.3f}"] + (["-t", f"{end - start:.3f}"] if end is not None else [])
    try:
        done = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *cut, "-i", str(film),
             "-vf", f"ass={ass}", *encoder_args(width, height), "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", "160k", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=3600)
        if done.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {done.returncode}: {done.stderr.decode(errors='replace')[-400:]}")
    finally:
        ass.unlink(missing_ok=True)
    return Path(out)
