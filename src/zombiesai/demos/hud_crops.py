"""Full-resolution HUD crops, saved beside the policy frames so a recording can be HUD-parsed later.

The policy sees 128x72, where the points and ammo counters are a smear of a few pixels. The HUD parser (M4)
reads them from the full-resolution screen instead -- and it can only do that for a recording if the
recorder kept those pixels. These crops are what it keeps.

Boxes are fractions of the captured frame, so a resolution change is a rescale rather than a redo (the same
rule the plan sets for the parser's own calibration). They were placed on a 2560x1440 capture of Nacht with
generous margins: the "+10" popups float up and left of the points, and the round counter grows from a
single tally mark into wide numerals by the later rounds. WaW anchors its HUD to the screen corners and
scales it with height, so the fractions hold at any 16:9 resolution; another aspect ratio needs new boxes.

The third box is not in a corner: it is the interaction prompt under the crosshair ("Press & hold F to buy
... [Cost: 1200]", "... to rebuild barrier", the box and the doors). At 128x72 that line is a grey smear one
pixel tall, so the policy can tell a prompt is up but not what it offers or what it costs. It was placed
from the recordings' own frames at every press of F -- the line sits at 0.64-0.67 of the height and the long
buy prompts span 0.31-0.65 of the width -- with a margin, since 128x72 only pins it to within 20 px.

The fourth is the countdown a timed power-up puts below that ("Double Points: 23", "Insta-Kill: 7"): whether
one is running and for how long, which should change how the agent plays. The recordings show Double Points
at 0.77 of the height and Insta-Kill at 0.83, each about 0.37-0.56 of the width, so one box covers both
lines, whichever is up.

Crops are area-downsampled by `HUD_SCALE`. At 0.5 a 2560x1440 capture keeps digits about 12 px tall -- ample
for template matching against a fixed bitmap font -- for 410 KB a decision, ~7.4 GB per 20 minutes (the prompt
and power-up boxes are 266 KB of it), instead of four times that. That is while recording: once the session
ends they are packed as verified video, ~20x smaller (demos/hud_video.py).

Live, the crops are cut from the capture's BGRX buffer as it is (`channels=frames.BGRX`): only the boxes are
read, and at 1440p the points and round boxes are exact 2x2 means (frames.block_mean) -- the same values the
float path gives there, since halving is exact in float32 too.
"""

import numpy as np

from zombiesai.demos.frames import area_resize

# name -> (left, top, width, height) as fractions of the captured frame
HUD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # points (with the "+N" popups), weapon name, grenade count, magazine ticks and reserve ammo
    "points_ammo": (0.8516, 0.8125, 0.1484, 0.1875),
    # the round counter: red tally marks early, numerals later
    "round": (0.0, 0.8056, 0.125, 0.1944),
    # the interaction prompt under the crosshair: what F would buy, open or rebuild here, and its cost
    "prompt": (0.25, 0.61, 0.5, 0.09),
    # the timed power-ups' countdowns below it: Double Points, Insta-Kill, and the seconds they have left
    "powerup": (0.31, 0.73, 0.34, 0.15),
}
HUD_SCALE = 0.5


def region_box(height: int, width: int, frac: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """(left, top, width, height) in pixels, clipped to the frame."""
    fx, fy, fw, fh = frac
    left, top = int(round(fx * width)), int(round(fy * height))
    right, bottom = min(width, int(round((fx + fw) * width))), min(height, int(round((fy + fh) * height)))
    if right <= left or bottom <= top:
        raise ValueError(f"HUD region {frac} is empty on a {width}x{height} frame")
    return left, top, right - left, bottom - top


def crop_shape(height: int, width: int, frac, scale: float = HUD_SCALE) -> tuple[int, int, int]:
    _, _, w, h = region_box(height, width, frac)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale))), 3


def crop_regions(
    frame: np.ndarray, regions: dict = HUD_REGIONS, scale: float = HUD_SCALE, *, channels=None
) -> dict[str, np.ndarray]:
    """Every HUD region of one full-resolution frame, area-downsampled by `scale`, as RGB crops that own their
    memory. `frame` is RGB, or BGRX with `channels=frames.BGRX` (see `frames.area_resize`)."""
    height, width = frame.shape[:2]
    out = {}
    for name, frac in regions.items():
        left, top, w, h = region_box(height, width, frac)
        out_h, out_w, _ = crop_shape(height, width, frac, scale)
        out[name] = area_resize(frame[top : top + h, left : left + w], out_h, out_w, channels=channels)
    return out
