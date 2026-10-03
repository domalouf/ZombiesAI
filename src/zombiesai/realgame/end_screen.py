"""The game-over scoreboard's numbers: Points, Kills and Headshots, read off the screen once a game ends.

When the player dies, co-op Nacht shows "GAME OVER / You Survived N Rounds" with the scoreboard drawn above it
(realgame/scoreboard.py finds its header): one row per player, the name and then the three numbers, white on a
translucent red brush stroke -- the same white-on-red as the HUD's points. Kills and headshots are on the screen
nowhere else: the HUD has no kill counter, and a kill's points are not told apart from a hit's by their size
alone (a game can end with kills and no points to show for them).

The numbers are the HUD points' font, drawn smaller: 18 px tall at 1440p where the points are 20, and
narrower still. So the row is cut from the *full-resolution* frame and area-resampled once, by `ROW_SCALE`
(rows, columns), to the shape of the glyphs the points atlas holds, and read by the points reader
(hud/parse.py) column by column. The two factors were fitted on real game-over screens: uniformly by 10/18 a
digit reads at confidence 0.0-0.3, below the reader's bar; these give 0.69-1.0 (median 0.86). Resampling the
HUD's half-scale crop again instead of the full frame is two resamplings, and reads worse still.

Refuse rather than guess, as everywhere the HUD is read: a read is None unless the header is up, every column
holds exactly one confidently read number, the points are a multiple of 10 and the headshots are no more than
the kills. The env (`RealGameEnv`) takes a number only once two looks in a row agree.
"""

from dataclasses import dataclass

import numpy as np

from zombiesai.demos.frames import area_resize
from zombiesai.demos.hud_crops import crop_regions, region_box
from zombiesai.hud.glyphs import LINES, ink, load_atlas
from zombiesai.hud.parse import OK, read_number
from zombiesai.realgame.scoreboard import SCOREBOARD_REGION, scoreboard_shown

# The first player's row under the header, the numbers only (x, y, w, h fractions; the name is further left).
ROW_REGION = (1600 / 2560, 326 / 1440, 460 / 2560, 36 / 1440)
# (rows, columns) of 1440p: the scoreboard's digits come out the shape of the points atlas's.
ROW_SCALE = (0.583, 0.59)
# Where each number lies in the resampled row (271 columns at any 16:9 size). Each is drawn under its header:
# Points from column 24, Kills from 137 and Headshots from 231 for the numbers seen so far.
COLUMNS = {"points": (0, 90), "kills": (90, 180), "headshots": (180, 271)}
LINE = LINES["points"]
BAND = LINE.rows[1] - LINE.rows[0]  # the atlas's rows: one blank above the digits, which fill the rest


@dataclass(frozen=True)
class EndScreen:
    points: int
    kills: int
    headshots: int
    confidence: float  # the least confident digit's

    @property
    def values(self) -> tuple[int, int, int]:
        return self.points, self.kills, self.headshots


def row_crop(frame: np.ndarray) -> np.ndarray:
    """The row's numbers from an (H, W, 3) RGB full-resolution frame, resampled to the atlas's digit size."""
    height, width = frame.shape[:2]
    left, top, w, h = region_box(height, width, ROW_REGION)
    sy, sx = (s * 1440 / height for s in ROW_SCALE)
    return area_resize(frame[top : top + h, left : left + w], max(1, round(h * sy)), max(1, round(w * sx)))


def read_column(inked: np.ndarray):
    """One column's ink -> a `NumberRead`. The band starts a row above the column's first row of ink, as the
    atlas's patches do; a number too near the bottom to fit is padded and will not read confidently."""
    on = np.flatnonzero((inked > LINE.threshold).any(axis=1))
    top = max(0, int(on[0]) - 1) if len(on) else 0
    band = inked[top : top + BAND]
    if band.shape[0] < BAND:
        band = np.pad(band, ((0, BAND - band.shape[0]), (0, 0)))
    return read_number(band, LINE, load_atlas()["points"])


def read_end_screen(frame: np.ndarray | None) -> EndScreen | None:
    """The scoreboard's Points, Kills and Headshots in a full-resolution RGB frame, or None if any is not there
    to read or does not read cleanly."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    frame = frame[..., :3]
    header = crop_regions(frame, {"scores": SCOREBOARD_REGION}, 720 / frame.shape[0])["scores"]
    if not scoreboard_shown(header):
        return None
    inked = ink(row_crop(frame), LINE.ink)
    reads = {name: read_column(inked[:, a:b]) for name, (a, b) in COLUMNS.items()}
    if any(r.status != OK for r in reads.values()):
        return None
    points, kills, headshots = (reads[k].value for k in COLUMNS)
    if points % 10 or headshots > kills:
        return None
    return EndScreen(points, kills, headshots, min(r.confidence for r in reads.values()))
