"""Is the scoreboard drawn? In Plutonium's co-op Nacht that is how going down shows.

`map` from Plutonium's LAN menu starts a *co-op* game, and co-op WaW has last stand: a downed player lies on
the floor with a pistol until bleed-out, and nothing on the HUD says so -- no points penalty, unlike the solo
game the demos were recorded in, where a down is the game over. What co-op does do is draw the scoreboard, by
itself, the moment the player is downed or dead, and keep it up through the game-over screen. The policy has
no scores key, so once a reset has cleared the scoreboard (a fresh co-op game opens with it drawn, until the
scores key is pressed), the scoreboard coming back means the player went down: the episode's end, as in solo.

Read by the shape of its header, "Points  Kills  Headshots": the template's letter pixels against the ring just
around them. White text scores ~+140 over any background; fog, sky and scenery score near zero because their
brightness does not follow the letters' outline.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

# The header and the first row under it, as a HUD crop (x, y, w, h fractions; half scale at 1440p).
SCOREBOARD_REGION = (0.60, 0.185, 0.22, 0.06)
TEMPLATE_PATH = Path(__file__).with_name("scoreboard_header.npy")  # (26, W) bool: the header's letters
MIN_CONTRAST = 60.0


@lru_cache(maxsize=1)
def _template() -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import binary_dilation

    letters = np.load(TEMPLATE_PATH)
    ring = binary_dilation(letters, iterations=2) & ~letters
    return letters, ring


def header_contrast(crop: np.ndarray | None) -> float:
    """Mean brightness on the header's letters minus the ring around them, in a SCOREBOARD_REGION crop."""
    letters, ring = _template()
    if crop is None or crop.ndim != 3 or crop.shape[0] < letters.shape[0] or crop.shape[1] != letters.shape[1]:
        return 0.0
    lum = crop[: letters.shape[0]].astype(np.float32).mean(axis=-1)
    return float(lum[letters].mean() - lum[ring].mean())


def scoreboard_shown(crop: np.ndarray | None) -> bool:
    return header_contrast(crop) > MIN_CONTRAST
