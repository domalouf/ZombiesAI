"""Is the game's console open? Read off the screen, because the console key is a toggle.

Resets type `map nazi_zombie_prototype` into the console. A toggle pressed blind goes wrong the moment the
console is already open -- left open by a reset whose closing press landed during the map load, say. The
policy's keys then type into it ("wwwsssaaa..."), and the next reset's press *closes* it, so the command goes
to the game as key presses instead: `t` opens the chat and the game says "type". So nothing here toggles the
console without first looking.

Plutonium T4's console is a one-line bar across the top of the screen: a flat olive fill, (64, 64, 51), above
and below the input line, with a darker border. Nothing in Nacht's scene is that flat across the whole width.
"""

import numpy as np

# The top strip of the frame, as a HUD crop (demos/hud_crops.py convention: x, y, w, h as fractions). The
# right-hand fifth is left out: Plutonium's watermark sits there.
CONSOLE_REGION = (0.02, 0.0, 0.76, 0.03)
# The fill's two bands inside that crop, as fractions of its height: measured at 2560x1440, where the bar
# spans y = 12..39 with its input text on y = 22..30.
FILL_BANDS = ((0.37, 0.47), (0.74, 0.84))
FILL_RGB = np.array([64.0, 64.0, 51.0])
MAX_COLOUR_ERROR = 10.0  # per channel, of a band's mean
MAX_BAND_STD = 12.0


def console_open(crop: np.ndarray | None) -> bool:
    """True when the console bar is drawn in `crop`, the CONSOLE_REGION crop of a frame."""
    if crop is None or crop.ndim != 3 or crop.shape[0] < 8:
        return False
    h = crop.shape[0]
    for lo, hi in FILL_BANDS:
        band = crop[int(lo * h) : max(int(lo * h) + 1, int(hi * h)), :].astype(np.float32)
        mean = band.reshape(-1, 3).mean(axis=0)
        if np.abs(mean - FILL_RGB).max() > MAX_COLOUR_ERROR or band.std() > MAX_BAND_STD:
            return False
    return True
