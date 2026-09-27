import numpy as np

from zombiesai.realgame.scoreboard import _template, header_contrast, scoreboard_shown


def crop_over(background: np.ndarray, text: bool) -> np.ndarray:
    """The header as WaW draws it: near-white letters with a dark drop shadow around them."""
    letters, ring = _template()
    crop = background.copy()
    if text:
        top = crop[: letters.shape[0]]
        top[ring] = (top[ring] * 0.35).astype(np.uint8)
        top[letters] = 235
    return crop


def test_the_header_is_seen_over_any_background_and_scenery_is_not():
    letters, _ = _template()
    h, w = letters.shape[0] + 18, letters.shape[1]
    rng = np.random.default_rng(1)
    backgrounds = {
        "dark": np.full((h, w, 3), 30, np.uint8),
        "fog": np.full((h, w, 3), 200, np.uint8),
        "busy": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
    }
    for name, bg in backgrounds.items():
        assert scoreboard_shown(crop_over(bg, True)), name
        assert not scoreboard_shown(crop_over(bg, False)), name
    assert header_contrast(None) == 0.0 and not scoreboard_shown(np.zeros((5, 5, 3), np.uint8))
