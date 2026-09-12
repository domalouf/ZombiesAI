"""A 3x5 pixel font for the rendered HUD: digits, capitals, and the few symbols a WaW HUD needs."""

import functools

import numpy as np

GLYPH_W, GLYPH_H = 3, 5
# Each glyph is 5 rows of 3 pixels, top to bottom.
_GLYPHS = {
    "0": "111101101101111",
    "1": "010110010010111",
    "2": "111001111100111",
    "3": "111001111001111",
    "4": "101101111001001",
    "5": "111100111001111",
    "6": "111100111101111",
    "7": "111001001001001",
    "8": "111101111101111",
    "9": "111101111001111",
    "A": "010101111101101",
    "B": "110101110101110",
    "C": "011100100100011",
    "D": "110101101101110",
    "E": "111100110100111",
    "F": "111100110100100",
    "G": "011100101101011",
    "H": "101101111101101",
    "I": "111010010010111",
    "J": "001001001101010",
    "K": "101101110101101",
    "L": "100100100100111",
    "M": "101111111101101",
    "N": "110101101101101",
    "O": "010101101101010",
    "P": "110101110100100",
    "Q": "010101101110011",
    "R": "110101110101101",
    "S": "011100010001110",
    "T": "111010010010010",
    "U": "101101101101111",
    "V": "101101101101010",
    "W": "101101111111101",
    "X": "101101010101101",
    "Y": "101101010010010",
    "Z": "111001010100111",
    "/": "001001010100100",
    "[": "110100100100110",
    "]": "011001001001011",
    ":": "000010000010000",
    ".": "000000000000010",
    "-": "000000111000000",
    "+": "000010111010000",
    " ": "000000000000000",
}


@functools.cache
def _mask(ch: str, scale: int) -> np.ndarray:
    bits = np.array([b == "1" for b in _GLYPHS.get(ch, _GLYPHS[" "])]).reshape(GLYPH_H, GLYPH_W)
    return np.kron(bits, np.ones((scale, scale), dtype=bool)).astype(bool)


def text_width(text: str, scale: int) -> int:
    return max(0, len(text) * (GLYPH_W + 1) - 1) * scale


def _blit(img: np.ndarray, mask: np.ndarray, x: int, y: int, color) -> None:
    h, w = mask.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1][mask[y0 - y : y1 - y, x0 - x : x1 - x]] = color


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int, color, shadow=(0, 0, 0)) -> None:
    """Draw text with its top-left corner at (x, y), clipped to the image. Shadows only at scale 2 and up."""
    passes = [(0, color)]
    if shadow is not None and scale >= 2:
        passes.insert(0, (scale // 2, shadow))
    for offset, fill in passes:
        cx = x + offset
        for ch in text.upper():
            _blit(img, _mask(ch, scale), cx, y + offset, fill)
            cx += (GLYPH_W + 1) * scale
