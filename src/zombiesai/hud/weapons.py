"""Which gun is up: the weapon's name, read off the HUD above the grenade count.

WaW prints the held weapon's name ("Colt M1911", "Kar98k") in white between the points and the grenade icon,
right-aligned, in the points_ammo crop. It is drawn whenever the ammo counter is -- a weapon swap or a shot
brings both up, and both fade out a few seconds after the last shot -- so a step without the counter has no
name either; `track.HudTracker` holds the last one through those.

The name is read as a whole word, not letter by letter: it is drawn at a fixed place in a fixed font, so each
weapon's name is one fixed picture, and a template per weapon (harvested from real frames, like the digits;
docs/hud.md, "Rebuilding the atlas") recognises it whatever the scene behind it. The ink rule is the ammo
digits' (light, neutral, thinner than a 5 px opening), keeping only white: the text is ~(235, 235, 230) with a
dark outline. A name with no template -- a box gun no recording has shown yet -- matches nothing well and reads
UNREADABLE rather than as the closest known gun.

`WEAPONS` is the vocabulary a read's value indexes. Its order is append-only, so a stored read keeps meaning
the same gun. `mag` is the gun's stock magazine (WaW's own sizes); the magazine reader only uses it to say how
many rounds a gun has when its marks run off the crop (parse.read_mag), so a wrong or missing size costs an
exact read, never a wrong one.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

from zombiesai.hud.glyphs import PATCH_FLOOR, REJECT, TOPHAT, GlyphSet

NAMES_PATH = Path(__file__).with_name("weapon_names.npz")


@dataclass(frozen=True)
class Weapon:
    key: str  # what the template labels say
    name: str  # what the HUD says
    mag: int | None = None  # rounds in a full magazine; None where the HUD does not count marks


WEAPONS = (
    Weapon("colt", "Colt M1911", 8),
    Weapon("kar98k", "Kar98k", 5),
    Weapon("m1_carbine", "M1A1 Carbine", 15),
    Weapon("thompson", "Thompson", 20),
    Weapon("double_barrel", "Double-Barreled Shotgun", 2),
    Weapon("sawed_off", "Sawed-Off Double-Barreled Shotgun", 2),
    Weapon("trench_gun", "M1897 Trench Gun", 6),
    Weapon("m1_garand", "M1 Garand", 8),
    Weapon("gewehr43", "Gewehr 43", 10),
    Weapon("springfield", "Springfield", 5),
    Weapon("bar", "BAR", 20),
    Weapon("stg44", "STG-44", 30),
    Weapon("mp40", "MP40", 32),
    Weapon("type100", "Type 100", 30),
    Weapon("ppsh", "PPSh-41", 71),
    Weapon("fg42", "FG42", 20),
    Weapon("mg42", "MG42", 125),
    Weapon("browning", "Browning M1919", 125),
    Weapon("ptrs41", "PTRS-41", 5),
    Weapon("panzerschreck", "Panzerschreck"),
    Weapon("flamethrower", "M2 Flamethrower"),
    Weapon("ray_gun", "Ray Gun", 20),
)
WEAPON_INDEX = {w.key: i for i, w in enumerate(WEAPONS)}

# The name's band in the reference points_ammo crop (hud_crops.py at 2560x1440, half scale): "Colt M1911"
# fills rows 56-68 and ends at column ~123; shorter names end at the same place, longer ones start further left.
NAME_ROWS = (53, 72)
NAME_COLS = (0, 150)
NAME_SHAPE = (NAME_ROWS[1] - NAME_ROWS[0], NAME_COLS[1] - NAME_COLS[0])
# A name is a few hundred inked pixels; less than this is a speck, a fading tail or scene.
NAME_MIN_INK = 30.0
# Mean squared difference (of 0..1 ink) to the best template. Measured on 1,173 live frames: a name sits at
# 0.0001-0.008 from its own weapon's templates (also with the templates built from other games), and at 0.086+
# from the other weapon's; a pale wall with no name is 0.073+ from any name. Within NAME_MAX_DIST confidence
# falls linearly; the margin to the best other label must be NAME_FULL_MARGIN for full confidence (measured
# 0.035+).
NAME_MAX_DIST = 0.04
NAME_FULL_MARGIN = 0.02
NAME_MIN_CONFIDENCE = 0.5


def name_ink(band: np.ndarray) -> np.ndarray:
    """(..., h, w, 3) uint8 -> (..., h, w) float32: how much each pixel looks like the name's white text."""
    x = band.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]  # channel by channel: a reduction over the last axis is ~50x slower
    lo = np.minimum(np.minimum(r, g), b)
    hi = np.maximum(np.maximum(r, g), b)
    v = np.clip((lo - 100) * (1 / 120), 0.0, 1.0) * (hi - lo <= 40)
    size = (1,) * (v.ndim - 2) + (TOPHAT, TOPHAT)
    v = v - ndimage.grey_opening(v.astype(np.float32), size=size)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def name_band(crops: np.ndarray) -> np.ndarray:
    return crops[..., NAME_ROWS[0] : NAME_ROWS[1], NAME_COLS[0] : NAME_COLS[1], :]


def signature(inked: np.ndarray) -> np.ndarray:
    """A name's ink scaled so its brightest pixel is 1 (a fading name keeps its shape, not its level)."""
    peak = inked.max(axis=(-2, -1), keepdims=True)
    return (inked / np.maximum(peak, PATCH_FLOOR)).astype(np.float32)


@lru_cache(maxsize=4)
def load_names(path: str | Path = NAMES_PATH) -> GlyphSet:
    """The weapon-name templates: labels are `Weapon.key`s, or REJECT for things that are not a name."""
    with np.load(path) as z:
        return GlyphSet(z["templates"].astype(np.float32), z["labels"])


def save_names(path: str | Path, templates: np.ndarray, labels, meta: str = "{}") -> None:
    unknown = sorted(set(labels) - set(WEAPON_INDEX) - {REJECT})
    if unknown:
        raise ValueError(f"labels name no weapon in WEAPONS: {unknown}")
    np.savez_compressed(path, templates=np.asarray(templates, np.float16), labels=np.asarray(labels, dtype=str),
                        meta=np.array(meta))


@dataclass
class WeaponRead:
    value: int = -1  # index into WEAPONS
    confidence: float = 0.0
    status: int = 1  # parse.ABSENT


def read_weapon(crop: np.ndarray, names: GlyphSet) -> WeaponRead:
    """One reference-size points_ammo crop -> the weapon whose name is drawn in it."""
    from zombiesai.hud.parse import ABSENT, OK, UNREADABLE

    inked = name_ink(name_band(crop))
    if float(inked.sum()) < NAME_MIN_INK:
        return WeaponRead(status=ABSENT)
    labels, dist, margin = names.classify(signature(inked)[None])
    label, d, m = str(labels[0]), float(dist[0]), float(margin[0])
    conf = float(np.clip(1 - d / NAME_MAX_DIST, 0, 1) * np.clip(m / NAME_FULL_MARGIN, 0, 1))
    if label == REJECT or conf < NAME_MIN_CONFIDENCE:
        return WeaponRead(-1, conf, UNREADABLE)
    return WeaponRead(WEAPON_INDEX[label], conf, OK)
