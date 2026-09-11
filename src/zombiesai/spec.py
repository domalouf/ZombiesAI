"""Versioned contract shared by every backend, agent, checkpoint, and episode directory."""

import functools
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

DECISION_HZ = 15
FRAMES_PER_DECISION = 4

# ---------------------------------------------------------------- actions
# Positive yaw turns right (mouse right); positive pitch looks up.
ACTION_HEADS = ("strafe", "forward", "yaw", "pitch", "fire", "ads", "sprint", "button")
STRAFE, FORWARD, YAW, PITCH, FIRE, ADS, SPRINT, BUTTON = range(len(ACTION_HEADS))
STRAFE_VALUES = (-1, 0, 1)
FORWARD_VALUES = (-1, 0, 1)
YAW_BINS_DEG = (-30.0, -14.0, -6.0, -2.0, 0.0, 2.0, 6.0, 14.0, 30.0)
PITCH_BINS_DEG = (-6.0, -2.0, 0.0, 2.0, 6.0)
BUTTONS = ("none", "use", "reload", "melee", "grenade", "swap")
HOLD_HEADS = ("fire", "ads", "sprint")
ACTION_NVEC = (
    len(STRAFE_VALUES),
    len(FORWARD_VALUES),
    len(YAW_BINS_DEG),
    len(PITCH_BINS_DEG),
    2,
    2,
    2,
    len(BUTTONS),
)
ACT_ENC_DIM = sum(ACTION_NVEC)
PREV_ACTION_HISTORY = 2


def make_action(
    strafe: int = 0,
    forward: int = 0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    fire: int = 0,
    ads: int = 0,
    sprint: int = 0,
    button: str = "none",
) -> np.ndarray:
    """Build a factored action from physical values (e.g. yaw in degrees, button by name)."""
    return np.array(
        (
            STRAFE_VALUES.index(strafe),
            FORWARD_VALUES.index(forward),
            YAW_BINS_DEG.index(float(yaw)),
            PITCH_BINS_DEG.index(float(pitch)),
            int(bool(fire)),
            int(bool(ads)),
            int(bool(sprint)),
            BUTTONS.index(button),
        ),
        dtype=np.int64,
    )


NEUTRAL_ACTION = tuple(int(v) for v in make_action())


def action_tuple(action) -> tuple[int, ...]:
    """Validated factored action as a tuple of Python ints: the per-step fast path for backends."""
    arr = action if isinstance(action, np.ndarray) else np.asarray(action)
    if arr.shape != (len(ACTION_NVEC),) or arr.dtype.kind not in "iu":
        raise ValueError(f"factored action must be {len(ACTION_NVEC)} integers, got {action!r}")
    values = tuple(arr.tolist())
    if not all(0 <= v < n for v, n in zip(values, ACTION_NVEC)):
        raise ValueError(f"factored action {list(values)} out of range for nvec {ACTION_NVEC}")
    return values


def validate_action(action) -> np.ndarray:
    return np.array(action_tuple(action), dtype=np.int64)


def factored_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array(ACTION_NVEC, dtype=np.int64))


# ---------------------------------------------------------------- compact profile (value methods)
_COMPACT_SPECS = (
    {},
    {"forward": 1},
    {"forward": -1},
    {"strafe": -1},
    {"strafe": 1},
    {"forward": 1, "strafe": -1},
    {"forward": 1, "strafe": 1},
    {"forward": 1, "sprint": 1},
    *({"yaw": y} for y in (-30, -14, -6, -2, 2, 6, 14, 30)),
    *({"pitch": p} for p in (-6, -2, 2, 6)),
    *({"fire": 1, "yaw": y} for y in (-6, -2, 0, 2, 6)),
    *({"fire": 1, "ads": 1, "yaw": y} for y in (-2, 0, 2)),
    {"ads": 1},
    {"fire": 1, "pitch": -2},
    {"fire": 1, "pitch": 2},
    *({"forward": 1, "yaw": y} for y in (-14, -6, 6, 14)),
    {"forward": 1, "sprint": 1, "yaw": -14},
    {"forward": 1, "sprint": 1, "yaw": 14},
    {"forward": -1, "fire": 1},
    {"forward": -1, "fire": 1, "yaw": -6},
    {"forward": -1, "fire": 1, "yaw": 6},
    {"strafe": -1, "fire": 1},
    {"strafe": 1, "fire": 1},
    {"forward": 1, "fire": 1},
    *({"button": b} for b in BUTTONS[1:]),
)
COMPACT_ACTIONS = tuple(tuple(int(v) for v in make_action(**kw)) for kw in _COMPACT_SPECS)
N_COMPACT = len(COMPACT_ACTIONS)
_COMPACT_ARRAY = np.array(COMPACT_ACTIONS, dtype=np.int64)
_COMPACT_ARRAY.setflags(write=False)

# Dyadic weights keep the projection's float sums exact, so argmin ties break identically everywhere.
PROJECTION_WEIGHTS = {
    "strafe": 1.0,
    "forward": 1.0,
    "yaw_per_deg": 0.125,
    "pitch_per_deg": 0.25,
    "fire": 3.0,
    "ads": 1.0,
    "sprint": 1.0,
    "button": 4.0,
}


def compact_action_space() -> spaces.Discrete:
    return spaces.Discrete(N_COMPACT)


def expand_compact(index: int) -> np.ndarray:
    return _COMPACT_ARRAY[int(index)].copy()


def _head_cost(head: int) -> np.ndarray:
    """(n_head_values, N_COMPACT) cost of emitting compact action c when the true head value is a."""
    w = PROJECTION_WEIGHTS
    c = _COMPACT_ARRAY[:, head]
    a = np.arange(ACTION_NVEC[head])[:, None]
    if head in (STRAFE, FORWARD):
        values = np.array(STRAFE_VALUES if head == STRAFE else FORWARD_VALUES, dtype=np.float64)
        return np.abs(values[a] - values[c]) * w[ACTION_HEADS[head]]
    if head in (YAW, PITCH):
        bins = np.array(YAW_BINS_DEG if head == YAW else PITCH_BINS_DEG)
        per_deg = w["yaw_per_deg"] if head == YAW else w["pitch_per_deg"]
        return np.abs(bins[a] - bins[c]) * per_deg
    return (a != c).astype(np.float64) * w[ACTION_HEADS[head]]


@functools.cache
def _projection_table() -> np.ndarray:
    total = np.zeros(ACTION_NVEC + (N_COMPACT,), dtype=np.float64)
    for head in range(len(ACTION_NVEC)):
        shape = [1] * len(ACTION_NVEC) + [N_COMPACT]
        shape[head] = ACTION_NVEC[head]
        total += _head_cost(head).reshape(shape)
    table = total.reshape(-1, N_COMPACT).argmin(axis=1).astype(np.uint8)
    table.setflags(write=False)
    return table


def project_to_compact(action) -> int:
    return int(_projection_table()[np.ravel_multi_index(tuple(validate_action(action)), ACTION_NVEC)])


def project_to_compact_batch(actions: np.ndarray) -> np.ndarray:
    return _projection_table()[np.ravel_multi_index(np.asarray(actions).T, ACTION_NVEC)]


# ---------------------------------------------------------------- HUD vector
@dataclass(frozen=True)
class HudField:
    name: str
    scale: float
    log: bool = False


HUD_FIELDS = (
    HudField("round", 30.0),
    HudField("points", 100_000.0, log=True),
    HudField("mag_ammo", 200.0, log=True),
    HudField("reserve_ammo", 1000.0, log=True),
    HudField("grenades", 4.0),
    HudField("damage_flash", 1.0),
    HudField("time_since_damage", 10.0),
    HudField("hud_confidence", 1.0),
    HudField("round_transition", 1.0),
    HudField("downed", 1.0),
    HudField("prompt_price", 100_000.0, log=True),
    HudField("prompt_door", 1.0),
    HudField("prompt_weapon", 1.0),
    HudField("prompt_box", 1.0),
    HudField("prompt_repair", 1.0),
    HudField("time_in_round", 600.0, log=True),
)
HUD_DIM = len(HUD_FIELDS)
HUD_INDEX = {f.name: i for i, f in enumerate(HUD_FIELDS)}
HUD_CLIP = 4.0
_HUD_LOG = np.array([f.log for f in HUD_FIELDS])
_HUD_LOG_IDX = np.flatnonzero(_HUD_LOG)
_HUD_DENOM = np.array([math.log1p(f.scale) if f.log else f.scale for f in HUD_FIELDS])
_HUD_INV_DENOM = 1.0 / _HUD_DENOM


def hud_raw(values: Mapping[str, float]) -> np.ndarray:
    """Raw (unnormalized) HUD readings in field order; every field is required."""
    missing = [f.name for f in HUD_FIELDS if f.name not in values]
    if missing:
        raise KeyError(f"missing HUD fields: {missing}")
    return np.array([float(values[f.name]) for f in HUD_FIELDS], dtype=np.float64)


def encode_hud(raw) -> np.ndarray:
    x = np.array(raw, dtype=np.float64)
    np.maximum(x, 0.0, out=x)
    x[_HUD_LOG_IDX] = np.log1p(x[_HUD_LOG_IDX])
    x *= _HUD_INV_DENOM
    np.minimum(x, HUD_CLIP, out=x)
    return x.astype(np.float32)


def decode_hud(encoded: np.ndarray) -> np.ndarray:
    """Inverse of encode_hud (exact below HUD_CLIP)."""
    x = np.asarray(encoded, dtype=np.float64) * _HUD_DENOM
    return np.where(_HUD_LOG, np.expm1(x), x)


# ---------------------------------------------------------------- sim state vector (state profile only)
# Bearings are egocentric and clockwise-positive (to the right), matching the yaw bins.
ZONES = ("start", "help", "upstairs")
DOORS = ("help_door", "start_debris", "upstairs_door")
STATE_K_ZOMBIES = 8
STATE_DIST_SCALE_M = 20.0
STATE_PITCH_SCALE_DEG = 90.0
STATE_DAMAGE_LOG_SCALE = 1000.0
STATE_RATE_SCALE = 15.0
MAX_PLANKS = 6
STATE_FIELDS = (
    "player_x",
    "player_y",
    "yaw_sin",
    "yaw_cos",
    "pitch",
    *(f"zone_{z}" for z in ZONES),
    *(f"open_{d}" for d in DOORS),
    "weapon_damage",
    "weapon_rate",
    "weapon_auto",
    *(f"zombie{i}_{k}" for i in range(STATE_K_ZOMBIES) for k in ("present", "sin", "cos", "dist")),
    "barrier_sin",
    "barrier_cos",
    "barrier_dist",
    "barrier_planks",
)
STATE_DIM = len(STATE_FIELDS)
STATE_INDEX = {name: i for i, name in enumerate(STATE_FIELDS)}
STATE_ZOMBIE_START = STATE_INDEX["zombie0_present"]


# ---------------------------------------------------------------- observations
PIXELS_SHAPE = (72, 128, 3)
AUDIO_SHAPE = (2, 64)
OBS_KEYS = {
    "pixels": (PIXELS_SHAPE, "uint8", 0, 255),
    "hud": ((HUD_DIM,), "float32", 0.0, HUD_CLIP),
    "prev_actions": ((PREV_ACTION_HISTORY * ACT_ENC_DIM,), "float32", 0.0, 1.0),
    "state": ((STATE_DIM,), "float32", -1.0, 1.0),
    "audio": (AUDIO_SHAPE, "float32", -math.inf, math.inf),
}
OBS_PROFILES = {
    "render": ("pixels", "hud", "prev_actions"),
    "state": ("state", "hud", "prev_actions"),
}


def observation_space(profile: str, audio: bool = False) -> spaces.Dict:
    """Audio is an opt-in flag rather than a profile so enabling it never changes SPEC_VERSION."""
    keys = OBS_PROFILES[profile] + (("audio",) if audio else ())
    out = {}
    for key in keys:
        shape, dtype, low, high = OBS_KEYS[key]
        out[key] = spaces.Box(low=low, high=high, shape=shape, dtype=np.dtype(dtype))
    return spaces.Dict(out)


_ACT_OFFSETS = np.concatenate(([0], np.cumsum(ACTION_NVEC)[:-1]))


@functools.cache
def _action_onehot(action: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(ACT_ENC_DIM, dtype=np.float32)
    out[np.asarray(action) + _ACT_OFFSETS] = 1.0
    out.setflags(write=False)
    return out


def encode_prev_actions(history) -> np.ndarray:
    """One-hot concat of PREV_ACTION_HISTORY factored actions, most recent first."""
    if len(history) != PREV_ACTION_HISTORY:
        raise ValueError(f"need {PREV_ACTION_HISTORY} past actions, got {len(history)}")
    return np.concatenate([_action_onehot(a if isinstance(a, tuple) else action_tuple(a)) for a in history])


# ---------------------------------------------------------------- version hash
def spec_canonical() -> dict:
    return {
        "decision_hz": DECISION_HZ,
        "frames_per_decision": FRAMES_PER_DECISION,
        "action": {
            "heads": ACTION_HEADS,
            "nvec": ACTION_NVEC,
            "strafe_values": STRAFE_VALUES,
            "forward_values": FORWARD_VALUES,
            "yaw_bins_deg": YAW_BINS_DEG,
            "pitch_bins_deg": PITCH_BINS_DEG,
            "buttons": BUTTONS,
            "hold_heads": HOLD_HEADS,
            "prev_action_history": PREV_ACTION_HISTORY,
        },
        "compact": {"actions": COMPACT_ACTIONS, "projection_weights": PROJECTION_WEIGHTS},
        "hud": {"fields": [(f.name, f.scale, f.log) for f in HUD_FIELDS], "clip": HUD_CLIP},
        "state": {
            "fields": STATE_FIELDS,
            "zones": ZONES,
            "doors": DOORS,
            "dist_scale_m": STATE_DIST_SCALE_M,
            "pitch_scale_deg": STATE_PITCH_SCALE_DEG,
            "damage_log_scale": STATE_DAMAGE_LOG_SCALE,
            "rate_scale": STATE_RATE_SCALE,
            "max_planks": MAX_PLANKS,
        },
        "obs": {
            k: {"shape": shape, "dtype": dtype, "low": str(float(lo)), "high": str(float(hi))}
            for k, (shape, dtype, lo, hi) in OBS_KEYS.items()
        },
        "profiles": OBS_PROFILES,
    }


SPEC_VERSION = hashlib.sha256(
    json.dumps(spec_canonical(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]


class SpecMismatchError(RuntimeError):
    pass


def require_spec_version(found: str, where: str) -> None:
    if found != SPEC_VERSION:
        raise SpecMismatchError(
            f"{where} was written under spec {found}, but this code is spec {SPEC_VERSION}; "
            "refusing to mix data whose action/observation layout may differ"
        )
