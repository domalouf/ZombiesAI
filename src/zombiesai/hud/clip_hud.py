"""Parse a whole clip's HUD crops and write what was read: hud.npz plus a JSON summary.

    hud.npz
      <field>, <field>_conf, <field>_status   the per-step reads (hud/parse.py): points, round, grenades,
                                               reserve, mag; status 0 ok, 1 absent, 2 unreadable,
                                               3 transition (round only); value -1 when there is none
      round_flags, reserve_low                 round 8 that may be 9; reserve drawn red
      points_good, round_good                  (T,) bool: the read passed the temporal checks (hud/track.py)
      points_settled                           (T,) int32: the last settled points value (-1 before one)
      round_inferred                           (T,) int32: the round, from settled counts and round changes
      meta                                     JSON: atlas and parser constants, the summary
    hud_summary.json                           hud/summary.py's summary
"""

import json
import time
from pathlib import Path

import numpy as np

from zombiesai.demos.clips import FLAG_BAD_STEP, FLAG_NOT_PLAYING, Clip, load_clip
from zombiesai.hud.parse import HudParser
from zombiesai.hud.summary import summarize
from zombiesai.hud.track import check_clip

REGIONS = ("points_ammo", "round")
HUD_FORMAT = 1


def parse_clip(clip: Clip | str | Path, parser: HudParser | None = None, lowres: bool = False) -> tuple[dict, object, dict]:
    """-> (parsed arrays, ClipCheck, summary). A clip without HUD crops raises ValueError, unless `lowres`:
    then the round alone is read from the policy frames (hud/lowres.py) and every other field is absent."""
    clip = clip if isinstance(clip, Clip) else load_clip(clip)
    crops = {name: clip.hud(name) for name in REGIONS}
    missing = [k for k, v in crops.items() if v is None]
    t0 = time.perf_counter()
    source = "hud_crops"
    if missing and not lowres:
        raise ValueError(f"{clip.path} has no HUD crops for {missing} (recorded before HUD crops existed?)")
    elif missing:
        from zombiesai.hud.lowres import find_offset, round_from_frames

        offset = find_offset(clip.frames)
        parsed = _absent_fields(clip.n_steps)
        parsed.update(round_from_frames(clip.frames, offset))
        source = f"policy_frames (round only, tally offset {offset})"
    else:
        parsed = (parser or HudParser()).parse_many(crops)
    seconds = time.perf_counter() - t0
    n = len(parsed["points"])
    playing = (clip.flags[:n] & (FLAG_NOT_PLAYING | FLAG_BAD_STEP)) == 0
    fire = None if clip.actions is None else clip.actions[:n, 4]
    check = check_clip(parsed, fire=fire, playing=playing)
    summary = summarize(check, playing=playing)
    summary["hud_source"] = source
    summary["parse_seconds"] = round(seconds, 2)
    summary["parse_ms_per_step"] = round(1000 * seconds / max(1, n), 3)
    return parsed, check, summary


def _absent_fields(n: int) -> dict[str, np.ndarray]:
    from zombiesai.hud.parse import ABSENT, HudReading, _as_array

    blank = HudReading()
    out = {k: _as_array(k, [getattr(blank, k)] * n) for k in HudReading.field_names()}
    out.update({k: np.full(n, ABSENT, np.int8) for k in out if k.endswith("_status")})
    return out


def tracks(check, n: int) -> dict[str, np.ndarray]:
    settled = np.full(n, -1, np.int32)
    for r in check.points_runs:
        settled[r.start :] = r.value  # later runs overwrite: each holds until the next settles
    return {
        "points_good": check.points_good,
        "round_good": check.round_good,
        "points_settled": settled,
        "round_inferred": check.round_inferred,
    }


def write_hud(out_dir: str | Path, parsed: dict, check, summary: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(parsed["points"])
    meta = {"format": HUD_FORMAT, "summary": summary}
    np.savez_compressed(out_dir / "hud.npz", **parsed, **tracks(check, n), meta=np.array(json.dumps(meta)))
    (out_dir / "hud_summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir / "hud.npz"


def load_hud(path: str | Path) -> dict:
    """hud.npz (or the clip directory holding one) -> {array name: array, "meta": dict}."""
    path = Path(path)
    if path.is_dir():
        path = path / "hud.npz"
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out
