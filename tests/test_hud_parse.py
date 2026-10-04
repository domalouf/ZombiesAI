"""The HUD parser on real crops, its temporal checks on synthetic sequences, and parse_hud.py's output.

tests/fixtures/hud/crops.npz holds a few real crops from the recordings (their sources are stored beside
them), with the parts the parser never looks at blanked so the file stays small. What each one shows was
checked by eye.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zombiesai import spec
from zombiesai.demos.clips import ClipWriter
from zombiesai.hud.glyphs import LINES, load_atlas
from zombiesai.hud.parse import ABSENT, OK, ROUND_CAPPED, TRANSITION, UNREADABLE, HudParser, fit_crops
from zombiesai.hud.summary import summarize
from zombiesai.hud.track import (
    Run,
    check_clip,
    classify_points,
    counter_blips,
    roll_consistent,
    round_track,
    settled_runs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hud"
ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def crops():
    with np.load(FIXTURES / "crops.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def parser():
    return HudParser()


def test_atlas_covers_every_digit_in_both_fonts():
    atlas = load_atlas()
    for name in ("points", "ammo"):
        assert set("0123456789") <= set(atlas[name].labels.tolist())
        assert atlas[name].shape == (LINES["points" if name == "points" else "reserve"].rows[1]
                                     - LINES["points" if name == "points" else "reserve"].rows[0],
                                     LINES["points" if name == "points" else "reserve"].width)


# (points, grenades, reserve, reserve drawn red, magazine) per points_ammo fixture; None = must not be OK
POINTS_AMMO = [
    (940, 2, 14, True, 2),  # Colt, red low reserve, 2 loaded ticks
    (820, 4, 131, False, 26),  # STG-44, a yellow "+N" popup at the points' left
    (None, 4, 19, False, 8),  # 1390 over a sunlit wall: the points must not misread
    (4050, None, None, False, None),  # the game-over count-up; the ammo counter is gone
    (3710, 4, 174, False, 20),  # Thompson, full 20
    (3770, 4, 34, False, 1),  # double-barrelled shotgun: dashes, one loaded
    (1100, 4, 46, False, None),  # Kar98k: its dashes run off the crop, so no magazine count
    (2170, 4, 33, False, 1),  # Kar98k with its last round drawn red
]


@pytest.mark.parametrize("i", range(len(POINTS_AMMO)))
def test_points_and_ammo_on_real_crops(crops, parser, i):
    points, grenades, reserve, low, mag = POINTS_AMMO[i]
    r = parser.parse({"points_ammo": crops["points_ammo"][i]})
    source = crops["points_ammo_source"][i]
    for name, want in (("points", points), ("grenades", grenades), ("reserve", reserve), ("mag", mag)):
        got, status = getattr(r, name), getattr(r, f"{name}_status")
        if want is None:
            assert status != OK and got == -1, (source, name, got, status)
        else:
            assert (got, status) == (want, OK), (source, name)
            assert getattr(r, f"{name}_conf") >= 0.5
    assert r.reserve_low == low


def test_game_over_screen_has_no_ammo_counter(crops, parser):
    r = parser.parse({"points_ammo": crops["points_ammo"][3]})
    assert r.grenades_status == ABSENT and r.reserve_status == ABSENT and r.mag_status == ABSENT


# (round, status, flags) per round fixture
ROUNDS = [
    (1, OK, 0),
    (3, OK, 0),
    (4, TRANSITION, 0),  # the round-change flash: white tallies
    (6, TRANSITION, 0),  # the first group white, the new sixth stroke already red
    (8, OK, ROUND_CAPPED),  # 8 strokes: could be round 9, whose stroke is off the crop
]


@pytest.mark.parametrize("i", range(len(ROUNDS)))
def test_round_tallies_on_real_crops(crops, parser, i):
    r = parser.parse({"round": crops["round"][i]})
    assert (r.round, r.round_status, r.round_flags) == ROUNDS[i], crops["round_source"][i]
    assert r.round_conf >= 0.5


def test_black_frames_read_as_absent_not_as_numbers(parser):
    r = parser.parse({"points_ammo": np.zeros((135, 190, 3), np.uint8), "round": np.zeros((140, 160, 3), np.uint8)})
    for name in ("points", "round", "grenades", "reserve", "mag"):
        assert getattr(r, f"{name}_status") == ABSENT and getattr(r, name) == -1


def test_noise_is_never_a_confident_number(parser):
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = parser.parse({"points_ammo": rng.integers(0, 256, (135, 190, 3), dtype=np.uint8),
                          "round": rng.integers(0, 256, (140, 160, 3), dtype=np.uint8)})
        assert r.points_status != OK and r.round_status not in (OK, TRANSITION)


def test_red_scene_over_the_round_counter_is_unreadable(crops, parser):
    crop = crops["round"][1].copy()  # round 3
    crop[60:120, 100:150] = (110, 2, 2)  # something tally-red where no stroke belongs
    assert parser.parse({"round": crop}).round_status == UNREADABLE


def test_the_hurt_screen_red_wash_is_not_round_ten(parser):
    """Measured on a live game going down in round 1: the scene ~(78, 1, 1), the one stroke ~(101, 1, 0). Every
    stroke is tally red then, and the reader used to settle a confident round 10 on it."""
    from zombiesai.hud.parse import _strokes, stroke_evidence

    crop = np.empty((140, 160, 3), np.uint8)
    crop[:] = (78, 1, 1)
    crop.reshape(-1, 3)[_strokes().idx[0]] = (101, 1, 0)
    reds, _ = stroke_evidence(crop)
    assert (reds[~np.isnan(reds)] == 1).all()  # red alone says all ten strokes
    r = parser.parse({"round": crop})
    assert (r.round, r.round_status) == (-1, UNREADABLE)


def test_popup_covering_a_digit_is_not_misread(crops, parser):
    crop = crops["points_ammo"][0].copy()  # 940
    crop[24:34, 60:68] = (255, 255, 0)  # a yellow "+N" over the last digit
    r = parser.parse({"points_ammo": crop})
    assert r.points_status != OK or r.points == 940


def test_other_crop_sizes_are_rescaled_to_the_reference(crops):
    small = crops["points_ammo"][:2, ::2, ::2]
    assert fit_crops(small, "points_ammo").shape == (2, 135, 190, 3)
    assert fit_crops(crops["points_ammo"][:2], "points_ammo") is not None


def test_parse_many_matches_parse(crops, parser):
    n = len(crops["round"])
    out = parser.parse_many({"points_ammo": crops["points_ammo"][:n], "round": crops["round"]}, batch=2)
    for i in range(n):
        r = parser.parse({"points_ammo": crops["points_ammo"][i], "round": crops["round"][i]})
        assert out["points"][i] == r.points and out["round"][i] == r.round and out["mag"][i] == r.mag
    assert out["points_conf"].dtype == np.float32 and out["round_status"].dtype == np.int8


# ---------------------------------------------------------------------------------------- temporal checks


def _ok(values):
    values = np.asarray(values)
    return values, values >= 0


def test_settled_runs_skip_unreadable_steps_and_drop_short_runs():
    v, ok = _ok([500, 500, -1, 500, 510, 520, 530, 530, 530, -1, 530])
    runs = settled_runs(v, ok)
    assert [(r.start, r.end, r.value) for r in runs] == [(0, 3, 500), (6, 10, 530)]
    assert [(r.start, r.end) for r in settled_runs(v, ok, max_gap=0)] == [(6, 8)]


def test_point_changes_are_classified_by_what_the_game_allows():
    vals = [500, 560, 1560, 610, 580, 4050, 500, 500 + 7, 1200, 30000, 30010]
    runs = [Run(10 * i, 10 * i + 5, v, 6) for i, v in enumerate(vals)]
    kinds = [e.kind for e in classify_points(runs)]
    assert kinds == ["gain",  # a kill
                     "gain",  # a train of kills, doubled
                     "spend",  # a door
                     "downed",  # 5% of 610, rounded to 10
                     "game_over",  # the count-up to the total score, then a new game
                     "new_game",
                     "implausible",  # not a multiple of 10
                     "implausible",  # 507 -> 1200
                     "implausible",  # far too much at once
                     "gain"]


def test_a_restart_after_dying_in_round_one_is_a_new_game():
    runs = [Run(0, 50, 500, 51), Run(55, 90, 470, 36), Run(160, 200, 500, 41)]
    assert [e.kind for e in classify_points(runs)] == ["downed", "new_game"]


def test_the_rolling_counter_is_consistent_and_a_misread_is_not():
    v = np.array([2070] * 4 + [2139, 2164] + [2170] * 4 + [2170, 2179, 2170, 2170, 2170])
    ok = np.ones(len(v), bool)
    runs = settled_runs(v, ok)
    good = roll_consistent(v, ok, runs)
    assert good[:10].all()
    v2 = v.copy()
    v2[5] = 8164  # a digit misread mid-roll
    good2 = roll_consistent(v2, ok, settled_runs(v2, ok))
    assert not good2[5] and good2[:5].all()


def test_counter_blips_flag_one_step_misreads():
    v, ok = _ok([28, 28, 2, 28, 28, 27, 27, -1, 27])
    assert np.flatnonzero(counter_blips(v, ok)).tolist() == [2]


def _round_sequence():
    """Round 7 -> 8 -> (flash) -> 9, which still shows 8 strokes, then a new game."""
    seq = [(7, OK)] * 30 + [(7, TRANSITION)] * 20 + [(-1, ABSENT)] * 20 + [(8, TRANSITION)] * 10 + [(8, OK)] * 40
    seq += [(8, TRANSITION)] * 20 + [(-1, ABSENT)] * 20 + [(8, OK)] * 40 + [(-1, ABSENT)] * 30 + [(1, OK)] * 20
    values = np.array([v for v, _ in seq], np.int32)
    status = np.array([s for _, s in seq], np.int8)
    flags = np.where(values == 8, ROUND_CAPPED, 0).astype(np.int8)
    return values, status, flags


def test_round_nine_is_inferred_from_the_round_change_flash():
    values, status, flags = _round_sequence()
    _, events, inferred, good = round_track(values, status, flags)
    assert [(e.before, e.after, e.kind) for e in events] == [(7, 8, "next"), (8, 9, "inferred_next"), (9, 1, "new_game")]
    assert inferred.max() == 9 and inferred[-1] == 1
    assert good[status != ABSENT].all()


def test_a_round_that_jumps_is_implausible():
    values = np.array([2] * 20 + [5] * 20, np.int32)
    status = np.zeros(40, np.int8)
    _, events, _, _ = round_track(values, status, np.zeros(40, np.int8))
    assert [e.kind for e in events] == ["implausible"]


def _parsed(points, rounds):
    n = len(points)
    out = {"points": np.asarray(points, np.int32), "round": np.asarray(rounds, np.int32)}
    out["points_status"] = np.where(out["points"] >= 0, OK, ABSENT).astype(np.int8)
    out["round_status"] = np.where(out["round"] >= 0, OK, ABSENT).astype(np.int8)
    out["round_flags"] = np.zeros(n, np.int8)
    for name in ("grenades", "reserve", "mag"):
        out[name] = np.full(n, -1, np.int32)
        out[f"{name}_status"] = np.full(n, ABSENT, np.int8)
    return out


def test_summary_counts_games_rounds_and_scores():
    points = [500] * 30 + [560] * 30 + [620] * 30 + [590] * 10 + [2000, 3000] + [3120] * 20 + [-1] * 20 + [500] * 40
    rounds = [1] * 60 + [2] * 62 + [-1] * 20 + [-1] * 10 + [1] * 30
    check = check_clip(_parsed(points, rounds))
    s = summarize(check, hz=15.0)
    assert s["game_overs"] == 1 and len(s["games"]) == 2
    g = s["games"][0]
    assert g["rounds_reached"] == 2 and g["score"] == 3120 and g["downs"] == 1 and g["points_gained"] == 120
    assert g["peak_points"] == 620 and g["game_over"] and g["started_in_clip"]
    assert s["highest_round"] == 2 and s["best_score"] == 3120 and s["final_points"] == 500
    assert s["read"]["points"]["consistent_rate"] == 1.0


# ---------------------------------------------------------------------------------------- the script


def test_parse_hud_script_writes_hud_npz_and_a_summary(tmp_path, crops):
    clip = tmp_path / "clips" / "demo_x"
    writer = ClipWriter(clip, source={"kind": "test"})
    pa, rd = crops["points_ammo"], crops["round"]
    for k in range(40):
        writer.add(np.zeros(spec.PIXELS_SHAPE, np.uint8), hud={"points_ammo": pa[0 if k < 20 else 1],
                                                               "round": rd[1 if k < 20 else 4]})
    writer.close()
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "parse_hud.py"), str(tmp_path / "clips"),
                           "--out", str(out)], capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stderr
    assert not (clip / "hud.npz").exists()  # --out: the clip itself is left alone
    with np.load(out / "demo_x" / "hud.npz") as z:
        for name in ("points", "round", "grenades", "reserve", "mag"):
            assert z[name].shape == (40,) and z[f"{name}_conf"].shape == (40,) and z[f"{name}_status"].shape == (40,)
        assert z["points"][0] == 940 and z["points"][-1] == 820 and z["round"][0] == 3 and z["round"][-1] == 8
        assert z["points_settled"].shape == (40,) and z["round_inferred"][-1] == 8
        meta = json.loads(str(z["meta"]))
    summary = json.loads((out / "demo_x" / "hud_summary.json").read_text())
    assert meta["summary"]["highest_round"] == summary["highest_round"] == 8
    assert summary["read"]["round"]["implausible_changes"] == 1  # 3 -> 8 in one step: flagged, not hidden
    assert summary["peak_points"] == 940 and "points_per_minute" in summary and summary["games"]
    assert "demo_x" in done.stdout


def test_online_tracker_agrees_with_the_offline_checks():
    from zombiesai.hud.parse import HudReading
    from zombiesai.hud.track import HudTracker

    values, status, flags = _round_sequence()
    points = np.array([500] * 60 + [560] * 20 + [5555, 560, 560] + [620] * 30 + [590] * 20 + [-1] * 60 + [500] * 40)
    points = np.resize(points, len(values))
    tracker = HudTracker()
    seen_rounds, events = [], []
    for p, rv, rs in zip(points, values, status):
        t = tracker.step(HudReading(points=int(p), points_status=OK if p >= 0 else ABSENT, round=int(rv), round_status=int(rs)))
        seen_rounds.append(t.round)
        if t.points_event:
            events.append((t.points_event, t.points))
    _, _, inferred, _ = round_track(values, status, flags)
    assert max(seen_rounds) == inferred.max() == 9 and seen_rounds[-1] == 1
    assert events[0] == ("gain", 560)  # the one-step 5555 never settled
    assert [k for k, _ in events][:3] == ["gain", "gain", "downed"]


def test_online_tracker_holds_the_round_through_a_jump_the_game_cannot_make():
    from zombiesai.hud.parse import HudReading
    from zombiesai.hud.track import HudTracker

    tracker = HudTracker()
    seq = [(1, OK)] * 20 + [(10, OK)] * 40 + [(8, OK)] * 20 + [(1, OK)] * 10 + [(2, OK)] * 10
    seen = [tracker.step(HudReading(round=v, round_status=s)).round for v, s in seq]
    assert seen[19] == 1 and max(seen[:90]) == 1  # 1 -> 10 and 1 -> 8 are misreads: round 1 held
    assert seen[-1] == 2  # the next round is still taken


def test_online_tracker_holds_the_gun_while_its_name_is_faded_out():
    from zombiesai.hud.parse import HudReading
    from zombiesai.hud.track import HudTracker
    from zombiesai.hud.weapons import WEAPON_INDEX

    colt, kar = WEAPON_INDEX["colt"], WEAPON_INDEX["kar98k"]
    tracker = HudTracker()
    seq = [(colt, OK)] * 5 + [(-1, ABSENT)] * 30 + [(kar, OK)] + [(-1, UNREADABLE)] * 3 + [(kar, OK)] * 3
    seen = []
    for v, s in seq:
        t = tracker.step(HudReading(weapon=v, weapon_status=s))
        seen.append((t.weapon, t.weapon_changed))
    assert seen[1] == (-1, False) and seen[2] == (colt, False)  # settles on the third read; no swap from nothing
    assert all(w == colt for w, _ in seen[2:40])  # held through the fade; one kar98k read is not a swap yet
    # unreadable steps do not break a run of reads: the third kar98k read settles it
    assert seen[40] == (kar, True) and seen[41] == (kar, False) and sum(c for _, c in seen) == 1


def test_lowres_round_reader_counts_tallies_in_policy_frames(crops):
    from zombiesai.demos.frames import area_resize
    from zombiesai.hud.lowres import find_offset, round_from_frames

    frames = np.zeros((3, *spec.PIXELS_SHAPE), np.uint8)
    frames[:] = 30
    for i, k in enumerate((0, 1, 4)):  # round 1, 3, 8 in red
        frames[i, 58:72, 0:16] = area_resize(crops["round"][k], 14, 16)
    assert find_offset(np.repeat(frames, 4, axis=0), stride=1) == (0, 0)
    out = round_from_frames(frames)
    assert out["round"].tolist() == [1, 3, 8] and (out["round_status"] == OK).all()
    assert out["round_flags"].tolist() == [0, 0, ROUND_CAPPED]


def test_stroke_evidence_is_the_plain_per_stroke_means_bit_for_bit(crops):
    """The round reader only converts the pixels its strokes and rings read, and skips mean()'s wrapper;
    the evidence must be exactly the plain computation's, or a confidence would move."""
    from zombiesai.hud.parse import N_STROKES, _strokes, stroke_evidence

    s = _strokes()

    def plain(crop):
        x = crop.reshape(-1, 3).astype(np.int16)
        red = (x[:, 0] > 60) & (x[:, 1] < 30) & (x[:, 2] < 30)
        lum = x.sum(1) * (1 / 3)
        reds, contrast = np.zeros(N_STROKES), np.zeros(N_STROKES)
        for k in range(N_STROKES):
            if len(s.idx[k]) == 0:
                reds[k] = contrast[k] = np.nan
                continue
            reds[k] = red[s.idx[k]].mean()
            contrast[k] = lum[s.idx[k]].mean() - (lum[s.ring[k]].mean() if len(s.ring[k]) else 0.0)
        return reds, contrast

    rng = np.random.default_rng(0)
    samples = list(crops["round"]) + [rng.integers(0, 256, (140, 160, 3), dtype=np.uint8) for _ in range(10)]
    samples += [np.clip(c.astype(int) + rng.integers(-40, 40, c.shape), 0, 255).astype(np.uint8) for c in crops["round"]]
    for crop in samples:
        for got, want in zip(stroke_evidence(crop), plain(crop)):
            np.testing.assert_array_equal(got.view(np.uint64), want.view(np.uint64))
