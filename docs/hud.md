# Reading the HUD (M4)

`src/zombiesai/hud/` turns a recording's full-resolution HUD crops (`demos/hud_crops.py`) into numbers:
points, round, grenades, reserve ammo and the magazine, each with a confidence and a status, then checks
them against how the game can actually change them. It runs on anything with the crops: demos from
`record_demo.py` and AI runs from `play_real.py`, raw or packed as video.

```
uv run python scripts/parse_hud.py data/demos/demo_0002        # writes demo_0002/hud.npz + hud_summary.json
uv run python scripts/parse_hud.py data/demos runs/play --dry-run   # just the comparison table
uv run python scripts/parse_hud.py runs/play --out /tmp/hud    # write elsewhere, leave the clips alone
uv run python scripts/parse_hud.py data/demos --lowres         # clips without crops: round only, see below
```

## How it reads

WaW draws its numbers in a fixed font at fixed places, so each digit is matched against glyphs harvested
from the recordings themselves, not OCR'd (PLAN.md, "Perception"). Per text line:

1. **Ink.** A colour rule turns the line's pixels into 0..1 "looks like text". Points are white on an
   opaque red brush stroke: ink is min(G, B), which is ~255 on a digit, ~3 on the brush and ~0 on the
   yellow "+N" popups that float over the points. Grenades and reserve ammo are light grey, or red when
   low, drawn straight on the scene: ink is R where G ≈ B (brown walls, blue stone and muzzle flash are
   not), followed by a 5 px top-hat so a pale wall behind the counter is not ink.
2. **Segment.** Columns with ink form runs, one per digit (the font leaves a gap). Runs shorter than a
   digit (tick marks, dashes, specks) are dropped; digits closer than 7 columns belong to one number.
   The points line is anchored: it must start at columns 41-48.
3. **Match.** Each run becomes a 12x10 (points) or 14x11 (ammo) patch, left-aligned, scaled to a peak of
   1, and goes to its nearest template in `hud/glyph_atlas.npz` (sum of squared differences). The atlas
   holds ~33 digit templates per font -- several per digit, one per sub-pixel phase the 0.5x downsample
   produces -- plus a few reject templates ("?") for things that segment like digits and are not.
   Confidence per digit comes from the distance to the best template and the margin to the best other
   digit; a number's confidence is its worst digit's. A run too wide for one digit is split at the cut
   that reads best; if the right half is scene showing through the brush, it is dropped.
4. **Refuse rather than guess.** A read becomes UNREADABLE when a yellow popup sits on the number or
   where a further digit would be (a popup over the last digit would otherwise turn 940 into 94), when a
   text-bright wall is right beside an ammo number (it can swallow a digit the same way), when another
   blob sits right beside an unanchored number, or when any glyph is a reject.

**Round** is not glyph-matched. Rounds 1-10 are chalk tallies at fixed places, stored in the atlas as one
mask per stroke (harvested from frames at each count; the second group is the first shifted 103 px, which
is how strokes 9 and 10, never recorded, are placed). A stroke is present when it is solid tally red
(~(108, 1, 0), which nothing in Nacht's scene matches) and, during a round change, when it is clearly
brighter than the scene around it: the tallies flash white/pink for ~10 s, go blank, and come back with
the new count. Strokes must be present in order. The ninth stroke is just off the right edge of the round
crop, so 9 looks like 8; the reader reports 8 with `ROUND_CAPPED` and the tracker counts the round change.
Round 11+ is drawn in numerals, which no recording shows yet -- they read as UNREADABLE.

**Magazine** is the row of marks left of the reserve: thin ticks two columns apart for most guns, dashes
for the Kar98k and the shotguns, bright when loaded, dim when spent, spent from the left. It is the length
of the chain of bright marks ending at column ~72. A chain that runs off the crop's left edge (the
Kar98k's fifth dash, long box-gun magazines) is UNREADABLE.

Statuses: `0` ok, `1` absent (nothing drawn: the ammo counter fades out a few seconds after the last shot,
the round counter is blank between rounds, menus, black frames), `2` unreadable, `3` transition (round
only: the flash, count still valid). Values are -1 unless the status is ok or transition.

## Temporal checks (`hud/track.py`)

- **Points roll**: the counter counts toward a new value over a few frames (2070 → 2139 → 2164 → 2170),
  so a single step's change means nothing. A value *settles* when read on 3 ok steps; the change between
  settled values must be a multiple of 10 and a gain (≤ 2000), a spend no larger than what was there, the
  downed penalty (5%, rounded to 10 -- solo Nacht has no revive, so this is the death), the game-over
  count-up to the game's total score, or the reset to 500. Reads between two settled values must run
  monotonically from one to the other. Measured over all recordings: every settled change was one of
  these (gains 10, 20, 50-170, …; spends 200, 950, 1000, 1200, 1500).
- **Round** settles on 5 solid-red reads; it goes up by one, or back to 1 for a new game. A flash between
  two settles of 8 is round 9.
- **Grenades** go down by 1-2 and back up to ≤ 4 at a round start. **Reserve/mag**: one-step "blips"
  (a read that differs from both neighbours while they agree) are counted as misreads; magazine drops are
  cross-checked with the recorded fire button.
- `HudTracker` does the same step by step for a live loop (hold the last settled value, flag `hud_lost`
  after 3 implausible changes in a row). It is not wired into `play_real.py` yet.

## Accuracy

Measured on demo_0001, demo_0002 and runs/play/play_000{0,1,2}: 22,331 steps, 21,671 of them not marked
as menus or bad steps, which are what count. "Read" is the share of steps with an ok (or, for round,
transition) read; "of visible" leaves out steps where the field is not drawn at all; "consistent" is the
share of reads that pass the checks.

| field | read, of steps | read, of visible | consistent | misread evidence |
|---|---|---|---|---|
| points | 99.3% | 99.4% | 100% of 21,515 | 0 blips; 279 settled changes, 0 implausible |
| round | 96.8% | 98.7% | 100% of 20,976 | 0 blips; 13 round changes, 0 implausible |
| grenades | 81.5% | 98.4% | 100% of 17,657 | 0 blips; 8 changes, 0 implausible |
| reserve | 81.1% | 96.9% | 99.99% of 17,568 | 2 blips |
| magazine | 68.1% | 81.3% | 99.7% of 14,752 | 46 blips; 90.5% of 482 one-step drops have fire held |

Plus by eye: 76 random and deliberately hard steps (bright walls behind the digits, popups, low-ammo red)
rendered with their reads -- every ok read of points, round, grenades and reserve was right, and the
refusals were on the frames a person also squints at. The honest estimate: points and round are wrong on
well under 1 in 1,000 ok reads (none found); grenades and reserve about 1 in 10,000; the magazine is
approximate -- about 1 read in 300 is off by a few rounds for a step, on busy backgrounds.

What it does not see: rounds past 10 (numerals: no examples yet), round 9 as distinct from 8 in a single
frame, points behind a solid-white flash, the ammo counter over a wall as pale as its text (unreadable,
~3% of visible steps), the Kar98k's magazine.

Speed: 0.7 ms a step on one core for all fields (a 20-minute recording in ~13 s; ~1.1 ms with the
packed-video decode), p99 0.8 ms a step live -- inside the plan's 3 ms budget.

## The summary (`hud_summary.json`)

Per game in the clip (a clip can hold several, or start mid-game): rounds reached, the game-over score,
peak and final points, points gained and spent (from settled changes, so a lower bound when a gain and a
spend fall between two settles), downs, time survived, and how long each fully seen round took. Per clip:
`highest_round`, `best_score`, `peak_points`, `points_per_minute` (gains over played minutes),
`game_overs`, and the read/consistency rates above -- which is also the first thing to look at when a
number looks odd.

| clip | minutes | games | game overs | highest round | score | peak points | points/min |
|---|---|---|---|---|---|---|---|
| demo_0001 | 2 | 1 | 0 | 2 | – | 1,540 | 655 |
| demo_0002 | 20 | 2 | 2 | 9 (4, then 9) | 12,270 (4,050, then 12,270) | 4,910 | 814 |
| play_0000-2 | ~1 each | 1-2 | 0-1 | 1 | – | 500 | ~0 |

demo_0000 has no HUD crops (it predates them), and its framing puts the tallies almost entirely below
the frame, so even `--lowres` reads the round on only 16% of its steps: not a usable number.

## Round from 128x72 frames (`--lowres`)

For clips without crops, `hud/lowres.py` reads the round alone from the policy frames: at 128x72 a tally
stroke is under a pixel wide but still pure red, and the round box starts on a pixel boundary, so the
stroke masks block-averaged by 10 predict each count's red coverage. On demo_0001/0002 it agrees with the
full-resolution reader on every one of 16,217 steps both call readable (it reads ~80%), and finds the same
round changes except the ninth. No points or ammo.

## Rebuilding the atlas

Needed when the font or resolution changes, or to add glyphs (round numerals, once a recording reaches
round 11):

```
uv run --with pillow python scripts/build_hud_atlas.py harvest data/demos/demo_0001 data/demos/demo_0002 \
    runs/play --out /tmp/atlas
# look at /tmp/atlas/{points,ammo,tally}.png: row k is cluster k (its centre, then members);
# /tmp/atlas/<set>.txt has the current atlas's guess for each cluster
# edit configs/hud/atlas_labels.json: one character per cluster ('?' reject, '-' drop), and for the
# tally clusters how many strokes each shows
uv run python scripts/build_hud_atlas.py build /tmp/atlas configs/hud/atlas_labels.json
uv run pytest -q tests/test_hud_parse.py
```

Harvesting is deterministic (seeded k-means over the same clips and stride), so the committed labels
rebuild the committed atlas. Other 16:9 resolutions work by area-resizing crops to the 1440p reference;
only 1440p has been measured.
