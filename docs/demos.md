# Learning from real gameplay

Everything here is about the same scarce resource: a person's time at the game. The agent gets ~54,000
decisions an hour from live play, and a weekend of it is a rounding error next to what RL usually assumes.
Recorded human play is the one source of data that is both *real* and *cheap to collect in bulk* — you play,
or you use footage you already have.

There are two routes in, and they do different jobs.

## Route A — record your own play, with input logging

`scripts/record_demo.py --source screen` captures the screen at the decision rate and logs raw mouse counts
and key transitions on the same clock, then writes a clip whose every frame is paired with the action you
took while looking at it.

```sh
uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20 \
    --notes "camping the help room, deliberately bad positioning after round 8"

# On the Linux machine, with its standing settings (window, bindings, counts per degree, --wait, --audio):
scripts/record_waw.sh 20 "camping the help room, deliberately bad positioning after round 8"
```

Every recording also carries the game's own settings -- sensitivity, `m_yaw`, resolution, key bindings --
read from WaW's `config.cfg` (`demos/game_settings.py`), and the recorder warns before you play when they
disagree with `--counts-per-degree`, or when mouse smoothing or toggle-ADS would spoil the labels. The game
writes that file on exit, so a setting changed mid-session shows up in the next recording's copy.

Keys pressed while Super is held are the desktop's, not the game's (Super+1 switches workspace; `1` is
weapon swap), so they and the mouse under Super never become labels. Closing the terminal or killing the
recorder stops it as cleanly as Ctrl-C: the clip is closed with its labels, and a clip that was killed
outright can still be re-quantized, because its start time is written before the first step.

The command is the same on either OS. Add `--wait` to launch it from a terminal elsewhere: it starts once the
game window can be captured, after a three-second countdown ([`linux.md`](./linux.md) says why). On Linux the pixels come from the game's XWayland window and the input
log from `/dev/input/event*`, which reports the same device counts Windows Raw Input does. The Linux setup
those two need -- group membership, a udev rule, and flat pointer acceleration -- is in
[`linux.md`](./linux.md), along with the spike order and what to check when the engine ignores the virtual
mouse.

Switching workspace or minimising the game mid-recording does not end it: while the window cannot be grabbed
the recorder repeats the last good frame, flags those steps `bad_step` so training skips them, and prints a
line when capture is lost and when it is back. A window that comes back at a different size counts as still
gone (HUD crops cannot change shape mid-clip), and the recording stops cleanly once the window has been gone
for 30 s, or at once if it was destroyed.

This is the only route that produces ground-truth labels, and its output is what trains the inverse dynamics
model that makes Route B possible. Two things decide whether the labels are worth anything:

- **`--counts-per-degree` is spike S4's number** — mouse counts per degree of yaw *at the sensitivity you
  play at*, with in-game smoothing and acceleration off. Wrong number, wrong look labels, every single step.
  `scripts/calibrate_mouse.py` measures it for you by turning the view and watching the pixels move.
- **Raw counts, not cursor deltas.** In a mouse-look FPS the cursor is captured and re-centred, so cursor
  positions carry no information about how far you turned. `demos/evdev_input.py` reads the kernel's
  event devices and `demos/win32_input.py` reads Windows Raw Input
  (`WM_INPUT`), which reports the device's own relative counts — the same unit the agent's synthetic mouse
  will emit. That symmetry is the reason a human's action means anything to the policy.
- **The wheel is a button.** WaW cycles weapons on the mouse wheel, so each notch is logged as a tap of
  `wheelup`/`wheeldown`, and the default bindings (and `configs/waw_bindings.json`) map both to `swap`.
  Several notches inside one decision are still one `swap` label.

The raw log is stored next to the clip as `inputs.jsonl`, so a wrong sensitivity, a rebound key, or a change
to the spec's yaw bins costs a re-quantization, not another evening of play:

```python
from zombiesai.demos.recorder import requantize
from zombiesai.demos.inputs import InputConfig
requantize("data/demos/demo_0000", InputConfig(counts_per_degree=6.9))
```

**Tap F8 when you stop playing, and again when you start.** Menus, the pause screen, loading, the game-over
card, alt-tabbing out: tap the mark key (`--mark-key`, F8 by default, which World at War leaves unbound) on
the way in and on the way out, and the terminal says `NOT PLAYING` / `playing again`. A step is marked if any
part of its input window was not play -- so the step holding each press is excluded, and play resumes on the
step after the second one. Marked steps stay in the clip (flag `FLAG_NOT_PLAYING`) but never reach training,
and frame stacks don't reach back across them. The key presses are in `inputs.jsonl` like any other, so
`requantize` re-derives the marking, under a different key if you pass `InputConfig(mark_key=...)`.

`record_demo.py` checks the recording the moment it finishes, while the game is still open: overrun rate,
time marked not playing,
label confidence, what your hands did, and — the one that matters — whether your yaw labels track the
direction the image actually moved (`yaw_flow_agreement`, criteria in `FlowCheck`). It reports three
things. The **lag** at which yaw and image motion agree best (Spearman rank correlation over lags −3…+3) is
the closed-loop delay measured from your own recording: 0 on the real game (demo_0000 peaks sharply there),
the sim's own input latency on `--source sim`. A peak confidently elsewhere means the input log and the
capture are out of step in time — stop and fix it, because no amount of training absorbs a timing bug. The
**px per degree** of image shift, as a horizontal FOV, catches a wrong `--counts-per-degree`, which no
correlation can see: WaW reads 81–96° at 16:9, the sim 80°, and a 2× error lands outside 62–110°. That one
is fixed with `requantize`, not by re-recording. The rank correlation itself only has to clear 0.3: on real
footage the optical-flow estimate locks onto fog, zombies and the gun often enough that 0.7 is a good
recording (demo_0000 scores 0.74, where the old Pearson threshold of 0.9 read 0.52 and cried wolf).

### Game audio (`--audio`, Linux)

`--audio` also keeps the sound, for the audio features the plan reserves the `audio` observation key for
(PLAN.md, risk 11). The policy does not hear it yet; the point is that a recording made without it can never
be given it later.

```sh
uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20 --audio
```

- **What is recorded:** the default sink's *monitor* through `parec` (PipeWire via pipewire-pulse, or
  PulseAudio), s16le 48 kHz stereo. **Everything playing through that output is recorded, not only the
  game** -- mute music, videos and voice chat, or point `--audio-device` at another `<sink>.monitor`
  (`pactl list short sources`). Only a `.monitor` is ever opened; a microphone name is refused. Capturing
  the game alone would mean one `parec --monitor-stream=<sink-input>` per stream the game opens (World at
  War under Proton opens four) and a mix, or routing it to its own null sink -- not done.
- **Alignment:** every chunk read off the pipe (10 ms) is stamped with `time.monotonic()` -- the clock of
  `t0_mono` and the input log -- and indexed in `audio_index.bin`. Loading replaces each chunk's clock
  offset with the promptest chunk's in the next second, which strips read jitter (measured here: 0-5 ms,
  median 2 ms) and follows crystal drift and dropouts. What no timestamp can see is the audio graph's own
  buffering before the pipe; it is *estimated* at 5 ms (one 256-sample PipeWire quantum), not measured, and
  sits in `clip.json["audio"]["latency_s"]` so a better number fixes old recordings on load. The game's own
  output buffering under Wine is unmeasured too and plays the same role as display latency on the frames;
  expect the pair to agree within ~20 ms, a third of a decision.
- **Files:** `audio.s16` raw while recording (11.5 MB/min, append-only, so a crash keeps everything up to
  the last read), replaced by lossless `audio.flac` after a clean stop when `ffmpeg` is installed
  (`--audio-raw` skips that). `clip.json["audio"]` holds rate, channels, device, latency and chunk stats.
- **Reading it:** `clip.audio_for_step(k, window_s=0.2)` is the 200 ms that had played by the time step k's
  frame was grabbed -- causal, like the frame -- as `(9600, 2)` int16, silence where nothing was captured.
  `clip.audio()` gives the whole stream with `sample_at(t)` / `time_of(sample)` on the monotonic clock.
- **Windows:** not implemented. `demos/audio.py` takes any stream with `read() -> (bytes, t_mono)`, so WASAPI
  loopback slots in as one more class.

**Play deliberately varied games.** Camping, trains, bad positioning, early deaths, running out of ammo. A
policy cloned from expert-only play has no idea what to do the moment it drifts off-distribution, and there
is no DAgger loop here to rescue it.

**On Linux, `--source sim` records NachtSim instead**, with a scripted agent at the controls, writing exactly
the same format. It is how the recording path stays testable without a Windows box, and how you can have
labelled clips before you have recorded anything real. It is not a substitute for real footage: the sim's
raycast view is a crude stand-in, and visual sim-to-real transfer is a non-goal (`sim_lies.md`).

## Route B — video nobody logged input for

Any recording of the game works: OBS, ShadowPlay, a file from years ago.

```sh
uv run python scripts/ingest_video.py ~/Videos/waw/*.mp4 --out data/clips/session1
uv run python scripts/train_idm.py data/demos --out runs/idm1          # needs Route A data
uv run python scripts/label_clips.py runs/idm1/idm.pt data/clips/session1 --min-confidence 0.5
uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1
```

Ingest decodes to 15 Hz at 128×72 with area averaging, removes letterbox bars, and splits the result into
continuous runs — menus, loading screens, paused captures and hard cuts are dropped rather than glued
together into transitions that never happened.

Then the **inverse dynamics model** (Baker et al., *Video PreTraining*, 2022) fills in the missing actions.
It is allowed to see the frames on *both* sides of a decision, which makes its job far easier than the
policy's: a turn to the right is visible as the scene sliding left, and a reload is visible as the animation
that follows it. So a small amount of labelled play buys labels for an unbounded amount of unlabelled video.

The catch, and it is worth being blunt about it: **an IDM trained on NachtSim renders will not label real
World at War footage.** The sim's view is untextured flat shading with a made-up font. Route B needs an IDM
trained on Route A recordings of the real game — half an hour of logged play is a reasonable start, and it
is spent far better there than on half an hour of demonstrations.

## What comes out

Both routes write to the clip store:

```
data/clips/<name>/
  frames.u8    (T, 72, 128, 3) uint8, decision rate, area-averaged
  clip.json    where the pixels came from, how they were cropped, spec stamps, label source
  labels.npz   actions (T, 8), per-step confidence, raw pre-quantization yaw/pitch degrees
  inputs.jsonl raw input log (Route A only)
  hud_<name>.u8  full-resolution HUD crops, (T, h, w, 3), same step index as frames.u8 (Route A only)
  audio.flac   game audio, 48 kHz stereo (audio.s16 raw if not compressed), with --audio (Route A only)
  audio_index.bin  per-chunk (sample, monotonic time) map that aligns it to the steps
```

The policy's 128×72 frame turns the points and ammo counters into a smear, so screen recordings also keep
the two HUD corners at full resolution (`demos/hud_crops.py`): `points_ammo` in the bottom right (points
and their "+N" popups, weapon, grenades, magazine and reserve ammo) and `round` in the bottom left. They are
cut from the same grab as each frame, area-downsampled by half -- digits stay about 12 px tall at 1440p --
and cost ~2.6 GB per 20 minutes. That is what lets M4's parser put points, ammo and round on a recording
after the fact. `--no-hud` turns them off. The boxes are screen fractions placed on a 16:9 capture; another
aspect ratio needs new ones.

Frames and labels are versioned separately on purpose. A spec change that touches the HUD layout must not
invalidate a weekend of ingested video; one that moves the yaw bins must invalidate its labels. Recorded
*episodes* (`store/episode_store.py`) read as clips too, via `clips.clip_from_episode`, so a BC run can mix
human video with sim episodes — and the episodes bring Monte-Carlo returns and auxiliary targets the video
cannot.

## Behavioural cloning

```sh
uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1
uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos-heldout --episodes 20
uv run python scripts/watch.py --checkpoint runs/bc1/bc.pt      # watch it play NachtSim
```

The policy sees a causal stack of 4 decision frames — nothing you could not see, in nothing but pixels — and
predicts the eight action heads. Details that are decisions rather than defaults:

- **Class-balanced, focal cross-entropy.** `button` is `none` ~95% of the time; plain cross-entropy answers
  that by never reloading. `class_balance_power` controls how hard the correction is; the default softens it
  to square-root inverse frequency, because full correction overshoots on the look heads and produces a
  policy that turns constantly.
- **A value head fitted to Monte-Carlo returns, plus auxiliary heads for "did I just score" and "am I being
  hit."** They cost nothing at BC time, they hand M7's RL run a critic that is already worth something, and
  they force the encoder to represent exactly what a value function needs. They train only on the clips whose
  source could supply the targets.
- **Prev-action conditioning is off by default.** It is the sufficient statistic for a delayed MDP and it
  belongs in the real environment — but in BC it is also the single strongest predictor of the label, and a
  policy that learns to copy its last action scores beautifully per frame and stands still in the game. Turn
  it on with `--prev-actions` and watch the copy rate.
- **Augmentation is random shift ±4 px and brightness jitter, applied identically across a stack. Never
  horizontal flips** — a flip inverts the yaw label and mirrors the HUD, and Nacht is not mirror-symmetric.

## Reading the evaluation

Per-frame accuracy lies. A policy at 70% per-frame accuracy can be one that never pulls the trigger, because
"don't fire" is the majority label. `eval_bc.py` reports four things instead:

1. **Balanced per-head accuracy against the majority baseline.** If a head does not beat its baseline, it has
   learned nothing, whatever the raw number says.
2. **Rollout statistics within 2× of the human's** — fire duty cycle, mean |yaw|/s, reload rate, how often a
   button is pressed at all. Measured from *sampled* actions, because that is what the policy will do in the
   game; an argmax policy systematically under-fires.
3. **The action-inertia check** — its copy rate against the human's own action autocorrelation.
4. **Rounds survived in NachtSim**, which is a smoke test and not a score.

A few failures and what they usually mean:

| Symptom | Likely cause |
|---|---|
| `button_any_frac` far above the human's | class weighting overshooting — lower `class_balance_power`, or sample below temperature 1 |
| every head at its majority baseline | not enough data, or `min_confidence` threw most of it away |
| yaw balanced accuracy near chance, raw accuracy high | it predicts "no turn" always; check the IDM's own yaw accuracy first |
| copy rate far above the human's | action inertia — train without `--prev-actions` |
| quality check says `misaligned` (yaw/flow peak at the wrong lag) | log and frames out of step in time; fix before anything else |
| quality check says `wrong_scale` (implied FOV outside 62–110°) | wrong `counts_per_degree`; `requantize` with the suggested value |

## What is not built yet

- **No HUD parse.** Real clips carry no `hud` vector yet, so BC here is pixels-only and there is no reward on
  real footage. Screen recordings keep full-resolution HUD crops for M4's parser to read later; ingested
  video and recordings made before the crops existed have none.
- **Nothing here has been run against the game.** The Linux capture path is tested against a real X
  server and the dispatcher's output round-trips through the same decoder a recording uses, but an X
  server in CI is not World at War under Proton, and `win32_input.py` has never run on Windows at all.
  Spikes S1-S3 exist to check exactly this; treat the first session as one.
- **The IDM is trained per-game, not per-project.** An IDM fit to your sensitivity and your bindings labels
  your footage. Someone else's video, at a different sensitivity, needs its own `counts_per_degree` at
  minimum — and, if the difference is large, its own IDM.
