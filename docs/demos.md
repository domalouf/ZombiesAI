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
```

This is the only route that produces ground-truth labels, and its output is what trains the inverse dynamics
model that makes Route B possible. Two things decide whether the labels are worth anything:

- **`--counts-per-degree` is spike S4's number** — mouse counts per degree of yaw *at the sensitivity you
  play at*, with in-game smoothing and acceleration off. Wrong number, wrong look labels, every single step.
- **Raw counts, not cursor deltas.** In a mouse-look FPS the cursor is captured and re-centred, so cursor
  positions carry no information about how far you turned. `demos/win32_input.py` reads Windows Raw Input
  (`WM_INPUT`), which reports the device's own relative counts — the same unit the agent's synthetic mouse
  will emit. That symmetry is the reason a human's action means anything to the policy.

The raw log is stored next to the clip as `inputs.jsonl`, so a wrong sensitivity, a rebound key, or a change
to the spec's yaw bins costs a re-quantization, not another evening of play:

```python
from zombiesai.demos.recorder import requantize
from zombiesai.demos.inputs import InputConfig
requantize("data/demos/demo_0000", InputConfig(counts_per_degree=6.9))
```

`record_demo.py` checks the recording the moment it finishes, while the game is still open: overrun rate,
label confidence, what your hands did, and — the one that matters — whether your yaw labels correlate with
the direction the image actually moved, at which lag. The lag is the closed-loop delay measured from your own
recording, and it should match spike S3. Below about 0.9 correlation at every lag, the input log and the
capture are out of step in time: stop and fix it, because no amount of training absorbs a timing bug. (On
`--source sim` the number runs lower — the raycast view is flat-shaded, so there is less for the optical-flow
estimate to lock onto.)

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
```

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
| yaw/flow correlation below 0.9 at every lag | log and frames misaligned in time; fix before anything else |

## What is not built yet

- **No HUD parse.** Real clips carry no `hud` vector, so BC here is pixels-only and there is no reward on
  real footage. That arrives with M4's parser; the clip format leaves room for it.
- **`ScreenCapture` and `RawInputRecorder` have not been run against the game.** They are written against
  Desktop Duplication (via `dxcam` or `mss`) and Win32 Raw Input, and their pure parts are unit-tested on
  Linux, but the message loop, the device registration and the capture timing are exactly what spikes S1–S3
  exist to verify. Treat the first recording session as the spike.
- **The IDM is trained per-game, not per-project.** An IDM fit to your sensitivity and your bindings labels
  your footage. Someone else's video, at a different sensitivity, needs its own `counts_per_degree` at
  minimum — and, if the difference is large, its own IDM.
