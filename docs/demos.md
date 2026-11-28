# Learning from real gameplay

Everything here is about the same scarce resource: a person's time at the game. The agent gets ~54,000
decisions an hour from live play, and a weekend of it is a rounding error next to what RL usually assumes.
Recorded human play is the one source of data that is both *real* and *cheap to collect in bulk* — you play,
or you use footage you already have.

There are two routes in, and they do different jobs.

## Route A — record your own play, with input logging

`scripts/record_demo.py` captures the game's window at the decision rate and logs raw mouse counts
and key transitions on the same clock, then writes a clip whose every frame is paired with the action you
took while looking at it.

```sh
uv run python scripts/record_demo.py --counts-per-degree 6.4 --minutes 20 \
    --notes "camping the help room, deliberately bad positioning after round 8"

# On the Linux machine, with its standing settings (window, bindings, counts per degree, --wait, --audio):
scripts/record_waw.sh 20 "camping the help room, deliberately bad positioning after round 8"

# The same, on the fleet's own game (recommended since 2026-11):
scripts/record_lan.sh 20 "camping the help room, deliberately bad positioning after round 8"
```

**Record on Plutonium when you can.** The fleet trains on Plutonium T4's LAN co-op Nacht, which has last
stand, and plays with the fleet's config.cfg. Steam's solo game has no last stand. `scripts/record_lan.sh` starts
that game on your desktop (`scripts/desktop_game.py up`; `realgame/desktop.py` explains each choice) and
records it. The game runs offline in a loopback-only network namespace, with its own copies of Plutonium and
the prefix under `runs/desktop/` and its own sound sink. That sink is looped to your speakers, so the
recording's audio is the game alone. **At the main menu press F9 to start Nacht**, and again after each game
over: in LAN mode the menu cannot start a co-op game, so the launcher binds `map nazi_zombie_prototype` to F9
(the console, `~`, takes it too). The game stays open when a recording ends. `uv run python
scripts/desktop_game.py down` quits it.

### The live overlay, and watching a session back with your inputs

`record_waw.sh` and `record_lan.sh` put an overlay over the game while you record (`record_demo.py
--overlay`; `ZOMBIESAI_OVERLAY=0` leaves it off). It is `scripts/record_overlay.py`, a GTK 4 layer-shell
surface on the compositor's overlay layer. It sits above the fullscreen game, never takes focus, and lets
every click through. It has two panels:

- **Top centre, the recorder's state.** Waiting for the window, the countdown, then `REC 03:12 / 20:00
  PLAYING`. While F8 has the session marked as not playing, a wide amber **NOT PLAYING, press F8 when you're
  back in the game** banner stays up. A capture outage (the window not on screen) shows in red. At the end it
  shows the saved clip's name for five seconds, then closes.
- **Left, your inputs.** The keys around WASD, the mouse buttons and wheel, and a look meter showing which way
  the mouse is turning and how fast, in degrees a second. A key bound to a control lights amber and is labelled
  with that control. A key that is not a control (space, ctrl, c) lights grey: the game sees it, the labels do
  not. With `--pad`, the gamepad's sticks, triggers and buttons appear under it.

**It never reaches the training frames.** The recorder grabs the game's own XWayland window, and the overlay is
a Wayland surface the X server never sees. A screen recorder such as OBS does see it. The recorder publishes its
state over a Unix datagram socket (`viz/live_bus.py`, `$XDG_RUNTIME_DIR/zombiesai/record.sock`), which costs
the decision loop one non-blocking send per step whether or not the overlay is open. The overlay reads the keys
straight off the event devices, as the recorder does, so the display keeps up with your hand rather than with
the 15 Hz decisions. It runs under the system Python (`/usr/bin/python3`, which has GTK 4 and
gtk4-layer-shell; the venv has neither). To see it without recording:
`/usr/bin/python3 scripts/record_overlay.py --demo`. Its log is `$XDG_RUNTIME_DIR/zombiesai/record_overlay.log`.

To watch a recorded session back with the same input display burned in:

```sh
uv run python scripts/render_inputs.py data/demos/demo_0005                       # -> demo_0005/inputs.mp4
uv run python scripts/render_inputs.py data/demos/demo_0005 --from 300 --to 360 --out fight.mp4
```

The picture is the clip's 640x360 scene crop, scaled to 1280x720. Clips from before the scene crop use the
128x72 policy frames, blown up. The game's audio is included when it was recorded. Keys light at the input
log's own timestamps, so a tap shorter than a decision still shows. The look meter shows each decision's
mouse motion. The F8 not-playing stretches carry the same amber banner (`viz/input_video.py`). A 30 s
stretch renders in about 2 s.

### What to play: one strategy, many situations

Behavioural cloning learns the *average* of what you did in moments that look alike. If you sometimes go
left around a corner and sometimes right, it learns to hover in between -- the indecisive, twitchy turning
the first policies showed. So:

- **Keep one core strategy across sessions.** The same route between rooms, the same spot to camp or loop
  to train zombies, the same rules for when to open the help room and when to hit the box. Write it in
  `--notes` so the sessions that share it can be picked out later. A second strategy is fine once the first
  has hours behind it; mixed in early, it halves the data for both.
- **Vary the situations, not the plan.** Early and late rounds, whatever the box hands you, low ammo,
  getting cornered, a zombie behind you. Rare actions -- reloading, rebuilding, buying, the box, knifing,
  grenades -- should happen as often as they naturally do in good play; the policy cannot learn a button it
  saw twice.
- **Play at your normal level and recover from your mistakes.** A recovery is exactly the data the policy
  lacks when it gets itself into trouble. Don't play badly on purpose, and mark menus, pauses and the
  game-over screen with F8.
- **Quantity still matters most.** Twenty-minute sessions; the plan asks for 2-4 hours before BC is judged.

Every recording also carries the game's own settings -- sensitivity, `m_yaw`, resolution, key bindings --
read from WaW's `config.cfg` (`demos/game_settings.py`), and the recorder warns before you play when they
disagree with `--counts-per-degree`, or when mouse smoothing or toggle-ADS would spoil the labels. The game
writes that file on exit, so a setting changed mid-session shows up in the next recording's copy.

Keys pressed while Super is held are the desktop's, not the game's (Super+1 switches workspace; `1` is
weapon swap), so they and the mouse under Super never become labels. Closing the terminal or killing the
recorder stops it as cleanly as Ctrl-C: the clip is closed with its labels, and a clip that was killed
outright can still be re-quantized, because its start time is written before the first step.

Add `--wait` to launch it from a terminal elsewhere: it starts once the game window can be captured, after a
three-second countdown ([`linux.md`](./linux.md) says why). The pixels come from the game's XWayland window
and the input log from `/dev/input/event*`, which reports the mouse's own counts. The setup those two need -- group membership, a udev rule, and flat pointer acceleration -- is in
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
  event devices, which report the device's own relative counts — the same unit the agent's synthetic mouse
  emits. That symmetry is the reason a human's action means anything to the policy.
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

After a change to the spec's look bins, `scripts/requantize.py data/demos` re-labels every recording with the
counts per degree, bindings and mark key it was made with. The labels it replaces stay beside the new ones as
`labels.<old spec>.npz`. It did this for the half- and one-degree look bins (2026-11: yaw
±30/14/6/2/1/0.5/0, pitch ±6/2/1/0.5/0). Before them, the ~28% of decisions with a mouse move under a degree
were all labelled "no turn".

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
or a source's own declared input latency (the synthetic stand-in's, in the tests). A peak confidently elsewhere means the input log and the
capture are out of step in time — stop and fix it, because no amount of training absorbs a timing bug. The
**px per degree** of image shift, as a horizontal FOV, catches a wrong `--counts-per-degree`, which no
correlation can see: WaW reads 81–96° at 16:9, and a 2× error lands outside 62–110°. That one
is fixed with `requantize`, not by re-recording. The rank correlation itself only has to clear 0.3: on real
footage the optical-flow estimate locks onto fog, zombies and the gun often enough that 0.7 is a good
recording (demo_0000 scores 0.74, where the old Pearson threshold of 0.9 read 0.52 and cried wolf).

### Game audio (`--audio`)

`--audio` also keeps the sound, which a policy trained with `train_bc.py --audio` hears (PLAN.md, risk 11;
see "Hearing" under Behavioural cloning). A recording made without it can never be given it later.

```sh
uv run python scripts/record_demo.py --counts-per-degree 6.4 --minutes 20 --audio
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

**Play deliberately varied games.** Camping, trains, bad positioning, early deaths, running out of ammo. A
policy cloned from expert-only play has no idea what to do the moment it drifts off-distribution. The
DAgger loop that rescues it is the live player: take the controls while it plays and your fix is recorded as
training data (Route C below).

The recording path is tested without the game: `zombiesai/synthetic.py`'s `SyntheticSource` is a scripted
player in a stand-in world, read by the recorder exactly as a human at the game is, so CI checks the labels,
the pairing of frame and action, and the flow check on every commit. It is a test fixture, not training data:
nothing learned on its pixels transfers to World at War's.

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

The catch, and it is worth being blunt about it: **the IDM is only as good as the real recordings it was
trained on.** Route B needs an IDM trained on Route A recordings of the real game — half an hour of logged play is a reasonable start, and it
is spent far better there than on half an hour of demonstrations.

## Route C — correct the policy while it plays

`scripts/play_real.py` records every run to `runs/play/play_NNNN` (label_source `play`), and whenever you
take the controls back, your input is decoded exactly as a demo's and kept as the label of those steps
([`linux.md`](./linux.md), "Letting a policy play"). These corrections are HG-DAgger data: the human's action
in the states the *policy* reaches -- facing a wall, stuck on a barricade -- which no demo covers.
`labels.npz["actor"]` says who acted at each step (`clips.ACTOR_POLICY`, `ACTOR_HUMAN`, `ACTOR_HUMAN_IDLE`
for the wait before handing back, `ACTOR_NONE` when nobody could), and `Clip.usable()` on a play run keeps
only `ACTOR_HUMAN` steps -- so a play run can be passed to training like any demo directory. Their frame
history is the policy's steps before the takeover, on purpose. The raw input is in `inputs.jsonl` and each
step's poll instant in `labels.npz["t_mono"]`; `requantize` refuses play runs (their steps are not on
`t0 + k/15`). Runs recorded before corrections existed are recognised and offer nothing.

## What comes out

Both routes write to the clip store:

```
data/clips/<name>/
  frames.u8    (T, 72, 128, 3) uint8, decision rate, area-averaged
  clip.json    where the pixels came from, how they were cropped, spec stamps, label source
  labels.npz   actions (T, 8), per-step confidence, raw pre-quantization yaw/pitch degrees
  inputs.jsonl raw input log (Route A only)
  hud_<name>.u8  full-resolution HUD crops, (T, h, w, 3), same step index as frames.u8 (Route A only)
  hud_<name>/  the same crops as verified H.264 chunks, replacing the .u8 once the session has ended
  audio.flac   game audio, 48 kHz stereo (audio.s16 raw if not compressed), with --audio (Route A only)
  audio_index.bin  per-chunk (sample, monotonic time) map that aligns it to the steps
```

The policy's 128×72 frame turns the points and ammo counters into a smear, so screen recordings also keep
four HUD boxes at full resolution (`demos/hud_crops.py`): `points_ammo` in the bottom right (points
and their "+N" popups, weapon, grenades, magazine and reserve ammo), `round` in the bottom left, `prompt`,
the interaction line under the crosshair ("Press & hold F to buy ... [Cost: 1200]") -- at 128×72 the policy
sees that a prompt is up but not what it offers or costs -- and `powerup`, the countdowns timed power-ups
put below it ("Double Points: 23", "Insta-Kill: 7"). They are cut from the same grab as each frame,
area-downsampled by half -- digits stay about 12 px tall at 1440p -- and cost ~7.4 GB per 20 minutes. That
is what lets M4's parser (docs/hud.md; points, round, grenades and ammo so far) put points, ammo, round,
what is on offer and which power-ups are running on a recording after the fact. Recordings made before the prompt and power-up boxes existed have only the two
corners.

That size is only while recording. When the session ends, `record_demo.py` (and `play_real.py`) pack each
region as H.264 video (`demos/hud_video.py`): yuv444p so red keeps full colour resolution, CRF 12, in
30-second chunks. Every chunk is decoded back and compared with the raw crops, and the raw file is deleted
only when all of them are within a mean error of 3 and a 99th-percentile error of 16 levels in 255 -- they
measure about 1.3 and 6, which is invisible, digits included. About 20x smaller: ~0.4 GB a session instead
of ~7.4. `clip.hud(name)` then returns a `HudVideo` that indexes like the raw array and decodes on demand.
Packing takes about half a minute and never runs during the recording; Ctrl-C leaves the raw crops, and
`scripts/pack_hud.py data/demos` packs anything left raw, including recordings from before this existed.
`--keep-raw-hud` skips it. `--no-hud` turns them off.

Recordings since 2026-11 also keep a fifth region, `scene`: the whole picture at a quarter of the capture
(640×360 at 1440p, `hud_crops.RECORD_REGIONS`). At 128×72 a zombie's head ten metres off is a pixel or two,
and a clip that kept only that can never be given more. With `scene`, a sharper view for the policy (a crop
round the crosshair, a bigger frame) can be cut later from recordings already made. It costs ~12 GB per
20 minutes while recording and is packed as video with the HUD boxes when the session ends. `--no-scene`
skips it. No model reads it yet, and packing it has not been checked on real footage. The boxes are screen fractions placed on a 16:9 capture; another
aspect ratio needs new ones.

Frames and labels are versioned separately on purpose. A spec change that touches the HUD layout must not
invalidate a weekend of ingested video; one that moves the yaw bins must invalidate its labels. A clip whose
source knew its rewards can also carry per-step `mc_return`, `aux_dpoints` and `aux_damage` columns, the
value and auxiliary heads' targets, which video cannot supply.

## Behavioural cloning

```sh
uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1
uv run python scripts/train_bc.py data/demos runs/play --out runs/bc2        # plus your corrections
uv run python scripts/train_bc.py data/demos --val-clips data/demos/demo_0001 --out runs/bc3   # fixed held-out clip
uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos-heldout
uv run python scripts/play_real.py runs/bc1/bc.pt --minutes 3   # watch it play the real game
```

The policy sees a causal stack of 4 decision frames — nothing you could not see, in nothing but pixels — and
predicts the eight action heads. Details that are decisions rather than defaults:

- **Memory is a choice of which past frames to stack.** `--frame-offsets 0 1 2 4 8 16 30` feeds the network
  frames from those many steps back (two seconds in seven frames) instead of the last `--frame-stack`
  (`frame_stack=4` is exactly offsets 0-3, and checkpoints from before the option load unchanged). Every
  offset clamps at the step's segment start — a clip start, or play resuming after a pause, outage or marked
  menu — and the live `BCAgent` keeps just enough frames to reach the oldest offset, clamping the same way and
  forgetting them all on `reset()`, which the live loop calls exactly where its recording sets
  `FLAG_CLIP_START`; `tests/test_bc_memory.py` checks the two inputs are identical frame for frame. It is
  **off by default because it did not help yet**: on 32 minutes of play, strided histories (0,1,2,4,8,16,30
  and 0,1,2,3,8,15,30) lost 0.01–0.025 mean balanced accuracy to the plain 4-stack on every held-out split
  and seed, with lower training loss — more input to memorise, not more to learn from. A GRU prototype over
  per-step embeddings (hidden state carried and reset at segment starts) tied the 4-stack. Re-run the
  comparison when there are hours of data, not minutes.

- **Plain cross-entropy, sampled** (the default since 2026-11). `button` is `none` ~99% of the time, so the
  *argmax* of a cross-entropy policy never reloads. But nothing plays the argmax: the live agent and RL
  sample every head, and a sampled cross-entropy policy presses at the person's own rate. The checkpoint
  kept is the one with the lowest held-out NLL, which comes after 2-4 epochs on an hour of play. The old
  class-balanced focal loss is still there (`focal_gamma 1`, `class_balance 0.999`, what every checkpoint
  before 2026-11 used). It buys balanced argmax accuracy by flattening every head, which RL then has to
  undo (`--calibrate`, docs/rl.md).
- **A value head fitted to Monte-Carlo returns, plus auxiliary heads for "did I just score" and "am I being
  hit."** They cost nothing at BC time, they hand M7's RL run a critic that is already worth something, and
  they force the encoder to represent exactly what a value function needs. They train only on the clips whose
  source could supply the targets -- which human recordings cannot yet: that waits on rewards read off their
  HUD crops.
- **Prev-action conditioning is off by default.** It is the sufficient statistic for a delayed MDP and it
  belongs in the real environment — but in BC it is also the single strongest predictor of the label, and a
  policy that learns to copy its last action scores beautifully per frame and stands still in the game. Turn
  it on with `--prev-actions` and watch the copy rate.
- **Corrections weigh `--correction-weight` (default 2) times a demo step.** They are few and they are the
  only labels for the states the policy drifts into; 2 lets them count without a handful of hurried fixes
  outweighing hours of deliberate play. Play runs nobody took over in are skipped.
- **Augmentation is random shift ±4 px and brightness jitter, applied identically across a stack. Never
  horizontal flips** — a flip inverts the yaw label and mirrors the HUD, and Nacht is not mirror-symmetric.

### Hearing (`--audio`)

```sh
uv run python scripts/train_bc.py data/demos --out runs/bc_audio --audio
uv run python scripts/compare_audio.py --out runs/audio_compare --train data/demos/demo_0000 \
    data/demos/demo_0002:0:0.8 --val data/demos/demo_0002:0.8:1 data/demos/demo_0001 --seeds 0 1
```

`--audio` adds the `audio` key: for every step, a **stereo log-mel of the 0.5 s that ended at the step's
frame** (`demos/hearing.py`) — 25 frames of 21 ms Hann windows every 20 ms, 64 mel bands to 16 kHz, left and
right kept as separate channels (their difference is the only direction cue), dB floored at -100 and scaled
by fixed constants. Half a second rather than the plan's 200 ms because a growl, a plank being torn off or a
run of footsteps lasts about that long. A small conv encoder turns it into a 128-d embedding that joins the
pixel features in the mixer.

- **One feature function for training and play.** `features_at(audio, t)` is called per step on a recording
  (cached per clip under `~/.cache/zombiesai/audio_features`, `--audio-cache` to move it — never beside the
  clips) and per tick on the live ring buffer; a test checks the two agree to the bit on the same PCM. The
  feature config is saved in the checkpoint.
- **Clips without audio still train**, with their audio *masked*: the embedding is multiplied by a has-audio
  flag that also goes into the mixer. Excluding them would throw away vision data (demo_0000 is a third of
  the footage); padding them with silence would teach that silence means "nothing is coming". The mask also
  lets a hearing policy play deaf when the live stream dies.
- **Random ±6 dB gain** on training features, so the policy does not hinge on where the volume sat.
- **Causal and aligned.** Gunfire shows up 25-30 ms after the fire button in the recorded audio and never
  before it (demo_0001, demo_0002), i.e. in the step after the press.
- **Caveat: audio carries an echo of the player's own actions** — gunshots, reloads, footsteps — much as
  prev-actions do. `compare_audio.py` therefore also reports accuracy on *change steps* (the label differs
  from the step before) and the copy rate, next to a `masked` arm: the audio network with nothing to hear.
- Checkpoints from before audio load exactly as they were (same modules, same outputs).
- **First result: no gain yet.** Trained on demo_0000 + the first 80% of demo_0002, validated on the last 20%
  of demo_0002 + demo_0001 (5212 steps), 8 epochs, 2 seeds: mean balanced accuracy 0.413/0.419 (vision),
  0.408/0.411 (audio), 0.418/0.416 (masked); final val NLL 0.541/0.549, 0.592/0.599, 0.560/0.559. The audio
  arm reaches the lowest val NLL of any run early (0.516 at epoch 2-3, against 0.521 for vision) and then
  overfits fastest (training loss 0.49 against 0.63): ~15 minutes of audio is enough to memorise, not to
  generalise. More recorded audio, or audio dropout / a smaller encoder, before this is worth switching on.

## Reading the evaluation

Per-frame accuracy lies. A policy at 70% per-frame accuracy can be one that never pulls the trigger, because
"don't fire" is the majority label. `eval_bc.py` reports three things instead, on held-out recordings:

1. **Balanced per-head accuracy against the majority baseline.** If a head does not beat its baseline, it has
   learned nothing, whatever the raw number says.
   Training also logs each head's plain held-out cross-entropy (`val.nll` in `metrics.jsonl`, no class
   weights or focal term), which moves before the accuracies do and shows overfitting first.
2. **Behaviour statistics within 2× of the human's** — fire duty cycle, mean |yaw|/s, reload rate, how often a
   button is pressed at all. Measured from *sampled* actions on the human's own frames, because that is what
   the policy will do in the game; an argmax policy systematically under-fires.
3. **The action-inertia check** — its copy rate against the human's own action autocorrelation.

Rounds survived are the real game's to say: `play_real.py` plays a checkpoint, and an RL run's
`episodes.jsonl` records every game's round.

A few failures and what they usually mean:

| Symptom | Likely cause |
|---|---|
| `button_any_frac` far above the human's | a focal / class-weighted checkpoint sampled uncalibrated; or, with plain CE, overfitting (check `val.nll`) |
| every head at its majority baseline | not enough data, or `min_confidence` threw most of it away |
| yaw balanced accuracy near chance, raw accuracy high | it predicts "no turn" always; check the IDM's own yaw accuracy first |
| copy rate far above the human's | action inertia — train without `--prev-actions` |
| quality check says `misaligned` (yaw/flow peak at the wrong lag) | log and frames out of step in time; fix before anything else |
| quality check says `wrong_scale` (implied FOV outside 62–110°) | wrong `counts_per_degree`; `requantize` with the suggested value |

## What is not built yet

- **The HUD parse is not in the training data yet.** `scripts/parse_hud.py` reads points, round, grenades
  and ammo from a clip's HUD crops (docs/hud.md) into `hud.npz`, but BC does not consume it and there is
  no reward on real footage yet. Ingested video and recordings made before the crops existed have no
  crops; `--lowres` reads only the round from their frames.
- **CI never touches the game.** The capture path is tested against a real X server and the dispatcher's
  output round-trips through the same decoder a recording uses, but an X server in CI is not World at War
  under Proton. After a change to capture or input, the spikes (S1-S3) and a short recording's quality check
  are the real test.
- **The IDM is trained per-game, not per-project.** An IDM fit to your sensitivity and your bindings labels
  your footage. Someone else's video, at a different sensitivity, needs its own `counts_per_degree` at
  minimum — and, if the difference is large, its own IDM.

## Recording with a gamepad

`--pad` records with a controller instead of (or as well as) the mouse and keyboard. The pad does not go to
the game: `demos/gamepad.py` turns it into a virtual mouse and keyboard (`zombiesai-pad-bridge`), and the
recording logs exactly the counts and keys that device sent. The labels are therefore made the same way as
a mouse demo's, through the same `--bindings` and `--counts-per-degree`: no stick model, no second
calibration, no controller aim assist. The mouse and keyboard keep working beside it and are logged as usual.

```sh
uv run python -m zombiesai.demos.gamepad --watch         # which pad, and what each control is called
scripts/record_lan.sh 20 "pad, training in the courtyard" --pad
scripts/record_lan.sh 20 "pad" --pad /dev/input/event21 --pad-config configs/my_pad.json
```

Setup (once): the Hyprland flat-acceleration block for `zombiesai-pad-bridge`, and if the pad is a
PlayStation or Switch one, hidraw kept away from the game (docs/linux.md, steps 4 and 4b).

The default layout (`configs/pad_bindings.json`). Buttons are named by position: south is Xbox A /
PlayStation Cross, and so on. The xpad driver swaps north and west, so on an Xbox pad check with `--watch` and
swap the two lines if Y reloads.

| Pad | Sends | Control |
|---|---|---|
| left stick | W / A / S / D past half tilt | forward, back, left, right |
| right stick | mouse movement, up to 240°/s yaw and 80°/s pitch | look |
| RT / LT | mouse1 / mouse2 | fire / aim |
| L3 (click) | shift, held until you click again or stop pushing forward | sprint |
| R3, LB | v | melee |
| RB | g | grenade |
| west (X / Square) | r | reload |
| east (B / Circle) | f | use (buy, open, rebuild) |
| north (Y / Triangle) | 1 | swap |
| south (A / Cross) | space | jump (unlabelled: the policy cannot jump) |
| start / select | escape / tab | pause / scoreboard (unlabelled) |
| d-pad down | f8 | the mark key: not playing / playing again |

Tuning, all in the JSON: `look.deadzone` (radial), `look.exponent` (2 is gentle near the centre for fine aim,
1 is linear), `look.max_yaw_deg_s` and `look.max_pitch_deg_s` (real degrees a second at full tilt, whatever
the in-game sensitivity), `look.invert_pitch`, the stick and trigger press/release thresholds, and `toggles`.
Keep yaw under 450°/s and pitch under 90°/s: one decision at full tilt then still fits the widest look bin
(30° and 6°). The recorder warns at the start if not. Every recording stamps the pad, the axes it used and the
whole config into clip.json under `gamepad`, so a pad demo can always be told from a mouse one.

The pad's live state (sticks, triggers, buttons, the keys it holds) goes to the live recording overlay
(`--overlay`) on the record bus.

What it does not do: the policy still turns in per-decision mouse bins, so a pad demo teaches the same kind of
turns as a mouse one, only smoother and slower. Rumble is not forwarded, and the pad's own gyro is ignored.
