# Nacht der Untoten RL Agent — Project Plan

## Context

Build a reinforcement-learning agent that learns to play *Call of Duty: World at
War* Nazi Zombies on Nacht der Untoten, trained on a personal Windows gaming PC
(RTX 5070, 12 GB).

Decisions already made:

- **Learning RL is the point.** Final bot skill is secondary to a clean
  environment, fast iteration, good logging, and swappable algorithms.
- **Screen capture only.** No mod tools, no GSC scripting, no process-memory
  reading. Pixels in, keyboard/mouse out; reward and episode boundaries read off
  the HUD.
- **Strong ML/PyTorch, new to RL.** PPO and DQN implemented from scratch as the
  learning exercise; the harder sample-efficient algorithm is borrowed.
- **Multi-day unattended runs available**, so real-game training is viable.
- **Its own repo** (`ZombiesAI`), matching the one-repo-per-project pattern.

Worth knowing: **there is no public RL agent for CoD Zombies.** A search across
GitHub and the literature turned up scripted GSC bots and nothing else.

### Three facts that drive every decision

1. **The real game cannot be parallelized — checked, not assumed.** WaW's
   DirectInput is foreground-only, compiled into the 2008 binary: one window
   holds input focus per desktop session, period. Every way around that on one
   consumer GPU was researched and ruled out — RDP sessions fall back to
   software rendering (Microsoft killed the GPU-accelerated path, RemoteFX, in
   2021), multi-seat software needs one GPU per seat, GPU partitioning needs
   enterprise hardware or a Windows Server host. A real technique exists
   (DLL-injection focus-faking — the class of tool behind LAN-party split-screen
   mods, with purpose-built support for DirectInput's background-window
   handling) but nobody has finished it for WaW specifically, and building it is
   a multi-week systems project orthogonal to learning RL. Not undertaken here.
   Net result: **~54,000 steps/hour** from the one real instance, where an Atari
   DQN baseline expects 10,000,000; a 48-hour weekend yields ~2.6M steps, one
   Atari-scale run if nothing crashes. **The simulator has no such ceiling** —
   pure CPU, no window, no input-focus problem — parallelized to every core the
   machine has (M1), it's the actual source of data volume. Trade compute for
   sample efficiency everywhere.
2. **The reward signal is a computer-vision artifact.** If the HUD parser is
   wrong, RL optimizes noise and you won't notice for days. Perception
   reliability is a first-class subsystem, not a utility function.
3. **~85% of the code can be written and tested on Linux with no game.** The
   Windows-only surface must be squeezed into thin, fakeable adapters so it
   can't hold the rest of the project hostage.

The closest published analogue is
[Counter-Strike Deathmatch with Large-Scale Behavioural Cloning](https://arxiv.org/abs/2104.04258)
(Pearce & Zhu, IEEE CoG 2022, Best Paper). Same genre, same constraints; they
state the blocker outright — real-time-only data generation **"precludes many
reinforcement learning algorithms."** Their answer was behavioral cloning on
5.5M frames, reaching casual-human play from pixels. **Demonstrations first.**

## The honest cost of the screen-capture-only constraint

Stated once, then dropped. Low-dimensional state observations are far more
sample-efficient than pixels — model-free pixel methods routinely need tens of
millions of steps where state-based methods need 10⁵–10⁶. At 54k steps/hour,
"tens of millions" is months.

Two things that exist, should you ever relax the constraint: WaW has **no ASLR**,
so published static addresses ([e7ite/WaWDll](https://github.com/e7ite/WaWDll))
read game state with no pointer chasing; and
[`t4sp_bot_warfare`](https://github.com/JezuzLizard/t4sp_bot_warfare) proves a bot
can be driven by engine builtins in WaW zombies. Neither is in scope.

**The mitigation that stays inside the constraint:** build a state vector *from*
pixels. Tier 1 — HUD scalars via template matching (M4). Tier 2, optional — a
small zombie detector trained on a few hundred hand-labeled frames, emitting
egocentric (bearing, apparent size) tuples. Pure computer vision, no game
modification, and likely the highest-leverage optional upgrade here.

## Architecture

One spec, several backends. Agent code never knows which it's talking to — that's
what lets you develop RL on Linux at thousands of steps/sec and deploy identical
code to Windows at 15.

```
              ┌───────────────────────────────────────┐
              │   spec.py — versioned, hashed contract │
              └──┬──────────────┬─────────────────┬───┘
                 │              │                 │
     ┌───────────▼──────┐ ┌─────▼──────────┐ ┌────▼──────────────┐
     │  NachtSim        │ │ ViZDoom ASYNC  │ │  RealGameEnv      │
     │  exact constants │ │ real-time      │ │  ~15 steps/sec    │
     │  from game source│ │ rehearsal      │ │  capture → HUD    │
     │  ≥5000 steps/s   │ │ free resets    │ │                   │
     └──────────────────┘ └────────────────┘ └───────────────────┘
          Linux, fully testable here        Windows-only adapters
```

**`spec.py` is the contract and it comes first.** It holds the observation and
action spaces, the compact-action list, and the HUD field order — plus
`SPEC_VERSION`, a hash over all of them, written into every checkpoint and every
episode directory. Loading mismatched data is a hard error. This prevents the
nastiest greenfield bug class: silently training on data whose action indices
moved.

Two surrogates, de-risking different things:

- **`NachtSim`** — the *strategy* problem: economy, kiting, round progression.
  Cheap to build faithfully because the constants are known (below), and it can
  **start at any round**, which the real game cannot.
- **ViZDoom `ASYNC_PLAYER`** — the *systems* problem. Fixed 35 FPS, drops your
  frames if you think too slowly: exactly WaW's failure mode, but free and
  instantly resettable.

([NZ:P](https://github.com/nzp-team/nzportable), a GPLv2 CoD-Zombies demake on a
Quake engine with QuakeC logic and a Nacht remake that builds on Linux, would be
a superb surrogate with a ViZDoom-style step API added — filed as a stretch
option, not the plan.)

### Ground-truth mechanics

From `raw/maps/_zombiemode_prototype.gsc` in the WaW mod tools — the script Nacht
actually runs (**not** `_zombiemode.gsc`, the later Der Riese-era fork; most
community guides get this wrong).

| Mechanic | Value |
|---|---|
| Zombie health | R1 = 150, **+100/round** through R9 (950), then **×1.10 compounding** from R10 |
| Zombies per round, **solo** | R1 = 4, R2 = 9, R3 = 14, R4 = 19, **R5+ = 24 (flat cap forever)** |
| Movement speed | `zombie_move_speed = round_number * 8` |
| Spawn pacing | 3 s between spawns; hard gate `while(enemy_count > 31)` |
| Starting points | 500 |
| Scoring | non-lethal hit 5 · kill 50 · +10 torso · +50 head · +80 melee |
| Penalties | downed −5% of points, no-revive −10% |
| Doors/debris | 2 doors + 1 debris, **1000 each** |
| Mystery box | 950 |
| Wall weapons | Kar98k 200, M1A1 Carbine 600 (start room); Double-Barreled 1000, Thompson 1200 (Help room) |

The solo 24-zombie cap is the most important fact for strategy: Nacht solo never
gets *denser*, only tankier and faster. That's what makes training the horde
viable indefinitely, and what the agent must eventually discover.

⚠ **Do not hardcode the scoring table into the reward function.** The source says
`zombie_score_damage 5` while community lore says 10 per hit, and per-map wall
prices live in entity keyvalues in `map_source/nazi_zombie_prototype.map`, not
the shared weapon table. Instead, `tools/fit_delta_alphabet.py` histograms
`Δpoints` over 30 minutes of recorded play and extracts the modes. **The
whitelist is measured, not assumed.** Geometry lives in
`configs/env/nacht_geometry.yaml`, calibrated from a map screenshot.

## The environment

### Observation

| key | shape | notes |
|---|---|---|
| `pixels` | `(72, 128, 3)` uint8 | current frame; stacking is a wrapper |
| `hud` | `(16,)` float32 | parsed scalars, fixed order |
| `prev_actions` | `(H·ACT_ENC_DIM,)` | H = 2 |
| `state` | `(S,)` | sim state-vector mode only |
| `audio` | `(2, 64)` | **reserved for v2** — see Risks |

**128×72 RGB, not 84×84 grayscale.** Three reasons, and the third is decisive:
preserving 16:9 keeps the peripheral horizontal FOV that matters in a 360° threat
environment; the hint prompt is amber; and **the damage vignette is red —
grayscale would destroy the agent's only health signal.** Downsample with area
averaging, not nearest: at ~15× reduction, nearest-neighbour aliases thin zombie
limbs in and out between frames, injecting noise that looks exactly like motion.

Don't crop the HUD out of the pixels — the floating "+10" popups and hit markers
are useful cues, and cropping creates a needless sim/real mismatch. Store uint8;
divide by 255 on the GPU. Stack 4 *decision* frames (267 ms) so approach velocity
is inferable.

**The `hud` vector must include a damage proxy** (normalized red-ring intensity)
and time-since-last-damage. WaW has no health bar, so **without these the value
function literally cannot represent "I am about to die,"** which makes the death
penalty pure variance. Also include `hud_confidence` so the policy can learn to
distrust bad parses.

Also worth parsing: WaW's centre-bottom prompt ("Press and hold [F] to buy X
[1200]"). Detect its presence and class by a perceptual hash of the prompt region
plus a digit read of the bracketed price — you don't need the weapon name. It
gives you an affordance signal, an action mask for `use`, and the classifier that
makes spend-aware reward shaping work.

### Action — factored canonical, compact projection for value methods

```python
FACTORED = MultiDiscrete([3, 3, 9, 5, 2, 2, 2, 6])
#  strafe, forward, yaw bin, pitch bin, fire, ads, sprint, button
YAW_BINS_DEG   = [-30, -14, -6, -2, 0, +2, +6, +14, +30]
PITCH_BINS_DEG = [-6, -2, 0, +2, +6]
```

32 logits; joint cardinality 29,160. No jump, no crouch in v1 — neither is
load-bearing on Nacht and each doubles the joint space.

**Why factored:** the scarce resource is experienced transitions, and
factorization turns one transition into eight supervised signals (BC) or eight
policy-gradient terms (PPO) instead of one draw from a 29k-way categorical. The
cost is that within-step head correlations aren't representable — acceptable,
because at 15 Hz temporal correlation matters far more.

**Why also a compact profile:** DQN needs `max_a Q(s,a)`, which doesn't decompose
over factored heads without a branching architecture. Rather than complicate the
from-scratch DQN — whose whole point is learning RL clearly — define ~48 curated
joint actions with `project_to_compact` / `expand_compact`, hashed into
`SPEC_VERSION`. Everything works in factored space; a wrapper exposes
`Discrete(48)` when needed.

`fire`, `ads`, `sprint` are **hold states, not edges** — the dispatcher diffs
requested against currently-held state. This is precisely why `prev_actions` must
be observable: `fire=1` means different things depending on whether it was
already held.

**Mouse aiming: discrete signed yaw/pitch bins specified in degrees, dispatched
as 3 sub-moves across the tick.** Old DirectX 9 raw-input paths clamp or drop
large single deltas, and smooth motion gives a better-behaved counts→degrees
relationship. An M0 spike fits `counts_per_degree` once; both the real env and
the sim consume it, so **the same action index means the same rotation in both
backends** — which is what lets sim intuitions and BC labels transfer.

Rejected: a continuous Gaussian head (entropy collapse and per-dim std tuning are
exactly the pathologies a first-time RL implementer shouldn't debug alongside
everything else, and it forecloses DQN); and absolute "look at pixel (x,y)"
(needs a stable screen→counts mapping you don't have, and the target moves during
your 66 ms of latency).

### Real-time loop

Fixed 15 Hz, action repeat 4 at 60 fps. Budget per tick: capture ≤8 ms,
downsample+crops ≤3, HUD parse ≤3, inference ≤10, dispatch ≤1, ~40 ms slack.
Absolute monotonic deadlines `t_k = t_0 + k·T`, never `sleep(T)`.

**Overrun policy: skip, hold, drop — never catch up.** Leave held input as-is (an
overrun degrades gracefully to a longer action repeat), record true `dt`, and if
`dt > 1.5·T` flag the transition `bad_step` and exclude it from training. Do
*not* attempt variable-γ correction; keep the MDP clean and pay a little data
loss. Never run two decisions back-to-back to catch up — that produces a
temporally warped MDP and silently costs a weekend.

**Delay: make it observable, don't compensate.** Closed-loop delay will be
60–120 ms (1–2 decisions). For constant delay `d`, `(s_{t-d}, a_{t-d}, …,
a_{t-1})` is Markov — so `prev_actions` isn't a heuristic, it's the sufficient
statistic. Measure `d` in M0, reproduce it as the sim's latency knob, and confirm
the algorithm can still learn to aim there *before* spending a real weekend. No
learned forward model in v1.

The known downside of prev-action conditioning is **action inertia** — a
self-imitation loop where the policy copies its last action because that's the
strongest predictor in BC data. Fix it with entropy bonuses and by monitoring
"copy rate" against the human's own action autocorrelation. Don't fix it by
dropping prev-action.

Use [`rtgym`](https://github.com/yannbouteiller/rtgym) (built for Delayed MDPs;
[TMRL](https://github.com/trackmania-rl/tmrl) is built on it and is the closest
existing project to yours) rather than reinventing the elastic-timestep logic.
And do **not** reuse Gym's `AtariPreprocessing` — its max-pool-over-two-frames
semantics assume a pausable emulator.

### Actor/learner split

Two processes, one machine, Windows `spawn` only.

- **Transitions, actor → learner:** a single-producer/single-consumer
  shared-memory ring, struct-of-arrays, preallocated, no queue, no lock. Actor
  writes slot `k`, then publishes `cursor = k+1` last. If the learner falls
  behind nothing backs up — it just samples slightly staler data.
- **Frames stored once, not stacked.** Stacking is index arithmetic at sample
  time, clamped at episode boundaries. Stacking at write time is 4× storage for
  zero information. At 27.6 KB/frame, a weekend is ~72 GB — memmap the frame
  array on NVMe (~1.5M capacity), keep the small arrays in RAM.
- **Weights, learner → actor:** a double-buffered shm parameter vector with a
  version counter. The actor checks it once per ~15 ticks, copies only if the
  remaining tick budget exceeds 15 ms, and **never waits**. Policy lag of ~1 s is
  irrelevant here.
- **Control/telemetry:** a tiny queue, `put_nowait`/`get_nowait` only, never on
  the hot path.

**The governor is not optional.** A free-running learner will do 100k gradient
steps on the first 1,000 transitions and destroy the network before real data
arrives. `governor.py` reads the actor's write cursor and throttles the learner
to `RR × (env steps produced) − grace`. Thirty lines; essential.

### The episode store — the highest-leverage decision in the plan

The actor writes to disk *in addition to* the ring:

```
runs/<run_id>/ep_<n>/
  frames.u8      # 72×128×3 policy frames @15 Hz      27 KB/step
  hud_crops.u8   # raw HUD field crops @15 Hz         ~6 KB/step
  fullres.mp4    # 1–3 Hz debug video
  meta.npz       # actions, rewards, parsed HUD, timestamps, flags
  events.jsonl   # round boundaries, purchases, watchdog incidents, parse rejects
  spec.json      # SPEC_VERSION, git SHA, resolved config
```

**Storing the raw HUD crops at 15 Hz costs ~15 GB/weekend and means you can
rewrite the parser, re-run it over every frame you have ever collected, and
recompute the rewards for the entire replay buffer without touching the game.**
When — not if — the parser has a bug, this is the difference between re-running a
script and losing the weekend.

The demo recorder writes this identical format with `is_demo=True`, so BC
loading, replay seeding, offline eval, and video review are all one code path.

### Episodes and resets

Episodes are 4,500–18,000 steps. `terminated` only on a confirmed `GAME_OVER`
screen; `truncated` (with `V(s_T)` bootstrap — a classic silent bug, get it
right) on step caps, watchdog intervention, or `hud_lost`. Round transitions are
**not** episode boundaries; they set a flag.

**Headline metric: `rounds_survived`, read from the game-over screen's own
digits.** The game produces this number, not your reward function, so it cannot
be reward-hacked — and it's a free per-episode cross-check on your round counter.

Per-round logging is where learning curves actually come from: 40 rounds per
episode gives 40 data points per 20 minutes instead of 1. Log round duration,
points, kills, hits-without-kills, headshot rate, shots fired, reloads, barrier
repairs, purchases, damage events, and the per-term reward decomposition.

Reset is an explicit FSM driven by the screen-state classifier with per-state
timeouts — **never blind `sleep()`** — escalating to the watchdog after N
failures. For episode diversity, given the real game stays single-instance by
design, not just by default: treat the *round* as the unit of diversity (free); add a ~2 s **in-episode scramble** at round
boundaries (random yaw sweep + walk, flagged `bad_step`) to decorrelate state at
40× less cost than a reset; vary exploration temperature *within* an episode; and
add a `suicide_after_round_k` option so you can get dozens of resets per hour
while debugging the reset FSM.

## Perception

**Template matching, not Tesseract.** Tesseract is 10–50 ms per call against a
66 ms total tick budget — that alone decides it. Beyond latency: the font is a
fixed bitmap, at fixed positions, with exactly 10 glyphs, so generic OCR solves a
much harder problem badly; you need per-digit confidences for the sanity filters;
and it's pure numpy, so it runs deterministically in CI on Linux. Tesseract would
fail exactly where you need it most — muzzle flash, explosions, the round-change
flash — and fail *silently with plausible output*.

Pipeline per field: crop → HSV threshold → binarize → column-projection glyph
segmentation → normalized SAD against 10 masks → right-aligned assembly.

**The sanity filters are where reliability actually comes from.** A stateful
`HudTracker` that may *reject* readings and hold the previous value:

- **round**: monotone, +1 only, must be 1 after reset.
- **points**: accept Δ only if it's in the empirically-fitted delta alphabet
  (≤3 gain atoms under a per-tick cap, or a measured price). 3 consecutive
  suspects → `hud_lost` → truncate.
- **ammo**: mag decreases by a plausible shot count for `dt` or jumps to
  `mag_size` on reload; reserve decreases only by `mag_size − mag_before`. This
  makes ammo self-validating **and** gives a free `shots_fired` counter — an
  excellent reward-hacking probe.
- Every rejection logs its crop to a 200-entry ring buffer, dumped at episode
  end. Free HUD-drift early warning.

**Calibration tool**: boxes stored as screen fractions (so a resolution change is
a rescale, not a redo); auto-suggested thresholds from 2-clustering in-box
pixels; and **glyph harvesting rather than hand-drawing** — segment candidates
across a few thousand frames, cluster them (a bitmap font clusters beautifully),
label 10–14 medoids once.

**Screen states** (`LOADING`, `IN_GAME`, `DOWNED`, `GAME_OVER`, `PAUSED`,
`ANOMALY`, …) via hand-written rules over a 32×18 grayscale downsample plus
probe crops, with hysteresis. No learned classifier — you have no labels, you
need <1 ms, and you need to understand failures. `GAME_OVER` by dHash;
`DOWNED` by mean-saturation collapse (last stand desaturates); `LOADING` by
near-zero frame difference.

## Reward

All terms in points-equivalent units, divided by `POINTS_SCALE = 100`, every term
logged separately every step.

`max(0, Δpoints)` alone is wrong: it makes buying reward-*neutral*, so the agent
never opens doors or buys off the wall — and the starting M1911 caps you around
round 5–8. Full spend-tracking is principled but fragile. **Use `max(0, Δ)` as a
robust floor plus a separate, explicitly-tuned progression term**, which
decomposes cleanly and degrades gracefully if the hint detector misfires.

| term | value |
|---|---|
| kill / hit | +10…+130 (free, from Δpoints) |
| barrier repair | +10 × **0.2** |
| round complete | +500 (flat — do not scale with round) |
| door opened (novel) | +300 |
| wall weapon (novel) | +200 |
| ammo rebuy | +50 |
| damage event | −100 |
| death | −1500 |

Then `r = clip(Σ / 100, −20, +8)`.

- **`GAIN_CAP = 400` is a safety property, not a tuning knob.** A misparse
  reading 4500 as 45000 must not be convertible into 45,000 reward. Alarm on
  every clip.
- **Novelty-gate** door and first wall-buy bonuses (once per kind per episode) so
  there's no buy/rebuy farming loop.
- **No per-step survival bonus.** A `+ε` liveness term makes "stand in a corner"
  a strong local optimum on Nacht — it genuinely works for several rounds. The
  round bonus already rewards survival, conditioned on progress.
- **Flat round bonus**, not scaling: increasing late-episode rewards create a
  variance problem and can dominate. Discounting already prices "you had to
  survive to get here." This is the constant I'd tune first, in the sim.

### Reward hacking, ranked by likelihood

1. **Board farming — the big one.** Repairing a barrier plank gives points, and
   zombies keep tearing them down, so an agent can park at a window in an
   infinite repair loop, never killing anything. This is *a real technique humans
   use*, which is exactly why RL will find it. Hence the 0.2× weight, a per-round
   cap, and an alarm at `points_from_repairs / total_points > 25%`.
2. **Hit-farming without killing.** In later rounds, spraying a high-HP zombie
   pays more than efficient kills. Detect via `kills_per_round` vs
   `points_per_round` divergence, and `shots_fired` vs `kills`.
3. **HUD misparse exploitation.** If the parser occasionally reads a huge number,
   the policy may find a visual configuration that triggers it. Sounds exotic; it
   is precisely the failure class RL finds. `GAIN_CAP` + the delta alphabet make
   it structurally unconvertible.
4. **Gaming the damage detector** by avoiding *looking at* red things rather than
   avoiding damage. Mitigate with EMA baseline subtraction, round-transition
   masking, and a border-ring mask. Validation: episodes with more detected
   damage should end sooner — if that correlation is absent, the detector is
   measuring scenery.
5. **Pause exploitation.** ESC freezes the world at zero risk — an infinite-value
   state. Keep ESC out of the action space, and if `PAUSED` is detected
   mid-episode, un-pause and flag the steps `bad_step`. Cheap; catastrophic if
   omitted.

> **The discriminator:** never optimize `rounds_survived` directly, always report
> it. If shaped return climbs while `rounds_survived` doesn't, you're being
> hacked. That divergence plot is the single best anti-hacking instrument.

## The simulator

**Visual sim-to-real transfer is a NON-GOAL** — no sim-trained weights ever load
into the real agent's encoder. Say it out loud in `docs/sim_lies.md`, a
maintained list of what the sim doesn't model (ballistics, hitbox geometry,
animation lockouts, box RNG, pathing jank, audio, the actual visual scene).

The sim exists for five things: a plumbing testbed for the whole runtime at
10,000× speed; an algorithm-correctness lab; a hyperparameter and reward-shaping
lab (**board farming will show up here**, because you modelled the economy); the
data-budget experiment (below); and — now that real-game parallelism is off the
table — **the actual source of data volume.** Pure CPU, no window, no
input-focus problem, so it scales to every core the machine has, not to a small
fixed number.

Run it as `os.cpu_count()` worker processes, not threads (numpy releases the
GIL inconsistently across ops; separate processes vectorize cleanly). Pin each
worker to one BLAS thread (`OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`) or
per-process numpy calls fight each other for cores and throughput stops scaling
linearly well before you run out of cores — verify the scaling curve in M1
rather than assuming it.

Keeping it honest: domain-randomize every number you're unsure of, per episode,
and report performance *across* the range; randomize the **timing** structure too
(latency ∈ {0,1,2,3} frames, action repeat ∈ {3,4,5}, 5% action dropout); inject
HUD noise mirroring the real parser's failure modes; expose one `sim_hardness ∈
[0,1]` dial that scales all of it. Sim score is a pass/fail smoke test, never a
leaderboard.

Design: Nacht as a 2D polygonal floor plan over two levels; zombies as
struct-of-arrays (never per-object Python); navigation as a **0.5 m grid with a
BFS distance field to the player recomputed every 3 ticks** — ~2,400 cells,
microseconds in numpy, and it captures the only thing that matters (zombies path
to you; doors and windows gate the paths).

**The critical honesty constraint: the player has hidden regenerating HP and the
sim does not expose it.** It exposes only the decaying `damage_flash` scalar with
the same dynamics as the real detector, false positives included. A sim that
gives the agent a health bar teaches a lesson the real game won't honour.
Likewise, shooting uses an aim cone with randomized σ — **the sim must not make
aiming trivial**, or "spray at everything" becomes optimal in sim and disastrous
in game.

Render mode is a crude DOOM-style **first-person raycast**, not a top-down view:
the sim's job is to reproduce the *shape* of the perception problem — partial
observability, things behind you are invisible, targets small and needing to be
centred. A top-down view is a different, much easier POMDP that would validate
nothing. Draw a fake HUD strip using the **real digit glyph atlas**, which lets
you run the real HUD parser against sim frames as a CI integration test.

## Milestones

**M0 — Feasibility spikes (Windows, ~1 day, throwaway).** Each has a binary
pass/fail and writes a number into `docs/spikes.md` that becomes a config
constant.

| # | Spike | PASS |
|---|---|---|
| S1 | **Synthetic input** | View rotates repeatably, W moves. Ladder: scancode `SendInput` → kernel virtual HID → hardware HID emulator |
| S2 | **Capture** | ≥60 fps, p99 latency <10 ms, no black frames, **in a mode that also passes S1** |
| S3 | **End-to-end latency** | A *stable, measurable* number <120 ms. It needn't be small — it must be known |
| S4 | **Mouse linearity** | R² > 0.98 with in-game smoothing/acceleration off. If this fails the discretized-yaw design needs rethinking — which is why it's in M0 |
| S5 | HUD legibility | Glyphs crisp, boxes stable |
| S6 | 4 h idle soak | No crash, hang, memory growth, driver reset |
| S7 | Blackwell/PyTorch | sm_120 kernels run; ≥200 grad steps/s at batch 256 **while the game runs**, no framerate impact |

Plus: **`timescale`** is `sv_cheats`-gated and reachable from the console, so
it's *inside* your constraint. If zombie AI and spawn logic stay correct at
`timescale 2` you double your data rate — the biggest possible single win. Verify
spawn counts and health per round against the table above.

Plan for **borderless windowed** as the default: DirectX 9 exclusive fullscreen
generally cannot be captured by the Desktop Duplication API at all.

**M1 — Spec + sim skeleton (Linux).** `spec.py` frozen and hashed; `NachtSim` in
state mode; random + scripted agents; logging, episode store, tests.
*Exit:* scripted beats random by ≥2 rounds over 100 episodes; ≥5,000 steps/s
single-process; vectorized across `os.cpu_count()` worker processes with
near-linear scaling confirmed (not assumed) up to that count — this is the real
data-volume lever now that real-game parallelism is out of scope; golden test on
space shapes and `SPEC_VERSION`.

**M2 — From-scratch PPO (Linux).** *Exit, all three:* (a) **solves CartPole and
LunarLander** — do this, it's the only way to know your PPO is correct before
blaming the environment; (b) beats scripted in state mode; (c) beats random in
render mode, validating the CNN path.

**M3 — From-scratch DQN (Linux).** Double + Dueling + n-step + PER on the compact
profile. *Exit:* (a) CartPole control; (b) within 1 round of PPO; (c) **the
data-budget experiment** — rate-limit the sim to 15 steps/s, cap at 2.6M steps,
and sweep replay ratio ∈ {1,2,4,8,16,32}. This tells you whether anything learns
in a weekend's worth of data *before* you spend a weekend. It is the single most
informative result in the plan.

**M4 — Real-game perception (record Windows, verify Linux).** *Exit on a held-out
10-minute recording:* parse rate ≥99.5% with sanity filters; ≤1 `hud_lost` per
10 min; screen-state F1 ≥0.99 for LOADING/IN_GAME/GAME_OVER; damage-detector
precision/recall reported against hand labels; parse <3 ms.

**M5 — Real-game env loop (Windows).** *Exit:* the **random** agent runs 2 hours
unattended, ≥20 auto-reset episodes, overrun rate <2%, zero stuck keys, zero
interventions. Random is deliberate — it can't be blamed for harness failures.

**M6 — Demos + BC (record Windows, train Linux).** 2–4 hours (~110k–220k
decisions) of *deliberately varied* play — camping, trains, bad positioning,
early deaths. BC on expert-only data has no idea what to do once it drifts
off-distribution, and DAgger is impractical here.

*Exit:* BC beats the scripted baseline on real `rounds_survived`.

**M7 — Real-game RL (Windows).** Sample-efficient algorithm from BC init, replay
ratio 8–16, actor/learner split, governor. *Exit tiers:* T0 a 48-hour unattended
run with <5 watchdog restarts and zero data loss; T1 `rounds_survived` over the
last 20 episodes exceeds BC by ≥1 round *and* holds under frozen-policy eval;
T2 (stretch) beats the human demonstrator. **Anti-hacking gate for T1:** no
single reward term >60% of return, and repairs <25% of points.

### Demonstrations and BC — the details that decide whether it works

**Capture raw mouse counts, not cursor deltas.** In a mouse-look FPS the cursor
is captured and re-centred, so position deltas are useless. Raw Input gives
counts — **the same unit your synthetic mouse action emits.** That symmetry is
what makes BC work at all. Timestamps must come from the same monotonic clock as
the capture thread. Store both raw summed counts and the bin index, so changing
the bins later is a re-quantization rather than a re-recording.

Cross-validate the labels cheaply: `fire` must correlate with mag-ammo
decrements; yaw must correlate with optical-flow horizontal motion. Below ~0.9
means a timing bug.

**The `button` head is `none` ~95% of the time — with plain cross-entropy you
will get a policy that never reloads.** Use class-balanced or focal loss. This is
the classic, predictable failure here.

**Pretrain the value head on demo Monte-Carlo returns.** A random critic in a
15 Hz environment with 20-minute episodes takes an enormous number of real steps
to become useful; this is one of the highest-return-per-line items in the plan.
Add auxiliary heads predicting next-step Δpoints bucket and damage flag — that
forces the encoder to represent "am I hitting things" and "am I being hit,"
exactly the features the value function will need.

Augment with random shift ±4 px and mild brightness jitter. **No horizontal
flips** — a flip inverts the yaw label and mirrors the HUD, and Nacht isn't
mirror-symmetric.

Evaluate at four levels, because per-frame accuracy lies: balanced per-head
accuracy vs the majority-class baseline; **rollout-statistic divergence** in the
sim (fire duty cycle, mean |yaw|/s, reload frequency — a policy at 70% per-frame
accuracy can still never pull the trigger, so assert each statistic is within 2×
of the human's); the action-inertia check; then real-game episodes.

### Why a high replay ratio, stated as arithmetic

The environment produces **15 transitions/s**, fixed. The RTX 5070 will do
roughly **200–500 gradient steps/s** on a 128×72 CNN at batch 256. Atari DQN's
canonical replay ratio is **0.25** — one update per 4 frames — which at 15
steps/s is **4 updates/s, leaving the GPU ~99% idle.** You'd be throwing away the
only resource you have in abundance. Push to **8–16**.

The known failure mode is loss of plasticity / primacy bias. Implement the
countermeasures from day one, not "if needed": periodic shrink-and-perturb resets
of later layers, LayerNorm and weight decay in the trunk, n-step returns with
annealed n (10 → 3), annealed γ (0.97 → 0.997), EMA targets.

For the borrowed algorithm slot: **a BBF-style high-replay-ratio discrete value
method** is the closest thing to the DQN you'll have just written, so you'll
actually understand it, and its reset machinery is exactly what this regime
demands. **DreamerV3-small** (~2.7 GB VRAM in the PyTorch reimplementations) is
the escape hatch if BBF plateaus — model-based methods extract more from a fixed
buffer, and you can train the world model offline from recorded episodes. Genuine
uncertainty about which wins; don't pre-commit.

## Verification

- `pytest` over recorded frames for HUD parsing and state detection, in CI on
  Linux.
- **`FakeCapture` (replays a recorded episode at simulated timing) and
  `FakeInput` (appends to a list) let `RealGameEnv` itself run end-to-end in CI
  on Linux** — scheduler, parser, reward shaper, reset FSM, episode store
  included. Only the two adapter implementations and the timing constants are
  genuinely un-testable off the gaming PC. Insist on this; it's what keeps the
  project moving on days you can't touch the Windows box.
- `NachtSim` asserted against the constants table (R1=4/R2=9/R3=14/R4=19/R5+=24;
  the 150/+100/×1.10 health curve).
- The real HUD parser run against sim render frames as an integration test.
- End-of-episode cross-check: tracked round count must equal the game-over
  screen's "rounds survived" digits. A mismatch is an unambiguous alarm, free.
- Latency histograms per stage; alarm if p99 exceeds budget.

## Risks, ranked by P(kills project) × P(discovered late)

1. **Synthetic input ignored by the engine.** Spike day one. Ladder: scancode
   `SendInput` → kernel virtual HID → hardware HID emulator (a microcontroller
   presenting as a USB mouse). ⚠ A kernel input driver is system-wide and
   **Vanguard and some EAC versions will not boot while it's installed** — if you
   play those, skip to the hardware route. Related: the game must be
   **foregrounded**, so add a focus guard that releases all inputs and pauses the
   agent if the window loses focus.
2. **Capture fails / black frames.** Borderless windowed by default; two
   implementations behind one `grab() -> (ndarray, t)` adapter.
3. **The game crashes or hangs during a multi-day run.** A 2008 DX9 title plus 48
   hours. The watchdog is a **separate supervisor process** watching an shm
   heartbeat and frozen frame hashes; it kills, relaunches, re-runs the reset
   FSM, logs the incident, and **truncates the in-flight episode so a crash never
   enters the buffer as a normal termination**. Plus a Windows setup checklist
   (no auto-restart, sleep, or screensaver). The most underrated killer here.
4. **HUD parse drift / silent reward corruption.** Continuous `reject_rate`,
   `confidence_p05`, and Δpoints-tail alarms; the rejected-crop ring buffer; the
   round-count cross-check.
5. **Reward hacking**, board farming above all — see above.
6. **Nothing learns, because there isn't enough data.** The most likely *quiet*
   failure. M3's rate-limited experiment tells you before the weekend is spent.
   If it fires: lean harder on BC and auxiliary losses, shrink the net, raise the
   replay ratio, switch to the model-based fallback, and adjust the target —
   beating the scripted baseline is a legitimate success for this project.
7. **Latency makes aiming unlearnable.** Set the sim's latency knob to the
   measured value and check whether PPO still learns to aim. An afternoon instead
   of a weekend.
8. **Actor/learner GPU contention blowing the deadline.** Measure overrun rate
   **with the learner under load**, not idle — this is the one people test wrong.
   Worth measuring rather than assuming: **running the actor's policy on CPU**
   (~5–10 ms for this net) fits the budget and removes contention entirely.
9. **Round-1 oversampling.** No start-round dvar without a mod, so real episodes
   only ever teach rounds 1–5. The sim curriculum is the answer.
10. **RTX 5070 is Blackwell (sm_120)** — needs PyTorch built against CUDA 12.8+.
    The default pip wheel will likely give a runtime that can't see the GPU.
11. **The damage detector may not be good enough.** Screen redness is a weak
    proxy. The best cheap upgrade is **audio** — WASAPI loopback, a 64-bin
    log-mel over 200 ms, 2 channels for coarse L/R. The hurt grunt, heartbeat and
    proximity growls are extremely informative when the agent sees only ~65° of a
    360° threat space. **Reserve the `audio` key in `spec.py` now** so adding it
    is a profile flag, not a spec migration that invalidates the replay buffer.
12. **Scope creep and motivation loss** — the actual leading cause of death for
    hobby projects. Structural mitigation: M1–M3 are the stated learning goal
    (PPO and DQN from scratch) and are **entirely achievable on Linux with no
    game at all.** If the Windows side stalls on S1, you still get the thing you
    actually wanted.

## Timeline expectations

A BC agent that moves and shoots plausibly: days. An RL agent that beats the
scripted baseline: a serious multi-week project, and a legitimate success. One
that beats a competent human: unlikely — and given the goal is learning RL, fine.
The surrogates are where most of the learning happens, and they're the part you
can start on immediately, on any machine.
