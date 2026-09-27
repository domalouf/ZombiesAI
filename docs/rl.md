# Reinforcement learning on the real game, several games at once

Behavioural cloning gets a policy that plays like the person it watched, mistakes included. This is the next
step: the policy plays, the HUD scores it, and PPO moves it toward whatever scored -- with several copies of
World at War playing at once, so there is enough experience to learn from.

```sh
uv run python scripts/instances.py up --n 4                            # four games, four private X servers
uv run python scripts/spike_instances.py --counts-per-degree 9.09     # does each one take input, alone?
uv run python scripts/train_rl.py runs/bc_real2/bc.pt --actors 4 --counts-per-degree 9.09 --out runs/rl1
uv run python scripts/dashboard.py --watch 30                          # the run beside every other
uv run python scripts/play_real.py runs/rl1/checkpoint.pt --minutes 3  # play the result yourself
```

## Why the game can run in parallel after all

`PLAN.md` opens with "the real game cannot be parallelized": WaW's DirectInput only listens to the foreground
window, a desktop has one, and on Windows more desktops means more GPUs. That was true of Windows. On Linux the
game runs under Proton as an X11 client, and **the thing with one foreground window is an X server** -- which
is cheap. So each instance gets:

| | how | module |
|---|---|---|
| a display | a rootful Xwayland, `:60`, `:61`, ... on a hidden Hyprland special workspace, floating, at a fixed size | `realgame/instances.py` |
| a game | `umu-run CoDWaW.exe` in a private copy of the Steam prefix, settings as `+set` arguments, `+map` straight into Nacht | `realgame/instances.py` |
| input | XTEST into that X server only -- the same key names and counts as the uinput device | `realgame/xtest.py` |
| capture | MIT-SHM grabs of the game window on that display (the existing capture, pointed at `:6x`) | `demos/x11_capture.py` |
| sound | a PulseAudio null sink per instance, so a policy that hears hears its own game | `realgame/instances.py` |

Nothing touches your desktop's X server, mouse or keyboard: you can keep using the computer while it trains.
`scripts/instances.py show` toggles the workspace if you want to watch.

**What was measured on this machine before any of it was written** (a throwaway rootful Xwayland on the
hidden special workspace, never focused):

- XTEST relative motion arrives as XInput2 **raw motion** carrying exactly the counts sent (`[7, -3]` in,
  `[7.0, -3.0]` out), on the master pointer. That is the event stream Wine's X11 driver turns into DirectInput
  and raw-input mouse deltas; raw values also skip the server's pointer acceleration, so counts per degree
  stays a straight line.
- Key and button events reach the window holding that server's focus, whatever Hyprland has focused.
- A Vulkan-over-X11 client (DXVK's path) renders at **60 fps** there while hidden, and capture reads every
  frame (Xwayland paces copy-presents on its own timer, not on the compositor's frame callbacks). The GPU is
  the RTX 5070, not a software renderer. Grabs take ~5 ms at that size.
- With a `float; size` rule the root is exactly the requested size, so a 16:9 game fits it.

**What only the game can answer**, and `scripts/spike_instances.py` asks: whether WaW under Proton, in such a
server, takes that input (DirectInput acquires, the view turns), and whether several can run at once (Steam
API, memory). Run it before anything long; each check is pass/fail.

## The loop

```
 instance 0 ── RealGameEnv ── actor 0 (CPU policy) ──┐  segments        ┌── weights.pt
 instance 1 ── RealGameEnv ── actor 1 (CPU policy) ──┼──────────────►  learner (GPU) ──┤
 instance 2 ── RealGameEnv ── actor 2 (CPU policy) ──┤                 PPO + KL to BC  └── metrics.jsonl, checkpoint.pt
 instance 3 ── RealGameEnv ── actor 3 (CPU policy) ──┘
```

**`realgame/env.py` — one game as an environment.** 15 decisions a second on absolute deadlines; a late step
is flagged `bad` and the schedule moves on from now (skip, never catch up). The observation is the frame
grabbed at the deadline. The reward comes off the HUD crops of the same grab, through the parser, the tracker's
rules, and the shaper:

- `terminated` on a settled death (solo Nacht has no revive: the downed penalty is the game over);
- `truncated` -- bootstrapped, since nothing proves the game ended -- on `hud_lost` (three implausible
  changes in a row), a frozen or lost picture for 5 s, the HUD gone for 8 s, or the 45-minute cap;
- **reset** is a state machine: release every key, type `map nazi_zombie_prototype` into the console, and
  wait for the HUD to settle on 500 points on a live picture; three failed tries relaunch the game, and two
  failed relaunches raise, which restarts the actor.

**`realgame/hud_reward.py` — what pays.** Settled gains (a kill is one gain, not four partial reads of a
rolling counter); a round when the tracked round goes up; death; purchases by price, novelty-gated (1000 is
a door or the debris, 200/600/1200 a wall gun). **Repairs are the one thing the HUD can't tell from a hit** --
both are +10 -- so the actor's own actions decide it: a small gain within ~1.3 s of pressing use, with the
trigger untouched, is a plank. That keeps `PLAN.md`'s board-farming defences working: the 0.2 weight, the
per-round cap, and the repair-share gate on the dashboard.

**`rl/parallel_ppo.py` — the learner.**

- Actors run the policy on a CPU thread (0.5 ms a decision for the BC net), sample every head, and ship
  segments of up to 256 decisions (~17 s). A segment ends at an episode's end, and carries the frames before
  its first step, so the learner rebuilds each step's frame stack by index arithmetic -- frames stored once.
  A test checks that it is exactly the stack the actor acted on.
- The learner gathers 4096 decisions (~70 s of four games), recomputes values with its current network,
  runs GAE per segment, and does the clipped PPO update. Actors are at most `max_policy_lag` versions behind
  (the log shows `policy_lag_mean`); the recorded behaviour log-probs make that a correct ratio.
- **Critic warm-up.** The BC value head never saw this reward. For the first 5 updates only the critic head
  trains, on a frozen encoder; the policy stays exactly as cloned. Advantages from a random critic are noise,
  and noise is what wrecks a pretrained policy fastest.
- **A KL anchor to the BC policy**, `kl_coef * KL(pi || pi_BC)` over the action heads, decaying from 0.2 to
  0.02 -- VPT's recipe for fine-tuning a behavioural prior with RL without forgetting it. `kl_ref` on the
  dashboard is how far it has moved from BC.
- Bad steps stay in the GAE chain (time passed) but are masked out of the losses (nobody chose them).
- Rewards are scaled by a running estimate of the discounted return's spread.
- An actor that crashes -- a game that died, a reset that failed -- is restarted, up to 20 times each.
- Actors never block on the learner: a segment that can't be delivered in 250 ms is dropped, because an actor
  stalled mid-game leaves its keys held down.

Actions are dispatched per tick as sampled -- the look bins as three sub-moves, no mouse motor -- because a PPO
ratio is only right if what was sent is what was sampled. `play_real.py`'s smoothed "mean look" is for
watching a policy play, not for training one.

**Output.** `runs/<run>/metrics.jsonl` uses the names the dashboard already charts for PPO, plus `kl_ref`,
`kl_coef`, `policy_lag_mean`, `dropped_segments`, `bad_step_frac` and `actors_alive`. `checkpoint.pt` is a
BC-format checkpoint with an `rl` section, so `play_real.py`, `eval_bc.py` and `watch.py` play it unchanged,
and it can be passed back to `train_rl.py` to continue. `--record-every 5` keeps every fifth episode of each
actor as a clip (frames, actions, HUD crops, rewards) under `runs/<run>/episodes/`.

## Rehearse on the sim first

The same actors and learner run on NachtSim's rendered view with `--env sim`, as fast as the CPU allows (about
300 steps/s per actor process on this machine) -- the whole pipeline, minus the game:

```sh
uv run python scripts/train_rl.py runs/bc_real2/bc.pt --env sim --actors 8 --total-steps 300000
```

A BC policy cloned from the real game will not play the sim well (`docs/sim_lies.md`: the pixels are not the
same), so treat this as a plumbing and hyperparameter check, not a result.

## What to expect, in numbers

Four instances at 15 Hz are **60 decisions a second, ~216,000 an hour** -- four times what `PLAN.md` budgeted
for one game, and a weekend is ~10M. That is on-policy-PPO territory for fine-tuning a policy that already
plays, not for learning from scratch. How many instances fit is the spike's other question: each game is
roughly 1-1.5 GB of RAM, the GPU has 12 GB, and the machine has 12 threads. Start with 2, then 4; watch
`bad_step_frac` (overruns) as you add more.

The number that matters is `round_reached_mean`, not the return: the game's own round counter can't be
reward-hacked, the shaped return can. If return climbs and rounds don't, read the reward terms before
anything else. The dashboard grades the two anti-hacking gates on every run (largest term under 60% of the
return, repairs under 25% of points).

## When the spike fails

| check | likely cause | try |
|---|---|---|
| window | the game didn't start: Steam API, Proton, the prefix | `runs/instances/i0/game.log`; is Steam running? `scripts/instances.py up --visible` to see it |
| window, but only some instances | too many at once for RAM, or Steam refused a second copy | fewer instances; a longer `--stagger` |
| capture | the window is bigger than the X server | `--width/--height` must match what the game runs at (the fleet passes `r_mode`) |
| hud | the HUD at 720p is read from native pixels, the glyph atlas from 1440p ones | `scripts/instances.py up --width 2560 --height 1440` (more GPU per instance) |
| focus | no window manager to hand focus over | it sets focus itself; if Wine ignores it, file what `status` prints |
| mouse | Wine isn't treating XTEST motion as mouse-look | check keys first; then Wine's DirectInput `MouseWarpOverride` = `force` in the instance's prefix |
| keys | the console is off, or the key isn't `grave` | `monkeytoy 0` is passed at launch; check the binding |

Every instance's prefix is a copy, made once, of Steam's: settings you change in one game stay in that copy.
To start over, `scripts/instances.py down` and delete `runs/instances/`.

## Not done yet

- **A damage detector.** `damage_event` is zero on the real game: nothing reads the red vignette yet (the sim
  has one). Death still costs 15; being hurt costs nothing until it kills you.
- **Ammo rebuys** pay nothing: a rebuy costs half the gun, which collides with other prices. The prompt
  reader (`hud_crops.py` already keeps the crop) would fix both this and the door/double-barrel ambiguity.
- **The eval gate.** M7's T1 wants `rounds_survived` over 20 episodes of the frozen policy against BC's;
  `eval_bc.py` does that on the sim, not yet on the fleet.
- **Off-policy reuse.** Every segment is used once. The recorded episodes are there for a replay-based
  learner (the plan's BBF-style slot) when on-policy stops being enough.
