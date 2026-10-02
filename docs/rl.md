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
| a display | a rootful Xwayland with glamor, `:60`, `:61`, ..., inside a headless Weston (GL renderer) of exactly the game's size (or, without weston, in `-shm` mode on a hidden Hyprland special workspace) | `realgame/instances.py` |
| a game | Plutonium's T4 client in LAN mode (`umu-run plutonium-bootstrapper-win32.exe t4sp <game dir> -lan`), with a private copy of the Steam prefix and of Plutonium, settings as `+set` arguments | `realgame/instances.py` |
| no network | a network namespace with loopback only: the game cannot reach anything outside the machine | `realgame/instances.py` |
| input | XTEST into that X server only -- the same key names and counts as the uinput device | `realgame/xtest.py` |
| capture | MIT-SHM grabs of the game window on that display (the existing capture, pointed at `:6x`) | `demos/x11_capture.py` |
| sound | a PulseAudio null sink per instance, so a policy that hears hears its own game | `realgame/instances.py` |

Nothing touches your desktop's X server, mouse or keyboard: you can keep using the computer while it trains.
Under Weston nothing appears on your desktop at all; `scripts/watch.py`-style captures, or a screenshot of a
display (`DISPLAY=:60 import -window root shot.png`), are how to look. (With the Hyprland host,
`scripts/instances.py show` toggles the workspace.)

### Setup, once

```sh
sudo pacman -S weston                       # the headless host (without it the fleet falls back to Hyprland;
                                            # ~/.local/bin/weston is a no-root stand-in, unpacked from the package)
# Plutonium, from its CDN, with the open-source CLI updater (github.com/mxve/plutonium-updater.rs):
curl -L https://github.com/mxve/plutonium-updater.rs/releases/latest/download/plutonium-updater-x86_64-unknown-linux-gnu.tar.gz | tar xz
./plutonium-updater -d ~/.local/share/plutonium
```

Plutonium runs the Steam copy's own game files; nothing in the Steam install changes, and the Steam game keeps
working as before. Your Steam `config.cfg` (bindings, sensitivity) is copied into each instance's Plutonium
profile at every launch, so the instances play with the settings the demos were recorded with.

### Why Plutonium, Weston, `-shm`, 1440p -- each was a failure first (2026-09-27)

- **Steam's `CoDWaW.exe` cannot start outside the Steam client**: it is wrapped in SteamStub DRM and stops at
  "Application load error P:0000065432". Plutonium starts the same game files through its own executable,
  and in `-lan` mode never logs in, contacts its servers or runs its anti-cheat.
- **Xwayland's default GPU path crashed under Hyprland** (abort in `xwl_glamor_gbm_dispose_syncpts`, NVIDIA
  explicit sync) as soon as the game started, and drew ~2 fps while hidden, so the Hyprland host uses `-shm`.
- **But `-shm` costs the game most of its frames, and tears every capture.** Without DRI3, NVIDIA's Vulkan
  reads each frame back and sends it as PutImage: a 1440p frame arrives as four 4 MB strips over ~65 ms, so
  the game ran at ~11 fps on ~1.2 cores, and a grab between strips got two frames in four bands. Under a
  headless Weston with the GL renderer, glamor works (DRI3 through linux-dmabuf; no crash) and a frame is one
  GPU copy: 60 fps, whole frames, the game ~0.6 of a core, a 1440p grab 1.4 ms. It needs `-noreset`: a reset
  (last X client gone) re-creates Xwayland's window, which crashes Weston 15.0.1's kiosk shell.
- **Hyprland's special workspace does not survive the monitor sleeping.** When the last physical monitor
  disconnects, Hyprland folds special workspaces into normal ones and tiles the windows -- and a rootful
  Xwayland resizes its root, and the game, to the tile (941x1030 here). A headless Weston per instance, sized
  to the game, has no monitor to lose.
- **One Plutonium folder cannot serve two games**: it keeps the running game's profile and logs beside its
  executable. Each instance gets a reflink copy (free on btrfs), remade when Plutonium updates.
- **The game will not finish loading until its window has focus**, and there is no window manager to give it:
  `instances.py up` focuses each window as it appears, and the env's focuser keeps it focused.
- **`+map` on the command line is thrown back to the menu** in LAN mode (the co-op menu asks for an "online
  profile" -- a local save profile despite the name -- that LAN mode cannot create). The same `map` typed into
  the console loads and stays, so the reset types it, then taps Enter through "Click to Start the Mission".
  From launch to a settled 500 takes ~1-4 minutes at 1440p; later resets are just the map load.
- **The HUD reads at 2560x1440, not 1280x720**: the glyph atlas is 1440p text at half scale, and natively
  rendered 720p text does not match it. The fleet defaults to 1440p.

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

**What only the game can answer**, and `scripts/spike_instances.py` asks: whether WaW in such a server takes
that input (DirectInput acquires, the view turns), and whether several can run at once. Run it before anything
long; each check is pass/fail. **Result on 2026-09-27, two Plutonium instances at 1440p under Weston,
offline: all six checks pass on both** -- a 20-degree turn moved the turned game 27 px (expected ~32) and the
other 0 px; points read 500; the console opens on a key; grabs ~30 ms.

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

## What the first real runs taught (2026-09-27)

Four Plutonium games, `bc_real3` (all five demos, 61k decisions) as the starting policy. Each of these was
found by watching a run go wrong, then fixed:

- **95% of steps late** (`bad_step_frac`). Every actor's NumPy started an OpenBLAS pool of one thread per core
  -- the frame resize is a matrix product -- and four of them fought over 12 threads (load average 52).
  `train()` now pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` before spawning actors: a step's grab, resize and HUD
  read take ~25 ms single-threaded, and late steps fell to under 5%.
- **The console left open.** The console key is a toggle, and a reset pressing it blind closes a console that
  was already open -- so `map nazi_zombie_prototype` went to the game as key presses (`t` opened the chat:
  "(Dead)zombiesai0: type"), and the policy's WASD went into the console. `realgame/console.py` now reads
  whether the console bar is on screen; `console_command` only toggles when it must, types nothing unless it
  sees the console open, and the env closes a console it finds open mid-episode (a bad step).
- **Every agent staring at the floor.** The cloned policy drifts ~1.4 deg/s in pitch and never learned to look
  back up. `RealGameEnv` adds a pitch spring back to level (2 s time constant): the policy's look bins go out as
  chosen, plus the spring's pull -- part of the environment, like aim assist, so the PPO ratio stays right.
- **A game-over screen taken for a fresh game.** It still shows the dead game's 500 points; after typing a
  command the reset now also requires the HUD to disappear (the load) before a 500 counts.
- **A slow load restarted forever.** A reset try that times out types `map` again, which starts the load over;
  beside three live games a 1440p load can outlast a 90 s try, so it never finished until the game was
  relaunched. Tries are now 240 s (420 s after a relaunch).
- **A relaunch that waited for a map nobody started.** After a relaunch the command is typed too (Plutonium's
  LAN mode cannot start one from its command line), and retried every 3 s until the console takes it.
- **Resets cost 60-90 s.** `fast_restart` restarts Nacht in ~4 s from play or from the game-over screen; the
  reset tries it first and falls back to `map`.
- **Deaths the HUD never showed.** `map` from Plutonium's LAN menu starts a *co-op* game, and co-op has last
  stand: downed on the floor with a pistol until bleed-out, no points penalty -- unlike the solo game the demos
  were recorded in, where going down is the game over. Co-op does draw the scoreboard by itself when the player
  goes down (`realgame/scoreboard.py` reads its header by shape). A fresh co-op game also opens with it drawn,
  so the reset clears it with the scores key -- held 150 ms: 30 ms taps are too short for the game to see --
  and from then on its return for 3 steps ends the episode as a death (`reason: down`), as solo Nacht would.

## Running it unattended

```sh
uv run python scripts/instances.py up --n 4        # once; the games keep running between training runs
setsid nohup uv run python -u scripts/train_rl.py runs/bc_real3/bc.pt --actors 4 --out runs/rlN \
    > runs/rlN.log 2>&1 < /dev/null &              # survives the terminal closing
tail -f runs/rlN.log                               # one line per PPO update, one per episode
pkill -INT -f scripts/train_rl.py                  # stop: releases every key, writes checkpoint.pt
```

To continue, pass the last run's `checkpoint.pt` as the first argument with a new `--out`: the update count
(so no second critic warm-up), the KL weight and the KL anchor carry over -- the anchor is always the BC
policy the chain started from (`rl/parallel_ppo.py: root_prior`), never the checkpoint being continued.

## Several PCs

One run can take games from every gaming PC in the house. The machine with the GPU you want to train on runs the
**learner** (and its own games, as before); every other gaming PC runs a **worker** that plays its games for it
(`rl/fleet.py`, `scripts/fleet_worker.py`). The actors on a worker are the same processes `train()` starts --
they still ship segments through a local queue and follow a local `weights.pt` -- and the worker forwards the
one and pulls the other over the LAN. So lag, bad steps and "never stall a live game" work as described above.

```
 learner PC (RTX 5070)                                   other gaming PC
 4 games ── actors 0..3 ──► learner ◄── POST /segment ── forwarder ◄── actors 100..103 ── 4 games
                            weights.pt ── GET /weights ──► puller ──► weights.pt
```

The lts laptop is neither. It has no GPU worth training on, and it doesn't need to be on the training path: it
stays the site, and the place the PCs report to (`publish_live.py`). Putting the learner there would add a hop
for every frame, for nothing.

```sh
# Once: a shared secret, the same on every machine (in the environment, never in the repo), SSH from the
# learner to each PC with a key (`fleet.py` runs non-interactively), and the repo cloned at ~/Projects/ZombiesAI.
python -c 'import secrets; print(secrets.token_hex(16))'      # -> ZOMBIES_FLEET_TOKEN

# Before each run, on the learner PC: every other PC to this commit and this machine's game settings, games up.
git push                                                       # the PCs fetch the commit from origin
uv run python scripts/fleet.py prep rig2.lan                   # or rig2.lan=2 for 2 games there; $ZOMBIES_FLEET_HOSTS
uv run python scripts/instances.py up --n 4
ZOMBIES_FLEET_TOKEN=... uv run python scripts/train_rl.py runs/bc_real3/bc.pt --actors 4 --listen :47860 --out runs/rl5

# Each other gaming PC (or leave deploy/zombiesai-worker.service running there):
ZOMBIES_FLEET_TOKEN=... uv run python scripts/fleet_worker.py --learner <learner-host>.lan --actors 4
```

**`scripts/fleet.py prep`** (`rl/fleet_admin.py`) does, on each PC over SSH and all PCs at once, what a refusal
would otherwise send you over there to do:

| step | what | |
|---|---|---|
| reach | the checkout is there, with nothing uncommitted | a PC with uncommitted work is left alone |
| commit | `git fetch`, check out the learner's commit, detached | no branch of theirs moves; it must be pushed |
| sync | `uv sync` | |
| settings | the learner's `config.cfg` installed in the fleet's root | the PC's own Steam profile is not touched; a `--client steam` fleet is warned, not installed (only Plutonium reads the file) |
| games | `instances.py up`, restarted if their settings just changed | a game reads config.cfg at launch |
| worker | `zombiesai-worker` restarted if the code or settings changed | |
| check | `fleet_worker.py --describe`, judged as hello judges it | |

It refuses to start while the learner's own checkout has uncommitted changes: the PCs can only get the last
commit, and hello compares commits, so they would train on different code under the same name.
`scripts/fleet.py check` runs only the last step, and changes nothing.

A worker can be left running (`deploy/zombiesai-worker.service`). It waits for a learner, joins whatever run
the learner starts, and stops its actors (every key released) when it hasn't heard from the learner for 60 s.
Then it waits for the next run. The games stay up between runs, as they do on the learner.

**What the learner refuses**, at hello, before a single segment:

- **A different commit or spec version.** The actor code, reward shaping and observation layout must be the
  learner's. `scripts/fleet.py prep` fixes it.
- **Different game settings.** Sensitivity, `m_yaw`/`m_pitch`, field of view and every key binding come from
  the `config.cfg` `instances.py` copies into each instance: one installed in the fleet's root
  (`runs/instances/config.cfg`, which is what `prep` puts there), else the machine's own Steam profile. A
  different sensitivity makes every look bin turn by a different angle, and a different binding makes a key do
  something else, and neither shows up in any number. The refusal lists what differs. Resolution, fps and vsync
  are set per instance on the command line, so they're not compared. (A `--client steam` fleet plays with the
  config inside each instance's copied Steam prefix instead, and that is what it reports.)
- **A name another live worker is using.** Workers are told apart by `--name` (the host name by default) and a
  random id per process. A second process under a name the learner heard from in the last 60 s is refused, so two
  PCs that share a host name are never merged into one machine; it waits and asks again, so a worker that was just
  restarted takes its PC's place once the old one has gone quiet.

**What it takes care of:**

- **Actor numbers.** Worker k's actors are `100k`, `100k+1`, ...: `episodes.jsonl`, the stream's numbers and the
  reward scaler's per-actor returns keep the machines apart. Its RNG seed moves by the same offset.
- **The starting checkpoint.** The worker downloads it from the learner (checked by SHA-256), along with the run's
  settings, so nothing has to be copied around by hand. `--counts-per-degree` and `--record-every` on the worker
  override the learner's for that machine. Clips are written on the machine that played them.
- **What it accepts over the network.** Segments are `.npz` read with `allow_pickle=False` and shape-checked;
  weights and the checkpoint are opened with `weights_only=True`; every request needs the token. It is plain
  HTTP, meant for a home LAN: for anything wider, put the machines on Tailscale or WireGuard.

**What to watch: each machine on its own.** One PC's games can feed the batch worse data than the rest -- a
slower GPU makes more late steps, a broken install plays worse -- and a run-wide average hides it. So every
update, the learner also scores each machine separately (`MachineStats` in `rl/parallel_ppo.py`):

- `metrics.jsonl` gains `machines` and `per_machine`, keyed by machine number (0 is the learner, 1 the first
  worker to join, ...; numbers and not names, because the last row is published with the site's training page):
  segments taken, `dropped_segments` (too stale for `max_policy_lag`), steps, `bad_step_frac`, `policy_lag_mean`,
  and over each machine's last 50 games `round_reached_mean`, `return_mean` and `seconds_mean`.
- `runs/<run>/fleet.json` has the same per machine, by name, plus what each worker says it sent and lost on
  the way (the link, or the learner's inbox full), and when it was last heard from.
- The dashboard (`scripts/dashboard.py`, and the site's training page) draws them in an **Each machine** section
  for any run trained on several PCs: late steps against the 10% line, round reached, each machine's share of the
  training data, and segments too stale to use (pooled over 10 updates), one line per machine. Pick the run and
  hide machines with the chips above the charts. The local page names each worker; the site's says "Machine 1".
- The log says it when a machine's late steps pass 10% of its steps, and again when they fall back under 5%:
  "rig2: 18% of its steps late -- ... fewer --actors there?". `actors_alive` counts every machine's actors.

Traffic is ~415 KB/s per game before compression (128x72 frames at 15 Hz): nothing for wired gigabit,
worth checking on Wi-Fi.

**On the site.** Each worker PC can report itself to lts the way the training PC does (`publish_live.py
--worker <id>`; README.md, "Live on the site"): the Training Room shows a card per machine, with what its worker
is doing, read from the `runs/fleet/status.json` the worker rewrites every few seconds.

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
