# ZombiesAI

[![tests](https://github.com/domalouf/ZombiesAI/actions/workflows/tests.yml/badge.svg)](https://github.com/domalouf/ZombiesAI/actions/workflows/tests.yml)

A reinforcement-learning agent learning to survive *Nacht der Untoten* — Call
of Duty: World at War's Nazi Zombies map — by screen capture and synthetic
keyboard/mouse input only, the way a person would. No mod tools, no memory
reading, no game scripting.

## Status: M2 (from-scratch PPO) in progress, M6 (demos + behavioural cloning) started

The full plan, including architecture, environment spec, reward design, the
milestone ladder (M0–M7), and ranked risks, is in [`PLAN.md`](./PLAN.md), and
as a formatted page: **[Undead Loop](https://claude.ai/code/artifact/9feeeb1c-8561-4347-912f-2fb98c9c441b)**.

Built so far (all Linux, no game needed):

- `src/zombiesai/spec.py` — the versioned contract: factored action space, the
  48-action compact profile, HUD/state layouts, and a `SPEC_VERSION` hash that
  every checkpoint and episode is stamped with.
- `src/zombiesai/sim/` — **NachtSim** in state or render mode: ground-truth
  round mechanics from the game script, domain-randomized guesses for everything
  else, latency/action-repeat/dropout randomization, and HUD noise. It also
  renders a first-person raycast view with a HUD, at the agent's 128×72 or any
  size, and can hand those pixels to the policy as its observation. What it gets
  wrong is listed in [`docs/sim_lies.md`](./docs/sim_lies.md).
- `src/zombiesai/reward.py` — the shaped reward, term by term, with the gain
  cap, novelty gating, and repair cap.
- `src/zombiesai/store/` — the episode store (crash-tolerant, spec-checked).
- `src/zombiesai/agents/` — random and scripted baselines.
- `src/zombiesai/rl/` — **PPO from scratch**: GAE with correct truncation
  bootstrapping, the clipped surrogate, a factored policy with one categorical
  per action head, and checkpoints stamped with `SPEC_VERSION`. Plus the
  convolutional trunk every pixel model here shares.
- `src/zombiesai/demos/` — **learning from real gameplay**: a demo recorder that
  logs raw mouse counts alongside the screen, video ingest for footage nobody
  logged input for, an inverse dynamics model that labels it, and behavioural
  cloning over pixels. See [`docs/demos.md`](./docs/demos.md).
- `src/zombiesai/realgame/` — the agent's hands: a factored action diffed against
  what is currently held, turned into key transitions and mouse sub-moves, and
  handed to a kernel virtual device. With capture (`demos/x11_capture.py`) and
  raw input (`demos/evdev_input.py`), the whole Linux side of the real-game loop
  exists. See [`docs/linux.md`](./docs/linux.md).

```sh
uv sync                                  # Python 3.13 env + deps (PyTorch: CPU on Linux, CUDA 12.8 on Windows)
uv run pytest                            # test suite
uv run python scripts/bench_sim.py       # steps/s and multi-process scaling
uv run python scripts/eval_baselines.py  # scripted vs random, 100 episodes each
uv run python scripts/watch.py --seed 1  # play a game and open its replay in your browser
uv run python scripts/film.py --seed 10033  # film a game in first-person pixels, MP4 in runs/films/

uv run python scripts/train_ppo.py cartpole     # PPO correctness check (solves in ~5 min on CPU)
uv run python scripts/train_ppo.py lunarlander  # the harder check
uv run python scripts/train_ppo.py nacht-state  # the real thing: PPO on NachtSim's state vector
uv run python scripts/curve.py runs/<run>                      # learning curve so far
uv run python scripts/eval_policy.py runs/<run>/checkpoint.pt
uv run python scripts/watch.py --checkpoint runs/<run>/checkpoint.pt
```

Learning from real play, from either direction (details in
[`docs/demos.md`](./docs/demos.md)):

```sh
# Windows: record yourself playing, frames paired with your own raw mouse counts.
uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20

# Anywhere: any footage of the game, no input log needed.
uv run python scripts/ingest_video.py ~/Videos/waw/*.mp4 --out data/clips/session1
uv run python scripts/train_idm.py data/demos --out runs/idm1        # what did they press between these frames?
uv run python scripts/label_clips.py runs/idm1/idm.pt data/clips/session1
uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1
uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos --episodes 20
uv run python scripts/watch.py --checkpoint runs/bc1/bc.pt           # watch what it learned

# No game yet? The same recorder path, driven by NachtSim, for labelled clips today.
uv run python scripts/record_demo.py --source sim --episodes 20 --counts-per-degree 10
```

On the machine that runs the game (Linux; see [`docs/linux.md`](./docs/linux.md)
for why, and for the udev and libinput setup), the M0 spikes are two commands:

```sh
uv run python scripts/calibrate_mouse.py --window "World at War" --full-turn  # S1 + S4
uv run python scripts/spike_capture.py --window "World at War" --latency      # S2 + S3
```

The first answers whether the engine sees synthetic input at all and measures
mouse counts per degree; the second measures capture rate and the closed-loop
delay the whole delayed-MDP design is built around.

Training runs write `config.json`, `metrics.jsonl` (one line per update), and
`checkpoint.pt` to `runs/<run>/`.

`watch.py` writes a self-contained replay to `runs/replays/`: a top-down map with
play/pause, speed, scrubbing, what the agent could and couldn't see, its action
each step, and a clickable match log. Add `#t=90` to the file's URL to open it at
90 seconds in. `--agent random` shows the baseline dying.

## The shape of it

- **Screen capture only.** Pixels in, keyboard/mouse out. Reward and episode
  boundaries are read off the HUD.
- **Real-time, single instance.** ~54,000 agent steps/hour from the real
  game — where an Atari DQN baseline assumes 10,000,000. A faithful simulator
  built from the game's own source constants, from-scratch PPO and DQN, and
  behavioral cloning from human play are how the plan closes that gap.
- **Human play is the other half of the data problem.** Recorded demonstrations
  and ordinary gameplay video are the only source of real experience that does
  not cost real time at 15 decisions a second — an inverse dynamics model turns
  unlabelled footage into training data, and behavioural cloning turns that into
  the policy RL starts from.
- **Learning RL is the point.** Final bot skill is secondary to a clean
  environment, fast iteration, and swappable algorithms.
- **Hardware:** RTX 5070, 12GB, trained locally, on Linux (the game runs under
  Proton as an XWayland client; synthetic input is a kernel virtual device).

See `PLAN.md` for everything else.
