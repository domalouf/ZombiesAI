# ZombiesAI

A reinforcement-learning agent learning to survive *Nacht der Untoten* — Call
of Duty: World at War's Nazi Zombies map — by screen capture and synthetic
keyboard/mouse input only, the way a person would. No mod tools, no memory
reading, no game scripting.

## Status: M2 (from-scratch PPO) in progress

The full plan, including architecture, environment spec, reward design, the
milestone ladder (M0–M7), and ranked risks, is in [`PLAN.md`](./PLAN.md), and
as a formatted page: **[Undead Loop](https://claude.ai/code/artifact/9feeeb1c-8561-4347-912f-2fb98c9c441b)**.

Built so far (all Linux, no game needed):

- `src/zombiesai/spec.py` — the versioned contract: factored action space, the
  48-action compact profile, HUD/state layouts, and a `SPEC_VERSION` hash that
  every checkpoint and episode is stamped with.
- `src/zombiesai/sim/` — **NachtSim** in state mode: ground-truth round
  mechanics from the game script, domain-randomized guesses for everything
  else, latency/action-repeat/dropout randomization, and HUD noise. What it
  gets wrong is listed in [`docs/sim_lies.md`](./docs/sim_lies.md).
- `src/zombiesai/reward.py` — the shaped reward, term by term, with the gain
  cap, novelty gating, and repair cap.
- `src/zombiesai/store/` — the episode store (crash-tolerant, spec-checked).
- `src/zombiesai/agents/` — random and scripted baselines.
- `src/zombiesai/rl/` — **PPO from scratch**: GAE with correct truncation
  bootstrapping, the clipped surrogate, a factored policy with one categorical
  per action head, and checkpoints stamped with `SPEC_VERSION`.

```sh
uv sync                                  # Python 3.13 env + deps (PyTorch: CPU on Linux, CUDA 12.8 on Windows)
uv run pytest                            # test suite
uv run python scripts/bench_sim.py       # steps/s and multi-process scaling
uv run python scripts/eval_baselines.py  # scripted vs random, 100 episodes each
uv run python scripts/watch.py --seed 1  # play a game and open its replay in your browser

uv run python scripts/train_ppo.py cartpole     # PPO correctness check (solves in ~5 min on CPU)
uv run python scripts/train_ppo.py lunarlander  # the harder check
uv run python scripts/train_ppo.py nacht-state  # the real thing: PPO on NachtSim's state vector
uv run python scripts/train_ppo.py nacht-state --start-rounds 1 5  # curriculum: train from rounds 1-5 (eval stays at 1)
uv run python scripts/curve.py runs/<run>                      # learning curve so far
uv run python scripts/eval_policy.py runs/<run>/checkpoint.pt
uv run python scripts/watch.py --checkpoint runs/<run>/checkpoint.pt
```

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
- **Learning RL is the point.** Final bot skill is secondary to a clean
  environment, fast iteration, and swappable algorithms.
- **Hardware:** RTX 5070, 12GB, trained locally.

See `PLAN.md` for everything else.
