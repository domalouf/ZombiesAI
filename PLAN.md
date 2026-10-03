# Nacht der Untoten RL Agent — Plan

The original 700-line plan (Windows host, the NachtSim simulator, from-scratch gym PPO) is in git history at tag
`pre-linux-pixels`. This is what is still true and what is next.

## The goal

A policy that plays a good game of *Call of Duty: World at War*'s Nacht der Untoten from **screen pixels and game
audio only**, with synthetic keyboard and mouse out — no mod tools, no memory reading, no game scripting. It is
trained on the owner's own gaming PCs, all Linux.

## Constraints that drive everything

1. **Real time is the data budget.** One game gives 15 decisions a second, ~54k an hour. The answers are
   parallelism and priors, never a simulator:
   - **Many games per PC.** Under Proton each game is an X11 client, so each instance gets its own rootful
     Xwayland under a headless Weston, private XTEST input, MIT-SHM capture and its own audio sink
     (`realgame/instances.py`, `docs/rl.md`). The per-step CPU cost of an actor is the ceiling on how many fit
     (`docs/linux.md`, "Per-step cost").
   - **Many PCs per run.** Any gaming PC that is free runs a fleet worker and feeds the same PPO learner
     (`rl/fleet.py`, `docs/rl.md`, "Several PCs").
   - **Human play first.** Recorded demos with raw input, and ordinary gameplay video labelled by an inverse
     dynamics model, train a behavioural-cloning policy that RL starts from (`docs/demos.md`).
2. **The reward is a computer-vision artifact.** Points, round and death are read off the HUD
   (`hud/`, `docs/hud.md`); a misread must become a rejected change, never a reward. `round_reached` is the
   number that matters — it cannot be reward-hacked; the shaped return can.
3. **One contract.** `spec.py` fixes the action space, observation layout and HUD fields, and its hash
   (`SPEC_VERSION`) is stamped on every clip and checkpoint. Never change it casually: every recording on every
   PC is refused under a new one.

## The pipeline

```
demos (raw input) ─┐
video ── IDM labels ┴─► BC (pixels + audio) ─► PPO fine-tune, KL-anchored to BC ─► checkpoint
                                                 ▲
                         actors: N games per PC × M PCs (fleet workers)
```

- **Observation:** a strided stack of 128×72 RGB frames (area-averaged, never nearest) plus a stereo log-mel of
  the last half second of the game's own audio (`demos/hearing.py`). Audio is on by default for new models:
  zombies are heard before they are seen, and most deaths come from behind.
- **Action:** factored — strafe, forward, yaw bin, pitch bin, fire, ADS, sprint, one button.
- **Reward:** settled point gains, rounds, purchases (novelty-gated), damage and death, with repairs capped so
  board farming cannot dominate (`reward.py`, `realgame/hud_reward.py`). Dashboard gates: no term above 60% of
  the return, repairs under 25% of points.
- **RL:** PPO with a critic warm-up and a decaying KL penalty to the BC prior (VPT's recipe); bad steps stay in
  GAE but out of the losses; segments more than `max_policy_lag` versions stale are dropped.

`src/zombiesai/synthetic.py` is a stand-in that only exists so all of this can be tested, and rehearsed with
`train_rl.py --env synthetic`, without the game. Nothing learned on it is meant to transfer.

## Status (October 2026)

Done: instances in parallel on one PC, the real-game env with HUD reward and reset FSM, BC from demos and
labelled video, PPO fine-tuning across several PCs, the live dashboard, site and stream overlay. Linux-only
cleanup and a ~4.5x cheaper actor step (1440p) landed on branch `linux-pixels-fleet`.

Also merged: the PPO update staged on the GPU once with mixed precision; RL from a fresh pixel+audio network
(`train_rl.py fresh`); LAN discovery of the learner, `--actors auto`, workers that start their own games and step
aside while someone is gaming.

Still on its own branch (`~/Projects/zai-wt/actor-runtime`, early, not wired in): one inference process per PC so
actors do not each load torch (~600 MB each). Not done at all: the BC data-loader speedups.

## Next

1. Finish the per-PC inference process; speed up the BC data loader.
2. A damage detector (the red vignette) so being hit costs something before it kills.
3. The prompt reader (door / weapon / ammo prices) for ammo rebuys and affordance signals.
4. The eval gate on the real game: `round_reached` of the frozen policy over 20 games against BC's.
5. Memory beyond the frame stack (a recurrent core) once the feed-forward policy plateaus.

## Risks

| Risk | Mitigation |
|---|---|
| HUD misreads become reward | tracker settles changes; `hud_lost` ends the episode; anti-hacking gates on the dashboard |
| Late steps on a busy PC | per-machine `bad_step_frac` in metrics; fewer actors there; the per-step cost work |
| RL wrecks the BC prior | critic warm-up, KL anchor, `kl_ref` on the dashboard |
| PCs on different code or settings | the learner refuses a worker on another commit, spec or `config.cfg` |
| The game crashes or hangs | reset FSM with relaunches; actors restarted up to 20 times each |
