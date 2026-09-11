# ZombiesAI

A reinforcement-learning agent learning to survive *Nacht der Untoten* — Call
of Duty: World at War's Nazi Zombies map — by screen capture and synthetic
keyboard/mouse input only, the way a person would. No mod tools, no memory
reading, no game scripting.

## Status: planning

No code yet — this repo currently holds the implementation plan. The full
plan, including architecture, environment spec, reward design, the milestone
ladder (M0–M7), and ranked risks, is in [`PLAN.md`](./PLAN.md), and as a
formatted page: **[Undead Loop](https://claude.ai/code/artifact/9feeeb1c-8561-4347-912f-2fb98c9c441b)**.

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
