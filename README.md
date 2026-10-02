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
- **RL on the real game, several games at once** — each World at War instance
  in an X server of its own (a hidden rootful Xwayland) with private XTEST input
  (`realgame/instances.py`, `realgame/xtest.py`); the game as an environment
  whose reward comes off the HUD (`realgame/env.py`, `realgame/hud_reward.py`);
  and PPO fine-tuning of the BC policy with a critic warm-up and a KL anchor to
  BC, one actor process per game (`rl/parallel_ppo.py`). See
  [`docs/rl.md`](./docs/rl.md).

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
uv run python scripts/train_ppo.py nacht-state --start-rounds 1 5  # curriculum: train from rounds 1-5 (eval stays at 1)
uv run python scripts/curve.py runs/<run>                      # one run's learning curve, in the terminal
uv run python scripts/dashboard.py                            # every run, as a page: curves, health, gates
uv run python scripts/dashboard.py --watch 30                 # ...rebuilt every 30s while a run trains
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

Then let it teach itself, with several games playing at once (details in
[`docs/rl.md`](./docs/rl.md)):

```sh
uv run python scripts/train_rl.py runs/bc1/bc.pt --env sim --actors 8   # rehearse the pipeline on NachtSim
uv run python scripts/instances.py up --n 4                             # four games, four private X servers
uv run python scripts/spike_instances.py --counts-per-degree 9.09      # does each take input, alone?
uv run python scripts/train_rl.py runs/bc1/bc.pt --actors 4 --counts-per-degree 9.09 --out runs/rl1
uv run python scripts/instances.py down
```

More games than one PC can run: the other gaming PCs join the same run as **workers** (`scripts/fleet_worker.py`),
sending their games to the learner over the LAN. The learner turns away any PC on a different commit or with
different game settings. See [`docs/rl.md`](./docs/rl.md), "Several PCs".

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

`dashboard.py` reads all of them and writes `runs/dashboard.html`: **how training
is going**, in one self-contained page. Every run PPO, BC or the IDM wrote —
return and round reached, entropy, KL, clip fraction, explained variance, value
loss and throughput — each curve a bucket mean over the spread it was averaged
out of, so nothing is smoothed away silently. Charts are scoped to one
environment at a time, because a CartPole return and a NachtSim return are not
the same number. It also grades each run against the gates in `PLAN.md`: the
M7 anti-hacking pair (no reward term above 60% of return, repairs below 25% of
points) and M2's solved-at returns, and says which way the headline metric is
actually moving and whether that beats its own noise. It needs nothing but the
standard library, so it also runs on a `runs/` directory copied off the
training box.

For the website, `--site` builds it the way the replay page is built — a static
directory (`index.html` + `fonts/`) with the fonts as files rather than `data:`
URIs, which the site's `default-src 'self'` CSP refuses, and with the local paths
a config carries (clip filenames, output directories) stripped out. Nothing is
served or fetched at view time: it is a snapshot of the numbers as they stood
when the page was built, and it says so. `deploy/deploy.sh` publishes it to
[domalouf.com/zombies/training/](https://domalouf.com/zombies/training/)
alongside the replay.

```sh
uv run python scripts/dashboard.py --site site/zombies/training
```

### Live on the site

`--site ... --live` (what `deploy/deploy.sh` builds) makes that page live: the
training PC pushes two files into the site's `zombies/training/live/` — the
machine as `viz/system.py` samples it, and the runs training right now, every
5 s (`machine.json`); every run's curves every minute (`runs.json`) — and the
page polls them. It shows "Training now", the machine (CPU, GPU, temperatures,
memory, disks, network, the busiest processes by name), and every run, and says
**offline** when the PC stops reporting. The PC only ever connects out; nothing
on it is opened to the network. Both files go through the same scrub as the
site build, plus: no host name, no process ids, users or command lines.

One-time setup, with the site on `lts`:

```sh
# 1. On the training PC: a key that can do one thing.
ssh-keygen -t ed25519 -f ~/.ssh/zombies_live -N '' -C zombies-live
cat >> ~/.ssh/config <<'EOF'
Host zombies-live
    HostName lts.lan
    User <your user on lts>
    IdentityFile ~/.ssh/zombies_live
    IdentitiesOnly yes
EOF

# 2. On lts: the directory, and the key allowed to write into it and nowhere else
#    (rrsync ships with rsync; -wo is write-only).
mkdir -p ~/site/www/zombies/training/live   # the site's web root (MyWebsite: server/.env SITE_WEB_ROOT)
echo "command=\"rrsync -wo $HOME/site/www/zombies/training/live\",restrict $(cat zombies_live.pub)" \
  >> ~/.ssh/authorized_keys

# 3. Back on the PC: try one push, then keep it running.
uv run python scripts/publish_live.py --once --dest zombies-live:
cp deploy/zombiesai-live.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now zombiesai-live
deploy/deploy.sh    # the page itself; it leaves training/live/ alone
```

nginx serves the two files as it does any static file, and the page asks for
them with `cache: "no-store"`. If a CDN sits in front, keep it from caching
`/zombies/training/live/`.

**The other gaming PCs** (the ones playing for the learner, `scripts/fleet_worker.py`)
show up on the same page, one card each beside the training PC: live or offline,
CPU, GPU and RAM, and what its worker is doing (waiting for a run, or playing N
games for which run, segments sent, best round). Click a card for that machine in
full. Each PC pushes only its own `live/machine-<id>.json`, never `runs.json` or
`stream.json`; nothing is relayed through the learner, so a PC shows up between
runs too. The id is the file name on the site (a-z, 0-9, `-`); the label is what
the page calls it, and the host name is never published.

```sh
# On each such PC: steps 1-3 above, with a key of its own (one more line in lts's
# authorized_keys, the same rrsync -wo directory), then say which machine it is:
mkdir -p ~/.config/zombiesai
printf 'ZOMBIES_LIVE_WORKER=rig2\nZOMBIES_LIVE_LABEL=Gaming PC 2\n' > ~/.config/zombiesai/live.env
systemctl --user restart zombiesai-live

# On the training PC: rebuild the page with the PCs' ids (a static site cannot list a directory).
LIVE_MACHINES="rig2" deploy/deploy.sh
```

### The Twitch stream

`domalouf.com/zombies/live/` is the stream's page: the Twitch player, the
overlay's four numbers beside it, and the PPO run under it — progress, steps,
games, throughput, the round, survival and accuracy curves over its games, the
policy's health (entropy, KL, clip fraction, explained variance, KL to the
human prior), its last games and where they end. `/zombies/live/overlay/` is
the overlay alone on a transparent page, for OBS: **best round, average round,
shot accuracy and average survival time**, in a small panel over the game.

Both poll `training/live/stream.json`, which `publish_live.py` now pushes
beside the other two (same key, same directory). Its games are the run's
`episodes.jsonl`: one line per finished game, written by `train_rl.py`
(`rl/parallel_ppo.py`). The run is the one training now (the real game before a
sim rehearsal), or `--stream-run <name>`. Averages are over the last 100 games;
the best round is the run's. On the real game, accuracy is read off the HUD:
magazine marks that went while the trigger was held, against the hits the
points counter paid for — two hits inside one counter roll settle as one gain,
so it reads low.

```sh
TWITCH_CHANNEL=<channel> deploy/deploy.sh          # builds /zombies/live/ with the player
uv run python scripts/build_stream.py --demo        # made-up games, to look at the layout locally
```

In OBS: add a **Browser** source, 1920×1080, URL
`https://domalouf.com/zombies/live/overlay/`, on top of the game capture. The
query string places it: `?corner=tl|tr|bl|br` (default `tl`, clear of Nacht's
HUD), `?scale=1.25`, `?layout=column`, `?inset=28`, and `?demo=1` to position
it before there are games.

The site's CSP has to let the player in: add `frame-src https://player.twitch.tv`
(in MyWebsite's nginx config). Twitch also checks the page's host against the
`parent` the page sends, which is the host it is served from.

`watch.py` writes a self-contained replay to `runs/replays/`: a top-down map with
play/pause, speed, scrubbing, what the agent could and couldn't see, its action
each step, and a clickable match log. Add `#t=90` to the file's URL to open it at
90 seconds in. `--agent random` shows the baseline dying.

## The shape of it

- **Screen capture only.** Pixels in, keyboard/mouse out. Reward and episode
  boundaries are read off the HUD.
- **Real-time.** ~54,000 agent steps/hour from one real game — where an
  Atari DQN baseline assumes 10,000,000. A faithful simulator built from the
  game's own source constants, from-scratch PPO and DQN, and behavioral cloning
  from human play are how the plan closes that gap — and on Linux, several
  games at once, each in its own X server (`docs/rl.md`).
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
