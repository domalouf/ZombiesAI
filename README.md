# ZombiesAI

[![tests](https://github.com/domalouf/ZombiesAI/actions/workflows/tests.yml/badge.svg)](https://github.com/domalouf/ZombiesAI/actions/workflows/tests.yml)

An agent learning to survive *Nacht der Untoten* — Call of Duty: World at War's
Nazi Zombies map — from the game's pixels and sound alone, with synthetic
keyboard and mouse as its hands, the way a person plays. No mod tools, no memory
reading, no game scripting, no simulator: every frame it learns from is the real
game's.

It runs on Linux. The game runs under Proton (Plutonium's T4 client), and each
gaming PC runs several copies at once, each in an X server of its own, so the
agent can play many games in parallel while you keep using the desktop.

## How it learns

1. **Behavioural cloning from real play.** Record yourself playing, with your
   raw mouse counts and keys logged against the frames (and the game's audio);
   label ordinary gameplay video nobody logged input for with an inverse
   dynamics model trained on those recordings; clone a pixel policy from both.
   Taking the controls back while a policy plays records corrections too.
   [`docs/demos.md`](./docs/demos.md)
2. **PPO fine-tuning on many games at once.** The cloned policy plays several
   real game instances per PC, the HUD scores it, and a PPO learner with a
   critic warm-up and a KL anchor to the clone moves it toward whatever scored.
   Any other gaming PC that is free joins the same run as a worker, playing its
   own games for the one learner. [`docs/rl.md`](./docs/rl.md)

The reward is read off the HUD ([`docs/hud.md`](./docs/hud.md)); the game's own
round counter is the number that matters, because it cannot be reward-hacked.
Constraints, reward design, risks and milestone status are in
[`PLAN.md`](./PLAN.md); running the game, the capture and the virtual input on
Linux is [`docs/linux.md`](./docs/linux.md).

## What's here

- `src/zombiesai/spec.py` — the versioned contract: the factored action space,
  the observation layout, and a `SPEC_VERSION` hash every clip and checkpoint is
  stamped with.
- `src/zombiesai/demos/` — learning from real gameplay: the demo recorder (X11
  capture, evdev input, game audio), video ingest, the inverse dynamics model,
  hearing, and behavioural cloning.
- `src/zombiesai/hud/` — reading points, round, grenades, ammo and the held weapon off the HUD.
- `src/zombiesai/realgame/` — the game as an environment: the agent's hands
  (uinput, and XTEST into one instance's X server), the instances themselves,
  the HUD reward, console and reset handling, and the live player.
- `src/zombiesai/rl/` — PPO fine-tuning: actors (one per game), the learner, and
  the fleet protocol that lets other PCs play for it.
- `src/zombiesai/session.py` — `./zai start | status | stop`: a training session's games, trainer and
  cleanup in one command each.
- `src/zombiesai/viz/` — the Training Room dashboard, the live site, the
  supervision page and the stream overlay.
- `src/zombiesai/synthetic.py` — a stand-in for the game when there is no game:
  first-person frames that answer to the agent's actions, on the interfaces the
  real game has. It keeps the recorder, BC, the actors, the learner and the fleet
  testable in CI and lets a PC with no game rehearse the RL plumbing. Nothing
  learned on it is meant to transfer.

## Training: start and stop

Training on this PC is three commands, run from the top of the checkout:

```sh
./zai start      # bring the games up and start training in the background
./zai status     # how the run is doing
./zai stop       # save everything and close everything
```

The game, Plutonium and the fleet need a one-time setup first: [`docs/linux.md`](./docs/linux.md) and
[`docs/rl.md`](./docs/rl.md) ("Setup, once").

### Starting a run

`./zai start` does, in order:

1. **Checks nothing is already training.** If a run is already playing the games, it says so and stops.
   Run `./zai stop` first.
2. **Picks what to start from.** By default it continues the newest real-game run's `checkpoint.pt`.
   If there is none, it uses the newest `bc.pt`, and failing that a fresh policy. Continuing loses
   nothing: the update count, the KL weight and the anchor to the BC policy all carry over.
3. **Brings the games up**, one at a time, waiting for each window. This takes a minute or so per game.
   Games that are already running are kept.
4. **Starts the trainer in the background** as the next free `runs/rl<N>`, with its log in
   `runs/rl<N>.log`. It runs in a session of its own, so closing the terminal does not stop it.

Options:

```sh
./zai start --games 6                    # play six games (remembered for next time)
./zai start --games auto                 # as many games as this PC can carry (see "How many games")
./zai start --from runs/bc1/bc.pt        # start from this checkpoint instead; --from fresh for a new policy
./zai start --name rl-lr1e4 -- --lr 1e-4 # choose the run's name; anything after -- goes to train_rl.py
./zai start --watch                      # also open the game viewers on Hyprland workspace 9
./zai start -- --env synthetic           # a rehearsal of the plumbing with no game: nothing is launched
```

### How many games

Without `--games`, `start` plays the same number of games as last time: the fleet's size in
`runs/instances/fleet.json`. `--games N` changes it, and the new number is kept.

With `--games auto`, or the first time when there is no fleet yet, the PC is measured instead
(`src/zombiesai/rl/capacity.py`, the same as `train_rl.py --actors auto`). CPUs, free RAM and free VRAM
each give a limit, after setting aside a share for the desktop and the learner. The smallest limit wins,
and the total is capped at 8, the most ever tried on one PC:

| resource | each game costs | set aside | this PC (2026-10-04) |
|---|---|---|---|
| CPU | 1 logical CPU (game + its actor) | 2 desktop + 2 learner | 16 CPUs: 12 games |
| RAM | 1.3 GB game + 0.7 GB actor | 4 GB desktop + 4 GB learner | 24 GB free: 8 games |
| VRAM | 1 GB (1440p under DXVK) | 1.5 GB desktop + 2.5 GB learner | 10.2 GB free: **6 games** |

The per-game costs are estimates from earlier runs, not a benchmark. If the run's late-step share
(`bad` in the log and on the dashboard) climbs past about 10%, the PC is overloaded: use fewer games.

### Watching a run

```sh
./zai status                                     # update, steps/s, mean round and return; games up
tail -f runs/rl<N>.log                           # one line per PPO update and per finished game
uv run python scripts/view_instances.py          # the games themselves, on workspace 9
uv run python scripts/supervise.py               # the supervision page: curves, games, start and stop
```

The run's best game so far is kept as `runs/rl<N>/best/best.mp4`.

### Stopping a run

`./zai stop` saves first, then closes everything a training session uses on this PC:

1. **The trainer stops gracefully.** Every `train_rl.py` and fleet worker is asked to stop, as Ctrl-C
   does. The learner finishes the step it is on and **writes `checkpoint.pt` before anything else**,
   then stops its actors. Every key is released, and a game that has just ended can finish encoding
   as the run's best film. This normally takes a few seconds. A trainer that is still running after
   `--timeout` (300 s) is killed, keeping its last periodic checkpoint.
2. **The games come down**, with their X servers (Xwayland, headless weston) and per-game sound sinks.
3. **Everything else of ours is closed:** the game viewers, the supervision dashboard, and anything
   left behind, such as an actor orphaned by a crash. Processes are matched by command line,
   environment and parent, so the desktop's own Xwayland, a BC training run and other programs are
   never touched.
4. **It reports what was saved** for each run (checkpoint age, updates, steps, games, best round) and
   ends with `all stopped`. If anything would not stop, it lists it and exits non-zero.

What a stopped run leaves in `runs/rl<N>/`: `checkpoint.pt` (continue from it, or play it with
`play_real.py`), `metrics.jsonl`, `episodes.jsonl`, `config.json` and `best/`.

`./zai stop --keep-games` stops only the trainer and leaves the games up, so the next `start` skips the
wait for them to load.

### Good to know

- `./zai stop` is safe to run when nothing is running, and `./zai status` never changes anything.
- The trainer saves its checkpoint however it is stopped: `./zai stop`, Ctrl-C, `kill`, a closed
  terminal, or the dashboard's Stop button. The only exception is `kill -9`, which leaves the last
  periodic checkpoint.
- The site's reporter (`zombiesai-live.service`) is left running and shows the PC as idle. To stop it
  too, run `systemctl --user stop zombiesai-live`.
- `./zai` runs this checkout's `.venv` directly (a worktree without one borrows the main checkout's)
  and never runs `uv sync`.
- The individual scripts (`instances.py`, `train_rl.py`, `view_instances.py`) still work on their own
  when you need finer control; see [Commands](#commands).

## Commands

```sh
uv sync                                   # Python 3.13 env + deps (PyTorch from the CUDA 12.8 index)
uv run pytest                             # the test suite; no game needed

# Learning from real play (docs/demos.md, docs/linux.md for the one-time setup)
uv run python scripts/calibrate_mouse.py --window "World at War" --full-turn   # counts per degree
uv run python scripts/record_demo.py --counts-per-degree 9.09 --minutes 20 --wait --audio
uv run python scripts/ingest_video.py ~/Videos/waw/*.mp4 --out data/clips/session1
uv run python scripts/train_idm.py data/demos --out runs/idm1        # what did they press between these frames?
uv run python scripts/label_clips.py runs/idm1/idm.pt data/clips/session1
uv run python scripts/train_bc.py data/clips/session1 data/demos --out runs/bc1
uv run python scripts/eval_bc.py runs/bc1/bc.pt --clips data/demos-heldout
uv run python scripts/play_real.py runs/bc1/bc.pt --minutes 3       # watch it play; F7 hands it the controls

# Then PPO on several games at once (docs/rl.md); ./zai start / ./zai stop (above) do this in one step
uv run python scripts/instances.py up --n 4                          # four games, four private X servers
uv run python scripts/spike_instances.py --counts-per-degree 9.09   # does each take input, alone?
uv run python scripts/train_rl.py runs/bc1/bc.pt --actors auto --out runs/rl1   # as many games as this PC carries
uv run python scripts/train_rl.py fresh --actors auto --out runs/rl0          # no BC: a new pixel+audio policy
uv run python scripts/train_rl.py runs/bc1/bc.pt --env synthetic --actors 8   # rehearse the plumbing, no game
uv run python scripts/instances.py down

# Watching it learn
uv run python scripts/curve.py runs/rl1          # one run's learning curve, in the terminal
uv run python scripts/dashboard.py --watch 30    # every run, as a page: curves, health, gates
uv run python scripts/supervise.py               # the live supervision page: runs, games, start and stop
```

More games than one PC can run: the other gaming PCs join the same run as **workers** (`scripts/fleet_worker.py`),
sending their games to the learner over the LAN. A worker needs only the fleet token: it finds the learner by
its LAN beacon, starts as many games as its PC can carry, and with `--yield-to-games` steps aside while someone
plays on that PC (`deploy/zombiesai-worker.service` runs it that way). The learner turns away any PC on a different commit or with
different game settings; `scripts/fleet.py prep` brings every PC to this one's commit and settings and starts
its games, over SSH, and the learner scores each machine's games on their own so one bad PC stands out. See
[`docs/rl.md`](./docs/rl.md), "Several PCs".

Training runs write `config.json`, `metrics.jsonl` (one line per update), and
`checkpoint.pt` to `runs/<run>/`.

`dashboard.py` reads all of them and writes `runs/dashboard.html`: **how training
is going**, in one self-contained page. Every run PPO, BC or the IDM wrote —
return and round reached, entropy, KL, clip fraction, explained variance, value
loss and throughput — each curve a bucket mean over the spread it was averaged
out of, so nothing is smoothed away silently. Charts are scoped to one
environment at a time, because a rehearsal's return and the real game's are not
the same number. It also grades each run against the gates in `PLAN.md`: the
M7 anti-hacking pair (no reward term above 60% of return, repairs below 25% of
points), and says which way the headline metric is
actually moving and whether that beats its own noise. It needs nothing but the
standard library, so it also runs on a `runs/` directory copied off the
training box.

For the website, `--site` builds it as a static
directory (`index.html` + `fonts/`) with the fonts as files rather than `data:`
URIs, which the site's `default-src 'self'` CSP refuses, and with the local paths
a config carries (clip filenames, output directories) stripped out. Nothing is
served or fetched at view time: it is a snapshot of the numbers as they stood
when the page was built, and it says so. `deploy/deploy.sh` publishes it to
[domalouf.com/zombies/training/](https://domalouf.com/zombies/training/)
beside the stream's page.

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

Each run's best game goes up too, as the video its actors kept
(`live/best/<run>.mp4`), shown under "Best games" with its round, points and
kills. A film is pushed when it changes, on its own so the 5 s reports never
wait on it, and the page links it only once it is on the site. Long games make
films of hundreds of MB, one per run.

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
show up on the same page, one collapsible panel each below the training PC's. The
summary says live or offline, CPU, GPU and RAM, and what its worker is doing (waiting
for a run, or playing N games for which run, segments sent, best round); open it for
that machine in full. The training PC starts open, the others closed, and the page
remembers what each reader opened. Each PC pushes only its own `live/machine-<id>.json`, never `runs.json` or
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
rehearsal), or `--stream-run <name>`. Averages are over the last 100 games;
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

## The shape of it

- **Pixels and sound in, keyboard and mouse out.** Reward and episode boundaries
  are read off the HUD.
- **Real time.** One game gives ~54,000 decisions an hour at 15 a second, where
  an Atari DQN baseline assumes 10,000,000. Human play (recorded or just
  watched) is the cheapest real experience there is, so behavioural cloning
  gives RL its starting point; several games per PC and several PCs per run
  give it the volume.
- **One model, whichever PCs are free.** Every gaming PC can contribute games to
  the same PPO run, and the learner refuses any whose code or game settings
  differ, because a different sensitivity or binding would quietly change what
  an action means.
- **Hardware:** RTX 5070, 12 GB, on Linux (the game runs under Proton as an
  XWayland client; synthetic input is XTEST into each instance's own X server,
  or a kernel virtual device for the live player).
