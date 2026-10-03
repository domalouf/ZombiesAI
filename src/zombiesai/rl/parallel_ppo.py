"""Reinforcement learning on the policy's own play: PPO fine-tuning of the behavioural-cloning policy, fed by
several games at once.

The shape of it:

* **Actors** -- one process per game instance (`realgame/instances.py`), or per synthetic stand-in for a
  rehearsal (`zombiesai/synthetic.py`).
  Each runs its own copy of the policy on a CPU thread (0.5 ms a decision for the BC net, so the games and the
  learner have the GPU to themselves), acts in real time, and ships **segments** of up to `segment_steps`
  decisions to the learner. A segment ends early at an episode's end, so no segment spans a reset, and it
  carries the frames before its first step that the frame stack reaches back into -- the learner builds each
  step's stack by index arithmetic, frames stored once (PLAN.md, "Frames stored once, not stacked").
* **The learner** gathers `batch_steps` decisions, recomputes values with its current network (actors' values
  are up to `max_policy_lag` versions old), does GAE per segment -- bootstrapping from V(last frame) unless
  the episode was *terminated* -- and runs the clipped PPO update. The behaviour log-probs the actors recorded
  are the ratio's denominator, which is what makes a slightly stale segment still a correct sample. Segments
  more than `max_policy_lag` versions stale are dropped rather than corrected. New weights go to
  `weights.pt` (atomically); actors pick them up between segments, never inside a tick.
* **Where it starts matters more than anything here.** It starts from a BC checkpoint, and two things keep
  RL from wrecking what BC learned before it has learned anything itself:

  - **A critic warm-up.** The BC value head was fitted (if at all) to sim returns, not to this reward, so for
    the first `critic_warmup_updates` updates only the critic head trains, on a frozen encoder, while the
    policy is left exactly as cloned. Advantages from a random critic are noise, and noise is what destroys
    a pretrained policy fastest.
  - **A KL penalty toward the BC policy**, `kl_coef * KL(pi || pi_BC)` summed over the action heads, decaying
    to `kl_min`. This is how VPT fine-tuned a behavioural prior with RL without forgetting it (Baker et al.
    2022): the policy is free to improve on the human where reward says so, and pulled back everywhere else.

* **Bad steps** (overruns, a frozen picture, the game unfocused) stay in the GAE chain -- time still passed --
  but are masked out of the policy and value losses: nobody chose what happened in them.
* Rewards are divided by a running estimate of the discounted return's standard deviation (`reward_scale`),
  the usual PPO normalisation; the shaper's clip already bounds a single step.

Everything is logged to `metrics.jsonl` in the names `viz/dashboard.py` already charts for PPO (return,
round reached, entropy, KL, clip fraction, explained variance, the reward-hacking shares), plus `kl_ref`, the
distance from the BC policy. Every finished game is also a line of `episodes.jsonl` -- its round, how long it
lived, how many shots it fired and landed -- which the stream overlay is built from (viz/stream.py).
`checkpoint.pt` is a BC-format checkpoint with an `rl` section, so everything that plays a BC policy --
`play_real.py`, `eval_bc.py`, `watch.py` -- plays the fine-tuned one unchanged.
"""

import json
import math
import multiprocessing as mp
import numbers
import os
import queue as queue_mod
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec

# The pieces live in their own modules; their names are re-exported here, where the fleet, the scripts and the
# tests have always imported them from.
from zombiesai.rl.actors import (  # noqa: F401
    _actor_loop,
    _episode_writer,
    _offer,
    actor_main,
    make_actor_env,
    make_real_env,
)
from zombiesai.rl.config import RLConfig, resolve_device  # noqa: F401
from zombiesai.rl.learner import Learner, RunningStd, compute_gae, kl_to_reference, root_prior  # noqa: F401
from zombiesai.rl.segments import FrameHistory, Segment, stack_indices  # noqa: F401
from zombiesai.rl.weights import WeightFollower, publish  # noqa: F401


# What episodes.jsonl keeps of an actor's episode summary: the numbers, not the per-term breakdowns.
EPISODE_FIELDS = ("actor", "episode", "reason", "return", "length", "seconds", "round_reached", "shots", "hits",
                  "points_gained", "kills", "end_points", "end_kills", "end_headshots", "bad_steps", "repair_share",
                  "max_term_share")


class MachineStats:
    """Each machine's share of the batches, so one PC feeding the learner worse data than the rest stands out
    instead of being averaged away -- a slower GPU means more late steps, a different install other rounds.
    rl/fleet.py numbers machine k's actors from k * block; this machine's are 0..n-1, machine 0. Keyed by that
    number, never a name: the last metrics row is published with the site's training page."""

    BAD_STEP_WARN = 0.10  # docs/rl.md: under 5% is normal since the BLAS fix; twice that is a machine in trouble
    BAD_STEP_OK = 0.05
    MIN_STEPS = 256  # fewer steps than a segment says nothing

    def __init__(self, block: int, recent: int = 50):
        self.block = block
        self.window: dict[int, dict] = {}
        self.recent: dict[int, deque] = {}
        self._recent_len = recent
        self.warned: set[int] = set()

    def _w(self, actor: int) -> dict:
        return self.window.setdefault(actor // self.block, {"segments": 0, "dropped": 0, "steps": 0, "bad": 0,
                                                            "lag": 0, "episodes": 0})

    def segment(self, seg: "Segment", lag: int, accepted: bool) -> None:
        w = self._w(seg.actor)
        if not accepted:
            w["dropped"] += 1
            return
        w["segments"] += 1
        w["steps"] += seg.n
        w["bad"] += int(seg.bad.sum())
        w["lag"] += lag

    def episode(self, actor: int, payload: dict) -> None:
        self._w(actor)["episodes"] += 1
        self.recent.setdefault(actor // self.block, deque(maxlen=self._recent_len)).append(payload)

    def flush(self) -> dict[str, dict]:
        """Since the last flush: segments taken and dropped as too stale, steps, the share of them late, the
        mean lag; over each machine's last `recent` games: rounds, return and survival time."""
        out = {}
        for m in sorted(set(self.window) | set(self.recent)):
            w = self.window.get(m) or {"segments": 0, "dropped": 0, "steps": 0, "bad": 0, "lag": 0, "episodes": 0}
            row = {"segments": w["segments"], "dropped_segments": w["dropped"], "steps": w["steps"],
                   "episodes": w["episodes"],
                   "bad_step_frac": round(w["bad"] / w["steps"], 4) if w["steps"] else None,
                   "policy_lag_mean": round(w["lag"] / w["segments"], 3) if w["segments"] else None}
            games = self.recent.get(m) or ()
            for key in ("round_reached", "return", "seconds"):
                vals = [g[key] for g in games if isinstance(g.get(key), (int, float))]
                row[f"{key}_mean"] = round(float(np.mean(vals)), 3) if vals else None
            out[str(m)] = row
        self.window = {}
        return out

    def verdicts(self, stats: dict[str, dict]) -> list[tuple[int, str]]:
        """(machine, "late" | "recovered") when a machine's late-step share crosses the line, once each way."""
        out = []
        for key, row in stats.items():
            m, frac = int(key), row["bad_step_frac"]
            if frac is None or row["steps"] < self.MIN_STEPS:
                continue
            if frac > self.BAD_STEP_WARN and m not in self.warned:
                self.warned.add(m)
                out.append((m, "late"))
            elif frac < self.BAD_STEP_OK and m in self.warned:
                self.warned.discard(m)
                out.append((m, "recovered"))
        return out


def _next_item(out, inbox, timeout_s: float = 1.0):
    """The next thing an actor sent: this machine's actors first (they drop a segment rather than wait, so they
    must not queue behind a busy network), then a remote worker's, already decoded in this process."""
    if inbox is None:
        try:
            return out.get(timeout=timeout_s)
        except queue_mod.Empty:
            return None, None, None
    try:
        return out.get_nowait()
    except queue_mod.Empty:
        pass
    try:
        return inbox.get_nowait()
    except queue_mod.Empty:
        pass
    try:
        return out.get(timeout=0.2)
    except queue_mod.Empty:
        return None, None, None


def train(config: RLConfig, run_dir: str | Path, *, say=print, fleet_token: str | None = None) -> Path:
    """Run the learner here and `n_actors` actor processes, until `total_steps` decisions or Ctrl-C. With
    `config.listen`, other PCs' workers (rl/fleet.py) play for it too, authenticated by `fleet_token`."""
    run_dir = Path(run_dir)
    if not config.listen and config.n_actors < 1:
        raise ValueError("no actors: give this machine some, or listen for other machines' (--listen)")
    run_dir.mkdir(parents=True, exist_ok=False)
    # init="fresh": a new pixel+audio policy saved as run_dir/init.pt, with no prior to anchor to (rl/learner.py).
    # Done before anything reads config.init, so the actors and the fleet's workers start from that file.
    from zombiesai.rl.learner import prepare_init

    config = prepare_init(config, run_dir)
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(config), "env": config.env_name, "algorithm": "ppo-finetune", "spec_version": spec.SPEC_VERSION},
        indent=2))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    learner = Learner(config, resolve_device(config.device))
    version = 0
    publish(run_dir / "weights.pt", learner.net, version)
    fleet = None
    if config.listen:
        from zombiesai.rl.fleet import FleetServer, play_settings

        # Every segment must reach as far back as this policy's oldest frame and hear as it does
        # (FrameHistory.depth and the actor's audio); the server refuses one that does not fit.
        fleet = FleetServer(config.listen, fleet_token or "", config=config, run_dir=run_dir,
                            settings=play_settings(config.fleet_root) if config.env == "real" else None,
                            context=max(learner.offsets), audio_shape=learner.audio_shape, say=say).start()
        fleet.set_weights(version, run_dir / "weights.pt")
        host, port = fleet.address
        say(f"listening for other machines' games on {host}:{port}")
    # One thread of work per actor. Actors inherit this environment when spawned, before they import numpy:
    # otherwise OpenBLAS starts a thread per core in every actor (the frame resize is a matrix product), and
    # four actors' pools fighting over 12 threads made 95% of real-game steps late.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ctx = mp.get_context("spawn")
    out = ctx.Queue(maxsize=max(8, 4 * config.n_actors))
    stop = ctx.Event()
    procs: dict[int, mp.Process] = {}
    restarts = {i: 0 for i in range(config.n_actors)}

    def spawn(i: int) -> None:
        p = ctx.Process(target=actor_main, args=(i, config, str(run_dir), out, stop), daemon=True,
                        name=f"actor-{i}")
        p.start()
        procs[i] = p

    for i in range(config.n_actors):
        spawn(i)
    say(f"learner on {learner.device}, {config.n_actors} {config.env_name} actors here from {config.init}")
    machines = None
    if fleet is not None:
        from zombiesai.rl.fleet import ACTOR_BLOCK

        machines = MachineStats(ACTOR_BLOCK)
    log = open(run_dir / "metrics.jsonl", "a", buffering=1)
    # Every finished game, one line each: what the stream overlay and its page are built from (viz/stream.py).
    episode_log = open(run_dir / "episodes.jsonl", "a", buffering=1)
    recent: deque[dict] = deque(maxlen=100)
    batch: list[Segment] = []
    batch_n = step = episodes = dropped = 0
    lags: list[int] = []
    start = time.time()
    checkpoint = run_dir / "checkpoint.pt"
    try:
        while step < config.total_steps:
            kind, i, payload = _next_item(out, fleet.inbox if fleet is not None else None)
            if kind == "segment":
                lag = version - payload.version
                if machines is not None:
                    machines.segment(payload, lag, accepted=lag <= config.max_policy_lag)
                if lag > config.max_policy_lag:
                    dropped += 1
                else:
                    lags.append(lag)
                    batch.append(payload)
                    batch_n += payload.n
            elif kind == "episode":
                episodes += 1
                recent.append(payload)
                if machines is not None:
                    machines.episode(i, payload)
                episode_log.write(json.dumps({"t": round(time.time(), 1), "step": step, "version": version,
                                              **{k: v for k, v in payload.items() if k in EPISODE_FIELDS}},
                                             default=_json_scalar) + "\n")
                say(episode_line(i, payload))
            elif kind == "error":
                say(f"  actor {i} failed:\n{payload}")
            for j, p in list(procs.items()):
                if not p.is_alive() and not stop.is_set():
                    if restarts[j] >= config.actor_restarts:
                        say(f"  actor {j} has died {restarts[j]} times; leaving it down")
                        del procs[j]
                        continue
                    restarts[j] += 1
                    say(f"  actor {j} is down; restarting it ({restarts[j]}/{config.actor_restarts})")
                    spawn(j)
            if not procs and fleet is None:
                raise RuntimeError("every actor is down")
            if batch_n >= config.batch_steps:
                stats = learner.update(batch)
                step += batch_n
                version += 1
                publish(run_dir / "weights.pt", learner.net, version)
                alive = sum(p.is_alive() for p in procs.values())
                per_machine = None
                if fleet is not None:
                    per_machine = machines.flush()
                    fleet.set_weights(version, run_dir / "weights.pt")
                    alive += fleet.remote_actors_alive()
                    (run_dir / "fleet.json").write_text(json.dumps(fleet.snapshot(per_machine), indent=2))
                    for m, verdict in machines.verdicts(per_machine):
                        who = "this machine" if m == 0 else fleet.name_of(m)
                        say(f"  {who}: {100 * per_machine[str(m)]['bad_step_frac']:.0f}% of its steps late"
                            + (" -- its games are feeding the batch more steps nobody chose; fewer --actors there?"
                               if verdict == "late" else ", back to normal"))
                row = {"update": learner.updates, "step": step, "sps": int(step / max(time.time() - start, 1e-9)),
                       "episodes": episodes, "version": version, "dropped_segments": dropped,
                       "policy_lag_mean": float(np.mean(lags)) if lags else 0.0, "actors_alive": alive, **stats}
                if fleet is not None:
                    row["machines"] = (1 if config.n_actors else 0) + len(fleet.alive())
                    row["per_machine"] = per_machine
                if recent:
                    for key in ("return", "length", "round_reached", "repair_share", "max_term_share", "seconds"):
                        vals = [e[key] for e in recent if isinstance(e.get(key), (int, float))]
                        if vals:
                            row[f"{key}_mean"] = float(np.mean(vals))
                log.write(json.dumps(row) + "\n")
                _print_row(row, say)
                batch, batch_n, lags = [], 0, []
                if learner.updates % config.checkpoint_every == 0:
                    learner.checkpoint(checkpoint, step, {"episodes": episodes})
    except KeyboardInterrupt:
        say("interrupted: stopping the actors")
    finally:
        stop.set()
        if fleet is not None:
            fleet.close()  # the workers stop their games when they lose us
        deadline = time.time() + 20
        while any(p.is_alive() for p in procs.values()) and time.time() < deadline:
            try:  # keep draining so no actor blocks on a full queue while it shuts down
                out.get(timeout=0.2)
            except queue_mod.Empty:
                pass
        for p in procs.values():
            if p.is_alive():
                p.terminate()
        log.close()
        episode_log.close()
        learner.checkpoint(checkpoint, step, {"episodes": episodes})
    return checkpoint


def episode_line(actor, payload: dict) -> str:
    """The log line of a finished game. Summaries come from other machines too (rl/fleet.py cleans them, but a
    field can still be missing), so nothing here may raise on a missing, None or odd value: a bad summary must
    never end the run for every machine."""

    def number(key: str, fmt: str = "") -> str:
        value = payload.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value):
            return format(value, fmt)
        return "?"

    reason = payload.get("reason")
    return (f"  actor {actor} episode {number('episode')}: round {number('round_reached')}, "
            f"return {number('return', '.1f')}, {number('length')} steps"
            + (f" ({reason[:200]})" if isinstance(reason, str) and reason else ""))


def _json_scalar(value):
    """For json.dumps: a numpy scalar as its Python value, anything else unknown as its text."""
    return value.item() if isinstance(value, np.generic) else str(value)


def _print_row(row: dict, say) -> None:
    parts = [f"upd {row['update']:>4}", f"step {row['step']:>9,}", f"sps {row['sps']:>4}"]
    if "return_mean" in row:
        parts.append(f"return {row['return_mean']:>7.2f}")
    if "round_reached_mean" in row:
        parts.append(f"round {row['round_reached_mean']:.2f}")
    if row.get("warmup"):
        parts.append(f"critic warm-up, value loss {row.get('value_loss', float('nan')):.3f}")
    else:
        ev = row.get("explained_variance")
        kl_ref = row.get("kl_ref")  # absent with no prior to anchor to (init="fresh")
        parts.append(f"ent {row.get('entropy', float('nan')):.2f} kl {row['approx_kl']:.4f} "
                     + (f"kl_bc {kl_ref:.3f} " if isinstance(kl_ref, float) and math.isfinite(kl_ref) else "")
                     + f"clip {row.get('clipfrac', float('nan')):.2f} ev {'-' if ev is None else f'{ev:.2f}'}")
    if row.get("bad_step_frac"):
        parts.append(f"bad {row['bad_step_frac']:.1%}")
    say(" | ".join(parts))
