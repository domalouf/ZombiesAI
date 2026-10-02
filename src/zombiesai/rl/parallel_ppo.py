"""Reinforcement learning on the policy's own play: PPO fine-tuning of the behavioural-cloning policy, fed by
several games at once.

The shape of it:

* **Actors** -- one process per game instance (`realgame/instances.py`), or per NachtSim for a rehearsal.
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

import copy
import json
import multiprocessing as mp
import os
import queue as queue_mod
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zombiesai import spec
from zombiesai.rl.distributions import FactoredCategorical


# What episodes.jsonl keeps of an actor's episode summary: the numbers, not the per-term breakdowns.
EPISODE_FIELDS = ("actor", "episode", "reason", "return", "length", "seconds", "round_reached", "shots", "hits",
                  "points_gained", "bad_steps", "repair_share", "max_term_share")


@dataclass
class RLConfig:
    init: str = ""  # the BC checkpoint RL starts from
    env: str = "real"  # "real": the game instances of a fleet; "sim": NachtSim's rendered view, for rehearsal
    n_actors: int = 4
    total_steps: int = 2_000_000
    segment_steps: int = 256  # ~17 s of real play
    batch_steps: int = 4096  # ~70 s of four games
    gamma: float = 0.995  # a 13 s horizon at 15 Hz
    gae_lambda: float = 0.95
    lr: float = 5e-5
    update_epochs: int = 3
    minibatch_size: int = 256
    clip_coef: float = 0.2  # the ratio is a product over eight heads: 0.1 clipped half of every batch
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    kl_coef: float = 0.2
    kl_decay: float = 0.995  # per update, down to kl_min
    kl_min: float = 0.02
    target_kl: float | None = 0.03  # stop an update's epochs early past 1.5x this
    critic_warmup_updates: int = 5
    max_grad_norm: float = 0.5
    max_policy_lag: int = 2
    reward_scale: bool = True
    device: str = "auto"
    actor_device: str = "cpu"
    seed: int = 0
    checkpoint_every: int = 10
    # real env
    fleet_root: str = "runs/instances"
    counts_per_degree: float = 9.09
    bindings: str = "configs/waw_bindings.json"
    record_every: int = 0  # record every k-th episode of each actor as a clip (0: never)
    hear: bool = True  # a checkpoint trained with audio hears its own instance's sink
    # sim env
    sim: dict = field(default_factory=dict)
    actor_restarts: int = 20  # per actor, before the run gives up on it
    # Other PCs' games (rl/fleet.py): "host:port" to accept their workers on, "" for this machine's games only
    listen: str = ""

    @property
    def env_name(self) -> str:
        return "real-waw" if self.env == "real" else "nacht-render"


# ------------------------------------------------------------------------------------------------ segments


@dataclass
class Segment:
    """Up to `segment_steps` decisions of one actor under one policy version.

    `frames` holds `context` frames before step 0 (for the stack to reach into), then one frame per step, then
    the observation after the last step (for the bootstrap value): `context + T + 1` frames in all. Step t's
    stack is frames[context + t - back] for each of the policy's offsets."""

    actor: int
    version: int
    context: int
    frames: np.ndarray  # (context + T + 1, H, W, 3) uint8
    actions: np.ndarray  # (T, heads) int64
    logp: np.ndarray  # (T,) behaviour log-prob of the action taken
    rewards: np.ndarray  # (T,) float32
    bad: np.ndarray  # (T,) bool
    terminated: bool
    audio: np.ndarray | None = None  # (T + 1, *feature) float32, one per observation
    audio_mask: np.ndarray | None = None  # (T + 1,) float32

    @property
    def n(self) -> int:
        return len(self.actions)


class FrameHistory:
    """The actor's view of the past: enough frames for the oldest offset, clamped at the last reset exactly as
    `demos.agent.BCAgent` and the training loader clamp (the first frame since reset stands in for anything
    older)."""

    def __init__(self, offsets: tuple[int, ...]):
        self.offsets = tuple(offsets)  # steps back, oldest first
        self.depth = max(self.offsets)
        self.frames: deque[np.ndarray] = deque(maxlen=self.depth + 1)

    def reset(self, frame: np.ndarray) -> None:
        self.frames.clear()
        self.push(frame)

    def push(self, frame: np.ndarray) -> None:
        self.frames.append(np.array(frame, dtype=np.uint8))

    def stack(self) -> np.ndarray:
        last = len(self.frames) - 1
        return np.stack([self.frames[max(last - back, 0)] for back in self.offsets])

    def context(self) -> list[np.ndarray]:
        """The `depth` frames before the newest, oldest first, padded by repeating the oldest held."""
        held = list(self.frames)[:-1]
        pad = [self.frames[0]] * (self.depth - len(held))
        return pad + held


def stack_indices(context: int, n: int, offsets: tuple[int, ...]) -> np.ndarray:
    """(n + 1, len(offsets)) frame indices of each step's stack, and of the final observation's."""
    t = np.arange(n + 1)[:, None]
    return context + t - np.asarray(offsets)[None, :]


def compute_gae(rewards, values, last_value: float, terminated: bool, gamma: float, lam: float):
    """GAE over one segment: no episode boundary inside it, and the end bootstraps unless it terminated."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    next_value = 0.0 if terminated else float(last_value)
    gae = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


class RunningStd:
    """Running variance of the discounted return, per actor stream, for reward scaling."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.count, self.mean, self.m2 = 1e-4, 0.0, 1.0
        self.ret: dict[int, float] = {}

    def update(self, actor: int, rewards: np.ndarray, terminated: bool) -> None:
        ret = self.ret.get(actor, 0.0)
        for r in rewards:
            ret = ret * self.gamma + float(r)
            self.count += 1
            delta = ret - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (ret - self.mean)
        self.ret[actor] = 0.0 if terminated else ret

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.m2 / self.count, 1e-8)))


def kl_to_reference(logits: torch.Tensor, ref_logits: torch.Tensor, nvec) -> torch.Tensor:
    """(B,) KL(pi || pi_ref), summed over the factored heads."""
    p = FactoredCategorical(logits, nvec).log_probs
    q = FactoredCategorical(ref_logits, nvec).log_probs
    return (p.exp() * (p - q)).sum((-2, -1))


# ------------------------------------------------------------------------------------------------ weights


def publish(path: Path, net: nn.Module, version: int) -> None:
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"version": version, "model": state}, tmp)
    tmp.replace(path)


class WeightFollower:
    """The actor's side: reload `weights.pt` when it changes. ~3 ms for the BC net, done between segments."""

    def __init__(self, path: Path):
        self.path, self.version, self._stamp = path, -1, None

    def poll(self, net: nn.Module) -> int:
        try:
            stamp = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return self.version
        if stamp != self._stamp:
            try:
                blob = torch.load(self.path, map_location="cpu", weights_only=True)
            except (EOFError, RuntimeError, OSError):
                return self.version  # caught mid-replace; next poll
            net.load_state_dict(blob["model"])
            self.version, self._stamp = int(blob["version"]), stamp
        return self.version


# ------------------------------------------------------------------------------------------------ actor envs


class SimActorEnv:
    """NachtSim's rendered view behind the real environment's interface, as fast as it will go."""

    def __init__(self, sim: dict, seed: int):
        from zombiesai.sim.nacht_sim import NachtSim, SimConfig

        self.env = NachtSim(SimConfig(**{**sim, "obs_profile": "render"}))
        self.seed = seed
        self.episodes = 0
        self.capture = None

    def reset(self):
        obs, info = self.env.reset(seed=self.seed * 100_003 + self.episodes)
        self.episodes += 1
        self._ret, self._len = 0.0, 0
        self._shots = self._hits = 0
        self._rs, self._seen = None, (0, 0)
        return {"pixels": obs["pixels"]}, info

    def _count_shots(self) -> None:
        """The sim keeps shots per round (RoundStats, replaced at each round start); this sums them per game."""
        rs = self.env.rs
        if rs is not self._rs:
            self._rs, self._seen = rs, (0, 0)
        self._shots += rs.shots - self._seen[0]
        self._hits += rs.shot_hits - self._seen[1]
        self._seen = (rs.shots, rs.shot_hits)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action))
        self._ret += reward
        self._len += 1
        self._count_shots()
        out = {"bad": False}
        if terminated or truncated:
            out["episode"] = {"return": self._ret, "length": self._len, "shots": self._shots, "hits": self._hits,
                              "seconds": info.get("episode_time_s"),
                              **{k: info[k] for k in ("round_reached", "repair_share", "max_term_share") if k in info}}
        return {"pixels": obs["pixels"]}, float(reward), bool(terminated), bool(truncated), out

    def close(self) -> None:
        self.env.close()


def make_real_env(config: RLConfig, index: int, audio_features=None):
    """The game on instance `index` of the fleet at `config.fleet_root`, as a RealGameEnv."""
    from zombiesai.demos.inputs import DEFAULT_BINDINGS
    from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
    from zombiesai.realgame.env import RealGameEnv
    from zombiesai.realgame.instances import Instance, load_fleet, specs
    from zombiesai.realgame.console import console_open
    from zombiesai.realgame.scoreboard import scoreboard_shown
    from zombiesai.realgame.xtest import console_command, tap_key

    fleet = load_fleet(config.fleet_root)
    instance = Instance(fleet, specs(fleet)[index], say=lambda m: print(f"[actor {index}] {m}", flush=True))
    bindings_path = Path(config.bindings)
    bindings = json.loads(bindings_path.read_text()) if bindings_path.exists() else dict(DEFAULT_BINDINGS)
    sink = instance.sink()
    dispatcher = ActionDispatcher(sink, DispatchConfig(counts_per_degree=config.counts_per_degree, bindings=bindings))
    hearing = None
    if audio_features is not None and config.hear:
        from zombiesai.demos.audio import PulseMonitorStream
        from zombiesai.demos.hearing import LiveAudio

        hearing = LiveAudio(PulseMonitorStream(instance.spec.monitor, stream_name=f"actor {index}"),
                            audio_features).start()
    capture = instance.capture()

    def shows_console() -> bool:
        return console_open((capture.last_hud or {}).get("console"))

    def looks_open() -> bool:  # a fresh look, for the console typing between frames
        capture.read()
        return shows_console()

    return RealGameEnv(capture, dispatcher, focus=instance.focuser(sink),
                       console=lambda command: console_command(sink, command, is_open=looks_open),
                       # 30 ms taps are shorter than the game notices for some keys (the scores key); 150 ms is sure
                       restart=instance.restart_game, press=lambda key: tap_key(sink, key, hold_s=0.15),
                       console_open=shows_console,
                       downed=lambda: scoreboard_shown((capture.last_hud or {}).get("scores")),
                       hearing=hearing, say=lambda m: print(f"[actor {index}] {m}", flush=True))


def make_actor_env(config: RLConfig, index: int, audio_features=None):
    if config.env == "sim":
        return SimActorEnv(config.sim, config.seed + index)
    if config.env == "real":
        return make_real_env(config, index, audio_features)
    raise ValueError(f"unknown env {config.env!r}")


# ------------------------------------------------------------------------------------------------ actor


def actor_main(index: int, config: RLConfig, run_dir: str, out: "mp.Queue", stop) -> None:
    """One actor process: play, ship segments and episode summaries, pick up new weights between segments."""
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        _actor_loop(index, config, Path(run_dir), out, stop)
    except Exception:  # noqa: BLE001 -- reported to the learner, which decides whether to restart us
        out.put(("error", index, traceback.format_exc()))


def _actor_loop(index: int, config: RLConfig, run_dir: Path, out, stop) -> None:
    from zombiesai.demos import bc
    from zombiesai.demos.hearing import feature_config, silence

    device = torch.device(config.actor_device)
    net, bc_config, meta = bc.load(config.init, device)
    net.eval()
    audio_features = None
    if bc_config.use_audio:
        from zombiesai.demos.hearing import AudioFeatureConfig

        audio_features = feature_config(meta.get("audio_features")) or AudioFeatureConfig()
    quiet = silence(audio_features) if audio_features is not None else None
    follower = WeightFollower(run_dir / "weights.pt")
    history = FrameHistory(bc_config.offsets)
    rng = torch.Generator(device="cpu").manual_seed(config.seed * 7919 + index)
    env = make_actor_env(config, index, audio_features)
    episode = 0
    writer = None
    try:
        obs, _ = env.reset()
        history.reset(obs["pixels"])
        while not stop.is_set():
            version = follower.poll(net)
            if version < 0:
                time.sleep(0.1)  # the learner has not published yet
                continue
            frames = history.context() + [history.frames[-1]]
            actions, logps, rewards, bad, heard, masks = [], [], [], [], [], []
            terminated = ended = False

            def hear(o):
                if audio_features is None:
                    return
                a = o.get("audio")
                heard.append(quiet if a is None else np.asarray(a, np.float32))
                masks.append(float(o.get("audio_mask", 1.0)) if a is not None else 0.0)

            hear(obs)
            for _ in range(config.segment_steps):
                with torch.no_grad():
                    pixels = torch.from_numpy(history.stack()[None]).to(device)
                    audio = mask = None
                    if audio_features is not None:
                        audio = torch.from_numpy(heard[-1][None]).to(device)
                        mask = torch.tensor([masks[-1]], device=device)
                    logits, _, _ = net(pixels, None, audio, mask)
                    dist = FactoredCategorical(logits, spec.ACTION_NVEC)
                    u = torch.rand(dist.log_probs.shape, generator=rng).clamp_min(1e-12).to(device)
                    action = (dist.log_probs - torch.log(-torch.log(u))).argmax(-1)
                    logp = float(dist.log_prob(action)[0])
                action = action[0].cpu().numpy().astype(np.int64)
                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                logps.append(logp)
                rewards.append(reward)
                bad.append(bool(info.get("bad", False)))
                if writer is not None:
                    writer.add(obs["pixels"], action, flags=1 if bad[-1] else 0,
                               hud=getattr(getattr(env, "capture", None), "last_hud", None),
                               extras={"actor": np.uint8(1), "reward": np.float32(reward)})
                history.push(obs["pixels"])
                frames.append(history.frames[-1])
                hear(obs)
                if terminated or truncated:
                    _offer(out, ("episode", index, {**info.get("episode", {}), "actor": index, "episode": episode}))
                    ended = True
                    break
                if stop.is_set():
                    break
            if not actions:
                break
            segment = Segment(
                actor=index, version=version, context=history.depth, frames=np.stack(frames),
                actions=np.stack(actions), logp=np.asarray(logps, np.float32),
                rewards=np.asarray(rewards, np.float32), bad=np.asarray(bad, bool), terminated=bool(terminated),
                audio=np.stack(heard) if heard else None,
                audio_mask=np.asarray(masks, np.float32) if heard else None,
            )
            _offer(out, ("segment", index, segment))
            if ended:
                if writer is not None:
                    writer.close(summary=info.get("episode", {}))
                    writer = None
                episode += 1
                obs, _ = env.reset()
                history.reset(obs["pixels"])
                if config.record_every and episode % config.record_every == 0:
                    writer = _episode_writer(run_dir, index, episode, env)
    finally:
        if writer is not None:
            writer.close(summary={"ended": "actor stopped"})
        env.close()


def _offer(out, item, timeout_s: float = 0.25) -> bool:
    """Hand the learner something without ever stalling the game for long: a real-time actor that blocks on a
    full queue leaves its last action's keys held in a live game. A segment that cannot be delivered is lost
    (the learner is behind anyway); the game is not."""
    try:
        out.put(item, timeout=timeout_s)
        return True
    except queue_mod.Full:
        return False


def _episode_writer(run_dir: Path, index: int, episode: int, env):
    from zombiesai.demos.clips import ClipWriter

    source = env.capture.describe() if getattr(env, "capture", None) is not None else {"kind": "sim"}
    return ClipWriter(run_dir / "episodes" / f"a{index}_ep{episode:05d}", source={**source, "actor": index},
                      label_source="play", config={"decision_hz": spec.DECISION_HZ})


# ------------------------------------------------------------------------------------------------ learner


def root_prior(path: str | Path) -> str:
    """The behavioural-cloning checkpoint an RL checkpoint's chain of continuations started from."""
    seen = set()
    path = str(path)
    while path not in seen:
        seen.add(path)
        rl = torch.load(path, map_location="cpu", weights_only=False).get("rl") or {}
        if not rl:
            return path
        if rl.get("reference"):
            return rl["reference"]
        path = str(rl["init"])
    raise ValueError(f"checkpoint chain loops at {path}")


class Learner:
    """The PPO update over a batch of segments, on the learner's device. Separate from the process plumbing
    so it can be tested (and driven) in-process."""

    def __init__(self, config: RLConfig, device: torch.device):
        from zombiesai.demos import bc

        self.config, self.device = config, device
        self.net, self.bc_config, self.meta = bc.load(config.init, device)
        self.net.train()
        # Continuing from an RL checkpoint picks up where it stopped: the KL anchor is still the behavioural
        # prior the chain started from (not the checkpoint itself, or every restart would let the policy drift
        # further), and the update count -- so the critic warm-up is not redone -- and the anchor's weight carry
        # on.
        previous = self.meta.get("rl") or {}
        self.reference = previous.get("reference") or (root_prior(previous["init"]) if previous else config.init)
        if previous:
            self.ref, _, _ = bc.load(self.reference, device)
            self.ref.eval()
        else:
            self.ref = copy.deepcopy(self.net).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.net.critic.parameters(), lr=config.lr * 10, eps=1e-5)
        self.offsets = self.bc_config.offsets
        self.uses_audio = self.bc_config.use_audio
        self.scaler = RunningStd(config.gamma)
        self.updates = int(previous.get("updates", 0))
        self.kl_coef = float(previous.get("kl_coef", config.kl_coef))

    def _forward(self, net, pixels, audio=None, mask=None):
        return net(pixels, None, audio, mask)

    def _gather(self, segments: list[Segment], rows: np.ndarray, which: np.ndarray):
        """Stacked pixels (and audio) of rows (segment, t) -- t may be n for the final observation."""
        pixels, audio, masks = [], [], []
        for s_idx, t in zip(which, rows):
            seg = segments[s_idx]
            idx = seg.context + t - np.asarray(self.offsets)
            pixels.append(seg.frames[idx])
            if self.uses_audio:
                audio.append(seg.audio[t])
                masks.append(seg.audio_mask[t])
        out = [torch.from_numpy(np.stack(pixels)).to(self.device)]
        if self.uses_audio:
            out += [torch.from_numpy(np.stack(audio)).to(self.device), torch.from_numpy(np.asarray(masks)).to(self.device)]
        else:
            out += [None, None]
        return out

    @torch.no_grad()
    def _values(self, segments: list[Segment]) -> list[np.ndarray]:
        """Current-network values of every observation, final one included: (n + 1,) per segment."""
        self.net.eval()
        out = []
        for s_idx, seg in enumerate(segments):
            rows = np.arange(seg.n + 1)
            values = []
            for chunk in np.array_split(rows, max(1, len(rows) // 256)):
                px, au, mk = self._gather(segments, chunk, np.full(len(chunk), s_idx))
                values.append(self._forward(self.net, px, au, mk)[1].float().cpu().numpy())
            out.append(np.concatenate(values))
        self.net.train()
        return out

    def update(self, segments: list[Segment]) -> dict:
        c = self.config
        for seg in segments:
            if c.reward_scale:
                self.scaler.update(seg.actor, seg.rewards, seg.terminated)
        scale = self.scaler.std if c.reward_scale else 1.0
        values = self._values(segments)
        adv, ret, which, rows, old_logp, actions, valid = [], [], [], [], [], [], []
        for s_idx, (seg, v) in enumerate(zip(segments, values)):
            a, r = compute_gae(seg.rewards / scale, v[:-1], v[-1], seg.terminated, c.gamma, c.gae_lambda)
            adv.append(a)
            ret.append(r)
            which.append(np.full(seg.n, s_idx))
            rows.append(np.arange(seg.n))
            old_logp.append(seg.logp)
            actions.append(seg.actions)
            valid.append(~seg.bad)
        adv, ret = np.concatenate(adv), np.concatenate(ret)
        which, rows = np.concatenate(which), np.concatenate(rows)
        old_logp, actions, valid = np.concatenate(old_logp), np.concatenate(actions), np.concatenate(valid)
        pred = np.concatenate([v[:-1] for v in values])
        usable = np.flatnonzero(valid)
        stats: dict[str, list[float]] = {k: [] for k in
                                         ("policy_loss", "value_loss", "entropy", "clipfrac", "kl_ref")}
        warmup = self.updates < c.critic_warmup_updates
        epoch_kl: list[float] = []
        for _ in range(c.update_epochs):
            epoch_kl = []
            perm = np.random.permutation(usable)
            for start in range(0, len(perm), c.minibatch_size):
                mb = perm[start : start + c.minibatch_size]
                if len(mb) < 2:
                    continue
                px, au, mk = self._gather(segments, rows[mb], which[mb])
                target = torch.from_numpy(ret[mb]).to(self.device)
                if warmup:
                    with torch.no_grad():
                        h = self.net.features(px, None, au, mk)
                    value = self.net.critic(h).squeeze(-1)
                    value_loss = 0.5 * ((value - target) ** 2).mean()
                    self.critic_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    self.critic_opt.step()
                    stats["value_loss"].append(value_loss.item())
                    continue
                logits, value, _ = self._forward(self.net, px, au, mk)
                with torch.no_grad():
                    ref_logits = self._forward(self.ref, px, au, mk)[0]
                dist = FactoredCategorical(logits, spec.ACTION_NVEC)
                act = torch.from_numpy(actions[mb]).to(self.device)
                log_ratio = dist.log_prob(act) - torch.from_numpy(old_logp[mb]).to(self.device)
                ratio = log_ratio.exp()
                a = torch.from_numpy(adv[mb]).to(self.device)
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()
                value_loss = 0.5 * ((value - target) ** 2).mean()
                entropy = dist.entropy().mean()
                kl_ref = kl_to_reference(logits, ref_logits, spec.ACTION_NVEC).mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy + self.kl_coef * kl_ref
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    epoch_kl.append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip_coef).float().mean().item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["kl_ref"].append(kl_ref.item())
            if not warmup and c.target_kl is not None and epoch_kl and np.mean(epoch_kl) > 1.5 * c.target_kl:
                break
        self.updates += 1
        if not warmup:
            self.kl_coef = max(c.kl_min, self.kl_coef * c.kl_decay)
        out = {k: float(np.mean(v)) for k, v in stats.items() if v}
        out["approx_kl"] = float(np.mean(epoch_kl)) if epoch_kl else 0.0
        var = float(np.var(ret[usable])) if len(usable) else 0.0
        out["explained_variance"] = 1.0 - float(np.var(ret[usable] - pred[usable])) / var if var > 0 else None
        out.update({"warmup": warmup, "kl_coef": self.kl_coef, "reward_scale": scale,
                    "bad_step_frac": float(1.0 - valid.mean()) if len(valid) else 0.0,
                    "batch_steps": int(len(valid))})
        return out

    def checkpoint(self, path: Path, step: int, extra: dict | None = None) -> None:
        """A BC-format checkpoint (everything that plays a BC policy plays this) with an `rl` section."""
        tmp = path.with_suffix(".tmp")
        blob = {k: v for k, v in self.meta.items() if k not in ("model",)}
        blob.update({
            "kind": "bc",
            "model": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "rl": {"init": self.config.init, "reference": self.reference, "kl_coef": self.kl_coef,
                   "updates": self.updates, "step": step, "env": self.config.env_name,
                   "config": asdict(self.config), **(extra or {})},
        })
        torch.save(blob, tmp)
        tmp.replace(path)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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

        fleet = FleetServer(config.listen, fleet_token or "", config=config, run_dir=run_dir,
                            settings=play_settings() if config.env == "real" else None, say=say).start()
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
                if lag > config.max_policy_lag:
                    dropped += 1
                else:
                    lags.append(lag)
                    batch.append(payload)
                    batch_n += payload.n
            elif kind == "episode":
                episodes += 1
                recent.append(payload)
                episode_log.write(json.dumps({"t": round(time.time(), 1), "step": step, "version": version,
                                              **{k: v for k, v in payload.items() if k in EPISODE_FIELDS}}) + "\n")
                say(f"  actor {i} episode {payload.get('episode')}: round {payload.get('round_reached', '?')}, "
                    f"return {payload.get('return', float('nan')):.1f}, {payload.get('length', 0)} steps"
                    + (f" ({payload['reason']})" if payload.get("reason") else ""))
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
                if fleet is not None:
                    fleet.set_weights(version, run_dir / "weights.pt")
                    alive += fleet.remote_actors_alive()
                    (run_dir / "fleet.json").write_text(json.dumps(fleet.snapshot(), indent=2))
                row = {"update": learner.updates, "step": step, "sps": int(step / max(time.time() - start, 1e-9)),
                       "episodes": episodes, "version": version, "dropped_segments": dropped,
                       "policy_lag_mean": float(np.mean(lags)) if lags else 0.0, "actors_alive": alive, **stats}
                if fleet is not None:
                    row["machines"] = (1 if config.n_actors else 0) + len(fleet.alive())
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
        parts.append(f"ent {row.get('entropy', float('nan')):.2f} kl {row['approx_kl']:.4f} "
                     f"kl_bc {row.get('kl_ref', float('nan')):.3f} clip {row.get('clipfrac', float('nan')):.2f} "
                     f"ev {'-' if ev is None else f'{ev:.2f}'}")
    if row.get("bad_step_frac"):
        parts.append(f"bad {row['bad_step_frac']:.1%}")
    say(" | ".join(parts))
