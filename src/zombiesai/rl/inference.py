"""One copy of the policy per machine: an inference process that acts for every actor on this PC.

Every actor used to import torch and load its own copy of the policy -- ~600 MB of resident memory each before
its game had drawn a frame, so eight games spent ~5 GB on identical copies, memory a ninth game could have had. Now
one **server** process holds the policy (on `actor_device`: the GPU, or a few CPU threads), follows `weights.pt`
itself, and answers every local actor; the actors are numpy, the env and the frame history, and never import torch.

How an actor and the server talk, and why each piece is there:

* **Shared memory, one slot per actor** (`Slots`): the stacked frames (uint8), the audio feature and its mask, and
  the actor's sampling noise go in; the action, its log-prob and the policy version that chose it come out. Nothing
  is pickled or copied through a pipe on the way, and the server builds its batch with one fancy-index copy.
* **A semaphore each way** for the wake-up: one the actors post to after writing a request, and one per actor the
  server posts to after writing the reply -- a futex wake, tens of microseconds.
* **The server batches whatever is ready now** and never waits to fill a batch: at 15 Hz the games' ticks are not
  aligned, and waiting for a fuller batch would only add a late game's wait to every other's.
* **A request number per slot, used as a seqlock.** The actor marks the slot odd while it writes and even when the
  request is whole; the server reads the number before and after copying, and drops a copy the actor wrote over
  (it gave up on that request and made the next one). A reply carries the number it answers, so a late reply to an
  abandoned request is never taken for the current one's.
* **The actor owns its randomness.** It draws the Gumbel noise from its own stream (seeded `seed * 7919 + index`,
  as the actors always were) and sends it with the frames; the server only adds it to the log-probs. So the
  server keeps no per-actor state, a restarted server does not replay anyone's stream, and an action does not
  depend on which batch it happened to be in. `LocalPolicy` samples with the same noise the same way.
* **The real-time guard.** An actor waits at most `inference_timeout_ms` for its reply; then it sends the neutral
  action (every key released) and marks the step bad -- bad steps are masked out of the losses. While the server
  is known to be down or has no weights (`Slots.version` < 0) it does not wait at all.
* **Weights change per step now, not per segment.** Each reply says which version chose it; a segment is labelled
  with the oldest version in it, so the learner's lag filter stays conservative, and the per-step behaviour
  log-probs keep the PPO ratio exact whatever version each step came from.

The slots and the semaphores belong to the `ActorPool` (rl/pool.py) that starts the server and the actors, not to
the server: a server that dies is restarted onto the same slots, and the actors carry on without re-attaching.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from zombiesai import spec

HEADS = len(spec.ACTION_NVEC)
WIDTH = max(spec.ACTION_NVEC)  # the Gumbel noise is drawn over FactoredCategorical's padded (heads, width) grid
NEUTRAL = np.asarray(spec.NEUTRAL_ACTION, np.int64)  # nothing held, no look: what an actor without a reply sends

# Per-slot control words, each slot's on a cache line of its own so actors do not false-share.
REQ, REP, VERSION = 0, 1, 2
CTRL_WORDS = 8
# The block's own words: the version the server is serving (-1: no server, or no weights yet), its pid, and the
# monotonic time of its last loop (beat), which the pool watches for a hung server.
G_VERSION, G_PID = 0, 1


@dataclass(frozen=True)
class PolicyShape:
    """What an actor needs to know about the policy to feed it -- which past frames it stacks, what it hears --
    without loading it, so without torch."""

    offsets: tuple[int, ...]  # steps back, oldest first (BCConfig.offsets)
    audio_features: dict | None = None  # AudioFeatureConfig's fields; None for a policy that does not hear

    @property
    def audio_shape(self) -> tuple[int, ...] | None:
        if self.audio_features is None:
            return None
        from zombiesai.demos.hearing import AudioFeatureConfig

        return tuple(AudioFeatureConfig(**self.audio_features).shape)


def shape_of(bc_config, meta: dict) -> PolicyShape:
    """The shape of a loaded BC policy: a hearing checkpoint without saved features hears with the defaults, as
    `bc.load` builds it."""
    audio = None
    if bc_config.use_audio:
        from dataclasses import asdict

        from zombiesai.demos.hearing import AudioFeatureConfig

        audio = meta.get("audio_features") or asdict(AudioFeatureConfig())
    return PolicyShape(tuple(int(o) for o in bc_config.offsets), audio)


def policy_shape(init: str | Path) -> PolicyShape:
    """The shape of the checkpoint at `init`. Reads it with torch -- in the pool's process, never an actor's."""
    import torch

    from zombiesai.demos.bc import BCConfig

    blob = torch.load(init, map_location="cpu", weights_only=True, mmap=True)
    spec.require_spec_version(blob["spec_version"], str(init))
    return shape_of(BCConfig(**blob["config"]), blob)


# ------------------------------------------------------------------------------------------------ the slots


def _layout(n: int, shape: PolicyShape) -> tuple[list[tuple[str, type, tuple, int]], int]:
    """(name, dtype, dims, byte offset) of every array in a block of `n` slots, each on its own cache line, and the
    block's size. Both sides compute it from (n, shape), so nothing about the layout has to travel."""
    audio = shape.audio_shape
    arrays = [("globals", np.int64, (8,)), ("beat", np.float64, (1,)), ("ctrl", np.int64, (n, CTRL_WORDS)),
              ("pixels", np.uint8, (n, len(shape.offsets), *spec.PIXELS_SHAPE)),
              ("audio", np.float32, (n, *audio) if audio else None), ("mask", np.float32, (n,)),
              ("noise", np.float32, (n, HEADS, WIDTH)), ("action", np.int64, (n, HEADS)), ("logp", np.float64, (n,))]
    out, offset = [], 0
    for name, dtype, dims in arrays:
        if dims is None:
            continue
        offset = -(-offset // 64) * 64
        out.append((name, dtype, dims, offset))
        offset += int(np.prod(dims)) * np.dtype(dtype).itemsize
    return out, offset


class Slots:
    """Every actor's slot in one shared-memory block, as arrays with the slot first, so the server gathers a batch
    of frames with one indexed copy."""

    audio = None  # a policy that does not hear has no audio array

    def __init__(self, shm: shared_memory.SharedMemory, n: int, shape: PolicyShape, owner: bool):
        self.shm, self.n, self.shape, self.owner = shm, n, shape, owner
        self.name = shm.name
        for name, dtype, dims, offset in _layout(n, shape)[0]:
            setattr(self, name, np.ndarray(dims, dtype=dtype, buffer=shm.buf, offset=offset))

    @classmethod
    def create(cls, n: int, shape: PolicyShape) -> "Slots":
        shm = shared_memory.SharedMemory(create=True, size=_layout(n, shape)[1])
        slots = cls(shm, n, shape, owner=True)
        slots.globals[:] = 0
        slots.globals[G_VERSION] = -1
        slots.beat[0] = 0.0
        slots.ctrl[:] = 0
        return slots

    @classmethod
    def attach(cls, name: str, n: int, shape: PolicyShape) -> "Slots":
        # track=False: the pool owns the block; an actor or server exiting must not unlink it under the others.
        return cls(shared_memory.SharedMemory(name=name, track=False), n, shape, owner=False)

    @property
    def version(self) -> int:
        return int(self.globals[G_VERSION])

    def close(self) -> None:
        # Drop the numpy views first: a mapping with views still exported cannot be closed.
        for key in ("globals", "beat", "ctrl", "pixels", "audio", "mask", "noise", "action", "logp"):
            setattr(self, key, None)
        try:
            self.shm.close()
        except BufferError:
            pass  # someone still holds a view; the mapping goes when the process does
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


@dataclass
class InferenceLink:
    """Everything an actor or the server needs to find the slots, handed to each process when it is spawned (the
    semaphores can only travel that way)."""

    name: str
    n: int
    shape: PolicyShape
    request: object  # multiprocessing Semaphore: an actor has written a request
    replies: list  # one Semaphore per actor: its reply is written
    timeout_s: float
    server_stop: object  # multiprocessing Event: the pool is done with the server


# ------------------------------------------------------------------------------------------------ the actor's side


class InferenceClient:
    """An actor's end of its slot: `act` returns (action, log-prob, version), or None when no reply came in time."""

    def __init__(self, link: InferenceLink, index: int):
        self.link, self.index = link, index
        self.shape = link.shape
        self.slots = Slots.attach(link.name, link.n, link.shape)
        self.reply = link.replies[index]
        # Carry on from the slot's last request number: a restarted actor must not reuse one the server has served.
        last = int(self.slots.ctrl[index, REQ])
        self.seq = last + (last % 2)

    def refresh(self) -> int:
        """The version the server is serving, or -1 while it has no weights (or is being restarted)."""
        return self.slots.version

    def act(self, pixels: np.ndarray, audio: np.ndarray | None, mask: float | None, noise: np.ndarray):
        s, i = self.slots, self.index
        if s.version < 0:
            return None  # nobody to ask: do not spend the tick waiting
        seq = self.seq + 2
        s.ctrl[i, REQ] = seq - 1  # odd: being written
        s.pixels[i] = pixels
        if s.audio is not None:
            s.audio[i] = audio
            s.mask[i] = mask
        s.noise[i] = noise
        s.ctrl[i, REQ] = seq  # even: whole
        self.seq = seq
        while self.reply.acquire(False):
            pass  # wake-ups for requests this actor already gave up on
        self.link.request.release()
        deadline = time.perf_counter() + self.link.timeout_s
        while True:
            left = deadline - time.perf_counter()
            if left <= 0 or not self.reply.acquire(timeout=left):
                return None
            if s.ctrl[i, REP] == seq:
                return s.action[i].copy(), float(s.logp[i]), int(s.ctrl[i, VERSION])

    def close(self) -> None:
        self.slots.close()


class LocalPolicy:
    """The old way, still there as `inference="local"`: the actor's own copy of the policy on its own CPU thread,
    reloaded from `weights.pt` between segments, so every step of a segment has the same version."""

    def __init__(self, config, run_dir: Path):
        import torch

        from zombiesai.demos import bc
        from zombiesai.rl.config import resolve_device
        from zombiesai.rl.weights import WeightFollower

        torch.set_num_threads(1)
        self.device = resolve_device(config.actor_device)
        self.net, bc_config, meta = bc.load(config.init, self.device)
        self.net.eval()
        self.shape = shape_of(bc_config, meta)
        self.follower = WeightFollower(Path(run_dir) / "weights.pt")
        self.version = -1

    def refresh(self) -> int:
        self.version = self.follower.poll(self.net)
        return self.version

    def act(self, pixels, audio, mask, noise):
        actions, logps = sample(self.net, self.device, pixels[None],
                                None if audio is None else audio[None],
                                None if audio is None else np.asarray([mask], np.float32), noise[None])
        return actions[0], float(logps[0]), self.version

    def close(self) -> None:
        pass


def sample(net, device, pixels, audio, mask, noise):
    """(B, HEADS) actions and (B,) behaviour log-probs for a batch: Gumbel-max over every head with the noise the
    actors drew. The one place either way of running the policy samples."""
    import torch

    from zombiesai.rl.distributions import FactoredCategorical

    with torch.no_grad():
        px = torch.from_numpy(pixels).to(device)
        au = mk = None
        if audio is not None:
            au, mk = torch.from_numpy(audio).to(device), torch.from_numpy(mask).to(device)
        logits, _, _ = net(px, None, au, mk)
        dist = FactoredCategorical(logits, spec.ACTION_NVEC)
        u = torch.from_numpy(noise).to(device).clamp_min(1e-12)
        action = (dist.log_probs - torch.log(-torch.log(u))).argmax(-1)
        logp = dist.log_prob(action)
    return action.cpu().numpy().astype(np.int64), logp.double().cpu().numpy()


# ------------------------------------------------------------------------------------------------ the server


def server_threads(n_actors: int) -> int:
    """CPU threads for a CPU server: one forward of eight is ~4x one of one on a single thread, and a couple of
    threads bring a full batch back under a single actor's old latency without taking cores from the games."""
    return max(1, min(4, n_actors // 2))


def server_main(config, run_dir: str, link: InferenceLink) -> None:
    """The inference process: load the policy, follow `weights.pt`, answer whatever is waiting, until the pool
    says stop. A crash is the pool's to restart; Ctrl-C (which reaches every process) just ends it."""
    try:
        _serve(config, Path(run_dir), link)
    except KeyboardInterrupt:
        pass


def _serve(config, run_dir: Path, link: InferenceLink) -> None:
    import torch

    from zombiesai.demos import bc
    from zombiesai.rl.config import resolve_device
    from zombiesai.rl.weights import WeightFollower

    device = resolve_device(config.actor_device)
    torch.set_num_threads(server_threads(link.n))
    net, bc_config, meta = bc.load(config.init, device)
    net.eval()
    if shape_of(bc_config, meta) != link.shape:
        raise RuntimeError(f"{config.init} is not the policy the slots were laid out for")
    slots = Slots.attach(link.name, link.n, link.shape)
    slots.globals[G_PID] = os.getpid()
    follower = WeightFollower(run_dir / "weights.pt")
    served = np.full(link.n, -1, np.int64)  # the request number each slot was last answered for
    hears = slots.audio is not None
    next_poll = 0.0
    while not link.server_stop.is_set():
        now = time.monotonic()
        slots.beat[0] = now
        if now >= next_poll:  # a stat() a tick is plenty; a load is ~3 ms and happens between batches
            slots.globals[G_VERSION] = follower.poll(net)
            next_poll = now + 0.05
        if not link.request.acquire(timeout=0.1):
            continue
        while link.request.acquire(False):
            pass  # one scan answers every request posted so far
        version = int(slots.globals[G_VERSION])
        if version < 0:
            continue  # no weights yet: the actors do not ask, and one that did gives up on its own
        seqs = slots.ctrl[:, REQ].copy()
        ready = np.flatnonzero((seqs > 0) & (seqs % 2 == 0) & (seqs != served))
        if not len(ready):
            continue
        pixels, noise = slots.pixels[ready], slots.noise[ready]
        audio = slots.audio[ready] if hears else None
        mask = slots.mask[ready] if hears else None
        whole = slots.ctrl[ready, REQ] == seqs[ready]  # an actor that wrote over its request meanwhile is skipped
        ready, seqs_ready = ready[whole], seqs[ready][whole]
        if not len(ready):
            continue
        if not whole.all():
            pixels, noise = pixels[whole], noise[whole]
            if hears:
                audio, mask = audio[whole], mask[whole]
        actions, logps = sample(net, device, pixels, audio, mask, noise)
        for row, (i, seq) in enumerate(zip(ready, seqs_ready)):
            slots.action[i] = actions[row]
            slots.logp[i] = logps[row]
            slots.ctrl[i, VERSION] = version
            slots.ctrl[i, REP] = seq
            served[i] = seq
            link.replies[i].release()
