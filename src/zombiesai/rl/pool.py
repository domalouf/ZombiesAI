"""This machine's actors, and the inference server that plays for them: started, kept alive and stopped in one place,
for the learner (`train()` in rl/parallel_ppo.py) and for a fleet worker (`FleetWorker.session` in rl/fleet.py)
alike -- the actors cannot tell which of the two they are playing for.

What it takes care of:

* **One thread of work per process.** The environment pins OMP/OpenBLAS/MKL to one thread before anything is
  spawned: otherwise OpenBLAS starts a thread per core in every actor (the frame resize is a matrix product), and
  four actors' pools fighting over 12 threads made 95% of real-game steps late (docs/rl.md). The server sets its
  own torch thread count.
* **Spawned, never forked**, and **without the parent's script**: multiprocessing's spawn re-runs the main
  script in every child so that anything pickled from `__main__` can be found -- and train_rl.py's imports torch,
  which would quietly put ~400 MB of torch back into every torch-free actor. Nothing a pool hands its processes
  lives in `__main__`, so the script is hidden from them while they start.
* **Restarts.** An actor that died (a game that crashed, a reset that failed) is started again, up to
  `actor_restarts` times each. A server that died, or stopped beating for `SERVER_HUNG_S`, is replaced on the same
  slots -- meanwhile its actors send neutral actions, as bad steps -- up to `actor_restarts` times too; past that
  the actors are stopped, and the caller sees every actor down.
* **Shutdown** keeps draining the queue while the actors stop (an actor blocked on a full queue cannot release
  its keys), stops the server only once the actors are done with it, and terminates stragglers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

from zombiesai.rl.actors import actor_main
from zombiesai.rl.inference import G_VERSION, InferenceLink, Slots, policy_shape, server_main

_SPAWNING = threading.Lock()  # a learner with --listen in a test runs a worker's pool beside its own


@contextmanager
def _without_main_script():
    """While processes are started: no `__main__` for multiprocessing to re-run in them (see the module's doc)."""
    with _SPAWNING:
        main = sys.modules["__main__"]
        sys.modules["__main__"] = types.ModuleType("__main__")  # no __file__, no __spec__: nothing to re-run
        try:
            yield
        finally:
            sys.modules["__main__"] = main


class ActorPool:
    """`config.n_actors` actor processes (numbered 0..n-1 on this machine) and, with `config.inference ==
    "server"`, the one inference process they share. Use `start()`, then `get()` and `tend()` in the caller's loop,
    and `close()` in its `finally`."""

    SERVER_HUNG_S = 15.0  # a live server that has not looped this long is stuck (a GPU hang): replace it
    SERVER_START_S = 300.0  # importing torch and starting CUDA on a busy machine can take a while; not this long

    def __init__(self, config, run_dir: str | Path, *, say=print):
        self.config, self.run_dir, self.say = config, Path(run_dir), say
        if config.inference not in ("server", "local"):
            raise ValueError(f"inference must be 'server' or 'local', not {config.inference!r}")
        self.ctx = mp.get_context("spawn")
        self.out = self.ctx.Queue(maxsize=max(8, 4 * config.n_actors))
        self.stop = self.ctx.Event()
        self.procs: dict[int, mp.Process] = {}  # the live actors by number; one that is given up on leaves it
        self.restarts = {i: 0 for i in range(config.n_actors)}
        self.server: mp.Process | None = None
        self.server_started = 0.0
        self.server_restarts = 0
        self.link: InferenceLink | None = None
        self.slots: Slots | None = None
        self._retired: list[mp.Process] = []  # stopped by a failed server; close() still waits for them

    # ---------------------------------------------------------------------------------------------- starting

    def start(self) -> "ActorPool":
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        n = self.config.n_actors
        if self.config.inference == "server" and n:
            self.slots = Slots.create(n, policy_shape(self.config.init))
            self.link = InferenceLink(self.slots.name, n, self.slots.shape, self.ctx.Semaphore(0),
                                      [self.ctx.Semaphore(0) for _ in range(n)],
                                      self.config.inference_timeout_ms / 1000.0, self.ctx.Event())
            self._start_server()
        for i in range(n):
            self._start_actor(i)
        return self

    def _process(self, target, args, name: str) -> mp.Process:
        p = self.ctx.Process(target=target, args=args, daemon=True, name=name)
        with _without_main_script():
            p.start()
        return p

    def _start_actor(self, i: int) -> None:
        self.procs[i] = self._process(actor_main, (i, self.config, str(self.run_dir), self.out, self.stop, self.link),
                                      f"actor-{i}")

    def _start_server(self) -> None:
        self.slots.globals[G_VERSION] = -1  # the actors stop asking until the new server has weights
        self.slots.beat[0] = 0.0
        self.server_started = time.monotonic()
        self.server = self._process(server_main, (self.config, str(self.run_dir), self.link), "inference")

    # ---------------------------------------------------------------------------------------------- running

    def get(self, timeout_s: float):
        """The next (kind, actor, payload) an actor sent, or (None, None, None)."""
        try:
            return self.out.get(timeout=timeout_s)
        except queue_mod.Empty:
            return None, None, None

    def alive(self) -> int:
        return sum(p.is_alive() for p in list(self.procs.values()))

    def tend(self) -> None:
        """Restart whatever died; call it every pass of the caller's loop."""
        if self.stop.is_set():
            return
        if self.server is not None:
            self._tend_server()
        for j, p in list(self.procs.items()):
            if p.is_alive():
                continue
            if self.restarts[j] >= self.config.actor_restarts:
                self.say(f"  actor {j} has died {self.restarts[j]} times; leaving it down")
                del self.procs[j]
                continue
            self.restarts[j] += 1
            self.say(f"  actor {j} is down; restarting it ({self.restarts[j]}/{self.config.actor_restarts})")
            self._start_actor(j)

    def _tend_server(self) -> None:
        beat, now = float(self.slots.beat[0]), time.monotonic()
        if self.server.is_alive():
            hung = now - beat > self.SERVER_HUNG_S if beat else now - self.server_started > self.SERVER_START_S
            if not hung:
                return
            self.say(f"  the inference server has not answered for {now - (beat or self.server_started):.0f}s; "
                     "replacing it")
            self.server.kill()
        self.server.join(timeout=5)
        self.slots.globals[G_VERSION] = -1
        if self.server_restarts >= self.config.actor_restarts:
            self.say(f"  the inference server has died {self.server_restarts} times; stopping this machine's actors")
            self.stop.set()
            self._retired += self.procs.values()
            self.procs.clear()
            self.server = None
            return
        self.server_restarts += 1
        self.say(f"  the inference server is down (exit {self.server.exitcode}); restarting it "
                 f"({self.server_restarts}/{self.config.actor_restarts}); its actors send neutral actions meanwhile")
        self._start_server()

    # ---------------------------------------------------------------------------------------------- stopping

    def close(self, wait_s: float = 20.0) -> None:
        """Stop the actors (each releases its keys), then the server; terminate whatever will not stop."""
        self.stop.set()
        actors = list(self.procs.values()) + self._retired
        deadline = time.time() + wait_s
        while any(p.is_alive() for p in actors) and time.time() < deadline:
            try:  # keep draining so no actor blocks on a full queue while it shuts down
                self.out.get(timeout=0.2)
            except queue_mod.Empty:
                pass
        for p in actors:
            if p.is_alive():
                p.terminate()
        if self.server is not None:
            self.link.server_stop.set()
            self.server.join(timeout=5)
            if self.server.is_alive():
                self.server.kill()
                self.server.join(timeout=5)
        for p in actors:
            p.join(timeout=1)
        if self.slots is not None:
            self.slots.close()
            self.slots = None

    def __enter__(self) -> "ActorPool":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
