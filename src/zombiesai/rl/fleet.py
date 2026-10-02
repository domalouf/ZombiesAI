"""Training across several PCs: the learner's games plus every other gaming PC's, one PPO run.

The learner (`train()` with `listen` set) opens a small HTTP endpoint beside its own actors. Each other PC runs a
**worker** (`scripts/fleet_worker.py`): its own games and actor processes, exactly as `train()` starts them, and a
forwarder in place of the learner. The actors cannot tell the difference -- they still ship segments through a
local queue and follow a local `weights.pt` -- so everything `parallel_ppo.py` says about lag, bad steps and
never stalling a live game holds unchanged. Only the two hops between the queue and the learner are new:

    worker PC                                                        learner PC
    actor 0 ─┐                       POST /segment  (npz)
    actor 1 ─┼─ queue ─ forwarder ─────────────────────────────►  FleetServer ─ inbox ─ learner loop
    actor 2 ─┘                       GET  /weights  (on change)          │
    weights.pt ◄─────────── puller ◄──────────────────────────── weights.pt (the learner's, in memory)

What the network must not be allowed to do, and what stops it:

* **Train on games played under different rules.** A worker says hello before anything else, and is refused
  unless its `SPEC_VERSION` and git commit match the learner's, and its game's sensitivity, field of view and key
  bindings match too (`play_settings`): a different `config.cfg` changes what every look bin and key means,
  silently. Every segment carries the spec version again.
* **Mix two machines' actor 0.** The server numbers each worker's actors from its own block (100, 200, ...), so
  `episodes.jsonl`, the reward scaler's per-actor returns and the stream's numbers keep them apart; the worker's
  RNG seed moves by the same offset, so two machines never play the same random stream. A machine is known by
  its name and a random id its worker draws at start: a second PC under a name already playing is refused (one
  of them needs `--name`), unless the first has gone quiet -- then it was the same PC's worker, restarted.
  An actor number outside the machine's block is refused, as is a segment whose frame context or audio does not
  fit the learner's policy (it would train on the wrong frames, or crash the update).
* **Run arbitrary code.** Segments travel as `.npz` loaded with `allow_pickle=False`, and weights and the starting
  checkpoint as `torch.save` blobs the receiver opens with `weights_only=True`. Every request needs the fleet's
  shared token (`ZOMBIES_FLEET_TOKEN`), and a body over `MAX_BODY` is refused before it is read.
* **Leave keys held when the learner goes away.** A worker that cannot reach the learner for `lost_s` stops its
  actors (which releases every key, as Ctrl-C on `train_rl.py` does), then waits for the next run.

The traffic is small for a LAN: a 128x72 frame at 15 Hz is ~415 KB/s per game before compression.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import io
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from zombiesai import spec

DEFAULT_PORT = 47860
TOKEN_ENV = "ZOMBIES_FLEET_TOKEN"
MAX_BODY = 256 << 20  # a 256-step segment is ~7 MB raw; this is far past any real one
STATUS_FILE = "status.json"  # in the worker's out_root
ACTOR_BLOCK = 100  # worker k's actors are numbered from k * ACTOR_BLOCK; the learner's own stay at 0..n-1
INSTANCE_HEADER = "X-Worker-Instance"  # the worker process's random id, beside its name in X-Worker
# A worker heartbeats every 5 s, so one silent this long has stopped: a hello under its name from another
# process is the same PC's worker restarted, and takes over its block. Sooner, it is a second PC with that name.
STALE_S = 60.0
# The settings that decide what an action does in the game. Resolution, fps, vsync and the rest are set per
# instance on the command line (FleetConfig.dvars), so they cannot differ between machines and are not compared.
PLAY_DVARS = ("sensitivity", "m_yaw", "m_pitch", "input_viewSensitivity", "cg_fov", "cg_fovscale")


class FleetError(RuntimeError):
    """The learner refused us, for a reason a person has to fix (a different commit, different settings)."""


# ------------------------------------------------------------------------------------------------ wire format


def encode_segment(segment) -> bytes:
    """A Segment as a compressed .npz: arrays as arrays, the rest as a JSON header. No pickles."""
    meta = {"version": segment.version, "context": segment.context, "terminated": bool(segment.terminated),
            "spec_version": spec.SPEC_VERSION}
    arrays = {"frames": segment.frames, "actions": segment.actions, "logp": segment.logp,
              "rewards": segment.rewards, "bad": segment.bad,
              "meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    if segment.audio is not None:
        arrays["audio"] = segment.audio
        arrays["audio_mask"] = segment.audio_mask
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def decode_segment(data: bytes, actor: int, *, context: int, audio_shape: tuple[int, ...] | None):
    """The inverse of `encode_segment`, checked against the learner's policy: a malformed segment raises
    ValueError, never trains. `context` is the policy's frame history depth (the oldest offset), which the
    learner's index arithmetic assumes every segment carries; `audio_shape` is one step's audio feature, or None
    for a policy that does not hear."""
    from zombiesai.rl.parallel_ppo import Segment

    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        if not isinstance(meta, dict):
            raise ValueError("a segment's header is a JSON object")
        audio = z["audio"] if "audio" in z.files else None
        audio_mask = z["audio_mask"] if "audio_mask" in z.files else None
        seg = Segment(actor=actor, version=int(meta["version"]), context=int(meta["context"]), frames=z["frames"],
                      actions=z["actions"].astype(np.int64), logp=z["logp"].astype(np.float32),
                      rewards=z["rewards"].astype(np.float32), bad=z["bad"].astype(bool),
                      terminated=bool(meta["terminated"]), audio=audio, audio_mask=audio_mask)
    if meta.get("spec_version") != spec.SPEC_VERSION:
        raise ValueError(f"segment from spec {meta.get('spec_version')}, learner is on {spec.SPEC_VERSION}")
    if seg.context != context:
        raise ValueError(f"a frame context of {seg.context}, the learner's policy looks back {context}")
    n = seg.n
    if seg.frames.dtype != np.uint8 or seg.frames.shape != (seg.context + n + 1, *spec.PIXELS_SHAPE):
        raise ValueError(f"frames {seg.frames.dtype} {seg.frames.shape} for {n} steps after {seg.context}")
    if seg.actions.shape != (n, len(spec.ACTION_NVEC)) or not (seg.logp.shape == seg.rewards.shape == seg.bad.shape
                                                                == (n,)):
        raise ValueError("actions, log-probs, rewards and bad flags disagree on the segment's length")
    if (seg.actions < 0).any() or (seg.actions >= np.asarray(spec.ACTION_NVEC)).any():
        raise ValueError("an action outside its head's range")
    if audio_shape is None:
        if seg.audio is not None or seg.audio_mask is not None:
            raise ValueError("audio for a policy that does not hear")
        return seg
    # The learner stacks these straight into the network's audio input: anything but float32 of the feature's
    # shape fails there, mid-update, after the batch is gathered. Narrower types widen safely; float64 does not.
    if seg.audio is None or seg.audio_mask is None:
        raise ValueError("no audio for a policy that hears")
    if seg.audio.shape != (n + 1, *audio_shape) or seg.audio_mask.shape != (n + 1,):
        raise ValueError(f"audio {seg.audio.shape} and mask {seg.audio_mask.shape} for {n + 1} observations of "
                         f"{tuple(audio_shape)}")
    for name in ("audio", "audio_mask"):
        array = getattr(seg, name)
        if not np.can_cast(array.dtype, np.float32, casting="safe"):
            raise ValueError(f"{name} is {array.dtype}, not float32")
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} is not finite")
        setattr(seg, name, array)
    return seg


EPISODE_TEXT_FIELDS = ("reason",)  # the summary's one field that is words, not a number


def clean_episode(body: dict) -> dict:
    """What the learner keeps of a worker's episode summary: the fields episodes.jsonl and its log line use
    (parallel_ppo.EPISODE_FIELDS), numbers as finite ints or floats and the reason as a short string. Anything
    else -- a null, a string where a number goes, NaN -- is dropped, so the learner never has to guess."""
    from zombiesai.rl.parallel_ppo import EPISODE_FIELDS

    out = {}
    for key in EPISODE_FIELDS:
        value = body.get(key)
        if key in EPISODE_TEXT_FIELDS:
            if isinstance(value, str) and value:
                out[key] = value[:200]
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[key] = value
    return out


def _count(value, default: int | None = None) -> int:
    """A count from a request: an int (a JSON true is not one), else ValueError, which the handler turns into
    a 400."""
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{value!r} is not a whole number")
    return value


# ------------------------------------------------------------------------------------------------ agreement


def play_settings(fleet_root: str | Path = "runs/instances") -> dict | None:
    """What this machine's games play with -- `instances.game_config`: for the Plutonium client the config.cfg
    copied into every instance (one installed in the fleet's root by `scripts/fleet.py prep`, else the Steam
    profile's), for the steam client the profile in the instances' prefixes -- reduced to what changes an
    action's meaning. None without the game (a sim-only machine)."""
    from zombiesai.demos.game_settings import dvar, parse_config
    from zombiesai.realgame.instances import game_config

    path = game_config(fleet_root)
    if path is None:
        return None
    dvars, binds = parse_config(path.read_text(errors="replace"))
    return {"dvars": {name: dvar(dvars, name) for name in PLAY_DVARS},
            "binds": {k.lower(): v for k, v in binds.items()}}


def describe(fleet_root: str | Path = "runs/instances") -> dict:
    """This machine as a learner would judge it, for `scripts/fleet.py`: its commit, the game settings its games
    play with and where they come from, and how many of its games are running."""
    from zombiesai.realgame.instances import fleet, game_config, load_fleet

    path = game_config(fleet_root)
    out = {**provenance(), "settings": play_settings(fleet_root),
           "settings_from": None if path is None else ("installed" if path.parent == Path(fleet_root)
                                                       else "prefix" if Path(fleet_root) in path.parents else "steam"),
           "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None,
           "client": None, "instances": None, "games_running": 0}
    try:
        config = load_fleet(fleet_root)
    except FileNotFoundError:
        return out
    instances = fleet(config, say=lambda m: None)
    out["client"] = config.client
    out["instances"] = len(instances)
    out["games_running"] = sum(i.game_running() for i in instances)
    return out


def settings_differences(ours: dict, theirs: dict | None) -> list[str]:
    if theirs is None:
        return ["no World at War config.cfg on that machine: Plutonium would play with its own bindings"]
    out = [f"{name}: {theirs['dvars'].get(name)!r}, here {value!r}"
           for name, value in ours["dvars"].items() if theirs["dvars"].get(name) != value]
    for key in sorted(set(ours["binds"]) | set(theirs["binds"])):
        if ours["binds"].get(key) != theirs["binds"].get(key):
            out.append(f"bind {key}: {theirs['binds'].get(key)!r}, here {ours['binds'].get(key)!r}")
    return out


def provenance() -> dict:
    from zombiesai.store.episode_store import git_provenance

    return {"spec_version": spec.SPEC_VERSION, **git_provenance()}


def parse_address(address: str, default_host: str = "0.0.0.0") -> tuple[str, int]:
    """"host:port", ":port", "host" or "port" -> (host, port)."""
    address = address.strip()
    if address.isdigit():
        return default_host, int(address)
    host, sep, port = address.rpartition(":")
    if not sep:
        return address, DEFAULT_PORT
    return host or default_host, int(port)


def token_from_env() -> str | None:
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None


# ------------------------------------------------------------------------------------------------ learner side


@dataclass
class WorkerState:
    name: str
    first_actor: int
    actors: int = 0
    alive: int = 0
    last_seen: float = 0.0
    segments: int = 0
    episodes: int = 0
    refused: int = 0
    sent: int = 0  # as the worker counts them: what it posted, and what never arrived (the link, or a full inbox)
    lost: int = 0
    host: str = ""
    commit: str = ""
    instance: str = ""  # the worker process's random id: two PCs with one name differ in it
    settings: dict | None = field(default=None, repr=False)


class FleetServer:
    """The learner's door for other PCs' actors. Runs on its own threads; the learner loop reads `inbox` beside
    its own actors' queue and calls `set_weights` after every publish. `context` and `audio_shape` are what the
    learner's policy needs of a segment (its frame history depth; one step's audio feature, None if it does not
    hear), for `decode_segment` to check."""

    def __init__(self, address: str, token: str, *, config, run_dir: Path, settings: dict | None, context: int,
                 audio_shape: tuple[int, ...] | None, inbox_size: int = 32, say=print):
        if not token:
            raise ValueError(f"a fleet needs a shared token: set {TOKEN_ENV} on every machine")
        self.token, self.config, self.run_dir, self.say = token, config, Path(run_dir), say
        self.context, self.audio_shape = int(context), None if audio_shape is None else tuple(audio_shape)
        self.settings = settings  # None until a worker brings some, if this machine has no game
        self.provenance = provenance()
        self.inbox: queue_mod.Queue = queue_mod.Queue(maxsize=inbox_size)
        self.workers: dict[str, WorkerState] = {}
        self._lock = threading.Lock()
        self._weights: tuple[int, bytes] = (-1, b"")
        self._init_bytes = Path(config.init).read_bytes()
        self._init_sha = hashlib.sha256(self._init_bytes).hexdigest()
        host, port = parse_address(address)
        server = self

        class Handler(_Handler):
            fleet = server

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.address = self.httpd.server_address[:2]
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="fleet-server", daemon=True)

    def start(self) -> "FleetServer":
        self._thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def set_weights(self, version: int, path: Path) -> None:
        self._weights = (version, Path(path).read_bytes())

    # -- state the learner reports

    def alive(self, within_s: float = 30.0) -> list[WorkerState]:
        now = time.time()
        with self._lock:
            return [w for w in self.workers.values() if now - w.last_seen < within_s]

    def remote_actors_alive(self) -> int:
        return sum(w.alive for w in self.alive())

    def name_of(self, machine: int) -> str:
        with self._lock:
            for w in self.workers.values():
                if w.first_actor // ACTOR_BLOCK == machine:
                    return w.name
        return f"machine {machine}"

    def snapshot(self, per_machine: dict[str, dict] | None = None) -> dict:
        """For runs/<run>/fleet.json, which stays on the learner: names are fine here, unlike metrics.jsonl.
        `per_machine` is the learner's own view of each machine's last batches (parallel_ppo.MachineStats)."""
        now = time.time()
        stats = per_machine or {}
        with self._lock:
            return {"t": round(now, 1), "address": f"{self.address[0]}:{self.address[1]}",
                    "learner": {"machine": 0, "stats": stats.get("0")},
                    "workers": [{"name": w.name, "machine": w.first_actor // ACTOR_BLOCK, "first_actor": w.first_actor,
                                 "actors": w.actors, "alive": w.alive, "seen_s_ago": round(now - w.last_seen, 1),
                                 "segments": w.segments, "episodes": w.episodes, "refused": w.refused,
                                 "sent": w.sent, "lost_on_the_way": w.lost,
                                 "stats": stats.get(str(w.first_actor // ACTOR_BLOCK))}
                                for w in self.workers.values()]}

    # -- requests

    def hello(self, body: dict, host: str) -> tuple[int, dict]:
        name = str(body.get("name") or host)[:64]
        actors = body.get("actors")
        # Checked before anything is registered: the block a machine gets holds ACTOR_BLOCK actors, no more.
        if isinstance(actors, bool) or not isinstance(actors, int) or not 1 <= actors <= ACTOR_BLOCK:
            return 400, {"error": f"actors {actors!r}: a machine plays 1..{ACTOR_BLOCK}"}
        ours = self.provenance
        if body.get("spec_version") != ours["spec_version"]:
            return 409, {"error": f"spec {body.get('spec_version')} here is {ours['spec_version']}: "
                                  "check out the learner's commit on that machine"}
        if "unknown" not in (body.get("sha"), ours["sha"]) and body.get("sha") != ours["sha"]:
            return 409, {"error": f"that machine is on commit {str(body.get('sha'))[:12]}, the learner on "
                                  f"{ours['sha'][:12]}: `scripts/fleet.py prep` on the learner brings it over"}
        instance = body.get("instance")
        if not isinstance(instance, str) or not instance:
            return 400, {"error": "no worker instance id: update the worker"}
        instance = instance[:64]
        if self.config.env == "real":
            with self._lock:
                if self.settings is None and body.get("settings") is not None:
                    self.settings = body["settings"]
                    self.say(f"  fleet: {name}'s game settings are the reference (this machine has no config.cfg)")
                reference = self.settings
            if reference is not None:
                diffs = settings_differences(reference, body.get("settings"))
                if diffs:
                    return 409, {"error": "its game settings differ from the learner's -- `scripts/fleet.py prep` "
                                          "on the learner installs its config.cfg there: " + "; ".join(diffs[:8])
                                          + (f" (and {len(diffs) - 8} more)" if len(diffs) > 8 else "")}
        now = time.time()
        with self._lock:
            worker = self.workers.get(name)
            if worker is None:
                worker = WorkerState(name=name, first_actor=ACTOR_BLOCK * (len(self.workers) + 1), instance=instance)
                self.workers[name] = worker
                self.say(f"  fleet: {name} joined from {host} with {actors} actors (actors {worker.first_actor}+)")
            elif worker.instance != instance:
                # Two processes under one name would share a block of actor numbers and an RNG stream. A quiet
                # one has stopped -- this is its PC's worker, restarted -- otherwise it is another machine.
                if now - worker.last_seen < STALE_S:
                    return 409, {"error": f"another machine is already playing as {name} (from {worker.host}): "
                                          "give one of them --name"}
                self.say(f"  fleet: {name} is back from {host}, a new worker taking over actors {worker.first_actor}+")
                worker.instance, worker.alive = instance, 0
            worker.actors = actors
            worker.last_seen, worker.host, worker.commit = now, host, str(body.get("sha"))
        return 200, {"run": self.run_dir.name, "first_actor": worker.first_actor, "config": asdict(self.config),
                     "version": self._weights[0], "init_sha256": self._init_sha}

    def worker(self, name: str | None, instance: str | None) -> WorkerState | None:
        """The worker a request comes from, if it said hello -- as this process, not another under its name."""
        with self._lock:
            w = self.workers.get(name or "")
            if w is None or not instance or w.instance != instance:
                return None
            w.last_seen = time.time()
            return w

    def deliver(self, item, timeout_s: float = 5.0) -> bool:
        try:
            self.inbox.put(item, timeout=timeout_s)
            return True
        except queue_mod.Full:
            return False


class _Handler(BaseHTTPRequestHandler):
    fleet: FleetServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: the learner's log is for training
        pass

    def _reply(self, code: int, body: dict | bytes, headers: dict | None = None) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream" if isinstance(body, bytes) else "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _authorized(self) -> bool:
        given = self.headers.get("X-Fleet-Token", "")
        if hmac.compare_digest(given.encode(), self.fleet.token.encode()):
            return True
        self.close_connection = True  # the body, if any, is left unread
        self._reply(401, {"error": "wrong or missing fleet token"})
        return False

    def _body(self) -> bytes | None:
        try:
            n = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True
            self._reply(413 if n > MAX_BODY else 411, {"error": f"a body of 0..{MAX_BODY} bytes, with its length"})
            return None
        return self.rfile.read(n)

    def _json(self, data: bytes) -> dict | None:
        try:
            body = json.loads(data or b"{}")
            if isinstance(body, dict):
                return body
        except ValueError:
            pass
        self._reply(400, {"error": "expected a JSON object"})
        return None

    def _known_worker(self) -> WorkerState | None:
        """A 409 for a stranger, and for a process other than the one that said hello under that name: the
        worker ends its session on a 409 and says hello again, which settles which of two is playing."""
        worker = self.fleet.worker(self.headers.get("X-Worker"), self.headers.get(INSTANCE_HEADER))
        if worker is None:
            self._reply(409, {"error": "say hello first", "hello": True})
        return worker

    @staticmethod
    def _local_actor(local: int, worker: WorkerState) -> int:
        """One of the actors the machine said hello with, numbered from 0, or ValueError."""
        if not 0 <= local < worker.actors:
            raise ValueError(f"actor {local} of a machine with {worker.actors}")
        return local

    def do_GET(self):
        if not self._authorized():
            return
        url = urlparse(self.path)
        if url.path == "/weights":
            if self._known_worker() is None:
                return
            version, data = self.fleet._weights
            have = parse_qs(url.query).get("have", ["-2"])[0]
            if version < 0 or str(version) == have:
                self._reply(304 if version >= 0 else 204, b"", {"X-Version": version})
            else:
                self._reply(200, data, {"X-Version": version})
        elif url.path == "/init":
            if self._known_worker() is None:
                return
            self._reply(200, self.fleet._init_bytes)
        else:
            self._reply(404, {"error": f"no {url.path}"})

    def do_POST(self):
        if not self._authorized():
            return
        data = self._body()
        if data is None:
            return
        path = urlparse(self.path).path
        fleet = self.fleet
        if path == "/hello":
            body = self._json(data)
            if body is not None:
                self._reply(*fleet.hello(body, self.client_address[0]))
            return
        worker = self._known_worker()
        if worker is None:
            return
        if path == "/segment":
            try:
                local = self._local_actor(int(self.headers.get("X-Actor", "-1")), worker)
                segment = decode_segment(data, worker.first_actor + local, context=fleet.context,
                                         audio_shape=fleet.audio_shape)
            except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile) as error:
                worker.refused += 1
                self._reply(400, {"error": f"bad segment: {error}"})
                return
            if fleet.deliver(("segment", segment.actor, segment)):
                worker.segments += 1
                self._reply(200, {"ok": True})
            else:
                self._reply(503, {"error": "the learner is behind; segment dropped"})
        elif path == "/episode":
            body = self._json(data)
            if body is None:
                return
            try:
                local = self._local_actor(_count(body.get("actor")), worker)
            except ValueError as error:
                self._reply(400, {"error": f"bad episode: {error}"})
                return
            payload = clean_episode(body)
            payload["actor"] = worker.first_actor + local
            payload["machine"] = worker.name
            worker.episodes += 1
            fleet.deliver(("episode", payload["actor"], payload), timeout_s=1.0)
            self._reply(200, {"ok": True})
        elif path == "/heartbeat":
            body = self._json(data)
            if body is None:
                return
            # The actor count is the one hello registered: a heartbeat reports on those actors, never adds more.
            try:
                alive = _count(body.get("alive"), 0)
                sent, lost = _count(body.get("sent"), worker.sent), _count(body.get("dropped"), worker.lost)
            except ValueError as error:
                self._reply(400, {"error": f"bad heartbeat: {error}"})
                return
            worker.alive = min(max(alive, 0), worker.actors)
            worker.sent, worker.lost = max(sent, 0), max(lost, 0)
            self._reply(200, {"ok": True, "version": fleet._weights[0]})
        else:
            self._reply(404, {"error": f"no {path}"})


# ------------------------------------------------------------------------------------------------ worker side


class FleetClient:
    """The worker's side of the protocol, over plain urllib."""

    def __init__(self, learner: str, token: str, name: str, timeout_s: float = 30.0):
        host, port = parse_address(learner, default_host="127.0.0.1")
        self.base, self.token, self.name, self.timeout_s = f"http://{host}:{port}", token, name, timeout_s
        # Drawn once per process: what tells this worker apart from another PC (or a second copy) with its name.
        self.instance = uuid.uuid4().hex

    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None):
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-Fleet-Token": self.token, "X-Worker": self.name,
                                              INSTANCE_HEADER: self.instance, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def _post_json(self, path: str, body: dict) -> tuple[int, dict]:
        code, data, _ = self._request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            return code, json.loads(data or b"{}")
        except ValueError:
            return code, {"error": data[:200].decode(errors="replace")}

    def hello(self, actors: int, settings: dict | None) -> dict:
        code, body = self._post_json("/hello", {"name": self.name, "instance": self.instance, "actors": actors,
                                                "settings": settings, **provenance()})
        if code == 401:
            raise FleetError(f"the learner refused the fleet token: set the same {TOKEN_ENV} on both machines")
        if code != 200:
            raise FleetError(body.get("error") or f"hello: HTTP {code}")
        return body

    def init(self) -> bytes:
        code, data, _ = self._request("GET", "/init")
        if code != 200:
            raise ConnectionError(f"init: HTTP {code}")
        return data

    def weights(self, have: int) -> tuple[int, int, bytes]:
        """(HTTP status, version, torch blob): 200 with the blob when the learner has newer weights than `have`,
        304 when it does not, 204 before its first publish."""
        code, data, headers = self._request("GET", f"/weights?have={have}")
        try:
            version = int(headers.get("X-Version", "-1")) if headers else -1
        except ValueError:  # not our learner's answer
            version = -1
        return code, version, data if code == 200 else b""

    def segment(self, local_actor: int, payload: bytes) -> int:
        code, _, _ = self._request("POST", "/segment", payload, {"X-Actor": str(local_actor),
                                                                 "Content-Type": "application/octet-stream"})
        return code

    def episode(self, payload: dict) -> int:
        return self._post_json("/episode", payload)[0]

    def heartbeat(self, actors: int, alive: int, sent: int = 0, dropped: int = 0) -> int:
        return self._post_json("/heartbeat", {"actors": actors, "alive": alive, "sent": sent, "dropped": dropped})[0]


@dataclass
class WorkerOptions:
    actors: int = 4
    fleet_root: str = "runs/instances"
    out_root: str = "runs/fleet"
    overrides: dict = field(default_factory=dict)  # RLConfig fields this machine sets for itself
    lost_s: float = 60.0  # no contact with the learner this long: stop the actors and wait for the next run
    poll_s: float = 1.0  # how often to ask for new weights
    retry_s: float = 10.0  # how often to look for a learner while there is none


class _Gone(Exception):
    """The learner went away (or forgot us): end this session."""


# What a request to the learner can raise when it cannot be reached, or something else answers on its port.
_NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, OSError, http.client.HTTPException)


def response_verdict(code: int, what: str, success: tuple[int, ...] = (200,)) -> tuple[bool, str | None]:
    """What one answer to the worker says about its session: (whether it is contact -- the learner is there and
    still takes this machine -- and why the session is over, or None).

    Only a success is contact. Anything else that answers -- a 404 from some other service on the port, a 400, a
    503 -- proves nothing about the learner, so it leaves the `lost_s` watchdog running: a learner that refuses
    every request must stop this machine's games as surely as one that is gone. 409 is the learner not knowing
    this machine any more (a new run: say hello again), 401 the token no longer matching (it restarted with
    another one), and both end the session at once."""
    if code in success:
        return True, None
    if code == 409:
        return False, f"the learner no longer knows this machine ({what}): a new run"
    if code == 401:
        return False, (f"the learner no longer takes this machine's fleet token ({what}): did it restart with "
                       f"another {TOKEN_ENV}?")
    return False, None


class FleetWorker:
    """Plays this machine's games for a learner elsewhere: hello, the run's starting checkpoint and config, then
    the same actor processes `train()` would start, with their segments forwarded and weights pulled."""

    def __init__(self, client: FleetClient, options: WorkerOptions, *, settings: dict | None = None, say=print):
        self.client, self.options, self.say = client, options, say
        self.settings = settings

    @property
    def status_path(self) -> Path:
        """What this worker is doing, rewritten every few seconds: publish_live.py puts it on the site."""
        return Path(self.options.out_root) / STATUS_FILE

    def _status(self, state: str, **fields) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.status_path, json.dumps({"at": round(time.time(), 1), "state": state,
                                                        "actors": self.options.actors, **fields}).encode())
        except OSError:
            pass  # the status is for people watching; never a reason to stop playing

    def run(self, stop: threading.Event | None = None, *, once: bool = False) -> None:
        """Serve every run the learner starts, until `stop` (or after one run, with `once`)."""
        stop = stop or threading.Event()
        waiting_said = refused = False
        try:
            while not stop.is_set():
                try:
                    hello = self.client.hello(self.options.actors, self.settings)
                except _NETWORK_ERRORS as error:
                    if not waiting_said:
                        self.say(f"waiting for the learner at {self.client.base} ({getattr(error, 'reason', error)})")
                        waiting_said = True
                    self._status("waiting")
                    stop.wait(self.options.retry_s)
                    continue
                except FleetError as error:
                    self._status("refused", reason=str(error)[:300])
                    refused = True
                    raise
                waiting_said = False
                try:
                    self.session(hello, stop)
                except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as error:
                    self.say(f"could not join the run: {error}")
                if once:
                    return
                self._status("waiting")
                stop.wait(self.options.retry_s)
        finally:
            if not refused:  # a refusal stays on the page: it is the one state a person has to act on
                self._status("stopped")

    def session(self, hello: dict, stop: threading.Event) -> None:
        from zombiesai.rl.parallel_ppo import RLConfig, actor_main

        name = Path(str(hello["run"])).name
        if name in ("", ".", ".."):
            raise ConnectionError(f"the learner named its run {hello['run']!r}")
        run_dir = Path(self.options.out_root) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        init = run_dir / "init.pt"
        if not init.exists() or hashlib.sha256(init.read_bytes()).hexdigest() != hello["init_sha256"]:
            data = self.client.init()
            if hashlib.sha256(data).hexdigest() != hello["init_sha256"]:
                raise ConnectionError("the starting checkpoint arrived damaged")
            _atomic_write(init, data)
        (run_dir / "weights.pt").unlink(missing_ok=True)  # never act on a previous run's weights
        learner_config = RLConfig(**hello["config"])
        if learner_config.env == "real":
            from zombiesai.realgame.instances import load_fleet

            have = load_fleet(self.options.fleet_root).n
            if have < self.options.actors:
                raise FleetError(f"{self.options.actors} actors but this machine's fleet has {have} instances: "
                                 f"scripts/instances.py up --n {self.options.actors}")
        config = replace(learner_config, init=str(init), n_actors=self.options.actors,
                         fleet_root=self.options.fleet_root, seed=learner_config.seed + hello["first_actor"],
                         **self.options.overrides)
        (run_dir / "config.json").write_text(json.dumps({**asdict(config), "first_actor": hello["first_actor"]}, indent=2))
        self.say(f"joined run {hello['run']} as actors {hello['first_actor']}..{hello['first_actor'] + config.n_actors - 1}")

        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # as train() does, before the actors import numpy
        ctx = mp.get_context("spawn")
        out = ctx.Queue(maxsize=max(8, 4 * config.n_actors))
        actors_stop = ctx.Event()
        procs: dict[int, mp.Process] = {}
        restarts = {i: 0 for i in range(config.n_actors)}
        outbox: queue_mod.Queue = queue_mod.Queue(maxsize=max(4, 2 * config.n_actors))
        state = {"last_ok": time.time(), "version": -1, "sent": 0, "dropped": 0, "gone": None, "episodes": 0,
                 "last_round": None, "best_round": None}
        counting = threading.Lock()  # "sent" and "dropped" move on the sender thread and on this one
        session_over = threading.Event()

        def count(key: str, n: int = 1) -> None:
            if n:
                with counting:
                    state[key] += n

        def check(code: int, what: str, success: tuple[int, ...] = (200,)) -> None:
            contact, gone = response_verdict(code, what, success)
            if contact:
                state["last_ok"] = time.time()
            elif gone is not None:
                state["gone"] = state["gone"] or gone
                session_over.set()

        def send_segments() -> None:
            while not session_over.is_set():
                try:
                    index, seg = outbox.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                try:
                    code = self.client.segment(index, encode_segment(seg))
                except _NETWORK_ERRORS:
                    count("dropped")
                    continue
                check(code, "segment")
                count("sent" if code == 200 else "dropped")

        def pull_weights() -> None:
            while not session_over.is_set():
                try:
                    code, version, blob = self.client.weights(state["version"])
                    check(code, "weights", success=(200, 204, 304))
                    if code == 200 and version >= 0:
                        _atomic_write(run_dir / "weights.pt", blob)
                        state["version"] = version
                except _NETWORK_ERRORS:
                    pass
                session_over.wait(self.options.poll_s)

        def heartbeat() -> None:
            while not session_over.is_set():
                alive = sum(p.is_alive() for p in list(procs.values()))
                self._status("playing", run=name, alive=alive,
                             **{k: state[k] for k in ("sent", "dropped", "version", "episodes", "last_round",
                                                      "best_round")})
                try:
                    check(self.client.heartbeat(config.n_actors, alive, state["sent"], state["dropped"]), "heartbeat")
                except _NETWORK_ERRORS:
                    pass
                session_over.wait(5.0)

        def spawn(i: int) -> None:
            p = ctx.Process(target=actor_main, args=(i, config, str(run_dir), out, actors_stop), daemon=True,
                            name=f"actor-{i}")
            p.start()
            procs[i] = p

        threads = [threading.Thread(target=f, daemon=True, name=f"fleet-{f.__name__}")
                   for f in (send_segments, pull_weights, heartbeat)]
        for t in threads:
            t.start()
        for i in range(config.n_actors):
            spawn(i)
        try:
            while not stop.is_set() and not session_over.is_set():
                if time.time() - state["last_ok"] > self.options.lost_s:
                    state["gone"] = f"no word from the learner for {self.options.lost_s:.0f}s"
                    break
                try:
                    kind, i, payload = out.get(timeout=0.5)
                except queue_mod.Empty:
                    kind = None
                if kind == "segment":
                    count("dropped", _offer_latest(outbox, (i, payload)))
                elif kind == "episode":
                    state["episodes"] += 1
                    reached = payload.get("round_reached")
                    if isinstance(reached, (int, float)):
                        state["last_round"] = reached
                        state["best_round"] = max(reached, state["best_round"] or reached)
                    try:
                        check(self.client.episode({**payload, "actor": i}), "episode")
                    except _NETWORK_ERRORS:
                        pass
                    self.say(f"  actor {i} episode {payload.get('episode')}: round {payload.get('round_reached', '?')}")
                elif kind == "error":
                    self.say(f"  actor {i} failed:\n{payload}")
                for j, p in list(procs.items()):
                    if not p.is_alive() and not actors_stop.is_set():
                        if restarts[j] >= config.actor_restarts:
                            self.say(f"  actor {j} has died {restarts[j]} times; leaving it down")
                            del procs[j]
                            continue
                        restarts[j] += 1
                        self.say(f"  actor {j} is down; restarting it ({restarts[j]}/{config.actor_restarts})")
                        spawn(j)
                if not procs:
                    state["gone"] = "every actor here is down"
                    break
        except KeyboardInterrupt:
            stop.set()
        finally:
            session_over.set()
            actors_stop.set()
            deadline = time.time() + 20
            while any(p.is_alive() for p in procs.values()) and time.time() < deadline:
                try:  # keep draining so no actor blocks on a full queue while it shuts down
                    out.get(timeout=0.2)
                except queue_mod.Empty:
                    pass
            for p in procs.values():
                if p.is_alive():
                    p.terminate()
            for t in threads:
                t.join(timeout=self.options.poll_s + 1)
            self.say(f"left run {hello['run']}: {state['sent']} segments sent, {state['dropped']} dropped"
                     + (f" ({state['gone']})" if state["gone"] else ""))


def _offer_latest(q: queue_mod.Queue, item) -> int:
    """Queue a segment for sending; when the link is behind, the oldest waiting segment goes -- it is the one
    most likely to be past the learner's lag limit by the time it arrives. Returns how many went, for the
    worker's `dropped`: a segment thrown away here is as lost to training as one the link lost."""
    evicted = 0
    while True:
        try:
            q.put_nowait(item)
            return evicted
        except queue_mod.Full:
            try:
                q.get_nowait()
                evicted += 1
            except queue_mod.Empty:
                pass


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
