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
  RNG seed moves by the same offset, so two machines never play the same random stream.
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
import io
import json
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
import urllib.error
import urllib.request
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


def decode_segment(data: bytes, actor: int):
    """The inverse of `encode_segment`, checked for shape: a malformed segment raises ValueError, never trains."""
    from zombiesai.rl.parallel_ppo import Segment

    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        audio = z["audio"] if "audio" in z.files else None
        audio_mask = z["audio_mask"] if "audio_mask" in z.files else None
        seg = Segment(actor=actor, version=int(meta["version"]), context=int(meta["context"]), frames=z["frames"],
                      actions=z["actions"].astype(np.int64), logp=z["logp"].astype(np.float32),
                      rewards=z["rewards"].astype(np.float32), bad=z["bad"].astype(bool),
                      terminated=bool(meta["terminated"]), audio=audio, audio_mask=audio_mask)
    if meta.get("spec_version") != spec.SPEC_VERSION:
        raise ValueError(f"segment from spec {meta.get('spec_version')}, learner is on {spec.SPEC_VERSION}")
    n = seg.n
    if seg.frames.dtype != np.uint8 or seg.frames.shape != (seg.context + n + 1, *spec.PIXELS_SHAPE):
        raise ValueError(f"frames {seg.frames.dtype} {seg.frames.shape} for {n} steps after {seg.context}")
    if seg.actions.shape != (n, len(spec.ACTION_NVEC)) or not (len(seg.logp) == len(seg.rewards) == len(seg.bad) == n):
        raise ValueError("actions, log-probs, rewards and bad flags disagree on the segment's length")
    if (seg.actions < 0).any() or (seg.actions >= np.asarray(spec.ACTION_NVEC)).any():
        raise ValueError("an action outside its head's range")
    if seg.audio is not None and (seg.audio_mask is None or len(seg.audio) != n + 1 or len(seg.audio_mask) != n + 1):
        raise ValueError("audio does not have one feature per observation")
    return seg


# ------------------------------------------------------------------------------------------------ agreement


def play_settings() -> dict | None:
    """What this machine's game plays with: the Steam profile's config.cfg that `instances.py` copies into every
    instance, reduced to what changes an action's meaning. None without the game (a sim-only machine)."""
    from zombiesai.demos.game_settings import dvar, parse_config, candidate_configs

    found = candidate_configs()
    if not found:
        return None
    dvars, binds = parse_config(found[0].read_text(errors="replace"))
    return {"dvars": {name: dvar(dvars, name) for name in PLAY_DVARS},
            "binds": {k.lower(): v for k, v in binds.items()}}


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
    host: str = ""
    commit: str = ""
    settings: dict | None = field(default=None, repr=False)


class FleetServer:
    """The learner's door for other PCs' actors. Runs on its own threads; the learner loop reads `inbox` beside
    its own actors' queue and calls `set_weights` after every publish."""

    def __init__(self, address: str, token: str, *, config, run_dir: Path, settings: dict | None,
                 inbox_size: int = 32, say=print):
        if not token:
            raise ValueError(f"a fleet needs a shared token: set {TOKEN_ENV} on every machine")
        self.token, self.config, self.run_dir, self.say = token, config, Path(run_dir), say
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

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            return {"t": round(now, 1), "address": f"{self.address[0]}:{self.address[1]}",
                    "workers": [{"name": w.name, "first_actor": w.first_actor, "actors": w.actors, "alive": w.alive,
                                 "seen_s_ago": round(now - w.last_seen, 1), "segments": w.segments,
                                 "episodes": w.episodes, "refused": w.refused}
                                for w in self.workers.values()]}

    # -- requests

    def hello(self, body: dict, host: str) -> tuple[int, dict]:
        name = str(body.get("name") or host)[:64]
        ours = self.provenance
        if body.get("spec_version") != ours["spec_version"]:
            return 409, {"error": f"spec {body.get('spec_version')} here is {ours['spec_version']}: "
                                  "check out the learner's commit on that machine"}
        if "unknown" not in (body.get("sha"), ours["sha"]) and body.get("sha") != ours["sha"]:
            return 409, {"error": f"that machine is on commit {str(body.get('sha'))[:12]}, the learner on "
                                  f"{ours['sha'][:12]}: git pull (or check out the same commit) and uv sync"}
        if self.config.env == "real":
            with self._lock:
                if self.settings is None and body.get("settings") is not None:
                    self.settings = body["settings"]
                    self.say(f"  fleet: {name}'s game settings are the reference (this machine has no config.cfg)")
                reference = self.settings
            if reference is not None:
                diffs = settings_differences(reference, body.get("settings"))
                if diffs:
                    return 409, {"error": "its game settings differ from the learner's -- copy the learner's "
                                          "Steam config.cfg there: " + "; ".join(diffs[:8])
                                          + (f" (and {len(diffs) - 8} more)" if len(diffs) > 8 else "")}
        with self._lock:
            worker = self.workers.get(name)
            if worker is None:
                worker = WorkerState(name=name, first_actor=ACTOR_BLOCK * (len(self.workers) + 1))
                self.workers[name] = worker
                self.say(f"  fleet: {name} joined from {host} with {body.get('actors')} actors "
                         f"(actors {worker.first_actor}+)")
            worker.actors = int(body.get("actors") or 0)
            worker.last_seen, worker.host, worker.commit = time.time(), host, str(body.get("sha"))
        if worker.actors > ACTOR_BLOCK:
            return 400, {"error": f"at most {ACTOR_BLOCK} actors a machine"}
        return 200, {"run": self.run_dir.name, "first_actor": worker.first_actor, "config": asdict(self.config),
                     "version": self._weights[0], "init_sha256": self._init_sha}

    def worker(self, name: str | None) -> WorkerState | None:
        with self._lock:
            w = self.workers.get(name or "")
            if w is not None:
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
        worker = self.fleet.worker(self.headers.get("X-Worker"))
        if worker is None:
            self._reply(409, {"error": "say hello first", "hello": True})
        return worker

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
                local = int(self.headers.get("X-Actor", "-1"))
                if not 0 <= local < max(worker.actors, 1):
                    raise ValueError(f"actor {local} of a machine with {worker.actors}")
                segment = decode_segment(data, worker.first_actor + local)
            except (ValueError, KeyError, OSError) as error:
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
            local = int(body.get("actor", 0))
            payload = {k: v for k, v in body.items() if isinstance(v, (int, float, str, bool)) or v is None}
            payload["actor"] = worker.first_actor + local
            payload["machine"] = worker.name
            worker.episodes += 1
            fleet.deliver(("episode", payload["actor"], payload), timeout_s=1.0)
            self._reply(200, {"ok": True})
        elif path == "/heartbeat":
            body = self._json(data)
            if body is None:
                return
            worker.alive = int(body.get("alive", 0))
            worker.actors = int(body.get("actors", worker.actors))
            self._reply(200, {"ok": True, "version": fleet._weights[0]})
        else:
            self._reply(404, {"error": f"no {path}"})


# ------------------------------------------------------------------------------------------------ worker side


class FleetClient:
    """The worker's side of the protocol, over plain urllib."""

    def __init__(self, learner: str, token: str, name: str, timeout_s: float = 30.0):
        host, port = parse_address(learner, default_host="127.0.0.1")
        self.base, self.token, self.name, self.timeout_s = f"http://{host}:{port}", token, name, timeout_s

    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None):
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-Fleet-Token": self.token, "X-Worker": self.name, **(headers or {})})
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
        code, body = self._post_json("/hello", {"name": self.name, "actors": actors, "settings": settings,
                                                **provenance()})
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
        return code, int(headers.get("X-Version", "-1") if headers else -1), data if code == 200 else b""

    def segment(self, local_actor: int, payload: bytes) -> int:
        code, _, _ = self._request("POST", "/segment", payload, {"X-Actor": str(local_actor),
                                                                 "Content-Type": "application/octet-stream"})
        return code

    def episode(self, payload: dict) -> int:
        return self._post_json("/episode", payload)[0]

    def heartbeat(self, actors: int, alive: int) -> int:
        return self._post_json("/heartbeat", {"actors": actors, "alive": alive})[0]


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
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
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
                except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
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
        session_over = threading.Event()

        def ok() -> None:
            state["last_ok"] = time.time()

        def check(code: int, what: str) -> None:
            if code == 409:
                state["gone"] = f"the learner no longer knows this machine ({what}): a new run"
                session_over.set()
            elif code < 500:
                ok()

        def send_segments() -> None:
            while not session_over.is_set():
                try:
                    index, seg = outbox.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                try:
                    code = self.client.segment(index, encode_segment(seg))
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    state["dropped"] += 1
                    continue
                check(code, "segment")
                if code == 200:
                    state["sent"] += 1
                else:
                    state["dropped"] += 1

        def pull_weights() -> None:
            while not session_over.is_set():
                try:
                    code, version, blob = self.client.weights(state["version"])
                    check(code, "weights")
                    if code == 200:
                        _atomic_write(run_dir / "weights.pt", blob)
                        state["version"] = version
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    pass
                session_over.wait(self.options.poll_s)

        def heartbeat() -> None:
            while not session_over.is_set():
                alive = sum(p.is_alive() for p in list(procs.values()))
                self._status("playing", run=name, alive=alive,
                             **{k: state[k] for k in ("sent", "dropped", "version", "episodes", "last_round",
                                                      "best_round")})
                try:
                    check(self.client.heartbeat(config.n_actors, alive), "heartbeat")
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
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
                    _offer_latest(outbox, (i, payload))
                elif kind == "episode":
                    state["episodes"] += 1
                    reached = payload.get("round_reached")
                    if isinstance(reached, (int, float)):
                        state["last_round"] = reached
                        state["best_round"] = max(reached, state["best_round"] or reached)
                    try:
                        check(self.client.episode({**payload, "actor": i}), "episode")
                    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
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


def _offer_latest(q: queue_mod.Queue, item) -> None:
    """Queue a segment for sending; when the link is behind, the oldest waiting segment goes -- it is the one
    most likely to be past the learner's lag limit by the time it arrives."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue_mod.Full:
            try:
                q.get_nowait()
            except queue_mod.Empty:
                pass


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
