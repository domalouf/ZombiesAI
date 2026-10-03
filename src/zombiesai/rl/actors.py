"""One actor process per game: play in real time, ship segments and episode summaries to whoever is listening
(the learner here, or a fleet worker's forwarder), with actions from this machine's inference server
(rl/inference.py) -- or, with `inference="local"`, from a copy of the policy of its own.

Nothing here imports torch at module level: with the server, an actor is numpy, its game and its frame history,
~600 MB lighter than an actor with a policy inside it. `actor_main` is the process; rl/pool.py starts them."""

import json
import os
import queue as queue_mod
import time
import traceback
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.rl.config import RLConfig
from zombiesai.rl.inference import HEADS, NEUTRAL, WIDTH, InferenceClient, InferenceLink, LocalPolicy
from zombiesai.rl.segments import FrameHistory, Segment


# ------------------------------------------------------------------------------------------------ actor envs


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
    if config.env == "synthetic":
        from zombiesai.synthetic import SyntheticActorEnv

        hears = audio_features is not None and config.hear
        env = SyntheticActorEnv(config.seed + index, config.synthetic or None,
                                audio_shape=audio_features.shape if hears else None)
        return Paced(env, config.synthetic_hz) if config.synthetic_hz > 0 else env
    if config.env == "real":
        return make_real_env(config, index, audio_features)
    raise ValueError(f"unknown env {config.env!r}")


class Paced:
    """The synthetic stand-in on the real game's clock: one step per tick on absolute deadlines, and a step that
    starts after its deadline is flagged `bad` with the schedule moving on from now (skip, never catch up) -- as
    RealGameEnv does. A rehearsal paced like this shows what the real-time guard and the late-step count will do
    with as many actors as the real run will have."""

    def __init__(self, env, hz: float):
        self.env, self.period = env, 1.0 / hz
        self.capture = getattr(env, "capture", None)
        self._next: float | None = None

    def reset(self):
        self._next = None
        return self.env.reset()

    def step(self, action):
        now = time.monotonic()
        if self._next is None:
            self._next = now + self.period
        late = now > self._next
        if late:
            self._next = now
        else:
            time.sleep(self._next - now)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._next += self.period
        if late:
            info = {**info, "bad": True}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


# ------------------------------------------------------------------------------------------------ actor


def actor_main(index: int, config: RLConfig, run_dir: str, out: "mp.Queue", stop,
               link: InferenceLink | None = None) -> None:
    """One actor process: play, ship segments and episode summaries; actions from the server behind `link`, or
    from a policy of its own when there is none."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        _actor_loop(index, config, Path(run_dir), out, stop, link)
    except Exception:  # noqa: BLE001 -- reported to the learner, which decides whether to restart us
        out.put(("error", index, traceback.format_exc()))


def _actor_loop(index: int, config: RLConfig, run_dir: Path, out, stop, link: InferenceLink | None = None) -> None:
    from zombiesai.demos.hearing import AudioFeatureConfig, silence

    policy = InferenceClient(link, index) if link is not None else LocalPolicy(config, run_dir)
    shape = policy.shape
    audio_features = AudioFeatureConfig(**shape.audio_features) if shape.audio_features is not None else None
    quiet = silence(audio_features) if audio_features is not None else None
    history = FrameHistory(shape.offsets)
    # The actor's own random stream, seeded as it always was: the Gumbel noise of every head, every step, drawn here
    # and sent with the frames, so the action does not depend on who ran the network (rl/inference.py).
    rng = np.random.default_rng(config.seed * 7919 + index)
    env = make_actor_env(config, index, audio_features)
    episode = 0
    writer = None
    try:
        obs, _ = env.reset()
        history.reset(obs["pixels"])
        while not stop.is_set():
            version = policy.refresh()
            if version < 0:
                time.sleep(0.1)  # the learner has not published yet, or the server is still starting
                continue
            frames = history.context() + [history.frames[-1]]
            actions, logps, rewards, bad, heard, masks = [], [], [], [], [], []
            versions, waited_ms, timeouts = [], [], 0
            terminated = ended = False

            def hear(o):
                if audio_features is None:
                    return
                a = o.get("audio")
                heard.append(quiet if a is None else np.asarray(a, np.float32))
                masks.append(float(o.get("audio_mask", 1.0)) if a is not None else 0.0)

            hear(obs)
            for _ in range(config.segment_steps):
                noise = rng.random((HEADS, WIDTH), dtype=np.float32)
                asked = time.perf_counter()
                reply = policy.act(history.stack(), heard[-1] if heard else None, masks[-1] if masks else None,
                                   noise)
                waited_ms.append(1000.0 * (time.perf_counter() - asked))
                if reply is None:  # no answer in time: nothing held, and nobody chose this step
                    action, logp, forced = NEUTRAL.copy(), 0.0, True
                    timeouts += 1
                else:
                    action, logp, used = reply
                    versions.append(used)
                    forced = False
                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                logps.append(logp)
                rewards.append(reward)
                bad.append(forced or bool(info.get("bad", False)))
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
            # With a server the version can change from one step to the next: the segment answers for its oldest,
            # so the learner's lag limit errs on the safe side (the per-step log-probs keep the ratio exact).
            segment = Segment(
                actor=index, version=min(versions) if versions else version, context=history.depth,
                frames=np.stack(frames), actions=np.stack(actions), logp=np.asarray(logps, np.float32),
                rewards=np.asarray(rewards, np.float32), bad=np.asarray(bad, bool), terminated=bool(terminated),
                audio=np.stack(heard) if heard else None,
                audio_mask=np.asarray(masks, np.float32) if heard else None,
            )
            _offer(out, ("segment", index, segment))
            _note(out, ("latency", index, {"ms": np.asarray(waited_ms, np.float32), "timeouts": timeouts}))
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
        policy.close()


def _note(out, item) -> None:
    """Something the learner may like to know but can do without (how long actions took): never waits at all."""
    try:
        out.put_nowait(item)
    except queue_mod.Full:
        pass


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

    source = env.capture.describe() if getattr(env, "capture", None) is not None else {"kind": "synthetic"}
    return ClipWriter(run_dir / "episodes" / f"a{index}_ep{episode:05d}", source={**source, "actor": index},
                      label_source="play", config={"decision_hz": spec.DECISION_HZ})
