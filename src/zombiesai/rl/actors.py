"""One actor process per game: play in real time, ship segments and episode summaries to whoever is listening
(the learner here, or a fleet worker's forwarder), and pick up new weights between segments."""

import json
import os
import queue as queue_mod
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from zombiesai import spec
from zombiesai.demos.hud_crops import hud_view
from zombiesai.rl.config import RLConfig
from zombiesai.rl.distributions import FactoredCategorical
from zombiesai.rl.segments import FrameHistory, Segment
from zombiesai.rl.weights import WeightFollower


# ------------------------------------------------------------------------------------------------ actor envs


def make_real_env(config: RLConfig, index: int, audio_features=None):
    """The game on instance `index` of the fleet at `config.fleet_root`, as a RealGameEnv."""
    from zombiesai.demos.inputs import DEFAULT_BINDINGS
    from zombiesai.realgame.dispatch import ActionDispatcher, DispatchConfig
    from zombiesai.realgame.env import RealGameEnv
    from zombiesai.realgame.instances import Instance, load_fleet, specs
    from zombiesai.realgame.console import console_open
    from zombiesai.realgame.end_screen import read_end_screen
    from zombiesai.realgame.scoreboard import scoreboard_shown
    from zombiesai.realgame.xtest import console_command, tap_key

    fleet = load_fleet(config.fleet_root)
    instance = Instance(fleet, specs(fleet)[index], say=lambda m: print(f"[actor {index}] {m}", flush=True))
    bindings_path = Path(config.bindings)
    bindings = json.loads(bindings_path.read_text()) if bindings_path.exists() else dict(DEFAULT_BINDINGS)
    sink = instance.sink()
    cpd = counts_per_degree(config, say=lambda m: print(f"[actor {index}] {m}", flush=True))
    # Sampled bins are whole turns the policy chose, so no dead zone: the motor's is for play_real's mean look,
    # where it keeps a near-zero average from drifting the view, and here it would shave 2-degree turns by a fifth.
    smoothing = config.look_smoothing_s
    dispatcher = ActionDispatcher(sink, DispatchConfig(counts_per_degree=cpd, bindings=bindings),
                                  motor=smoothing > 0, motor_time_constant_s=smoothing or 0.08,
                                  motor_dead_zone_deg_s=(0.0, 0.0))
    hearing = None
    if audio_features is not None and config.hear:
        from zombiesai.demos.audio import PulseMonitorStream
        from zombiesai.demos.hearing import LiveAudio

        hearing = LiveAudio(PulseMonitorStream(instance.spec.monitor, stream_name=f"actor {index}"),
                            audio_features).start()
    from zombiesai.rl.best_episode import VIDEO_HEIGHT

    capture = instance.capture(video_height=VIDEO_HEIGHT if config.record_best else None)

    def shows_console() -> bool:
        return console_open((capture.last_hud or {}).get("console"))

    def looks_open() -> bool:  # a fresh look, for the console typing between frames
        capture.read()
        return shows_console()

    env = RealGameEnv(capture, dispatcher, focus=instance.focuser(sink),
                       console=lambda command: console_command(sink, command, is_open=looks_open),
                       # 30 ms taps are shorter than the game notices for some keys (the scores key); 150 ms is sure
                       restart=instance.restart_game, press=lambda key: tap_key(sink, key, hold_s=0.15),
                       console_open=shows_console,
                       downed=lambda: scoreboard_shown((capture.last_hud or {}).get("scores")),
                       end_screen=lambda: read_end_screen(capture.grab()),
                       hearing=hearing, say=lambda m: print(f"[actor {index}] {m}", flush=True))
    # The sink this game alone plays into, for the best game's film to have its sound; a fleet started without
    # sinks of its own plays into the desktop's, mixed with every other game, so its films stay silent.
    env.sound_monitor = instance.spec.monitor if fleet.audio_sinks else None
    return env


# What --counts-per-degree defaulted to before it was derived: sensitivity 5 x m_yaw 0.022, the demos' settings.
FALLBACK_COUNTS_PER_DEGREE = 9.09


def counts_per_degree(config: RLConfig, say=print) -> float:
    """Mouse counts per degree for this machine's games. The engine turns `sensitivity * m_yaw` degrees a
    count, so derived from the config the games play with, a look bin means the same turn on every PC; the
    learner already turns away a PC whose sensitivity differs (rl/fleet.py), so every PC derives the same
    number. An explicit `counts_per_degree` (a calibration) wins, but a large gap from the config is said."""
    from zombiesai.demos.game_settings import CPD_MISMATCH, implied_counts_per_degree
    from zombiesai.rl.fleet import play_settings

    settings = play_settings(config.fleet_root)
    implied = implied_counts_per_degree(settings["dvars"]) if settings is not None else None
    if config.counts_per_degree is not None:
        if implied and abs(config.counts_per_degree / implied - 1.0) > CPD_MISMATCH:
            say(f"counts_per_degree {config.counts_per_degree:g}, but the game's sensitivity implies "
                f"{implied:.2f}: every turn will be {config.counts_per_degree / implied:.2f}x what the policy chose")
        return config.counts_per_degree
    if implied is None:
        say(f"no sensitivity and m_yaw in the game's config: assuming {FALLBACK_COUNTS_PER_DEGREE} counts per degree")
        return FALLBACK_COUNTS_PER_DEGREE
    return implied


def make_actor_env(config: RLConfig, index: int, audio_features=None):
    if config.env == "synthetic":
        from zombiesai.synthetic import SyntheticActorEnv

        hears = audio_features is not None and config.hear
        return SyntheticActorEnv(config.seed + index, config.synthetic or None,
                                 audio_shape=audio_features.shape if hears else None)
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
    from zombiesai.rl.best_episode import BestEpisodeRecorder

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
    best = BestEpisodeRecorder(run_dir, index, say=lambda m: print(f"[actor {index}] {m}", flush=True),
                               sound=_sound_stream(env, index)) if config.record_best else None
    # A learner killed outright (SIGKILL, a crash) never sets `stop`, and its actors are re-parented: an actor
    # whose parent changed stops as if told to, rather than play on for nobody with keys held in a live game.
    parent = os.getppid()

    def stopping() -> bool:
        return stop.is_set() or os.getppid() != parent

    from zombiesai.realgame.env import Stopped

    if hasattr(env, "should_stop"):  # a reset can wait minutes for a map load or a relaunch; a stop cannot
        env.should_stop = stopping
    try:
        obs, _ = env.reset()
        history.reset(obs["pixels"])
        if best is not None:
            best.start(episode)
            best.add(*_video_frame(env, obs))
        while not stopping():
            version = follower.poll(net)
            if version < 0:
                time.sleep(0.1)  # the learner has not published yet
                continue
            frames = history.context() + [history.frames[-1]]
            actions, logps, rewards, bad, heard, masks, views = [], [], [], [], [], [], []
            terminated = ended = False

            def hear(o):
                if audio_features is not None:
                    a = o.get("audio")
                    heard.append(quiet if a is None else np.asarray(a, np.float32))
                    masks.append(float(o.get("audio_mask", 1.0)) if a is not None else 0.0)
                if bc_config.use_hud_view:
                    v = o.get("hud_view")
                    views.append(hud_view(None) if v is None else np.asarray(v, np.uint8))

            hear(obs)
            for _ in range(config.segment_steps):
                with torch.no_grad():
                    pixels = torch.from_numpy(history.stack()[None]).to(device)
                    audio = mask = hud = None
                    if audio_features is not None:
                        audio = torch.from_numpy(heard[-1][None]).to(device)
                        mask = torch.tensor([masks[-1]], device=device)
                    if views:
                        hud = torch.from_numpy(views[-1][None]).to(device)
                    logits, _, _ = net(pixels, None, audio, mask, hud)
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
                if best is not None:
                    best.add(*_video_frame(env, obs))
                history.push(obs["pixels"])
                frames.append(history.frames[-1])
                hear(obs)
                if terminated or truncated:
                    _offer(out, ("episode", index, {**info.get("episode", {}), "actor": index, "episode": episode}))
                    ended = True
                    break
                if stopping():
                    break
            if not actions:
                break
            segment = Segment(
                actor=index, version=version, context=history.depth, frames=np.stack(frames),
                actions=np.stack(actions), logp=np.asarray(logps, np.float32),
                rewards=np.asarray(rewards, np.float32), bad=np.asarray(bad, bool), terminated=bool(terminated),
                audio=np.stack(heard) if heard else None,
                audio_mask=np.asarray(masks, np.float32) if heard else None,
                hud_view=np.stack(views) if views else None,
            )
            _offer(out, ("segment", index, segment))
            if ended:
                if writer is not None:
                    writer.close(summary=info.get("episode", {}))
                    writer = None
                if best is not None:
                    best.finish(info.get("episode", {}))
                episode += 1
                obs, _ = env.reset()
                history.reset(obs["pixels"])
                if best is not None:
                    best.start(episode)
                    best.add(*_video_frame(env, obs))
                if config.record_every and episode % config.record_every == 0:
                    writer = _episode_writer(run_dir, index, episode, env)
    except Stopped:
        pass  # told to stop while a reset waited for a fresh game: an ordinary stop, not a failure
    finally:
        if writer is not None:
            writer.close(summary={"ended": "actor stopped"})
        if best is not None:
            best.close()
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


def _video_frame(env, obs):
    """What the best game's film shows of this step, and when it was grabbed: the game's own picture, or, for a
    game with no screen behind it (the synthetic stand-in), the observation, with no time."""
    capture = getattr(env, "capture", None)
    if capture is None:
        return obs["pixels"], None
    return getattr(capture, "last_video", None), getattr(capture, "last_t", None)


def _sound_stream(env, index: int):
    """A function opening a fresh capture of this game's own sink for the best game's film, or None."""
    monitor = getattr(env, "sound_monitor", None)
    if monitor is None:
        return None
    from zombiesai.demos.audio import PulseMonitorStream

    return lambda: PulseMonitorStream(monitor, stream_name=f"best game {index}")


def _episode_writer(run_dir: Path, index: int, episode: int, env):
    from zombiesai.demos.clips import ClipWriter

    source = env.capture.describe() if getattr(env, "capture", None) is not None else {"kind": "synthetic"}
    return ClipWriter(run_dir / "episodes" / f"a{index}_ep{episode:05d}", source={**source, "actor": index},
                      label_source="play", config={"decision_hz": spec.DECISION_HZ})
