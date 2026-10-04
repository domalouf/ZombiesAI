"""A stand-in for the game when there is no game: first-person frames that answer to the agent's actions, on the
interfaces the real game has.

It is not a simulator of Nacht, and nothing learned on it is meant to transfer -- the real game's pixels are the
only ones a policy here trains on. It exists so that everything *around* the game runs end to end in CI and on a
PC with no game installed: the actors, the learner and the fleet (`SyntheticActorEnv`, `train_rl.py --env
synthetic`), and the recorder, the inverse dynamics model and behavioural cloning (`SyntheticSource`). For that it
has to get a few things exactly right, and nothing else:

* **Looking turns the picture by the lens.** The view is a window onto a 360-degree panorama at the centre
  resolution of an FOV_DEG pinhole lens on the frame's 128 pixels, so a turn of `yaw` degrees drags the image by
  `yaw * PX_PER_DEG` pixels -- the relation the recorder's yaw-versus-flow check (demos/inputs.py) measures on
  real footage and turns back into a field of view. Pitch moves it vertically at the same scale. The walls are
  smoothed noise, never periodic, so a shift is always measurable.
* **Input latency** is a whole number of decisions, fixed per world, and reported where the recorder looks for
  it (`SyntheticSource.describe()["latency_steps"]`).
* **Something to learn.** Zombies come in through four windows, grow as they close in, and die to shots that land
  where the crosshair is in the picture; the screen edges go red while the player is being hit, as WaW's do.
  `ScriptedPlayer` turns towards the nearest and fires, so a policy cloned from it, or an inverse dynamics model
  trained on it, has a real signal to find. With `audio_shape` set, each step also carries a stereo feature in
  which a zombie is loud as it nears and louder on its own side -- the cue the real game's sound gives.
* **The real environment's bookkeeping.** Rewards go through the same `RewardShaper` and the episode summary has
  the fields the learner, `episodes.jsonl` and the stream read (round reached, shots, hits, term shares).

Everything is deterministic in the seed, and a step with its scripted player costs well under a millisecond.
"""

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from zombiesai import spec
from zombiesai.demos.hud_crops import HUD_VIEW_SHAPE
from zombiesai.reward import REWARD_TERMS, RewardConfig, RewardShaper, StepSignals

FRAME_H, FRAME_W = spec.PIXELS_SHAPE[:2]
# WaW's cg_fov 65 is the horizontal FOV at 4:3; its Hor+ widescreen makes that about 80 degrees at 16:9.
FOV_DEG = 80.0
# What a turn of one degree moves the centre of an FOV_DEG pinhole view, which is what the flow check measures
# (demos/inputs.implied_fov_deg). The panorama is drawn at that scale everywhere, so the frame spans a little
# more than FOV_DEG of it -- a cylinder rather than a pinhole, which nothing here can tell apart.
PX_PER_DEG = (FRAME_W / 2) / math.tan(math.radians(FOV_DEG) / 2) * math.pi / 180.0
PANORAMA_W = round(360.0 * PX_PER_DEG)
PITCH_LIMIT_DEG = 20.0
_PITCH_PX = round(PITCH_LIMIT_DEG * PX_PER_DEG)
PANORAMA_H = FRAME_H + 2 * _PITCH_PX
WINDOW_BEARINGS_DEG = (0.0, 90.0, 180.0, 270.0)
ROOM_RADIUS_M = 6.0  # how far the player can back away from the middle of the room before a wall stops it
START_POINTS = 500
DT = 1.0 / spec.DECISION_HZ
# Nacht solo: 4, 9, 14, 19, then 24 a round forever (PLAN.md, "Ground-truth mechanics").
ZOMBIES_PER_ROUND = (4, 9, 14, 19, 24)
MAX_ALIVE = 24

_ZOMBIE_RGB = np.array((96, 70, 52), np.uint8)
_HEAD_RGB = np.array((150, 128, 104), np.uint8)
_COLS = np.arange(FRAME_W) - FRAME_W // 2
_YAW_BINS = np.asarray(spec.YAW_BINS_DEG)
_PITCH_BINS = np.asarray(spec.PITCH_BINS_DEG)


def _wrap180(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def _nearest(bins: np.ndarray, value: float) -> float:
    return float(bins[np.abs(bins - value).argmin()])


def _smooth_circular(x: np.ndarray, sigma: float) -> np.ndarray:
    k = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    padded = np.concatenate((x[-k:], x, x[:k]))
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def _panorama(rng: np.random.Generator) -> np.ndarray:
    """The room, unrolled: textured walls round the horizon, a dark ceiling and a floor, and four boarded windows."""
    h, w, cy = PANORAMA_H, PANORAMA_W, PANORAMA_H // 2
    fine, coarse = _smooth_circular(rng.normal(size=w), 2.0), _smooth_circular(rng.normal(size=w), 9.0)
    columns = 115 + 38 * fine / fine.std() + 24 * coarse / coarse.std()
    rows = _smooth_circular(rng.normal(size=h), 3.0)
    wall = columns[None, :] + 10 * (rows / rows.std())[:, None]
    image = wall[..., None] * (np.array((1.0, 0.9, 0.78)) * rng.uniform(0.9, 1.1, size=3))
    top, bottom = cy - 24, cy + 16
    image[:top] *= np.linspace(0.25, 0.5, top)[:, None, None]
    floor = 70 + 18 * _smooth_circular(rng.normal(size=w), 4.0)[None, :] + np.linspace(0, 30, h - bottom)[:, None]
    image[bottom:] = floor[..., None] * np.array((0.85, 0.8, 0.7))
    for bearing in WINDOW_BEARINGS_DEG:
        cols = (round(bearing * PX_PER_DEG) + np.arange(-7, 8)) % w
        image[cy - 14 : cy + 6, cols] = (38, 32, 28)
        image[cy - 12 : cy + 4 : 4, cols] = (120, 92, 60)  # the planks
    return np.clip(image, 0, 255).astype(np.uint8)


def _vignette() -> np.ndarray:
    y, x = np.mgrid[0:FRAME_H, 0:FRAME_W]
    r = np.hypot((x - FRAME_W / 2) / (FRAME_W / 2), (y - FRAME_H / 2) / (FRAME_H / 2))
    return np.clip((r - 0.55) / 0.5, 0.0, 1.0)[..., None]


_VIGNETTE = _vignette()


@dataclass(frozen=True)
class SyntheticConfig:
    latency_steps: int = 0  # decisions between an action and its effect on the picture
    max_steps: int = 18_000  # 20 minutes at 15 Hz; then the episode is truncated
    start_round: int = 1
    spawn_every_s: float = 2.0
    spawn_distance_m: float = 14.0
    zombie_speed_mps: float = 1.2  # round 1; +0.25 a round, so by round 4 backpedalling no longer outruns them
    zombie_health: float = 150.0  # round 1; +100 a round, as Nacht's are
    shot_damage: float = 100.0
    fire_every_steps: int = 3  # five shots a second
    mag_size: int = 8
    reserve: int = 96
    reload_steps: int = 22  # about 1.5 s
    move_speed_mps: float = 3.0  # forward; backwards and sideways at backpedal_factor of it, sprinting at 1.5x
    backpedal_factor: float = 0.6
    melee_every_steps: int = 8
    contact_m: float = 1.0
    attack_every_steps: int = 15  # a zombie in contact hits once a second
    attack_damage: float = 40.0  # three hits in a row are a death; WaW's health comes back, so does this
    regen_after_s: float = 3.0
    hit_tolerance_px: float = 1.0  # hip fire; ADS halves it
    audio_shape: tuple[int, int, int] | None = None  # (2, frames, mels): what a hearing policy reads
    reward: RewardConfig = field(default_factory=RewardConfig)


class SyntheticWorld:
    """One room, a horde, and a player who can only turn, step, shoot and reload."""

    def __init__(self, config: SyntheticConfig | None = None, seed: int = 0):
        self.config = config or SyntheticConfig()
        self.shaper = RewardShaper(self.config.reward)
        self.reset(seed)

    # ---------------------------------------------------------------------------------------------- state
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = int(seed)
        c = self.config
        self.rng = np.random.default_rng(self.seed)
        self._pano = _panorama(self.rng)
        self.yaw = float(self.rng.uniform(0.0, 360.0))
        self.pitch = 0.0
        self.pending: deque = deque([spec.NEUTRAL_ACTION] * c.latency_steps)
        self.pos = np.zeros(2)  # the player's place in the room; the zombies are kept relative to it
        self.bearing = np.zeros(0)
        self.distance = np.zeros(0)
        self.health_z = np.zeros(0)
        self.attack_in = np.zeros(0, dtype=np.int64)
        self.round = max(1, int(c.start_round))
        self.to_spawn = self._round_size()
        self.spawn_in = 0
        self.health = 100.0
        self.since_damage = 10_000
        self.points = START_POINTS
        self.mag, self.reserve = c.mag_size, c.reserve
        self.reload_left = 0
        self.cooldown = 0
        self.melee_in = 0
        self.fired = False
        self.steps = 0
        self.shots = self.hits = 0
        self.return_ = 0.0
        self.shaper.reset()
        return self.observe()

    def _round_size(self) -> int:
        return ZOMBIES_PER_ROUND[min(self.round, len(ZOMBIES_PER_ROUND)) - 1]

    @property
    def reloading(self) -> bool:
        return self.reload_left > 0

    def nearest(self) -> int | None:
        return int(self.distance.argmin()) if len(self.distance) else None

    def _half_size_px(self, i) -> tuple[np.ndarray, np.ndarray]:
        """A zombie's half width and half height on screen, in pixels: 1.8 m tall, 0.45 m wide."""
        half_h = np.degrees(np.arctan2(0.9, self.distance[i])) * PX_PER_DEG
        return 0.25 * half_h, half_h

    def half_width_deg(self, i: int) -> float:
        return float(self._half_size_px(i)[0] / PX_PER_DEG)

    def _box(self, i):
        """Where zombie(s) `i` are drawn: centre x, centre y, half width, half height, in frame pixels."""
        half_w, half_h = self._half_size_px(i)
        cx = FRAME_W / 2 + _wrap180(self.bearing[i] - self.yaw) * PX_PER_DEG
        cy = FRAME_H / 2 + self.pitch * PX_PER_DEG + 0.15 * half_h
        return cx, cy, half_w, half_h

    # ---------------------------------------------------------------------------------------------- step
    def step(self, action):
        """Apply `action` (after the world's latency), advance a decision, and return (obs, reward, terminated,
        truncated, info) -- info carries `bad` (always False here) and, at the end, the episode `summary`."""
        c = self.config
        self.pending.append(spec.action_tuple(action))
        a = self.pending.popleft()
        self.steps += 1
        delta = 0.0
        damaged = died = False
        rounds_completed = 0

        self.yaw = (self.yaw + _YAW_BINS[a[spec.YAW]]) % 360.0
        self.pitch = float(np.clip(self.pitch + _PITCH_BINS[a[spec.PITCH]], -PITCH_LIMIT_DEG, PITCH_LIMIT_DEG))
        self._move(spec.FORWARD_VALUES[a[spec.FORWARD]], spec.STRAFE_VALUES[a[spec.STRAFE]],
                   sprint=bool(a[spec.SPRINT]))

        button = spec.BUTTONS[a[spec.BUTTON]]
        if self.reload_left:
            self.reload_left -= 1
            if not self.reload_left:
                take = min(c.mag_size - self.mag, self.reserve)
                self.mag, self.reserve = self.mag + take, self.reserve - take
        elif button == "reload" and self.mag < c.mag_size and self.reserve > 0:
            self.reload_left = c.reload_steps
        self.cooldown = max(0, self.cooldown - 1)
        self.fired = False
        if a[spec.FIRE] and not self.reload_left and self.cooldown == 0 and self.mag > 0:
            self.mag -= 1
            self.shots += 1
            self.cooldown = c.fire_every_steps
            self.fired = True
            delta += self._shoot(ads=bool(a[spec.ADS]))
        self.melee_in = max(0, self.melee_in - 1)
        if button == "melee" and len(self.distance) and self.melee_in == 0:
            self.melee_in = c.melee_every_steps
            i = self.nearest()
            if self.distance[i] < 1.6 and abs(float(_wrap180(self.bearing[i] - self.yaw))) < 30.0:
                delta += self._damage(i, 150.0, kill_points=120)

        if len(self.distance):
            speed = c.zombie_speed_mps + 0.25 * (self.round - 1)
            self.distance = np.maximum(self.distance - speed * DT, c.contact_m)
            touching = self.distance <= c.contact_m
            self.attack_in = np.where(touching, self.attack_in - 1, c.attack_every_steps)
            hits = int((self.attack_in <= 0).sum())
            if hits:
                self.attack_in[self.attack_in <= 0] = c.attack_every_steps
                self.health -= c.attack_damage * hits
                self.since_damage = 0
                damaged = True
                died = self.health <= 0
        self.since_damage += 1
        if self.since_damage * DT >= c.regen_after_s:
            self.health = 100.0

        self.spawn_in -= 1
        if self.to_spawn and self.spawn_in <= 0 and len(self.distance) < MAX_ALIVE:
            self._spawn()
        if not self.to_spawn and not len(self.distance):
            rounds_completed = 1
            self.round += 1
            self.to_spawn = self._round_size()
            self.spawn_in = round(3.0 / DT)  # the break between rounds
            self.reserve = c.reserve  # a full reserve each round: this world is about aiming, not the economy

        self.points += int(delta)
        result = self.shaper(StepSignals(delta_points=delta, rounds_completed=rounds_completed,
                                         damage_event=damaged, death=died))
        self.return_ += result.reward
        terminated = died
        truncated = not terminated and self.steps >= c.max_steps
        info = {"bad": False, "points": self.points, "round": self.round, "terms": result.terms}
        if terminated or truncated:
            info["episode"] = self.summary("death" if terminated else "time_limit")
        return self.observe(), float(result.reward), terminated, truncated, info

    def _move(self, forward: int, strafe: int, sprint: bool) -> None:
        if not (forward or strafe):
            return
        c = self.config
        factor = (1.5 if sprint else 1.0) if forward > 0 else c.backpedal_factor
        speed = c.move_speed_mps * factor * DT
        heading = math.radians(self.yaw)
        step = speed * np.array((forward * math.sin(heading) + strafe * math.cos(heading),
                                 forward * math.cos(heading) - strafe * math.sin(heading)))
        target = self.pos + step
        if np.hypot(*target) > ROOM_RADIUS_M:  # the wall: stop at it
            target *= ROOM_RADIUS_M / np.hypot(*target)
        (dx, dy), self.pos = target - self.pos, target
        if not len(self.distance):
            return
        bearing = np.radians(self.bearing)
        x, y = self.distance * np.sin(bearing) - dx, self.distance * np.cos(bearing) - dy
        self.distance = np.maximum(np.hypot(x, y), c.contact_m)
        self.bearing = np.degrees(np.arctan2(x, y)) % 360.0

    def _shoot(self, ads: bool) -> float:
        """A shot lands on the nearest zombie drawn under the crosshair, if any. Returns the points it paid."""
        if not len(self.distance):
            return 0.0
        cx, cy, half_w, half_h = self._box(np.arange(len(self.distance)))
        tolerance = self.config.hit_tolerance_px * (0.5 if ads else 1.0)
        on = (np.abs(cx - FRAME_W / 2) <= half_w + tolerance) & (np.abs(cy - FRAME_H / 2) <= half_h)
        if not on.any():
            return 0.0
        i = int(np.flatnonzero(on)[self.distance[on].argmin()])
        self.hits += 1
        return self._damage(i, self.config.shot_damage, kill_points=60)

    def _damage(self, i: int, amount: float, kill_points: int) -> float:
        self.health_z[i] -= amount
        if self.health_z[i] > 0:
            return 10.0
        keep = np.arange(len(self.distance)) != i
        self.bearing, self.distance = self.bearing[keep], self.distance[keep]
        self.health_z, self.attack_in = self.health_z[keep], self.attack_in[keep]
        return float(kill_points)

    def _spawn(self) -> None:
        c = self.config
        window = WINDOW_BEARINGS_DEG[int(self.rng.integers(len(WINDOW_BEARINGS_DEG)))]
        self.bearing = np.append(self.bearing, (window + self.rng.uniform(-4.0, 4.0)) % 360.0)
        self.distance = np.append(self.distance, c.spawn_distance_m + self.rng.uniform(-1.5, 1.5))
        self.health_z = np.append(self.health_z, c.zombie_health + 100.0 * (self.round - 1))
        self.attack_in = np.append(self.attack_in, c.attack_every_steps)
        self.to_spawn -= 1
        self.spawn_in = round(c.spawn_every_s / DT)

    # ---------------------------------------------------------------------------------------------- views
    def render(self) -> np.ndarray:
        """The 128x72 RGB frame the player sees now."""
        cols = (round(self.yaw * PX_PER_DEG) + _COLS) % PANORAMA_W
        top = _PITCH_PX - round(self.pitch * PX_PER_DEG)
        frame = self._pano[top : top + FRAME_H][:, cols]
        if len(self.distance):
            cx, cy, half_w, half_h = self._box(np.arange(len(self.distance)))
            for i in np.argsort(-self.distance):  # far ones first, so the near ones stand in front
                if abs(cx[i] - FRAME_W / 2) > FRAME_W / 2 + half_w[i] + 1:
                    continue
                x0, x1 = (int(np.clip(round(v), 0, FRAME_W)) for v in (cx[i] - half_w[i], cx[i] + half_w[i] + 1))
                y0, y1 = (int(np.clip(round(v), 0, FRAME_H)) for v in (cy[i] - half_h[i], cy[i] + half_h[i] + 1))
                frame[y0:y1, x0:x1] = _ZOMBIE_RGB
                head = int(np.clip(round(cy[i] - half_h[i] + 0.35 * half_h[i]), 0, FRAME_H))
                frame[y0:head, x0:x1] = _HEAD_RGB
        if self.since_damage * DT < 1.0:
            alpha = _VIGNETTE * (0.7 * (1.0 - self.since_damage * DT))
            frame = (frame * (1.0 - alpha) + np.array((170.0, 10.0, 10.0)) * alpha).astype(np.uint8)
        return frame

    def hear(self) -> np.ndarray:
        """A stereo feature of `config.audio_shape`: quiet, plus each zombie in its own side's low bands, louder as
        it nears, and a broadband burst on a shot. Shaped like demos/hearing.py's features, meant like none of them."""
        channels, frames, mels = self.config.audio_shape
        out = np.full((channels, frames, mels), -2.5, np.float32)
        low = slice(2, max(3, mels // 4))
        for bearing, distance in zip(self.bearing, self.distance):
            level = max(0.0, 1.0 - distance / 15.0)
            pan = math.sin(math.radians(float(_wrap180(bearing - self.yaw))))
            for ch, gain in enumerate(((1.0 - pan) / 2.0, (1.0 + pan) / 2.0)[:channels]):
                out[ch, :, low] += 3.0 * level * gain
        if self.fired:
            out[:, -3:, :] += 2.0
        return out

    def hud_view(self) -> np.ndarray:
        """The HUD corner a policy with `use_hud_view` gets (demos/hud_crops.py), drawn the way the real one
        shows a pistol's magazine: a row of one-pixel marks ending at column 35, loaded ones bright, spent ones
        dim, spent from the left. Nothing else of the real corner is drawn."""
        view = np.full(HUD_VIEW_SHAPE, 30, np.uint8)
        size = self.config.mag_size
        for k in range(size):
            view[56:58, 35 - k] = 75 if k < self.mag else 40  # the real view's contrasts: ~+40 loaded, ~+10 spent
        return view

    def observe(self) -> dict:
        obs = {"pixels": self.render(), "hud_view": self.hud_view()}
        if self.config.audio_shape is not None:
            obs["audio"], obs["audio_mask"] = self.hear(), 1.0
        return obs

    def summary(self, reason: str | None = None) -> dict:
        stats = self.shaper.stats
        return {
            "reason": reason,
            "return": self.return_,
            "length": self.steps,
            "seconds": self.steps * DT,
            "round_reached": self.round,
            "points_gained": stats.points_gained,
            "shots": self.shots,
            "hits": self.hits,
            "bad_steps": 0,
            "repair_share": stats.repair_share(),
            "max_term_share": stats.max_term_share(),
            "reward_term_sums": dict(zip(REWARD_TERMS, stats.term_sums.tolist())),
        }


class ScriptedPlayer:
    """Turn towards the nearest zombie, fire once it is under the crosshair, back off when it is close, reload when
    dry, and look round the windows between waves -- with a little wandering, so a recording of it uses every
    head (the inverse dynamics model and BC need more than one class to tell apart). Reads the world's state, as
    a scripted player may; deterministic in its seed."""

    def __init__(self, seed: int = 0, wander: float = 0.1):
        self.seed, self.wander = seed, wander
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed + 7919)
        self._scan = 1.0

    def act(self, world: SyntheticWorld) -> np.ndarray:
        in_flight_yaw = sum(_YAW_BINS[a[spec.YAW]] for a in world.pending)
        in_flight_pitch = sum(_PITCH_BINS[a[spec.PITCH]] for a in world.pending)
        kw: dict = {"pitch": _nearest(_PITCH_BINS, -(world.pitch + in_flight_pitch))}
        target = world.nearest()
        if target is None:
            if self.rng.random() < 0.04:
                self._scan = -self._scan
            kw["yaw"] = 6.0 * self._scan
            if world.mag < world.config.mag_size and world.reserve and not world.reloading:
                kw["button"] = "reload"
        else:
            rel = float(_wrap180(world.bearing[target] - world.yaw - in_flight_yaw))
            limit = 30.0 if abs(rel) > 60.0 else 14.0
            kw["yaw"] = _nearest(_YAW_BINS, float(np.clip(rel, -limit, limit)))
            if abs(rel - kw["yaw"]) <= world.half_width_deg(target) and world.mag and not world.reloading:
                kw["fire"] = 1
                kw["ads"] = int(world.distance[target] > 6.0)
            armed = world.mag or world.reserve or world.reloading
            if world.distance[target] < 2.5 and armed:
                kw["forward"] = -1
            if not world.mag and world.reserve and not world.reloading:
                kw["button"] = "reload"
            elif world.distance[target] < 1.4:
                kw["button"] = "melee"
        if self.rng.random() < self.wander:
            kw["yaw"] = float(self.rng.choice((-14.0, -6.0, -2.0, 2.0, 6.0, 14.0)))
            kw["strafe"] = int(self.rng.choice((-1, 0, 1)))
            kw["sprint"] = int(self.rng.random() < 0.3)
            if kw.get("button", "none") == "none" and self.rng.random() < 0.2:
                kw["button"] = str(self.rng.choice(("use", "grenade", "swap")))
        return spec.make_action(**kw)


class SyntheticActorEnv:
    """The real environment's interface for an RL actor (rl/actors.py), over a SyntheticWorld: `reset()` and
    `step(action)` with `info["bad"]` and, at an episode's end, `info["episode"]`. Each episode is a new seed."""

    capture = None  # no screen behind it: the actor records clips without a capture description

    def __init__(self, seed: int, config: SyntheticConfig | dict | None = None,
                 audio_shape: tuple[int, int, int] | None = None):
        if isinstance(config, dict):
            config = SyntheticConfig(**config)
        config = config or SyntheticConfig()
        if audio_shape is not None:
            config = replace(config, audio_shape=tuple(audio_shape))
        self.seed = int(seed)
        self.episodes = 0
        self.world = SyntheticWorld(config, seed=self.seed * 100_003)

    def reset(self):
        obs = self.world.reset(self.seed * 100_003 + self.episodes)
        self.episodes += 1
        return obs, {}

    def step(self, action):
        return self.world.step(action)

    def close(self) -> None:
        pass


class SyntheticSource:
    """A recorder source (demos/recorder.py) with a player at the controls: `read()` is the frame on screen and
    `drain(start, end)` plays one decision and returns the raw input events that player's hands would have made
    for it (demos/inputs.synthesize) -- so a recording of it is labelled exactly as a human demo is."""

    def __init__(self, player: ScriptedPlayer | None = None, *, seed: int = 0, max_steps: int = 18_000,
                 latency_steps: int = 0, input_config=None, config: SyntheticConfig | None = None):
        from zombiesai.demos.inputs import InputConfig

        base = config or SyntheticConfig()
        self.config = replace(base, max_steps=max_steps, latency_steps=latency_steps)
        self.input_config = input_config or InputConfig(counts_per_degree=10.0)
        self.player = player or ScriptedPlayer(seed)
        self.seed = seed
        self.world = SyntheticWorld(self.config, seed=seed)
        self.player.reset()
        self.done = False
        self.info: dict = {}
        self.steps = 0

    def read(self) -> np.ndarray:
        return self.world.render()

    def drain(self, start: float, end: float) -> list[dict]:
        from zombiesai.demos.inputs import synthesize

        if self.done:
            return []
        action = np.asarray(self.player.act(self.world))
        events = synthesize(action, start, end - start, self.input_config)
        _, _, terminated, truncated, self.info = self.world.step(action)
        self.done = terminated or truncated
        self.steps += 1
        return events

    def describe(self) -> dict:
        return {"kind": "synthetic", "seed": self.seed, "player": type(self.player).__name__,
                "latency_steps": int(self.config.latency_steps)}

    def close(self) -> None:
        pass


def scripted_actions(steps: int, *, seed: int = 0, latency_steps: int = 0) -> np.ndarray:
    """The actions `SyntheticSource(seed=seed, latency_steps=...)` plays, without recording them: what a correct
    recording of it must be labelled with."""
    source = SyntheticSource(seed=seed, max_steps=steps, latency_steps=latency_steps)
    out = []
    while not source.done and len(out) < steps:
        out.append(np.asarray(source.player.act(source.world)))
        _, _, terminated, truncated, _ = source.world.step(out[-1])
        source.done = terminated or truncated
    return np.array(out, dtype=np.uint8)
