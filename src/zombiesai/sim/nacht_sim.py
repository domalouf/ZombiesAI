"""NachtSim: a fast 2D Nacht der Untoten simulator emitting the spec's state-profile observations."""

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import gymnasium as gym
import numpy as np

from zombiesai import spec
from zombiesai.reward import REWARD_TERMS, RewardConfig, RewardShaper, StepSignals
from zombiesai.sim import mechanics
from zombiesai.sim.geometry import load_geometry
from zombiesai.sim.params import (
    BOX_WEIGHTS,
    STARTING_RESERVE,
    STARTING_WEAPON,
    WEAPONS,
    SimParams,
    Weapon,
    sample_params,
    sample_timing,
)

MAX_ZOMBIES = 32
APPROACH, TEARING, CLIMBING, INSIDE = range(4)
PROMPT_NONE, PROMPT_DOOR, PROMPT_WEAPON, PROMPT_BOX, PROMPT_REPAIR = range(5)
NO_PROMPT = (PROMPT_NONE, 0, -1)
PITCH_LIMIT_DEG = 70.0
DIRECT_CHASE_M = 2.0
BLOCK_CHECK_M = 3.0
GRENADE_LOCKOUT_S = 0.6
TWO_PI = 2.0 * math.pi
INF = math.inf

_YAW_RAD = tuple(math.radians(d) for d in spec.YAW_BINS_DEG)
_FWD = spec.FORWARD_VALUES
_STRAFE = spec.STRAFE_VALUES
_USE, _RELOAD, _MELEE, _GRENADE, _SWAP = (spec.BUTTONS.index(b) for b in ("use", "reload", "melee", "grenade", "swap"))
_H = spec.HUD_INDEX
_S = spec.STATE_INDEX
_Z0 = spec.STATE_ZOMBIE_START
_K = spec.STATE_K_ZOMBIES
_EMPTY_ZOMBIES = [0.0] * (4 * _K)
_STATE_ORDER = (
    ("player_x", "player_y", "yaw_sin", "yaw_cos", "pitch")
    + tuple(f"zone_{z}" for z in spec.ZONES)
    + tuple(f"open_{d}" for d in spec.DOORS)
    + ("weapon_damage", "weapon_rate", "weapon_auto")
    + tuple(f"zombie{i}_{k}" for i in range(_K) for k in ("present", "sin", "cos", "dist"))
    + ("barrier_sin", "barrier_cos", "barrier_dist", "barrier_planks")
)
# _state_vector builds the vector positionally, so it must agree with the spec's field order.
assert _STATE_ORDER == spec.STATE_FIELDS
_PROMPT_HUD = {
    PROMPT_DOOR: _H["prompt_door"],
    PROMPT_WEAPON: _H["prompt_weapon"],
    PROMPT_BOX: _H["prompt_box"],
    PROMPT_REPAIR: _H["prompt_repair"],
}
_MISREAD_FIELDS = (_H["round"], _H["points"], _H["mag_ammo"], _H["reserve_ammo"])
_KILL_BONUS = {
    "head": mechanics.SCORE_HEAD_BONUS,
    "upper": mechanics.SCORE_TORSO_BONUS,
    "melee": mechanics.SCORE_MELEE_BONUS,
}


def weapon_features(w: Weapon) -> tuple[float, float, float]:
    return (
        min(1.0, math.log1p(w.damage) / math.log1p(spec.STATE_DAMAGE_LOG_SCALE)),
        min(1.0, w.rpm / 60.0 / spec.STATE_RATE_SCALE),
        float(w.automatic),
    )


_WEAPON_FEATURES = {name: weapon_features(w) for name, w in WEAPONS.items()}


@dataclass(frozen=True)
class SimConfig:
    hardness: float = 0.5
    max_steps: int = 18_000
    # "state" is the fast path the RL algorithms train on; "render" feeds the raycast view to the policy
    # instead, which is what a pixel network (behavioural cloning, M2's CNN check) has to be evaluated on.
    obs_profile: str = "state"
    latency_steps: int | None = None
    frames_per_step: int | None = None
    action_dropout: float | None = None
    params: SimParams = field(default_factory=SimParams)
    reward: RewardConfig = field(default_factory=RewardConfig)
    geometry_path: str | None = None


@dataclass(slots=True)
class HeldWeapon:
    w: Weapon
    mag: int
    reserve: int


@dataclass(slots=True)
class RoundStats:
    spawned: int = 0
    kills: int = 0
    headshot_kills: int = 0
    melee_kills: int = 0
    hits: int = 0
    shots: int = 0
    shot_hits: int = 0  # shots that damaged at least one zombie (knife and grenade hits don't count)
    reloads: int = 0
    repairs: int = 0
    purchases: int = 0
    damage_taken: int = 0
    points_gained: int = 0


@dataclass(slots=True)
class _StepAccum:
    repair_points: float = 0.0
    rounds_completed: int = 0
    doors_opened: list = field(default_factory=list)
    wall_weapons_bought: list = field(default_factory=list)
    ammo_fractions: list = field(default_factory=list)


def _wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def _corrupt_digits(value: float, rng: np.random.Generator) -> int:
    """One plausible template-matching failure: a dropped, doubled, or substituted digit."""
    digits = list(str(max(0, int(value))))
    pos = int(rng.integers(len(digits)))
    op = int(rng.integers(3))
    if op == 0 and len(digits) > 1:
        del digits[pos]
    elif op == 1:
        digits.insert(pos, digits[pos])
    else:
        digits[pos] = str(int(rng.integers(10)))
    return int("".join(digits))


class NachtSim(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": spec.DECISION_HZ}

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        self.render_mode = render_mode
        self._renderer = None
        self.shots_this_step = 0  # lets the renderer draw muzzle flash
        self.config = config or SimConfig()
        if not 0.0 <= self.config.hardness <= 1.0:
            raise ValueError(f"hardness must be in [0, 1], got {self.config.hardness}")
        if self.config.obs_profile not in spec.OBS_PROFILES:
            raise ValueError(f"obs_profile must be one of {tuple(spec.OBS_PROFILES)}, got {self.config.obs_profile!r}")
        self.pixel_obs = self.config.obs_profile == "render"
        self.geo = geo = load_geometry(self.config.geometry_path)
        self.observation_space = spec.observation_space(self.config.obs_profile)
        self.action_space = spec.factored_action_space()
        self.reward_shaper = RewardShaper(self.config.reward)

        n = MAX_ZOMBIES
        self.z_alive = np.zeros(n, dtype=bool)
        self.z_phase = np.zeros(n, dtype=np.int8)
        # Complex positions halve the numpy calls in the hot 2D vector math; x/y are live views onto them.
        self.z_pos = np.zeros(n, dtype=np.complex128)
        self.z_x = self.z_pos.real
        self.z_y = self.z_pos.imag
        self.z_hp = np.zeros(n)
        self.z_speed = np.zeros(n)
        self.z_window = np.zeros(n, dtype=np.int64)
        # Absolute sim times, so nothing needs a per-step countdown.
        self.z_event_t = np.zeros(n)  # next plank torn (TEARING) or climb finished (CLIMBING)
        self.z_hit_t = np.full(n, INF)  # current swing lands (inf = not swinging)
        self.z_ready_t = np.zeros(n)  # next swing may start

        x0, y0, x1, y1 = geo.bounds
        self._grid = (x0, y0, 1.0 / geo.cell_size, geo.nx)
        self._norm = (x0, y0, 1.0 / (x1 - x0), 1.0 / (y1 - y0))
        self._origin = complex(x0, y0)
        self._zone_list = geo.zone_of_cell.tolist()
        self._node_list = geo.node_of_cell.tolist()
        self._node_c = geo.node_xy[:, 0] + 1j * geo.node_xy[:, 1]
        self._win_out = geo.window_outside[:, 0] + 1j * geo.window_outside[:, 1]
        self._win_in = geo.window_inside[:, 0] + 1j * geo.window_inside[:, 1]
        self._window_xy = tuple((float(x), float(y)) for x, y in geo.window_pos)
        self._door_rects = tuple(tuple(float(v) for v in r) for r in geo.door_rects)
        self._weapon_xy = tuple((float(x), float(y)) for x, y in geo.wall_weapon_pos)
        self._box_xy = (float(geo.box_pos[0]), float(geo.box_pos[1]))
        self._box_names = tuple(BOX_WEIGHTS)
        weights = np.array([BOX_WEIGHTS[k] for k in self._box_names], dtype=np.float64)
        self._box_p = weights / weights.sum()

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self._rng = self.np_random
        cfg = self.config
        p = self.p = sample_params(rng, cfg.hardness, cfg.params)
        self.timing = sample_timing(rng, cfg.hardness, cfg.latency_steps, cfg.frames_per_step, cfg.action_dropout)
        self.dt = self.timing.frames_per_step / 60.0
        self.reward_shaper.reset()

        self.open_mask = 0
        for door_id in options.get("open_doors", ()):
            self.open_mask |= 1 << spec.DOORS.index(door_id)
        self.planks = np.full(len(self.geo.windows), spec.MAX_PLANKS, dtype=np.int64)
        self._doors_changed()

        spawn = self.geo.player_spawn
        self.px = float(spawn[0] + rng.uniform(-1.0, 1.0))
        self.py = float(spawn[1] + rng.uniform(-1.0, 1.0))
        self.yaw = float(rng.uniform(-math.pi, math.pi))
        self.pitch = 0.0
        self.zone = self._zone_list[self.geo.cell(self.px, self.py)]
        self.hp = p.player_max_hp
        self.time_since_hit = INF
        self.stamina = 1.0
        self.sprinting = False
        self.ads = False

        loadout = tuple(options.get("loadout", (STARTING_WEAPON,)))
        if not 1 <= len(loadout) <= 2:
            raise ValueError(f"loadout must hold 1 or 2 weapons, got {loadout}")
        self.weapons = [
            HeldWeapon(
                WEAPONS[n], WEAPONS[n].mag_size, STARTING_RESERVE if n == STARTING_WEAPON else WEAPONS[n].max_reserve
            )
            for n in loadout
        ]
        self.slot = 0
        self.reload_t = self.swap_t = self.fire_cd = self.melee_cd = self.lock_t = 0.0
        self.grenades = p.grenades_start
        self.live_grenades: list[list[float]] = []
        self.points = int(options.get("start_points", mechanics.STARTING_POINTS))
        self.repair_t = 0.0

        self.z_alive[:] = False
        self.z_hit_t[:] = INF
        self._nearest_inside = INF
        self.round = int(options.get("start_round", 1))
        self._start_round()
        self.intermission_t = 0.0
        self.t = 0.0
        self.steps = 0

        self.flash = 0.0
        self.flash_fp = 0.0
        self.flash_obs = 0.0
        self.time_since_damage = INF
        self._damage_detected = False

        neutral = spec.NEUTRAL_ACTION
        self._pending: deque[tuple[int, ...]] = deque()
        self._applied = neutral
        self._prev_applied = neutral
        self._history = [neutral] * spec.PREV_ACTION_HISTORY
        self._ubuf: list[float] = []
        self._nbuf: list[float] = []
        self._ui = self._ni = 0
        self._events: list[dict] = []
        self.shots_this_step = 0
        self._refresh_distance_field()
        return self._observation(), {"round": self.round, "timing": asdict(self.timing)}

    def step(self, action):
        chosen = spec.action_tuple(action)
        a = self._apply_timing(chosen)
        dt = self.dt
        self.t += dt
        self._events = []
        self.shots_this_step = 0
        acc = _StepAccum()
        points_before = self.points

        self._look(a)
        self._timers(dt)
        self._hold_states(a, dt)
        self._buttons(a, dt, acc)
        self._fire(a)
        self._move(a, dt)
        self._refresh_distance_field()
        self._update_zombies(dt)
        self._update_grenades(dt)
        self._update_health(dt)
        died = self.hp <= 0.0
        if not died:
            self._update_round(dt, acc)
        self.steps += 1
        self.rs.points_gained += max(0, self.points - points_before)
        self._history = [chosen] + self._history[:-1]

        result = self.reward_shaper(
            StepSignals(
                delta_points=float(self.points - points_before),
                repair_points=acc.repair_points,
                rounds_completed=acc.rounds_completed,
                doors_opened=tuple(acc.doors_opened),
                wall_weapons_bought=tuple(acc.wall_weapons_bought),
                ammo_rebuy_reserve_fractions=tuple(acc.ammo_fractions),
                damage_event=self._damage_detected,
                death=died,
            )
        )
        terminated = died
        truncated = not terminated and self.steps >= self.config.max_steps
        if died:
            self._events.append({"type": "game_over", "round": self.round, "t": self.t})
        info = {
            "round": self.round,
            "points": self.points,
            "reward_terms": result.terms,
            "gain_clipped": result.gain_clipped,
            "bad_step": False,
            "dt": dt,
            "events": self._events,
        }
        if terminated or truncated:
            info.update(self._episode_summary(terminated))
        return self._observation(downed=died), result.reward, terminated, truncated, info

    def render(self):
        """The agent's-eye view as a PIXELS_SHAPE uint8 frame; None unless built with render_mode="rgb_array"."""
        return self.frame() if self.render_mode == "rgb_array" else None

    def state(self) -> np.ndarray:
        """The state-profile vector, whichever profile this sim was built with -- the render profile hides it
        from the policy, but a scripted agent or a diagnostic still has every right to ask."""
        return self._state_vector()

    def frame(self) -> np.ndarray:
        """The same view, drawn whether or not anyone asked to watch: the render profile's observation."""
        if self._renderer is None:
            from zombiesai.sim.render import Renderer

            self._renderer = Renderer(self.geo)
        return self._renderer.render(self)

    # ------------------------------------------------------------ batched per-step randomness
    def _uniform(self) -> float:
        if self._ui == len(self._ubuf):
            self._ubuf = self._rng.random(4096).tolist()
            self._ui = 0
        self._ui += 1
        return self._ubuf[self._ui - 1]

    def _normal(self) -> float:
        if self._ni == len(self._nbuf):
            self._nbuf = self._rng.standard_normal(4096).tolist()
            self._ni = 0
        self._ni += 1
        return self._nbuf[self._ni - 1]

    # ------------------------------------------------------------ timing & input
    def _apply_timing(self, chosen: tuple[int, ...]) -> tuple[int, ...]:
        self._pending.append(chosen)
        a = self._pending.popleft() if len(self._pending) > self.timing.latency_steps else self._applied
        if self.timing.action_dropout and self._uniform() < self.timing.action_dropout:
            a = self._applied
        self._prev_applied = self._applied
        self._applied = a
        return a

    def _look(self, a) -> None:
        self.yaw = _wrap(self.yaw - _YAW_RAD[a[spec.YAW]])
        pitch = self.pitch + spec.PITCH_BINS_DEG[a[spec.PITCH]]
        self.pitch = min(max(pitch, -PITCH_LIMIT_DEG), PITCH_LIMIT_DEG)

    def _timers(self, dt: float) -> None:
        if self.reload_t > 0.0:
            self.reload_t -= dt
            if self.reload_t <= 0.0:
                self.reload_t = 0.0
                hw = self.weapons[self.slot]
                take = min(hw.w.mag_size - hw.mag, hw.reserve)
                hw.mag += take
                hw.reserve -= take
        self.swap_t = max(0.0, self.swap_t - dt)
        self.melee_cd = max(0.0, self.melee_cd - dt)
        self.lock_t = max(0.0, self.lock_t - dt)
        self.fire_cd -= dt

    def _hold_states(self, a, dt: float) -> None:
        p = self.p
        self.sprinting = bool(
            a[spec.SPRINT] and _FWD[a[spec.FORWARD]] == 1 and not a[spec.FIRE] and self.stamina > 0.0
        )
        if self.sprinting:
            self.stamina = max(0.0, self.stamina - dt / p.sprint_duration_s)
        else:
            self.stamina = min(1.0, self.stamina + dt / p.sprint_recovery_s)
        self.ads = bool(a[spec.ADS]) and not self.sprinting

    def _buttons(self, a, dt: float, acc: _StepAccum) -> None:
        button = a[spec.BUTTON]
        kind, price, target = self.prompt
        if button == _USE and kind == PROMPT_REPAIR:
            self._repair(target, dt, acc)
        else:
            self.repair_t = 0.0
        if button == 0 or button == self._prev_applied[spec.BUTTON]:
            return
        if button == _USE:
            self._use(kind, price, target, acc)
        elif button == _RELOAD:
            self._start_reload()
        elif button == _MELEE:
            self._melee()
        elif button == _GRENADE:
            self._throw_grenade()
        elif button == _SWAP:
            self._swap()

    # ------------------------------------------------------------ interactions
    def _prompt(self) -> tuple[int, int, int]:
        """(kind, price, target index) of the nearest interactable in range, as WaW's hint string shows it."""
        p, px, py = self.p, self.px, self.py
        best, best_d = NO_PROMPT, INF
        rr, ir = p.repair_radius_m, p.interact_radius_m
        for i, (wx, wy) in enumerate(self._window_xy):
            d = math.hypot(wx - px, wy - py)
            if d <= rr and d < best_d and self.planks[i] < spec.MAX_PLANKS:
                best, best_d = (PROMPT_REPAIR, 0, i), d
        for i, (x0, y0, x1, y1) in enumerate(self._door_rects):
            if self.open_mask >> i & 1:
                continue
            d = math.hypot(max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1))
            if d <= ir and d < best_d:
                best, best_d = (PROMPT_DOOR, self.geo.doors[i].price, i), d
        for i, (wx, wy) in enumerate(self._weapon_xy):
            d = math.hypot(wx - px, wy - py)
            if d <= ir and d < best_d:
                ww = self.geo.wall_weapons[i]
                price = round(ww.price * p.ammo_cost_fraction) if self._owned(ww.weapon) else ww.price
                best, best_d = (PROMPT_WEAPON, price, i), d
        d = math.hypot(self._box_xy[0] - px, self._box_xy[1] - py)
        if d <= ir and d < best_d:
            best = (PROMPT_BOX, mechanics.BOX_PRICE, 0)
        return best

    def _owned(self, name: str) -> HeldWeapon | None:
        for hw in self.weapons:
            if hw.w.name == name:
                return hw
        return None

    def _use(self, kind: int, price: int, target: int, acc: _StepAccum) -> None:
        if kind in (PROMPT_NONE, PROMPT_REPAIR) or self.points < price:
            return
        if kind == PROMPT_DOOR:
            self.open_mask |= 1 << target
            self._doors_changed()
            acc.doors_opened.append(self.geo.doors[target].id)
            item = self.geo.doors[target].id
        elif kind == PROMPT_WEAPON:
            name = self.geo.wall_weapons[target].weapon
            hw = self._owned(name)
            if hw is not None:
                if hw.reserve >= hw.w.max_reserve and hw.mag >= hw.w.mag_size:
                    return
                acc.ammo_fractions.append(hw.reserve / hw.w.max_reserve)
                hw.mag, hw.reserve = hw.w.mag_size, hw.w.max_reserve
                item = f"{name}_ammo"
            else:
                self._give_weapon(name)
                acc.wall_weapons_bought.append(name)
                item = name
        else:
            owned = {hw.w.name for hw in self.weapons}
            probs = np.array([0.0 if n in owned else w for n, w in zip(self._box_names, self._box_p)])
            name = self._box_names[int(self._rng.choice(len(probs), p=probs / probs.sum()))]
            self._give_weapon(name)
            item = f"box:{name}"
        self.points -= price
        self.rs.purchases += 1
        self._events.append({"type": "purchase", "item": item, "price": price, "t": self.t})

    def _give_weapon(self, name: str) -> None:
        w = WEAPONS[name]
        hw = HeldWeapon(w, w.mag_size, w.max_reserve)
        if len(self.weapons) < 2:
            self.weapons.append(hw)
            self.slot = len(self.weapons) - 1
        else:
            self.weapons[self.slot] = hw
        self.reload_t = 0.0
        self.swap_t = self.p.weapon_swap_s

    def _repair(self, window: int, dt: float, acc: _StepAccum) -> None:
        self.repair_t += dt
        interval = self.p.repair_interval_s
        while self.repair_t >= interval and self.planks[window] < spec.MAX_PLANKS:
            self.planks[window] += 1
            self.points += mechanics.SCORE_BARRIER_PLANK
            acc.repair_points += mechanics.SCORE_BARRIER_PLANK
            self.rs.repairs += 1
            self.repair_t -= interval
        if self.planks[window] >= spec.MAX_PLANKS:
            self.repair_t = 0.0

    def _start_reload(self) -> None:
        hw = self.weapons[self.slot]
        if self.reload_t > 0.0 or self.swap_t > 0.0 or hw.mag >= hw.w.mag_size or hw.reserve <= 0:
            return
        self.reload_t = hw.w.reload_s
        self.rs.reloads += 1

    def _swap(self) -> None:
        if len(self.weapons) < 2 or self.swap_t > 0.0:
            return
        self.slot = 1 - self.slot
        self.swap_t = self.p.weapon_swap_s
        self.reload_t = 0.0

    def _doors_changed(self) -> None:
        active = self.geo.active_zones(self.open_mask)
        self._active_windows = np.flatnonzero(active[self.geo.window_zone])
        self._active_window_list = self._active_windows.tolist()
        self._walkable = self.geo.walkable(self.open_mask)
        self._walk_list = self._walkable.tolist()
        self._player_node = -1

    # ------------------------------------------------------------ combat
    def _fire(self, a) -> None:
        if not a[spec.FIRE] or self.reload_t > 0.0 or self.swap_t > 0.0 or self.lock_t > 0.0:
            self.fire_cd = max(self.fire_cd, 0.0)
            return
        hw = self.weapons[self.slot]
        if hw.mag <= 0:
            self.fire_cd = max(self.fire_cd, 0.0)
            self._start_reload()
            return
        moving = a[spec.FORWARD] != 1 or a[spec.STRAFE] != 1
        interval = 60.0 / hw.w.rpm
        if hw.w.automatic:
            while self.fire_cd <= 0.0 and hw.mag > 0:
                self._shoot(hw, moving)
                self.fire_cd += interval
        elif not self._prev_applied[spec.FIRE] and self.fire_cd <= 0.0:
            # Semi-auto and bolt weapons fire on the press edge only; holding fire does nothing.
            self._shoot(hw, moving)
            self.fire_cd = interval
        else:
            self.fire_cd = max(self.fire_cd, 0.0)
        if hw.mag == 0:
            self._start_reload()

    def _shoot(self, hw: HeldWeapon, moving: bool) -> None:
        hw.mag -= 1
        self.rs.shots += 1
        self.shots_this_step += 1
        idx = np.flatnonzero(self.z_alive)
        if idx.size == 0:
            return
        p, w, rng = self.p, hw.w, self._rng
        dx = self.z_x[idx] - self.px
        dy = self.z_y[idx] - self.py
        dist = np.maximum(np.hypot(dx, dy), 0.1)
        bearing = _wrap(self.yaw - np.arctan2(dy, dx))

        spread = (w.ads_spread_deg if self.ads else w.hip_spread_deg) * p.aim_sigma_mult
        sigma = math.radians(spread * (p.move_spread_mult if moving else 1.0))
        aim_h, aim_v = rng.normal(0.0, sigma, 2)
        pitch = math.radians(self.pitch)
        if w.pellets > 1:
            pellet = rng.normal(0.0, math.radians(w.pellet_spread_deg), (2, w.pellets))
            shot_h, shot_v = aim_h + pellet[0], pitch + aim_v + pellet[1]
        else:
            shot_h, shot_v = np.array([aim_h]), np.array([pitch + aim_v])

        eye = p.eye_height_m
        head_el = np.arctan2(p.zombie_head_z_m - eye, dist)
        torso_el = np.arctan2(p.zombie_torso_z_m - eye, dist)
        head_r = np.arctan2(p.zombie_head_r_m, dist)
        half_w = np.arctan2(p.zombie_torso_half_width_m, dist)
        half_h = np.arctan2(p.zombie_torso_half_height_m, dist)

        dh = bearing[None, :] - shot_h[:, None]
        dv_torso = shot_v[:, None] - torso_el[None, :]
        head = dh**2 + (shot_v[:, None] - head_el[None, :]) ** 2 <= head_r**2
        body = (np.abs(dh) <= half_w) & (np.abs(dv_torso) <= half_h)
        hit = head | body
        if not hit.any():
            return
        masked = np.where(hit, dist[None, :], INF)
        first = masked.argmin(axis=1)
        pellets = np.flatnonzero(np.isfinite(masked[np.arange(len(first)), first]))
        tgt = first[pellets]
        targets = np.unique(tgt)
        visible = self.geo.line_of_sight(
            self.open_mask, self.px, self.py, self.z_x[idx[targets]], self.z_y[idx[targets]]
        )
        self.rs.shot_hits += bool(visible.any())
        for j, t in enumerate(targets):
            if not visible[j]:
                continue
            mine = pellets[tgt == t]
            is_head = head[mine, t]
            dmg = w.damage * p.weapon_damage_mult * np.where(is_head, w.head_mult, 1.0)
            if dist[t] > w.range_m:
                dmg = dmg * w.falloff_mult
            if is_head.any():
                where = "head"
            elif (dv_torso[mine, t] > 0.0).any():
                where = "upper"
            else:
                where = "lower"
            self._damage_zombie(int(idx[t]), float(dmg.sum()), where)

    def _damage_zombie(self, i: int, damage: float, where: str) -> None:
        self.z_hp[i] -= damage
        if self.z_hp[i] > 0.0:
            self.points += self.p.hit_points
            self.rs.hits += 1
            return
        self.z_alive[i] = False
        self.points += mechanics.SCORE_KILL + _KILL_BONUS.get(where, 0)
        self.rs.kills += 1
        self.rs.headshot_kills += where == "head"
        self.rs.melee_kills += where == "melee"

    def _melee(self) -> None:
        if self.melee_cd > 0.0:
            return
        p = self.p
        self.melee_cd = p.melee_cooldown_s
        self.lock_t = max(self.lock_t, p.melee_lockout_s)
        idx = np.flatnonzero(self.z_alive)
        if idx.size == 0:
            return
        dx = self.z_x[idx] - self.px
        dy = self.z_y[idx] - self.py
        dist = np.hypot(dx, dy)
        bearing = np.abs(_wrap(self.yaw - np.arctan2(dy, dx)))
        ok = (dist <= p.melee_range_m) & (bearing <= math.radians(p.melee_cone_deg))
        if ok.any():
            j = int(np.where(ok, dist, INF).argmin())
            # Windows don't block the knife (knifing through barricades is real); solid walls do.
            target = idx[j : j + 1]
            if self.geo.line_of_sight(self.open_mask, self.px, self.py, self.z_x[target], self.z_y[target])[0]:
                self._damage_zombie(int(idx[j]), p.melee_damage, "melee")

    def _throw_grenade(self) -> None:
        if self.grenades <= 0 or self.lock_t > 0.0:
            return
        self.grenades -= 1
        self.lock_t = GRENADE_LOCKOUT_S
        ux, uy = math.cos(self.yaw), math.sin(self.yaw)
        d = self.geo.free_distance(self.open_mask, self.px, self.py, ux, uy, self.p.grenade_throw_m)
        self.live_grenades.append([self.px + ux * d, self.py + uy * d, self.p.grenade_fuse_s])

    def _update_grenades(self, dt: float) -> None:
        if not self.live_grenades:
            return
        p = self.p
        remaining = []
        for g in self.live_grenades:
            g[2] -= dt
            if g[2] > 0.0:
                remaining.append(g)
                continue
            for i in np.flatnonzero(self.z_alive):
                d = math.hypot(self.z_x[i] - g[0], self.z_y[i] - g[1])
                if d < p.grenade_radius_m:
                    self._damage_zombie(int(i), p.grenade_damage * (1.0 - d / p.grenade_radius_m), "explosive")
            d = math.hypot(self.px - g[0], self.py - g[1])
            if d < p.grenade_radius_m:
                self._hurt_player(p.grenade_damage * p.grenade_self_damage_mult * (1.0 - d / p.grenade_radius_m))
        self.live_grenades = remaining

    # ------------------------------------------------------------ player movement & health
    def _move(self, a, dt: float) -> None:
        f, s = _FWD[a[spec.FORWARD]], _STRAFE[a[spec.STRAFE]]
        if f == 0 and s == 0:
            return
        p = self.p
        speed_f = (p.player_run_speed * (p.sprint_mult if self.sprinting else 1.0)) if f > 0 else p.player_back_speed
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vx = f * speed_f * cy + s * p.player_strafe_speed * sy
        vy = f * speed_f * sy - s * p.player_strafe_speed * cy
        cap = max(speed_f if f else 0.0, p.player_strafe_speed if s else 0.0)
        k = dt * min(1.0, cap / math.hypot(vx, vy)) * (p.ads_move_mult if self.ads else 1.0)
        nx, ny = self.px + vx * k, self.py + vy * k

        self._blockers = None
        if self._nearest_inside < BLOCK_CHECK_M:
            self._blockers = self.z_pos[self.z_alive & (self.z_phase == INSIDE)]
        if self._free(nx, ny):
            self.px, self.py = nx, ny
        elif self._free(nx, self.py):
            self.px = nx
        elif self._free(self.px, ny):
            self.py = ny
        x0, y0, inv, gnx = self._grid
        zone = self._zone_list[int((self.py - y0) * inv) * gnx + int((self.px - x0) * inv)]
        if zone >= 0:
            self.zone = zone

    def _free(self, x: float, y: float) -> bool:
        walk, r = self._walk_list, self.p.player_radius_m
        x0, y0, inv, nx = self._grid
        for qx, qy in ((x, y), (x + r, y), (x - r, y), (x, y + r), (x, y - r)):
            if not walk[int((qy - y0) * inv) * nx + int((qx - x0) * inv)]:
                return False
        if self._blockers is None or self._blockers.size == 0:
            return True
        # Zombies body-block the player (getting cornered is real), but never trap one already overlapping.
        d_new = np.abs(self._blockers - complex(x, y))
        d_old = np.abs(self._blockers - complex(self.px, self.py))
        return not ((d_new < r + self.p.zombie_radius_m) & (d_new < d_old)).any()

    def _hurt_player(self, damage: float) -> None:
        self.hp -= damage
        self.time_since_hit = 0.0
        self.rs.damage_taken += 1

    def _update_health(self, dt: float) -> None:
        p, h = self.p, self.config.hardness
        self.time_since_hit += dt
        if self.time_since_hit >= p.player_regen_delay_s:
            self.hp = min(p.player_max_hp, self.hp + p.player_regen_rate * dt)
        target = 1.0 if self.hp <= 0.0 else min(1.0, p.flash_gain * (1.0 - self.hp / p.player_max_hp))
        decay = math.exp(-dt / p.flash_decay_tau_s)
        self.flash = max(self.flash * decay, target)
        observed = self.flash
        if h > 0.0:
            self.flash_fp *= decay
            if self._uniform() < p.flash_false_positive_hz * h * dt:
                self.flash_fp += 0.2 + 0.4 * self._uniform()
            observed = min(1.0, max(0.0, observed + self.flash_fp + p.flash_noise_std * h * self._normal()))
        self._damage_detected = observed - self.flash_obs > p.flash_detect_jump
        self.flash_obs = observed
        self.time_since_damage = 0.0 if self._damage_detected else self.time_since_damage + dt

    # ------------------------------------------------------------ rounds & zombies
    def _refresh_distance_field(self) -> None:
        x0, y0, inv, nx = self._grid
        node = self._node_list[int((self.py - y0) * inv) * nx + int((self.px - x0) * inv)]
        if node >= 0 and node != self._player_node:
            self._player_node = node
            self._dist = self.geo.distance_field(self.open_mask, node)

    def _start_round(self) -> None:
        self.to_spawn = mechanics.zombies_in_round_solo(self.round)
        self.zombie_max_hp = float(mechanics.zombie_health(self.round))
        self.spawn_t = self.p.first_spawn_delay_s
        self.round_time = 0.0
        self.rs = RoundStats()

    def _update_round(self, dt: float, acc: _StepAccum) -> None:
        if self.intermission_t > 0.0:
            self.intermission_t -= dt
            if self.intermission_t <= 0.0:
                self.intermission_t = 0.0
                self.round += 1
                self._start_round()
                self.grenades = min(self.p.grenades_max, self.grenades + self.p.grenades_per_round)
                self._events.append({"type": "round_start", "round": self.round, "t": self.t})
            return
        self.round_time += dt
        if self.to_spawn > 0:
            self.spawn_t -= dt
            if self.spawn_t <= 0.0:
                alive = int(self.z_alive.sum())
                if alive <= mechanics.MAX_ENEMY_COUNT and alive < MAX_ZOMBIES:
                    self._spawn_zombie()
                    self.to_spawn -= 1
                    self.spawn_t += mechanics.SPAWN_DELAY_S
        elif not self.z_alive.any():
            acc.rounds_completed += 1
            self._events.append(
                {"type": "round_complete", "round": self.round, "duration_s": self.round_time, "t": self.t}
                | asdict(self.rs)
            )
            self.intermission_t = self.p.round_intermission_s

    def _spawn_zombie(self) -> None:
        p, rng = self.p, self._rng
        i = int(np.flatnonzero(~self.z_alive)[0])
        w = self._active_window_list[int(rng.integers(len(self._active_window_list)))]
        win = self.geo.windows[w]
        jitter = rng.uniform(-0.3, 0.3)
        self.z_pos[i] = complex(win.spawn[0] - win.normal[1] * jitter, win.spawn[1] + win.normal[0] * jitter)
        self.z_phase[i] = APPROACH
        self.z_window[i] = w
        self.z_hp[i] = self.zombie_max_hp
        self.z_event_t[i] = self.z_ready_t[i] = 0.0
        self.z_hit_t[i] = INF
        ms = mechanics.zombie_move_speed(self.round)
        tier = int(rng.integers(ms - p.zombie_speed_jitter, ms + p.zombie_speed_jitter))
        if tier <= p.zombie_walk_max:
            speed = p.zombie_walk_speed
        elif tier <= p.zombie_run_max:
            speed = p.zombie_run_speed
        else:
            speed = p.zombie_sprint_speed
        self.z_speed[i] = speed * rng.uniform(0.9, 1.1)
        self.z_alive[i] = True
        self.rs.spawned += 1

    def _update_zombies(self, dt: float) -> None:
        idx = np.flatnonzero(self.z_alive)
        if idx.size == 0:
            self._nearest_inside = INF
            return
        p, now = self.p, self.t
        ph = self.z_phase[idx]
        counts = np.bincount(ph, minlength=4).tolist()

        if counts[APPROACH]:
            ia = idx[ph == APPROACH]
            delta = self._win_out[self.z_window[ia]] - self.z_pos[ia]
            d = np.abs(delta)
            step = self.z_speed[ia] * dt
            self.z_pos[ia] += delta * np.minimum(step / np.maximum(d, 1e-9), 1.0)
            arrived = ia[d <= step]
            self.z_phase[arrived] = TEARING
            self.z_event_t[arrived] = now + p.zombie_tear_interval_s

        if counts[TEARING]:
            it = idx[ph == TEARING]
            win = self.z_window[it]
            ready = self.z_event_t[it] <= now
            if ready.any():
                removed = np.bincount(win[ready], minlength=self.planks.size)
                np.maximum(self.planks - removed, 0, out=self.planks)
                self.z_event_t[it[ready]] += p.zombie_tear_interval_s
            through = it[self.planks[win] == 0]
            self.z_phase[through] = CLIMBING
            self.z_event_t[through] = now + p.zombie_climb_s

        if counts[CLIMBING]:
            ic = idx[ph == CLIMBING]
            done = ic[self.z_event_t[ic] <= now]
            self.z_phase[done] = INSIDE
            self.z_pos[done] = self._win_in[self.z_window[done]]

        if counts[INSIDE]:
            self._steer(idx[ph == INSIDE], dt)
        self._zombie_attacks(idx)

    def _cells(self, z: np.ndarray) -> np.ndarray:
        """Grid cells of in-bounds complex positions (inside zombies and their candidate moves)."""
        _, _, inv, nx = self._grid
        c = (z - self._origin) * inv
        return c.imag.astype(np.int64) * nx + c.real.astype(np.int64)

    def _steer(self, idx: np.ndarray, dt: float) -> None:
        geo, p = self.geo, self.p
        pos = self.z_pos[idx]
        nodes = geo.node_of_cell[self._cells(pos)]
        D = self._dist
        here = np.where(nodes >= 0, D[nodes], INF)
        nbrs = geo.node_nbrs[nodes]
        nd = np.where(nbrs >= 0, D[nbrs], INF)
        best = nd.argmin(axis=1)
        rows = np.arange(idx.size)
        follow = (nodes >= 0) & (here > DIRECT_CHASE_M) & (nd[rows, best] < here)
        target = np.where(follow, self._node_c[nbrs[rows, best]], complex(self.px, self.py))

        delta = target - pos
        d = np.abs(delta)
        speed = self.z_speed[idx] * dt
        swinging = self.z_hit_t[idx] < INF
        if swinging.any():
            speed = np.where(swinging, 0.3 * speed, speed)
        reach = np.where(follow, d, np.maximum(d - 0.7 * p.zombie_attack_range_m, 0.0))
        new = pos + delta * (np.minimum(speed, reach) / np.maximum(d, 1e-9))

        if idx.size > 1:
            sep = new[:, None] - new[None, :]
            sd = np.abs(sep)
            np.fill_diagonal(sd, INF)
            push = np.maximum(2.0 * p.zombie_radius_m - sd, 0.0) / np.maximum(sd, 1e-6)
            new = new + 0.5 * (push * sep).sum(axis=1)

        self.z_pos[idx] = np.where(self._walkable[self._cells(new)], new, pos)

    def _zombie_attacks(self, idx: np.ndarray) -> None:
        p, now = self.p, self.t
        ph = self.z_phase[idx]
        d = np.abs(self.z_pos[idx] - complex(self.px, self.py))
        inside = ph == INSIDE
        self._nearest_inside = float(d[inside].min(initial=INF))
        reach = (inside & (d <= p.zombie_attack_range_m)) | (
            (ph == TEARING) & (d <= p.zombie_reach_through_window_m)
        )
        hit_t = self.z_hit_t[idx]
        swinging = hit_t < INF
        if not (reach.any() or swinging.any()):
            return
        # A swing lands only if the player is still in reach when it finishes: backing off dodges it.
        landed = swinging & (hit_t <= now)
        hits = int((landed & reach).sum())
        self.z_hit_t[idx[landed]] = INF
        start = idx[reach & ~swinging & (self.z_ready_t[idx] <= now)]
        self.z_hit_t[start] = now + p.zombie_attack_windup_s
        self.z_ready_t[start] = now + p.zombie_attack_cooldown_s
        for _ in range(hits):
            self._hurt_player(p.zombie_damage)

    # ------------------------------------------------------------ observations
    def _observation(self, downed: bool = False) -> dict[str, np.ndarray]:
        # The prompt is read first either way: the state vector does not use it, but the HUD and the
        # rendered view both draw it, and they must agree about the same step.
        self.prompt = self._prompt()
        obs = {"hud": self._hud(downed), "prev_actions": spec.encode_prev_actions(self._history)}
        obs["pixels" if self.pixel_obs else "state"] = self.frame() if self.pixel_obs else self._state_vector()
        return obs

    def _state_vector(self) -> np.ndarray:
        px, py, yaw = self.px, self.py, self.yaw
        x0, y0, ix, iy = self._norm
        zones = [0.0] * len(spec.ZONES)
        zones[self.zone] = 1.0
        best_d, best_w = INF, -1
        for w in self._active_window_list:
            wx, wy = self._window_xy[w]
            dd = math.hypot(wx - px, wy - py)
            if dd < best_d:
                best_d, best_w = dd, w
        wx, wy = self._window_xy[best_w]
        b = yaw - math.atan2(wy - py, wx - px)
        s = np.array(
            [(px - x0) * ix, (py - y0) * iy, math.sin(yaw), math.cos(yaw), self.pitch / spec.STATE_PITCH_SCALE_DEG]
            + zones
            + [float(self.open_mask >> i & 1) for i in range(len(spec.DOORS))]
            + list(_WEAPON_FEATURES[self.weapons[self.slot].w.name])
            + _EMPTY_ZOMBIES
            + [
                math.sin(b),
                math.cos(b),
                min(1.0, best_d / spec.STATE_DIST_SCALE_M),
                self.planks[best_w] / spec.MAX_PLANKS,
            ],
            dtype=np.float32,
        )

        idx = np.flatnonzero(self.z_alive)
        if idx.size:
            rel = self.z_pos[idx] - complex(px, py)
            d = np.abs(rel)
            order = np.argsort(d)[:_K]
            dn = np.maximum(d[order], 1e-9)
            # Rotating by -yaw puts the facing direction on +x; clockwise-positive bearing b has
            # cos b = Re / |rel| and sin b = -Im / |rel|, no trig needed.
            rot = rel[order] * complex(math.cos(yaw), -math.sin(yaw))
            end = _Z0 + 4 * order.size
            s[_Z0:end:4] = 1.0
            s[_Z0 + 1 : end : 4] = -rot.imag / dn
            s[_Z0 + 2 : end : 4] = rot.real / dn
            s[_Z0 + 3 : end : 4] = np.minimum(d[order] / spec.STATE_DIST_SCALE_M, 1.0)
        return s

    def _hud(self, downed: bool) -> np.ndarray:
        rng, h = self._rng, self.config.hardness
        hw = self.weapons[self.slot]
        raw = [0.0] * spec.HUD_DIM
        raw[_H["round"]] = self.round
        raw[_H["points"]] = self.points
        raw[_H["mag_ammo"]] = hw.mag
        raw[_H["reserve_ammo"]] = hw.reserve
        raw[_H["grenades"]] = self.grenades
        raw[_H["damage_flash"]] = self.flash_obs
        raw[_H["time_since_damage"]] = min(self.time_since_damage, 60.0)
        raw[_H["round_transition"]] = float(self.intermission_t > 0.0)
        raw[_H["downed"]] = float(downed)
        kind, price, _ = self.prompt
        if kind != PROMPT_NONE:
            raw[_PROMPT_HUD[kind]] = 1.0
            raw[_H["prompt_price"]] = price
        raw[_H["time_in_round"]] = self.round_time
        confidence = 1.0
        if h > 0.0:
            confidence -= 0.1 * h * self._uniform()
            if self._uniform() < len(_MISREAD_FIELDS) * self.p.hud_misread_prob * h:
                f = _MISREAD_FIELDS[int(rng.integers(len(_MISREAD_FIELDS)))]
                raw[f] = _corrupt_digits(raw[f], rng)
                confidence = rng.uniform(0.3, 0.9)
        raw[_H["hud_confidence"]] = confidence
        return spec.encode_hud(raw)

    def _episode_summary(self, terminated: bool) -> dict:
        st = self.reward_shaper.stats
        out = {
            "round_reached": self.round,
            "episode_steps": self.steps,
            "episode_time_s": self.t,
            "repair_share": st.repair_share(),
            "max_term_share": st.max_term_share(),
            "gain_clips": st.gain_clips,
            "reward_term_sums": dict(zip(REWARD_TERMS, st.term_sums.tolist())),
        }
        if terminated:
            out["rounds_survived"] = self.round
        return out
