"""Hand-written state-mode baseline: fight the nearest visible threat, knife when dry, shop and repair in lulls."""

import math

import numpy as np

from zombiesai import spec
from zombiesai.sim.geometry import Geometry, load_geometry
from zombiesai.sim.nacht_sim import weapon_features
from zombiesai.sim.params import SimParams, WEAPONS

_H = spec.HUD_INDEX
_S = spec.STATE_INDEX
_Z0 = spec.STATE_ZOMBIE_START
_K = spec.STATE_K_ZOMBIES
_OPEN0 = _S[f"open_{spec.DOORS[0]}"]
_YAW_BINS = np.array(spec.YAW_BINS_DEG)
_PITCH_BINS = np.array(spec.PITCH_BINS_DEG)
SHOP_WEAPON = "m1a1_carbine"
HIP_AIM_HEIGHT_M = 1.30  # upper torso: the forgiving target when spread is wide
ADS_AIM_HEIGHT_M = 1.62  # head: tight ADS spread makes the 2.5x multiplier worth it
ADS_MIN_M = 2.5
SAFE_TO_SHOP_M = 5.0
CLOSE_FIRE_M = 3.0
KITE_RADIUS_M = 4.0
KITE_TURN_DEG = 20.0
# Actions land 0-2 steps late, so re-aiming before the last turn has shown up in the observation
# double-counts it and oscillates. Wait this many steps after any turn before trusting the bearing.
SETTLE_STEPS = 2
_EYE_HEIGHT_M = SimParams().eye_height_m


def _nearest_bin(bins: np.ndarray, value: float) -> float:
    return float(bins[np.abs(bins - value).argmin()])


class ScriptedAgent:
    def __init__(
        self,
        geometry: Geometry | None = None,
        engage_m: float = 12.0,
        backpedal_m: float = 2.5,
        repair_m: float = 6.0,
    ):
        self.geo = geometry or load_geometry()
        self.engage_m = engage_m
        self.backpedal_m = backpedal_m
        self.repair_m = repair_m
        x0, y0, x1, y1 = self.geo.bounds
        self._origin = (x0, y0)
        self._extent = (x1 - x0, y1 - y0)
        self._home = tuple(float(v) for v in self.geo.player_spawn)
        shop = next(w for w in self.geo.wall_weapons if w.weapon == SHOP_WEAPON)
        self._shop_pos = tuple(float(v) for v in shop.pos)
        self._shop_price = shop.price
        self._shop_features = np.array(weapon_features(WEAPONS[SHOP_WEAPON]))
        self.reset()

    def reset(self) -> None:
        self._prev_fire = 0
        self._prev_button = "none"
        self._swapped_dry = False
        self._since_turn = SETTLE_STEPS

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        s = obs["state"]
        hud = spec.decode_hud(obs["hud"])
        mag, reserve = round(hud[_H["mag_ammo"]]), round(hud[_H["reserve_ammo"]])
        if mag or reserve:
            self._swapped_dry = False

        px = self._origin[0] + s[_S["player_x"]] * self._extent[0]
        py = self._origin[1] + s[_S["player_y"]] * self._extent[1]
        yaw = math.atan2(s[_S["yaw_sin"]], s[_S["yaw_cos"]])
        open_mask = sum(1 << i for i in range(len(spec.DOORS)) if s[_OPEN0 + i] > 0.5)
        pitch = float(s[_S["pitch"]]) * spec.STATE_PITCH_SCALE_DEG

        kw: dict = {}
        target = self._visible_target(s, px, py, yaw, open_mask)
        if target is not None and not (target[1] > SAFE_TO_SHOP_M and self._shopping(s, hud)):
            self._fight(s, target, mag, reserve, pitch, px, py, yaw, open_mask, kw)
        elif mag == 0 and reserve > 0:
            kw["button"] = "reload"
        else:
            if abs(pitch) > 2.0 and self._since_turn >= SETTLE_STEPS:
                kw["pitch"] = _nearest_bin(_PITCH_BINS, -pitch)
            self._lull(s, hud, px, py, yaw, kw)

        if kw.get("button", "none") == self._prev_button and self._prev_button in ("melee", "swap", "reload"):
            kw["button"] = "none"
        self._prev_button = kw.get("button", "none")
        self._prev_fire = kw.get("fire", 0)
        turned = kw.get("yaw", 0.0) != 0.0 or kw.get("pitch", 0.0) != 0.0
        self._since_turn = 0 if turned else self._since_turn + 1
        return spec.make_action(**kw)

    def _visible_target(self, s, px, py, yaw, open_mask) -> tuple[float, float] | None:
        """(bearing_deg, dist_m) of the nearest zombie with line of sight, from the state's egocentric slots."""
        z = s[_Z0 : _Z0 + 4 * _K].reshape(_K, 4)
        present = np.flatnonzero(z[:, 0] > 0.5)
        if present.size == 0:
            return None
        bearing = np.arctan2(z[present, 1], z[present, 2])
        dist = z[present, 3] * spec.STATE_DIST_SCALE_M
        heading = yaw - bearing
        visible = self.geo.line_of_sight(
            open_mask, px, py, px + dist * np.cos(heading), py + dist * np.sin(heading)
        ) & (dist <= self.engage_m)
        if not visible.any():
            return None
        j = int(np.flatnonzero(visible)[0])
        return math.degrees(bearing[j]), float(dist[j])

    def _fight(self, s, target, mag, reserve, pitch, px, py, yaw, open_mask, kw) -> None:
        bearing, dist = target
        settled = self._since_turn >= SETTLE_STEPS
        kiting = dist < self.backpedal_m
        cone = max(1.5, math.degrees(math.atan2(0.2, max(dist, 0.1))))
        aim_z = ADS_AIM_HEIGHT_M if dist > ADS_MIN_M else HIP_AIM_HEIGHT_M
        aim_pitch = math.degrees(math.atan2(aim_z - _EYE_HEIGHT_M, max(dist, 0.3)))
        # While kiting, only fix big bearing errors: every turn rotates the frame the move is expressed in.
        turn_threshold = KITE_TURN_DEG if kiting else cone
        if (settled or dist < CLOSE_FIRE_M) and abs(bearing) > turn_threshold:
            kw["yaw"] = _nearest_bin(_YAW_BINS, bearing)
        if settled and abs(aim_pitch - pitch) > 2.0:
            kw["pitch"] = _nearest_bin(_PITCH_BINS, aim_pitch - pitch)
        aimed = abs(bearing) <= cone and (settled or dist < CLOSE_FIRE_M)

        if kiting:
            self._kite(s, px, py, yaw, open_mask, kw)
        elif dist > ADS_MIN_M:
            kw["ads"] = 1

        if mag == 0 and reserve == 0:
            if not self._swapped_dry:
                kw["button"] = "swap"
                self._swapped_dry = True
            elif dist < 1.6 and abs(bearing) < 30.0:
                kw["button"] = "melee"
        elif mag == 0:
            kw["button"] = "reload"
        elif aimed:
            automatic = s[_S["weapon_auto"]] > 0.5
            kw["fire"] = 1 if automatic else 1 - self._prev_fire

    def _kite(self, s, px, py, yaw, open_mask, kw) -> None:
        """Pick the move direction with the most open floor that heads least toward nearby zombies."""
        walk = self.geo.walkable(open_mask)
        c, sn = math.cos(yaw), math.sin(yaw)
        z = s[_Z0 : _Z0 + 4 * _K].reshape(_K, 4)
        near = (z[:, 0] > 0.5) & (z[:, 3] * spec.STATE_DIST_SCALE_M < KITE_RADIUS_M)
        zd = z[near, 3] * spec.STATE_DIST_SCALE_M
        # Egocentric (forward, right) frame; bearings are clockwise-positive, so right = sin(bearing).
        zdir = np.stack((z[near, 2], z[near, 1]), axis=1)

        best, best_score = None, -math.inf
        for f, r in ((-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 1), (1, -1), (1, 0)):
            norm = math.hypot(f, r)
            wx, wy = (f * c + r * sn) / norm, (f * sn - r * c) / norm
            free = 0.0
            for d in (0.6, 1.2, 1.8, 2.4):
                if not walk[self.geo.cell(px + d * wx, py + d * wy)]:
                    break
                free = d
            toward = np.clip(zdir @ np.array((f / norm, r / norm)), 0.0, None)
            score = free - 3.0 * float((toward * (KITE_RADIUS_M - zd) / KITE_RADIUS_M).sum())
            if score > best_score:
                best, best_score = (f, r), score
        if best is not None:
            kw["forward"], kw["strafe"] = best

    def _shopping(self, s, hud) -> bool:
        points = hud[_H["points"]]
        weapon = s[_S["weapon_damage"] : _S["weapon_damage"] + 3]
        if not np.allclose(weapon, self._shop_features, atol=1e-4):
            return points >= self._shop_price
        low = hud[_H["reserve_ammo"]] < 0.5 * WEAPONS[SHOP_WEAPON].max_reserve
        return low and points >= round(self._shop_price * 0.5)

    def _lull(self, s, hud, px, py, yaw, kw) -> None:
        points = hud[_H["points"]]
        if self._shopping(s, hud):
            if hud[_H["prompt_weapon"]] > 0.5 and hud[_H["prompt_price"]] <= points:
                kw["button"] = "use" if self._prev_button != "use" else "none"
            else:
                self._walk_to(px, py, yaw, self._shop_pos, kw)
            return

        barrier_dist = s[_S["barrier_dist"]] * spec.STATE_DIST_SCALE_M
        if s[_S["barrier_planks"]] < 1.0 and barrier_dist <= self.repair_m:
            if hud[_H["prompt_repair"]] > 0.5:
                kw["button"] = "use"
            else:
                bearing = math.degrees(math.atan2(s[_S["barrier_sin"]], s[_S["barrier_cos"]]))
                kw["yaw"] = _nearest_bin(_YAW_BINS, bearing)
                if abs(bearing) < 30.0:
                    kw["forward"] = 1
            return

        z0 = s[_Z0 : _Z0 + 4]
        if z0[0] > 0.5:
            kw["yaw"] = _nearest_bin(_YAW_BINS, math.degrees(math.atan2(z0[1], z0[2])))
        else:
            self._walk_to(px, py, yaw, self._home, kw, arrive_m=1.0)

    def _walk_to(self, px, py, yaw, target, kw, arrive_m: float = 0.6) -> None:
        dx, dy = target[0] - px, target[1] - py
        if math.hypot(dx, dy) <= arrive_m:
            return
        bearing = math.degrees((yaw - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi)
        kw["yaw"] = _nearest_bin(_YAW_BINS, bearing)
        if abs(bearing) < 30.0:
            kw["forward"] = 1
