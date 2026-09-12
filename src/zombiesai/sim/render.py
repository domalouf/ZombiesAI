"""First-person raycast renderer for NachtSim: the plan's crude DOOM-style render mode.

It reproduces the shape of the real perception problem rather than the real scene: partial observability, small
targets that have to be centred, a red damage vignette, and a HUD strip. Visual sim-to-real transfer is a non-goal;
see docs/sim_lies.md.
"""

import math

import numpy as np

from zombiesai import spec
from zombiesai.sim.bitmap_font import GLYPH_H, draw_text, text_width
from zombiesai.sim.geometry import DOOR, WINDOW, Geometry

# WaW's cg_fov 65 is the horizontal FOV at 4:3; its Hor+ widescreen widens that to about 80 degrees at 16:9.
FOV_DEG = 80.0
WALL_HEIGHT_M = 3.0
SILL_M, LINTEL_M = 0.9, 2.1
PANEL_M = 1.0  # top of the darker wainscot band along the walls
MARCH_STEP_M = 0.1
MAX_VIEW_M = 50.0
NEAR_M = 0.15
FOG_M = 22.0


def _c(r: int, g: int, b: int) -> np.ndarray:
    return np.array((r, g, b), dtype=np.float32)


CEILING_FAR, CEILING_NEAR = _c(36, 32, 30), _c(62, 55, 50)
FLOOR_FAR, FLOOR_NEAR = _c(44, 38, 32), _c(104, 88, 70)
WALL = _c(138, 120, 98)
DOOR_COLORS = {"door": _c(112, 74, 42), "debris": _c(92, 84, 74)}
PLANK = _c(150, 108, 64)
GUN, GUN_EDGE, FLASH, BLADE = _c(46, 46, 50), _c(88, 88, 94), _c(255, 236, 160), _c(192, 198, 206)
ROUND_RED, ROUND_LULL = _c(178, 22, 18), _c(232, 214, 204)
POINTS, AMMO, PROMPT, GRENADE_ICON = _c(242, 228, 176), _c(236, 236, 236), _c(232, 164, 40), _c(150, 160, 130)
CROSSHAIR = _c(220, 220, 220)
VIGNETTE = _c(150, 0, 0)

# Billboards as (bottom m, top m, lateral offset m, half-width m, colour), drawn in order.
_TROUSERS, _TUNIC, _SKIN, _EYES = (50, 54, 48), (88, 96, 80), (156, 164, 132), (24, 18, 16)
ZOMBIE = (
    (0.0, 0.85, -0.10, 0.075, _TROUSERS),
    (0.0, 0.85, 0.10, 0.075, _TROUSERS),
    (0.95, 1.40, -0.31, 0.06, _TUNIC),
    (0.95, 1.40, 0.31, 0.06, _TUNIC),
    (0.80, 1.48, 0.0, 0.24, _TUNIC),
    (1.50, 1.74, 0.0, 0.12, _SKIN),
    (1.60, 1.65, -0.05, 0.025, _EYES),
    (1.60, 1.65, 0.05, 0.025, _EYES),
)
MYSTERY_BOX = ((0.0, 0.75, 0.0, 0.55, (84, 58, 32)), (0.62, 0.68, 0.0, 0.56, (120, 170, 220)))
CHALK_OUTLINE = (
    (1.10, 1.50, 0.0, 0.42, (214, 212, 200)),
    (1.14, 1.46, 0.0, 0.38, (104, 90, 74)),
    (1.26, 1.34, 0.0, 0.30, (214, 212, 200)),
)
LIVE_GRENADE = ((0.0, 0.12, 0.0, 0.07, (40, 46, 36)),)


def _fog(dist):
    return np.clip(1.1 - dist / FOG_M, 0.3, 1.0)


class Renderer:
    """Draws a NachtSim's current state. Build one per resolution: it caches per-column and per-pixel tables."""

    def __init__(
        self,
        geo: Geometry,
        width: int = spec.PIXELS_SHAPE[1],
        height: int = spec.PIXELS_SHAPE[0],
        fov_deg: float = FOV_DEG,
    ):
        self.geo, self.w, self.h = geo, width, height
        self.focal = width / 2 / math.tan(math.radians(fov_deg) / 2)
        self._offsets = np.arctan((np.arange(width) + 0.5 - width / 2) / self.focal)  # right-positive
        self._cos = np.cos(self._offsets)
        self._t = np.arange(1, int(MAX_VIEW_M / MARCH_STEP_M) + 1) * MARCH_STEP_M
        self._cols = np.arange(width)
        self._rows = (np.arange(height, dtype=np.float32) + 0.5)[:, None]
        self.s = max(1, round(height / 144))  # HUD pixel size: 1 at the agent's 72 rows
        yy = (np.arange(height) + 0.5 - height / 2) / (height / 2)
        xx = (np.arange(width) + 0.5 - width / 2) / (width / 2)
        r = np.hypot(xx[None, :], yy[:, None]) / math.sqrt(2)
        self._vignette = (np.clip((r - 0.35) / 0.65, 0.0, 1.0) ** 1.5).astype(np.float32)[..., None]
        self._door_colors = np.array([DOOR_COLORS[d.kind] for d in geo.doors] + [WALL], dtype=np.float32)
        # Each WINDOW cell belongs to its nearest window, so a ray through it can look up that window's planks.
        self._window_of_cell = np.full(geo.kind.size, -1, dtype=np.int64)
        cells = np.flatnonzero(geo.kind == WINDOW)
        xy = np.array([geo.cell_center(int(c)) for c in cells]).reshape(-1, 2)
        self._window_of_cell[cells] = np.linalg.norm(xy[:, None] - geo.window_pos[None], axis=2).argmin(axis=1)

    def render(self, sim) -> np.ndarray:
        eye = sim.p.eye_height_m
        horizon = self.h / 2 + self.focal * math.tan(math.radians(sim.pitch))
        img = self._background(horizon)
        zbuf = np.full((self.h, self.w), np.inf, dtype=np.float32)
        rays = self._cast(sim)
        self._walls(img, zbuf, sim, rays, horizon, eye)
        self._sprites(img, zbuf, sim, horizon, eye)
        self._windows(img, zbuf, sim, rays, horizon, eye)
        self._viewmodel(img, sim)
        if sim.flash_obs > 0.0:
            alpha = self._vignette * (0.85 * min(1.0, float(sim.flash_obs)))
            img = img * (1.0 - alpha) + VIGNETTE * alpha
        if sim.hp <= 0.0:  # downed: the last-stand screen drains of colour
            img = img.mean(axis=2, keepdims=True) * 0.55 + VIGNETTE * 0.35
        self._hud(img, sim)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ world
    def _background(self, horizon: float) -> np.ndarray:
        rows = self._rows[:, 0]
        near = np.clip(np.abs(rows - horizon) / (self.h / 2), 0.0, 1.0)[:, None]
        ceiling = CEILING_FAR + (CEILING_NEAR - CEILING_FAR) * near
        floor = FLOOR_FAR + (FLOOR_NEAR - FLOOR_FAR) * near
        column = np.where((rows >= horizon)[:, None], floor, ceiling)
        return np.repeat(column[:, None, :], self.w, axis=1)

    def _entry(self, cells: np.ndarray, px: float, py: float, dx: np.ndarray, dy: np.ndarray):
        """Exact distance at which each ray enters its cell, and whether it came through an x = const face."""
        geo, cs = self.geo, self.geo.cell_size
        iy, ix = np.divmod(cells, geo.nx)
        x0 = geo.bounds[0] + ix * cs
        y0 = geo.bounds[1] + iy * cs
        tx = np.minimum((x0 - px) / dx, (x0 + cs - px) / dx)
        ty = np.minimum((y0 - py) / dy, (y0 + cs - py) / dy)
        return np.maximum(tx, ty), tx > ty

    def _cast(self, sim):
        """March every column's ray through the grid at once, then solve each first hit exactly."""
        angle = sim.yaw - self._offsets  # screen-right is clockwise
        dx, dy = np.cos(angle), np.sin(angle)
        dx[np.abs(dx) < 1e-9] = 1e-9
        dy[np.abs(dy) < 1e-9] = 1e-9
        cells = self.geo.cells(sim.px + dx[:, None] * self._t, sim.py + dy[:, None] * self._t)
        blocked = self.geo.sight_block(sim.open_mask)[cells]
        hit = blocked.argmax(axis=1)
        hit_cell = cells[self._cols, hit]
        t, x_face = self._entry(hit_cell, sim.px, sim.py, dx, dy)
        t = np.where(blocked[self._cols, hit], t, MAX_VIEW_M)
        return cells, hit, hit_cell, t, x_face, dx, dy

    def _walls(self, img, zbuf, sim, rays, horizon: float, eye: float) -> None:
        cells, hit, hit_cell, t, x_face, dx, dy = rays
        geo, f = self.geo, self.focal
        perp = np.maximum(t * self._cos, NEAR_M).astype(np.float32)
        door = geo.kind[hit_cell] == DOOR
        base = np.where(door[:, None], self._door_colors[geo.door_of_cell[hit_cell]], WALL)
        along = np.where(x_face, sim.py + dy * t, sim.px + dx * t)
        spacing = np.where(door, 0.25, 1.0)
        seam = (along % spacing) < 0.05 * spacing
        shade = _fog(perp) * np.where(x_face, 1.0, 0.8)

        rows = self._rows
        mask = (rows >= horizon - f * (WALL_HEIGHT_M - eye) / perp) & (rows < horizon + f * eye / perp)
        height = eye + (horizon - rows) * perp / f
        panel = np.where(height < PANEL_M, 0.78, 1.0) * np.where(np.abs(height - PANEL_M) < 0.03, 0.7, 1.0)
        tone = np.where(door[None, :], 1.0, panel) * np.where(seam, 0.72, 1.0)[None, :]
        color = base[None, :, :] * (shade[None, :] * tone)[..., None]
        img[mask] = color[mask]
        zbuf[mask] = np.broadcast_to(perp, mask.shape)[mask]

    def _sprites(self, img, zbuf, sim, horizon: float, eye: float) -> None:
        c, s = math.cos(sim.yaw), math.sin(sim.yaw)
        things = [(float(sim.z_x[i]), float(sim.z_y[i]), ZOMBIE) for i in np.flatnonzero(sim.z_alive)]
        things.append((float(self.geo.box_pos[0]), float(self.geo.box_pos[1]), MYSTERY_BOX))
        things += [(float(x), float(y), CHALK_OUTLINE) for x, y in self.geo.wall_weapon_pos]
        things += [(g[0], g[1], LIVE_GRENADE) for g in sim.live_grenades]
        projected = []
        for x, y, parts in things:
            rx, ry = x - sim.px, y - sim.py
            forward = rx * c + ry * s
            if forward > NEAR_M:
                projected.append((forward, rx * s - ry * c, parts))
        projected.sort(key=lambda p: -p[0])  # far to near
        for forward, right, parts in projected:
            self._billboard(img, zbuf, forward, right, parts, horizon, eye)

    def _billboard(self, img, zbuf, forward: float, right: float, parts, horizon: float, eye: float) -> None:
        k = self.focal / forward
        cx = self.w / 2 + right * k
        shade = float(_fog(forward))
        for n, (lo, hi, offset, half_w, color) in enumerate(parts):
            x0, x1 = max(0, round(cx + (offset - half_w) * k)), min(self.w, round(cx + (offset + half_w) * k))
            y0, y1 = max(0, round(horizon - (hi - eye) * k)), min(self.h, round(horizon - (lo - eye) * k))
            if x0 >= x1 or y0 >= y1:
                continue
            depth = forward - n * 1e-4  # later parts of the same sprite sit just in front of earlier ones
            region = zbuf[y0:y1, x0:x1]
            visible = region > depth
            img[y0:y1, x0:x1][visible] = np.asarray(color, dtype=np.float32) * shade
            region[visible] = depth

    def _windows(self, img, zbuf, sim, rays, horizon: float, eye: float) -> None:
        """The wall above and below each window opening, and its planks, in front of whatever is outside."""
        cells, hit, _, _, _, dx, dy = rays
        is_window = (self.geo.kind[cells] == WINDOW) & (np.arange(cells.shape[1])[None, :] < hit[:, None])
        first = is_window.argmax(axis=1)
        cols = self._cols[is_window[self._cols, first]]
        if cols.size == 0:
            return
        window_cell = cells[cols, first[cols]]
        t, x_face = self._entry(window_cell, sim.px, sim.py, dx[cols], dy[cols])
        perp = np.maximum(t * self._cos[cols], NEAR_M).astype(np.float32)
        planks = sim.planks[self._window_of_cell[window_cell]]

        height = eye + (self._rows - horizon) * (-perp) / self.focal
        frame = ((height >= 0.0) & (height < SILL_M)) | ((height > LINTEL_M) & (height <= WALL_HEIGHT_M))
        board = (height - SILL_M) / (LINTEL_M - SILL_M) * spec.MAX_PLANKS
        boarded = (height >= SILL_M) & (height <= LINTEL_M) & (board % 1.0 < 0.72) & (np.floor(board) < planks)
        sub_img, sub_z = img[:, cols], zbuf[:, cols]
        in_front = sub_z > perp
        shade = (_fog(perp) * np.where(x_face, 1.0, 0.8))[None, :, None]
        for mask, color in ((frame & in_front, WALL), (boarded & in_front, PLANK)):
            fill = np.broadcast_to(color * shade, sub_img.shape)
            sub_img[mask] = fill[mask]
            sub_z[mask] = np.broadcast_to(perp, mask.shape)[mask]
        img[:, cols], zbuf[:, cols] = sub_img, sub_z

    # ------------------------------------------------------------ player overlay
    @staticmethod
    def _rect(img, x0: float, y0: float, x1: float, y1: float, color) -> None:
        h, w = img.shape[:2]
        x0, x1 = max(0, round(x0)), min(w, round(x1))
        y0, y1 = max(0, round(y0)), min(h, round(y1))
        if x0 < x1 and y0 < y1:
            img[y0:y1, x0:x1] = color

    def _viewmodel(self, img, sim) -> None:
        W, H = self.w, self.h
        u = H / 72.0
        applied = sim._applied
        moving = applied[spec.FORWARD] != 1 or applied[spec.STRAFE] != 1  # index 1 is "no movement"
        bob = math.sin(sim.t * (12.0 if sim.sprinting else 8.0)) * 1.5 * u if moving else 0.0
        drop = 20 * u if (sim.reload_t > 0.0 or sim.swap_t > 0.0) else 0.0
        cx = W * (0.5 if sim.ads else 0.66) + 0.6 * bob
        y = H - (4 * u if sim.ads else 0.0) + drop + abs(bob)
        if sim.weapons[sim.slot].w.name == "m1911":
            self._rect(img, cx - 5 * u, y - 17 * u, cx + 5 * u, y - 11 * u, GUN_EDGE)
            self._rect(img, cx - 4 * u, y - 16 * u, cx + 4 * u, y - 11 * u, GUN)
            self._rect(img, cx - 1 * u, y - 11 * u, cx + 5 * u, H, GUN)
            muzzle = y - 17 * u
        else:
            self._rect(img, cx - 6 * u, y - 14 * u, cx + 7 * u, H, GUN)
            self._rect(img, cx - 1.5 * u, y - 30 * u, cx + 1.5 * u, y - 14 * u, GUN_EDGE)
            muzzle = y - 30 * u
        if sim.shots_this_step:
            self._rect(img, cx - 4 * u, muzzle - 7 * u, cx + 4 * u, muzzle, FLASH)
            self._rect(img, cx - 2 * u, muzzle - 10 * u, cx + 2 * u, muzzle - 7 * u, FLASH)
        if sim.melee_cd > sim.p.melee_cooldown_s - 0.3:  # knifed within the last 0.3 s
            steps = max(1, round(H * 0.5))
            for k in range(steps):
                x = W * 0.8 + (W * 0.42 - W * 0.8) * k / steps
                self._rect(img, x - 1.5 * u, H - k - 1, x + 1.5 * u, H - k, BLADE)
        if not (sim.ads or sim.sprinting):
            s, gap = self.s, 3 * self.s
            mx, my = W // 2, H // 2
            for x0, y0, x1, y1 in (
                (mx - gap - 2 * s, my, mx - gap, my + s),
                (mx + gap + s, my, mx + gap + 3 * s, my + s),
                (mx, my - gap - 2 * s, mx + s, my - gap),
                (mx, my + gap + s, mx + s, my + gap + 3 * s),
            ):
                self._rect(img, x0, y0, x1, y1, CROSSHAIR)

    def _hud(self, img, sim) -> None:
        W, H, s = self.w, self.h, self.s
        pad = 3 * s
        bottom = H - pad
        lull = sim.intermission_t > 0.0 and int(sim.intermission_t * 2.0) % 2 == 0
        color = ROUND_LULL if lull else ROUND_RED
        if sim.round <= 5:
            # Nacht counts rounds in red chalk tallies up to five, then switches to numerals.
            mark_h, gap = 8 * s, 3 * s
            for m in range(min(sim.round, 4)):
                self._rect(img, pad + m * gap, bottom - mark_h, pad + m * gap + s, bottom, color)
            if sim.round == 5:
                span = 3 * gap + 3 * s
                for k in range(span):
                    yy = bottom - 1 - k * (mark_h - 1) / span
                    self._rect(img, pad - s + k, yy - s + 1, pad + k, yy + 1, color)
        else:
            draw_text(img, str(sim.round), pad, bottom - GLYPH_H * 2 * s, 2 * s, color)

        held = sim.weapons[sim.slot]
        ammo = f"{held.mag}/{held.reserve}"
        ammo_x = W - pad - text_width(ammo, s)
        draw_text(img, ammo, ammo_x, bottom - GLYPH_H * s, s, AMMO)
        points = str(sim.points)
        draw_text(img, points, W - pad - text_width(points, s), bottom - 2 * GLYPH_H * s - 2 * s, s, POINTS)
        for g in range(sim.grenades):
            gx = ammo_x - (g + 1) * 3 * s - s
            self._rect(img, gx, bottom - 2 * s, gx + 2 * s, bottom, GRENADE_ICON)
        kind, price, _ = sim.prompt
        if kind:  # PROMPT_NONE is 0
            text = f"HOLD F [{price}]" if price else "HOLD F"
            draw_text(img, text, (W - text_width(text, s)) // 2, round(H * 0.7), s, PROMPT)
