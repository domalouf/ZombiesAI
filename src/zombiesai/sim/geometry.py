"""Nacht floor plan rasterized to a nav/sight grid, with cached door-dependent distance fields."""

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from zombiesai import spec
from zombiesai.sim import mechanics

DEFAULT_GEOMETRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "env" / "nacht_geometry.yaml"

WALL, FLOOR, DOOR, WINDOW, APPROACH = range(5)
_NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class Window:
    id: str
    zone: int
    pos: np.ndarray
    normal: np.ndarray
    inside: np.ndarray  # where a zombie lands after climbing
    outside: np.ndarray  # where a zombie stands to tear planks
    spawn: np.ndarray


@dataclass(frozen=True)
class Door:
    id: str
    kind: str
    unlocks: int
    rect: tuple[float, float, float, float]
    price: int


@dataclass(frozen=True)
class WallWeapon:
    weapon: str
    pos: np.ndarray
    price: int


class Geometry:
    def __init__(self, path: str | Path = DEFAULT_GEOMETRY_PATH):
        cfg = yaml.safe_load(Path(path).read_text())
        self.cell_size = cs = float(cfg["cell_size"])
        self.bounds = x0, y0, x1, y1 = tuple(float(v) for v in cfg["bounds"])
        self.nx, self.ny = round((x1 - x0) / cs), round((y1 - y0) / cs)
        self._inv_cs = 1.0 / cs
        self.zones = tuple(cfg["zones"])
        if self.zones != spec.ZONES:
            raise ValueError(f"geometry zones {self.zones} must match spec.ZONES {spec.ZONES}")

        cx = x0 + (np.arange(self.nx) + 0.5) * cs
        cy = y0 + (np.arange(self.ny) + 0.5) * cs
        self._cx, self._cy = np.meshgrid(cx, cy)
        kind = np.full((self.ny, self.nx), WALL, dtype=np.int8)
        zone = np.full((self.ny, self.nx), -1, dtype=np.int8)
        door_of = np.full((self.ny, self.nx), -1, dtype=np.int8)

        for room in cfg["rooms"]:
            m = self._shape_mask(room)
            kind[m] = FLOOR
            zone[m] = self.zones.index(room["zone"])

        doors = []
        for i, d in enumerate(cfg["doors"]):
            m = self._shape_mask(d)
            kind[m] = DOOR
            door_of[m] = i
            zone[m] = self.zones.index(d["unlocks"])
            doors.append(
                Door(d["id"], d["kind"], self.zones.index(d["unlocks"]), tuple(d["rect"]), mechanics.DOOR_PRICE)
            )
        self.doors = tuple(doors)
        if tuple(d.id for d in self.doors) != spec.DOORS:
            raise ValueError(f"geometry doors must be {spec.DOORS} in that order")
        self.door_rects = np.array([d.rect for d in self.doors], dtype=np.float64)
        self.all_doors_mask = (1 << len(self.doors)) - 1

        half_w = float(cfg["window_width"]) / 2
        approach_len = float(cfg["approach_length"])
        windows = []
        for w in cfg["windows"]:
            pos = np.array(w["pos"], dtype=np.float64)
            normal = np.array(w["normal"], dtype=np.float64)
            if abs(normal).sum() != 1.0:
                raise ValueError(f"window {w['id']}: normals must be axis-aligned unit vectors")
            tangent = np.array((-normal[1], normal[0]))
            opening = self._rect_mask(pos - tangent * half_w, pos + tangent * half_w + normal * cs)
            approach = self._rect_mask(
                pos + normal * cs - tangent * half_w, pos + normal * approach_len + tangent * half_w
            )
            kind[opening & (kind == WALL)] = WINDOW
            kind[approach & (kind == WALL)] = APPROACH
            windows.append(
                Window(
                    w["id"],
                    self.zones.index(w["zone"]),
                    pos,
                    normal,
                    pos - normal * 0.6,
                    pos + normal * 0.6,
                    pos + normal * (approach_len - 0.5),
                )
            )
        self.windows = tuple(windows)
        self.window_pos = np.array([w.pos for w in self.windows])
        self.window_inside = np.array([w.inside for w in self.windows])
        self.window_outside = np.array([w.outside for w in self.windows])
        self.window_zone = np.array([w.zone for w in self.windows], dtype=np.int64)

        self.wall_weapons = tuple(
            WallWeapon(w["weapon"], np.array(w["pos"], dtype=np.float64), mechanics.WALL_WEAPON_PRICES[w["weapon"]])
            for w in cfg["wall_weapons"]
        )
        self.wall_weapon_pos = np.array([w.pos for w in self.wall_weapons])
        self.box_pos = np.array(cfg["box"]["pos"], dtype=np.float64)
        self.player_spawn = np.array(cfg["player_spawn"]["pos"], dtype=np.float64)

        self.kind = kind.ravel()
        self.zone_of_cell = zone.ravel()
        self.door_of_cell = door_of.ravel()
        self._walkable: dict[int, np.ndarray] = {}
        self._sight_block: dict[int, np.ndarray] = {}
        self._graphs: dict[int, csr_matrix] = {}
        self._all_pairs: dict[int, np.ndarray] = {}
        self._validate()
        self._build_nav_graph()

    # ------------------------------------------------------------ rasterization
    def _rect_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return (self._cx > lo[0]) & (self._cx < hi[0]) & (self._cy > lo[1]) & (self._cy < hi[1])

    def _shape_mask(self, item: dict) -> np.ndarray:
        if "rect" in item:
            x0, y0, x1, y1 = item["rect"]
            return self._rect_mask(np.array((x0, y0), float), np.array((x1, y1), float))
        inside = np.zeros(self._cx.shape, dtype=bool)
        poly = item["poly"]
        for (xa, ya), (xb, yb) in zip(poly, poly[1:] + poly[:1]):
            if ya == yb:
                continue
            crosses = (ya > self._cy) != (yb > self._cy)
            x_cross = xa + (self._cy - ya) * (xb - xa) / (yb - ya)
            inside ^= crosses & (self._cx < x_cross)
        return inside

    def _validate(self) -> None:
        def on(kind: int, xy: np.ndarray, what: str) -> None:
            if self.kind[self.cell(xy[0], xy[1])] != kind:
                raise ValueError(f"{what} at {xy.tolist()} is not on the expected cell kind {kind}")

        for w in self.windows:
            on(FLOOR, w.inside, f"window {w.id} inside anchor")
            on(APPROACH, w.outside, f"window {w.id} outside anchor")
            on(APPROACH, w.spawn, f"window {w.id} spawn")
        for ww in self.wall_weapons:
            on(FLOOR, ww.pos, f"wall weapon {ww.weapon}")
        on(FLOOR, self.box_pos, "box")
        on(FLOOR, self.player_spawn, "player spawn")

    # ------------------------------------------------------------ lookups
    def cell(self, x: float, y: float) -> int:
        ix = min(max(int((x - self.bounds[0]) * self._inv_cs), 0), self.nx - 1)
        iy = min(max(int((y - self.bounds[1]) * self._inv_cs), 0), self.ny - 1)
        return iy * self.nx + ix

    def cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        # Truncation equals floor for in-bounds points; out-of-bounds ones are clamped either way.
        ix = ((xs - self.bounds[0]) * self._inv_cs).astype(np.int64)
        iy = ((ys - self.bounds[1]) * self._inv_cs).astype(np.int64)
        np.minimum(np.maximum(ix, 0, out=ix), self.nx - 1, out=ix)
        np.minimum(np.maximum(iy, 0, out=iy), self.ny - 1, out=iy)
        return iy * self.nx + ix

    def cell_center(self, cell: int) -> tuple[float, float]:
        iy, ix = divmod(cell, self.nx)
        return float(self._cx[iy, ix]), float(self._cy[iy, ix])

    def normalize_xy(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)

    def active_zones(self, open_mask: int) -> np.ndarray:
        active = np.zeros(len(self.zones), dtype=bool)
        active[self.zones.index("start")] = True
        for i, d in enumerate(self.doors):
            if open_mask >> i & 1:
                active[d.unlocks] = True
        return active

    def door_distances(self, x: float, y: float) -> np.ndarray:
        r = self.door_rects
        dx = np.maximum(np.maximum(r[:, 0] - x, 0.0), x - r[:, 2])
        dy = np.maximum(np.maximum(r[:, 1] - y, 0.0), y - r[:, 3])
        return np.hypot(dx, dy)

    def _door_open_lookup(self, open_mask: int) -> np.ndarray:
        """Indexable by door_of_cell; the trailing entry answers for door_of_cell == -1."""
        return np.array([bool(open_mask >> i & 1) for i in range(len(self.doors))] + [False])

    def walkable(self, open_mask: int) -> np.ndarray:
        out = self._walkable.get(open_mask)
        if out is None:
            door_open = self._door_open_lookup(open_mask)
            out = (self.kind == FLOOR) | ((self.kind == DOOR) & door_open[self.door_of_cell])
            out.setflags(write=False)
            self._walkable[open_mask] = out
        return out

    def sight_block(self, open_mask: int) -> np.ndarray:
        out = self._sight_block.get(open_mask)
        if out is None:
            door_open = self._door_open_lookup(open_mask)
            out = (self.kind == WALL) | ((self.kind == DOOR) & ~door_open[self.door_of_cell])
            out.setflags(write=False)
            self._sight_block[open_mask] = out
        return out

    def line_of_sight(self, open_mask: int, ox: float, oy: float, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """Per target, whether the segment from (ox, oy) crosses no sight-blocking cell (sampled every cs/2)."""
        dx, dy = tx - ox, ty - oy
        n = max(3, int(math.ceil(float(np.hypot(dx, dy).max(initial=0.0)) * 2 * self._inv_cs)) + 2)
        t = np.linspace(0.0, 1.0, n)[1:-1]
        cells = self.cells(ox + dx[:, None] * t, oy + dy[:, None] * t)
        return ~self.sight_block(open_mask)[cells].any(axis=1)

    def free_distance(self, open_mask: int, ox: float, oy: float, ux: float, uy: float, max_d: float) -> float:
        """How far a thrown object travels along unit direction (ux, uy) before hitting a sight-blocking cell."""
        step = self.cell_size / 2
        d = np.arange(step, max_d + 1e-9, step)
        blocked = self.sight_block(open_mask)[self.cells(ox + ux * d, oy + uy * d)]
        hit = np.flatnonzero(blocked)
        return float(d[hit[0]] - step) if hit.size else max_d

    # ------------------------------------------------------------ navigation
    def _build_nav_graph(self) -> None:
        node_mask = (self.kind == FLOOR) | (self.kind == DOOR)
        self.n_nodes = int(node_mask.sum())
        self.node_of_cell = np.full(self.kind.size, -1, dtype=np.int64)
        self.node_of_cell[node_mask] = np.arange(self.n_nodes)
        self.cell_of_node = np.flatnonzero(node_mask)
        self.node_xy = np.stack(
            (self._cx.ravel()[self.cell_of_node], self._cy.ravel()[self.cell_of_node]), axis=1
        )
        door_bit = np.where(self.door_of_cell >= 0, 1 << np.maximum(self.door_of_cell, 0).astype(np.int64), 0)

        iy, ix = np.divmod(self.cell_of_node, self.nx)
        self.node_nbrs = np.full((self.n_nodes, len(_NEIGHBOR_OFFSETS)), -1, dtype=np.int64)
        src, dst, weight, bits = [], [], [], []
        for k, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
            jcell = np.where(ok, jy * self.nx + jx, 0)
            jnode = np.where(ok, self.node_of_cell[jcell], -1)
            ok &= jnode >= 0
            edge_bits = door_bit[self.cell_of_node] | door_bit[jcell]
            if dx and dy:
                side_a = np.where(ok, iy * self.nx + np.clip(jx, 0, self.nx - 1), 0)
                side_b = np.where(ok, np.clip(jy, 0, self.ny - 1) * self.nx + ix, 0)
                ok &= (self.node_of_cell[side_a] >= 0) & (self.node_of_cell[side_b] >= 0)
                edge_bits = edge_bits | door_bit[side_a] | door_bit[side_b]
            self.node_nbrs[ok, k] = jnode[ok]
            src.append(np.flatnonzero(ok))
            dst.append(jnode[ok])
            weight.append(np.full(int(ok.sum()), self.cell_size * (math.sqrt(2.0) if dx and dy else 1.0)))
            bits.append(edge_bits[ok])
        self._edge_src = np.concatenate(src)
        self._edge_dst = np.concatenate(dst)
        self._edge_w = np.concatenate(weight)
        self._edge_bits = np.concatenate(bits)

    def _graph(self, open_mask: int) -> csr_matrix:
        g = self._graphs.get(open_mask)
        if g is None:
            closed = self.all_doors_mask & ~open_mask
            keep = (self._edge_bits & closed) == 0
            g = csr_matrix(
                (self._edge_w[keep], (self._edge_src[keep], self._edge_dst[keep])),
                shape=(self.n_nodes, self.n_nodes),
            )
            self._graphs[open_mask] = g
        return g

    def distance_field(self, open_mask: int, node: int) -> np.ndarray:
        """Path distance in meters from every nav node to `node` (inf where unreachable)."""
        # All-pairs per door mask: ~0.1 s and ~8 MB once, then every lookup is free.
        table = self._all_pairs.get(open_mask)
        if table is None:
            table = dijkstra(self._graph(open_mask), directed=True).astype(np.float32)
            table.setflags(write=False)
            self._all_pairs[open_mask] = table
        return table[node]


@functools.cache
def load_geometry(path: str | None = None) -> Geometry:
    return Geometry(path or DEFAULT_GEOMETRY_PATH)
