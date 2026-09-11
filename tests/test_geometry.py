import numpy as np
import pytest

from zombiesai import spec
from zombiesai.sim.geometry import FLOOR, load_geometry


@pytest.fixture(scope="module")
def geo():
    return load_geometry()


def mask(*door_ids: str) -> int:
    return sum(1 << spec.DOORS.index(d) for d in door_ids)


def reachable_zones(geo, open_mask: int) -> set[str]:
    src = geo.node_of_cell[geo.cell(*geo.player_spawn)]
    d = geo.distance_field(open_mask, src)
    return {geo.zones[z] for z in set(geo.zone_of_cell[geo.cell_of_node[np.isfinite(d)]].tolist())}


def test_door_topology(geo):
    assert reachable_zones(geo, 0) == {"start"}
    assert reachable_zones(geo, mask("help_door")) == {"start", "help"}
    assert reachable_zones(geo, mask("start_debris")) == {"start", "upstairs"}
    # The upstairs door is reached through the help room, so alone it opens nothing.
    assert reachable_zones(geo, mask("upstairs_door")) == {"start"}
    assert reachable_zones(geo, geo.all_doors_mask) == {"start", "help", "upstairs"}


def test_active_zones(geo):
    assert geo.active_zones(0).tolist() == [True, False, False]
    assert geo.active_zones(mask("upstairs_door")).tolist() == [True, False, True]


def test_every_window_is_reachable_with_all_doors_open(geo):
    src = geo.node_of_cell[geo.cell(*geo.player_spawn)]
    d = geo.distance_field(geo.all_doors_mask, src)
    for w in geo.windows:
        node = geo.node_of_cell[geo.cell(*w.inside)]
        assert node >= 0 and np.isfinite(d[node]), w.id


def test_distance_field_is_a_metric(geo):
    a = geo.node_of_cell[geo.cell(*geo.player_spawn)]
    b = geo.node_of_cell[geo.cell(*geo.wall_weapons[0].pos)]
    da = geo.distance_field(0, a)
    assert da[a] == 0.0
    assert da[b] == pytest.approx(geo.distance_field(0, b)[a])
    straight = np.hypot(*(geo.node_xy[a] - geo.node_xy[b]))
    assert straight - 1e-6 <= da[b] <= straight * 1.1


def test_line_of_sight(geo):
    los = lambda m, a, b: bool(geo.line_of_sight(m, a[0], a[1], np.array([b[0]]), np.array([b[1]]))[0])
    assert los(0, (2, 2), (12, 8))
    assert not los(geo.all_doors_mask, (5, 9), (5, 15))  # the solid gap between the unfolded floors
    assert not los(0, (12, 4), (17, 4))  # through the closed help door
    assert los(mask("help_door"), (12, 4), (17, 4))
    assert los(0, (4, 1), (4, -3))  # out through a window into its approach lane


def test_closed_doors_block_walking(geo):
    door_cell = geo.cell(14.5, 4.0)
    assert not geo.walkable(0)[door_cell]
    assert geo.walkable(mask("help_door"))[door_cell]
    assert geo.kind[geo.cell(7, 5)] == FLOOR
