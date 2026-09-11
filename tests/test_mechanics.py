import pytest

from zombiesai.sim import mechanics


def test_solo_zombie_counts_match_prototype_script():
    assert [mechanics.zombies_in_round_solo(r) for r in range(1, 8)] == [4, 9, 14, 19, 24, 24, 24]
    assert mechanics.zombies_in_round_solo(60) == 24


def test_zombie_health_curve():
    assert mechanics.zombie_health(1) == 150
    assert mechanics.zombie_health(2) == 250
    assert mechanics.zombie_health(9) == 950
    assert mechanics.zombie_health(10) == 1045  # 950 + Int(950 * 10 / 100)
    assert mechanics.zombie_health(11) == 1149  # 1045 + Int(104.5)
    healths = [mechanics.zombie_health(r) for r in range(1, 40)]
    assert all(b > a for a, b in zip(healths, healths[1:]))


def test_move_speed():
    assert mechanics.zombie_move_speed(1) == 8
    assert mechanics.zombie_move_speed(10) == 80


def test_prices():
    assert mechanics.STARTING_POINTS == 500
    assert mechanics.DOOR_PRICE == 1000
    assert mechanics.BOX_PRICE == 950
    assert mechanics.WALL_WEAPON_PRICES == {"kar98k": 200, "m1a1_carbine": 600, "double_barrel": 1000, "thompson": 1200}


@pytest.mark.parametrize("fn", [mechanics.zombie_health, mechanics.zombies_in_round_solo])
def test_round_zero_rejected(fn):
    with pytest.raises(ValueError):
        fn(0)
