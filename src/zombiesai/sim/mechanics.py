"""Ground-truth Nacht der Untoten mechanics, from raw/maps/_zombiemode_prototype.gsc in the WaW mod tools."""

# Only numbers read from the script belong here; guesses live in sim/params.py and get domain-randomized.

import functools

STARTING_POINTS = 500

ZOMBIE_HEALTH_START = 150
ZOMBIE_HEALTH_INCREASE = 100
ZOMBIE_HEALTH_INCREASE_PERCENT = 10
ZOMBIE_HEALTH_COMPOUNDING_FROM_ROUND = 10

SOLO_ZOMBIES_ROUND_1 = 4
SOLO_ZOMBIES_PER_ROUND = 5
SOLO_ZOMBIE_CAP = 24

ZOMBIE_MOVE_SPEED_PER_ROUND = 8
SPAWN_DELAY_S = 3.0
MAX_ENEMY_COUNT = 31

DOOR_PRICE = 1000
BOX_PRICE = 950
WALL_WEAPON_PRICES = {"kar98k": 200, "m1a1_carbine": 600, "double_barrel": 1000, "thompson": 1200}

# zombie_score_damage is 5 in the source; community lore says 10. The sim randomizes between them.
SCORE_HIT_SOURCE = 5
SCORE_HIT_COMMUNITY = 10
SCORE_KILL = 50
SCORE_TORSO_BONUS = 10
SCORE_HEAD_BONUS = 50
SCORE_MELEE_BONUS = 80
SCORE_BARRIER_PLANK = 10
DOWNED_PENALTY_FRACTION = 0.05
NO_REVIVE_PENALTY_FRACTION = 0.10


@functools.cache
def zombie_health(round_number: int) -> int:
    """GSC ai_calculate_health: +100/round through R9, then += Int(health * 10 / 100) from R10."""
    if round_number < 1:
        raise ValueError(f"round_number must be >= 1, got {round_number}")
    health = ZOMBIE_HEALTH_START
    for r in range(2, round_number + 1):
        if r >= ZOMBIE_HEALTH_COMPOUNDING_FROM_ROUND:
            health += int(health * ZOMBIE_HEALTH_INCREASE_PERCENT / 100)
        else:
            health += ZOMBIE_HEALTH_INCREASE
    return health


def zombies_in_round_solo(round_number: int) -> int:
    if round_number < 1:
        raise ValueError(f"round_number must be >= 1, got {round_number}")
    return min(SOLO_ZOMBIE_CAP, SOLO_ZOMBIES_ROUND_1 + SOLO_ZOMBIES_PER_ROUND * (round_number - 1))


def zombie_move_speed(round_number: int) -> int:
    return round_number * ZOMBIE_MOVE_SPEED_PER_ROUND
