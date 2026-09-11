"""Record a NachtSim episode frame by frame and write it as a self-contained HTML replay viewer."""

import json
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.reward import REWARD_TERMS
from zombiesai.sim.geometry import FLOOR, WALL, Geometry
from zombiesai.sim.nacht_sim import (
    PROMPT_BOX,
    PROMPT_DOOR,
    PROMPT_NONE,
    PROMPT_REPAIR,
    PROMPT_WEAPON,
    NachtSim,
)
from zombiesai.sim.params import WEAPONS

TEMPLATE = Path(__file__).with_name("replay_template.html")
WEAPON_NAMES = {
    "m1911": "M1911",
    "kar98k": "Kar98k",
    "m1a1_carbine": "M1A1 Carbine",
    "double_barrel": "Double-Barreled Shotgun",
    "thompson": "Thompson",
    "mp40": "MP40",
    "stg44": "STG-44",
    "bar": "BAR",
    "trench_gun": "Trench Gun",
    "m1_garand": "M1 Garand",
    "gewehr43": "Gewehr 43",
    "ptrs41": "PTRS-41",
    "ray_gun": "Ray Gun",
}
MAP_NAMES = {"double_barrel": "Double-Barrel", "m1a1_carbine": "M1A1"}
DOOR_NAMES = {"help_door": "the help room door", "start_debris": "the stairway debris", "upstairs_door": "the upstairs door"}
ZONE_LABELS = {"start": "Start room", "help": "Help room", "upstairs": "Upstairs"}
FLAG_ADS, FLAG_SPRINT, FLAG_INTERMISSION, FLAG_RELOADING, FLAG_SWAPPING = (1, 2, 4, 8, 16)
KILL_BODY, KILL_HEAD, KILL_KNIFE = range(3)
_WEAPON_INDEX = {name: i for i, name in enumerate(WEAPONS)}
_H = spec.HUD_INDEX


def _r(v: float, nd: int = 2) -> float:
    return round(float(v), nd)


def _map_payload(geo: Geometry) -> dict:
    def wall_normal(pos: np.ndarray) -> list[float]:
        for nx, ny in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if geo.kind[geo.cell(pos[0] - nx * 0.6, pos[1] - ny * 0.6)] == WALL:
                return [nx, ny]
        return [0, 1]

    labels = []
    for z, name in enumerate(geo.zones):
        cells = np.flatnonzero((geo.zone_of_cell == z) & (geo.kind == FLOOR))
        xy = np.array([geo.cell_center(int(c)) for c in cells])
        # Interquartile midpoint, so a zone's stair corridor doesn't drag its label out of the room.
        lo = np.percentile(xy, 25, axis=0)
        hi = np.percentile(xy, 75, axis=0)
        labels.append({"text": ZONE_LABELS[name], "x": _r((lo[0] + hi[0]) / 2), "y": _r((lo[1] + hi[1]) / 2)})
    return {
        "bounds": list(geo.bounds),
        "cell": geo.cell_size,
        "nx": geo.nx,
        "ny": geo.ny,
        "kind": "".join(str(int(k)) for k in geo.kind),
        "labels": labels,
        "doors": [{"id": d.id, "kind": d.kind, "rect": list(d.rect), "price": d.price} for d in geo.doors],
        "windows": [
            {"pos": w.pos.tolist(), "normal": w.normal.tolist(), "outside": w.outside.tolist(), "spawn": w.spawn.tolist()}
            for w in geo.windows
        ],
        "weapons": [
            {
                "name": MAP_NAMES.get(w.weapon, WEAPON_NAMES[w.weapon]),
                "price": w.price,
                "pos": w.pos.tolist(),
                "normal": wall_normal(w.pos),
            }
            for w in geo.wall_weapons
        ],
        "box": geo.box_pos.tolist(),
    }


class _Recorder:
    COLUMNS = (
        "t", "x", "y", "yaw", "pitch", "hp", "flash", "weapon", "mag", "reserve", "points", "round", "grenades",
        "flags", "act", "shots", "hits", "kills", "z", "planks", "doors", "prompt", "gren", "r", "ret", "obs", "terms",
    )

    def __init__(self, env: NachtSim):
        self.env = env
        self.f: dict[str, list] = {k: [] for k in self.COLUMNS}
        self.events: list[dict] = []
        self.prompts: list[str] = []
        self._prompt_index: dict[str, int] = {}
        self.ret = 0.0
        self.totals = {"shots": 0, "hits": 0, "kills": 0, "headshots": 0, "knife": 0, "points_earned": 0}

    def _prompt(self) -> int:
        env = self.env
        kind, price, target = env.prompt
        if kind == PROMPT_NONE:
            return -1
        if kind == PROMPT_REPAIR:
            text = "to rebuild the barrier"
        elif kind == PROMPT_DOOR:
            door = env.geo.doors[target]
            verb = "clear" if door.kind == "debris" else "open"
            text = f"to {verb} {DOOR_NAMES[door.id]} [{price}]"
        elif kind == PROMPT_WEAPON:
            name = env.geo.wall_weapons[target].weapon
            ammo = " ammo" if env._owned(name) else ""
            text = f"to buy {WEAPON_NAMES[name]}{ammo} [{price}]"
        else:
            assert kind == PROMPT_BOX
            text = f"for the Mystery Box [{price}]"
        if text not in self._prompt_index:
            self._prompt_index[text] = len(self.prompts)
            self.prompts.append(text)
        return self._prompt_index[text]

    def event(self, kind: str, text: str) -> None:
        self.events.append({"i": len(self.f["t"]) - 1, "t": _r(self.env.t, 1), "kind": kind, "text": text})

    def frame(self, obs, reward=0.0, terms=None, shots=0, hits=(), kills=()) -> None:
        env, f = self.env, self.f
        hw = env.weapons[env.slot]
        idx = np.flatnonzero(env.z_alive)
        zs = []
        for i in idx:
            zs += [_r(env.z_x[i]), _r(env.z_y[i]), int(env.z_phase[i]), _r(min(1.0, env.z_hp[i] / env.zombie_max_hp))]
        seen = spec.decode_hud(obs["hud"])
        self.ret += reward
        flags = (
            FLAG_ADS * env.ads
            | FLAG_SPRINT * env.sprinting
            | FLAG_INTERMISSION * (env.intermission_t > 0.0)
            | FLAG_RELOADING * (env.reload_t > 0.0)
            | FLAG_SWAPPING * (env.swap_t > 0.0)
        )
        row = {
            "t": _r(env.t, 3),
            "x": _r(env.px),
            "y": _r(env.py),
            "yaw": _r(env.yaw, 3),
            "pitch": _r(env.pitch, 1),
            "hp": _r(max(env.hp, 0.0), 1),
            "flash": _r(env.flash_obs),
            "weapon": _WEAPON_INDEX[hw.w.name],
            "mag": hw.mag,
            "reserve": hw.reserve,
            "points": env.points,
            "round": env.round,
            "grenades": env.grenades,
            "flags": int(flags),
            "act": list(env._applied),
            "shots": shots,
            "hits": [list(h) for h in hits],
            "kills": [list(k) for k in kills],
            "z": zs,
            "planks": env.planks.tolist(),
            "doors": env.open_mask,
            "prompt": self._prompt(),
            "gren": [v for g in env.live_grenades for v in (_r(g[0]), _r(g[1]), _r(g[2]))],
            "r": _r(reward, 3),
            "ret": _r(self.ret, 2),
            "obs": [
                int(round(seen[_H["round"]])),
                int(round(seen[_H["points"]])),
                int(round(seen[_H["mag_ammo"]])),
                int(round(seen[_H["reserve_ammo"]])),
                _r(seen[_H["hud_confidence"]]),
            ],
            "terms": [[k, _r(v, 3)] for k, v in enumerate(terms if terms is not None else ()) if abs(v) > 1e-9],
        }
        for key, value in row.items():
            f[key].append(value)


def record_replay(env: NachtSim, agent, *, seed: int, agent_name: str, options: dict | None = None) -> dict:
    """Play one episode, capturing the full world state after every step (not just what the agent saw)."""
    obs, _ = env.reset(seed=seed, options=options)
    agent.reset()
    rec = _Recorder(env)
    rec.frame(obs)
    rec.event("round_start", "Round 1 begins")
    info: dict = {}
    terminated = truncated = False
    while not (terminated or truncated):
        hp_before, alive_before = env.z_hp.copy(), env.z_alive.copy()
        rs_before = env.rs
        counts = (rs_before.shots, rs_before.hits, rs_before.kills, rs_before.headshot_kills, rs_before.melee_kills)
        player_hp, points = env.hp, env.points

        action = np.asarray(agent.act(obs))
        obs, reward, terminated, truncated, info = env.step(action)

        # A round can roll over mid-step, so count deltas on the old stats object plus the new one.
        new = env.rs if env.rs is not rs_before else None
        delta = [
            getattr(rs_before, k) - c + (getattr(new, k) if new else 0)
            for k, c in zip(("shots", "hits", "kills", "headshot_kills", "melee_kills"), counts)
        ]
        shots, hits_n, kills_n, heads_n, knife_n = delta
        damaged = alive_before & (env.z_hp < hp_before)
        killed = np.flatnonzero(alive_before & ~env.z_alive)
        kinds = [KILL_HEAD] * heads_n + [KILL_KNIFE] * knife_n
        kills = [
            (_r(env.z_x[i]), _r(env.z_y[i]), kinds[j] if j < len(kinds) else KILL_BODY) for j, i in enumerate(killed)
        ]
        hits = [(_r(env.z_x[i]), _r(env.z_y[i])) for i in np.flatnonzero(damaged)]
        rec.frame(obs, reward, info["reward_terms"], shots, hits, kills)

        t = rec.totals
        t["shots"] += shots
        t["hits"] += hits_n + kills_n
        t["kills"] += kills_n
        t["headshots"] += heads_n
        t["knife"] += knife_n
        t["points_earned"] += max(0, env.points - points)
        for kind in kinds[: len(killed)] + [KILL_BODY] * max(0, len(killed) - len(kinds)):
            label = {KILL_HEAD: "Headshot kill", KILL_KNIFE: "Knife kill", KILL_BODY: "Kill"}[kind]
            rec.event("kill", label)
        if env.hp < player_hp and env.hp > 0:
            rec.event("hit", f"Hit by a zombie — health {env.hp:.0f} of {env.p.player_max_hp:.0f}")
        for e in info["events"]:
            if e["type"] == "purchase":
                item = e["item"]
                if item in DOOR_NAMES:
                    text = f"Opened {DOOR_NAMES[item]}"
                elif item.startswith("box:"):
                    text = f"Mystery Box: {WEAPON_NAMES[item[4:]]}"
                elif item.endswith("_ammo"):
                    text = f"Bought {WEAPON_NAMES[item[:-5]]} ammo"
                else:
                    text = f"Bought the {WEAPON_NAMES[item]}"
                rec.event("purchase", f"{text} (−{e['price']})")
            elif e["type"] == "round_complete":
                rec.event(
                    "round",
                    f"Round {e['round']} cleared: {e['kills']} kills, {e['headshot_kills']} headshots, "
                    f"{e['duration_s']:.0f} s",
                )
            elif e["type"] == "round_start":
                rec.event("round_start", f"Round {e['round']} begins")
            elif e["type"] == "game_over":
                rec.event("death", f"Downed in round {e['round']}: game over")
    if truncated:
        rec.event("end", f"Step cap reached in round {env.round}")

    t = rec.totals
    summary = {
        "agent": agent_name,
        "seed": seed,
        "hardness": env.config.hardness,
        "latency_steps": env.timing.latency_steps,
        "frames_per_step": env.timing.frames_per_step,
        "round_reached": env.round,
        "rounds_survived": info.get("rounds_survived"),
        "ended": "game_over" if terminated else "step_cap",
        "game_time_s": _r(env.t, 1),
        "steps": env.steps,
        "max_hp": env.p.player_max_hp,
        "return": _r(rec.ret, 2),
        "accuracy": _r(t["hits"] / t["shots"], 3) if t["shots"] else None,
        **t,
    }
    return {
        "spec_version": spec.SPEC_VERSION,
        "summary": summary,
        "map": _map_payload(env.geo),
        "weapons": [WEAPON_NAMES[n] for n in WEAPONS],
        "prompts": rec.prompts,
        "reward_terms": list(REWARD_TERMS),
        "action_spec": {
            "yaw_bins_deg": list(spec.YAW_BINS_DEG),
            "pitch_bins_deg": list(spec.PITCH_BINS_DEG),
            "buttons": list(spec.BUTTONS),
        },
        "events": rec.events,
        "frames": rec.f,
    }


def write_replay_html(replay: dict, path: str | Path, standalone: bool = True) -> Path:
    """standalone wraps the page in a full HTML document; without it, the body fragment an Artifact expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(replay, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__REPLAY_DATA__*/null", data, 1)
    if standalone:
        html = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
            f"{html}\n</body>\n</html>\n"
        )
    path.write_text(html)
    return path
