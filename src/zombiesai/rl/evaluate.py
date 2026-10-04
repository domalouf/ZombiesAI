"""The agent at hour N, played for the record: the progress curve in rounds, and the human it is measured against.

Training's own numbers are rolling averages over whatever the policy was while they were gathered, by a policy
that changed in the middle of them. For the video's chart each kept snapshot (`snapshots/h<hours>.pt`, written
by the learner every `snapshot_every_h` hours of training, rl/parallel_ppo.py) is played on its own, frozen,
for a set number of games, and scored by the game's round counter -- the number that cannot be reward-hacked.

The games are played by the training's own actors (rl/actors.py) on the fleet's instances, so the agent plays
exactly as it did in training: the same hands, the same 15 Hz, actions sampled the same way. They only follow a
weights.pt that never changes. Each snapshot's games and best film go in `<video dir>/evals/<snapshot>/`, and a
line per snapshot in `evals.jsonl` there (and `evals.csv`, for a chart in an editor).

`human_baseline` is the same numbers for the owner's own play: every game that started and ended inside a
recorded demo, as `scripts/parse_hud.py` summarised it (`hud_summary.json` beside each clip).
"""

import csv
import json
import multiprocessing as mp
import os
import queue as queue_mod
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from zombiesai.rl.config import RLConfig

SNAPSHOT = re.compile(r"h(\d+\.\d+)\.pt$")


def snapshot_hours(path: Path) -> float | None:
    match = SNAPSHOT.search(Path(path).name)
    return float(match[1]) if match else None


def find_snapshots(roots: list[Path]) -> list[Path]:
    """Every snapshot under `roots` (run directories, or directories of runs), earliest in training first."""
    found = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.add(root)
            continue
        for pattern in ("snapshots/h*.pt", "*/snapshots/h*.pt"):
            found |= {p for p in root.glob(pattern) if snapshot_hours(p) is not None}
    return sorted(found, key=lambda p: (snapshot_hours(p) or 0.0, str(p)))


def pick(snapshots: list[Path], every_h: float) -> list[Path]:
    """One snapshot per `every_h` hours of training (the earliest in each), and always the latest."""
    if every_h <= 0 or not snapshots:
        return list(snapshots)
    chosen, seen = [], set()
    for path in snapshots:
        bucket = int((snapshot_hours(path) or 0.0) / every_h)
        if bucket not in seen:
            seen.add(bucket)
            chosen.append(path)
    if snapshots[-1] not in chosen:
        chosen.append(snapshots[-1])
    return chosen


def score(games: list[dict]) -> dict:
    """What a set of games comes to: rounds (mean, median, best), kills and survival time."""
    if not games:
        return {"games": 0}
    rounds = np.array([g.get("round_reached") or 0 for g in games], float)
    kills = [g["end_kills"] if g.get("end_kills") is not None else g.get("kills") for g in games]
    kills = np.array([k for k in kills if isinstance(k, (int, float))], float)
    seconds = np.array([g["seconds"] for g in games if isinstance(g.get("seconds"), (int, float))], float)
    return {"games": len(games), "round_mean": round(float(rounds.mean()), 2),
            "round_median": float(np.median(rounds)), "round_best": int(rounds.max()),
            "kills_mean": round(float(kills.mean()), 1) if len(kills) else None,
            "seconds_mean": round(float(seconds.mean()), 1) if len(seconds) else None}


def human_baseline(roots: list[Path]) -> dict:
    """The owner's own games, from the demos' HUD summaries: only games seen from their start to their end."""
    games = []
    for root in roots:
        for path in Path(root).rglob("hud_summary.json"):
            try:
                summary = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for g in summary.get("games") or []:
                if g.get("started_in_clip") and g.get("game_over") and (g.get("rounds_reached") or 0) > 0:
                    games.append({"round_reached": g["rounds_reached"], "seconds": g.get("seconds")})
    return score(games)


def evaluate_snapshot(snapshot: Path, games: int, out_dir: Path, base: RLConfig, *, say=print,
                      timeout_s: float = 6 * 3600) -> dict:
    """Play `snapshot`, frozen, on `base.n_actors` of the fleet's games until `games` games have finished.
    Returns its line for evals.jsonl."""
    from zombiesai.rl.actors import actor_main

    snapshot, out_dir = Path(snapshot), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(snapshot, map_location="cpu", weights_only=False)
    clock = (blob.get("rl") or {}).get("clock") or {}
    torch.save({"version": 0, "model": blob["model"], "clock": clock}, out_dir / "weights.pt")
    config = replace(base, init=str(snapshot), record_best=True, record_moments=False, film_every_h=0,
                     record_records=False, record_every=0)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ctx = mp.get_context("spawn")
    out, stop = ctx.Queue(maxsize=max(8, 4 * config.n_actors)), ctx.Event()
    procs = [ctx.Process(target=actor_main, args=(i, config, str(out_dir), out, stop), daemon=True,
                         name=f"eval-actor-{i}") for i in range(config.n_actors)]
    for p in procs:
        p.start()
    finished: list[dict] = []
    deadline = time.time() + timeout_s
    try:
        with open(out_dir / "episodes.jsonl", "a", buffering=1) as log:
            while len(finished) < games and time.time() < deadline and any(p.is_alive() for p in procs):
                try:
                    kind, i, payload = out.get(timeout=1.0)
                except queue_mod.Empty:
                    continue
                if kind == "episode":
                    finished.append(payload)
                    log.write(json.dumps(payload, default=str) + "\n")
                    say(f"  {snapshot.name}: game {len(finished)}/{games}, round {payload.get('round_reached')}")
                elif kind == "error":
                    say(f"  actor {i} failed:\n{payload}")
    finally:
        stop.set()
        end = time.time() + 60
        while any(p.is_alive() for p in procs) and time.time() < end:
            try:
                out.get(timeout=0.2)
            except queue_mod.Empty:
                pass
        for p in procs:
            if p.is_alive():
                p.terminate()
    return {"snapshot": str(snapshot), "hours": snapshot_hours(snapshot), "clock": clock, **score(finished),
            "evaluated_unix": round(time.time())}


def write_results(directory: Path, rows: list[dict], human: dict | None) -> None:
    """evals.jsonl (appended, one line per snapshot) and evals.csv (the whole table, rewritten)."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "evals.jsonl", "a") as log:
        for row in rows:
            log.write(json.dumps(row, default=str) + "\n")
    every = [json.loads(line) for line in (directory / "evals.jsonl").read_text().splitlines() if line.strip()]
    columns = ("hours", "games", "round_mean", "round_median", "round_best", "kills_mean", "seconds_mean",
               "snapshot")
    with open(directory / "evals.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in sorted(every, key=lambda r: r.get("hours") or 0):
            writer.writerow([row.get(c) for c in columns])
        if human and human.get("games"):
            writer.writerow(["human", *[human.get(c) for c in columns[1:-1]], "demos"])
