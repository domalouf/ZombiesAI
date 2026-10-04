"""The films kept for the video about training, beside the run's best game (rl/best_episode.py).

They all live in one directory, `runs/video` by default, shared by every run on this PC so that a run continued
from another carries on the same collection (a synthetic rehearsal keeps its own, in the run's directory):

    firsts/     the agent's firsts: its first kill, door, box, Ray Gun, round 5 ... (rl/moments.py)
    progress/   one whole game every `film_every_h` hours of training: the check-ins ("after 6 hours ...")
    records/    the superlatives: the shortest game, the most repairs, the worst aim, the longest without a kill
    grid/       every game at once (scripts/record_grid.py)
    evals/      snapshots of the policy played for the record (scripts/eval_snapshots.py)

A fleet worker on another PC keeps its own (`runs/fleet/video` there); `scripts/gather_video.py` copies them
here and `merge`s them in, each shelf by its own rule.

Each kept film has its sidecar: `<name>.brain.jsonl`, what the policy thought on each frame (its value
estimate, how sure it was, what it pressed, what it was paid) for `scripts/overlay.py` to draw over it, and an
entry in the directory's index (`firsts.json`, `progress.json`, `records.json`) saying when in training it was
filmed (`clock`, rl/clock.py), in which run, on which PC.

Every shelf decides under a file lock, as `keep_if_best` does, because the actors on a PC share it: a candidate
is kept when its slot is empty or it beats the one there by the shelf's own rule.
"""

import fcntl
import json
import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path


def default_dir(run_dir: Path, synthetic: bool) -> Path:
    """Every real run on this PC shares one collection, beside the runs; a rehearsal keeps its own."""
    run_dir = Path(run_dir)
    return run_dir / "video" if synthetic else run_dir.parent / "video"


def place(source: Path, dest: Path) -> None:
    """`source` at `dest` too, leaving `source` where it is: a hard link where the file system allows one."""
    dest.unlink(missing_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


class Shelf:
    """One directory of kept films, indexed by `<index>.json` (key -> record). `better(new, old)` says whether a
    candidate replaces the film already kept under its key."""

    def __init__(self, directory: Path, index: str):
        self.dir, self.index = Path(directory), index

    def read(self) -> dict:
        try:
            return json.loads((self.dir / f"{self.index}.json").read_text())
        except (OSError, ValueError):
            return {}

    def better(self, new: dict, old: dict) -> bool:
        raise NotImplementedError

    def wants(self, key: str, record: dict, kept: dict | None = None) -> bool:
        kept = self.read() if kept is None else kept
        return key not in kept or self.better(record, kept[key])

    def offer(self, key: str, record: dict, files: dict[str, Path], *, move: bool = False) -> bool:
        """Keep `files` (suffix -> path, ".mp4" first) as `<key><suffix>` if the record still wins its slot.
        With `move` the files are taken (and deleted when they lose), else they are left for the caller."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            kept = self.read()
            if not self.wants(key, record, kept):
                if move:
                    for path in files.values():
                        Path(path).unlink(missing_ok=True)
                return False
            names = {}
            for suffix, path in files.items():
                dest = self.dir / f"{key}{suffix}"
                if move:
                    os.replace(path, dest)
                else:
                    place(Path(path), dest)
                names[suffix] = dest.name
            for suffix in {".mp4", ".brain.jsonl"} - set(files):  # a sidecar left from the film replaced
                (self.dir / f"{key}{suffix}").unlink(missing_ok=True)
            kept[key] = {**record, "clip": names.get(".mp4"), "brain": names.get(".brain.jsonl"),
                         "host": record.get("host") or socket.gethostname()}
            tmp = self.dir / f"{self.index}.json.tmp"
            tmp.write_text(json.dumps(kept, indent=2, default=str))
            tmp.replace(self.dir / f"{self.index}.json")
            return True


class Progress(Shelf):
    """One game every `every_h` hours of training: slot k is the first game started at or after k * every_h
    hours of the training clock, so the films are the agent's check-ins whatever the run they came from."""

    def __init__(self, directory: Path, every_h: float):
        super().__init__(directory, "progress")
        self.every_h = every_h

    def key(self, start_clock: dict | None) -> str | None:
        if self.every_h <= 0 or not start_clock:
            return None
        hours = float(start_clock.get("train_s") or 0.0) / 3600.0
        return f"h{int(hours / self.every_h) * self.every_h:07.2f}"

    def better(self, new: dict, old: dict) -> bool:
        return (new.get("start_clock") or {}).get("train_s", 0) < (old.get("start_clock") or {}).get("train_s", 0)


@dataclass(frozen=True)
class Record:
    title: str
    lower: bool  # the record is the lowest value (else the highest)
    unit: str = ""


RECORDS = {
    "shortest_game": Record("Shortest game", lower=True, unit="s"),
    "most_repairs": Record("Most barricade repairs in one game", lower=False),
    "worst_aim": Record("Worst aim", lower=True, unit="hit rate"),
    "longest_without_kill": Record("Longest stretch without a kill", lower=False, unit="s"),
}
MIN_SHOTS_FOR_AIM = 30  # fewer shots than this says nothing about aim
DEATHS = ("down", "death")


def record_values(summary: dict) -> dict[str, float]:
    """The finished game's value for each record it can hold: the shortest game only counts a death (a game cut
    short by a lost window is not short play), worst aim only a game with enough shots to judge."""
    values = {}
    seconds = summary.get("seconds")
    if summary.get("reason") in DEATHS and isinstance(seconds, (int, float)):
        values["shortest_game"] = round(float(seconds), 1)
    repairs = (summary.get("events") or {}).get("repair")
    if isinstance(repairs, int) and repairs > 0:
        values["most_repairs"] = float(repairs)
    shots, hits = summary.get("shots"), summary.get("hits")
    if isinstance(shots, int) and shots >= MIN_SHOTS_FOR_AIM and isinstance(hits, int):
        values["worst_aim"] = round(hits / shots, 4)
    drought = summary.get("longest_without_kill_s")
    if isinstance(drought, (int, float)) and drought > 0:
        values["longest_without_kill"] = round(float(drought), 1)
    return values


class Records(Shelf):
    def __init__(self, directory: Path):
        super().__init__(directory, "records")

    def better(self, new: dict, old: dict) -> bool:
        a, b = new.get("value"), old.get("value")
        if b is None:
            return True
        return a < b if RECORDS[new["record"]].lower else a > b

    def candidates(self, summary: dict) -> list[tuple[str, dict]]:
        kept = self.read()
        out = []
        for name, value in record_values(summary).items():
            record = {"record": name, "title": RECORDS[name].title, "value": value, "unit": RECORDS[name].unit}
            if self.wants(name, record, kept):
                out.append((name, record))
        return out


class Keepsakes:
    """The shelves an actor offers its finished games to (any of them None: not kept)."""

    def __init__(self, directory: Path, *, firsts: bool = True, progress_every_h: float = 1.0,
                 records: bool = True):
        from zombiesai.rl.moments import MomentBook

        self.dir = Path(directory)
        self.firsts = MomentBook(self.dir / "firsts") if firsts else None
        self.progress = Progress(self.dir / "progress", progress_every_h) if progress_every_h > 0 else None
        self.records = Records(self.dir / "records") if records else None

    def __bool__(self) -> bool:
        return any(s is not None for s in (self.firsts, self.progress, self.records))

    def shelves(self) -> dict[str, Shelf]:
        return {name: shelf for name, shelf in (("firsts", self.firsts), ("progress", self.progress),
                                                 ("records", self.records)) if shelf is not None}


def merge(into: Path, source: Path) -> dict[str, int]:
    """Offer every film kept under `source` (another PC's video directory, copied here) to the shelves under
    `into`, each by its shelf's own rule -- the earlier first, the earlier check-in, the more extreme record --
    so the collection ends up as if every PC's games had been played on this one. Counts what was taken."""
    ours, theirs = Keepsakes(into), Keepsakes(source)
    taken = {}
    for name, shelf in theirs.shelves().items():
        mine = ours.shelves()[name]
        taken[name] = 0
        for key, record in shelf.read().items():
            files = {suffix: shelf.dir / record[field] for suffix, field in ((".mp4", "clip"),
                                                                            (".brain.jsonl", "brain"))
                     if record.get(field) and (shelf.dir / record[field]).exists()}
            if ".mp4" in files:
                taken[name] += mine.offer(key, record, files)
    return taken
