"""Games kept on the site for good. A run's best game is replaced whenever the run does better (rl/best_episode.py),
so one worth keeping for itself -- a funny ending, a first -- is copied out of the run (scripts/save_game.py) and
stays, under "Saved games" on the Training Room, whatever the run does next.

They live beside the runs, in `runs/saved/`:

    <name>.mp4          the film, as the run's actors kept it
    <name>.json         the run's best.json as it was when saved: the record (it names the host, so stays here)
    <name>.brain.jsonl  what the policy thought on each frame, for scripts/overlay.py, when the run kept one
    saved.json          the index, name -> what the site says about the game

The live page's build (viz/live_site.py) reads every checkout's index and ships each film as
zombies/training/saved/<name>.mp4, beside the page; deploy/deploy.sh publishes them with it.
"""

import json
import os
import re
import time
from pathlib import Path

from zombiesai.rl.keepsakes import place

SAVED_DIR = "saved"
INDEX = "saved.json"
# A saved game's name is its file name on the site: short, lower case, no dots or slashes.
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
# What the site says about a game: the run's name is public already; the host that played it is not.
PUBLIC_FIELDS = ("name", "title", "caption", "run", "round", "points", "kills", "seconds", "recorded", "saved")


def read_index(saved_dir: Path) -> dict:
    try:
        index = json.loads((Path(saved_dir) / INDEX).read_text())
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


def save_game(run_dir: Path, saved_dir: Path, name: str, title: str, caption: str = "",
              now: float | None = None) -> dict:
    """Keep `run_dir`'s best game as `name`, with what the site says about it. A name is saved once: a second game
    under it would change what a link to the first shows."""
    run_dir, saved_dir = Path(run_dir), Path(saved_dir)
    if not NAME.fullmatch(name):
        raise ValueError(f"name {name!r}: lower-case letters, digits and dashes, 64 at most")
    if not title.strip():
        raise ValueError("a saved game needs a title")
    best = run_dir / "best"
    record = json.loads((best / "best.json").read_text())
    if name in read_index(saved_dir) or (saved_dir / f"{name}.mp4").exists():
        raise FileExistsError(f"{name} is saved already")
    saved_dir.mkdir(parents=True, exist_ok=True)
    # best.mp4 is only ever replaced (os.replace), never written in place, so a hard link keeps this one.
    place(best / "best.mp4", saved_dir / f"{name}.mp4")
    if (best / "best.brain.jsonl").exists():
        place(best / "best.brain.jsonl", saved_dir / f"{name}.brain.jsonl")
    (saved_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    entry = {
        "name": name, "title": title.strip(), "caption": caption.strip(), "run": run_dir.name,
        "round": record.get("round"), "points": record.get("points"), "kills": record.get("kills"),
        "seconds": stats.get("seconds", summary.get("seconds")), "recorded": record.get("recorded_unix"),
        "saved": time.time() if now is None else now,
    }
    index = read_index(saved_dir)
    index[name] = entry
    tmp = saved_dir / (INDEX + ".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    os.replace(tmp, saved_dir / INDEX)
    return entry


def saved_games(runs_dirs: list[Path]) -> list[tuple[dict, Path]]:
    """(what the site says, the film) for every saved game in these runs dirs whose film is there, newest saved
    first. A name saved in two checkouts is the first one's."""
    games: dict[str, tuple[dict, Path]] = {}
    for runs_dir in runs_dirs:
        saved_dir = Path(runs_dir) / SAVED_DIR
        for name, entry in read_index(saved_dir).items():
            film = saved_dir / f"{name}.mp4"
            if name in games or not NAME.fullmatch(name) or not isinstance(entry, dict) or not film.is_file():
                continue
            public = {k: entry.get(k) for k in PUBLIC_FIELDS}
            games[name] = (dict(public, name=name, video=f"{SAVED_DIR}/{name}.mp4?v={int(film.stat().st_mtime)}"), film)
    return sorted(games.values(), key=lambda g: g[0].get("saved") or 0, reverse=True)


def ship(games: list[tuple[dict, Path]], out_dir: Path) -> list[dict]:
    """Put each film in `out_dir`/saved/ (the page's directory), and return what the page says about them."""
    if not games:
        return []
    films = Path(out_dir) / SAVED_DIR
    films.mkdir(parents=True, exist_ok=True)
    for entry, film in games:
        place(film, films / f"{entry['name']}.mp4")
    return [entry for entry, _ in games]
