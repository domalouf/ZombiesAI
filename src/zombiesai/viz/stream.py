"""The Twitch stream's numbers: the overlay drawn on the stream, and the page around it (domalouf.com/zombies/live/).

Both read one file, stream.json, which the live publisher (viz/live_site.py) writes beside machine.json and
runs.json every few seconds and pushes to the site's zombies/training/live/:

    stats    the four the overlay shows: best round, average round, shot accuracy, average survival time
             (and the average points and kills a game earned, which the live viewer shows: realgame/viewer.py.
             Kills are the game-over scoreboard's (`end_kills`, realgame/end_screen.py), averaged over the games
             it was read in; only a window with no such game -- an older run -- falls back to the HUD-gain
             estimate (`kills`, a lower bound). The two are never mixed in one average, so the number cannot
             change definition mid-window. A run with neither averages to "no value", not 0)
    recent   the last games, newest first, for the page's table
    rounds   how many games ended in each round
    curves   the round, survival and accuracy as rolling means over the games, for the page's charts
    ppo      the run's training progress and its policy-health numbers, from the same payload as runs.json

The games come from the run's `episodes.jsonl`, one line per finished game (rl/parallel_ppo.py). Which run:
the one `--stream-run` names, else one a trainer is writing now (a real-game run before a rehearsal), else
the run whose games were written last. Averages are over the last `WINDOW` games, so the overlay says how the
agent plays now and not how it played a day ago; the best round is the run's best.

Deliberately stdlib-only, like the dashboard: it reads JSON a trainer wrote and nothing else.
"""

import json
import math
import shutil
from html import escape
from pathlib import Path

from zombiesai.viz.dashboard import FONTS, _local_fonts

HERE = Path(__file__).parent
WINDOW = 100  # games the averages are over
RECENT = 25  # games the page lists
CURVE_POINTS = 120  # at most this many points per rolling curve
EPISODES = "episodes.jsonl"
# The PPO numbers the page shows as policy health: (metrics.jsonl key, label, what it means).
HEALTH = (
    ("entropy", "Policy entropy", "nats; falling means the policy is committing"),
    ("approx_kl", "Approx. KL", "how far each update moves the policy"),
    ("clipfrac", "Clip fraction", "share of the batch the surrogate clipped"),
    ("explained_variance", "Explained variance", "1 is a perfect critic, 0 is predicting the mean"),
    ("kl_ref", "KL to the human prior", "how far it has moved from the behavioural-cloning policy"),
    ("value_loss", "Value loss", ""),
)
# The run's curves the page charts against steps, from runs.json's series for it.
PPO_CURVES = ("return_mean", "round_reached_mean", "entropy", "approx_kl", "clipfrac", "explained_variance")


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _r(v, nd: int = 3):
    return None if v is None else round(float(v), nd)


class EpisodeTail:
    """episodes.jsonl, read incrementally: a run plays thousands of games, and a 5 s tick should read the new
    lines, not all of them. A file that shrank or changed is read again from the start."""

    def __init__(self):
        self.path: Path | None = None
        self.offset = 0
        self.ino = None
        self.games: list[dict] = []
        self._partial = b""

    def read(self, path: Path | None) -> list[dict]:
        if path != self.path:
            self.__init__()
            self.path = path
        if path is None:
            return self.games
        try:
            st = path.stat()
        except OSError:
            return self.games
        if st.st_ino != self.ino or st.st_size < self.offset:
            self.offset, self.games, self._partial, self.ino = 0, [], b"", st.st_ino
        if st.st_size == self.offset:
            return self.games
        with open(path, "rb") as f:
            f.seek(self.offset)
            data = self._partial + f.read()
            self.offset = f.tell()
        *lines, self._partial = data.split(b"\n")  # a trainer's last line can be half-written
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                self.games.append(row)
        return self.games


def accuracy(games: list[dict]) -> float | None:
    """Landed over fired, summed over the games: one game of three shots must not weigh what one of 300 does."""
    shots = sum(_num(g.get("shots")) or 0 for g in games)
    hits = sum(_num(g.get("hits")) or 0 for g in games)
    return min(1.0, hits / shots) if shots else None


def _mean(values) -> float | None:
    values = [v for v in (_num(v) for v in values) if v is not None]
    return sum(values) / len(values) if values else None


def average_kills(games: list[dict]) -> float | None:
    """The scoreboard's kills over the games it was read in; the HUD estimate only if it was read in none."""
    read = [g.get("end_kills") for g in games if _num(g.get("end_kills")) is not None]
    return _mean(read) if read else _mean(g.get("kills") for g in games)


def summarize(games: list[dict], window: int = WINDOW) -> dict:
    """The overlay's four numbers. The best round is the run's; the rest are over the last `window` games."""
    recent = games[-window:]
    rounds = [r for r in (_num(g.get("round_reached")) for g in games) if r is not None]
    return {
        "games": len(games),
        "window": len(recent),
        "best_round": int(max(rounds)) if rounds else None,
        "avg_round": _r(_mean(g.get("round_reached") for g in recent), 2),
        "accuracy": _r(accuracy(recent), 4),
        "avg_survival_s": _r(_mean(g.get("seconds") for g in recent), 1),
        "avg_points": _r(_mean(g.get("points_gained") for g in recent), 1),
        "avg_kills": _r(average_kills(recent), 2),
    }


def rolling(games: list[dict], window: int = WINDOW, points: int = CURVE_POINTS) -> dict:
    """Round, survival and accuracy as means over the `window` games ending at each of at most `points` games.
    x is the game's number in the run (1-based)."""
    n = len(games)
    out = {"x": [], "round": [], "survival": [], "accuracy": []}
    if not n:
        return out
    ends = sorted({max(1, round(n * (k + 1) / points)) for k in range(min(points, n))})
    for end in ends:
        span = games[max(0, end - window):end]
        out["x"].append(end)
        out["round"].append(_r(_mean(g.get("round_reached") for g in span), 2))
        out["survival"].append(_r(_mean(g.get("seconds") for g in span), 1))
        out["accuracy"].append(_r(accuracy(span), 4))
    return out


def round_counts(games: list[dict]) -> list[list[int]]:
    """[[round, games that ended there], ...] from round 1 to the best, empty rounds included."""
    counts: dict[int, int] = {}
    for g in games:
        r = _num(g.get("round_reached"))
        if r is not None:
            counts[int(r)] = counts.get(int(r), 0) + 1
    return [[r, counts.get(r, 0)] for r in range(1, max(counts) + 1)] if counts else []


def recent_games(games: list[dict], n: int = RECENT) -> list[dict]:
    out = []
    for i, g in zip(range(len(games), 0, -1), reversed(games[-n:])):
        shots, hits = _num(g.get("shots")), _num(g.get("hits"))
        out.append({
            "n": i, "t": _num(g.get("t")), "round": _num(g.get("round_reached")), "seconds": _r(_num(g.get("seconds")), 1),
            "accuracy": _r(min(1.0, hits / shots), 4) if shots and hits is not None else None,
            "shots": shots, "return": _r(_num(g.get("return")), 2), "reason": g.get("reason"),
        })
    return out


def ppo_section(run: dict | None) -> dict | None:
    """The run's training progress and health, from its entry in runs.json (already scrubbed)."""
    if run is None:
        return None
    last = run.get("last_row") or {}
    series = {}
    for key in PPO_CURVES:
        s = (run.get("series") or {}).get(key)
        if s:
            series[key] = {k: s.get(k) for k in ("x", "y", "label", "hint", "last", "best")}
            series[key]["trend"] = (s.get("trend") or {}).get("verdict")
    return {
        "status": run.get("status"), "progress": run.get("progress"), "progress_text": run.get("progress_text"),
        "elapsed_s": run.get("elapsed_s"), "eta_s": run.get("eta_s"),
        "steps": _num(last.get("step")), "updates": _num(last.get("update")), "episodes": _num(last.get("episodes")),
        "sps": _num(last.get("sps")), "actors": _num(last.get("actors_alive")), "warmup": bool(last.get("warmup")),
        "health": [{"key": k, "label": label, "hint": hint, "value": _r(_num(last.get(k)), 4)}
                   for k, label, hint in HEALTH if _num(last.get(k)) is not None],
        "series": series,
    }


def stream_payload(name: str | None, games: list[dict], run: dict | None, *, live: bool, now: float) -> dict:
    return {
        "at": now, "run": name, "live": live, "env": run and run.get("env"),
        "stats": summarize(games), "recent": recent_games(games), "rounds": round_counts(games),
        "curves": rolling(games), "window": WINDOW, "ppo": ppo_section(run),
    }


def pick_run(run_paths: dict[str, str], trainers: list[dict], prefer: str | None = None) -> tuple[str, Path] | None:
    """(public name, run dir) of the run the stream follows; see the module docstring. Only runs that have
    written games count."""
    by_name = {name: Path(path) for path, name in run_paths.items() if (Path(path) / EPISODES).exists()}
    if prefer:
        return (prefer, by_name[prefer]) if prefer in by_name else None
    writing = [t for t in trainers if t.get("run") in by_name]
    writing.sort(key=lambda t: bool(t.get("rehearsal")))  # the real game first: it is what the stream shows
    if writing:
        return writing[0]["run"], by_name[writing[0]["run"]]
    if not by_name:
        return None
    name = max(by_name, key=lambda n: (by_name[n] / EPISODES).stat().st_mtime)
    return name, by_name[name]


class StreamFeed:
    """What the live publisher calls each tick: which run, its games so far, and the payload for stream.json."""

    def __init__(self, prefer: str | None = None):
        self.prefer = prefer
        self.tail = EpisodeTail()
        self.choice: tuple[str, Path] | None = None

    def choose(self, run_paths: dict[str, str], trainers: list[dict]) -> None:
        self.choice = pick_run(run_paths, trainers, self.prefer)

    def payload(self, runs: dict | None, trainers: list[dict], now: float) -> dict:
        name, run_dir = self.choice or (None, None)
        games = self.tail.read(run_dir / EPISODES if run_dir else None)
        run = next((r for r in (runs or {}).get("runs", []) if r["name"] == name), None)
        live = any(t.get("run") == name for t in trainers) if name else False
        return stream_payload(name, games, run, live=live, now=now)


# ------------------------------------------------------------------------------------------------ the pages


def _page(template: str, data: dict | None, replacements: dict[str, str]) -> str:
    html = (HERE / template).read_text()
    snapshot = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = html.replace("/*__STREAM_DATA__*/null", snapshot, 1)
    for key, value in replacements.items():
        html = html.replace(key, value)
    return _local_fonts(html, embed=False)


def _copy_fonts(out_dir: Path) -> None:
    (out_dir / "fonts").mkdir(parents=True, exist_ok=True)
    for f in FONTS.iterdir():
        if f.suffix in (".woff2", ".txt"):
            shutil.copy2(f, out_dir / "fonts" / f.name)


def write_stream_site(out_dir: str | Path, *, channel: str | None, snapshot: dict | None = None,
                      links: list[tuple[str, str]] = (), description: str = "") -> tuple[Path, Path]:
    """The stream's two pages as a static directory, published at the site's zombies/live/:

        index.html          the Twitch player and the training around it
        overlay/index.html  the overlay alone, on a transparent page, for OBS's browser source

    Both poll ../training/live/stream.json (relative, so it works on any host serving the zombies/ tree). The
    page starts from `snapshot`, the numbers as they stood at build time, so it says something with the PC off.
    `channel` is the Twitch channel; without one the player is a placeholder until there is a stream."""
    out_dir = Path(out_dir)
    channel = (channel or "").strip().lstrip("@") or None
    if channel and not channel.replace("_", "").isalnum():
        raise ValueError(f"not a Twitch channel name: {channel!r}")
    nav = "".join(f'<a href="{escape(href)}">{escape(label)}</a>' for label, href in links)
    page = _page("stream_page.html", snapshot, {
        "__CHANNEL__": escape(channel or ""), "<!--nav-->": nav, "__DESCRIPTION__": escape(description),
    })
    index = out_dir / "index.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    index.write_text(page)
    _copy_fonts(out_dir)
    overlay = out_dir / "overlay" / "index.html"
    overlay.parent.mkdir(exist_ok=True)
    overlay.write_text(_page("stream_overlay.html", None, {}))
    _copy_fonts(overlay.parent)
    return index, overlay


def demo_games(n: int = 400, seed: int = 7) -> list[dict]:
    """Made-up games that get better, for laying the pages out before a run has played any (--demo)."""
    import random

    rng = random.Random(seed)
    games, t = [], 1.79e9
    for i in range(n):
        skill = i / n
        rnd = max(1, min(12, int(rng.gauss(1.6 + 4.5 * skill, 1.1))))
        seconds = max(20.0, rng.gauss(70 + 95 * rnd, 40))
        shots = int(rng.gauss(30 + 70 * rnd, 15))
        t += seconds + 20
        games.append({"t": round(t, 1), "step": i * 1800, "actor": i % 4, "episode": i // 4,
                      "reason": "death" if rng.random() > 0.08 else "hud_lost", "return": round(rng.gauss(rnd * 8, 4), 2),
                      "seconds": round(seconds, 1), "round_reached": rnd, "shots": max(0, shots),
                      "hits": max(0, int(shots * min(0.9, rng.gauss(0.18 + 0.25 * skill, 0.05))))})
    return games
