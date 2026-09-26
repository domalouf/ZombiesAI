"""A clip's HUD, summarised: the numbers to compare AI play runs and demos by.

Built from the checked tracks (`hud/track.py`), not the raw reads, so a misread cannot become a record:
rounds come from settled tally counts and the round changes between them, points from settled values and
the classified changes between them. A clip can hold several games (a death is a game over in solo Nacht;
the next game starts at 500 points and round 1) or start mid-game.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from zombiesai.hud.track import START_POINTS, ClipCheck


@dataclass
class Game:
    start_step: int
    end_step: int  # last step of the game in this clip (inclusive)
    started_in_clip: bool  # False: the clip begins mid-game
    game_over: bool  # the game ended in this clip: a down (solo Nacht has no revive) or the game-over count-up
    rounds_reached: int  # highest round seen (-1: never read)
    score: int  # the game-over screen's total, or -1 when the game did not end in the clip
    peak_points: int
    final_points: int  # last settled points before the game over (or the clip's end)
    points_gained: int  # sum of settled gains
    points_spent: int  # sum of settled spends (purchases), positive
    downs: int  # downed penalties seen
    seconds: float  # from the game's first step in the clip to its game over (or the clip's end)
    round_seconds: list = field(default_factory=list)  # time each fully seen round took


def summarize(check: ClipCheck, hz: float = 15.0, playing: np.ndarray | None = None) -> dict:
    n = check.n_steps
    playing = np.ones(n, bool) if playing is None else playing.astype(bool)
    events = check.points_events
    bounds = [0] + [e.step for e in events if e.kind == "new_game"] + [n]
    games = []
    for gi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        evs = [e for e in events if a <= e.step < b]
        runs = [r for r in check.points_runs if a <= r.start < b]
        over = [e for e in evs if e.kind == "game_over"]
        ends = [e.step for e in evs if e.kind in ("downed", "game_over")]
        end_play = ends[0] if ends else b
        in_game = [r for r in runs if r.start < end_play]
        # The tallies reset a little after the points do; until then the old game's count is still up.
        reset = [e.step for e in check.round_events if e.kind == "new_game" and a <= e.step < b]
        r0 = reset[0] if (gi > 0 and reset) else a
        rounds = check.round_inferred[r0:end_play] if check.round_inferred is not None else np.array([-1])
        round_changes = [e for e in check.round_events if r0 <= e.step < end_play and e.kind in ("next", "inferred_next")]
        games.append(Game(
            start_step=a,
            end_step=b - 1,
            started_in_clip=gi > 0 or bool(runs and runs[0].value == START_POINTS and rounds[:1].max(initial=-1) <= 1),
            game_over=bool(over) or any(e.kind == "downed" for e in evs),
            rounds_reached=int(max(rounds.max(initial=-1), max((e.after for e in round_changes), default=-1))),
            score=over[-1].after if over else -1,
            peak_points=max((r.value for r in in_game), default=-1),
            final_points=in_game[-1].value if in_game else -1,
            points_gained=sum(e.delta for e in evs if e.kind == "gain"),
            points_spent=-sum(e.delta for e in evs if e.kind == "spend"),
            downs=sum(e.kind == "downed" for e in evs),
            seconds=round((end_play - a) / hz, 1),
            round_seconds=[round((y.step - x.step) / hz, 1) for x, y in zip(round_changes, round_changes[1:])],
        ))
    played_min = playing.sum() / hz / 60
    gained = sum(g.points_gained for g in games)
    return {
        "steps": n,
        "seconds": round(n / hz, 1),
        "played_seconds": round(playing.sum() / hz, 1),
        "games": [asdict(g) for g in games],
        "game_overs": sum(g.game_over for g in games),
        "highest_round": max((g.rounds_reached for g in games), default=-1),
        "best_score": max((g.score for g in games), default=-1),
        "peak_points": max((g.peak_points for g in games), default=-1),
        "final_points": check.points_runs[-1].value if check.points_runs else -1,
        "points_gained": gained,
        "points_per_minute": round(gained / played_min, 1) if played_min > 0 else float("nan"),
        "downs": sum(g.downs for g in games),
        "read": {k: {m: (round(x, 4) if isinstance(x, float) else x) for m, x in v.items()} for k, v in check.stats.items()},
    }
