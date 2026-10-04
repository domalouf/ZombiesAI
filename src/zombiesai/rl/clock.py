"""How much training the policy has had, counted across every run it was continued through.

A run's own counters start again at zero when the next run continues from its checkpoint (`version` in
weights.pt, `step` in metrics.jsonl), but the policy does not: it is the same policy, a few more hours in. The
clock is what a video says over a clip -- "hour 37, game 4,812" -- so it is carried in the checkpoint's `rl`
section (`Learner.checkpoint`), picked up by the next run, and handed to the actors with each new set of
weights (rl/weights.py), which is how a fleet worker on another PC knows it too.

* `train_s` -- seconds the learner has spent training, across runs (wall time from its start to its stop).
* `steps` -- decisions the learner has trained on.
* `games` -- finished games, on every PC.
* `played_s` -- seconds of game played in those games: with eight games at once it runs ~8x `train_s`.
* `updates` -- PPO updates (the learner's own count, which already carries over).
"""

from dataclasses import asdict, dataclass


@dataclass
class TrainingClock:
    train_s: float = 0.0
    steps: int = 0
    games: int = 0
    played_s: float = 0.0
    updates: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainingClock":
        d = d or {}
        return cls(train_s=float(d.get("train_s") or 0.0), steps=int(d.get("steps") or 0),
                   games=int(d.get("games") or 0), played_s=float(d.get("played_s") or 0.0),
                   updates=int(d.get("updates") or 0))

    def to_dict(self) -> dict:
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in asdict(self).items()}

    @property
    def hours(self) -> float:
        return self.train_s / 3600.0

    def label(self) -> str:
        """What a video says over a clip: "Hour 6 · game 812"."""
        return f"Hour {int(self.hours)} · game {self.games:,}"


class RunClock:
    """The learner's side: the clock the run started from, plus this run's training time, steps and games."""

    def __init__(self, start: TrainingClock, now: float):
        self.start, self._t0 = start, now
        self.games = 0
        self.played_s = 0.0

    def game(self, seconds) -> None:
        self.games += 1
        if isinstance(seconds, (int, float)) and seconds > 0:
            self.played_s += float(seconds)

    def at(self, now: float, steps: int, updates: int) -> TrainingClock:
        """The clock at `now`, `steps` into this run, after `updates` updates in all."""
        s = self.start
        return TrainingClock(train_s=s.train_s + max(0.0, now - self._t0), steps=s.steps + steps,
                             games=s.games + self.games, played_s=s.played_s + self.played_s, updates=updates)
