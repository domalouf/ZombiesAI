"""One real game as an RL environment: `reset()` until a fresh game is on, then `step(action)` fifteen times a
second, with the reward read off the HUD.

It is a real-time environment, so `step` does not wait for anything but the clock. The action goes out the
moment it arrives, the environment pumps it until the next decision's deadline, and the observation is the
frame grabbed at that deadline. Deadlines are absolute (`t0 + k * dt`); a step that finishes late is flagged
`bad` and the schedule moves on from *now* -- skip, never catch up (PLAN.md, "Real-time loop"). The agent's
own compute (a policy forward is ~0.5 ms on a CPU thread) happens between two calls and is part of the tick.

The reward is `reward.RewardShaper` on signals from `hud_reward.HudSignals`, which reads the settled events of
`hud/track.HudTracker` -- so a misread digit never becomes a reward, it becomes a rejected change, and three
in a row become `hud_lost`, which ends the episode.

Episode ends, and why they are what they are:

* **terminated** on a settled death (the downed penalty: solo Nacht has no revive). The one confirmed end.
* **truncated** -- bootstrapped from V(s), because nothing proves the game is over -- on `hud_lost`, on the
  picture staying frozen or the window staying lost for `stale_s`, on the HUD staying absent for
  `hud_absent_s` (a game-over screen whose death the tracker missed, a crash to the menu), and on the
  `max_episode_s` cap.

Reset is an explicit state machine, never a blind sleep. It releases every key, types the start command into
the game's console (`map nazi_zombie_prototype`), and waits for the HUD to show a *settled* 500 points on a live
picture. A try that does not get there within `start_timeout_s` is repeated; after `reset_attempts` of them
the game is relaunched (`restart`, which the instance manager provides), and the wait starts over.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from zombiesai import spec
from zombiesai.hud.parse import ABSENT, OK
from zombiesai.hud.track import START_POINTS, HudTracker
from zombiesai.realgame.hud_reward import HudSignals, SignalConfig
from zombiesai.realgame.instances import NACHT
from zombiesai.reward import REWARD_TERMS, RewardConfig, RewardShaper


@dataclass(frozen=True)
class EnvConfig:
    max_episode_s: float = 45 * 60.0
    overrun_factor: float = 1.5
    stale_s: float = 5.0  # the picture frozen or the window lost this long: truncate
    hud_absent_s: float = 8.0  # no points counter drawn this long, after it was: truncate
    start_command: str = f"map {NACHT}"
    start_timeout_s: float = 90.0  # per try, from the command to a settled 500
    launch_timeout_s: float = 240.0  # after relaunching the game (Proton start-up, then the map load)
    reset_attempts: int = 3  # console tries before a relaunch
    max_relaunches: int = 2  # per reset, before giving up
    first_look_s: float = 3.0  # the first reset of a run: is a fresh game already on?
    after_death_s: float = 4.0  # let the game-over screen play before typing into the console
    signals: SignalConfig = field(default_factory=SignalConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def dt(self) -> float:
        return 1.0 / spec.DECISION_HZ


class ResetFailed(RuntimeError):
    """No fresh game after every console try and a relaunch: the instance needs a human (or a supervisor)."""


class RealGameEnv:
    """`capture` reads frames and HUD crops (`read()`, `last_hud`, `last_stale`; `Instance.capture()`),
    `dispatcher` is an `ActionDispatcher` on this game's own input (`Instance.sink()`), `reader` turns crops
    into a `HudReading` (default `HudParser().parse`), `focus()` gives the game window input focus and says
    whether there is one, `console(command)` types into the game's console, and `restart()` relaunches the
    game. `hearing`, if given, is a `LiveAudio` on this instance's own sink."""

    def __init__(
        self,
        capture,
        dispatcher,
        *,
        reader=None,
        focus=None,
        console=None,
        restart=None,
        hearing=None,
        config: EnvConfig | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
        say=print,
    ):
        if reader is None:
            from zombiesai.hud.parse import HudParser

            reader = HudParser().parse
        self.capture, self.dispatcher, self.reader = capture, dispatcher, reader
        self.focus = focus or (lambda: True)
        self.console, self.restart, self.hearing = console, restart, hearing
        self.config = config or EnvConfig()
        self.clock, self.sleep, self.say = clock, sleep, say
        self.tracker = HudTracker()
        self.signals = HudSignals(self.config.signals)
        self.shaper = RewardShaper(self.config.reward)
        self._deadline = 0.0
        self._needs_restart = False  # set after the first reset: from then on every reset starts a new game
        self._ended_by_death = False
        self.episodes = 0
        self.resets = {"console": 0, "relaunch": 0}
        self._reset_episode_counters()

    # ------------------------------------------------------------------------------------------------ helpers

    def _reset_episode_counters(self) -> None:
        self.t_start = self.clock()
        self.steps = 0
        self.bad_steps = 0
        self.return_ = 0.0
        self._stale_since = None
        self._absent_since = None
        self._seen_hud = False

    def _observe(self):
        """Grab now: (obs, reading, stale). The HUD is parsed from the same grab as the frame."""
        t = self.clock()
        frame = self.capture.read()
        stale = bool(getattr(self.capture, "last_stale", False))
        crops = getattr(self.capture, "last_hud", None)
        reading = self.reader(crops) if crops is not None and not stale else None
        obs = {"pixels": frame}
        if self.hearing is not None:
            obs["audio"], obs["audio_mask"] = self.hearing.observe(t)
        return obs, reading, stale

    def _wait_tick(self) -> None:
        self._deadline += self.config.dt
        now = self.clock()
        if now < self._deadline:
            self.dispatcher.pump_until(self._deadline)

    # ------------------------------------------------------------------------------------------------ reset

    def reset(self):
        """Block until a fresh game is on (settled 500 points, live picture) and return its first obs.

        The first reset of a run looks before it types: a game just launched with `+map` is already on its way
        to a fresh start. After that every reset types the start command; `reset_attempts` failed tries earn a
        relaunch (which starts the map by itself, so the first wait after it types nothing), and more than
        `max_relaunches` of those raise `ResetFailed`."""
        self.dispatcher.release_all()
        if self._ended_by_death:
            self._pump_for(self.config.after_death_s)
        obs = self._await_fresh_game(self.config.first_look_s) if not self._needs_restart else None
        tries = relaunches = 0
        typed = True  # whether the next wait follows a command (or a relaunch) rather than just looking
        while obs is None:
            if tries == self.config.reset_attempts:
                if self.restart is None or relaunches == self.config.max_relaunches:
                    raise ResetFailed(f"no fresh game after {tries} tries and {relaunches} relaunches")
                self.say("  reset: relaunching the game")
                self.restart()
                self.resets["relaunch"] += 1
                relaunches += 1
                tries, typed = 0, False
                timeout = self.config.launch_timeout_s
            else:
                if typed and self.console is not None:
                    self._type(self.config.start_command)
                    self.resets["console"] += 1
                typed = True
                timeout = self.config.start_timeout_s
            obs = self._await_fresh_game(timeout)
            tries += 1
        self._needs_restart = True  # every later reset must start a new game
        self._ended_by_death = False
        self.signals.reset()
        self.shaper.reset()
        self.episodes += 1
        self._reset_episode_counters()
        self._seen_hud = True  # the reset just read a settled 500 off it
        self._deadline = self.clock()
        return obs, {"resets": dict(self.resets)}

    def _type(self, command: str) -> None:
        if self.focus():
            self.console(command)

    def _pump_for(self, seconds: float) -> None:
        end = self.clock() + seconds
        while self.clock() < end:
            self.dispatcher.pump_until(min(end, self.clock() + 0.1))

    def _await_fresh_game(self, timeout_s: float):
        """Poll at the decision rate until the HUD settles on 500 points in a live picture."""
        self.tracker.reset()
        end = self.clock() + timeout_s
        self._deadline = self.clock()
        while self.clock() < end:
            self.focus()
            obs, reading, stale = self._observe()
            if reading is not None:
                tracked = self.tracker.step(reading)
                if not stale and tracked.points == START_POINTS and reading.points_status == OK \
                        and reading.points == START_POINTS:
                    return obs
            self._wait_tick()
        return None

    # ------------------------------------------------------------------------------------------------ step

    def step(self, action):
        """Send `action`, wait out the tick, observe. Returns (obs, reward, terminated, truncated, info)."""
        action = np.asarray(action, dtype=np.int64)
        focused = self.focus()
        now = self.clock()
        if focused:
            self.dispatcher.apply(action, now, self.config.dt)
        else:
            self.dispatcher.release_all()
        self._wait_tick()
        late = self.clock() > self._deadline + (self.config.overrun_factor - 1.0) * self.config.dt
        if late:
            self._deadline = self.clock()  # skip, never catch up
        obs, reading, stale = self._observe()
        self.steps += 1

        tracked = self.tracker.step(reading) if reading is not None else self.tracker.state
        if reading is None:
            # Nothing new read: carry the settled state, but no event is re-reported.
            tracked.points_event, tracked.points_delta, tracked.round_changed = "", 0, False
        signals = self.signals.step(tracked, action)
        result = self.shaper(signals)
        self.return_ += result.reward

        now = self.clock()
        self._stale_since = (self._stale_since or now) if stale else None
        absent = reading is not None and reading.points_status == ABSENT
        self._seen_hud = self._seen_hud or (reading is not None and reading.points_status == OK)
        self._absent_since = (self._absent_since or now) if absent and self._seen_hud else None

        terminated = signals.death
        reason = "death" if terminated else None
        if not terminated:
            if tracked.hud_lost:
                reason = "hud_lost"
            elif self._stale_since is not None and now - self._stale_since >= self.config.stale_s:
                reason = "picture_lost"
            elif self._absent_since is not None and now - self._absent_since >= self.config.hud_absent_s:
                reason = "hud_gone"
            elif now - self.t_start >= self.config.max_episode_s:
                reason = "time_limit"
        truncated = reason is not None and not terminated
        bad = late or stale or not focused
        self.bad_steps += bad
        info = {"bad": bad, "points": tracked.points, "round": self.signals.round, "terms": result.terms}
        if terminated or truncated:
            self.dispatcher.release_all()
            self._ended_by_death = terminated
            info["episode"] = self.summary(reason)
        return obs, result.reward, terminated, truncated, info

    def summary(self, reason: str | None = None) -> dict:
        stats = self.shaper.stats
        return {
            "reason": reason,
            "return": self.return_,
            "length": self.steps,
            "seconds": self.clock() - self.t_start,
            "round_reached": max(self.signals.round, 1),
            "points_gained": self.signals.points_gained,
            "bad_steps": self.bad_steps,
            "repair_share": stats.repair_share(),
            "max_term_share": stats.max_term_share(),
            "gain_clips": stats.gain_clips,
            "events": dict(self.signals.events),
            "reward_term_sums": dict(zip(REWARD_TERMS, stats.term_sums.tolist())),
        }

    def close(self) -> None:
        self.dispatcher.close()
        self.capture.close()
        if self.hearing is not None:
            self.hearing.close()
