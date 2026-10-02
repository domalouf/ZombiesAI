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

The episode summary also carries an estimate of how well it shoots, for the stream overlay (viz/stream.py):
`shots` are the magazine marks that went away between two good reads while the fire button was held, and
`hits` the settled non-repair gains -- every one of those is at least one bullet that landed, but a counter
still rolling from one hit when the next lands settles as a single gain, so `hits / shots` is a floor.
"""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from zombiesai import spec
from zombiesai.hud.parse import ABSENT, OK
from zombiesai.hud.track import START_POINTS, HudTracker
from zombiesai.realgame.hud_reward import HudSignals, SignalConfig
from zombiesai.realgame.instances import NACHT
from zombiesai.reward import REWARD_TERMS, RewardConfig, RewardShaper


# More marks than this gone between two good reads is a misread or a weapon swap, not a burst.
MAX_SHOTS_BETWEEN_READS = 12


@dataclass(frozen=True)
class EnvConfig:
    max_episode_s: float = 45 * 60.0
    overrun_factor: float = 1.5
    stale_s: float = 5.0  # the picture frozen or the window lost this long: truncate
    hud_absent_s: float = 8.0  # no points counter drawn this long, after it was: truncate
    start_command: str = f"map {NACHT}"
    # Tried first when the map is (probably) still loaded: restarts the level in ~4 s, where `map` reloads it
    # in 60-90 s at 1440p. From a live game and from the game-over screen alike. None skips it.
    quick_start_command: str | None = "fast_restart"
    quick_start_timeout_s: float = 20.0
    # Tapped once a fresh game is up: after a fast_restart from the game-over screen the scoreboard stays
    # drawn until the scores key is pressed and let go.
    after_start_key: str | None = "tab"
    clear_tries: int = 3  # taps of it, until `downed()` reads false
    # Consecutive steps `downed()` must hold to end the episode as a death: co-op draws the scoreboard when the
    # player goes down (realgame/scoreboard.py), and solo Nacht -- what the demos were -- ends there.
    down_steps: int = 3
    # The start-of-game scoreboard can come up a moment after the reset saw a clean screen: in an episode's
    # first seconds it is cleared again, not counted -- no zombie reaches the spawn that fast.
    down_grace_s: float = 4.0
    # Per try, from the command to a settled 500. Generous on purpose: the next try types `map` again, which
    # restarts a load from zero -- and a 1440p load beside three live games can take a few minutes, so a short
    # timeout restarts the same slow load forever, until the relaunch.
    start_timeout_s: float = 240.0
    launch_timeout_s: float = 420.0  # after relaunching the game (Proton start-up, the menu, then the map load)
    reset_attempts: int = 3  # console tries before a relaunch
    max_relaunches: int = 2  # per reset, before giving up
    first_look_s: float = 3.0  # the first reset of a run: is a fresh game already on?
    after_death_s: float = 4.0  # let the game-over screen play before typing into the console
    # A map loaded from the console stops at "Click to Start the Mission". Enter starts it too, without firing
    # a shot; it is tapped every `start_key_every_s` while a reset waits (it does nothing in a live game).
    start_key: str | None = "enter"
    start_key_every_s: float = 2.0
    retype_every_s: float = 3.0  # a start command the console did not take is tried again this often
    console_key: str = "grave"
    # The view's pitch springs back to level with this time constant (s); None turns it off. A cloned policy
    # drifts a degree or so a second and never learned to look back up -- humans barely use pitch (~2 deg/s
    # against ~17 of yaw) -- so without it every agent ends up staring at the floor and never sees a zombie.
    # It is part of the environment, like aim assist: the policy's look bins are sent as chosen, plus the
    # spring's pull. Pitch is tracked exactly (XTEST counts are exact, and every map load starts level).
    pitch_spring_s: float | None = 2.0
    pitch_limit_deg: float = 85.0  # where the engine stops the view
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
    whether there is one, `console(command)` types into the game's console, `press(key)` taps one key,
    `console_open()` says whether the last frame shows the console, `downed()` whether it shows the player
    down (co-op's scoreboard), and `restart()` relaunches the game. `hearing`, if given, is a `LiveAudio` on this instance's own sink."""

    def __init__(
        self,
        capture,
        dispatcher,
        *,
        reader=None,
        focus=None,
        console=None,
        restart=None,
        press=None,
        console_open=None,
        downed=None,
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
        self.console, self.restart, self.hearing, self.press = console, restart, hearing, press
        self.console_open, self.downed = console_open, downed
        self._down_run = 0
        self._next_clear = 0.0
        self.pitch = 0.0  # degrees from level, as sent
        self.config = config or EnvConfig()
        self.clock, self.sleep, self.say = clock, sleep, say
        self.tracker = HudTracker()
        self.signals = HudSignals(self.config.signals)
        self.shaper = RewardShaper(self.config.reward)
        self._deadline = 0.0
        self._needs_restart = False  # set after the first reset: from then on every reset starts a new game
        self._ended_by_death = False
        self.episodes = 0
        self.resets = {"console": 0, "quick": 0, "relaunch": 0}
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
        self.shots = 0
        self._mag = -1  # the last good magazine read, -1 when there is none to count from
        self._fired = False  # fire held since that read

    def _count_shots(self, reading, action) -> None:
        """Magazine marks that went away while the trigger was held. A drop without firing is a weapon swap,
        a rise is a reload; neither is a shot, and both start the count again from the new read."""
        self._fired = self._fired or bool(action[spec.FIRE])
        if reading is None or reading.mag_status != OK:
            return
        drop = self._mag - reading.mag
        if self._mag >= 0 and self._fired and 0 < drop <= MAX_SHOTS_BETWEEN_READS:
            self.shots += drop
        self._mag, self._fired = reading.mag, False

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

        The first reset of a run looks before it types: a game may already be on a fresh start. After that
        every wait carries the start command, retried until the console takes it (a game still loading, or
        on a screen that swallows the key, does not open it); `reset_attempts` failed tries earn a relaunch --
        after which the command is typed too, since Plutonium's LAN mode cannot start a map from its command
        line -- and more than `max_relaunches` of those raise `ResetFailed`."""
        self.dispatcher.release_all()
        if self._ended_by_death:
            self._pump_for(self.config.after_death_s)
        obs = self._await_fresh_game(self.config.first_look_s) if not self._needs_restart else None
        tries = relaunches = 0
        quick = self.config.quick_start_command is not None and self._needs_restart
        while obs is None:
            if tries == self.config.reset_attempts:
                if self.restart is None or relaunches == self.config.max_relaunches:
                    raise ResetFailed(f"no fresh game after {tries} tries and {relaunches} relaunches")
                self.say("  reset: relaunching the game")
                self.restart()
                self.resets["relaunch"] += 1
                relaunches += 1
                tries = 0
                quick = False  # nothing is loaded to restart
                timeout = self.config.launch_timeout_s
            elif quick:
                quick = False
                obs = self._await_fresh_game(self.config.quick_start_timeout_s,
                                             command=self.config.quick_start_command)
                if obs is not None:
                    self.resets["quick"] += 1
                continue
            else:
                timeout = self.config.start_timeout_s
            obs = self._await_fresh_game(timeout, command=self.config.start_command)
            tries += 1
        obs = self._clear_after_start(obs)
        self._needs_restart = True  # every later reset must start a new game
        self._down_run = 0
        self._next_clear = 0.0
        self.pitch = 0.0  # a map load starts level
        self._ended_by_death = False
        self.signals.reset()
        self.shaper.reset()
        self.episodes += 1
        self._reset_episode_counters()
        self._seen_hud = True  # the reset just read a settled 500 off it
        self._deadline = self.clock()
        return obs, {"resets": dict(self.resets)}

    def _clear_after_start(self, obs):
        """Tap `after_start_key` until the scoreboard a fresh co-op game opens with is gone, so that its coming
        back can mean a down. Returns the latest observation."""
        if self.press is None or not self.config.after_start_key:
            return obs
        for _ in range(self.config.clear_tries):
            if self.downed is not None and not self.downed():
                break
            self.press(self.config.after_start_key)
            self._pump_for(0.3)
            obs, _, _ = self._observe()
            if self.downed is None:
                break
        return obs

    def _look(self, action) -> tuple[float, float] | None:
        """The turn to send: the chosen bins, plus the pitch spring's pull back to level."""
        if self.config.pitch_spring_s is None:
            return None
        yaw = spec.YAW_BINS_DEG[int(action[spec.YAW])]
        limit = self.config.pitch_limit_deg
        wanted = float(np.clip(self.pitch + spec.PITCH_BINS_DEG[int(action[spec.PITCH])], -limit, limit))
        wanted -= wanted * min(1.0, self.config.dt / self.config.pitch_spring_s)
        sent, self.pitch = wanted - self.pitch, wanted
        return yaw, sent

    def _close_console(self) -> bool:
        """Close the console if the last frame shows it open. True when it had to."""
        if self.console_open is None or self.press is None or not self.console_open():
            return False
        self.dispatcher.release_all()
        self.press(self.config.console_key)
        return True

    def _type(self, command: str) -> bool:
        """Type `command` into the console. False when it could not be (no focus, or the console would not
        open); a console callback that cannot tell returns None, which counts as typed."""
        if not self.focus():
            return False
        return self.console(command) is not False

    def _pump_for(self, seconds: float) -> None:
        end = self.clock() + seconds
        while self.clock() < end:
            self.dispatcher.pump_until(min(end, self.clock() + 0.1))

    def _await_fresh_game(self, timeout_s: float, command: str | None = None):
        """Poll at the decision rate until the HUD settles on 500 points in a live picture.

        With `command`, it is typed as soon as the console takes it (retried every `retype_every_s`), and 500
        only counts once the HUD has gone away since -- the new map loading. Otherwise a game-over screen that
        still shows the dead game's 500 would pass for a fresh start. Between tries it taps `start_key`
        through "Click to Start the Mission", and closes the console if it finds it open."""
        self.tracker.reset()
        end = self.clock() + timeout_s
        self._deadline = self.clock()
        pending = command if self.console is not None else None
        next_type = self.clock()
        next_press = self.clock() + self.config.start_key_every_s
        reloaded = pending is None
        while self.clock() < end:
            now = self.clock()
            if self.focus():
                if pending is not None and now >= next_type:
                    if self._type(pending):
                        pending = None
                        self.resets["console"] += 1
                    next_type = now + self.config.retype_every_s
                elif pending is None and self.press is not None and now >= next_press:
                    if not self._close_console() and self.config.start_key:
                        self.press(self.config.start_key)
                    next_press = now + self.config.start_key_every_s
            obs, reading, stale = self._observe()
            if reading is not None:
                if pending is None and reading.points_status == ABSENT:
                    reloaded = True
                tracked = self.tracker.step(reading)
                if reloaded and not stale and tracked.points == START_POINTS and reading.points_status == OK \
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
            self.dispatcher.apply(action, now, self.config.dt, look_deg=self._look(action))
        else:
            self.dispatcher.release_all()
        self._wait_tick()
        late = self.clock() > self._deadline + (self.config.overrun_factor - 1.0) * self.config.dt
        if late:
            self._deadline = self.clock()  # skip, never catch up
        obs, reading, stale = self._observe()
        self.steps += 1
        console = focused and self._close_console()  # the policy's keys would be typing into it

        tracked = self.tracker.step(reading) if reading is not None else self.tracker.state
        if reading is None:
            # Nothing new read: carry the settled state, but no event is re-reported.
            tracked.points_event, tracked.points_delta, tracked.round_changed = "", 0, False
        signals = self.signals.step(tracked, action)
        self._count_shots(reading, action)
        shown = self.downed is not None and self.downed()
        if shown and self.clock() - self.t_start < self.config.down_grace_s:
            if self.press is not None and self.config.after_start_key and self.clock() >= self._next_clear:
                self.press(self.config.after_start_key)
                self._next_clear = self.clock() + 0.5
            shown = False
        self._down_run = self._down_run + 1 if shown else 0
        down = self._down_run >= self.config.down_steps and not signals.death
        if down:
            signals = replace(signals, death=True)
            self.signals.died = True  # the game-over count-up that follows is not a second death
        result = self.shaper(signals)
        self.return_ += result.reward

        now = self.clock()
        self._stale_since = (self._stale_since or now) if stale else None
        absent = reading is not None and reading.points_status == ABSENT
        self._seen_hud = self._seen_hud or (reading is not None and reading.points_status == OK)
        self._absent_since = (self._absent_since or now) if absent and self._seen_hud else None

        terminated = signals.death
        reason = ("down" if down else "death") if terminated else None
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
        bad = late or stale or not focused or console
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
            "shots": self.shots,
            "hits": self.signals.events["gain"],
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
