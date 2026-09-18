"""Turning a factored action into input transitions: what to press, what to release, how to move the mouse.

This is the half of synthetic input that has nothing to do with the operating system, which is why it lives
apart from `uinput.py` and is tested without a kernel device. Three rules come straight from PLAN.md:

* **`fire`, `ads` and `sprint` are hold states, not edges.** The dispatcher diffs what the policy asked for
  against what is currently held and emits only the difference -- so `fire=1` on two consecutive steps is one
  press, not two, and a policy that stops asking for it releases the button.
* **Mouse aiming goes out as several sub-moves spread across the tick.** Old DirectX 9 raw-input paths clamp
  or drop a single large delta, and smooth motion gives a better-behaved counts-to-degrees relationship --
  which is the relationship spike S4 fits and the whole action space is defined in.
* **Counts are integers and degrees are not.** The rounding residue is carried, within the tick and across
  ticks, so a long run of 2-degree turns does not quietly lose a degree a second to truncation.

The unit is the mouse count, the same unit `demos/evdev_input.py` reads off the human's mouse. That symmetry
is what makes a demonstration and a policy rollout the same kind of thing.
"""

import time
from dataclasses import dataclass, field

from zombiesai import spec
from zombiesai.demos.inputs import DEFAULT_BINDINGS, inverse_bindings

HOLD_CONTROLS = ("forward", "back", "left", "right", "fire", "ads", "sprint")
TAP_CONTROLS = tuple(b for b in spec.BUTTONS if b != "none")


@dataclass(frozen=True)
class DispatchConfig:
    """`counts_per_degree` is spike S4's number for the sensitivity the game is set to. Everything else here
    is timing: how finely to spread a turn, and how long a tap has to last for the engine to see it."""

    counts_per_degree: float = 1.0
    submoves: int = 3
    tap_hold_s: float = 0.05  # a press shorter than a frame at 60 fps can be missed entirely
    bindings: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))

    @property
    def codes(self) -> dict[str, str]:
        return inverse_bindings(self.bindings)


class FakeSink:
    """Records what would have been sent, in the log format the quantizer reads.

    It is the plan's `FakeInput`, and it closes the loop: dispatch an action through it, feed the events to
    `demos.inputs.quantize`, and the action that comes back must be the one that went in.
    """

    def __init__(self):
        self.events: list[dict] = []
        self.syncs = 0

    def key(self, code: str, down: bool, t: float) -> None:
        kind = "button" if code.startswith("mouse") else "key"
        self.events.append({"t": t, "type": kind, "code": code, "down": down})

    def move(self, dx: int, dy: int, t: float) -> None:
        self.events.append({"t": t, "type": "mouse", "dx": int(dx), "dy": int(dy)})

    def sync(self) -> None:
        self.syncs += 1

    def close(self) -> None:
        pass


class ActionDispatcher:
    """Holds the current input state and moves it to whatever the policy asked for.

    `apply` sends the key and button transitions immediately and schedules the tick's mouse sub-moves;
    `pump` sends the ones that have come due and releases finished taps. The loop calls `pump` in the slack
    it has anyway, so no part of this sleeps on the hot path.
    """

    def __init__(self, sink, config: DispatchConfig | None = None, *, clock=time.monotonic):
        self.sink = sink
        self.config = config or DispatchConfig()
        self.codes = self.config.codes  # inverted once, not per key event on the hot path
        self.clock = clock
        self.held: set[str] = set()
        self._tap_until: dict[str, float] = {}
        self._pending: list[tuple[float, int, int]] = []  # (due, dx, dy)
        self._residue = 0.0 + 0.0j  # sub-count rounding carried between ticks, x in real, y in imag

    def apply(self, action, now: float | None = None, dt: float = 1.0 / spec.DECISION_HZ) -> None:
        values = spec.action_tuple(action)
        now = self.clock() if now is None else now
        wanted = self._wanted(values)
        for control in sorted(self.held - wanted):
            self._send_key(control, False, now)
        for control in sorted(wanted - self.held):
            self._send_key(control, True, now)

        button = spec.BUTTONS[values[spec.BUTTON]]
        if button != "none":
            # A tap is press-now, release-later: the engine reads an edge, and the release is pumped rather
            # than slept through. Re-pressing a button already held down would be no edge at all.
            if button in self.held:
                self._send_key(button, False, now)
            self._send_key(button, True, now)
            self._tap_until[button] = now + self.config.tap_hold_s

        self._schedule_move(values, now, dt)
        self.pump(now)

    def _wanted(self, values: tuple[int, ...]) -> set[str]:
        forward = spec.FORWARD_VALUES[values[spec.FORWARD]]
        strafe = spec.STRAFE_VALUES[values[spec.STRAFE]]
        wanted = set()
        if forward > 0:
            wanted.add("forward")
        elif forward < 0:
            wanted.add("back")
        if strafe > 0:
            wanted.add("right")
        elif strafe < 0:
            wanted.add("left")
        for head, control in ((spec.FIRE, "fire"), (spec.ADS, "ads"), (spec.SPRINT, "sprint")):
            if values[head]:
                wanted.add(control)
        return wanted

    def _schedule_move(self, values: tuple[int, ...], now: float, dt: float) -> None:
        degrees = complex(spec.YAW_BINS_DEG[values[spec.YAW]], -spec.PITCH_BINS_DEG[values[spec.PITCH]])
        # Screen-down is positive dy on a mouse, and positive pitch looks up, so the sign flips here -- the
        # one place in the codebase it does, matching the same flip in the recorder's quantizer.
        target = degrees * self.config.counts_per_degree + self._residue
        n = max(1, self.config.submoves)
        sent = 0 + 0j
        for i in range(n):
            step = target * (i + 1) / n
            chunk = complex(round(step.real - sent.real), round(step.imag - sent.imag))
            sent += chunk
            if chunk:
                self._pending.append((now + i * dt / n, int(chunk.real), int(chunk.imag)))
        self._residue = target - sent

    def pump(self, now: float | None = None) -> None:
        """Send every sub-move that has come due and release every tap whose hold time has elapsed."""
        now = self.clock() if now is None else now
        due = [move for move in self._pending if move[0] <= now]
        if due:
            self._pending = [move for move in self._pending if move[0] > now]
            for _, dx, dy in due:
                self.sink.move(dx, dy, now)
            self.sink.sync()
        for control, until in list(self._tap_until.items()):
            if until <= now:
                self._send_key(control, False, now)
                del self._tap_until[control]

    def pump_until(self, deadline: float, poll_s: float = 0.002) -> None:
        """Standalone pumping, for scripts with nothing else to do inside the tick."""
        while True:
            now = self.clock()
            self.pump(now)
            if now >= deadline:
                return
            time.sleep(min(poll_s, max(0.0, deadline - now)))

    def flush(self) -> None:
        """Send every scheduled sub-move at once, ignoring its due time."""
        if self._pending:
            now = self.clock()
            for _, dx, dy in self._pending:
                self.sink.move(dx, dy, now)
            self._pending.clear()
            self.sink.sync()

    def release_all(self) -> None:
        """Drop everything: the focus guard, the end of an episode, and the last thing any crash handler
        should do. A key left held after the process dies is a keyboard nobody can use."""
        now = self.clock()
        self._pending.clear()
        for control in sorted(self.held):
            self._send_key(control, False, now)
        for control in list(self._tap_until):
            if control in self.held:
                self._send_key(control, False, now)
            del self._tap_until[control]
        self._residue = 0 + 0j

    def _send_key(self, control: str, down: bool, t: float) -> None:
        code = self.codes.get(control)
        if code is None:
            raise KeyError(f"no binding for {control!r}; give DispatchConfig a bindings map that covers it")
        self.sink.key(code, down, t)
        self.sink.sync()
        self.held.add(control) if down else self.held.discard(control)

    def close(self) -> None:
        self.release_all()
        self.sink.close()
