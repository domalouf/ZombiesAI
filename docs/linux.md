# Running the real game on Linux

The plan was written for a Windows gaming PC. It does not have to be one, and on balance Linux is the better
host for this project: the risk it removes is the one ranked first.

**Synthetic input (risk 1) stops being a ladder.** `PLAN.md` climbs from scancode `SendInput` through a
kernel virtual HID driver to a microcontroller pretending to be a mouse, because rung one often fails
silently on a 2008 DirectX 9 title. On Linux the middle rung is already in the kernel: `/dev/uinput` creates
a real input device that libinput, the compositor, XWayland, Wine and the game cannot distinguish from
hardware — because at the level they read it, it *is* hardware. `REL_X`/`REL_Y` are mouse counts, which is
the unit the action space is denominated in.

**Recording a human gets easier too.** `/dev/input/event*` reports the same counts, so the demo recorder is
a file read rather than a Win32 message loop.

**Capture (risk 2) is the piece that gets harder**, and it is the one to spike first. There is no
synchronous screen grab on Wayland — but Proton runs the game as an *XWayland* client, so the game is still
an X11 window with an XID, and an X11 window can be read in-process with MIT-SHM in a couple of
milliseconds. That is what `demos/x11_capture.py` does.

## Setup

### 1. The GPU and PyTorch

The RTX 5070 is Blackwell (sm_120) and needs a CUDA 12.8+ build. On Arch, the current `nvidia` package is
fine; check with `nvidia-smi`. **`pyproject.toml` selects the CUDA wheel index on Linux** — if you also work
on a CPU-only Linux box, it will pull the CUDA libraries there too (they install and fall back to CPU, just
large). Verify after `uv sync`:

```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` printing `False` on a machine with a working `nvidia-smi` means you have a CPU
build, not a driver problem.

### 2. Permission to read input devices

Reading `/dev/input/event*` needs the `input` group:

```sh
sudo usermod -aG input "$USER"   # log out and back in
```

### 3. Permission to create the virtual device

`/dev/uinput` is root-only by default. Load the module and hand the node to the same group:

```sh
sudo modprobe uinput
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
printf 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"\n' \
  | sudo tee /etc/udev/rules.d/99-zombiesai-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 4. Flat pointer acceleration for the virtual device

This one is easy to miss and it invalidates spike S4 when you do. libinput accelerates pointers by default,
so counts-to-degrees comes out as a curve instead of a line — which looks exactly like the game doing it. The
virtual device is named `zombiesai-virtual-input`; in Hyprland (`~/.config/hypr/hyprland.conf`):

```
device {
    name = zombiesai-virtual-input
    accel_profile = flat
    sensitivity = 0
}
```

Do the same for your real mouse while recording demos, and turn off in-game mouse smoothing and
acceleration. The agent and the human have to live in the same units.

### 5. The game

Run World at War through Proton. Solo Nazi Zombies is the good case; multiplayer is not what this needs.

- **In-game fullscreen at the monitor's resolution** (`r_fullscreen 1`, `r_mode` = the native size, 16:9).
  Not windowed: in a window Wine does not hold the pointer, and on Hyprland the view jumped on every mouse
  click -- with nothing of ours running, so it was the game, not the recorder. Proton's fullscreen is still an
  XWayland window, so capture finds and grabs it exactly as before. (The plan's advice to avoid exclusive
  fullscreen is about DXGI Desktop Duplication on Windows.) Keep it wholly on screen: X11 does not promise
  the contents of a window that is partly off the display, and a partly off-screen window fails with
  `BadMatch`.
- **A window on a hidden workspace can still be grabbed** -- XWayland hands back its last frame -- so the
  recorder treats a picture that stops changing as out of sight (`capture.FROZEN_READS`): `--wait` holds off
  until the picture moves, and frozen steps are flagged like any outage. WaW's pause menu freezes the picture
  too, so pauses mark themselves even when F8 is forgotten.
- Check the game really is an XWayland client and find its window:

  ```sh
  uv run python scripts/spike_capture.py --list-windows
  ```

  If nothing sensible appears, the game is running as a native Wayland client (rare for Proton, but
  `gamescope` can change this) — run it inside `gamescope` and capture gamescope's window, or force
  XWayland.
- `timescale` is `sv_cheats`-gated and reachable from the console either way. It is the single biggest
  data-rate lever in the plan; verify spawn counts and health per round before trusting a run made under it.

## The spike order

Do these before writing anything else. Each one is quick and each one can end the experiment early, which is
the point.

```sh
# S1 (does the engine see synthetic input?) and S4 (counts per degree), in one run.
uv run python scripts/calibrate_mouse.py --window "World at War" --full-turn

# S2 (capture rate) and S3 (input-to-pixels delay).
uv run python scripts/spike_capture.py --window "World at War" --latency --video /tmp/agentview.mp4

# Then record yourself playing, with the number S4 gave you. --wait holds the start until the game is
# on screen, then counts down 3 s (--countdown), so it can be launched from a terminal on another workspace.
uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20 --wait \
    --notes "varied play: camping, trains, deliberate bad positioning"
```

Capture reads the game's X11 window, and X refuses (`BadMatch`) to read one that is on a hidden workspace or
partly off-screen. Without `--wait` the recorder starts, and fails, the moment you press enter; with it, it
asks X every half second until the window is readable, and gives up after `--wait-timeout` (300 s). Input
from the wait and the countdown is thrown away rather than folded into the first decision. `--minutes`
counts recording only.

`calibrate_mouse.py --full-turn` measures counts per degree twice: once from an assumed field of view, and
once by turning all the way around until the view returns to where it started, which assumes nothing and
also tells you the real field of view. Put both numbers in `docs/spikes.md`; every look label and every turn
the agent makes depends on them.

`spike_capture.py --latency` is the closed-loop delay. It does not need to be small — it needs to be known
and stable. Set the sim's `latency_steps` to what it reports and confirm PPO still learns to aim there
before spending a weekend on the real game.

## Letting a policy play

`scripts/play_real.py` runs a trained policy against the game: capture, the model (about 0.4 ms a decision on
the GPU), and the virtual device, fifteen times a second (`realgame/play.py`).

```sh
uv run python scripts/play_real.py runs/bc_real1/bc.pt --dry-run --minutes 1   # everything but the input
uv run python scripts/play_real.py runs/bc_real1/bc.pt --minutes 3
```

Most of it is about when *not* to send input, because input goes to whatever window has focus:

- only while the game is the focused window, asked of Hyprland over its IPC socket every tick;
- only while the picture is live -- a frozen or lost frame (pause menu, hidden workspace) pauses it;
- every pause, standby and takeover also wipes the policy's frame history, so it never acts on a stack
  that reaches across the gap (the recording marks the same step with `FLAG_CLIP_START`);
- touching your own mouse or keyboard hands the controls back until you have been idle for 1.5 s;
- **F9 stops it**, and every exit path releases every key.

It starts in standby: **F7** in the game hands the AI the controls and takes them back, **F9** quits, and a
system sound says which (the fullscreen game hides notifications).

**Aim** goes through a mouse motor (`dispatch.MouseMotor`): the policy sets a turn *rate* from its
probability-weighted look, and a thread sends motion every 4 ms through two cascaded low-pass stages, so
speed and acceleration are continuous, with a soft dead zone that holds the view still through the policy's
idle drift. `--smoothness` is the stage time constant (default 0.08 s; lag is about twice it).

**The virtual device belongs to a session service** (`realgame/input_service.py`), started on first use and
left running; each player run borrows it over a Unix socket. Destroying a uinput device while the fullscreen
game has focus appears to drop its pointer lock -- after the first live runs quit, the human's clicks
jumped the view -- so the device is created once and never unplugged mid-session. Start the service before
launching the game if you can (`uv run python -m zombiesai.realgame.input_service &`); stop it with the game
closed. If a client dies mid-press the service releases every key it left down.

**A policy trained with `--audio` hears the game**, enabled automatically from the checkpoint: a background
thread reads the default sink's *monitor* through `parec` (never a microphone; `--audio-device` picks another
`.monitor`) into a 2 s ring buffer, and each acting tick takes the log-mel of the half second that ended when
the frame was grabbed — ~2 ms of numpy, never a wait on the pipe. The window ends at the newest sample that
has arrived (~5-15 ms before the frame) rather than zero-filling the rest. Pauses don't reset it: audio keeps
flowing through standby as it does in the recordings. A dead or silent-for-0.25 s stream switches the policy
to its vision-only mode (has-audio 0). `--deaf` forces that. The run summary prints how many ticks were heard,
the window lag and the cost per tick. Game audio from the play run is not recorded into its clip yet.

Each run is recorded to `runs/play/` like a demo (`label_source="play"`), with the steps nobody could play
(unfocused, frozen) flagged bad. Spike S1 -- does the engine see the virtual device at all -- is the first
live run's real test.

**Taking over is also teaching it** (HG-DAgger). Whenever you grab the controls, your input is decoded
exactly as a demo's -- the same `--bindings`, `--counts-per-degree` and hold rule, through the recorder's own
decoder -- and written as the label of those steps: the fix, in exactly the state the policy got itself into,
which is the data behavioural cloning lacks most. Each step says who acted in `labels.npz["actor"]`
(`clips.ACTOR_*`): the policy's own steps are kept to watch but never trained on; the step whose window
holds your first touch is yours; the idle ~1.5 s before it takes back over is you waiting, not playing, and
is left out (a shorter pause mid-takeover is kept). Holding a key counts as playing, so running with W held
does not hand the controls back. F7, F8 and F9 are never labels. The run's summary says how many correction
steps and seconds were captured; train on them beside the demos:

```sh
uv run python scripts/train_bc.py data/demos runs/play --out runs/bc2   # --correction-weight 2 by default
```

## What runs where

| Piece | Linux | Windows |
|---|---|---|
| Capture | `demos/x11_capture.py` (XWayland, MIT-SHM) | `dxcam` Desktop Duplication |
| Recording human input | `demos/evdev_input.py` | `demos/win32_input.py` |
| Synthetic input | `realgame/uinput.py` | not written yet (`SendInput` ladder) |
| Everything else | identical | identical |

`demos/capture.py` picks the backend by platform, so nothing above the adapter layer knows which it got.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `cannot open X display` | you are on a pure Wayland session with no XWayland; run the game under Proton or gamescope |
| `no window matching …` | the title differs (Wine sometimes uses the executable name) — run with `--list-windows` |
| `BadMatch` on grab | the window is partly off-screen or minimised; move it fully onto the display |
| capture p99 over 10 ms | the MIT-SHM path is not active (`describe()` says `xgetimage`), or the compositor is throttling an unfocused window |
| S1 fails, the view never moves | the game is not focused, or it enumerated input devices at startup — create the virtual device first, then launch the game |
| S4 R² below 0.98 | in-game mouse smoothing or acceleration is on, or libinput is still accelerating the virtual device |
| clean shutdown leaves keys held | something bypassed `ActionDispatcher.release_all()`; it is also what the watchdog must call first |
| `PermissionError` on `/dev/input/event*` or `/dev/uinput` | steps 2 and 3 above, and log out and back in for the group to take effect |
