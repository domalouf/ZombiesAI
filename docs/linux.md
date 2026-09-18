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

- **Borderless windowed**, and keep the window fully on screen and unobscured. X11 does not promise the
  contents of a window that is partly off the display, and a partly off-screen window fails capture with
  `BadMatch`.
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

# Then record yourself playing, with the number S4 gave you.
uv run python scripts/record_demo.py --source screen --counts-per-degree 6.4 --minutes 20 \
    --notes "varied play: camping, trains, deliberate bad positioning"
```

`calibrate_mouse.py --full-turn` measures counts per degree twice: once from an assumed field of view, and
once by turning all the way around until the view returns to where it started, which assumes nothing and
also tells you the real field of view. Put both numbers in `docs/spikes.md`; every look label and every turn
the agent makes depends on them.

`spike_capture.py --latency` is the closed-loop delay. It does not need to be small — it needs to be known
and stable. Set the sim's `latency_steps` to what it reports and confirm PPO still learns to aim there
before spending a weekend on the real game.

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
