"""Every game at once, as one film: the instances' pictures tiled in a grid and encoded together.

The opening shot of a training video is the agent playing many games at the same time, so this reads every
running instance's X server (as the live viewers do, realgame/viewer.py, and like them without sending
anything back: the games cannot tell), cuts each picture to every n-th pixel so the grid fits `max_width`,
and tiles them in display order, `fps` times a second, into an H.264 film.

A grab here is on a clock, not on the instance's damage, so on a `-shm` instance (whose frames arrive in four
strips, realgame/viewer.py) a tile can now and then show a frame half drawn. At tile size it is hard to see.
Each grab costs a few milliseconds of one core per game; the actors' own grabs are not touched.
"""

import math
import subprocess
import time
from pathlib import Path

import numpy as np

GAP = 4  # dark pixels between the tiles


def layout(n: int, width: int, height: int, max_width: int) -> tuple[int, int, int]:
    """(columns, rows, step) for `n` pictures of `width` x `height`: the most nearly 16:9 grid, and the
    smallest whole step that keeps it within `max_width`."""
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = cols * width / (rows * height)
        score = abs(math.log(aspect / (16 / 9)))
        if best is None or score < best[0]:
            best = (score, cols, rows)
    _, cols, rows = best
    step = 1
    while cols * (width // step) + (cols - 1) * GAP > max_width:
        step += 1
    return cols, rows, step


def mosaic(pictures: list[np.ndarray | None], cols: int, rows: int, tile_h: int, tile_w: int) -> np.ndarray:
    """BGRX pictures, already tile-sized (or None: a game whose picture could not be read, drawn dark), tiled
    in reading order into one BGRX frame with even sides (yuv420p needs them)."""
    h = rows * tile_h + (rows - 1) * GAP
    w = cols * tile_w + (cols - 1) * GAP
    out = np.zeros((h + h % 2, w + w % 2, 4), np.uint8)
    for i, picture in enumerate(pictures):
        if picture is None:
            continue
        r, c = divmod(i, cols)
        y, x = r * (tile_h + GAP), c * (tile_w + GAP)
        ph, pw = min(tile_h, picture.shape[0]), min(tile_w, picture.shape[1])
        out[y:y + ph, x:x + pw] = picture[:ph, :pw]
    return out


def record(out: Path, seconds: float, *, fps: float = 15.0, max_width: int = 1920, displays=None,
           say=print) -> Path:
    """Film every running instance (or those in `displays`, as ":60") for `seconds` into `out`."""
    from zombiesai.demos.x11_capture import X11Grabber
    from zombiesai.realgame.viewer import running_screens
    from zombiesai.rl.best_episode import encoder_args

    screens = [s for s in running_screens() if displays is None or s.display in displays]
    if not screens:
        raise SystemExit("no game instances are running (scripts/instances.py up, or ./zai start)")
    grabbers = [X11Grabber(display=s.display) for s in screens]
    width, height = grabbers[0].size
    cols, rows, step = layout(len(screens), width, height, max_width)
    tile_h, tile_w = -(-height // step), -(-width // step)
    frame = mosaic([None] * len(screens), cols, rows, tile_h, tile_w)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    say(f"filming {len(screens)} games ({', '.join(s.display for s in screens)}) as a {cols}x{rows} grid, "
        f"{frame.shape[1]}x{frame.shape[0]}, for {seconds:g} s -> {out}")
    encoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr0",
         "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-r", f"{fps:g}", "-i", "-",
         *encoder_args(frame.shape[1], frame.shape[0]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    t0 = time.monotonic()
    k = late = 0
    try:
        while k < seconds * fps:
            pictures = []
            for grabber in grabbers:
                try:
                    pictures.append(np.ascontiguousarray(grabber.grab_bgrx()[::step, ::step]))
                except Exception:  # noqa: BLE001 -- a game restarting is a dark tile, not the end of the film
                    pictures.append(None)
            encoder.stdin.write(mosaic(pictures, cols, rows, tile_h, tile_w).data)
            k += 1
            wait = t0 + k / fps - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                late += 1
    except KeyboardInterrupt:
        say("stopped early; keeping what was filmed")
    finally:
        encoder.stdin.close()
        encoder.wait()
        for grabber in grabbers:
            grabber.close()
    if late > k * 0.1:
        say(f"{late} of {k} frames were late: the film runs slower than the games did; try a lower --fps")
    return out
