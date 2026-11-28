"""A recorded demo as a video with the player's inputs drawn on it: the key display, burned in.

The live overlay (scripts/record_overlay.py) shows the keys while a session is played; this draws the same
panel (viz/input_layout.py) over the recording afterwards, from the raw input log, for watching a session back
or putting it in a video:

* **the picture** is the clip's own: the 640x360 scene it keeps beside the HUD boxes (packed or raw), scaled
  up, or for clips older than the scene crop, the policy's 128x72 frames blown up with whole pixels.
* **the keys, mouse buttons and wheel** light exactly when they went down and up in inputs.jsonl -- at the
  log's own timestamps, not rounded to decisions -- so a tap shorter than a decision still shows.
* **the look meter** shows each decision's mouse motion: which way, and how fast in degrees a second (the
  recording's counts-per-degree).
* **the top bar** says which clip, its notes and the time; while F8 had the session marked as not playing,
  an amber banner says so, as it did live.

The video's frame k is the clip's step k, shown from k/15 s: the frame the player saw at the start of the
decision, with the inputs given during it. An event at monotonic time t is drawn at t - t0, the recorder's own
clock (clip.json's t0_mono), so the keys line up with the frames exactly as the labels do. The game's audio, if
the clip has it, is aligned the same way (demos/audio.py `sample_at`).

The overlay is an ASS subtitle track burned in with ffmpeg's libass, as viz/overlay.py does for the policy's
films; nothing here draws a pixel itself.
"""

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from zombiesai import spec
from zombiesai.viz import input_layout as L

OUT_W, OUT_H = 1280, 720
FONT = "Liberation Sans"


def ass_colour(hex_rgb: str) -> str:
    """RGB hex -> ASS's &HBBGGRR&."""
    return f"&H{hex_rgb[4:6]}{hex_rgb[2:4]}{hex_rgb[0:2]}&"


def ass_alpha(opacity: float) -> str:
    return f"&H{int(round(255 * (1 - max(0.0, min(1.0, opacity))))):02X}&"


def _ts(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _event(start: float, end: float, text: str, layer: int = 0) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},Default,,0,0,0,,{text}"


def _plain(text: str) -> str:
    return str(text).replace("\\", "⧵").replace("{", "(").replace("}", ")").replace("\n", " ")


def rounded_path(w: float, h: float, r: float) -> str:
    """An ASS drawing of a w x h rectangle with corners of radius r (cubic beziers), from its top left."""
    r = min(r, w / 2, h / 2)
    k = r * 0.4477  # 1 - 0.5523: a bezier quarter circle's control points
    return (f"m {r:.1f} 0 l {w - r:.1f} 0 b {w - k:.1f} 0 {w:.1f} {k:.1f} {w:.1f} {r:.1f} "
            f"l {w:.1f} {h - r:.1f} b {w:.1f} {h - k:.1f} {w - k:.1f} {h:.1f} {w - r:.1f} {h:.1f} "
            f"l {r:.1f} {h:.1f} b {k:.1f} {h:.1f} 0 {h - k:.1f} 0 {h - r:.1f} "
            f"l 0 {r:.1f} b 0 {k:.1f} {k:.1f} 0 {r:.1f} 0")


def shape(x: float, y: float, path: str, fill: str, opacity: float, border: float = 0.0,
          border_colour: str = L.CHALK_DIM, border_opacity: float = 0.0) -> str:
    return (f"{{\\an7\\pos({x:.1f},{y:.1f})\\p1\\bord{border:.1f}\\shad0\\1c{ass_colour(fill)}\\1a{ass_alpha(opacity)}"
            f"\\3c{ass_colour(border_colour)}\\3a{ass_alpha(border_opacity)}}}{path}{{\\p0}}")


def label(x: float, y: float, size: float, colour: str, text: str, align: int = 5, bold: bool = True,
          opacity: float = 1.0) -> str:
    return (f"{{\\an{align}\\pos({x:.1f},{y:.1f})\\fn{FONT}\\fs{size:.1f}\\b{int(bold)}\\bord0\\shad0"
            f"\\1c{ass_colour(colour)}\\1a{ass_alpha(opacity)}}}{_plain(text)}")


def circle_path(r: float) -> str:
    k = r * 0.5523
    return (f"m 0 {-r:.1f} b {k:.1f} {-r:.1f} {r:.1f} {-k:.1f} {r:.1f} 0 b {r:.1f} {k:.1f} {k:.1f} {r:.1f} 0 {r:.1f} "
            f"b {-k:.1f} {r:.1f} {-r:.1f} {k:.1f} {-r:.1f} 0 b {-r:.1f} {-k:.1f} {-k:.1f} {-r:.1f} 0 {-r:.1f}")


# ------------------------------------------------------------------------------------------------ the timeline


def key_intervals(events: list[dict], t0: float, end_s: float) -> dict[str, list[tuple[float, float]]]:
    """code -> the [down, up) intervals in video seconds, for every key and mouse button in the log. A key
    still down at the end is held to `end_s`; a wheel notch lights for `WHEEL_FLASH_S`."""
    out: dict[str, list[tuple[float, float]]] = {}
    down: dict[str, float] = {}
    for event in events:
        if event.get("type") not in ("key", "button"):
            continue
        code, t = str(event.get("code", "")).lower(), float(event["t"]) - t0
        if code.startswith("wheel"):
            if event.get("down"):
                out.setdefault(code, []).append((t, t + L.WHEEL_FLASH_S))
            continue
        if event.get("down"):
            down.setdefault(code, t)  # a second down without an up (a missed release) keeps the first
        elif code in down:
            out.setdefault(code, []).append((down.pop(code), t))
    for code, t in down.items():
        out.setdefault(code, []).append((t, end_s))
    return {code: merge([(max(a, 0.0), min(b, end_s)) for a, b in spans if b > 0 and a < end_s])
            for code, spans in out.items()}


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        elif b > a:
            merged.append((a, b))
    return merged


def step_motion(events: list[dict], t0: float, n_steps: int, dt: float) -> np.ndarray:
    """(n_steps, 2) summed mouse counts per decision, [t0 + k*dt, t0 + (k+1)*dt) -- the recorder's windows."""
    motion = np.zeros((n_steps, 2))
    for event in events:
        if event.get("type") != "mouse":
            continue
        k = int(np.floor((float(event["t"]) - t0) / dt))
        if 0 <= k < n_steps:
            motion[k, 0] += event.get("dx", 0)
            motion[k, 1] += event.get("dy", 0)
    return motion


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) step ranges where `mask` is true."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


# ------------------------------------------------------------------------------------------------ the drawing


def build_ass(events: list[dict], t0: float, n_steps: int, *, bindings: dict, counts_per_degree: float,
              not_playing: np.ndarray | None = None, title: str = "", mark_key: str | None = "f8",
              width: int = OUT_W, height: int = OUT_H, start: float = 0.0, end: float | None = None,
              dt: float = 1.0 / spec.DECISION_HZ) -> str:
    """The input display over a clip of `n_steps` decisions, as an ASS script for a `width` x `height` video,
    for the stretch from `start` to `end` seconds, timed from `start`."""
    total = n_steps * dt
    end = total if end is None else min(end, total)
    k = height / 720
    u = 34 * k  # a key cap, in output pixels
    controls = L.control_of(bindings)
    lines: list[str] = []

    def at(a: float, b: float, text: str, layer: int) -> None:
        a, b = max(a, start) - start, min(b, end) - start
        if b > a:
            lines.append(_event(a, b, text, layer))

    # The panel: its backing and every cap unlit, for the whole video, then each cap lit while it was down.
    pad = 0.5 * u
    ox, oy = 24 * k + pad, height * 0.36 + pad + 0.75 * u
    panel_w, panel_h = L.PANEL_W * u + 2 * pad, (L.PANEL_H + 0.75) * u + 1.0 * u
    at(start, end, shape(ox - pad, oy - pad - 0.75 * u, rounded_path(panel_w, panel_h, 0.35 * u), L.INK, 0.45), 1)
    intervals = key_intervals(events, t0, total)
    for cap in L.ALL_CAPS:
        control = controls.get(cap.code)
        x, y, w, h = ox + cap.x * u, oy + cap.y * u, cap.w * u, cap.h * u
        cap_path = rounded_path(w, h, 0.16 * u)
        size = (0.42 if len(cap.label) <= 2 else 0.3) * u
        named = control is not None and cap.w >= 0.8

        def caption(ink: str, opacity: float) -> list[str]:
            if named:
                return [label(x + w / 2, y + h * 0.4, size, ink, cap.label, opacity=opacity),
                        label(x + w / 2, y + h * 0.78, 0.22 * u, ink, control, bold=False, opacity=opacity)]
            return [label(x + w / 2, y + h / 2, size, ink, cap.label, opacity=opacity)]

        at(start, end, shape(x, y, cap_path, L.INK, 0.55, 0.04 * u, L.CHALK if control else L.CHALK_DIM,
                             0.55 if control else 0.25), 2)
        for text in caption(L.CHALK if control else L.CHALK_DIM, 1.0 if control else 0.55):
            at(start, end, text, 3)
        lit = L.SANGUINE if control == "fire" else (L.AMBER if control else L.CHALK_DIM)
        for a, b in intervals.get(cap.code, []):
            at(a, b, shape(x, y, cap_path, lit, 0.95), 4)
            for text in caption(L.INK, 1.0):
                at(a, b, text, 5)

    # The look meter: its dial always, and each decision's motion as a needle.
    cx, cy, r = ox + L.LOOK_CX * u, oy + L.LOOK_CY * u, L.LOOK_R * u
    at(start, end, shape(cx, cy, circle_path(r), L.INK, 0.55, 0.04 * u, L.CHALK_DIM, 0.4), 2)
    at(start, end, shape(cx, cy, f"m {-0.85 * r:.1f} 0 l {0.85 * r:.1f} 0 l {0.85 * r:.1f} 1 l {-0.85 * r:.1f} 1",
                         L.CHALK_DIM, 0.2), 3)
    at(start, end, shape(cx, cy, f"m 0 {-0.85 * r:.1f} l 1 {-0.85 * r:.1f} l 1 {0.85 * r:.1f} l 0 {0.85 * r:.1f}",
                         L.CHALK_DIM, 0.2), 3)
    motion = step_motion(events, t0, n_steps, dt)
    first, last = int(start / dt), min(n_steps, int(np.ceil(end / dt)))
    for s in range(first, last):
        x, y, speed = L.look_vector(motion[s, 0], motion[s, 1], dt, counts_per_degree)
        a, b = s * dt, (s + 1) * dt
        caption = f"look {speed:.0f}°/s" if speed >= 1 else "look"
        at(a, b, label(cx, cy + r + 0.32 * u, 0.26 * u, L.CHALK_DIM, caption, bold=False), 4)
        if speed >= 1:
            tip_x, tip_y = x * r, y * r
            # The needle as a thin quad from the centre to the tip, and a dot at the tip.
            nx, ny = -tip_y, tip_x
            norm = max(1e-6, (nx * nx + ny * ny) ** 0.5)
            hw = 0.05 * u
            nx, ny = nx / norm * hw, ny / norm * hw
            needle = (f"m {nx:.1f} {ny:.1f} l {tip_x + nx:.1f} {tip_y + ny:.1f} l {tip_x - nx:.1f} {tip_y - ny:.1f} "
                      f"l {-nx:.1f} {-ny:.1f}")
            at(a, b, shape(cx, cy, needle, L.AMBER, 0.9), 5)
            at(a, b, shape(cx + tip_x, cy + tip_y, circle_path(0.14 * u), L.AMBER, 0.95), 6)

    # The top bar: the clip and the time, and the not-playing banner over it while F8 had it marked.
    bar_w, bar_h = 560 * k, 46 * k
    bx, by = (width - bar_w) / 2, 14 * k
    at(start, end, shape(bx, by, rounded_path(bar_w, bar_h, 10 * k), L.INK, 0.6), 1)
    head = title if len(title) <= 48 else title[:47] + "…"
    for second in range(int(start), int(np.ceil(end))):
        clock = f"{second // 60:02d}:{second % 60:02d}"
        at(second, second + 1, label(bx + 18 * k, by + bar_h / 2, 20 * k, L.CHALK_DIM, head, align=4, bold=False), 2)
        at(second, second + 1, label(bx + bar_w - 18 * k, by + bar_h / 2, 24 * k, L.CHALK, clock, align=6), 2)
    if not_playing is not None:
        key = (mark_key or "f8").upper()
        for a, b in runs(np.asarray(not_playing, bool)):
            ta, tb = a * dt, b * dt
            at(ta, tb, shape(bx - 40 * k, by, rounded_path(bar_w + 80 * k, bar_h, 10 * k), L.AMBER, 0.94), 7)
            at(ta, tb, label(width / 2, by + bar_h / 2, 22 * k, L.INK,
                             f"NOT PLAYING ({key}) · left out of training"), 8)

    head_lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
                  "ScaledBorderAndShadow: yes", "WrapStyle: 2", "", "[V4+ Styles]",
                  "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                  "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                  "Alignment, MarginL, MarginR, MarginV, Encoding",
                  f"Style: Default,{FONT},{20 * k:.0f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
                  "1,0,0,7,0,0,0,1", "", "[Events]",
                  "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    return "\n".join(head_lines + lines) + "\n"


# ------------------------------------------------------------------------------------------------ the render


def _picture(clip):
    """(array-like of (T, h, w, 3) RGB frames, ffmpeg scale filter): the scene crop if the clip kept one,
    else the policy frames, blown up with whole pixels."""
    scene = clip.hud("scene") if "scene" in clip.hud_regions else None
    if scene is not None:
        return scene, f"scale={OUT_W}:{OUT_H}:flags=lanczos"
    return clip.frames, f"scale={OUT_W}:{OUT_H}:flags=neighbor"


def _write_audio(clip, t0: float, start: float, end: float, path: Path) -> bool:
    """The clip's audio for [start, end) video seconds, aligned to the frames, as a WAV. False if it has none."""
    meta = clip.manifest.get("audio")
    if not isinstance(meta, dict):
        return False
    from zombiesai.demos.audio import load_audio

    audio = load_audio(clip.path, meta)
    if audio is None:
        return False
    with wave.open(str(path), "wb") as out:
        out.setnchannels(audio.channels)
        out.setsampwidth(2)
        out.setframerate(audio.rate)
        block = 10 * audio.rate
        n = int(round((end - start) * audio.rate))
        for i in range(0, n, block):
            m = min(block, n - i)
            s = audio.sample_at(t0 + start + (i + np.arange(m)) / audio.rate)
            chunk = np.zeros((m, audio.channels), np.int16)
            ok = s >= 0
            chunk[ok] = audio.samples[s[ok] - audio.first_sample]
            out.writeframes(chunk.astype("<i2").tobytes())
    return True


def clip_t0(clip, events: list[dict]) -> float:
    manifest = clip.manifest
    t0 = manifest.get("summary", {}).get("t0_mono", manifest.get("t0_mono"))
    if t0 is None and isinstance(manifest.get("audio"), dict):
        t0 = manifest["audio"].get("t0_mono")
    if t0 is None:
        if not events:
            raise ValueError(f"{clip.path} has no t0_mono and no input log to place one")
        t0 = min(e["t"] for e in events)
    return float(t0)


def render(clip_dir: str | Path, out: str | Path | None = None, *, start: float = 0.0, end: float | None = None,
           audio: bool = True, say=print) -> Path:
    """Render `clip_dir` with its inputs drawn on, to `out` (default <clip>/inputs.mp4)."""
    from zombiesai.demos.clips import FLAG_NOT_PLAYING, load_clip
    from zombiesai.demos.inputs import DEFAULT_BINDINGS, read_log
    from zombiesai.rl.best_episode import encoder_args

    clip_dir = Path(clip_dir)
    clip = load_clip(clip_dir, any_label_spec=True)
    log = clip_dir / "inputs.jsonl"
    if not log.exists():
        raise ValueError(f"{clip_dir} has no inputs.jsonl: only recorded demos have inputs to draw")
    events = read_log(log)
    t0 = clip_t0(clip, events)
    recorder = clip.manifest.get("config", {}).get("recorder", {}) if isinstance(clip.manifest.get("config"), dict) else {}
    input_config = recorder.get("input", {}) if isinstance(recorder, dict) else {}
    bindings = input_config.get("bindings") or dict(DEFAULT_BINDINGS)
    counts_per_degree = float(input_config.get("counts_per_degree") or 1.0)
    mark_key = clip.manifest.get("summary", {}).get("mark_key", input_config.get("mark_key", "f8"))
    dt = 1.0 / spec.DECISION_HZ
    n = clip.n_steps
    end = n * dt if end is None else min(end, n * dt)
    if end <= start:
        raise ValueError(f"nothing to render: {start:.1f}s to {end:.1f}s of a {n * dt:.1f}s clip")
    notes = clip.manifest.get("summary", {}).get("notes") or ""
    title = clip_dir.name + (f" · {notes}" if notes else "")
    not_playing = (np.asarray(clip.flags) & FLAG_NOT_PLAYING) != 0
    out = Path(out) if out else clip_dir / "inputs.mp4"
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(build_ass(events, t0, n, bindings=bindings, counts_per_degree=counts_per_degree,
                                  not_playing=not_playing, title=title, mark_key=mark_key, start=start, end=end))
    wav = out.with_suffix(".wav")
    has_audio = audio and _write_audio(clip, t0, start, end, wav)
    frames, scale = _picture(clip)
    first, last = int(round(start / dt)), int(round(end / dt))
    h, w = frames.shape[1:3]
    say(f"{clip_dir.name}: steps {first}-{last} ({(last - first) * dt:.0f}s), picture {w}x{h} -> {OUT_W}x{OUT_H}"
        + (", with audio" if has_audio else ""))
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(spec.DECISION_HZ), "-i", "-"]
    if has_audio:
        command += ["-i", str(wav)]
    command += ["-vf", f"{scale},ass={ass_path}", *encoder_args(OUT_W, OUT_H), "-pix_fmt", "yuv420p"]
    command += (["-c:a", "aac", "-b:a", "160k", "-shortest"] if has_audio else ["-an"])
    command += ["-movflags", "+faststart", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        chunk = 450
        for a in range(first, last, chunk):
            block = np.ascontiguousarray(np.asarray(frames[a:min(a + chunk, last)], dtype=np.uint8))
            process.stdin.write(block.tobytes())
        process.stdin.close()
        err = process.stderr.read()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited {process.returncode}: {err.decode(errors='replace')[-400:]}")
    finally:
        if process.poll() is None:
            process.kill()
        ass_path.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)
    return out


def describe(path: Path) -> dict:
    """The rendered video's size and length, for the script's last line."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
    data = json.loads(probe.stdout)
    stream = next((s for s in data.get("streams", []) if "width" in s), {})
    return {"width": stream.get("width"), "height": stream.get("height"),
            "seconds": float(data.get("format", {}).get("duration", 0.0))}
