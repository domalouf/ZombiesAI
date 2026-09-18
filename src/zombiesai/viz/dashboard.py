"""Read every run under runs/ and write a self-contained HTML dashboard of how training is going.

Deliberately stdlib-only. A training run's log is JSON on disk; reading it should not need torch,
numpy or gymnasium, so this can be pointed at a runs/ directory copied off the training box.
"""

import json
import math
import re
import time
from base64 import b64encode
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
FONTS = Path(__file__).with_name("fonts")

# The gates training is judged against, quoted from PLAN.md so the dashboard argues from the plan and
# not from taste. The shares are M7's anti-hacking gate for T1; the returns are M2's correctness checks.
GATES = {
    "max_term_share": {"limit": 0.60, "short": "largest term", "label": "no single reward term above 60% of return"},
    "repair_share": {"limit": 0.25, "short": "repairs", "label": "repairs below 25% of points earned"},
}
# Mirrors PRESETS in rl/envs.py, which is the source of truth but imports gymnasium.
SOLVED_AT = {"cartpole": 475.0, "lunarlander": 200.0}

# The validated dark categorical slots, in their fixed order. A run keeps its slot for the life of the
# page: hiding one must never repaint the others.
SERIES_COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")

# PPO logs one row per update; BC and the IDM log one per epoch. Everything below is (path, label, hint).
PPO_SERIES = (
    ("return_mean", "Return", "mean over the last 100 finished episodes"),
    ("round_reached_mean", "Round reached", "how far into the game it gets, mean of the last 100"),
    ("length_mean", "Episode length", "steps before it dies or hits the cap"),
    ("entropy", "Policy entropy", "nats; falling means the policy is committing"),
    ("approx_kl", "Approx. KL", "how far each update moves the policy"),
    ("clipfrac", "Clip fraction", "share of the batch the surrogate clipped"),
    ("explained_variance", "Explained variance", "1 is a perfect critic, 0 is predicting the mean"),
    ("value_loss", "Value loss", ""),
    ("policy_loss", "Policy loss", ""),
    ("max_term_share_mean", "Largest reward term", "share of return from one term; the gate is 0.60"),
    ("repair_share_mean", "Repair share", "share of points from barrier repairs; the gate is 0.25"),
    ("sps", "Steps/s", "throughput, not learning"),
)
BC_SERIES = (
    ("val.accuracy.mean_balanced", "Val balanced accuracy", "mean per-class recall across the action heads"),
    ("loss.policy", "Policy loss", "training, per epoch"),
    ("loss.value", "Value loss", ""),
)
IDM_SERIES = (
    ("val.mean_balanced", "Val balanced accuracy", "how well it recovers the actions between two frames"),
    ("loss", "Loss", "training, per epoch"),
)
KIND_SERIES = {"ppo": PPO_SERIES, "bc": BC_SERIES, "idm": IDM_SERIES}
# Charts are scoped to one of these at a time. Plotting CartPole's return beside NachtSim's would put two
# different units on one axis, and a 500k-step run beside a 20M-step one squashes the short one to nothing.
GROUP_LABEL = {"bc": "behavioural cloning", "idm": "inverse dynamics"}
# The one each kind leads with, and the x it is plotted against.
HEADLINE = {"ppo": "return_mean", "bc": "val.accuracy.mean_balanced", "idm": "val.mean_balanced"}
X_KEY = {"ppo": "step", "bc": "epoch", "idm": "epoch"}


def _dig(row: dict, path: str):
    """row['val']['accuracy']['mean_balanced'] for the path 'val.accuracy.mean_balanced'."""
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) and math.isfinite(cur) else None


def read_rows(path: Path) -> list[dict]:
    """Every parseable line. A live trainer's last line can be half-written, and that is not an error."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def classify(run_dir: Path, config: dict, rows: list[dict]) -> str:
    """Which trainer wrote this run. The checkpoint's filename is the surest signal; the rows are the fallback."""
    for name, kind in (("bc.pt", "bc"), ("idm.pt", "idm"), ("checkpoint.pt", "ppo")):
        if (run_dir / name).exists():
            return kind
    if "obs_keys" in config:
        return "bc"
    if any("update" in r for r in rows):
        return "ppo"
    if any(isinstance(r.get("loss"), dict) for r in rows):  # BC logs its loss split by head, the IDM a float
        return "bc"
    if any("epoch" in r for r in rows):
        return "idm"
    return "ppo" if "total_steps" in config else "unknown"


def bucket(xs: list[float], ys: list[float], buckets: int) -> dict:
    """Aggregate a per-update series into at most `buckets` points, keeping each bucket's spread.

    Plotting every update of a 20M-step run is a megabyte of noise. The mean is the line; min and max
    become the band behind it, so smoothing never hides how wide the scatter actually was.
    """
    n = len(xs)
    if n == 0:
        return {"x": [], "y": [], "lo": [], "hi": []}
    buckets = max(1, min(buckets, n))
    out = {"x": [], "y": [], "lo": [], "hi": []}
    for b in range(buckets):
        start, stop = n * b // buckets, n * (b + 1) // buckets
        if stop <= start:
            continue
        chunk = ys[start:stop]
        out["x"].append(_round(xs[stop - 1]))
        out["y"].append(_round(sum(chunk) / len(chunk), 4))
        out["lo"].append(_round(min(chunk), 4))
        out["hi"].append(_round(max(chunk), 4))
    return out


def _round(v: float, nd: int = 4) -> float:
    r = round(float(v), nd)
    return int(r) if r == int(r) and abs(r) < 1e15 else r


def trend(xs: list[float], ys: list[float], tail: float = 0.34) -> dict | None:
    """Least squares over the last `tail` of the run, with a verdict the slope alone cannot give.

    A slope is only news if it outruns the scatter it was fitted through, so the verdict compares the
    total rise across the window against the residual spread rather than testing the slope against zero.
    """
    n = len(ys)
    if n < 6:
        return None
    lo = max(0, n - max(4, int(round(n * tail))))
    x, y = xs[lo:], ys[lo:]
    m = len(x)
    mx, my = sum(x) / m, sum(y) / m
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx == 0:
        return None
    slope = sum((x[i] - mx) * (y[i] - my) for i in range(m)) / sxx
    intercept = my - slope * mx
    resid = [y[i] - (intercept + slope * x[i]) for i in range(m)]
    spread = math.sqrt(sum(r * r for r in resid) / m)
    span = x[-1] - x[0]
    rise = slope * span
    if abs(rise) <= spread:
        verdict = "flat"
    else:
        verdict = "climbing" if rise > 0 else "falling"
    return {"slope": slope, "rise": _round(rise, 3), "spread": _round(spread, 3), "span": span, "verdict": verdict}


def _fmt(v: float, signed: bool = False) -> str:
    """Enough digits to be worth printing: a spread of 0.006 must not read as 0.00."""
    a = abs(v)
    nd = 2 if a >= 1 else (3 if a >= 0.1 else 4)
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _span_text(span: float, x_key: str) -> str:
    """'0.0 M steps' is not a span. Say it in whatever unit the number is actually big in."""
    if x_key != "step":
        return f"{span:,.0f} epochs"
    if span >= 1e6:
        return f"{span / 1e6:,.1f}M steps"
    if span >= 1e4:
        return f"{span / 1e3:,.0f}K steps"
    return f"{span:,.0f} steps"


def _num(v, default=None):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _progress(kind: str, config: dict, rows: list[dict]) -> tuple[float | None, str]:
    """Fraction done and the text that says it, from whatever the trainer's config promised."""
    last = rows[-1] if rows else {}
    if kind == "ppo":
        total, per = _num(config.get("total_steps")), None
        if _num(config.get("rollout_steps")) and _num(config.get("num_envs")):
            per = config["rollout_steps"] * config["num_envs"]
        step = _num(last.get("step"))
        if total and per and _num(last.get("update")):
            updates = total // per
            done = min(1.0, last["update"] / updates) if updates else None
            return done, f"update {last['update']:,} of {updates:,}"
        if total and step:
            return min(1.0, step / total), f"{step:,} of {total:,} steps"
        return None, f"{step:,} steps" if step else "no updates logged"
    epochs, epoch = _num(config.get("epochs")), _num(last.get("epoch"))
    if epochs and epoch:
        return min(1.0, epoch / epochs), f"epoch {epoch} of {epochs}"
    return None, f"epoch {epoch}" if epoch else "no epochs logged"


def _elapsed(kind: str, rows: list[dict]) -> float | None:
    """Seconds of training. PPO logs no clock, but steps/s and the step count are one between them."""
    if not rows:
        return None
    last = rows[-1]
    if kind == "ppo":
        sps, step = _num(last.get("sps")), _num(last.get("step"))
        return step / sps if sps and step else None
    return _num(last.get("seconds"))


def read_run(run_dir: Path, buckets: int = 160, stale_after: float = 600.0, now: float | None = None) -> dict | None:
    """One run's config, curves, headline numbers and verdicts. None if the directory holds no metrics."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    now = time.time() if now is None else now
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
    rows = read_rows(metrics)
    kind = classify(run_dir, config, rows)
    x_key = X_KEY.get(kind, "step")

    series = {}
    for path, label, hint in KIND_SERIES.get(kind, ()):
        xs, ys = [], []
        for row in rows:
            x, y = _dig(row, x_key), _dig(row, path)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if not ys:
            continue
        entry = bucket(xs, ys, buckets)
        entry.update(
            label=label,
            hint=hint,
            n=len(ys),
            last=_round(ys[-1], 4),
            best=_round(max(ys), 4),
            worst=_round(min(ys), 4),
            trend=trend(xs, ys),
        )
        series[path] = entry

    updated = metrics.stat().st_mtime
    progress, progress_text = _progress(kind, config, rows)
    done = progress is not None and progress >= 0.999
    status = "done" if done else ("running" if now - updated <= stale_after else "stopped")
    elapsed = _elapsed(kind, rows)
    eta = None
    if status == "running" and progress and elapsed and progress < 1:
        eta = elapsed * (1 - progress) / progress

    env = config.get("env") or config.get("preset") or ("pixels" if kind != "ppo" else None)
    group = env if kind == "ppo" and env else kind
    run = {
        "name": run_dir.name,
        "kind": kind,
        "env": env,
        "group": group,
        "group_label": GROUP_LABEL.get(group, group),
        "status": status,
        "updated": updated,
        "age_s": _round(now - updated, 1),
        "progress": None if progress is None else _round(progress, 4),
        "progress_text": progress_text,
        "elapsed_s": None if elapsed is None else _round(elapsed, 1),
        "eta_s": None if eta is None else _round(eta, 1),
        "rows": len(rows),
        "spec_version": config.get("spec_version"),
        "config": config,
        "headline": HEADLINE.get(kind),
        "x_key": x_key,
        "series": series,
        "last_row": rows[-1] if rows else {},
    }
    run["stats"] = _stats(run)
    run["notes"] = _notes(run)
    return run


def _stats(run: dict) -> dict:
    """The few numbers that belong above the fold, per kind."""
    last, series = run["last_row"], run["series"]
    out = {
        "steps": _num(last.get("step")),
        "episodes": _num(last.get("episodes")),
        "sps": _num(last.get("sps")),
        "epoch": _num(last.get("epoch")),
        "train_steps": _num(run["config"].get("train_steps")),
    }
    for key in ("return_mean", "round_reached_mean", "val.accuracy.mean_balanced", "val.mean_balanced"):
        if key in series:
            out[key] = {"last": series[key]["last"], "best": series[key]["best"]}
    solved_at = SOLVED_AT.get(run["env"] or "")
    if solved_at is not None and "return_mean" in series:
        out["solved_at"] = solved_at
        # PPO's own rule: the mean over the trailing 100 episodes, which is exactly what return_mean is.
        out["solved"] = series["return_mean"]["last"] >= solved_at
    return out


def _notes(run: dict) -> list[dict]:
    """Verdicts, each one tied to a number on the page. Levels: good, warning, serious, critical, info."""
    notes, series, stats = [], run["series"], run["stats"]
    head = series.get(run["headline"] or "")
    if head and head.get("trend"):
        t = head["trend"]
        word = {"climbing": "good", "flat": "warning", "falling": "serious"}[t["verdict"]]
        notes.append(
            {
                "level": word,
                "text": f"{head['label'].lower()} is {t['verdict']}: {_fmt(t['rise'], signed=True)} over the last "
                f"{_span_text(t['span'], run['x_key'])}, against a spread of {_fmt(t['spread'])}",
            }
        )
    if "solved" in stats:
        notes.append(
            {
                "level": "good" if stats["solved"] else "warning",
                "text": f"{run['env']} counts as solved at {stats['solved_at']:g}; "
                f"the last 100 episodes average {series['return_mean']['last']:.1f}",
            }
        )
    pairs = (("max_term_share_mean", "max_term_share"), ("repair_share_mean", "repair_share"))
    gates = [(key, gate) for key, gate in pairs if key in series]
    broken = [(k, g) for k, g in gates if series[k]["last"] >= GATES[g]["limit"]]
    for key, gate in broken:
        notes.append(
            {
                "level": "critical",
                "text": f"anti-hacking gate broken — {GATES[gate]['label']}: now {series[key]['last']:.2f}, "
                f"over the {GATES[gate]['limit']:.2f} limit",
            }
        )
    if gates and not broken:
        # One line while they hold: a passing gate is worth saying once, not once per term.
        held = ", ".join(
            f"{GATES[g]['short']} {series[k]['last']:.2f} against {GATES[g]['limit']:.2f}" for k, g in gates
        )
        notes.append({"level": "good", "text": f"anti-hacking gates hold — {held}"})
    ev = series.get("explained_variance")
    if ev is not None and ev["last"] < 0:
        notes.append(
            {
                "level": "serious",
                "text": f"the critic explains less variance than predicting the mean would "
                f"({ev['last']:.2f}); its value targets are not being fitted",
            }
        )
    return notes


def collect_runs(root: Path, buckets: int = 160, stale_after: float = 600.0, now: float | None = None) -> list[dict]:
    """Every run directory under `root`, newest activity first, each holding the colour slot it is given here."""
    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        run = read_run(run_dir, buckets=buckets, stale_after=stale_after, now=now)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda r: r["updated"], reverse=True)
    for i, run in enumerate(runs):
        # Colour follows the run, not its rank in whatever the reader has filtered to.
        run["color"] = SERIES_COLORS[i % len(SERIES_COLORS)]
        run["slot"] = i
    return runs


def build_dashboard(root: Path, buckets: int = 160, stale_after: float = 600.0, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    runs = collect_runs(root, buckets=buckets, stale_after=stale_after, now=now)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "generated_epoch": now,
        "root": str(root),
        "stale_after_s": stale_after,
        "gates": GATES,
        "series_meta": {
            path: {"label": label, "hint": hint} for group in KIND_SERIES.values() for path, label, hint in group
        },
        "runs": runs,
    }


def _inline_fonts(html: str) -> str:
    """Swap the webfont link for the repo's own woff2, base64'd, so the page makes no external request."""
    if not FONTS.is_dir():
        return html
    css = (FONTS / "fonts.css").read_text()

    def embed(match: re.Match) -> str:
        path = FONTS / Path(match.group(1)).name
        if not path.exists():
            return match.group(0)
        return f"url(data:font/woff2;base64,{b64encode(path.read_bytes()).decode()}) format(\"woff2\")"

    css = re.sub(r"url\(([^)]+)\) format\(\"woff2\"\)", embed, css)
    return re.sub(r"<!--fonts-->.*?<!--/fonts-->", lambda _: f"<style>\n{css}</style>", html, count=1, flags=re.S)


def write_dashboard_html(payload: dict, path: str | Path, standalone: bool = True, refresh: float = 0) -> Path:
    """The dashboard as one file: data, styles, fonts and script inlined, no request to anywhere."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DASHBOARD_DATA__*/null", data, 1)
    html = _inline_fonts(html)
    if standalone:
        title = re.search(r"<title>.*?</title>\n", html).group(0)
        meta = '<meta name="color-scheme" content="dark">\n'
        if refresh:
            # A training run appends to metrics.jsonl for hours; --watch rewrites the file, this re-reads it.
            meta += f'<meta http-equiv="refresh" content="{int(refresh)}">\n'
        html = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n{title}{meta}</head>\n<body>\n'
            f"{html.replace(title, '', 1)}\n</body>\n</html>\n"
        )
    path.write_text(html)
    return path


def write_dashboard(
    root: str | Path,
    out: str | Path,
    buckets: int = 160,
    stale_after: float = 600.0,
    refresh: float = 0,
) -> tuple[Path, dict]:
    payload = build_dashboard(Path(root), buckets=buckets, stale_after=stale_after)
    return write_dashboard_html(payload, out, refresh=refresh), payload
