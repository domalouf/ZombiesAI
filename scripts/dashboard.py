"""Build a dashboard of every training run under runs/: learning curves, health, and the plan's gates."""

import argparse
import time
import webbrowser
from pathlib import Path

from zombiesai.viz.dashboard import write_dashboard


def summarize(payload: dict) -> None:
    """The same verdicts the page shows, for whoever is watching the terminal instead."""
    runs = payload["runs"]
    if not runs:
        print(f"no runs with a metrics.jsonl under {payload['root']}/ yet")
        return
    print(f"{len(runs)} run{'s' if len(runs) != 1 else ''} under {payload['root']}/:")
    for run in runs:
        head = run["series"].get(run["headline"] or "")
        value = "nothing logged yet"
        if head:
            value = f"{head['label'].lower()} {head['last']:.3g} (best {head['best']:.3g})"
        print(f"  {run['name']:<34} {run['status']:<8} {run['progress_text']:<26} {value}")
        for note in run["notes"]:
            if note["level"] in ("critical", "serious"):
                print(f"    ! {note['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="directory of run directories")
    parser.add_argument("--out", type=Path, help="default: <runs>/dashboard.html")
    parser.add_argument("--buckets", type=int, default=160, help="points per curve; each bucket keeps its spread")
    parser.add_argument("--stale-after", type=float, default=600.0,
                        help="seconds without a new row before a run reads as stopped")
    parser.add_argument("--watch", type=float, nargs="?", const=30.0, metavar="SECONDS",
                        help="rebuild on an interval while training runs; the page reloads itself to match")
    parser.add_argument("--no-open", action="store_true", help="don't open the dashboard in a browser")
    args = parser.parse_args()

    out = args.out or args.runs / "dashboard.html"
    refresh = args.watch or 0
    build = dict(buckets=args.buckets, stale_after=args.stale_after, refresh=refresh)
    path, payload = write_dashboard(args.runs, out, **build)
    summarize(payload)
    print(f"dashboard: {path.resolve()}")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    if not args.watch:
        return

    print(f"watching {args.runs}/ — rebuilding every {refresh:g}s, Ctrl-C to stop")
    try:
        while True:
            time.sleep(refresh)
            _, payload = write_dashboard(args.runs, out, **build)
            live = [r["name"] for r in payload["runs"] if r["status"] == "running"]
            names = f": {', '.join(live)}" if live else ""
            print(f"{time.strftime('%H:%M:%S')}  rebuilt — {len(live)} running{names}")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
