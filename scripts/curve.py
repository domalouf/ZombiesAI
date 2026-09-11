"""Print a training run's learning curve from its metrics.jsonl, as a table with a bar per row."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory, e.g. runs/ppo-nacht-state-s1")
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "return_mean" in r]
    if not rows:
        raise SystemExit("no finished episodes logged yet")
    picked = [rows[round(i * (len(rows) - 1) / max(1, args.rows - 1))] for i in range(min(args.rows, len(rows)))]
    lo = min(r["return_mean"] for r in picked)
    hi = max(r["return_mean"] for r in picked)
    extra = "round_reached_mean" if "round_reached_mean" in rows[-1] else None

    round_col = f"  {'round':>5}" if extra else ""
    header = f"{'step':>12}  {'return':>8}{round_col}  {'entropy':>7}  {'expl.var':>8}"
    print(header)
    for r in picked:
        bar = "█" * round(30 * (r["return_mean"] - lo) / (hi - lo)) if hi > lo else ""
        ev = r.get("explained_variance")
        line = f"{r['step']:>12,}  {r['return_mean']:>8.2f}"
        if extra:
            line += f"  {r.get(extra, float('nan')):>5.2f}"
        line += f"  {r['entropy']:>7.2f}  {'-' if ev is None else f'{ev:.2f}':>8}  {bar}"
        print(line)
    print(f"\n{rows[-1]['episodes']:,} episodes, {rows[-1]['sps']:,} steps/s")


if __name__ == "__main__":
    main()
