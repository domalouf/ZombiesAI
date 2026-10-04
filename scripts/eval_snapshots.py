"""Play the agent as it was at each hour of training, for the record (rl/evaluate.py): the progress chart in
rounds, and your own play to measure it against.

    ./zai stop --keep-games                                      # the games must be free: training uses them
    uv run python scripts/eval_snapshots.py runs --every-h 4 --games 20
    uv run python scripts/eval_snapshots.py runs/rl12/snapshots/h0030.00.pt --games 50
    uv run python scripts/eval_snapshots.py runs --list          # the snapshots there are, and nothing else
    uv run python scripts/eval_snapshots.py --human data/demos   # only your own numbers, from the demos

Each snapshot plays --games games on --actors of the fleet's instances, frozen, exactly as it played in
training. Its games and its best game's film go in runs/video/evals/<snapshot>/; the table of every snapshot
evaluated so far is runs/video/evals/evals.csv, with a "human" row from your demos when --human names them (run
scripts/parse_hud.py over them first). A snapshot already evaluated is skipped unless --again.
"""

import argparse
import json
from pathlib import Path

from zombiesai.rl.config import RLConfig
from zombiesai.rl.evaluate import evaluate_snapshot, find_snapshots, human_baseline, pick, snapshot_hours, write_results


def training_now(repo: Path) -> bool:
    from zombiesai.session import ours, trees

    found = ours([t for _, t in trees(repo)]).values()
    return any(k in ("trainer", "worker") and p.flag("--env") != "synthetic" for p, k in found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("where", nargs="*", type=Path, help="snapshots, run directories, or directories of runs")
    parser.add_argument("--every-h", type=float, default=0.0, help="one snapshot per this many hours (default: all)")
    parser.add_argument("--games", type=int, default=20, help="games per snapshot (default: 20)")
    parser.add_argument("--actors", type=int, default=None, help="games at once (default: the fleet's size)")
    parser.add_argument("--human", nargs="*", type=Path, default=None, help="demo directories for the human row")
    parser.add_argument("--out", type=Path, default=Path("runs/video/evals"))
    parser.add_argument("--list", action="store_true", help="list the snapshots and stop")
    parser.add_argument("--again", action="store_true", help="evaluate snapshots already evaluated too")
    parser.add_argument("--env", choices=("real", "synthetic"), default="real",
                        help="synthetic: a rehearsal of the plumbing, with no game")
    args = parser.parse_args()

    human = human_baseline(args.human) if args.human else None
    if human is not None:
        print(f"you: {json.dumps(human)}")
    chosen = pick(find_snapshots(args.where), args.every_h) if args.where else []
    if args.list:
        for path in chosen:
            print(f"hour {snapshot_hours(path):7.2f}  {path}")
        return
    done = set()
    if (args.out / "evals.jsonl").exists() and not args.again:
        done = {json.loads(line)["snapshot"] for line in (args.out / "evals.jsonl").read_text().splitlines()
                if line.strip()}
    todo = [p for p in chosen if str(p) not in done]
    if todo and args.env == "real" and training_now(Path.cwd()):
        raise SystemExit("training is playing the games right now: `./zai stop --keep-games` first")
    if todo:
        if args.env == "real":
            from zombiesai.realgame.instances import load_fleet

            actors = args.actors or load_fleet(RLConfig.fleet_root).n
        else:
            actors = args.actors or 2
        base = RLConfig(n_actors=actors, env=args.env)
    rows = []
    for path in todo:
        print(f"hour {snapshot_hours(path):.2f}: {path}, {args.games} games on {actors} instances")
        row = evaluate_snapshot(path, args.games, args.out / f"{path.parent.parent.name}-{path.stem}", base)
        print(f"  -> {json.dumps(row)}")
        rows.append(row)
        write_results(args.out, [row], human)
    if not rows and human is not None:
        write_results(args.out, [], human)
    if (args.out / "evals.csv").exists():
        print((args.out / "evals.csv").read_text())


if __name__ == "__main__":
    main()
