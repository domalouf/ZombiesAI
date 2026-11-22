"""Check the opening fast-forward (`--skip-opening-s`) on a whole fleet, as a training run would load it.

    uv run python scripts/check_skip.py --actors 7 --games 3 --skips 0,30

Every instance plays idle games (the neutral action every step, so a zombie walks up and kills it) in blocks:
`--games` with no skip, then `--games` with each skip given. All instances run the same block at once, so the
skip block carries the load a training run would -- every game fast-forwarding with its frame cap lifted.

The fast-forward cannot be timed from inside the game, so it is checked against the games that play the
opening: an idle player's first hit lands at a game time that does not depend on the skip, so
`skipped_s + (first step to first hit)` should match the no-skip block's first hit. Coming in *above* it means
the games ran slower than the timescale under load (`skipped_s` is a ceiling): the gap is the game-seconds
the skip did not get through. Also reported: wall time per game (reset included), how often the policy's
steps run late in each block (a fast-forwarding game takes GPU and CPU from the playing ones), the console
commands each game logged, and any console errors.

Writes one JSON line per game to --out, then prints a summary.
"""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path


def _console_log(fleet, index: int) -> Path:
    from zombiesai.realgame.instances import specs

    return specs(fleet)[index].plutonium / "console.log"


def worker(index: int, skips: list[float], games: int, fleet_root: str, max_game_s: float, barrier, out) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import numpy as np

    from zombiesai import spec
    from zombiesai.realgame.instances import load_fleet
    from zombiesai.rl.actors import make_real_env
    from zombiesai.rl.config import RLConfig

    noop = np.array(spec.NEUTRAL_ACTION, dtype=np.int64)
    fleet = load_fleet(fleet_root)
    log = _console_log(fleet, index)
    config = RLConfig(fleet_root=fleet_root, hear=False, record_best=False, record_moments=False,
                      record_records=False)
    env = None
    try:
        env = make_real_env(config, index)
        env._needs_restart = True  # a fresh game every time, never one already running
        for skip in skips:
            env.config = replace(env.config, skip_opening_s=skip)
            expected = list(env.skip_commands()) if skip else []
            barrier.wait()  # every instance starts the block together
            for game in range(games):
                mark = log.stat().st_size if log.exists() else 0
                t_reset = time.monotonic()
                env.reset()
                t0 = time.monotonic()
                first_hit = None
                info = {}
                while True:
                    _, _, term, trunc, info = env.step(noop)
                    if first_hit is None and env.damage.hits:
                        first_hit = time.monotonic() - t0
                    if term or trunc or time.monotonic() - t0 > max_game_s:
                        break
                episode = info.get("episode") or env.summary("cap")
                text = log.read_bytes()[mark:].decode("latin-1", "replace") if log.exists() else ""
                lines = text.splitlines()
                out.put({
                    "actor": index, "skip": skip, "game": game,
                    "reset_s": round(t0 - t_reset, 2), "episode_s": round(time.monotonic() - t0, 2),
                    "first_hit_s": None if first_hit is None else round(first_hit, 2),
                    "skipped_s": round(episode["skipped_s"], 2), "reason": episode["reason"],
                    "steps": episode["length"], "bad_steps": episode["bad_steps"],
                    "commands": [line[1:] for line in lines if line.startswith("]")],
                    "expected": expected,
                    # only what the skip's own commands could cause; the game logs plenty of harmless errors
                    "errors": [line for line in lines if "Unknown command" in line or "Error" in line
                               and any(word in line for word in ("timescale", "sv_cheats", "com_maxfps"))],
                })
    except Exception as exc:  # noqa: BLE001 -- reported, and the barrier released for the others
        out.put({"actor": index, "error": f"{type(exc).__name__}: {exc}"})
        barrier.abort()
    finally:
        if env is not None:
            env.close()


def summarize(rows: list[dict]) -> None:
    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return "-"
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return (f"mean {mean:5.1f} +- {sd / len(values) ** 0.5:4.1f}, median {statistics.median(values):5.1f} "
                f"(n={len(values)}, {min(values):.1f}-{max(values):.1f})")

    games = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            print(f"actor {r['actor']}: {r['error']}")
    base = [r for r in games if r["skip"] == 0]
    base_hit = statistics.fmean([r["first_hit_s"] for r in base if r["first_hit_s"] is not None] or [0.0])
    for skip in sorted({r["skip"] for r in games}):
        block = [r for r in games if r["skip"] == skip]
        hit_game_time = [r["skipped_s"] + r["first_hit_s"] for r in block if r["first_hit_s"] is not None]
        steps, bad = sum(r["steps"] for r in block), sum(r["bad_steps"] for r in block)
        print(f"\nskip {skip:g}: {len(block)} games, ends {dict((k, sum(r['reason'] == k for r in block)) for k in {r['reason'] for r in block})}")
        print(f"  reset (wall)                  {stats([r['reset_s'] for r in block])}")
        print(f"  first step to first hit       {stats([r['first_hit_s'] for r in block])}")
        print(f"  skipped_s + that (game time)  {stats(hit_game_time)}")
        print(f"  wall per game (reset + play)  {stats([r['reset_s'] + r['episode_s'] for r in block])}")
        print(f"  late steps                    {bad}/{steps} = {bad / max(1, steps):.1%}")
        if skip and base and hit_game_time:
            print(f"  vs no skip: first hit {statistics.fmean(hit_game_time) - base_hit:+.1f} game-s "
                  f"(above 0: the skip got through less than skipped_s says)")
        wanted = None
        for r in block:
            if r["commands"][-len(r["expected"]):] != r["expected"] if r["expected"] else False:
                wanted = wanted or []
                wanted.append((r["actor"], r["game"], r["commands"]))
        if wanted:
            print("  games whose console did not log the three skip commands:", wanted)
        errors = [(r["actor"], e) for r in block for e in r["errors"]]
        if errors:
            print("  console errors:", errors[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actors", type=int, default=7)
    parser.add_argument("--instances", default="", help="which instances, e.g. 0,1,2,4 (default: the first --actors)")
    parser.add_argument("--append", action="store_true", help="add to --out (and summarize all of it)")
    parser.add_argument("--games", type=int, default=3, help="per instance per block")
    parser.add_argument("--skips", default="0,30", help="the blocks, in order: skip_opening_s values")
    parser.add_argument("--fleet", default="runs/instances")
    parser.add_argument("--max-game-s", type=float, default=240.0)
    parser.add_argument("--out", type=Path, default=Path("runs/exp/skip_check.jsonl"))
    parser.add_argument("--summary", action="store_true", help="only summarize an existing --out")
    args = parser.parse_args()
    if args.summary:
        summarize([json.loads(line) for line in args.out.read_text().splitlines()])
        return
    skips = [float(s) for s in args.skips.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chosen = [int(i) for i in args.instances.split(",")] if args.instances else list(range(args.actors))
    ctx = mp.get_context("spawn")
    out, barrier = ctx.Queue(), ctx.Barrier(len(chosen))
    procs = [ctx.Process(target=worker, args=(i, skips, args.games, args.fleet, args.max_game_s, barrier, out))
             for i in chosen]
    for p in procs:
        p.start()
    earlier = [json.loads(line) for line in args.out.read_text().splitlines()] \
        if args.append and args.out.exists() else []
    rows, expected = [], len(chosen) * args.games * len(skips)
    with args.out.open("a" if args.append else "w") as f:
        while len([r for r in rows if "error" not in r]) < expected and any(p.is_alive() for p in procs) \
                or not out.empty():
            try:
                row = out.get(timeout=5.0)
            except Exception:  # noqa: BLE001 -- queue.Empty
                continue
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "error" in row:
                print(f"actor {row['actor']}: {row['error']}", flush=True)
            else:
                print(f"actor {row['actor']} skip {row['skip']:g} game {row['game']}: reset {row['reset_s']} s, "
                      f"first hit {row['first_hit_s']} s, skipped {row['skipped_s']}, {row['reason']}, "
                      f"late {row['bad_steps']}/{row['steps']}", flush=True)
    for p in procs:
        p.join(timeout=30)
    summarize(earlier + rows)


if __name__ == "__main__":
    main()
