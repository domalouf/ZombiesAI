"""What the actors cost: resident memory per actor process and how long each decision waits for the policy, with
the policy in every actor (`--inference local`) or in one inference server per machine, on the CPU or the GPU.

    uv run python scripts/bench_actors.py                       # 1, 4 and 8 actors, every mode, deaf and hearing
    uv run python scripts/bench_actors.py --actors 8 --modes server-cpu --seconds 60

No game is needed: the actors play the synthetic stand-in, paced like the real game (`--hz 15`; 0 runs flat out,
the worst case for a shared server), with a fresh BC policy of the real network's size (`BCConfig` defaults with
the test fixture's frame offsets; `--hidden` to change it), deaf and with audio (`use_audio=True`).

Reported per run: VmRSS of each actor process (what `top` shows; shared libraries count in every process) and the
total PSS of actors plus server (shared pages split between their users -- the honest total), whether any actor
mapped libtorch, the policy wait per decision (p50/p99: from the frames being ready to the action being back,
IPC and sampling included), the share of actions that never came (timeouts) and of late steps.
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

MODES = {"local": ("local", "cpu"), "server-cpu": ("server", "cpu"), "server-gpu": ("server", "cuda")}


def proc_kb(pid: int, path: str, key: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/{path}").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def maps_torch(pid: int) -> bool:
    try:
        return "libtorch" in Path(f"/proc/{pid}/maps").read_text()
    except OSError:
        return False


def checkpoint(directory: Path, audio: bool, hidden: int) -> Path:
    import torch

    from zombiesai.demos import bc

    config = bc.BCConfig(frame_offsets=(0, 1, 3), hidden=hidden, use_audio=audio)
    torch.manual_seed(0)
    path = directory / f"bc_{'audio' if audio else 'deaf'}_{hidden}.pt"
    bc.save(path, bc.build_net(config), config, 0, {}, {"fire_duty": 0.1})
    return path


def run(init: Path, mode: str, n: int, args, work: Path) -> dict:
    from zombiesai.demos import bc
    from zombiesai.rl.config import RLConfig
    from zombiesai.rl.pool import ActorPool
    from zombiesai.rl.weights import publish

    inference, device = MODES[mode]
    run_dir = work / f"{init.stem}_{mode}_{n}"
    run_dir.mkdir()
    net, _, _ = bc.load(init)
    publish(run_dir / "weights.pt", net, 0)
    config = RLConfig(init=str(init), env="synthetic", n_actors=n, segment_steps=64, synthetic_hz=args.hz,
                      inference=inference, actor_device=device, seed=0)
    pool = ActorPool(config, run_dir, say=lambda m: print(f"    {m}", file=sys.stderr))
    waits, timeouts, steps, bad = [], 0, 0, 0
    memory: dict = {}
    begin = time.time()
    try:
        pool.start()
        while time.time() - begin < args.seconds:
            kind, i, payload = pool.get(0.2)
            pool.tend()
            warm = time.time() - begin > args.warmup
            if kind == "latency" and warm:
                waits.append(payload["ms"])
                timeouts += payload["timeouts"]
            elif kind == "segment" and warm:
                steps += payload.n
                bad += int(payload.bad.sum())
            elif kind == "error":
                print(payload, file=sys.stderr)
            if not memory and time.time() - begin > (args.seconds + args.warmup) / 2:
                actors = [p.pid for p in pool.procs.values()]
                server = [pool.server.pid] if pool.server is not None else []
                rss = [proc_kb(pid, "status", "VmRSS") or 0 for pid in actors]
                memory = {
                    "rss_mb_per_actor": round(float(np.mean(rss)) / 1024),
                    "rss_mb_server": round((proc_kb(server[0], "status", "VmRSS") or 0) / 1024) if server else 0,
                    "pss_mb_total": round(sum(proc_kb(pid, "smaps_rollup", "Pss") or 0
                                              for pid in actors + server) / 1024),
                    "torch_in_actors": any(maps_torch(pid) for pid in actors),
                }
    finally:
        pool.close()
    ms = np.concatenate(waits) if waits else np.zeros(1, np.float32)
    return {"mode": mode, "audio": "audio" in init.stem, "actors": n, **memory,
            "p50_ms": round(float(np.percentile(ms, 50)), 2), "p99_ms": round(float(np.percentile(ms, 99)), 2),
            "decisions": int(ms.size), "timeouts": timeouts,
            "bad_step_frac": round(bad / steps, 4) if steps else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actors", default="1,4,8", help="comma-separated actor counts")
    parser.add_argument("--modes", default="local,server-cpu,server-gpu", help=f"some of {', '.join(MODES)}")
    parser.add_argument("--audio", choices=("both", "deaf", "hearing"), default="both")
    parser.add_argument("--hz", type=float, default=15.0, help="decisions a second per actor (0: flat out)")
    parser.add_argument("--hidden", type=int, default=512, help="the policy's width (BCConfig.hidden)")
    parser.add_argument("--seconds", type=float, default=25.0, help="per run, warm-up included")
    parser.add_argument("--warmup", type=float, default=6.0, help="seconds not counted: imports, CUDA, first frames")
    parser.add_argument("--json", type=Path, default=None, help="also write the rows here")
    args = parser.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    if "server-gpu" in modes:
        import torch

        if not torch.cuda.is_available():
            print("no CUDA here: skipping server-gpu", file=sys.stderr)
            modes.remove("server-gpu")
    rows = []
    with tempfile.TemporaryDirectory(prefix="bench_actors_") as tmp:
        work = Path(tmp)
        for audio in {"both": (False, True), "deaf": (False,), "hearing": (True,)}[args.audio]:
            init = checkpoint(work, audio, args.hidden)
            for n in (int(a) for a in args.actors.split(",")):
                for mode in modes:
                    row = run(init, mode, n, args, work)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
