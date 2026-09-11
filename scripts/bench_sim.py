"""Measure NachtSim throughput: single-process steps/s, then the multi-process scaling curve (M1 exit)."""

import os

# One BLAS thread per worker, or per-process numpy calls fight over cores and scaling flattens early.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import time

import numpy as np


def _worker(seconds: float, seed: int, barrier, results) -> None:
    from zombiesai import spec
    from zombiesai.sim.nacht_sim import NachtSim

    env = NachtSim()
    env.reset(seed=seed)
    actions = np.random.default_rng(seed).integers(0, spec.ACTION_NVEC, size=(4096, len(spec.ACTION_NVEC)))

    def run(duration: float) -> int:
        steps, start = 0, time.perf_counter()
        while time.perf_counter() - start < duration:
            for a in actions[steps % 4096 : steps % 4096 + 64]:
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    env.reset()
            steps += 64
        return steps

    run(1.0)  # warm-up: lazily built distance tables, allocator, caches
    barrier.wait()
    start = time.perf_counter()
    steps = run(seconds)
    results.put(steps / (time.perf_counter() - start))


def measure(n_workers: int, seconds: float) -> list[float]:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n_workers)
    results = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(seconds, i, barrier, results)) for i in range(n_workers)]
    for p in procs:
        p.start()
    rates = [results.get() for _ in procs]
    for p in procs:
        p.join()
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    counts = sorted({1, *(2**k for k in range(1, 8) if 2**k < args.max_workers), args.max_workers})
    single = None
    print(f"{'workers':>7} {'total steps/s':>14} {'per worker':>11} {'efficiency':>10}")
    for n in counts:
        rates = measure(n, args.seconds)
        total = sum(rates)
        single = single or total
        print(f"{n:>7} {total:>14,.0f} {total / n:>11,.0f} {total / (n * single):>10.0%}")
    print(f"\nM1 target: >=5,000 steps/s single-process -> {'PASS' if single >= 5000 else 'FAIL'} ({single:,.0f})")


if __name__ == "__main__":
    main()
