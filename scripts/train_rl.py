"""Reinforcement learning from the policy's own play: PPO fine-tuning of a BC policy across parallel games.

    # Rehearse the whole pipeline on NachtSim first -- same actors, same learner, no game:
    uv run python scripts/train_rl.py runs/bc_real2/bc.pt --env sim --actors 8 --total-steps 200000

    # The real thing: bring a fleet up and check it (docs/rl.md), then
    uv run python scripts/instances.py up --n 4
    uv run python scripts/spike_instances.py --counts-per-degree 9.09
    uv run python scripts/train_rl.py runs/bc_real2/bc.pt --actors 4 --counts-per-degree 9.09 --out runs/rl1

It writes runs/<out>/metrics.jsonl (charted by scripts/dashboard.py like any PPO run), weights.pt (what the
actors follow) and checkpoint.pt, a BC-format policy that play_real.py and eval_bc.py play unchanged:

    uv run python scripts/play_real.py runs/rl1/checkpoint.pt --minutes 3

Ctrl-C stops the actors (every key released) and writes the checkpoint.

Other PCs' games join with --listen (docs/rl.md, "Several PCs"): each runs scripts/fleet_worker.py, and every
machine has the same ZOMBIES_FLEET_TOKEN in its environment.

    ZOMBIES_FLEET_TOKEN=... uv run python scripts/train_rl.py runs/bc_real3/bc.pt --actors 4 --listen :47860 --out runs/rl5
"""

import argparse
from dataclasses import fields
from pathlib import Path

from zombiesai.rl.fleet import TOKEN_ENV, token_from_env
from zombiesai.rl.parallel_ppo import RLConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("init", type=Path, help="the BC checkpoint to start from (or an RL checkpoint to continue)")
    parser.add_argument("--env", choices=("real", "sim"), default="real")
    parser.add_argument("--actors", type=int, default=4, help="one per game instance (real) or sim process; "
                                                                 "0 with --listen: only other machines play")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fleet", default=RLConfig.fleet_root, help="the fleet's root (scripts/instances.py)")
    parser.add_argument("--sim-hardness", type=float, default=0.5)
    for f in fields(RLConfig):
        if f.name in ("init", "env", "n_actors", "fleet_root", "sim"):
            continue
        flag = "--" + f.name.replace("_", "-")
        kind = type(f.default) if f.default is not None else float
        if kind is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=kind, default=f.default)
    args = parser.parse_args()

    values = {f.name: getattr(args, f.name) for f in fields(RLConfig)
              if f.name not in ("init", "env", "n_actors", "fleet_root", "sim")}
    config = RLConfig(init=str(args.init), env=args.env, n_actors=args.actors, fleet_root=args.fleet,
                      sim={"hardness": args.sim_hardness}, **values)
    if args.out is None:
        root = Path("runs")
        k = len(list(root.glob(f"rl_{args.env}_*"))) if root.exists() else 0
        args.out = root / f"rl_{args.env}_{k:03d}"
    token = token_from_env()
    if config.listen and not token:
        raise SystemExit(f"--listen needs a shared secret: {TOKEN_ENV}=<the same value on every machine> "
                         "(python -c 'import secrets; print(secrets.token_hex(16))' makes one)")
    if args.env == "real" and args.actors > 0:
        from zombiesai.realgame.instances import load_fleet

        fleet = load_fleet(config.fleet_root)
        if fleet.n < args.actors:
            raise SystemExit(f"--actors {args.actors} but the fleet has {fleet.n} instances; "
                             f"scripts/instances.py up --n {args.actors}")
    print(f"writing {args.out}")
    checkpoint = train(config, args.out, fleet_token=token)
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
