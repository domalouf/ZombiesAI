"""Keep domalouf.com/zombies/training/ and the stream's page and overlay (/zombies/live/) live: push this machine,
its training runs and the stream's numbers to the site every few seconds (viz/live_site.py). It only ever connects out -- nothing on this PC is opened to the network.

    uv run python scripts/publish_live.py --dest zombies-live@lts.lan:     # every 5 s, until Ctrl-C
    uv run python scripts/publish_live.py --once                           # write the files, push nothing

On a PC that plays for the learner (scripts/fleet_worker.py), --worker <id> pushes only that machine and what its
worker is doing, as live/machine-<id>.json; the page shows it once deploy/deploy.sh is run with LIVE_MACHINES=<id>:

    uv run python scripts/publish_live.py --dest zombies-live: --worker rig2 --label "Gaming PC 2"

--dest is anything rsync takes. With the restricted key README.md sets up ("Live on the site"), the server pins
the directory, so the path after the colon is empty. ZOMBIES_LIVE_DEST stands in for --dest (the systemd unit sets it).
"""

import argparse
import os
import time
from pathlib import Path

from zombiesai.rl.fleet import STATUS_FILE
from zombiesai.viz.live_site import LivePublisher, machine_id, worker_file
from zombiesai.viz.system import SystemSampler

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=os.environ.get("ZOMBIES_LIVE_DEST"),
                        help="rsync destination: the site's zombies/training/live/ (default: $ZOMBIES_LIVE_DEST)")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between pushes (default: 5)")
    parser.add_argument("--runs-every", type=float, default=60.0, help="seconds between rebuilds of runs.json")
    parser.add_argument("--out", type=Path, help="staging directory (default: $XDG_RUNTIME_DIR/zombiesai-live)")
    parser.add_argument("--once", action="store_true", help="sample, write the files once, and push only with --dest")
    parser.add_argument("--stream-run", help="the run the stream overlay's numbers come from (default: the one "
                                             "training now, the real game before the sim)")
    parser.add_argument("--worker", metavar="ID", default=os.environ.get("ZOMBIES_LIVE_WORKER"),
                        help="this PC plays for the learner: push only live/machine-<ID>.json (a-z, 0-9, -; "
                             "$ZOMBIES_LIVE_WORKER)")
    parser.add_argument("--label", default=os.environ.get("ZOMBIES_LIVE_LABEL"),
                        help="what the page calls this machine (default: \"Training PC\", or the worker's id)")
    parser.add_argument("--worker-status", type=Path, default=REPO / "runs" / "fleet" / STATUS_FILE,
                        help="the status fleet_worker.py writes (default: runs/fleet/status.json)")
    args = parser.parse_args()
    if args.worker:
        try:
            machine_id(args.worker)
        except ValueError as error:
            parser.error(str(error))
    out = args.out or Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "zombiesai-live"
    if not args.dest and not args.once:
        parser.error("--dest (or ZOMBIES_LIVE_DEST) is where the site's zombies/training/live/ is; --once to try it")

    sampler = SystemSampler().start()
    publisher = LivePublisher(REPO, out, args.dest, sampler=sampler, runs_every_s=args.runs_every,
                              stream_run=args.stream_run, label=args.label, worker=args.worker,
                              worker_status=args.worker_status)
    time.sleep(2.5)  # the sampler's first rates need two samples
    try:
        if args.once:
            publisher.tick()
            wrote = f"{out}/{worker_file(args.worker)}" if args.worker else f"{out}/machine.json, runs.json and stream.json"
            print(f"wrote {wrote}" + (f", pushed to {args.dest}" if args.dest else ""))
            return
        print(f"publishing every {args.every:g}s to {args.dest} (Ctrl-C to stop)", flush=True)
        while True:
            started = time.monotonic()
            try:
                publisher.tick()
            except OSError as error:  # a run directory vanishing mid-read: the next tick will do
                print(f"{time.strftime('%H:%M:%S')}  skipped a tick: {error}", flush=True)
            time.sleep(max(0.5, args.every - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
