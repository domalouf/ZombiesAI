"""Build the stream's pages for domalouf.com/zombies/live/: the Twitch player with the agent's numbers and its PPO
training around it, and the overlay OBS draws on the stream (viz/stream.py).

    uv run python scripts/build_stream.py --channel <twitch name>      # site/zombies/live/ and .../overlay/
    uv run python scripts/build_stream.py --demo                       # made-up games, to see the layout

Both pages poll ../training/live/stream.json, which scripts/publish_live.py pushes every few seconds; the page
also carries the numbers as they stand now, for when the PC is off. TWITCH_CHANNEL stands in for --channel.
In OBS, add a Browser source at the canvas size (1920x1080) pointing at https://domalouf.com/zombies/live/overlay/
(?corner=tr, ?scale=1.25, ?layout=column, ?demo=1 to place it before there are games).
"""

import argparse
import os
import time
from pathlib import Path

from zombiesai.viz.live_site import public_runs
from zombiesai.viz.stream import StreamFeed, demo_games, stream_payload, write_stream_site
from zombiesai.viz.supervise import build_payload, live_trainers, run_roots

REPO = Path(__file__).resolve().parents[1]
LINKS = [("← domalouf.com", "/"), ("Training Room", "/zombies/training/"),
         ("Code on GitHub", "https://github.com/domalouf/ZombiesAI")]
DESCRIPTION = "Watch a reinforcement-learning agent learn Nacht der Untoten live on Twitch, and how its PPO training is going."


def snapshot(stream_run: str | None) -> dict:
    payload = build_payload(run_roots(REPO))
    trainers = live_trainers(payload["run_paths"])
    feed = StreamFeed(stream_run)
    feed.choose(payload["run_paths"], trainers)
    # Built, not live: the page says it is offline until the PC's first push.
    return dict(feed.payload(public_runs(payload, trainers), trainers, time.time()), live=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", default=os.environ.get("TWITCH_CHANNEL"),
                        help="the Twitch channel to embed (default: $TWITCH_CHANNEL; none: a placeholder)")
    parser.add_argument("--out", type=Path, default=Path("site/zombies/live"))
    parser.add_argument("--stream-run", help="the run whose numbers to show (default: the one training now)")
    parser.add_argument("--demo", action="store_true", help="bake in made-up games instead of the runs'")
    args = parser.parse_args()

    data = (stream_payload("demo", demo_games(), None, live=False, now=time.time()) if args.demo
            else snapshot(args.stream_run))
    index, overlay = write_stream_site(args.out, channel=args.channel, snapshot=data, links=LINKS, description=DESCRIPTION)
    games = data["stats"]["games"]
    print(f"{index} and {overlay}: {data['run'] or 'no run'} ({games} games), "
          + (f"Twitch channel {args.channel}" if args.channel else "no Twitch channel yet"))


if __name__ == "__main__":
    main()
