"""Turn recordings of real Nacht der Untoten play into clips of policy frames.

    uv run python scripts/ingest_video.py data/video/*.mp4 --out data/clips/session1

Feed it anything: an OBS capture, a ShadowPlay file, footage recorded years before this project existed.
Each video is decoded to 15 Hz at 128x72, letterbox bars are removed, and the result is split into
continuous runs -- menus, loading screens, pauses and hard cuts are dropped rather than glued together into
transitions that never happened. The clips come out unlabelled; `train_idm.py` and `label_clips.py` are what
give them actions.
"""

import argparse
import json
from pathlib import Path

from zombiesai.demos.clips import load_clip
from zombiesai.demos.frames import FITS
from zombiesai.demos.video import FFmpegMissing, IngestConfig, ingest, probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/clips"), help="where the clips go")
    parser.add_argument("--fit", choices=FITS, default="crop",
                        help="crop: trim to 16:9 (default). pad: keep a 4:3 frame whole, bars and all. stretch: squash")
    parser.add_argument("--min-seconds", type=float, default=4.0, help="drop runs shorter than this")
    parser.add_argument("--start", type=float, default=0.0, help="skip this many seconds of each video")
    parser.add_argument("--duration", type=float, help="ingest only this many seconds of each video")
    parser.add_argument("--cut-delta", type=float, default=IngestConfig.cut_delta,
                        help="mean luma change that counts as a scene cut")
    parser.add_argument("--dry-run", action="store_true", help="probe the videos and stop")
    args = parser.parse_args()

    config = IngestConfig(
        fit=args.fit,
        min_steps=max(1, int(args.min_seconds * IngestConfig.fps)),
        cut_delta=args.cut_delta,
        start_s=args.start,
        duration_s=args.duration,
    )
    total_steps = 0
    try:
        for video in args.videos:
            info = probe(video)
            print(f"{video.name}: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration_s / 60:.1f} min")
            if args.dry_run:
                continue
            written = ingest(video, args.out, config, name=video.stem)
            for path in written:
                clip = load_clip(path)
                total_steps += clip.n_steps
                seconds = clip.n_steps / config.fps
                print(f"  {path.name}: {clip.n_steps} steps ({seconds / 60:.1f} min)")
            if not written:
                print("  nothing kept -- all of it was menu, cut, or too short")
    except FFmpegMissing as error:
        raise SystemExit(str(error)) from error

    if not args.dry_run:
        print(f"\n{total_steps:,} decision frames ({total_steps / config.fps / 3600:.2f} h) under {args.out}")
        print("next: train an inverse dynamics model on labelled play, then label these with it")
        summary = {"clips_root": str(args.out), "steps": total_steps, "config": vars(config)}
        (args.out / "ingest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
