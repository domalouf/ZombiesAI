"""Film one NachtSim game in first-person pixels: an MP4 of the raycast view with the agent's own 128x72 frame inset."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from zombiesai.agents.random_agent import RandomAgent
from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.sim.bitmap_font import GLYPH_H, draw_text, text_width
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.sim.render import Renderer

INSET_SCALE = 2
END_CARD_S = 3.0
STATS = ("kills", "headshot_kills", "melee_kills", "shots", "shot_hits")


def compose(view: np.ndarray, agent_frame: np.ndarray, s: int) -> np.ndarray:
    """Pin the agent's actual frame, enlarged with hard pixel edges, to the film frame's top-right corner."""
    inset = np.kron(agent_frame, np.ones((INSET_SCALE, INSET_SCALE, 1), dtype=np.uint8))
    h, w = inset.shape[:2]
    margin, border = 3 * s, max(1, s // 2)
    x1, y0 = view.shape[1] - margin, margin
    x0 = x1 - w
    view[y0 - border : y0 + h + border, x0 - border : x1 + border] = (20, 18, 16)
    view[y0 : y0 + h, x0:x1] = inset
    label = "AGENT VIEW 128X72"
    draw_text(view, label, x1 - text_width(label, s), y0 + h + border + 2 * s, s, (220, 214, 200))
    return view


def end_card(frame: np.ndarray, rounds: int, truncated: bool, s: int) -> np.ndarray:
    card = (frame.astype(np.float32) * 0.3).astype(np.uint8)
    height, width = card.shape[:2]
    big, small = 5 * s, 2 * s
    lines = [
        ("GAME OVER" if not truncated else "STEP CAP", big, (178, 22, 18), height // 2 - GLYPH_H * big),
        (f"YOU SURVIVED {rounds} ROUND{'' if rounds == 1 else 'S'}", small, (236, 236, 236), height // 2 + 3 * s),
    ]
    for text, scale, color, y in lines:
        draw_text(card, text, (width - text_width(text, scale)) // 2, y, scale, color)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--checkpoint", type=Path, help="film a trained policy instead (PPO or BC; overrides --agent)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardness", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--size", default="960x540", help="film resolution as WIDTHxHEIGHT (both even)")
    parser.add_argument("--out", type=Path, help="default: runs/films/<agent>-seed<seed>.mp4")
    args = parser.parse_args()

    if args.checkpoint:
        import torch

        from zombiesai.rl.agent import load_agent

        torch.manual_seed(args.seed)  # the policy samples its actions; same seed, same game
        agent = load_agent(args.checkpoint)
        args.agent = "bc" if getattr(agent, "obs_profile", "state") == "render" else "ppo"
    else:
        agent = ScriptedAgent() if args.agent == "scripted" else RandomAgent(args.seed)

    width, height = (int(v) for v in args.size.lower().split("x"))
    profile = getattr(agent, "obs_profile", "state")
    env = NachtSim(
        SimConfig(hardness=args.hardness, max_steps=args.max_steps, obs_profile=profile), render_mode="rgb_array"
    )
    obs, _ = env.reset(seed=args.seed)
    agent.reset()
    camera = Renderer(env.geo, width, height)
    fps = 60 / env.timing.frames_per_step  # one film frame per decision, so playback runs at game speed
    out = args.out or Path("runs/films") / f"{args.agent}-seed{args.seed}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )

    totals, between_rounds = dict.fromkeys(STATS, 0), False
    frame = compose(camera.render(env), env.render(), camera.s)
    encoder.stdin.write(frame.tobytes())
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(np.asarray(agent.act(obs)))
        for e in info["events"]:
            if e["type"] == "round_complete":
                between_rounds = True
                for k in STATS:
                    totals[k] += e[k]
            elif e["type"] == "round_start":
                between_rounds = False
        frame = compose(camera.render(env), env.render(), camera.s)
        encoder.stdin.write(frame.tobytes())
        if env.steps % 500 == 0:
            print(f"  step {env.steps}, round {env.round}", flush=True)
    if not between_rounds:  # the round in progress isn't in any round_complete event yet
        for k in STATS:
            totals[k] += getattr(env.rs, k)
    card = end_card(frame, env.round, truncated, camera.s)
    for _ in range(round(END_CARD_S * fps)):
        encoder.stdin.write(card.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {out}")

    minutes, seconds = divmod(env.t, 60)
    print(
        f"{args.agent} agent, seed {args.seed}: round {env.round} after {int(minutes)}:{seconds:04.1f} of game time, "
        f"{totals['kills']} kills ({totals['headshot_kills']} headshots, {totals['melee_kills']} knifed), "
        f"{totals['shot_hits']}/{totals['shots']} shots hit, {env.steps} steps at {fps:g} fps"
    )
    print(f"film: {out.resolve()} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
