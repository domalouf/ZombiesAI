"""What one PPO run is configured with: `RLConfig`, shared by the learner, its actors and every fleet worker.

Each field becomes a `scripts/train_rl.py` flag of the same name (and a setting in the supervision page's form,
which reads this class from source -- viz/supervise.py), so a comment on a field is also its help text.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class RLConfig:
    init: str = ""  # the BC checkpoint RL starts from
    env: str = "real"  # "real": the game instances of a fleet; "synthetic": no game, a rehearsal of the plumbing
    n_actors: int = 4
    total_steps: int = 2_000_000
    segment_steps: int = 256  # ~17 s of real play
    batch_steps: int = 4096  # ~70 s of four games
    gamma: float = 0.995  # a 13 s horizon at 15 Hz
    gae_lambda: float = 0.95
    lr: float = 5e-5
    update_epochs: int = 3
    minibatch_size: int = 256
    clip_coef: float = 0.2  # the ratio is a product over eight heads: 0.1 clipped half of every batch
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    kl_coef: float = 0.2
    kl_decay: float = 0.995  # per update, down to kl_min
    kl_min: float = 0.02
    target_kl: float | None = 0.03  # stop an update's epochs early past 1.5x this
    critic_warmup_updates: int = 5
    max_grad_norm: float = 0.5
    max_policy_lag: int = 2
    reward_scale: bool = True
    device: str = "auto"
    actor_device: str = "cpu"
    seed: int = 0
    checkpoint_every: int = 10
    # Keep a copy of the policy every this many hours of training, counted across continued runs, in the run's
    # snapshots/ (0: never) -- to film or evaluate the agent as it was at hour N (scripts/eval_snapshots.py)
    snapshot_every_h: float = 1.0
    # real env
    fleet_root: str = "runs/instances"
    # Mouse counts per degree of turn. None: worked out from the sensitivity and m_yaw the games play with
    # (rl/actors.py), so a look bin turns the same number of degrees on every PC whatever its config says.
    counts_per_degree: float | None = None
    # Mouse motor time constant (s): a look sets a turn rate the mouse eases into, as a hand does, instead of a
    # burst per decision. 0 sends each decision's turn as sub-moves inside its tick.
    look_smoothing_s: float = 0.08
    bindings: str = "configs/waw_bindings.json"
    record_every: int = 0  # record every k-th episode of each actor as a clip (0: never)
    # Film every game and keep the run's best -- highest round, then most kills, then most points -- as
    # best/best.mp4 in the run's directory (rl/best_episode.py)
    record_best: bool = True
    # Keep the agent's firsts -- first kill, door, mystery box, headshot, Ray Gun, round 5, ... -- as clips with
    # the game's sound, in video_dir's firsts/ (rl/moments.py)
    record_moments: bool = True
    # Keep one whole game every this many hours of training, in video_dir's progress/ (0: never)
    film_every_h: float = 1.0
    # Keep the record games -- shortest, most repairs, worst aim, longest without a kill -- in video_dir's records/
    record_records: bool = True
    # Where the video's films are kept (rl/keepsakes.py); "" for runs/video beside the runs, shared by every run
    # on this PC (a synthetic rehearsal keeps its own in the run's directory)
    video_dir: str = ""
    hear: bool = True  # a checkpoint trained with audio hears its own instance's sink
    # synthetic env: SyntheticConfig overrides (zombiesai/synthetic.py)
    synthetic: dict = field(default_factory=dict)
    actor_restarts: int = 20  # per actor, before the run gives up on it
    # Other PCs' games (rl/fleet.py): "host:port" to accept their workers on, "" for this machine's games only
    listen: str = ""
    amp: str = "auto"  # learner precision: auto (bf16 where native, else fp16 with loss scaling), off, bf16, fp16

    @property
    def env_name(self) -> str:
        return "real-waw" if self.env == "real" else self.env


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
