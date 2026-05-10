from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Problem dimensions
    # state = [6 joint angles, 6 joint velocities, optimal_release_step / 35.0] -> 13
    state_dim: int = 13
    action_dim: int = 7  # 6 torques + 1 release signal

    # Environment
    max_steps: int = 40
    dt: float = 0.04
    k_tau: float = 7.5
    damping: float = 1.2
    max_vel: float = 12.0
    reset_angle_low: float = -0.3
    reset_angle_high: float = 0.3
    reset_vel_low: float = -0.05
    reset_vel_high: float = 0.05
    release_threshold: float = 0.40
    # Terminal reward add-on when released (set 0 for strict "no completion bonus" experiments).
    release_completion_bonus: float = 0.0
    # Linear taper: episode starts with p_force = max(0, 1 - completed_episodes / K) for K episodes.
    curriculum_taper_episodes: int = 2000
    min_release_steps: int = 10

    # RL
    gamma: float = 0.99
    tau: float = 0.005  # soft update rate for target critic

    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256

    # Replay buffer
    buffer_size: int = 300_000
    start_steps: int = 2_000  # random/exploration steps before training updates

    # Exploration noise (Gaussian on torques, mild on release)
    action_noise_std: float = 0.20
    release_noise_std: float = 0.25
    action_noise_clip: float = 0.50

    # Training loop
    total_env_steps: int = 100_000
    updates_per_step: int = 1
    # eval_every_steps > 0: fixed interval; == 0: use total_env_steps // eval_num_checkpoints (~10 evals/run).
    eval_every_steps: int = 0
    eval_num_checkpoints: int = 10
    eval_episodes: int = 10

    # SAC-only (trainSAC.py): prioritized replay (Schaul et al.)
    prioritized_replay: bool = True
    per_alpha: float = 0.6  # priority exponent on TD magnitude
    per_beta_start: float = 0.4  # importance-sampling exponent (annealed to per_beta_end)
    per_beta_end: float = 1.0
    per_eps: float = 1e-6

    # SAC (trainSAC.py): minimum fraction of batch from release-ending episode transitions
    min_release_fraction: float = 0.5

    # Misc
    seed: int = 0
    device: str = "cpu"
    save_dir: str = "checkpoints"


CFG = Config()
