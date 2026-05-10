"""
Soft Actor-Critic (SAC) training for DartEnv.

Stochastic tanh-Gaussian policy, twin Q-networks, automatic entropy temperature.
Same env, buffer, and config as train.py; SAC-specific checkpoints and CSV names.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CFG, Config
from env import DartEnv
from models import CriticNet, GaussianPolicy, make_target
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from train import (
    append_episode_csv,
    append_train_metrics_csv,
    env_action_torch,
    raw_policy_to_env_action,
    resolve_eval_interval,
    truncate_csv_log,
)
from utils import ensure_dir, set_seed, soft_update, to_tensor


@torch.no_grad()
def evaluate_sac_policy(
    env: DartEnv,
    policy: GaussianPolicy,
    device: str,
    n_episodes: int,
) -> dict[str, float]:
    rewards: list[float] = []
    rewards_released: list[float] = []
    rewards_timeout: list[float] = []
    landing_dists: list[float] = []
    release_flags: list[float] = []
    scores: list[float] = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r = 0.0
        done = False
        last_info = None
        while not done:
            st = to_tensor(s, device=device).unsqueeze(0)
            raw = policy.mean_action(st)
            a = raw_policy_to_env_action(raw.squeeze(0).cpu().numpy())
            s, r, done, step_out = env.step(a)
            ep_r += r
            last_info = step_out["info"]
        rewards.append(ep_r)
        if last_info is not None and last_info.landing_xy is not None:
            dy, dz = last_info.landing_xy
            d = float(np.hypot(dy - env.target_x, dz - env.target_y))
            landing_dists.append(d)
            release_flags.append(1.0 if last_info.released else 0.0)
            scores.append(float(last_info.score))
            if last_info.released:
                rewards_released.append(ep_r)
            else:
                rewards_timeout.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_reward_released": float(np.mean(rewards_released)) if rewards_released else float("nan"),
        "std_reward_released": float(np.std(rewards_released)) if rewards_released else float("nan"),
        "mean_reward_timeout": float(np.mean(rewards_timeout)) if rewards_timeout else float("nan"),
        "std_reward_timeout": float(np.std(rewards_timeout)) if rewards_timeout else float("nan"),
        "released_eval_count": float(len(rewards_released)),
        "timeout_eval_count": float(len(rewards_timeout)),
        "mean_landing_dist": float(np.mean(landing_dists)) if landing_dists else float("nan"),
        "std_landing_dist": float(np.std(landing_dists)) if landing_dists else float("nan"),
        "release_rate": float(np.mean(release_flags)) if release_flags else float("nan"),
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
    }


def train_sac(
    cfg: Config,
    *,
    train_metrics_csv: str | None = None,
    episode_log_csv: str | None = None,
    progress_every_episodes: int = 25,
) -> None:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=cfg.curriculum_taper_episodes,
        min_release_steps=cfg.min_release_steps,
        seed=cfg.seed,
        k_tau=cfg.k_tau,
        damping=cfg.damping,
        max_vel=cfg.max_vel,
        reset_angle_low=cfg.reset_angle_low,
        reset_angle_high=cfg.reset_angle_high,
        reset_vel_low=cfg.reset_vel_low,
        reset_vel_high=cfg.reset_vel_high,
    )
    eval_env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=0,
        min_release_steps=cfg.min_release_steps,
        seed=cfg.seed + 777777,
        k_tau=cfg.k_tau,
        damping=cfg.damping,
        max_vel=cfg.max_vel,
        reset_angle_low=cfg.reset_angle_low,
        reset_angle_high=cfg.reset_angle_high,
        reset_vel_low=cfg.reset_vel_low,
        reset_vel_high=cfg.reset_vel_high,
    )
    eval_interval = resolve_eval_interval(cfg)
    print(f"[SAC] eval every {eval_interval} env steps ({cfg.total_env_steps} total)")
    rng = np.random.default_rng(cfg.seed + 999)

    policy = GaussianPolicy(cfg.state_dim, cfg.action_dim).to(device)
    q1 = CriticNet(cfg.state_dim, cfg.action_dim).to(device)
    q2 = CriticNet(cfg.state_dim, cfg.action_dim).to(device)
    q1_t = make_target(q1).to(device)
    q2_t = make_target(q2).to(device)

    policy_opt = Adam(policy.parameters(), lr=cfg.actor_lr)
    q1_opt = Adam(q1.parameters(), lr=cfg.critic_lr)
    q2_opt = Adam(q2.parameters(), lr=cfg.critic_lr)

    # Less aggressive than -action_dim (=-7); the release dimension is effectively
    # discrete (commit) so very high target entropy fights commitment and worsens collapse.
    target_entropy = -7.0
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    alpha_opt = Adam([log_alpha], lr=cfg.actor_lr)

    buf: ReplayBuffer | PrioritizedReplayBuffer
    if cfg.prioritized_replay:
        buf = PrioritizedReplayBuffer(
            cfg.state_dim,
            cfg.action_dim,
            cfg.buffer_size,
            seed=cfg.seed + 42,
            alpha=cfg.per_alpha,
            eps=cfg.per_eps,
        )
    else:
        buf = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size, seed=cfg.seed + 42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.save_dir)
    ensure_dir(save_dir)

    s = env.reset()
    ep_reward = 0.0
    ep_len = 0
    best_composite = -1e18
    releases_in_last = 0
    episodes_in_last = 0
    total_episodes = 0
    recent_returns: deque[float] = deque(maxlen=200)
    recent_releases: deque[float] = deque(maxlen=200)
    # Sliding window for early-warning release-rate collapse detection (every 100 episodes).
    last100_releases: deque[float] = deque(maxlen=100)
    episode_storage_indices: list[int] = []

    for step in range(1, cfg.total_env_steps + 1):
        with torch.no_grad():
            st = to_tensor(s, device=device).unsqueeze(0)
            if step < cfg.start_steps:
                raw = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(1, 7))).float().to(device)
            else:
                raw, _ = policy.sample(st)
            a_env = env_action_torch(raw).squeeze(0).cpu().numpy()

        s2, r, done, step_out = env.step(a_env)
        slot = buf.add(s, a_env, r, s2, done)
        episode_storage_indices.append(slot)

        if done:
            released_flag = bool(step_out["info"].released)
            buf.mark_episode(episode_storage_indices, released=released_flag)
            episode_storage_indices.clear()
            s = env.reset()
        else:
            s = s2
        ep_reward += r
        ep_len += 1
        if done:
            releases_in_last += int(released_flag)
            episodes_in_last += 1
            total_episodes += 1
            recent_returns.append(float(ep_reward))
            recent_releases.append(float(released_flag))
            last100_releases.append(float(released_flag))

            if total_episodes % 100 == 0 and len(last100_releases) == last100_releases.maxlen:
                rr100 = float(np.mean(last100_releases))
                if rr100 < 0.3:
                    print(
                        f"[SAC WARN] release_rate over last 100 episodes = {rr100:.2f} "
                        f"(< 0.3) at env_step={step} episode={total_episodes} -- "
                        "possible release/commitment collapse",
                        file=sys.stderr,
                    )

            if episode_log_csv is not None:
                append_episode_csv(
                    episode_log_csv,
                    step,
                    total_episodes,
                    float(ep_reward),
                    ep_len,
                    released_flag,
                )

            if progress_every_episodes > 0 and total_episodes % progress_every_episodes == 0:
                rr = float(np.mean(recent_returns))
                relr = float(np.mean(recent_releases))
                print(
                    f"[SAC progress] env_step={step} episode={total_episodes} "
                    f"last_ep_return={ep_reward:.3f} ep_len={ep_len} released={int(released_flag)} | "
                    f"rolling_mean_return(n={len(recent_returns)})={rr:.3f} rolling_release_rate={relr:.2f}"
                )

            ep_reward = 0.0
            ep_len = 0

        if len(buf) >= cfg.batch_size and step >= cfg.start_steps:
            beta = cfg.per_beta_start + (cfg.per_beta_end - cfg.per_beta_start) * min(
                1.0, float(step) / float(max(1, cfg.total_env_steps))
            )
            for _ in range(cfg.updates_per_step):
                min_frac = cfg.min_release_fraction if cfg.min_release_fraction > 0 else None
                if cfg.prioritized_replay:
                    b = buf.sample(cfg.batch_size, beta, min_frac)
                    iw = to_tensor(b.weights, device)
                else:
                    b = buf.sample(cfg.batch_size, min_frac)
                    iw = None

                bs = to_tensor(b.state, device)
                ba = to_tensor(b.action, device)
                br = to_tensor(b.reward, device)
                bns = to_tensor(b.next_state, device)
                bd = to_tensor(b.done, device)

                alpha = log_alpha.exp().clamp(min=1e-8, max=100.0)

                with torch.no_grad():
                    next_raw, next_log_pi = policy.sample(bns)
                    next_a = env_action_torch(next_raw)
                    q_next = torch.min(q1_t(bns, next_a), q2_t(bns, next_a)) - alpha * next_log_pi
                    target_q = br + cfg.gamma * (1.0 - bd) * q_next

                q1_val = q1(bs, ba)
                q2_val = q2(bs, ba)
                if iw is not None:
                    q1_loss = (iw * (q1_val - target_q).pow(2)).mean()
                    q2_loss = (iw * (q2_val - target_q).pow(2)).mean()
                    with torch.no_grad():
                        td_mag = torch.max(
                            (q1_val - target_q).abs(),
                            (q2_val - target_q).abs(),
                        ).cpu().numpy()
                    buf.update_priorities(b.indices, td_mag)
                else:
                    q1_loss = F.mse_loss(q1_val, target_q)
                    q2_loss = F.mse_loss(q2_val, target_q)

                q1_opt.zero_grad()
                q1_loss.backward()
                q1_opt.step()
                q2_opt.zero_grad()
                q2_loss.backward()
                q2_opt.step()

                raw_pi, log_pi = policy.sample(bs)
                a_pi = env_action_torch(raw_pi)
                q_pi = torch.min(q1(bs, a_pi), q2(bs, a_pi))
                policy_loss = (alpha.detach() * log_pi - q_pi).mean()

                policy_opt.zero_grad()
                policy_loss.backward()
                policy_opt.step()

                alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                soft_update(q1_t, q1, cfg.tau)
                soft_update(q2_t, q2, cfg.tau)

        if step % eval_interval == 0 and step >= cfg.start_steps:
            stats = evaluate_sac_policy(eval_env, policy, device, cfg.eval_episodes)
            print(f"[SAC eval @ {step}] {stats}")
            if train_metrics_csv is not None:
                append_train_metrics_csv(train_metrics_csv, step, stats)
            # Composite "best" criterion: prioritize throw quality over timeout avoidance.
            # Requires at least one released eval episode so mean_reward_released and
            # mean_landing_dist are well-defined; otherwise skip saving.
            released_count = float(stats.get("released_eval_count", 0.0))
            mean_rel = float(stats.get("mean_reward_released", float("nan")))
            mean_land = float(stats.get("mean_landing_dist", float("nan")))
            if released_count > 0 and np.isfinite(mean_rel) and np.isfinite(mean_land):
                composite = mean_rel - 10.0 * mean_land
                if composite > best_composite:
                    best_composite = composite
                    path = os.path.join(save_dir, "policy_bestSAC.pt")
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "q1": q1.state_dict(),
                            "q2": q2.state_dict(),
                            "q1_target": q1_t.state_dict(),
                            "q2_target": q2_t.state_dict(),
                            "log_alpha": log_alpha.detach().cpu(),
                            "cfg": asdict(cfg),
                            "step": step,
                            "algorithm": "SAC",
                            "best_composite": composite,
                            "best_mean_reward_released": mean_rel,
                            "best_mean_landing_dist": mean_land,
                        },
                        path,
                    )
                    print(
                        f"[SAC best @ {step}] composite={composite:.3f} "
                        f"(mean_reward_released={mean_rel:.3f}, mean_landing_dist={mean_land:.3f})"
                    )

        if step % 1000 == 0:
            rel_rate = releases_in_last / max(1, episodes_in_last)
            print(
                f"[SAC 1k-window @ env_step={step}] episodes_finished_in_window={episodes_in_last} "
                f"release_rate={rel_rate:.2f}"
            )
            releases_in_last = 0
            episodes_in_last = 0

    final_path = os.path.join(save_dir, "policy_finalSAC.pt")
    torch.save(
        {
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "q1_target": q1_t.state_dict(),
            "q2_target": q2_t.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "cfg": asdict(cfg),
            "step": cfg.total_env_steps,
            "algorithm": "SAC",
        },
        final_path,
    )
    print(f"Saved SAC final checkpoint to {final_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train SAC on DartEnv.")
    p.add_argument("--total-env-steps", type=int, default=None)
    p.add_argument(
        "--total-steps",
        type=int,
        default=None,
        dest="total_steps_alias",
        help="Alias for --total-env-steps (paper checklist compatibility)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--eval-every-steps",
        type=int,
        default=None,
        help="Fixed eval interval (env steps). Omit to use total_env_steps // eval_num_checkpoints (~10/run).",
    )
    p.add_argument(
        "--eval-num-checkpoints",
        type=int,
        default=None,
        help="When eval_every_steps is auto (0), eval this many times evenly (default from config, usually 10).",
    )
    p.add_argument("--eval-episodes", type=int, default=None, help="Override number of eval episodes per checkpoint")
    p.add_argument(
        "--curriculum-taper-episodes",
        type=int,
        default=None,
        help="Override curriculum_taper_episodes (linear forced-release decay length)",
    )
    p.add_argument(
        "--curriculum-episodes",
        type=int,
        default=None,
        help="Deprecated: use --curriculum-taper-episodes (sets taper length if provided)",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for train_metrics_SAC.csv and train_episodes_SAC.csv",
    )
    p.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to SAC CSVs instead of truncating at start",
    )
    p.add_argument("--no-train-log", action="store_true")
    p.add_argument("--no-episode-log", action="store_true")
    p.add_argument("--no-prioritized-replay", action="store_true", help="Use uniform replay (disable PER for SAC)")
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    cfg = CFG
    te_override = args.total_env_steps
    if te_override is None and getattr(args, "total_steps_alias", None) is not None:
        te_override = args.total_steps_alias
    if te_override is not None:
        te = int(te_override)
        cfg = replace(
            cfg,
            total_env_steps=te,
            start_steps=min(cfg.start_steps, max(100, te // 10)),
        )
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.eval_every_steps is not None:
        cfg = replace(cfg, eval_every_steps=max(1, int(args.eval_every_steps)))
    if args.eval_num_checkpoints is not None:
        cfg = replace(cfg, eval_num_checkpoints=max(1, int(args.eval_num_checkpoints)))
    if args.eval_episodes is not None:
        cfg = replace(cfg, eval_episodes=max(1, int(args.eval_episodes)))
    taper_override = args.curriculum_taper_episodes
    if taper_override is None and args.curriculum_episodes is not None:
        print(
            "Warning: --curriculum-episodes is deprecated; use --curriculum-taper-episodes",
            file=sys.stderr,
        )
        taper_override = args.curriculum_episodes
    if taper_override is not None:
        cfg = replace(cfg, curriculum_taper_episodes=max(0, int(taper_override)))
    if args.no_prioritized_replay:
        cfg = replace(cfg, prioritized_replay=False)

    base = os.path.dirname(os.path.abspath(__file__))
    train_metrics_csv: str | None = None
    if not args.no_train_log:
        train_metrics_csv = os.path.join(base, args.log_dir, "train_metrics_SAC.csv")

    episode_log_csv: str | None = None
    if not args.no_episode_log:
        episode_log_csv = os.path.join(base, args.log_dir, "train_episodes_SAC.csv")

    if not args.append_logs:
        truncate_csv_log(train_metrics_csv)
        truncate_csv_log(episode_log_csv)

    train_sac(
        cfg,
        train_metrics_csv=train_metrics_csv,
        episode_log_csv=episode_log_csv,
        progress_every_episodes=max(0, args.progress_every),
    )


if __name__ == "__main__":
    main()
