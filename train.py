"""
Actor-critic (DDPG-style) training for DartEnv.

Rollout: state -> policy (+ Gaussian exploration) -> env action -> env.step ->
store (s,a,r,s',done).

Updates:
  critic: MSE(Q(s,a), r + gamma * (1-done) * Q_target(s', mu_target(s')))
  policy: maximize Q(s, mu(s))  -> minimize -mean(Q(...))
  soft-update Q_target toward Q periodically.
"""

from __future__ import annotations

import argparse
import csv
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
from models import CriticNet, PolicyNet, make_target
from replay_buffer import ReplayBuffer
from utils import ensure_dir, set_seed, soft_update, to_tensor


def resolve_eval_interval(cfg: Config) -> int:
    """
    Evaluation CSV/checkpoint cadence. If cfg.eval_every_steps > 0, use it; otherwise space
    eval_num_checkpoints evaluations evenly across total_env_steps (default 10 checkpoints).
    """
    if cfg.eval_every_steps > 0:
        return max(1, int(cfg.eval_every_steps))
    k = max(1, int(cfg.eval_num_checkpoints))
    return max(1, int(cfg.total_env_steps) // k)


def truncate_csv_log(path: str | None) -> None:
    """Overwrite path with an empty file so the next append writes a fresh header + rows."""
    if path is None:
        return
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    open(path, "w", newline="").close()


def raw_policy_to_env_action(raw: np.ndarray) -> np.ndarray:
    """Map policy output [-1,1]^7 to env torques [-1,1]^6 + release [0,1]."""
    raw = np.asarray(raw, dtype=np.float32).reshape(7,)
    tau = np.clip(raw[:6], -1.0, 1.0)
    rel = float(np.clip((raw[6] + 1.0) / 2.0, 0.0, 1.0))
    return np.concatenate([tau, np.array([rel], dtype=np.float32)])


def exploration_noise(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0.0, cfg.action_noise_std, size=(6,)).astype(np.float32)
    n = np.clip(n, -cfg.action_noise_clip, cfg.action_noise_clip)
    rn = float(rng.normal(0.0, cfg.release_noise_std))
    rn = float(np.clip(rn, -0.3, 0.3))
    return np.concatenate([n, np.array([rn], dtype=np.float32)])


def env_action_torch(raw_t: torch.Tensor) -> torch.Tensor:
    """Batch: raw [-1,1] -> concatenated env action."""
    tau = torch.clamp(raw_t[:, :6], -1.0, 1.0)
    rel = torch.clamp((raw_t[:, 6:7] + 1.0) / 2.0, 0.0, 1.0)
    return torch.cat([tau, rel], dim=-1)


@torch.no_grad()
def evaluate_policy(
    env: DartEnv,
    actor: PolicyNet,
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
            raw = actor(st).squeeze(0).cpu().numpy()
            a = raw_policy_to_env_action(raw)
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


_TRAIN_METRIC_FIELDS = (
    "step",
    "mean_reward",
    "std_reward",
    "mean_reward_released",
    "std_reward_released",
    "mean_reward_timeout",
    "std_reward_timeout",
    "released_eval_count",
    "timeout_eval_count",
    "mean_landing_dist",
    "std_landing_dist",
    "release_rate",
    "mean_score",
)


def append_train_metrics_csv(path: str, step: int, stats: dict[str, float]) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    write_header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
    row = {k: stats.get(k, float("nan")) for k in _TRAIN_METRIC_FIELDS if k != "step"}
    row["step"] = step
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_TRAIN_METRIC_FIELDS))
        if write_header:
            w.writeheader()
        w.writerow(row)


_EPISODE_LOG_FIELDS = ("env_step", "episode_index", "episode_return", "episode_length", "released")


def append_episode_csv(
    path: str, env_step: int, episode_index: int, ep_ret: float, ep_len: int, released: bool
) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    write_header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
    row = {
        "env_step": env_step,
        "episode_index": episode_index,
        "episode_return": ep_ret,
        "episode_length": ep_len,
        "released": 1 if released else 0,
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_EPISODE_LOG_FIELDS))
        if write_header:
            w.writeheader()
        w.writerow(row)


def train(
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
    print(f"[train] eval every {eval_interval} env steps")
    rng = np.random.default_rng(cfg.seed + 999)

    actor = PolicyNet(cfg.state_dim, cfg.action_dim).to(device)
    critic = CriticNet(cfg.state_dim, cfg.action_dim).to(device)
    target_actor = make_target(actor).to(device)
    target_critic = make_target(critic).to(device)

    actor_opt = Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = Adam(critic.parameters(), lr=cfg.critic_lr)

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
    last100_releases: deque[float] = deque(maxlen=100)
    episode_storage_indices: list[int] = []

    for step in range(1, cfg.total_env_steps + 1):
        # --- action ---
        with torch.no_grad():
            st = to_tensor(s, device=device).unsqueeze(0)
            raw = actor(st).squeeze(0).cpu().numpy()
        if step < cfg.start_steps:
            raw = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
        else:
            raw = np.clip(raw + exploration_noise(cfg, rng), -1.0, 1.0)

        a_env = raw_policy_to_env_action(raw)
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
            released = float(released_flag)
            releases_in_last += int(released_flag)
            episodes_in_last += 1
            total_episodes += 1
            recent_returns.append(float(ep_reward))
            recent_releases.append(released)
            last100_releases.append(released)

            if total_episodes % 100 == 0 and len(last100_releases) == last100_releases.maxlen:
                rr100 = float(np.mean(last100_releases))
                if rr100 < 0.3:
                    print(
                        f"[WARN] release_rate over last 100 episodes = {rr100:.2f} "
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
                    f"[progress] env_step={step} episode={total_episodes} "
                    f"last_ep_return={ep_reward:.3f} ep_len={ep_len} released={int(released_flag)} | "
                    f"rolling_mean_return(n={len(recent_returns)})={rr:.3f} rolling_release_rate={relr:.2f}"
                )

            ep_reward = 0.0
            ep_len = 0

        # --- updates ---
        if len(buf) >= cfg.batch_size and step >= cfg.start_steps:
            min_frac = cfg.min_release_fraction if cfg.min_release_fraction > 0 else None
            for _ in range(cfg.updates_per_step):
                b = buf.sample(cfg.batch_size, min_frac)
                bs = to_tensor(b.state, device)
                ba = to_tensor(b.action, device)
                br = to_tensor(b.reward, device)
                bns = to_tensor(b.next_state, device)
                bd = to_tensor(b.done, device)

                with torch.no_grad():
                    # True DDPG target: use target_actor and target_critic
                    next_raw = target_actor(bns)
                    next_a = env_action_torch(next_raw)
                    target_q = br + cfg.gamma * (1.0 - bd) * target_critic(bns, next_a)

                q = critic(bs, ba)
                critic_loss = F.mse_loss(q, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                cur_raw = actor(bs)
                cur_a = env_action_torch(cur_raw)
                policy_loss = -critic(bs, cur_a).mean()

                actor_opt.zero_grad()
                policy_loss.backward()
                actor_opt.step()

                soft_update(target_critic, critic, cfg.tau)
                soft_update(target_actor, actor, cfg.tau)

        if step % eval_interval == 0 and step >= cfg.start_steps:
            stats = evaluate_policy(eval_env, actor, device, cfg.eval_episodes)
            print(f"[eval @ {step}] {stats}")
            if train_metrics_csv is not None:
                append_train_metrics_csv(train_metrics_csv, step, stats)
            released_count = float(stats.get("released_eval_count", 0.0))
            mean_rel = float(stats.get("mean_reward_released", float("nan")))
            mean_land = float(stats.get("mean_landing_dist", float("nan")))
            if released_count > 0 and np.isfinite(mean_rel) and np.isfinite(mean_land):
                composite = mean_rel - 10.0 * mean_land
                if composite > best_composite:
                    best_composite = composite
                    path = os.path.join(save_dir, "policy_best.pt")
                    torch.save(
                        {
                            "actor": actor.state_dict(),
                            "target_actor": target_actor.state_dict(),
                            "critic": critic.state_dict(),
                            "target_critic": target_critic.state_dict(),
                            "cfg": asdict(cfg),
                            "step": step,
                            "best_composite": composite,
                            "best_mean_reward_released": mean_rel,
                            "best_mean_landing_dist": mean_land,
                        },
                        path,
                    )
                    print(
                        f"[best @ {step}] composite={composite:.3f} "
                        f"(mean_reward_released={mean_rel:.3f}, mean_landing_dist={mean_land:.3f})"
                    )

        if step % 1000 == 0:
            rel_rate = releases_in_last / max(1, episodes_in_last)
            print(
                f"[1k-window @ env_step={step}] episodes_finished_in_window={episodes_in_last} "
                f"release_rate={rel_rate:.2f}"
            )
            releases_in_last = 0
            episodes_in_last = 0

    final_path = os.path.join(save_dir, "policy_final.pt")
    torch.save(
        {
            "actor": actor.state_dict(),
            "target_actor": target_actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "cfg": asdict(cfg),
            "step": cfg.total_env_steps,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train DDPG-style agent on DartEnv.")
    p.add_argument("--total-env-steps", type=int, default=None, help="Override CFG.total_env_steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for train_metrics.csv and train_episodes.csv (under project root)",
    )
    p.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to existing CSVs; default is to truncate them at the start of training",
    )
    p.add_argument(
        "--no-train-log",
        action="store_true",
        help="Do not append eval metrics to CSV",
    )
    p.add_argument(
        "--no-episode-log",
        action="store_true",
        help="Do not write train_episodes.csv (per-episode returns)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print rolling stats every N finished episodes (0 disables)",
    )
    args = p.parse_args()

    cfg = CFG
    if args.total_env_steps is not None:
        te = args.total_env_steps
        cfg = replace(
            cfg,
            total_env_steps=te,
            start_steps=min(cfg.start_steps, max(100, te // 10)),
        )
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)

    base = os.path.dirname(os.path.abspath(__file__))
    train_metrics_csv: str | None = None
    if not args.no_train_log:
        train_metrics_csv = os.path.join(base, args.log_dir, "train_metrics.csv")

    episode_log_csv: str | None = None
    if not args.no_episode_log:
        episode_log_csv = os.path.join(base, args.log_dir, "train_episodes.csv")

    if not args.append_logs:
        truncate_csv_log(train_metrics_csv)
        truncate_csv_log(episode_log_csv)

    train(
        cfg,
        train_metrics_csv=train_metrics_csv,
        episode_log_csv=episode_log_csv,
        progress_every_episodes=max(0, args.progress_every),
    )


if __name__ == "__main__":
    main()
