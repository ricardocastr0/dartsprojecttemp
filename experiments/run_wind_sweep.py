#!/usr/bin/env python3
"""
Zero-shot wind robustness: load checkpoints/policy_finalSAC.pt and evaluate under
several fixed wind vectors plus random wind on the sphere (radius 1 m/s).

Does not retrain. Uses deterministic actor (tanh mean).

Outputs:
  experiments/results/wind_sweep.csv
  experiments/figures/wind_sweep.png
"""
from __future__ import annotations

import csv
import math
import os
import sys

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EXP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config  # noqa: E402
from env import DartEnv  # noqa: E402
from evaluate import load_actor  # noqa: E402
from models import GaussianPolicy  # noqa: E402
from train import raw_policy_to_env_action  # noqa: E402
from utils import set_seed, to_tensor  # noqa: E402

CHECKPOINT_REL = os.path.join("checkpoints", "policy_finalSAC.pt")
EVAL_EPISODES = 200
EVAL_SEED = 42


def _ensure_dirs() -> tuple[str, str]:
    res = os.path.join(_ROOT, "experiments", "results")
    fig = os.path.join(_ROOT, "experiments", "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    return res, fig


def checkpoint_path() -> str:
    p = os.path.join(_ROOT, CHECKPOINT_REL)
    if not os.path.isfile(p):
        print(f"ERROR: checkpoint not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def make_env_like_eval(cfg: Config, wind_xyz: tuple[float, float, float]) -> DartEnv:
    return DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=0,
        min_release_steps=cfg.min_release_steps,
        seed=EVAL_SEED,
        wind_xyz=wind_xyz,
        k_tau=cfg.k_tau,
        damping=cfg.damping,
        max_vel=cfg.max_vel,
        reset_angle_low=cfg.reset_angle_low,
        reset_angle_high=cfg.reset_angle_high,
        reset_vel_low=cfg.reset_vel_low,
        reset_vel_high=cfg.reset_vel_high,
    )


def eval_fixed_wind(
    env: DartEnv,
    policy: GaussianPolicy,
    device: str,
    n_episodes: int,
) -> dict[str, float]:
    landing_dists: list[float] = []
    release_flags: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for _ in range(n_episodes):
            s = env.reset()
            done = False
            last_info = None
            while not done:
                st = to_tensor(s, device=device).unsqueeze(0)
                raw = policy.mean_action(st)
                a = raw_policy_to_env_action(raw.squeeze(0).cpu().numpy())
                s, _, done, step_out = env.step(a)
                last_info = step_out["info"]
            if last_info is not None and last_info.landing_xy is not None:
                dy, dz = last_info.landing_xy
                d = float(np.hypot(dy - env.target_x, dz - env.target_y))
                landing_dists.append(d)
                release_flags.append(1.0 if last_info.released else 0.0)
                scores.append(float(last_info.score))
    return {
        "mean_miss_m": float(np.mean(landing_dists)) if landing_dists else float("nan"),
        "std_miss_m": float(np.std(landing_dists)) if landing_dists else float("nan"),
        "release_rate": float(np.mean(release_flags)) if release_flags else float("nan"),
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
    }


def eval_random_sphere_wind(
    env: DartEnv,
    policy: GaussianPolicy,
    device: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    landing_dists: list[float] = []
    release_flags: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for _ in range(n_episodes):
            v = rng.normal(size=3).astype(np.float64)
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-12:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                nrm = 1.0
            radius_ms = 1.0
            w = (radius_ms * v / nrm).tolist()
            env.wind_xyz = (float(w[0]), float(w[1]), float(w[2]))
            s = env.reset()
            done = False
            last_info = None
            while not done:
                st = to_tensor(s, device=device).unsqueeze(0)
                raw = policy.mean_action(st)
                a = raw_policy_to_env_action(raw.squeeze(0).cpu().numpy())
                s, _, done, step_out = env.step(a)
                last_info = step_out["info"]
            if last_info is not None and last_info.landing_xy is not None:
                dy, dz = last_info.landing_xy
                d = float(np.hypot(dy - env.target_x, dz - env.target_y))
                landing_dists.append(d)
                release_flags.append(1.0 if last_info.released else 0.0)
                scores.append(float(last_info.score))
    return {
        "mean_miss_m": float(np.mean(landing_dists)) if landing_dists else float("nan"),
        "std_miss_m": float(np.std(landing_dists)) if landing_dists else float("nan"),
        "release_rate": float(np.mean(release_flags)) if release_flags else float("nan"),
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
    }


def wind_speed_label(w: tuple[float, float, float]) -> float:
    return float(math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2))


def main() -> None:
    set_seed(EVAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir, figures_dir = _ensure_dirs()
    ckpt = checkpoint_path()

    policy, cfg = load_actor(ckpt, device)
    if not isinstance(policy, GaussianPolicy):
        print("ERROR: expected SAC GaussianPolicy checkpoint", file=sys.stderr)
        sys.exit(1)

    conditions: list[tuple[str, tuple[float, float, float] | None, float]] = [
        ("no_wind", (0.0, 0.0, 0.0), 0.0),
        ("light_crosswind", (0.0, 0.5, 0.0), wind_speed_label((0.0, 0.5, 0.0))),
        ("moderate_crosswind", (0.0, 1.0, 0.0), wind_speed_label((0.0, 1.0, 0.0))),
        ("strong_crosswind", (0.0, 2.0, 0.0), wind_speed_label((0.0, 2.0, 0.0))),
        ("headwind", (-1.0, 0.0, 0.0), wind_speed_label((-1.0, 0.0, 0.0))),
        ("random_sphere", None, 1.0),
    ]

    rng = np.random.default_rng(EVAL_SEED)
    rows: list[dict[str, float | str]] = []

    for name, wfix, wspeed in conditions:
        print(f"\n=== Condition: {name} ===")
        if wfix is not None:
            env = make_env_like_eval(cfg, wfix)
            stats = eval_fixed_wind(env, policy, device, EVAL_EPISODES)
        else:
            env = make_env_like_eval(cfg, (0.0, 0.0, 0.0))
            stats = eval_random_sphere_wind(env, policy, device, EVAL_EPISODES, rng)
        print(stats)
        rows.append(
            {
                "condition": name,
                "wind_speed_ms": wspeed,
                "mean_miss_m": stats["mean_miss_m"],
                "std_miss_m": stats["std_miss_m"],
                "release_rate": stats["release_rate"],
                "mean_score": stats["mean_score"],
            }
        )

    out_csv = os.path.join(results_dir, "wind_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = ("condition", "wind_speed_ms", "mean_miss_m", "std_miss_m", "release_rate", "mean_score")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")

    labels = [r["condition"] for r in rows]
    means = [r["mean_miss_m"] for r in rows]
    stds = [r["std_miss_m"] for r in rows]
    rel = [r["release_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85, ecolor="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean radial miss (m)")
    ax.set_title("Zero-shot wind sweep (policy trained with zero wind)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, rr) in enumerate(zip(means, rel, strict=True)):
        ax.text(i, m + stds[i] + 0.002, f"rel={rr:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, "wind_sweep.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
