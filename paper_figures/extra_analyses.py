#!/usr/bin/env python3
"""
Extra analyses on the trained SAC checkpoint (no retraining required). Generates:

  fig6_critic_calibration.png       — Q(s_0, a_0) vs realized episode return
  fig7_robustness_reset_noise.png   — accuracy as a function of N(0, σ_θ) on initial joints
  fig8_pca_trajectory_projection.png — all eval episode trajectories on PC1-PC2
  fig9_policy_entropy_per_step.png   — policy entropy over the 40-step horizon

Run from repo root:
  python3 paper_figures/extra_analyses.py

Override checkpoint:
  PAPER_CHECKPOINT=/path/to/policy_*.pt python3 paper_figures/extra_analyses.py
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import fields, replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CFG, Config  # noqa: E402
from env import DartEnv  # noqa: E402
from evaluate import (  # noqa: E402
    collect_trajectories,
    load_actor,
    make_eval_env,
    stack_states_from_runs,
)
from models import CriticNet, GaussianPolicy  # noqa: E402
from train import env_action_torch, raw_policy_to_env_action  # noqa: E402
from utils import to_tensor  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "paper_figures")
EVAL_SEED_MAIN = 1
N_EPISODES = 400


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
def resolve_checkpoint() -> str:
    env_ckpt = os.environ.get("PAPER_CHECKPOINT")
    if env_ckpt and os.path.isfile(env_ckpt):
        return env_ckpt
    for rel in ("checkpoints/policy_finalSAC.pt", "checkpoints/policy_bestSAC.pt"):
        p = os.path.join(_ROOT, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No SAC checkpoint found (set PAPER_CHECKPOINT or place one in checkpoints/)"
    )


def load_critic(ckpt_path: str, cfg: Config, device: str) -> CriticNet | None:
    """Load q1 from the SAC checkpoint dict (returns None if absent)."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("q1")
    if sd is None:
        return None
    q = CriticNet(cfg.state_dim, cfg.action_dim).to(device)
    q.load_state_dict(sd)
    q.eval()
    return q


# ---------------------------------------------------------------------------
# Episode loop variants
# ---------------------------------------------------------------------------
@torch.no_grad()
def episode_with_critic_and_entropy(
    env: DartEnv,
    actor: GaussianPolicy,
    critic: CriticNet | None,
    device: str,
    *,
    init_noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run one episode. Returns:
      q0      : Q(s_0, a_0)  (None if critic absent)
      ret     : sum of step rewards (undiscounted)
      ret_g   : discounted sum with cfg.gamma=0.99
      entropy : per-step Gaussian entropy (length T)
      released: bool
      dist_m  : radial miss (NaN if not released)
    """
    s = env.reset()
    if init_noise_sigma > 0.0 and rng is not None:
        env._angles = (env._angles + rng.normal(0.0, init_noise_sigma, size=6).astype(np.float32))
        env._angles = np.clip(env._angles, -env.max_angle, env.max_angle).astype(np.float32)
        s = env._get_state()

    ent_steps: list[float] = []
    rewards: list[float] = []
    done = False
    last_info = None

    a0 = None
    s0_t = to_tensor(s, device=device).unsqueeze(0)
    raw0 = actor.mean_action(s0_t)
    a0_env = env_action_torch(raw0)  # for critic input

    while not done:
        st = to_tensor(s, device=device).unsqueeze(0)
        mean, log_std = actor.forward(st)
        # Differential entropy of the pre-tanh Gaussian (closed form). Sum over action dims.
        sigma = log_std.exp()
        ent_per_dim = 0.5 * (math.log(2.0 * math.pi * math.e)) + log_std  # 0.5*log(2*pi*e) + log_sigma
        ent_steps.append(float(ent_per_dim.sum(dim=-1).item()))
        raw = actor.mean_action(st).squeeze(0).cpu().numpy()
        if a0 is None:
            a0 = raw
        a = raw_policy_to_env_action(raw)
        s, r, done, step_out = env.step(a)
        rewards.append(float(r))
        last_info = step_out["info"]

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    gamma = 0.99
    discounts = gamma ** np.arange(rewards_arr.size, dtype=np.float64)
    ret_g = float((rewards_arr * discounts).sum())
    ret = float(rewards_arr.sum())

    q0_val: float | None = None
    if critic is not None:
        q0_val = float(critic(s0_t, a0_env).item())

    released = bool(last_info.released) if last_info is not None else False
    if released and last_info is not None and last_info.landing_xy is not None:
        dy, dz = last_info.landing_xy
        dist = float(math.hypot(dy - env.target_x, dz - env.target_y))
    else:
        dist = float("nan")

    return {
        "q0": q0_val,
        "ret": ret,
        "ret_g": ret_g,
        "entropy": np.asarray(ent_steps, dtype=np.float64),
        "released": released,
        "dist_m": dist,
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def fig6_critic_calibration(
    q0_vals: np.ndarray, ret_g: np.ndarray, ret_undisc: np.ndarray, out_path: str
) -> dict:
    """Scatter Q(s0,a0) vs realized return; report Pearson r and best-fit slope."""
    mask = np.isfinite(q0_vals) & np.isfinite(ret_g)
    q = q0_vals[mask]
    r = ret_g[mask]
    if q.size < 2:
        return {}
    pearson = float(np.corrcoef(q, r)[0, 1])
    slope, intercept = np.polyfit(q, r, 1)
    rmse = float(np.sqrt(np.mean((q - r) ** 2)))

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(q, r, s=14, alpha=0.55, color="steelblue", edgecolor="white", linewidth=0.3)
    lo, hi = float(min(q.min(), r.min())), float(max(q.max(), r.max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, label="y = x (perfect)")
    xs = np.linspace(q.min(), q.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="C3", lw=1.4, label=f"fit: slope={slope:.2f}")
    ax.set_xlabel(r"Critic estimate $Q(s_0, \mu_\phi(s_0))$")
    ax.set_ylabel(r"Realized discounted return $\sum_t \gamma^t r_t$")
    ax.set_title(
        f"Critic value calibration (N={q.size}, Pearson r={pearson:.3f}, RMSE={rmse:.2f})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pearson": pearson, "slope": float(slope), "intercept": float(intercept), "rmse": rmse, "n": int(q.size)}


def fig7_reset_noise(rows: list[dict], out_path: str) -> None:
    sigmas = np.array([r["sigma"] for r in rows], dtype=np.float64)
    mean_d = np.array([r["mean_dist"] for r in rows], dtype=np.float64)
    std_d = np.array([r["std_dist"] for r in rows], dtype=np.float64)
    rel_r = np.array([r["release_rate"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax0, ax1 = axes
    ax0.errorbar(sigmas, mean_d, yerr=std_d, fmt="o-", color="C0", capsize=3, lw=1.4)
    ax0.set_xlabel(r"Initial-angle noise $\sigma_\theta$ (rad)")
    ax0.set_ylabel("Radial miss (m), mean ± std")
    ax0.set_title("Accuracy vs. initial-condition noise")
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0.0)

    ax1.plot(sigmas, rel_r, "o-", color="C2", lw=1.4)
    ax1.set_xlabel(r"Initial-angle noise $\sigma_\theta$ (rad)")
    ax1.set_ylabel("Release rate")
    ax1.set_title("Release reliability vs. initial-condition noise")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        f"Robustness sweep (deterministic actor, N={rows[0]['n']} episodes per σ, eval seed={rows[0]['seed']})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig8_pca_trajectory_projection(runs: list[dict], out_path: str) -> dict:
    """Project per-episode (angles, velocities) trajectories onto PC1-PC2 of the global PCA."""
    X = stack_states_from_runs(runs)
    pca = PCA(n_components=2).fit(X)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    n_ep = len(runs)
    cmap = plt.colormaps["viridis"]
    for i, d in enumerate(runs):
        traj = np.concatenate([d["angles"], d["velocities"]], axis=1).astype(np.float64)
        z = pca.transform(traj)
        ax.plot(z[:, 0], z[:, 1], color=cmap(i / max(1, n_ep - 1)), lw=0.5, alpha=0.35)
        ax.scatter(z[0, 0], z[0, 1], color=cmap(i / max(1, n_ep - 1)), s=6, marker="o", alpha=0.45, edgecolor="none")
        ax.scatter(z[-1, 0], z[-1, 1], color=cmap(i / max(1, n_ep - 1)), s=14, marker="x", alpha=0.6)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"All {n_ep} eval trajectories projected onto PC1–PC2", fontsize=11)
    ax.grid(True, alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dimgray", markersize=6, label="episode start"),
        plt.Line2D([0], [0], marker="x", color="dimgray", markersize=8, lw=0, label="episode end / release"),
        plt.Line2D([0], [0], color="dimgray", lw=1.2, label="trajectory (color = episode index)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "n_episodes": n_ep,
    }


def fig9_policy_entropy(entropies: list[np.ndarray], out_path: str) -> dict:
    """Mean ± std of per-step Gaussian entropy across episodes (padded to longest)."""
    T_max = max(e.size for e in entropies)
    M = np.full((len(entropies), T_max), np.nan, dtype=np.float64)
    for i, e in enumerate(entropies):
        M[i, : e.size] = e
    mean_t = np.nanmean(M, axis=0)
    std_t = np.nanstd(M, axis=0)
    n_t = np.sum(~np.isnan(M), axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xs = np.arange(1, T_max + 1)
    ax.fill_between(xs, mean_t - std_t, mean_t + std_t, alpha=0.25, color="C0", label="±1 std")
    ax.plot(xs, mean_t, color="C0", lw=1.6, label="mean entropy")
    ax2 = ax.twinx()
    ax2.bar(xs, n_t, color="lightgray", alpha=0.55, width=0.85, zorder=0, label="episodes still active")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Pre-tanh Gaussian entropy (sum over 7 action dims)")
    ax2.set_ylabel("# episodes still running at step t", color="dimgray")
    ax2.tick_params(axis="y", labelcolor="dimgray")
    ax.set_title(f"Policy entropy across the throw (N={len(entropies)} episodes)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    finite_first = float(mean_t[0])
    finite_last = float(mean_t[~np.isnan(mean_t)][-1])
    return {
        "n_episodes": len(entropies),
        "entropy_step1_mean": finite_first,
        "entropy_last_mean": finite_last,
        "T_max": int(T_max),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = resolve_checkpoint()
    print(f"checkpoint: {ckpt_path}")
    actor, cfg = load_actor(ckpt_path, device)
    if not isinstance(actor, GaussianPolicy):
        raise RuntimeError(
            f"extra_analyses requires a SAC GaussianPolicy checkpoint; got {type(actor).__name__}"
        )
    critic = load_critic(ckpt_path, cfg, device)
    if critic is None:
        print("Note: checkpoint has no q1 — Fig 6 will be skipped.")

    # ---------------- Figs 6 & 9 (with critic+entropy traces) ----------------
    env = make_eval_env(cfg, EVAL_SEED_MAIN)
    q0_vals: list[float] = []
    rets_g: list[float] = []
    rets: list[float] = []
    entropies: list[np.ndarray] = []
    for _ in range(N_EPISODES):
        ep = episode_with_critic_and_entropy(env, actor, critic, device)
        q0_vals.append(ep["q0"] if ep["q0"] is not None else float("nan"))
        rets_g.append(ep["ret_g"])
        rets.append(ep["ret"])
        entropies.append(ep["entropy"])
    q0_arr = np.asarray(q0_vals, dtype=np.float64)
    rg_arr = np.asarray(rets_g, dtype=np.float64)
    r_arr = np.asarray(rets, dtype=np.float64)

    fig6_path = os.path.join(OUT_DIR, "fig6_critic_calibration.png")
    fig9_path = os.path.join(OUT_DIR, "fig9_policy_entropy_per_step.png")
    fig6_stats = (
        fig6_critic_calibration(q0_arr, rg_arr, r_arr, fig6_path) if critic is not None else {}
    )
    if critic is not None:
        print(f"Saved {fig6_path}")
    fig9_stats = fig9_policy_entropy(entropies, fig9_path)
    print(f"Saved {fig9_path}")

    # ---------------- Fig 8 (PCA trajectory projection) ----------------
    env_pca = make_eval_env(cfg, EVAL_SEED_MAIN)
    runs, _meta = collect_trajectories(env_pca, cfg, device, N_EPISODES, actor=actor, rng=None)
    fig8_path = os.path.join(OUT_DIR, "fig8_pca_trajectory_projection.png")
    fig8_stats = fig8_pca_trajectory_projection(runs, fig8_path)
    print(f"Saved {fig8_path}")

    # ---------------- Fig 7 (init-condition noise sweep) ----------------
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20]
    rng = np.random.default_rng(EVAL_SEED_MAIN + 7777)
    rows: list[dict] = []
    n_per = 200
    for sig in sigmas:
        env_n = make_eval_env(cfg, EVAL_SEED_MAIN + 17)
        dists: list[float] = []
        rels: list[float] = []
        for _ in range(n_per):
            ep = episode_with_critic_and_entropy(
                env_n, actor, None, device, init_noise_sigma=sig, rng=rng
            )
            rels.append(1.0 if ep["released"] else 0.0)
            if ep["released"] and math.isfinite(ep["dist_m"]):
                dists.append(ep["dist_m"])
        rows.append(
            {
                "sigma": sig,
                "mean_dist": float(np.mean(dists)) if dists else float("nan"),
                "std_dist": float(np.std(dists)) if dists else float("nan"),
                "release_rate": float(np.mean(rels)),
                "n": n_per,
                "seed": EVAL_SEED_MAIN + 17,
            }
        )
        print(
            f"  σθ={sig:.2f}: rel={rows[-1]['release_rate']:.3f} "
            f"mean_d={rows[-1]['mean_dist']:.4f} std_d={rows[-1]['std_dist']:.4f}"
        )
    fig7_path = os.path.join(OUT_DIR, "fig7_robustness_reset_noise.png")
    fig7_reset_noise(rows, fig7_path)
    print(f"Saved {fig7_path}")

    # ---------------- Append to figure_stats.txt ----------------
    stats_path = os.path.join(OUT_DIR, "figure_stats.txt")
    with open(stats_path, "a") as f:
        f.write("\n--- extra analyses (extra_analyses.py) ---\n")
        if critic is not None and fig6_stats:
            f.write(
                f"Fig 6 critic calibration: Pearson r={fig6_stats['pearson']:.4f}, "
                f"slope={fig6_stats['slope']:.4f}, intercept={fig6_stats['intercept']:.4f}, "
                f"RMSE={fig6_stats['rmse']:.4f}, N={fig6_stats['n']}\n"
            )
        else:
            f.write("Fig 6 critic calibration: skipped (no q1 in checkpoint)\n")
        f.write(
            "Fig 7 reset-noise sweep (release_rate, mean_dist, std_dist) per sigma_theta (rad):\n"
        )
        for r in rows:
            f.write(
                f"  sigma={r['sigma']:.2f}: rel={r['release_rate']:.3f}, "
                f"mean_dist={r['mean_dist']:.4f}, std_dist={r['std_dist']:.4f} "
                f"(n={r['n']}, eval_seed={r['seed']})\n"
            )
        f.write(
            f"Fig 8 PCA trajectory projection: PC1={fig8_stats['pc1_var']:.4f}, "
            f"PC2={fig8_stats['pc2_var']:.4f}, n_episodes={fig8_stats['n_episodes']}\n"
        )
        f.write(
            f"Fig 9 policy entropy: mean entropy at step 1 = {fig9_stats['entropy_step1_mean']:.3f}, "
            f"at last step = {fig9_stats['entropy_last_mean']:.3f}, "
            f"T_max={fig9_stats['T_max']}, N={fig9_stats['n_episodes']}\n"
        )
    print(f"Appended to {stats_path}")


if __name__ == "__main__":
    main()
