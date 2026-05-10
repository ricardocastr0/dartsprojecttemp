#!/usr/bin/env python3
"""
PCA comparison: zero-pad trajectories to T_max vs truncate all episodes to T_min.

Loads checkpoints/policy_finalSAC.pt, runs 400 deterministic evaluation episodes,
then stacks joint states [angles | velocities] (12-D per timestep) two ways.

Outputs:
  experiments/results/pca_comparison.csv
  experiments/figures/pca_comparison.png
"""
from __future__ import annotations

import csv
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
from sklearn.decomposition import PCA

from evaluate import collect_trajectories, effective_dof_95, load_actor, make_eval_env  # noqa: E402
from utils import set_seed  # noqa: E402

CHECKPOINT_REL = os.path.join("checkpoints", "policy_finalSAC.pt")
EPISODES = 400
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


def stack_trajectories(runs: list[dict]) -> tuple[list[np.ndarray], int, int]:
    mats: list[np.ndarray] = []
    lengths: list[int] = []
    for d in runs:
        ang = np.asarray(d["angles"], dtype=np.float64)
        vel = np.asarray(d["velocities"], dtype=np.float64)
        m = np.concatenate([ang, vel], axis=1)
        mats.append(m)
        lengths.append(int(m.shape[0]))
    if not lengths:
        print("ERROR: no trajectory data", file=sys.stderr)
        sys.exit(1)
    return mats, max(lengths), min(lengths)


def pad_stack(mats: list[np.ndarray], t_pad: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for m in mats:
        t = m.shape[0]
        if t < t_pad:
            pad = np.zeros((t_pad - t, 12), dtype=np.float64)
            m = np.vstack([m, pad])
        elif t > t_pad:
            m = m[:t_pad]
        rows.append(m)
    return np.vstack(rows)


def truncate_stack(mats: list[np.ndarray], t_cut: int) -> np.ndarray:
    rows = [m[:t_cut] for m in mats]
    return np.vstack(rows)


def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, PCA]:
    n_comp = min(12, max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    ev = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    # Pad to 12 if fewer components (degenerate sample)
    if ev.size < 12:
        ev = np.pad(ev, (0, 12 - ev.size))
    cum = np.cumsum(ev[:12])
    dof = effective_dof_95(ev[:12])
    return ev[:12], cum[:12], dof, pca


def plot_side_by_side(
    ev_pad: np.ndarray,
    cum_pad: np.ndarray,
    ev_trunc: np.ndarray,
    cum_trunc: np.ndarray,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    titles = ("Zero-padded to T_max", "Truncated to T_min")
    for ax_i, (ev, cum, title) in enumerate(
        zip((ev_pad, ev_trunc), (cum_pad, cum_trunc), titles, strict=True)
    ):
        ax1 = axes[ax_i]
        xs = np.arange(1, 13)
        ax1.bar(xs, ev, color="C0", alpha=0.85)
        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Explained variance ratio", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_xticks(xs)
        ax2 = ax1.twinx()
        ax2.plot(xs, cum, color="C1", marker="o", lw=2)
        ax2.axhline(0.95, linestyle="--", color="red", lw=1, alpha=0.8)
        ax2.set_ylabel("Cumulative explained variance", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(0.0, 1.02)
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    set_seed(EVAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir, figures_dir = _ensure_dirs()
    ckpt = checkpoint_path()

    print(
        "Note: 'zero-padded' pads each episode with zero-state rows after release; "
        "evaluate.py concatenates raw timesteps with no episode alignment — "
        "that differs from both methods below.\n"
    )

    actor, cfg = load_actor(ckpt, device)
    env = make_eval_env(cfg, EVAL_SEED)
    runs, _ = collect_trajectories(env, cfg, device, EPISODES, actor=actor, rng=None)

    mats, t_max, t_min = stack_trajectories(runs)
    print(f"T_max = {t_max}, T_min = {t_min} (from {len(runs)} episodes)")

    X_pad = pad_stack(mats, t_max)
    X_trunc = truncate_stack(mats, t_min)

    ev_p, cum_p, dof_p, _ = run_pca(X_pad)
    ev_t, cum_t, dof_t, _ = run_pca(X_trunc)

    pc12_p = float(ev_p[0] + ev_p[1])
    pc12_t = float(ev_t[0] + ev_t[1])

    print(
        f"Zero-padded:  PC1+PC2 = {100.0 * pc12_p:.2f}%, eff DOF (95%) = {dof_p}\n"
        f"Truncated:      PC1+PC2 = {100.0 * pc12_t:.2f}%, eff DOF (95%) = {dof_t}"
    )

    csv_path = os.path.join(results_dir, "pca_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ("method", "pc_index", "explained_variance", "cumulative_variance", "effective_dof_95")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for method, ev, cum, dof in (
            ("zero_padded", ev_p, cum_p, dof_p),
            ("truncated", ev_t, cum_t, dof_t),
        ):
            for i in range(12):
                w.writerow(
                    {
                        "method": method,
                        "pc_index": i + 1,
                        "explained_variance": float(ev[i]),
                        "cumulative_variance": float(cum[i]),
                        "effective_dof_95": int(dof),
                    }
                )
    print(f"Wrote {csv_path}")

    fig_path = os.path.join(figures_dir, "pca_comparison.png")
    plot_side_by_side(ev_p, cum_p, ev_t, cum_t, fig_path)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
