#!/usr/bin/env python3
"""
Generate paper figures (300 DPI). Run from repo root:
  python3 paper_figures/generate_paper_figures.py

Checkpoint: ``checkpoints/policy_finalSAC.pt`` unless ``PAPER_CHECKPOINT`` is set
(see paper_figures/analysis_decisions.md for the best-vs-final decision).
Training curves (Fig 1): ``PAPER_TRAIN_METRICS`` or first match among
``paper_runs/regulation_1m/train_metrics_SAC.csv``, ``1MilStats/1Mil_metrics.csv`` (bundled 1M run), ``logs/...``

Fig 2 tiers match env dartboard ring bonuses (see ``R_INNER_BULL_M``, ``R_OUTER_BULL_M``, ``R_DOUBLES_WIRE_M``).
Fig 5 overlays inner/outer bull radii on landing-density contours.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from scipy.ndimage import gaussian_filter

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.decomposition import PCA

# Paper eval defaults (checkpoint resolved at runtime; see resolve_paper_checkpoint())
EVAL_SEED_MAIN = 1  # figures 2, 3, 5
# Drop rare huge misses from density-plot input (full-sample stats still from all episodes).
LANDING_DATA_RMAX_M = 0.22
# Plot window half-width (m): ±10 cm so landing spread and mm-scale bull rings are visible.
LANDING_VIEW_HALF_M = 0.10
LANDING_CONTOUR_BINS = 72
LANDING_CONTOUR_GAUSS_SIGMA = 1.35

# Repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluate import (  # noqa: E402
    collect_trajectories,
    load_actor,
    make_eval_env,
    stack_states_from_runs,
)
from env import (  # noqa: E402
    DartEnv,
    R_DOUBLES_WIRE_M,
    R_INNER_BULL_M,
    R_OUTER_BULL_M,
    dartboard_score_plane,
)


def _first_existing_file(rel_candidates: list[str]) -> str | None:
    for rel in rel_candidates:
        p = os.path.join(_ROOT, rel)
        if os.path.isfile(p):
            return p
    return None


def resolve_paper_checkpoint() -> str:
    env_ckpt = os.environ.get("PAPER_CHECKPOINT")
    if env_ckpt:
        if os.path.isfile(env_ckpt):
            return env_ckpt
        raise FileNotFoundError(f"PAPER_CHECKPOINT set but not found: {env_ckpt}")
    p = _first_existing_file(
        [
            "checkpoints/policy_finalSAC.pt",
            "checkpoints/policy_bestSAC.pt",
            "paper_runs/policy_finalSAC.pt",
        ]
    )
    if p is None:
        raise FileNotFoundError(
            "No SAC checkpoint found (try checkpoints/policy_finalSAC.pt or set PAPER_CHECKPOINT)."
        )
    return p


def _train_metrics_usable(path: str, *, min_data_rows: int = 1) -> bool:
    try:
        with open(path, newline="") as f:
            n = sum(1 for _ in f)
        return n >= min_data_rows + 1
    except OSError:
        return False


def resolve_train_metrics_csv() -> str | None:
    env_m = os.environ.get("PAPER_TRAIN_METRICS")
    if env_m:
        p = env_m if os.path.isabs(env_m) else os.path.join(_ROOT, env_m)
        if os.path.isfile(p) and _train_metrics_usable(p):
            return p
        print(f"Warning: PAPER_TRAIN_METRICS missing or empty: {env_m}", file=sys.stderr)
    for rel in (
        "paper_runs/regulation_1m/train_metrics_SAC.csv",
        "paper_runs/train_metrics_SAC.csv",
        "logs_sac_regulation_1m/train_metrics_SAC.csv",
        "1MilStats/1Mil_metrics.csv",
        "logs/train_metrics_SAC.csv",
    ):
        p = os.path.join(_ROOT, rel)
        if os.path.isfile(p) and _train_metrics_usable(p):
            return p
    return None


def effective_dof_95(explained_ratio: np.ndarray) -> int:
    c = np.cumsum(explained_ratio)
    for i, val in enumerate(c):
        if val >= 0.95:
            return i + 1
    return len(explained_ratio)


def score_boundaries_on_ray() -> tuple[float, float, float]:
    """Scores at radial tier boundaries on +Δy (matches env ring radii)."""
    return (
        dartboard_score_plane(R_INNER_BULL_M, 0.0),
        dartboard_score_plane(R_OUTER_BULL_M, 0.0),
        dartboard_score_plane(R_DOUBLES_WIRE_M, 0.0),
    )


def tier_fractions_from_rads(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=np.float64)
    n = max(1, r.size)
    inner = np.sum(r < R_INNER_BULL_M) / n
    bull = np.sum((r >= R_INNER_BULL_M) & (r < R_OUTER_BULL_M)) / n
    outer = np.sum((r >= R_OUTER_BULL_M) & (r < R_DOUBLES_WIRE_M)) / n
    outside = np.sum(r >= R_DOUBLES_WIRE_M) / n
    return {
        "tier_inner_bull": float(inner),
        "tier_outer_bull": float(bull),
        "tier_inside_doubles_wire": float(outer),
        "tier_outside_doubles": float(outside),
    }


def print_csv_preview(path: str, max_rows: int = 3) -> None:
    print(f"\n--- CSV: {path} ---")
    if not os.path.isfile(path):
        print("  (missing)")
        return
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= max_rows:
                break
    if not rows:
        print("  (empty)")
        return
    print("  columns:", rows[0])
    for row in rows[1:]:
        print("  row:", row)


def fig1_sac_training_curves(out_dir: str, sac_metrics_path: str) -> None:
    """Periodic SAC eval metrics vs environment steps (from train_metrics_SAC.csv)."""
    steps = []
    mr, rr, ld = [], [], []
    with open(sac_metrics_path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            steps.append(float(row["step"]))
            mr.append(float(row["mean_reward"]))
            rr.append(float(row["release_rate"]))
            ld.append(float(row["mean_landing_dist"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax0, ax1, ax2 = axes
    ax0.plot(steps, mr, color="C0", lw=2, label="SAC (eval rollouts)")
    ax0.set_xlabel("Environment step")
    ax0.set_ylabel("Mean eval reward")
    ax0.set_title("Mean reward (periodic eval)")
    ax0.grid(True, alpha=0.3)

    ax1.plot(steps, rr, color="C0", lw=2, label="SAC")
    ax1.set_xlabel("Environment step")
    ax1.set_ylabel("Release rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Release rate")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, ld, color="C0", lw=2, label="SAC")
    ax2.set_xlabel("Environment step")
    ax2.set_ylabel("Mean landing distance (m)")
    ax2.set_title("Mean radial miss")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SAC training (periodic evaluation)", y=1.02, fontsize=11)
    fig.tight_layout()
    outp = os.path.join(out_dir, "fig1_training_curves.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")


def rollout_final_policy(
    ckpt: str,
    seed: int,
    n_episodes: int,
    device: str,
) -> tuple[dict, DartEnv, object]:
    actor, cfg = load_actor(ckpt, device)
    env = make_eval_env(cfg, seed)
    runs, meta = collect_trajectories(env, cfg, device, n_episodes, actor=actor, rng=None)
    return meta, env, actor


def fig2_score_hist(out_dir: str, scores: np.ndarray, r_landing: np.ndarray, *, policy_label: str, seed: int) -> None:
    """Vertical lines at dartboard score values corresponding to env tier radii."""
    b_in, b_ob, b_dw = score_boundaries_on_ray()
    tiers = tier_fractions_from_rads(r_landing)
    xs_line = sorted([b_dw, b_ob, b_in])

    n_low = int(np.sum(scores < 42.0))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Zoom x-range so the main modes (~78, ~110) are readable; rare floor outliers omitted from axis
    xmin, xmax = 42.0, 118.0
    bins_zoom = np.linspace(xmin, xmax, 36)
    ax.hist(scores, bins=bins_zoom, density=False, color="steelblue", edgecolor="white", linewidth=0.45)
    ymax = ax.get_ylim()[1] * 1.08

    for x in xs_line:
        if xmin <= x <= xmax:
            ax.axvline(x, color="dimgray", ls="--", lw=1.4, alpha=0.95)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Terminal dartboard score")
    ax.set_ylabel("Episode count")
    stats_txt = (
        f"mean={np.mean(scores):.2f}, median={np.median(scores):.2f}\n"
        f"Inner bull: {tiers['tier_inner_bull']*100:.1f}% | "
        f"Outer bull: {tiers['tier_outer_bull']*100:.1f}% | "
        f"Inside doubles wire: {tiers['tier_inside_doubles_wire']*100:.1f}% | "
        f"Outside: {tiers['tier_outside_doubles']*100:.1f}%"
    )
    ax.set_title(
        f"Terminal dartboard score ({policy_label}, eval seed={seed}, N=400)\n{stats_txt}",
        fontsize=10,
    )
    if n_low > 0:
        fig.text(
            0.5,
            0.02,
            f"Note: x-axis restricted to [{xmin:.0f}, {xmax:.0f}] for readability; "
            f"{n_low} episode(s) with score < {xmin:.0f} not shown.",
            ha="center",
            fontsize=9,
            style="italic",
        )

    # Region legend (no overlapping text on bars)
    band_handles = [
        Line2D(
            [0],
            [0],
            color="dimgray",
            linestyle="--",
            linewidth=1.4,
            label=f"Boundary ≈{b_dw:.1f} (r={R_DOUBLES_WIRE_M:.3f} m)",
        ),
        Line2D(
            [0],
            [0],
            color="dimgray",
            linestyle="--",
            linewidth=1.4,
            label=f"Boundary ≈{b_ob:.1f} (r={R_OUTER_BULL_M:.3f} m)",
        ),
        Line2D(
            [0],
            [0],
            color="dimgray",
            linestyle="--",
            linewidth=1.4,
            label=f"Boundary ≈{b_in:.1f} (r={R_INNER_BULL_M:.4f} m)",
        ),
    ]
    ax.legend(handles=band_handles, loc="upper left", framealpha=0.92, title="Score tiers (see Methods)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    outp = os.path.join(out_dir, "fig2_score_histogram.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")


def fig3_release_steps(
    out_dir: str,
    release_steps: np.ndarray,
    _released_unused: np.ndarray,
    *,
    policy_label: str,
    seed: int,
) -> None:
    timeout = release_steps >= 40  # max_steps
    released_ok = ~timeout

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.arange(0, 42) - 0.5
    ax.hist(
        release_steps[released_ok],
        bins=bins,
        color="C0",
        alpha=0.88,
        label=f"Released (n={int(np.sum(released_ok))})",
        edgecolor="white",
        linewidth=0.45,
    )
    ax.hist(
        release_steps[timeout],
        bins=bins,
        color="C3",
        alpha=0.82,
        label=f"Timeout (n={int(np.sum(timeout))})",
        edgecolor="white",
        linewidth=0.45,
    )
    ax.axvline(40, color="k", ls="--", lw=1.3, label="Horizon (40)")
    ax.axvline(20, color="darkorange", ls="--", lw=1.3, label=r"$\mathbb{E}$[optimal release step]=20")
    # Zoom to where mass lives; timeouts listed in title
    ax.set_xlim(8.5, 18.5)
    ax.set_xlabel("Step index at release")
    ax.set_ylabel("Count")
    frac_rel = float(np.mean(released_ok))
    frac_to = float(np.mean(timeout))
    ax.set_title(
        f"Release step ({policy_label}, seed={seed}, N=400)\n"
        f"released {frac_rel*100:.1f}%, timeout {frac_to*100:.1f}% "
        f"({int(np.sum(timeout))} episodes at step 40)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    outp = os.path.join(out_dir, "fig3_release_step_histogram.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")


def fig4_pca_dual(
    out_dir: str,
    explained0: np.ndarray,
    dof0: int,
    explained1: np.ndarray,
    dof1: int,
    *,
    policy_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, explained, dof, title in [
        (axes[0], explained0, dof0, "Eval environment seed 0"),
        (axes[1], explained1, dof1, "Eval environment seed 1"),
    ]:
        n = len(explained)
        xs = np.arange(1, n + 1)
        cum = np.cumsum(explained)
        ax.bar(xs, explained, color="C0", alpha=0.78, width=0.65, label="Per-component")
        ax2 = ax.twinx()
        ax2.plot(xs, cum, color="C1", marker="o", ms=4, lw=1.8, label="Cumulative")
        ax2.axhline(0.95, color="red", ls="--", lw=1.2, alpha=0.9)
        cross = np.where(cum >= 0.95)[0]
        if cross.size:
            k = int(cross[0])
            xpos = float(xs[k])
            ax2.axvline(xpos, color="green", ls="--", lw=1.2)
            ax2.text(
                xpos + 0.25,
                0.52,
                f"Effective DOF (95%) = {dof}",
                fontsize=10,
                color="green",
            )
        ax.set_xlabel("Principal component index")
        ax.set_ylabel("Explained variance ratio", color="C0")
        ax2.set_ylabel("Cumulative explained variance", color="C1")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax.set_xticks(xs)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"PCA on joint angles and velocities (12-D), {policy_label}",
        y=1.03,
        fontsize=13,
    )
    fig.tight_layout()
    outp = os.path.join(out_dir, "fig4_pca_cumulative_variance.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")


def fig5_landing_density_contour(
    out_dir: str,
    lx: np.ndarray,
    ly: np.ndarray,
    tx: float,
    ty: float,
    *,
    view_half_m: float,
    bins: int,
    gauss_sigma: float,
) -> None:
    """Smoothed 2D histogram + filled contours; lighter regions = higher landing density."""
    ring_spec = [
        (R_INNER_BULL_M, "Inner bull (50)", "6.35"),
        (R_OUTER_BULL_M, "Outer bull (25)", "16"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    lx = np.asarray(lx, dtype=np.float64)
    ly = np.asarray(ly, dtype=np.float64)
    lim = view_half_m
    rng = [[tx - lim, tx + lim], [ty - lim, ty + lim]]

    if lx.size >= 1:
        H, xe, ye = np.histogram2d(lx, ly, bins=bins, range=rng)
        Z = gaussian_filter(H.astype(np.float64), sigma=gauss_sigma)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        zmin, zmax = float(Z.min()), float(Z.max())
        if zmax <= zmin:
            zmax = zmin + 1e-12
        levels = np.linspace(zmin, zmax, 36)
        cf = ax.contourf(
            X,
            Y,
            Z,
            levels=levels,
            cmap="hot",
            antialiased=True,
            zorder=1,
        )
        cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Smoothed bin count (lighter = denser)")

    # Standard bull circles (white + black halo); radii are ~mm scale — thinner strokes on inner ring.
    for r_m, _name, _mm in ring_spec:
        lw_b, lw_w = (2.2, 1.25) if r_m <= R_INNER_BULL_M * 1.01 else (3.0, 1.7)
        ax.add_patch(
            Circle(
                (tx, ty),
                r_m,
                fill=False,
                linestyle="--",
                edgecolor="black",
                linewidth=lw_b,
                zorder=25,
            )
        )
        ax.add_patch(
            Circle(
                (tx, ty),
                r_m,
                fill=False,
                linestyle="--",
                edgecolor="white",
                linewidth=lw_w,
                zorder=26,
            )
        )

    ax.set_xlim(tx - view_half_m, tx + view_half_m)
    ax.set_ylim(ty - view_half_m, ty + view_half_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta y$ (m), left")
    ax.set_ylabel(r"$\Delta z$ (m), relative to bull")
    ax.set_title("Landing density", fontsize=12)
    ax.grid(True, alpha=0.22)

    ring_handles = []
    for r_m, name, mm_s in ring_spec:
        lw_w = 1.25 if r_m <= R_INNER_BULL_M * 1.01 else 1.8
        lw_pe = 2.8 if r_m <= R_INNER_BULL_M * 1.01 else 3.4
        ring_handles.append(
            Line2D(
                [0],
                [0],
                color="white",
                linestyle="--",
                marker="o",
                markersize=0,
                linewidth=lw_w,
                path_effects=[pe.Stroke(linewidth=lw_pe, foreground="black"), pe.Normal()],
                label=f"{name} (r = {mm_s} mm)",
            )
        )
    ax.legend(
        handles=ring_handles,
        loc="lower left",
        fontsize=9,
        framealpha=0.9,
        title="Standard bull (steel-tip)",
    )

    fig.tight_layout()
    outp = os.path.join(out_dir, "fig5_landing_density_contour.png")
    fig.savefig(outp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {outp}")


def main() -> None:
    out_dir = os.path.join(_ROOT, "paper_figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Paper-relevant paths (see paper_runs/README.txt) ===")
    for label, base in [
        ("paper_runs", os.path.join(_ROOT, "paper_runs")),
        ("checkpoints", os.path.join(_ROOT, "checkpoints")),
        ("eval_outputs", os.path.join(_ROOT, "eval_outputs")),
    ]:
        print(f"\n[{label}]")
        if not os.path.isdir(base):
            print("  (missing)")
            continue
        for root, _, files in os.walk(base):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, _ROOT)
                print(f"  {rel}  ({sz} bytes)")

    sac_metrics = resolve_train_metrics_csv()
    if sac_metrics is None:
        print(
            "Skipping Fig 1: no usable train_metrics_SAC.csv (train with "
            "--log-dir paper_runs/regulation_1m or set PAPER_TRAIN_METRICS).",
            file=sys.stderr,
        )
    else:
        print_csv_preview(sac_metrics)
        fig1_sac_training_curves(out_dir, sac_metrics)

    ckpt = resolve_paper_checkpoint()
    policy_short = os.path.basename(ckpt).replace(".pt", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Figs 2,3,5: match eval_outputs/1mil_best_seed1 (best checkpoint, eval seed 1)
    meta_main, env_main, _ = rollout_final_policy(ckpt, seed=EVAL_SEED_MAIN, n_episodes=400, device=device)

    scores_m = meta_main["_meta_scores"].astype(np.float64)
    lx_full = meta_main["_meta_landing_x"].astype(np.float64)
    ly_full = meta_main["_meta_landing_y"].astype(np.float64)
    rs_m = meta_main["_meta_release_steps"].astype(np.int32)
    rads_m = np.sqrt((lx_full - env_main.target_x) ** 2 + (ly_full - env_main.target_y) ** 2)
    released_m = rs_m < env_main.max_steps

    fig2_score_hist(out_dir, scores_m, rads_m, policy_label=policy_short, seed=EVAL_SEED_MAIN)
    fig3_release_steps(out_dir, rs_m, released_m.astype(float), policy_label=policy_short, seed=EVAL_SEED_MAIN)

    actor0, cfg0 = load_actor(ckpt, device)
    runs0, meta0b = collect_trajectories(make_eval_env(cfg0, 0), cfg0, device, 400, actor=actor0, rng=None)
    X0 = stack_states_from_runs(runs0)
    pca0 = PCA(n_components=min(12, max(1, X0.shape[0] - 1)))
    pca0.fit(X0)
    ex0 = pca0.explained_variance_ratio_
    dof0 = effective_dof_95(ex0)

    runs1, _ = collect_trajectories(make_eval_env(cfg0, 1), cfg0, device, 400, actor=actor0, rng=None)
    X1 = stack_states_from_runs(runs1)
    pca1 = PCA(n_components=min(12, max(1, X1.shape[0] - 1)))
    pca1.fit(X1)
    ex1 = pca1.explained_variance_ratio_
    dof1 = effective_dof_95(ex1)

    fig4_pca_dual(out_dir, ex0, dof0, ex1, dof1, policy_label=policy_short)

    keep = rads_m <= LANDING_DATA_RMAX_M
    n_out = int(np.sum(~keep))
    lx_plot = lx_full[keep]
    ly_plot = ly_full[keep]
    fig5_landing_density_contour(
        out_dir,
        lx_plot,
        ly_plot,
        float(env_main.target_x),
        float(env_main.target_y),
        view_half_m=LANDING_VIEW_HALF_M,
        bins=LANDING_CONTOUR_BINS,
        gauss_sigma=LANDING_CONTOUR_GAUSS_SIGMA,
    )

    # Save stats for paper
    stats_path = os.path.join(out_dir, "figure_stats.txt")
    b1, b2, b3 = score_boundaries_on_ray()
    tiers = tier_fractions_from_rads(rads_m)
    n_plot = int(np.sum(keep))
    mean_miss = float(meta_main["_meta_mean_dist"])
    std_miss = float(meta_main["_meta_std_dist"])
    release_rate = float(meta_main["_meta_release_rate"])
    with open(stats_path, "w") as f:
        f.write(f"Figure statistics ({os.path.basename(ckpt)}, eval_seed={EVAL_SEED_MAIN}, 400 episodes)\n\n")
        f.write(
            f"Score boundary scores (on Δy axis): "
            f"r={R_INNER_BULL_M:.5f} m -> {b1:.4f}, "
            f"r={R_OUTER_BULL_M:.5f} m -> {b2:.4f}, "
            f"r={R_DOUBLES_WIRE_M:.3f} m -> {b3:.4f}\n"
        )
        f.write(f"Tier fractions: {tiers}\n")
        f.write(
            f"Fig 3: release fraction {float(np.mean(rs_m < env_main.max_steps)):.4f}, "
            f"timeout {float(np.mean(rs_m >= env_main.max_steps)):.4f}\n"
        )
        f.write(
            "Fig 5 (paper_figures/fig5_landing_density_contour.png):\n"
            f"  Policy label used in other figures: {policy_short}\n"
            f"  Density map: 2D histogram ({LANDING_CONTOUR_BINS}x{LANDING_CONTOUR_BINS} bins over "
            f"±{LANDING_VIEW_HALF_M:.2f} m), Gaussian smoothing sigma={LANDING_CONTOUR_GAUSS_SIGMA} "
            "(lighter areas = more landings in that region).\n"
            f"  Landings included in the density plot: radial miss r <= {LANDING_DATA_RMAX_M} m "
            f"(n={n_plot}/400); excluded as far outliers: {n_out}.\n"
            f"  All 400 episodes (full sample): mean radial miss={mean_miss:.4f} m, "
            f"std={std_miss:.4f} m, release rate={release_rate:.4f}.\n"
            "  Ring overlays: same inner/outer bull radii as env ring bonuses (see R_INNER_BULL_M, R_OUTER_BULL_M).\n"
            f"    inner bull r={R_INNER_BULL_M * 1000:.2f} mm (50 pts); outer bull r={R_OUTER_BULL_M * 1000:.1f} mm (25 pts).\n"
            f"  Third reward tier in env: r < {R_DOUBLES_WIRE_M:.3f} m (inside doubles wire), not drawn on Fig 5.\n"
        )
        f.write(f"PCA eff DOF seed0={dof0}, seed1={dof1}\n")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
