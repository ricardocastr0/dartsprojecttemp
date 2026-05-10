#!/usr/bin/env python3
"""
Four SAC ablations × 300k env steps each (same seed / hyperparams except the ablated piece).

Conditions:
  full       — baseline (prioritized replay + stratified batching + curriculum)
  no_strat   — min_release_fraction = 0 (no 50/50 stratified batching)
  no_curric  — curriculum_taper_episodes = 0
  no_per     — prioritized_replay = False (uniform replay)

Mid-training: eval every 50k steps with 50 episodes (trainSAC built-in).
Final: 200 deterministic eval episodes per condition.

Outputs:
  experiments/results/ablations.csv
  experiments/figures/ablation_release_rate.png
  experiments/figures/ablation_mean_miss.png
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
from dataclasses import replace

from config import CFG, Config  # noqa: E402
from evaluate import collect_trajectories, load_actor, make_eval_env  # noqa: E402
from train import truncate_csv_log  # noqa: E402
from trainSAC import train_sac  # noqa: E402
from utils import set_seed  # noqa: E402

EVAL_SEED = 42
TOTAL_ENV_STEPS = 300_000
EVAL_EVERY_STEPS = 50_000
MID_EVAL_EPISODES = 50
FINAL_EVAL_EPISODES = 200
# Synthetic step for post-training deterministic eval (distinct from last train checkpoint row at 300000)
FINAL_STEP_TAG = 300_001

CONDITIONS: list[tuple[str, dict[str, float | int | bool]]] = [
    ("full", {}),
    ("no_strat", {"min_release_fraction": 0.0}),
    ("no_curric", {"curriculum_taper_episodes": 0}),
    ("no_per", {"prioritized_replay": False}),
]


def _ensure_dirs() -> tuple[str, str]:
    res = os.path.join(_ROOT, "experiments", "results")
    fig = os.path.join(_ROOT, "experiments", "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    return res, fig


def build_cfg(condition_name: str, overrides: dict) -> Config:
    base = replace(
        CFG,
        seed=EVAL_SEED,
        total_env_steps=TOTAL_ENV_STEPS,
        eval_every_steps=EVAL_EVERY_STEPS,
        eval_episodes=MID_EVAL_EPISODES,
        save_dir=os.path.join("experiments", "checkpoints", f"ablation_{condition_name}"),
    )
    return replace(base, **overrides)


def read_train_metrics_rows(csv_path: str) -> list[dict[str, str]]:
    if not os.path.isfile(csv_path):
        print(f"ERROR: train metrics CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def run_final_eval(ckpt_path: str, device: str) -> dict[str, float]:
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    actor, cfg = load_actor(ckpt_path, device)
    env = make_eval_env(cfg, EVAL_SEED)
    _, meta = collect_trajectories(env, cfg, device, FINAL_EVAL_EPISODES, actor=actor, rng=None)
    return {
        "release_rate": float(meta["_meta_release_rate"]),
        "mean_miss_m": float(meta["_meta_mean_dist"]),
        "mean_score": float(np.mean(meta["_meta_scores"]))
        if meta["_meta_scores"].size
        else float("nan"),
    }


def main() -> None:
    set_seed(EVAL_SEED)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir, figures_dir = _ensure_dirs()

    all_csv_rows: list[dict[str, float | str | int]] = []

    for cond_name, overrides in CONDITIONS:
        print(f"\n=== Ablation: {cond_name} ===")
        cfg = build_cfg(cond_name, overrides)

        log_rel = os.path.join("experiments", "logs", f"ablation_{cond_name}")
        metrics_csv = os.path.join(_ROOT, log_rel, "train_metrics_SAC.csv")
        episode_csv = os.path.join(_ROOT, log_rel, "train_episodes_SAC.csv")
        truncate_csv_log(metrics_csv)
        truncate_csv_log(episode_csv)

        train_sac(
            cfg,
            train_metrics_csv=metrics_csv,
            episode_log_csv=episode_csv,
            progress_every_episodes=200,
        )

        ckpt_path = os.path.join(_ROOT, cfg.save_dir, "policy_finalSAC.pt")
        train_rows = read_train_metrics_rows(metrics_csv)

        for row in train_rows:
            step = int(float(row["step"]))
            if step % EVAL_EVERY_STEPS != 0 or step > TOTAL_ENV_STEPS:
                continue
            all_csv_rows.append(
                {
                    "condition": cond_name,
                    "step": step,
                    "release_rate": float(row["release_rate"]),
                    "mean_miss_m": float(row["mean_landing_dist"]),
                    "mean_score": float(row["mean_score"]),
                }
            )

        fin = run_final_eval(ckpt_path, device)
        all_csv_rows.append(
            {
                "condition": cond_name,
                "step": FINAL_STEP_TAG,
                "release_rate": fin["release_rate"],
                "mean_miss_m": fin["mean_miss_m"],
                "mean_score": fin["mean_score"],
            }
        )
        print(f"  Final deterministic eval @ step label {FINAL_STEP_TAG}: {fin}")

    out_csv = os.path.join(results_dir, "ablations.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = ("condition", "step", "release_rate", "mean_miss_m", "mean_score")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_csv_rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")

    # --- Line plots (mid-train curves + final markers) ---
    cond_order = [c[0] for c in CONDITIONS]
    colors = ["C0", "C1", "C2", "C3"]
    def plot_metric(metric_key: str, ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for ci, cond_name in enumerate(cond_order):
            xs: list[int] = []
            ys: list[float] = []
            for r in all_csv_rows:
                if r["condition"] != cond_name:
                    continue
                if r["step"] == FINAL_STEP_TAG:
                    continue
                xs.append(int(r["step"]))
                ys.append(float(r[metric_key]))
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, "-o", label=cond_name, color=colors[ci % len(colors)], lw=1.5, ms=4)

            xf = yf = None
            for r in all_csv_rows:
                if r["condition"] == cond_name and r["step"] == FINAL_STEP_TAG:
                    xf, yf = FINAL_STEP_TAG, float(r[metric_key])
                    break
            if xf is not None:
                ax.scatter([xf], [yf], color=colors[ci % len(colors)], s=80, marker="s", zorder=5)
        ax.set_xlabel("Environment step")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel} vs training step (square: {FINAL_EVAL_EPISODES}-ep deterministic @ end)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        p = os.path.join(figures_dir, fname)
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f"Saved {p}")

    plot_metric("release_rate", "Release rate", "ablation_release_rate.png")
    plot_metric("mean_miss_m", "Mean radial miss (m)", "ablation_mean_miss.png")


if __name__ == "__main__":
    main()
