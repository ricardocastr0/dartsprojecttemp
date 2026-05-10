#!/usr/bin/env python3
"""
Train three SAC policies with σ_t ∈ {3.0, 5.0, 15.0} (timing bonus width in env),
then run 200-episode deterministic evaluation each.

Does not modify env.py or trainSAC.py — uses a runtime monkey-patch on DartEnv.__init__
to set self.timing_std after the original constructor runs.

Outputs:
  experiments/results/sigma_t_sweep.csv
  experiments/figures/sigma_t_{σ}_release_hist.png
"""
from __future__ import annotations

import csv
import os
import sys

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EXP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import env as _env_mod  # noqa: E402

_ORIG_DARTENV_INIT = _env_mod.DartEnv.__init__
_TIMING_OVERRIDE: dict[str, float | None] = {"sigma_t": None}


def _patched_dartenv_init(self, *args, **kwargs) -> None:
    _ORIG_DARTENV_INIT(self, *args, **kwargs)
    if _TIMING_OVERRIDE["sigma_t"] is not None:
        self.timing_std = float(_TIMING_OVERRIDE["sigma_t"])


_env_mod.DartEnv.__init__ = _patched_dartenv_init

from dataclasses import replace  # noqa: E402

import matplotlib

matplotlib.use("Agg")

from config import CFG  # noqa: E402
from evaluate import collect_trajectories, load_actor, make_eval_env  # noqa: E402
from train import truncate_csv_log  # noqa: E402
from trainSAC import train_sac  # noqa: E402
from utils import plot_histogram, set_seed  # noqa: E402

EVAL_EPISODES = 200
EVAL_SEED = 42
SIGMA_VALUES = (3.0, 5.0, 15.0)
TOTAL_ENV_STEPS = 300_000


def _ensure_dirs() -> tuple[str, str]:
    res = os.path.join(_ROOT, "experiments", "results")
    fig = os.path.join(_ROOT, "experiments", "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    return res, fig


def run_final_eval(ckpt_path: str, device: str) -> dict[str, float]:
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    actor, cfg_loaded = load_actor(ckpt_path, device)
    env = make_eval_env(cfg_loaded, EVAL_SEED)
    _, meta = collect_trajectories(env, cfg_loaded, device, EVAL_EPISODES, actor=actor, rng=None)
    rs = meta["_meta_release_steps"].astype(np.float64)
    mean_rs = float(np.mean(rs)) if rs.size else float("nan")
    return {
        "mean_miss_m": float(meta["_meta_mean_dist"]),
        "std_miss_m": float(meta["_meta_std_dist"]),
        "release_rate": float(meta["_meta_release_rate"]),
        "mean_release_step": mean_rs,
        "mean_score": float(np.mean(meta["_meta_scores"])) if meta["_meta_scores"].size else float("nan"),
        "release_steps": rs,
    }


def main() -> None:
    set_seed(EVAL_SEED)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    results_dir, figures_dir = _ensure_dirs()

    rows: list[dict[str, float | str]] = []

    for sigma in SIGMA_VALUES:
        print(f"\n=== σ_t = {sigma} (timing_std override) ===")
        _TIMING_OVERRIDE["sigma_t"] = sigma

        name = f"sigma_t_{sigma}"
        ckpt_rel = os.path.join("experiments", "checkpoints", name)
        log_rel = os.path.join("experiments", "logs", name)

        cfg = replace(
            CFG,
            seed=EVAL_SEED,
            total_env_steps=TOTAL_ENV_STEPS,
            save_dir=ckpt_rel,
        )

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

        ckpt_path = os.path.join(_ROOT, ckpt_rel, "policy_finalSAC.pt")
        stats = run_final_eval(ckpt_path, device)

        hist_path = os.path.join(figures_dir, f"{name}_release_hist.png")
        plot_histogram(
            stats["release_steps"],
            title=f"Release step (σ_t={sigma}, N={EVAL_EPISODES})",
            xlabel="First release step",
            save_path=hist_path,
        )
        print(f"Saved histogram: {hist_path}")

        rows.append(
            {
                "sigma_t": sigma,
                "mean_miss_m": stats["mean_miss_m"],
                "std_miss_m": stats["std_miss_m"],
                "release_rate": stats["release_rate"],
                "mean_release_step": stats["mean_release_step"],
                "mean_score": stats["mean_score"],
            }
        )

    _TIMING_OVERRIDE["sigma_t"] = None

    out_csv = os.path.join(results_dir, "sigma_t_sweep.csv")
    fieldnames = (
        "sigma_t",
        "mean_miss_m",
        "std_miss_m",
        "release_rate",
        "mean_release_step",
        "mean_score",
    )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
