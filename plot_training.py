"""
Plot training logs: eval metrics from train_metrics*.csv and per-episode rows from train_episodes*.csv.

Prefers SAC filenames (`train_metrics_SAC.csv`, `train_episodes_SAC.csv`) when present.

Usage:
  python plot_training.py
  python plot_training.py --log-dir logs --out logs/training_curves.png --smooth 50
  python plot_training.py --curriculum-taper-episodes 2000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from config import CFG

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_log_paths(log_dir: str) -> tuple[str, str]:
    """Prefer SAC CSV names when both exist."""
    sac_m = os.path.join(log_dir, "train_metrics_SAC.csv")
    sac_e = os.path.join(log_dir, "train_episodes_SAC.csv")
    plain_m = os.path.join(log_dir, "train_metrics.csv")
    plain_e = os.path.join(log_dir, "train_episodes.csv")
    metrics_path = sac_m if os.path.isfile(sac_m) else plain_m
    episodes_path = sac_e if os.path.isfile(sac_e) else plain_e
    return metrics_path, episodes_path


def _read_train_metrics(path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return {}
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in out:
            try:
                v = float(row[k]) if row.get(k) not in (None, "") else float("nan")
            except (TypeError, ValueError):
                v = float("nan")
            out[k].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _read_episodes(path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return {}
    env_step: list[int] = []
    ep_idx: list[int] = []
    ep_ret: list[float] = []
    ep_len: list[float] = []
    released: list[float] = []
    for row in rows:
        try:
            env_step.append(int(row["env_step"]))
            ep_ret.append(float(row["episode_return"]))
            released.append(float(row.get("released", 0)))
            if "episode_index" in row and row["episode_index"] not in (None, ""):
                ep_idx.append(int(row["episode_index"]))
            else:
                ep_idx.append(len(ep_idx) + 1)
            if "episode_length" in row and row["episode_length"] not in (None, ""):
                ep_len.append(float(row["episode_length"]))
            else:
                ep_len.append(float("nan"))
        except (KeyError, ValueError):
            continue
    return {
        "env_step": np.asarray(env_step, dtype=np.int64),
        "episode_index": np.asarray(ep_idx, dtype=np.int64),
        "episode_return": np.asarray(ep_ret, dtype=np.float64),
        "episode_length": np.asarray(ep_len, dtype=np.float64),
        "released": np.asarray(released, dtype=np.float64),
    }


def rolling_mean(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_indices_aligned, smoothed) for simple trailing mean of length window."""
    if x.size == 0 or window <= 1:
        return np.arange(x.size, dtype=np.int64), x.astype(np.float64)
    w = min(window, x.size)
    if w < 2:
        return np.arange(x.size, dtype=np.int64), x.astype(np.float64)
    cum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    sm = (cum[w:] - cum[:-w]) / float(w)
    idx = np.arange(w - 1, w - 1 + sm.size, dtype=np.int64)
    return idx, sm


def main() -> None:
    p = argparse.ArgumentParser(description="Plot train_metrics / train_episodes CSV logs")
    p.add_argument("--log-dir", type=str, default="logs", help="Directory with CSV logs")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <log-dir>/training_curves.png)",
    )
    p.add_argument("--smooth", type=int, default=50, help="Moving average window for episode returns")
    p.add_argument(
        "--curriculum-taper-episodes",
        type=int,
        default=None,
        help="Episode-index boundary for curriculum vs post (vertical line on return plot); "
        "default: config.CFG.curriculum_taper_episodes",
    )
    args = p.parse_args()

    base = _SCRIPT_DIR
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(base, args.log_dir)
    metrics_path, episodes_path = _resolve_log_paths(log_dir)
    out_path = args.out or os.path.join(log_dir, "training_curves.png")
    if args.out and not os.path.isabs(args.out):
        out_path = os.path.join(base, args.out)

    metrics = _read_train_metrics(metrics_path)
    episodes = _read_episodes(episodes_path)

    if not metrics and not episodes:
        print(f"No data: missing or empty\n  {metrics_path}\n  {episodes_path}", file=sys.stderr)
        sys.exit(1)

    taper_k = args.curriculum_taper_episodes
    if taper_k is None:
        taper_k = int(CFG.curriculum_taper_episodes)

    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]
    ax20, ax21 = axes[2, 0], axes[2, 1]

    if metrics and "step" in metrics:
        st = metrics["step"]
        if "mean_reward" in metrics:
            ax00.plot(st, metrics["mean_reward"], "o-", ms=4, label="mean_reward")
            ax00.set_xlabel("env step (eval)")
            ax00.set_ylabel("mean_reward")
            ax00.set_title("Eval: mean return")
            ax00.grid(True, alpha=0.3)
            ax00.legend()
        if "release_rate" in metrics:
            ax01.plot(st, metrics["release_rate"], "s-", ms=4, color="C1", label="release_rate")
            ax01.set_xlabel("env step (eval)")
            ax01.set_ylabel("release_rate")
            ax01.set_title("Eval: release rate")
            ax01.set_ylim(-0.05, 1.05)
            ax01.grid(True, alpha=0.3)
            ax01.legend()
        if "mean_landing_dist" in metrics:
            ax10.plot(st, metrics["mean_landing_dist"], "^-", ms=4, color="C2", label="mean_landing_dist")
            ax10.set_xlabel("env step (eval)")
            ax10.set_ylabel("radial miss (m)")
            ax10.set_title("Eval: mean landing distance")
            ax10.grid(True, alpha=0.3)
            ax10.legend()
    else:
        ax00.text(0.5, 0.5, "No train_metrics*.csv", ha="center", va="center")
        ax01.text(0.5, 0.5, "No train_metrics*.csv", ha="center", va="center")
        ax10.text(0.5, 0.5, "No train_metrics*.csv", ha="center", va="center")

    if episodes and episodes["env_step"].size > 0:
        es = episodes["env_step"]
        er = episodes["episode_return"]
        ax11.scatter(es, er, s=6, alpha=0.35, c="C0", label="episode_return")
        idx, sm = rolling_mean(er, args.smooth)
        if sm.size > 0:
            ax11.plot(es[idx], sm, "-", lw=2, color="C3", label=f"rolling mean (w={args.smooth})")
        if taper_k > 0 and "episode_index" in episodes:
            ei = episodes["episode_index"]
            mask = ei > int(taper_k)
            if np.any(mask):
                vline_x = float(es[np.argmax(mask)])
                ax11.axvline(
                    vline_x,
                    color="gray",
                    ls="--",
                    alpha=0.7,
                    label=f"curriculum taper end (~ ep idx {taper_k})",
                )
        ax11.set_xlabel("env step")
        ax11.set_ylabel("episode return")
        ax11.set_title("Training episodes (SAC rollouts)")
        ax11.grid(True, alpha=0.3)
        ax11.legend(fontsize=8)

        epi = episodes["episode_index"]
        el = episodes["episode_length"]
        ok = np.isfinite(el)
        if np.any(ok):
            ax20.scatter(epi[ok], el[ok], s=6, alpha=0.4, c="C4")
            ax20.set_xlabel("episode index")
            ax20.set_ylabel("episode length (steps)")
            ax20.set_title("Episode length vs index")
            ax20.grid(True, alpha=0.3)
        else:
            ax20.text(0.5, 0.5, "No episode_length column", ha="center", va="center")

        ax21.scatter(es, episodes["released"], s=5, alpha=0.25, c="C5", label="released")
        ax21.set_xlabel("env step")
        ax21.set_ylabel("released (0/1)")
        ax21.set_title("Release outcomes vs time")
        ax21.set_ylim(-0.1, 1.1)
        ax21.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, "No train_episodes*.csv", ha="center", va="center")
        ax20.text(0.5, 0.5, "No train_episodes*.csv", ha="center", va="center")
        ax21.text(0.5, 0.5, "No train_episodes*.csv", ha="center", va="center")

    fig.suptitle("Training overview", fontsize=12)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"(metrics: {metrics_path}, episodes: {episodes_path})")


if __name__ == "__main__":
    main()
