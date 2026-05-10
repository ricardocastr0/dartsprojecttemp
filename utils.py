from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state - mean) / (std + 1e-8)


@dataclass
class TrajectoryLogger:
    """Stores a single episode trajectory for analysis/plots."""

    angles: list[np.ndarray] = field(default_factory=list)
    velocities: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.angles.append(state[:6].copy())
        self.velocities.append(state[6:12].copy())
        self.actions.append(action.copy())
        self.rewards.append(float(reward))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "angles": np.asarray(self.angles, dtype=np.float32),
            "velocities": np.asarray(self.velocities, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
        }


def plot_pca_variance(explained: np.ndarray, save_path: str) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    xs = np.arange(1, len(explained) + 1)
    cum = np.cumsum(explained)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, cum, marker="o")
    plt.axhline(0.95, linestyle="--", color="red", linewidth=1, label="95% threshold")
    plt.xticks(xs)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_landing_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    target_x: float,
    target_y: float,
    ring_radii: list[float],
    save_path: str,
    *,
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    title: str = "Landing positions",
) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, alpha=0.5, s=12, c="C0", label="Landings")
    ax.scatter([target_x], [target_y], c="red", s=80, marker="+", linewidths=2, label="Bull")
    for r in ring_radii:
        circ = Circle((target_x, target_y), r, fill=False, linestyle="--", edgecolor="gray", linewidth=1)
        ax.add_patch(circ)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_pca_explained_bars(explained: np.ndarray, save_path: str) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    n = len(explained)
    xs = np.arange(1, n + 1)
    cum = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(xs, explained, color="C0", alpha=0.85, label="Explained variance ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.plot(xs, cum, color="C1", marker="o", linewidth=2, label="Cumulative")
    ax2.axhline(0.95, linestyle="--", color="red", linewidth=1, alpha=0.8)
    ax2.set_ylabel("Cumulative explained variance", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0.0, 1.02)

    ax1.set_title("PCA explained variance by component")
    fig.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_landing_hexbin(
    xs: np.ndarray,
    ys: np.ndarray,
    target_x: float,
    target_y: float,
    ring_radii: list[float],
    save_path: str,
    *,
    gridsize: int = 35,
    mincnt: int = 1,
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    title: str = "Landing density (hexbin)",
) -> None:
    """2D landing density on board plane with ring overlay (same geometry as plot_landing_scatter)."""
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 7))
    if xs.size >= 2:
        hb = ax.hexbin(xs, ys, gridsize=gridsize, mincnt=mincnt, cmap="viridis", bins="log")
        plt.colorbar(hb, ax=ax, label="log10(count+1)")
    ax.scatter([target_x], [target_y], c="red", s=80, marker="+", linewidths=2, label="Bull", zorder=5)
    for r in ring_radii:
        circ = Circle((target_x, target_y), r, fill=False, linestyle="--", edgecolor="gray", linewidth=1)
        ax.add_patch(circ)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    save_path: str,
    bins: int | None = None,
) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(v, bins=bins if bins is not None else min(30, max(10, int(np.sqrt(v.size)))), color="C0", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_torque_variance(torque_var: np.ndarray, save_path: str) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    plt.figure(figsize=(7, 4))
    joints = np.arange(1, len(torque_var) + 1)
    plt.bar(joints, torque_var)
    plt.xticks(joints)
    plt.xlabel("Joint index")
    plt.ylabel("Torque variance")
    plt.title("Per-joint torque usage (variance)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def pretty_dict(d: dict[str, Any]) -> str:
    keys = sorted(d.keys())
    parts = []
    for k in keys:
        v = d[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

