"""
Matplotlib visualization of dart trajectory (SPEC frame) and vertical board plane.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from flight_physics import (
    BULLSEYE_CENTER_Z_M,
    OCHE_TO_BOARD_X_M,
    integrate_until_board,
)


def save_throw_figure(
    release_state6: np.ndarray,
    save_path: str,
    wind_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    n_samples: int = 400,
) -> dict | None:
    """
    Integrate from release_state6 and save x–z plot with board line and bull height.
    Returns integrate_until_board result dict, or None on failure.
    """
    out = integrate_until_board(
        release_state6,
        wind_xyz_mps=wind_xyz,
        drag_enabled=drag_enabled,
    )
    sol = out.get("sol")
    if sol is None or sol.sol is None:
        return out

    t_hit = out.get("t_hit")
    t1 = float(t_hit) * 1.05 if t_hit is not None else 0.5
    t1 = max(t1, 0.3)
    ts = np.linspace(0.0, t1, n_samples)
    xs: list[float] = []
    zs: list[float] = []
    for t in ts:
        y = sol.sol(t)
        xs.append(float(y[0]))
        zs.append(float(y[2]))

    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, zs, "b-", lw=1.5, label="Trajectory")
    ax.axvline(OCHE_TO_BOARD_X_M, color="saddlebrown", ls="--", lw=1.5, label=f"Board x={OCHE_TO_BOARD_X_M}")
    ax.scatter(
        [OCHE_TO_BOARD_X_M],
        [BULLSEYE_CENTER_Z_M],
        c="red",
        s=60,
        zorder=5,
        label="Bull center",
    )
    x0 = float(release_state6[0])
    z0 = float(release_state6[2])
    ax.scatter([x0], [z0], c="green", s=40, zorder=5, label="Release")
    ax.set_xlabel("x (m) toward board")
    ax.set_ylabel("z (m) up")
    ax.set_title("Dart flight (x–z)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return out
