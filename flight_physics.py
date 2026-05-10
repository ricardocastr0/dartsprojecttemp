"""
SPEC-style dart flight in world frame (+x toward board, +y left, +z up).

Transcribed from course SPEC (no external dartrobot import). Uses scipy ODE
integration with forward crossing of vertical board plane x = OCHE_TO_BOARD_X_M.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

# --- Board / world (SPEC quick reference) ---
OCHE_TO_BOARD_X_M = 2.37
BULLSEYE_CENTER_X_M = OCHE_TO_BOARD_X_M
BULLSEYE_CENTER_Y_M = 0.0
BULLSEYE_CENTER_Z_M = 1.73

# --- Dart / air (SPEC Track B1) ---
DART_MASS_KG = 0.022
AIR_DENSITY_KG_M3 = 1.225
DRAG_COEFFICIENT_CD = 0.47
CROSS_SECTION_AREA_M2 = 3.0e-4
GRAVITY_M_S2 = 9.81

# Plot reference only (mm from PDF) — not used for reward
R_INNER_BULL_MM = 6.35
R_OUTER_BULL_MM = 15.9
R_TRIPLE_INNER_MM = 99.0
R_TRIPLE_OUTER_MM = 107.0
R_DOUBLE_INNER_MM = 162.0
R_DOUBLE_OUTER_MM = 170.0
R_BOARD_MISS_MM = 170.0

# Radial miss when integrator reports no forward hit (shaping)
PRED_DIST_MISS_M = float(R_BOARD_MISS_MM) / 1000.0 + 0.5


def acceleration_total(
    vx: float,
    vy: float,
    vz: float,
    wind_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    mass_kg: float = DART_MASS_KG,
    rho: float = AIR_DENSITY_KG_M3,
    cd: float = DRAG_COEFFICIENT_CD,
    area_m2: float = CROSS_SECTION_AREA_M2,
    gravity_m_s2: float = GRAVITY_M_S2,
    drag_enabled: bool = True,
) -> tuple[float, float, float]:
    vrx = vx - wind_xyz_mps[0]
    vry = vy - wind_xyz_mps[1]
    vrz = vz - wind_xyz_mps[2]
    vmag = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)

    ax = ay = 0.0
    az = -gravity_m_s2

    if drag_enabled and vmag > 1e-12:
        k = -0.5 * rho * cd * area_m2 * vmag / mass_kg
        ax += k * vrx
        ay += k * vry
        az += k * vrz

    return ax, ay, az


def state_derivative(
    _t: float,
    state6: np.ndarray,
    wind_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> np.ndarray:
    x, y, z, vx, vy, vz = state6.tolist()
    ax, ay, az = acceleration_total(vx, vy, vz, wind_xyz_mps, drag_enabled=drag_enabled)
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def board_ring_radii_m_for_plot() -> list[float]:
    """Reference circles on board face (Δy, Δz plane), meters."""
    return sorted(
        {
            R_INNER_BULL_MM / 1000.0,
            R_OUTER_BULL_MM / 1000.0,
            R_TRIPLE_INNER_MM / 1000.0,
            R_TRIPLE_OUTER_MM / 1000.0,
            R_DOUBLE_INNER_MM / 1000.0,
            R_DOUBLE_OUTER_MM / 1000.0,
        }
    )


def integrate_until_board(
    release_state6: np.ndarray,
    wind_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    max_time_s: float = 3.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """
    Integrate until first forward crossing of x = OCHE_TO_BOARD_X_M.

    Returns keys: hit, t_hit, state_hit, delta_y_m, delta_z_m, sol
    """
    release_state6 = np.asarray(release_state6, dtype=float).reshape(6)
    wind = tuple(float(w) for w in wind_xyz_mps)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return state_derivative(t, y, wind_xyz_mps=wind, drag_enabled=drag_enabled)

    def event_board_plane(_t: float, y: np.ndarray) -> float:
        return float(y[0] - OCHE_TO_BOARD_X_M)

    event_board_plane.terminal = True  # type: ignore[attr-defined]
    event_board_plane.direction = 1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, max_time_s),
        release_state6,
        events=event_board_plane,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    miss: dict[str, Any] = {
        "hit": False,
        "t_hit": None,
        "state_hit": sol.y[:, -1] if sol.y.size else release_state6.copy(),
        "delta_y_m": 0.0,
        "delta_z_m": 0.0,
        "sol": sol,
    }

    if not sol.t_events or len(sol.t_events[0]) == 0:
        return miss

    t_hit = float(sol.t_events[0][0])
    assert sol.sol is not None
    y_hit = sol.sol(t_hit)
    vx_hit = float(y_hit[3])
    if vx_hit <= 0.0:
        return miss

    y_imp = float(y_hit[1])
    z_imp = float(y_hit[2])
    delta_y_m = y_imp - BULLSEYE_CENTER_Y_M
    delta_z_m = z_imp - BULLSEYE_CENTER_Z_M

    return {
        "hit": True,
        "t_hit": t_hit,
        "state_hit": y_hit,
        "delta_y_m": delta_y_m,
        "delta_z_m": delta_z_m,
        "sol": sol,
    }


def radial_miss_m(dy: float, dz: float) -> float:
    return float(math.hypot(dy, dz))
