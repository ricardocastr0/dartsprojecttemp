from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flight_physics import (
    PRED_DIST_MISS_M,
    integrate_until_board,
    radial_miss_m,
)

# Regulation-inspired dartboard radial tiers for `_dartboard_score` ring bonuses (meters).
# Inner/outer bull match steel-tip specification; outer tier is inside ~doubles wire (~170 mm).
R_INNER_BULL_M = 0.00635  # 12.7 mm ø inner bull (double bull / 50)
R_OUTER_BULL_M = 0.016  # ~32 mm ø outer bull (single bull / 25), r ≈ 16 mm
R_DOUBLES_WIRE_M = 0.170  # simplified outer bonus boundary (~standard doubles wire radius)


def dartboard_score_plane(
    x: float,
    y: float,
    *,
    tx: float = 0.0,
    ty: float = 0.0,
) -> float:
    """
    Same radial score as `DartEnv._dartboard_score`; bull target defaults to (tx, ty) = (0, 0).
    """
    dx = float(x) - float(tx)
    dy = float(y) - float(ty)
    r = float(np.sqrt(dx * dx + dy * dy))
    base = 50.0 * float(np.exp(-2.5 * r * r))
    ring_bonus = 0.0
    if r < R_INNER_BULL_M:
        ring_bonus = 60.0
    elif r < R_OUTER_BULL_M:
        ring_bonus = 28.0
    elif r < R_DOUBLES_WIRE_M:
        ring_bonus = 8.0
    return float(base + ring_bonus)


@dataclass
class StepInfo:
    released: bool
    # Board-face coordinates relative to bull: (Δy, Δz) in meters; None if unavailable
    landing_xy: tuple[float, float] | None
    score: float


class DartEnv:
    """
    Minimal 6-DOF arm + dart throw environment.

    State (13):
      - 6 joint angles
      - 6 joint velocities
      - 1 normalized optimal_release_step (resampled per reset; exposed so the timing
        bonus is not a hidden variable)

    Action (7):
      - 6 joint torques (continuous, [-1, 1] expected)
      - 1 release signal (continuous, [0, 1] expected)

    Joint dynamics (semi-implicit at dt): v_{t+1} = clip(v_t + dt*(k_tau*tau - damping*v_t));
    theta_{t+1} = theta_t + dt*v_{t+1} — velocity used for the position update is the post-step value.

    Release is **synthetic** (no FK): world-frame release point is a fixed pivot `_release_xyz`.
    Throw **direction** from joints 1–2 (azimuth `q0`, elevation `q1`); **speed** from the magnitude of
    joint velocities 3–6 mapped into [6, 12] m/s. Flight: SPEC frame integration in `flight_physics.py`.
    """

    def __init__(
        self,
        dt: float = 0.04,
        max_steps: int = 40,
        release_threshold: float = 0.40,
        release_bonus: float = 0.0,
        curriculum_taper_episodes: int = 0,
        min_release_steps: int = 10,
        seed: int = 0,
        release_xyz: tuple[float, float, float] | None = None,
        wind_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        drag_enabled: bool = True,
        flight_rtol: float = 1e-6,
        flight_atol: float = 1e-9,
        k_tau: float = 7.5,
        damping: float = 1.2,
        max_vel: float = 12.0,
        reset_angle_low: float = -0.1,
        reset_angle_high: float = 0.1,
        reset_vel_low: float = -0.05,
        reset_vel_high: float = 0.05,
    ) -> None:
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.release_threshold = float(release_threshold)
        self.release_bonus = float(release_bonus)
        self.curriculum_taper_episodes = int(curriculum_taper_episodes)
        self.min_release_steps = int(min_release_steps)
        self.rng = np.random.default_rng(seed)

        self.state_dim = 13
        self.action_dim = 7
        # Normalization constant for optimal_release_step in the observation
        # (matches the upper end of reset() clip below).
        self._optimal_step_max = 35.0

        # Joint dynamics parameters (kept explicit to allow future tuning)
        self.k_tau = float(k_tau)
        self.damping = float(damping)
        self.max_vel = float(max_vel)
        self._reset_angle_low = float(reset_angle_low)
        self._reset_angle_high = float(reset_angle_high)
        self._reset_vel_low = float(reset_vel_low)
        self._reset_vel_high = float(reset_vel_high)
        self.max_angle = np.pi

        self.release_height = 1.5  # default z at release if release_xyz not fully set
        rz = float(release_xyz[2]) if release_xyz is not None else float(self.release_height)
        rx = float(release_xyz[0]) if release_xyz is not None else 0.0
        ry = float(release_xyz[1]) if release_xyz is not None else 0.0
        self._release_xyz = (rx, ry, rz)

        self.wind_xyz = tuple(float(w) for w in wind_xyz)
        self.drag_enabled = bool(drag_enabled)
        self.flight_rtol = float(flight_rtol)
        self.flight_atol = float(flight_atol)

        # Completed episodes since construction (increments on each terminal step).
        self._episodes_completed = 0

        # Curriculum: p_force at episode start (set in reset)
        self._curriculum_p_force = 0.0

        # Bull is origin in (Δy, Δz) for _dartboard_score (same formula as legacy ground target)
        self.target_x = 0.0
        self.target_y = 0.0

        # Internal state (_t = timestep index within episode; incremented each step(), reset in reset())
        self._t = 0
        self._angles = np.zeros(6, dtype=np.float32)
        self._vels = np.zeros(6, dtype=np.float32)

        self.step_penalty = 0.005
        self.timeout_penalty = 20.0

        self.shaping_dist_weight = 0.02  # lighter shaping so terminal release signals dominate
        self.hit_sigma = 0.10
        self.release_dist_penalty = 15.0
        self.timing_std = 15.0  # Gaussian width for timing bonus vs optimal_release_step

    def reset(self) -> np.ndarray:
        self._t = 0
        self.optimal_release_step = int(
            np.clip(np.round(self.rng.normal(loc=20.0, scale=5.0)), 10, 35)
        )
        k = self.curriculum_taper_episodes
        if k > 0:
            self._curriculum_p_force = max(
                0.0, 1.0 - float(self._episodes_completed) / float(k)
            )
        else:
            self._curriculum_p_force = 0.0

        self._angles = self.rng.uniform(
            low=self._reset_angle_low, high=self._reset_angle_high, size=(6,)
        ).astype(np.float32)
        self._vels = self.rng.uniform(
            low=self._reset_vel_low, high=self._reset_vel_high, size=(6,)
        ).astype(np.float32)
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected action shape (7,), got {action.shape}.")

        torques = np.clip(action[:6], -1.0, 1.0)
        release = float(np.clip(action[6], 0.0, 1.0))

        acc = self.k_tau * torques - self.damping * self._vels
        self._vels = np.clip(self._vels + self.dt * acc, -self.max_vel, self.max_vel)
        self._angles = np.clip(self._angles + self.dt * self._vels, -self.max_angle, self.max_angle)

        self._t += 1

        if self._t >= self.min_release_steps and self._curriculum_p_force > 0.0:
            if self.rng.random() < self._curriculum_p_force:
                release = 1.0

        dy, dz, hit = self._board_deltas_now()
        pred_dist = radial_miss_m(dy, dz) if hit else PRED_DIST_MISS_M

        done = False
        can_release = self._t >= self.min_release_steps
        reward = -self.step_penalty - self.shaping_dist_weight * pred_dist
        info = StepInfo(released=False, landing_xy=None, score=0.0)

        if (can_release and release >= self.release_threshold) or self._t >= self.max_steps:
            done = True
            released = bool(can_release and release >= self.release_threshold)
            dy_t, dz_t, hit_t = self._board_deltas_now()
            if hit_t:
                dist = radial_miss_m(dy_t, dz_t)
                landing = (float(dy_t), float(dz_t))
            else:
                dist = PRED_DIST_MISS_M
                # Synthetic far point so _dartboard_score is low (not bull at 0,0)
                landing = (PRED_DIST_MISS_M, 0.0)
            score = self._dartboard_score(landing[0], landing[1])

            if released:
                hit = 100.0 * float(np.exp(-0.5 * (dist / self.hit_sigma) ** 2))
                terminal_reward = (
                    hit + 0.2 * score - self.release_dist_penalty * dist + self.release_bonus
                )
                # Bonus for releasing when predicted shot was already decent (pred_dist in meters).
                release_quality_bonus = max(0.0, 1.0 - float(pred_dist)) * 20.0
                terminal_reward += release_quality_bonus
                release_step = int(self._t)
                denom = float(self.timing_std) + 1e-8
                release_timing_bonus = 5.0 * float(
                    np.exp(-0.5 * ((release_step - int(self.optimal_release_step)) / denom) ** 2)
                )
                terminal_reward += release_timing_bonus
                reward = float(terminal_reward)
            else:
                reward = float(-self.timeout_penalty)
            info = StepInfo(released=released, landing_xy=landing, score=float(score))

        if done:
            self._episodes_completed += 1

        return self._get_state(), reward, done, {"info": info, "t": self._t, "pred_dist": pred_dist}

    def _get_state(self) -> np.ndarray:
        timing_norm = float(self.optimal_release_step) / float(self._optimal_step_max)
        return np.concatenate(
            [self._angles, self._vels, np.array([timing_norm], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

    def _release_position_xyz(self) -> np.ndarray:
        """Fixed pivot in world frame (no link geometry)."""
        ox, oy, oz = self._release_xyz
        return np.array([ox, oy, oz], dtype=np.float64)

    def _release_direction_unit(self) -> np.ndarray:
        """Unit direction from azimuth (q0) and elevation (q1); +x is toward the board at yaw=pitch=0."""
        yaw = float(self._angles[0])
        pitch = float(self._angles[1])
        cp = float(np.cos(pitch))
        d = np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)], dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return d / n

    def _throw_speed_mps(self) -> float:
        """Map joint angular speeds 3–6 (indices 2:6) to throw speed in [6, 12] m/s."""
        qd = self._vels[2:6].astype(np.float64)
        wn = float(np.linalg.norm(qd))
        scale = max(float(self.max_vel), 1e-8)
        u = min(1.0, wn / scale)
        return 6.0 + 6.0 * u

    def _synthetic_velocity_raw(self) -> np.ndarray:
        sp = self._throw_speed_mps()
        d = self._release_direction_unit()
        v = (sp * d).astype(np.float32)
        return v

    def get_release_state6(self) -> np.ndarray:
        """World-frame [x,y,z,vx,vy,vz] at current joint state (for flight / viz)."""
        release_pos = self._release_position_xyz()
        v = self._release_velocity_xyz()
        return np.asarray(
            [
                release_pos[0],
                release_pos[1],
                release_pos[2],
                float(v[0]),
                float(v[1]),
                float(v[2]),
            ],
            dtype=np.float64,
        )

    def _release_state6(self) -> np.ndarray:
        return self.get_release_state6()

    def _board_deltas_now(self) -> tuple[float, float, bool]:
        out = integrate_until_board(
            self._release_state6(),
            wind_xyz_mps=self.wind_xyz,
            drag_enabled=self.drag_enabled,
            rtol=self.flight_rtol,
            atol=self.flight_atol,
        )
        if not out["hit"]:
            return float(out["delta_y_m"]), float(out["delta_z_m"]), False
        return float(out["delta_y_m"]), float(out["delta_z_m"]), True

    def raw_release_velocity_xyz(self) -> np.ndarray:
        """Synthetic release velocity (same as `_release_velocity_xyz`)."""
        return self._synthetic_velocity_raw()

    def _release_velocity_xyz(self) -> np.ndarray:
        return self._synthetic_velocity_raw()

    def _dartboard_score(self, x: float, y: float) -> float:
        """
        Smooth radial base plus discrete ring bonuses; x,y are Δy, Δz relative to bull.
        Tier radii: `R_INNER_BULL_M`, `R_OUTER_BULL_M`, `R_DOUBLES_WIRE_M`.
        """
        return dartboard_score_plane(x, y, tx=self.target_x, ty=self.target_y)
