"""
Paper checklist smoke tests: env shapes, throw speed band, replay stratification.

Run from repo root: python3 tests_learning/verify_checklist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from config import CFG, Config
from env import DartEnv
from replay_buffer import ReplayBuffer


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    print("=== Checklist verification ===\n")
    cfg: Config = CFG
    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=0,
        min_release_steps=cfg.min_release_steps,
        seed=0,
        k_tau=cfg.k_tau,
        damping=cfg.damping,
        max_vel=cfg.max_vel,
        reset_angle_low=cfg.reset_angle_low,
        reset_angle_high=cfg.reset_angle_high,
        reset_vel_low=cfg.reset_vel_low,
        reset_vel_high=cfg.reset_vel_high,
    )
    ok_all = True

    s = env.reset()
    ok_all &= _ok("state shape (13,)", s.shape == (13,), str(s.shape))
    a = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    ok_all &= _ok("random action shape (7,)", a.shape == (7,), str(a.shape))
    s2, _r, _done, _out = env.step(a)
    ok_all &= _ok("step returns state (13,)", s2.shape == (13,), str(s2.shape))

    # Release speed in [6, 12] m/s (vector norm)
    bad = 0
    for _ in range(50):
        env.reset()
        v = env.raw_release_velocity_xyz()
        sp = float(np.linalg.norm(v))
        if not (6.0 - 1e-3 <= sp <= 12.0 + 1e-3):
            bad += 1
    ok_all &= _ok(
        "throw speed norm in [6,12] m/s (50 random resets)",
        bad == 0,
        f"{bad} violations",
    )

    # Synthetic buffer: alternating release-tagged single-step episodes
    buf = ReplayBuffer(12, 7, 10_000, seed=2)
    for ep in range(600):
        s = np.zeros(12, dtype=np.float32)
        a = np.zeros(7, dtype=np.float32)
        slot = buf.add(s, a, float(ep), s, True)
        buf.mark_episode([slot], released=(ep % 2 == 0))

    mean_fracs: list[float] = []
    for _ in range(300):
        idx = buf._stratified_indices(cfg.batch_size, cfg.min_release_fraction)
        frac = float(np.mean(buf.released_episode[idx] == 1))
        mean_fracs.append(frac)
    mf = float(np.mean(mean_fracs))
    ok_all &= _ok(
        f"stratified batches ~50% release-tagged (mean frac={mf:.3f})",
        abs(mf - 0.5) < 0.08,
        f"target 0.50 (batch_size={cfg.batch_size})",
    )

    # Timeout terminal reward
    env_to = DartEnv(
        curriculum_taper_episodes=0,
        seed=99,
        max_steps=5,
        min_release_steps=100,
        release_threshold=0.99,
    )
    env_to.reset()
    z = np.zeros(7, dtype=np.float32)
    last_r = 0.0
    while True:
        _s, last_r, done, _ = env_to.step(z)
        if done:
            break
    ok_all &= _ok(
        "timeout reward equals -timeout_penalty",
        abs(last_r + env_to.timeout_penalty) < 1e-5,
        f"r={last_r}",
    )

    print(f"\nconfig.min_release_fraction={cfg.min_release_fraction}")
    print(f"config.curriculum_taper_episodes={cfg.curriculum_taper_episodes}")
    print("\nDone.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
