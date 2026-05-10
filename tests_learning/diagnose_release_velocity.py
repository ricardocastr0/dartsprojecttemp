"""
Summarize synthetic release velocity after warmup torques (speed norm should stay in [6,12] m/s).

Run from repo root: python tests_learning/diagnose_release_velocity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from config import CFG, Config
from env import DartEnv


def _run(cfg: Config, *, n_samples: int, warmup_steps: int, seed: int) -> dict:
    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=0,
        min_release_steps=cfg.min_release_steps,
        seed=seed,
        k_tau=cfg.k_tau,
        damping=cfg.damping,
        max_vel=cfg.max_vel,
        reset_angle_low=cfg.reset_angle_low,
        reset_angle_high=cfg.reset_angle_high,
        reset_vel_low=cfg.reset_vel_low,
        reset_vel_high=cfg.reset_vel_high,
    )
    rng = np.random.default_rng(seed + 42)

    speeds: list[float] = []

    for i in range(n_samples):
        if n_samples >= 2000 and (i + 1) % max(1, n_samples // 8) == 0:
            print(f"  progress: {i + 1}/{n_samples}", file=sys.stderr, flush=True)
        env.reset()
        for _ in range(warmup_steps):
            a = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
            env.step(a)

        v_raw = env.raw_release_velocity_xyz()
        speeds.append(float(np.linalg.norm(v_raw)))

    sp_arr = np.asarray(speeds, dtype=np.float64)

    return {
        "config": {
            "k_tau": cfg.k_tau,
            "damping": cfg.damping,
            "max_vel": cfg.max_vel,
            "min_release_steps": cfg.min_release_steps,
            "warmup_steps": warmup_steps,
            "n_samples": n_samples,
        },
        "speed_norm_mean": float(np.mean(sp_arr)),
        "speed_norm_std": float(np.std(sp_arr)),
        "speed_norm_min": float(np.min(sp_arr)),
        "speed_norm_max": float(np.max(sp_arr)),
        "fraction_in_6_12": float(np.mean((sp_arr >= 6.0 - 1e-6) & (sp_arr <= 12.0 + 1e-6))),
        "targets": {"speed_norm_in_[6,12]_m/s": 1.0},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-samples",
        type=int,
        default=800,
        help="Each sample runs warmup_steps env.step() calls (flight integration each step); "
        "use 4000 only if you can wait.",
    )
    p.add_argument("--warmup-steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = _run(CFG, n_samples=args.n_samples, warmup_steps=args.warmup_steps, seed=args.seed)
    print(json.dumps(out, indent=2))

    ok = out["fraction_in_6_12"] >= 0.999
    print(f"\n=== Pass (expect speed norm in [6,12] m/s): {ok} ===")


if __name__ == "__main__":
    main()
