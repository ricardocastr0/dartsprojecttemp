"""
Analytic terminal reward (released branch) — mirrors env.py formulas without stepping physics.

Use to see how timing alignment and radial miss move the scalar reward.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def dartboard_score_plane(x: float, y: float) -> float:
    dx, dy = x - 0.0, y - 0.0
    r = float(np.sqrt(dx * dx + dy * dy))
    base = 50.0 * float(np.exp(-2.5 * r * r))
    ring_bonus = 0.0
    if r < 0.035:
        ring_bonus = 60.0
    elif r < 0.095:
        ring_bonus = 28.0
    elif r < 0.18:
        ring_bonus = 8.0
    return float(base + ring_bonus)


def terminal_reward_components(
    *,
    dist_m: float,
    pred_dist_m: float,
    release_step: int,
    optimal_release_step: int,
    hit_sigma: float = 0.10,
    release_dist_penalty: float = 15.0,
    release_bonus: float = 20.0,
    timing_std: float = 15.0,
) -> dict[str, float]:
    score = dartboard_score_plane(dist_m, 0.0) if dist_m < 2.0 else dartboard_score_plane(1.0, 0.0)
    hit = 100.0 * float(np.exp(-0.5 * (dist_m / hit_sigma) ** 2))
    base = hit + 0.2 * score - release_dist_penalty * dist_m + release_bonus
    quality = max(0.0, 1.0 - float(pred_dist_m)) * 20.0
    denom = float(timing_std) + 1e-8
    timing = 5.0 * float(np.exp(-0.5 * ((release_step - optimal_release_step) / denom) ** 2))
    total = base + quality + timing
    return {
        "hit_term": hit,
        "score_term": 0.2 * score,
        "dist_penalty": -release_dist_penalty * dist_m,
        "release_bonus": release_bonus,
        "quality_bonus": quality,
        "timing_bonus": timing,
        "total_terminal": total,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analytic terminal reward breakdown.")
    p.add_argument("--release-bonus", type=float, default=20.0)
    args = p.parse_args()

    rb = args.release_bonus

    scenarios = [
        ("ideal_bullseye_timing_aligned", dict(dist_m=0.0, pred_dist_m=0.0, release_step=20, optimal_release_step=20)),
        ("ideal_bull_curriculum_step10_opt25", dict(dist_m=0.0, pred_dist_m=0.0, release_step=10, optimal_release_step=25)),
        ("good_not_perfect", dict(dist_m=0.05, pred_dist_m=0.2, release_step=15, optimal_release_step=18)),
        ("bad_miss", dict(dist_m=0.5, pred_dist_m=0.8, release_step=12, optimal_release_step=30)),
    ]

    print("=== Terminal reward (released branch), analytic ===\n")
    rows = []
    for name, kw in scenarios:
        kw = {**kw, "release_bonus": rb}
        parts = terminal_reward_components(**kw)
        rows.append({"scenario": name, **parts})
        print(f"{name}:")
        for k, v in parts.items():
            print(f"  {k}: {v:.4f}")
        print()

    print(f"Theoretical ceiling (dist=0, pred=0, timing aligned, bonus={rb}): ~{terminal_reward_components(dist_m=0, pred_dist_m=0, release_step=20, optimal_release_step=20, release_bonus=rb)['total_terminal']:.2f}")


if __name__ == "__main__":
    main()
