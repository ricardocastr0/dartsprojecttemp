"""
Roll out DartEnv with random actions (same spirit as SAC warmup) and summarize:

- Episode length / return / released distributions
- Split by curriculum phase (first K episodes vs rest) when curriculum_taper_episodes > 0
- Timeout rate after curriculum (released=0)

Run from repo root: python tests_learning/rollout_diagnostics.py --episodes 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# bootstrap before project imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from config import CFG, Config
from env import DartEnv


def rollout_random(cfg: Config, *, episodes: int, seed: int, curriculum_taper_episodes: int | None) -> dict:
    ce = cfg.curriculum_taper_episodes if curriculum_taper_episodes is None else curriculum_taper_episodes
    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        release_bonus=cfg.release_completion_bonus,
        curriculum_taper_episodes=ce,
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
    rng = np.random.default_rng(seed + 404)

    records: list[dict] = []
    for ep in range(episodes):
        optimal = None
        s = env.reset()
        optimal = int(env.optimal_release_step)
        ep_ret = 0.0
        ep_len = 0
        done = False
        while not done:
            a = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
            s, r, done, out = env.step(a)
            ep_ret += float(r)
            ep_len += 1
        info = out["info"]
        records.append(
            {
                "episode_index": ep + 1,
                "episode_return": ep_ret,
                "episode_length": ep_len,
                "released": bool(info.released),
                "optimal_release_step": optimal,
                "in_curriculum_phase": (ep + 1) <= ce,
            }
        )

    def summarize(sub: list[dict]) -> dict:
        if not sub:
            return {}
        rets = [x["episode_return"] for x in sub]
        lens = [x["episode_length"] for x in sub]
        rel = [x["released"] for x in sub]
        return {
            "n": len(sub),
            "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)),
            "return_min": float(np.min(rets)),
            "return_max": float(np.max(rets)),
            "length_mean": float(np.mean(lens)),
            "length_std": float(np.std(lens)),
            "release_rate": float(np.mean(rel)),
            "timeout_count": int(sum(1 for x in sub if not x["released"])),
        }

    cur = [r for r in records if r["in_curriculum_phase"]]
    post = [r for r in records if not r["in_curriculum_phase"]]
    post_timeouts = [r for r in post if not r["released"]]

    out = {
        "config": {
            "max_steps": cfg.max_steps,
            "curriculum_taper_episodes": ce,
            "min_release_steps": cfg.min_release_steps,
            "release_threshold": cfg.release_threshold,
            "k_tau": cfg.k_tau,
            "max_vel": cfg.max_vel,
            "seed": seed,
        },
        "all_episodes": summarize(records),
        "curriculum_phase_only": summarize(cur),
        "post_curriculum_only": summarize(post),
        "post_curriculum_timeout_rate": len(post_timeouts) / max(1, len(post)),
        "records_sample_head": records[:5],
        "records_sample_post_curriculum": post[:5] if post else [],
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--curriculum-taper-episodes",
        type=int,
        default=-1,
        help="-1 = use config.py curriculum_taper_episodes",
    )
    args = p.parse_args()

    cfg = CFG
    ce = cfg.curriculum_taper_episodes if args.curriculum_taper_episodes < 0 else args.curriculum_taper_episodes

    result = rollout_random(cfg, episodes=args.episodes, seed=args.seed, curriculum_taper_episodes=ce)

    print(json.dumps(result, indent=2))
    print("\n=== Interpretation hints ===")
    print("- Curriculum phase (episode_index ≤ taper length): forced-release probability decreases linearly.")
    print("- Post taper: random actions → timeouts possible (released=0, length=max_steps).")
    print("- Compare return_mean between phases; apples-to-oranges if lengths differ.")


if __name__ == "__main__":
    main()
