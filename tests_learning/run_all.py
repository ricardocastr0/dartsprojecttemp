"""
Run all learning diagnostics (reward math + env rollouts), optionally analyze training CSVs.

Usage (from repository root):

  python3 tests_learning/run_all.py
  python3 tests_learning/run_all.py --episodes 800 --seed 1
  python3 tests_learning/run_all.py --analyze-logs --log-dir logs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--analyze-logs",
        action="store_true",
        help="Run analyze_training_logs.py if episode CSV exists under --log-dir",
    )
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--algorithm", type=str, default="auto", choices=("auto", "sac", "ddpg"))
    p.add_argument("--rolling", type=int, default=50)
    args = p.parse_args()

    scripts = [
        ("reward_analytic.py", []),
        (
            "rollout_diagnostics.py",
            ["--episodes", str(args.episodes), "--seed", str(args.seed)],
        ),
    ]

    for name, extra in scripts:
        path = ROOT / "tests_learning" / name
        cmd = [sys.executable, str(path), *extra]
        print(f"\n{'=' * 60}\n$ {' '.join(cmd)}\n{'=' * 60}\n")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)

    if args.analyze_logs:
        sac_ep = ROOT / args.log_dir / "train_episodes_SAC.csv"
        ddp_ep = ROOT / args.log_dir / "train_episodes.csv"
        has_any = (sac_ep.is_file() and sac_ep.stat().st_size > 0) or (
            ddp_ep.is_file() and ddp_ep.stat().st_size > 0
        )
        if not has_any:
            print(
                f"\n[analyze-logs] Skip: no non-empty train_episodes*.csv under {ROOT / args.log_dir}",
                file=sys.stderr,
            )
            return
        ana = ROOT / "tests_learning" / "analyze_training_logs.py"
        cmd = [
            sys.executable,
            str(ana),
            "--log-dir",
            args.log_dir,
            "--algorithm",
            args.algorithm,
            "--rolling",
            str(args.rolling),
            "--patterns",
            "--plot-dir",
            str(ROOT / "tests_learning" / "figures"),
            "--out-json",
            str(ROOT / "tests_learning" / "figures" / "summary.json"),
        ]
        print(f"\n{'=' * 60}\n$ {' '.join(cmd)}\n{'=' * 60}\n")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
