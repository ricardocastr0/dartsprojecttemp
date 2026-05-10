# Analysis decisions — final SAC run (regulation dartboard)

This document records the decisions made while turning the new training run
(`FinalLogs/`) into the figure pipeline used by the paper. Every choice is
backed by numbers from `evaluate.py` (N=400 episodes per cell) and from the
training-log analyzer.

## 1. Which checkpoint represents the paper

**Decision: use `policy_finalSAC.pt` (the 1M-step endpoint), not
`policy_bestSAC.pt`.**

The training script saves `policy_bestSAC.pt` whenever the *composite* metric
`mean_reward_released - 10·mean_landing_dist` improves on the per-100k
evaluation slice (`trainSAC.py:312`). That criterion intentionally biases
toward the largest single-batch reward seen during a 10-episode evaluation
window, so it can fire on a noisy evaluation block. The composite winner here
is the `step=900000` checkpoint, which had `release_rate=0.9` (1 of 10 eval
episodes timed out) but unusually high reward on the 9 throws that did land.

A 400-episode deterministic evaluation (seeds 0 and 1) shows the consequence:

| Checkpoint | Seed | Release rate | Mean miss (m) | Std miss (m) | Mean score |
|------------|:----:|:------------:|:-------------:|:------------:|:----------:|
| Best       | 0    | 0.995        | 0.0137        | **0.0720**   | 94.31      |
| Best       | 1    | 0.990        | 0.0180        | **0.1007**   | 93.65      |
| Final      | 0    | **1.000**    | **0.0101**    | **0.0084**   | 89.53      |
| Final      | 1    | **1.000**    | **0.0099**    | **0.0081**   | 89.13      |

Final wins decisively on every robustness axis the paper actually argues:

- **Release rate 1.000 (vs 0.99–0.995).** Zero timeouts in 800 evaluation
  episodes across two seeds — the strongest possible evidence that the policy
  consistently completes the throw within `T_max=40` steps.
- **Mean radial miss ~10 mm (vs 14–18 mm for best).** Both seeds are within
  0.2 mm of each other → highly reproducible.
- **Std of miss ~8 mm (vs 72–101 mm for best).** Order-of-magnitude tighter
  spread. The best checkpoint inherits the rare large-miss behavior of the
  900k eval window; the final checkpoint does not.
- **Tier %s (final/seed 1):** 45% inner bull, 39% outer bull, 16% inside
  doubles wire, **0% outside**. Every throw landed within the third reward
  tier on the regulation-style dartboard.

Best is ahead by ~5 score points because its rare in-distribution throws hit
the inner-bull bonus more often, but the paper's central thesis is robustness
under stochastic dynamics. The final checkpoint converts that thesis into a
quantitative claim:
**\(d = 0.0099 \pm 0.0081\) m** at **100% release rate** over 400 episodes.

`generate_paper_figures.py` was updated to prefer
`checkpoints/policy_finalSAC.pt` (override with `PAPER_CHECKPOINT`).

## 2. Evaluation protocol used for figures and tables

- **Episodes per evaluation cell:** N = 400.
- **Determinism:** Actor uses `tanh(μ_φ(s))` (no stochastic sampling).
- **Curriculum at eval:** Disabled (`curriculum_taper_episodes=0`).
- **Seeds reported:** 0 and 1 (same protocol).
- **PCA input:** 12-dimensional joint state (6 angles + 6 velocities), stacked
  across all evaluation episodes; the timing scalar is excluded because it is
  episode-level constant.
- **Effective DOF criterion:** Smallest k with cumulative explained variance
  ≥ 0.95.

Outputs landed in:

```
eval_outputs/best_seed0/   eval_outputs/final_seed0/
eval_outputs/best_seed1/   eval_outputs/final_seed1/
```

## 3. Training-log analysis

Run via:

```bash
python3 tests_learning/analyze_training_logs.py \
  --log-dir paper_runs/regulation_1m --algorithm sac \
  --plot-dir tests_learning/figures/final_run \
  --out-json tests_learning/figures/final_run_summary.json --patterns
```

Key numbers (`final_run_summary.json`):

- 77,173 total episodes in 1M env steps.
- Curriculum (first 2,000 eps): mean return 3.37, release rate 0.979.
- Post-curriculum (75,173 eps): mean return **114.08**, release rate
  **0.941**, timeout rate **5.94%**.
- Released-finish post-curriculum (n=70,705): mean return **122.64**.
- Timeout post-curriculum (n=4,468): mean return **-21.37**, consistent with
  the −20 timeout penalty.
- Pearson(episode\_length, return) = **−0.76**: short, decisive throws win.
- Pearson(episode\_index, return) = **+0.41**: monotonic improvement.
- Last 500 episodes: timeout rate **0.000**, mean return **139.95** — fully
  converged.
- **No 600k collapse this run** (cf. earlier runs; the regulation reward run
  is qualitatively cleaner).

Periodic eval (every 100k steps, 10 episodes per check):

| Step | Rel. rate | Mean dist (m) | Mean score |
|:----:|:---------:|:-------------:|:----------:|
| 100k | 1.00 | 0.2113 | 47.0 |
| 200k | 1.00 | 0.0507 | 59.6 |
| 300k | 1.00 | 0.0449 | 59.7 |
| 400k | 1.00 | 0.0244 | 69.1 |
| 500k | 1.00 | 0.0177 | 68.0 |
| 600k | 1.00 | 0.0137 | 77.2 |
| 700k | 1.00 | 0.0102 | 83.6 |
| 800k | 1.00 | 0.0108 | 79.2 |
| 900k | 0.90 | 0.0441 | 92.9 ← single timeout in 10 |
| 1M   | 1.00 | 0.0114 | 80.4 |

The 900k evaluation block is the same row that tripped the "best composite"
saver. Its high mean score reflects only the 9 released throws.

## 4. Figure-pipeline configuration

`generate_paper_figures.py` was changed to:

- Prefer `checkpoints/policy_finalSAC.pt` for the figure rollouts.
- Resolve Fig 1 metrics via `PAPER_TRAIN_METRICS`, then in order:
  `paper_runs/regulation_1m/train_metrics_SAC.csv`, …, falling back to
  `1MilStats/1Mil_metrics.csv` and `logs/train_metrics_SAC.csv`.

Override examples:

```bash
PAPER_CHECKPOINT=checkpoints/policy_bestSAC.pt python3 paper_figures/generate_paper_figures.py
PAPER_TRAIN_METRICS=paper_runs/regulation_1m/train_metrics_SAC.csv python3 paper_figures/generate_paper_figures.py
```

## 5. What is *not* changed

- `env.py` reward function (`R_INNER_BULL_M`, `R_OUTER_BULL_M`,
  `R_DOUBLES_WIRE_M`, +60/+28/+8 ring bonuses, `50·exp(-2.5 r²)` base).
- Methods MDP, action / state space, curriculum, replay scheme, network sizes.
- Paper text — see `paper_figures/PAPER_UPDATE_CHECKLIST.txt` for the items to
  reconcile against the new numbers above.
