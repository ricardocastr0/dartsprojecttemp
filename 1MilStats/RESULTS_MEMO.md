# 1M-step SAC run — paper-oriented results memo

**Last regenerated:** full pipeline re-run (four `evaluate.py` jobs + `analyze_training_logs.py`). Sources: `1Mil_metrics.csv`, `1Mil_episodes.csv` (**70,944** episode rows, ~2.5 MB — not a performance issue), frozen checkpoints `1MilbestPolicy.pt` / `1MilfinalPolicy.pt`, fresh `evaluate.py` rollouts (**400** episodes each). Training-time eval rows used **40 episodes** per checkpoint unless your CLI overrode `eval_episodes`.

**Analyzer file naming:** `tests_learning/analyze_training_logs.py` expects `train_episodes_SAC.csv` and `train_metrics_SAC.csv`. In `1MilStats/` these are **symlinks** to `1Mil_episodes.csv` and `1Mil_metrics.csv`.

---

## 1. Training eval trajectory (`1Mil_metrics.csv`)

| step (env) | release_rate | mean_landing_dist (m) | mean_score | Notes |
|------------|--------------|------------------------|------------|--------|
| 100k | 0.775 | 1.44 | 33.6 | Early learning |
| 200k | 0.900 | 0.127 | 68.5 | Strong gain |
| 300k | 1.000 | 0.061 | 82.2 | Stable release |
| 400k | 0.675 | 0.243 | 69.0 | **Variance spike** (13/40 timeout in eval) |
| 500k | 1.000 | **0.022** | 105.9 | **Best training-eval accuracy** in table |
| **600k** | **0.125** | **6.58** | 18.3 | **Severe collapse** (35/40 timeout) |
| 700k | 1.000 | 0.015 | 110.0 | Recovery |
| 800k | 1.000 | **0.012** | 109.2 | **Tightest** mean miss in table |
| 900k | 1.000 | 0.015 | 108.4 | Plateau |
| 1M | 1.000 | 0.022 | 106.7 | Slight drift vs 800k |

**Conclusions for text**

- **Mid-run instability:** At **600k steps** the periodic eval shows **commitment collapse** (release_rate 12.5%) and landing statistics dominated by timeouts—report as a transient training pathology, not held-out test performance.
- **Peak training-eval accuracy** sits around **500k–800k** (mean landing distance ≈ **1.2–2.2 cm**). The **400k** row shows milder noise (release_rate 67.5%).
- **Final row (1M)** is strong but **not strictly monotonic** vs 800k; report **best checkpoint** vs **final** separately.

---

## 2. Frozen checkpoints — off-policy evaluation (400 episodes, `mean_action`)

| Checkpoint | seed | release_rate | mean_landing_dist_m | std_landing_dist_m | mean_score | eff. DOF (95%) | PC1 explained |
|------------|------|--------------|---------------------|--------------------|------------|----------------|---------------|
| `1MilbestPolicy.pt` | 0 | 0.995 | 0.0420 | 0.3949 | 108.13 | 7 | 0.429 |
| `1MilfinalPolicy.pt` | 0 | 0.993 | **0.0296** | **0.0934** | 106.09 | 8 | 0.495 |
| `1MilbestPolicy.pt` | 1 | 0.983 | 0.0896 | 0.5853 | 106.44 | 5 | 0.630 |
| `1MilfinalPolicy.pt` | 1 | 0.985 | 0.0482 | 0.2855 | 105.25 | 8 | 0.481 |

**Takeaways**

- On **seed 0**, **final** beats **best** on **mean** radial miss and **std** — the filename “best” does not imply dominance at **N=400**. Prefer **final** for tight scatter, or report **both**.
- **Seed sensitivity:** “Best” mean distance shifts **4.2 cm → 9.0 cm** across seeds; **final** is more stable (**3.0 cm → 4.8 cm**).
- **PCA:** Effective DOF **5–8** (angles+velocities, 12-D); **joint 2** has highest torque variance — good for a behavior caption.

**Artifacts — `evaluate.py` outputs**

| Directory | Contents |
|-----------|----------|
| `eval_outputs/1mil_best/` | `1MilbestPolicy.pt`, seed **0** (`--print-cfg` in console): PCA, torque bars, landing scatter/hexbin, dual PCA, histograms |
| `eval_outputs/1mil_final/` | `1MilfinalPolicy.pt`, seed **0** |
| `eval_outputs/1mil_best_seed1/` | best, seed **1** |
| `eval_outputs/1mil_final_seed1/` | final, seed **1** |

---

## 3. Episode-level analyzer (`tests_learning/analyze_training_logs.py`)

**Command (default curriculum K from `Config`, 2000):**

```bash
python3 tests_learning/analyze_training_logs.py --log-dir 1MilStats --algorithm sac \
  --rolling 50 --patterns --out-json tests_learning/figures/summary_1mil.json \
  --plot-dir tests_learning/figures/1mil
```

**Scale:** **70,944** episodes — completes in **~2 s** on typical hardware.

### Summary statistics

| Segment | n episodes | return mean | release_rate | timeout_rate |
|---------|------------|-------------|--------------|--------------|
| Curriculum (ep index ≤ 2000) | 2,000 | 3.18 | 0.997 | 0.003 |
| Post-curriculum | 68,944 | 114.16 | 0.930 | 0.071 |
| All | 70,944 | 111.03 | 0.932 | 0.069 |

**Post-curriculum splits:** released finishes **64,035** (return mean **124.55**); timeout/horizon **4,909** (return mean **-21.35**).

**Rolling return** (post-curriculum, window **50**): first smoothed mean **≈ −18.3**, last **≈ 139.3** (last episode index **70,944**, env_step **≈ 999,989**).

**Training eval checkpoints (from metrics CSV):** **10** rows; **Δ mean_landing_dist** (last − first) = **−1.42 m** (improvement toward bull).

**Pattern scan**

- Overall timeout rate **~6.9%**; **last 100** episodes: return mean **~141**, timeout **1%**.
- **prev100 vs last100** timeouts: **0% → 1%** (no EOF collapse).
- **Early post** (~ep 501–600): return mean **−19.4**, timeout rate **93%** vs **late** window **141** / **1%** timeout — curriculum handoff is sharp.
- **Pearson (post-curriculum):** episode_index vs return **+0.43**; episode_length vs return **−0.77**.

**Outputs**

- **JSON:** [`tests_learning/figures/summary_1mil.json`](../tests_learning/figures/summary_1mil.json)
- **Plots:** [`tests_learning/figures/1mil/analysis_training.png`](../tests_learning/figures/1mil/analysis_training.png), [`training_eval_overlay.png`](../tests_learning/figures/1mil/training_eval_overlay.png)

---

## 4. Recommended figures for IEEE draft

| Figure | Path | Role |
|--------|------|------|
| Training curves (tabular / custom plot) | `1Mil_metrics.csv` | Collapse at **600k**, recovery |
| Episode return + curriculum | `tests_learning/figures/1mil/analysis_training.png` | Learning curve vs episode index |
| Eval overlay | `tests_learning/figures/1mil/training_eval_overlay.png` | Periodic eval metrics vs env step |
| Landing scatter / hexbin | `eval_outputs/1mil_final/` (or best) | Qualitative accuracy |
| PCA cumulative variance | `eval_outputs/.../pca_variance.png` | Effective dimensionality |
| PCA bars | `pca_explained_bars.png` | PC narrative |
| Torque variance | `torque_variance.png` | Joint usage |
| Dual PCA | `*_rollout_first.png` / `*_rollout_second.png` | Stability across eval halves |

---

## 5. Limitations (cite explicitly)

- **Deterministic eval:** `evaluate.py` uses **`policy.mean_action`** for SAC (not stochastic rollouts).
- **Sample sizes:** Training CSV uses **N=40** eval episodes per row; large-sample tables use **N=400** — state **N** wherever compared.
- **Checkpoint names:** `1MilbestPolicy.pt` / `1MilfinalPolicy.pt` are **user-named**; tie to training step / composite in methods if cited.

---

## 6. One-sentence synthesis for abstract/discussion

Learning reaches **centimeter-scale** mean miss with **near-complete release** under deterministic evaluation; **training-time eval at ~600k steps** shows a **sharp transient failure**, then **recovery**; **episode-level logs** confirm **long-run improvement** after curriculum while retaining **~7%** timeout episodes post-curriculum—**report checkpoint choice** and **eval protocol** clearly.
