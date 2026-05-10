# Learning diagnostics (`tests_learning/`)

Small scripts for **reward math**, **random-policy rollouts**, **smoke checks**, and **reading your real training CSVs**. Everything assumes you run commands from the **repository root** (`rl_dart_6dof/`).

---

## Which tool for which question?

| You want to… | Use this | What you get |
|--------------|-----------|----------------|
| See whether learning improved over time (returns, timeouts, curriculum vs after) using **`logs/train_episodes*.csv`** | **`analyze_training_logs.py`** | Text summary to stdout; optional **`analysis_training.png`**, **`training_eval_overlay.png`**, **`summary.json`** |
| Plot eval metrics vs env step without the full analysis panels | Root **[`plot_training.py`](../plot_training.py)** | Figures from `train_metrics*.csv` + episodes (simpler than this folder’s analyzer) |
| Know what “good” terminal reward *could* be (timing vs radial miss; ~165 ceiling) | **`reward_analytic.py`** | Printed JSON-ish breakdown of analytic reward components (no env, no your logs) |
| Know how a **random** policy behaves in the env (timeout rate, curriculum split) | **`rollout_diagnostics.py`** | Printed stats for uniform random actions—not a substitute for reading your training CSV |
| Sanity-check throw speed band **[6, 12] m/s** after warmup | **`diagnose_release_velocity.py`** | Speed norm stats |
| Quick env / replay checks (shapes, speed band, stratified batches) | **`verify_checklist.py`** | Pass/fail lines |

**Training diagnosis** = **`analyze_training_logs.py`** (plus optional **`plot_training.py`** for quick curves). The other scripts are **reference baselines** or **engineering checks**, not replacements for reading episode logs.

---

## Recommended workflow: diagnose a training run

1. **Confirm you have logs.** After `trainSAC.py` / `train.py`, expect under `--log-dir` (default `logs/`):
   - `train_episodes_SAC.csv` or `train_episodes.csv` — one row per finished training episode (`episode_return`, `episode_length`, `released`, …).
   - `train_metrics_SAC.csv` or `train_metrics.csv` — periodic **eval** snapshots (mean reward, landing distance, …).

2. **Run the analyzer** (pick algorithm if both SAC and DDPG logs exist):

   ```bash
   python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac --rolling 50
   ```

   - Read the **printed sections**: segment stats (curriculum vs post), **post-curriculum** split by released vs timeout, **rolling return** (last window), and **eval first / last / delta** when metrics CSV exists.

3. **Add `--patterns`** when you care *when* things broke (binned by `episode_index`, tail streaks, correlations)—not just averages:

   ```bash
   python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac --patterns --rolling 50
   ```

4. **Inspect artifacts** (default plot dir is `tests_learning/figures/` if you omit `--plot-dir`):

   - **`analysis_training.png`** — returns & rolling mean, episode length, curriculum vs post histograms, eval-related panels.
   - **`training_eval_overlay.png`** — eval **`mean_reward`**, **`mean_landing_dist`**, **`release_rate`** vs **`step`** when `train_metrics*.csv` has multiple rows.
   - **`summary.json`** — full structured output (including **`pattern_analysis`** when `--patterns`).

5. **Match curriculum settings to the run.** The CSV does **not** store `curriculum_taper_episodes`. By default the analyzer uses **[`config.CFG.curriculum_taper_episodes`](../config.py)**. If that value changed since training, pass the value from the **training run** explicitly:

   ```bash
   python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac --curriculum-episodes 2000
   ```

   (`--curriculum-episodes -1` keeps the config default.)

6. **`--rolling N`** is a **number of consecutive episodes** (trailing mean over `episode_return` rows), not env steps. Tune **`N`** if the curve is too noisy or too flat.

7. If training used a different horizon than current `config`, align timeout detection:

   ```bash
   python3 tests_learning/analyze_training_logs.py --log-dir logs --max-steps-override 80
   ```

---

## One command: synthetic checks + log analysis

**`run_all.py`** always runs **`reward_analytic.py`** then **`rollout_diagnostics.py`** (defaults: 600 episodes, seed 0). With **`--analyze-logs`** it **also** runs **`analyze_training_logs.py`** *if* a non-empty `train_episodes*.csv` exists under `--log-dir`, passing **`--patterns`**, writing **`tests_learning/figures/analysis_training.png`**, **`training_eval_overlay.png`**, and **`tests_learning/figures/summary.json`**.

```bash
python3 tests_learning/run_all.py
python3 tests_learning/run_all.py --episodes 800 --seed 1
python3 tests_learning/run_all.py --analyze-logs --log-dir logs
python3 tests_learning/run_all.py --analyze-logs --log-dir logs --algorithm sac --rolling 50
```

Use this when you want baselines + log analysis in one shot; use **`analyze_training_logs.py`** alone when you only care about CSVs.

---

## Direct commands (training logs)

```bash
# Default: stdout summary + figures under tests_learning/figures/
python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac --rolling 50

# Extra collapse / tail / correlation diagnostics
python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac --patterns

# Stats and JSON only (no PNG)
python3 tests_learning/analyze_training_logs.py --log-dir logs --no-plot --out-json tests_learning/figures/summary.json

# Let analyzer pick SAC vs DDPG logs automatically
python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm auto
```

**`--algorithm`:** `auto` prefers `*_SAC` when both exist and are non-empty; `sac` / `ddpg` force filenames.

---

## Other scripts (reference / smoke)

```bash
python3 tests_learning/reward_analytic.py
python3 tests_learning/rollout_diagnostics.py --episodes 600 --seed 0
python3 tests_learning/rollout_diagnostics.py --curriculum-taper-episodes 0 --episodes 200
python3 tests_learning/diagnose_release_velocity.py
python3 tests_learning/verify_checklist.py
```

### Local defaults (`analysis_params.py`)

Edit **`DEFAULT_ROLLING_WINDOW`** in **`analysis_params.py`** to change the default **`--rolling`** for **`analyze_training_logs.py`** without touching project-wide config.

---

## Script summary

| Script | Purpose |
|--------|---------|
| **`analyze_training_logs.py`** | **Main training diagnostic**: curriculum vs post segments, rolling returns, eval trends, optional **`--patterns`**, PNG + JSON. |
| **`reward_analytic.py`** | Closed-form **terminal reward** (released branch): timing vs radial miss; illustrates ~165-scale spikes. |
| **`rollout_diagnostics.py`** | **`DartEnv`** with **uniform random** actions: baseline lengths/returns/timeouts vs curriculum—not your trained policy. |
| **`verify_checklist.py`** | Smoke tests: shapes, **[6,12] m/s** throw speed, replay stratification. |
| **`diagnose_release_velocity.py`** | Stats for **`raw_release_velocity_xyz()`** after warmup (speed norm in band). |
| **`run_all.py`** | Runs reward + rollout diagnostics; optional **`--analyze-logs`** chains **`analyze_training_logs.py`** as above. |

---

## Interpreting analyzer output (short)

- **Curriculum phase** (`episode_index ≤ K`): forced-release probability ramps down over the first **`K`** episodes (`K` = taper you pass or config). Returns and lengths reflect that schedule plus RNG.
- **Post-curriculum**: more timeouts possible; JSON **`segments_detail.post_curriculum_conditionals`** splits **released** vs **timeout / horizon** mean returns.
- **`tail_window_timeout_fraction`**: near **1.0** means the **global** last rolling mean is dominated by timeouts—prefer **`rolling_return_post_curriculum_only`** in JSON or adjust **`--rolling`**.
- **`pattern_analysis`** (`--patterns`): **`bins_by_episode_index`** shows **where** timeout rate or mean return shifts (often clearer than one global post average); **`suffix_timeout_run_from_eof`** is the trailing timeout streak length from the end of the CSV.
- **Train return spikes ~160**: consistent with analytic ceiling; **`mean_landing_dist`** (eval metrics) is often a clearer **skill** signal than noisy training returns.

---

## Dependencies

- **`reward_analytic`**, **`rollout_diagnostics`**, **`analyze_training_logs`** (stats): `numpy`, stdlib `csv`.
- **`analyze_training_logs`** (plots): `matplotlib` (optional; use **`--no-plot`** if unavailable).

Plots use the **Agg** backend (no display required).
