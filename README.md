# RL Dart 6-DOF (actor–critic stub)

Self-contained PyTorch project: a **6-joint** arm plus a **continuous release** dimension is trained with a **DDPG-style** actor–critic (deterministic policy + Q with target network and soft updates).

## Coordinate frame and flight (SPEC-aligned)

World frame: **+x toward the board**, **+y left**, **+z up** (transcribed from course SPEC; no external `dartrobot` package).

- Vertical board plane at **x = 2.37 m** (oche-to-board).
- Bullseye center **(2.37, 0, 1.73) m**.
- After release, the dart is integrated with **gravity** and optional **quadratic drag** (`v_rel = v - wind`) until the trajectory crosses the board in **+x**; landing is reported as **Δy, Δz** relative to the bull.
- Reward **formulas** (Gaussian hit, penalties, smooth `_dartboard_score`) are unchanged; ballistic **Δy, Δz**
  come from integrating release state through `flight_physics.py`.

Implementation: `flight_physics.py` (constants + `scipy.integrate.solve_ivp`). **Projecto** is optional reading only, not a runtime dependency.

## Release model (synthetic, not FK)

World-frame release **position** is a **fixed pivot** (`release_xyz` / default z).  
Throw **direction** is set by joint angles **1–2** (azimuth / elevation); **speed** maps the norm of joint velocities **3–6** into **\[6, 12\] m/s**. No forward kinematics or link lengths.

For a paper: state clearly that learning uses this **abstract throwing plant**; SPEC-style flight integration still grounds landing geometry. Older checkpoints (e.g. different velocity maps) are **not comparable**—retrain for reported metrics.

## Setup

From this folder:

```bash
cd rl_dart_6dof
pip install torch numpy scipy scikit-learn matplotlib
```

(Use the [official PyTorch install](https://pytorch.org/get-started/locally/) if you need CUDA.)

## Train

```bash
python train.py
```

**Checkpoints** (relative to `rl_dart_6dof/`):

- `checkpoints/policy_best.pt` — best mean eval reward seen so far
- `checkpoints/policy_final.pt` — last weights after training

**Overrides** (optional):

```bash
python train.py --total-env-steps 10000 --seed 1
```

**Logs** (under `logs/` by default):

- `train_metrics.csv` — each **eval** checkpoint: mean/std reward, radial miss, release rate, mean score.
- `train_episodes.csv` — every **finished training episode**: return, length, whether release fired (with exploration noise; not the same as eval).

While training you get:

- `[progress]` every **25** episodes (rolling mean return / release rate; change with `--progress-every N`, or `0` to disable).
- `[1k-window]` every **1000** env steps: release rate over episodes that **ended** in that window.

Flags: `--log-dir`, `--no-train-log`, `--no-episode-log`, `--progress-every`, `--append-logs`.

By default, `train.py` **truncates** `train_metrics.csv` and `train_episodes.csv` at the start of each run. Use **`--append-logs`** to keep appending to existing files.

**Plot logs:**

```bash
python plot_training.py
python plot_training.py --log-dir logs --out logs/training_curves.png --smooth 50
```

Hyperparameters live in `config.py` (`state_dim=12`, `action_dim=7`, learning rates, `gamma`, buffer size, etc.).

Each episode samples a random optimal release timestep for the timing bonus so the policy cannot memorize a single release frame. `min_release_fraction` (default **0.5**) ensures at least half of each replay batch comes from transitions in episodes that ended with a release, limiting buffer contamination during collapse-prone phases.

**Stratified sampling vs PER (SAC):** When `min_release_fraction > 0`, index selection follows that quota instead of sum-tree priorities; importance-sampling weights are uniform for those batches while `update_priorities` still applies TD errors—acceptable tradeoff; document for writeups.

## Train with SAC (optional)

```bash
python trainSAC.py
python trainSAC.py --total-steps 100000
python trainSAC.py --total-env-steps 100000 --curriculum-taper-episodes 2000
```

Uses a **stochastic** tanh-Gaussian policy, **twin** Q-networks, **automatic** entropy temperature (`target_entropy = -action_dim`), and by default **prioritized replay** with **stratified batches** when `min_release_fraction > 0` (see `replay_buffer.py`). Defaults in `config.py`: **`total_env_steps=100000`**, **`eval_every_steps=0`** (auto: eval every **`total_env_steps // eval_num_checkpoints`** env steps, default **10** checkpoints per run), **`eval_episodes=10`**, **`curriculum_taper_episodes=2000`** (linear decay of forced-release probability over the first K completed-episode starts). Override cadence with **`--eval-every-steps`** or **`--eval-num-checkpoints`**. **`--total-steps`** is an alias for **`--total-env-steps`**. Disable PER with **`--no-prioritized-replay`**.

Checkpoints: `checkpoints/policy_bestSAC.pt`, `checkpoints/policy_finalSAC.pt`. Logs (default): `logs/train_metrics_SAC.csv`, `logs/train_episodes_SAC.csv`.

## Evaluate (effective DOF / PCA)

Evaluation matches **training’s eval environment**: **`curriculum_taper_episodes=0`** (no forced-release taper). **`min_release_steps`, `release_threshold`, `max_steps`, `release_completion_bonus`,** etc. come from the **checkpoint’s saved `cfg`** when loading a trained policy (aligned with `eval_env` in `train.py` / `trainSAC.py`).

### Paper-style eval workflow

```bash
python evaluate.py --checkpoint checkpoints/policy_bestSAC.pt --episodes 500
python evaluate.py --baseline random --episodes 500
python plot_training.py --log-dir logs --smooth 50 --curriculum-taper-episodes 2000
python tests_learning/verify_checklist.py
```

The evaluate script prints a **Paper summary** block (release rate, mean/std landing distance, mean/std score). Compare trained vs random baseline side-by-side from two runs.

```bash
python evaluate.py
python evaluate.py --checkpoint checkpoints/policy_bestSAC.pt --episodes 500 --seed 0 --print-cfg
python evaluate.py --checkpoint checkpoints/policy_bestSAC.pt --landing-heatmap --dual-pca-rollout-split --episodes 400
```

If `--checkpoint` is omitted, the script searches `checkpoints/` for `policy_best.pt`, `policy_bestSAC.pt`, `policy_final.pt`, `policy_finalSAC.pt` in that order. Supports **DDPG** (`PolicyNet`) and **SAC** (`GaussianPolicy`); deterministic evaluation (`mean_action` vs forward pass).

**Paper-style robustness:** use **`--episodes 200`**–**`500`**; repeat with **`--seed 1`**, **`--seed 2`** and compare PCA (e.g. PC1 explained variance) and landing stats. A stderr reminder appears if `--episodes` &lt; 100.

**Random baseline** (same raw-action mapping as training warm-up):

```bash
python evaluate.py --baseline random --episodes 500 --seed 0
```

Compare metrics and plots to a trained run with the **same** `--episodes` and `--seed`.

**Useful flags:** `--print-cfg`; **`--landing-heatmap`** (`landing_hexbin.png`); **`--dual-pca-rollout-split`** (PCA on first vs second half of eval episodes — detects rollout drift, **not** training curriculum phases); **`--viz-episode`** (`throw_xz.png`). For multiple throw figures, rerun with different **`--seed`** values.

Default plot directory: **`eval_outputs/`** (`--output-dir`). Non-interactive matplotlib only (headless/WSL safe).

```bash
python evaluate.py --checkpoint checkpoints/policy_final.pt --output-dir eval_outputs --viz-episode
python evaluate.py --baseline random
```

## Project layout

| File               | Role |
|--------------------|------|
| `env.py`           | `DartEnv`: toy arm + SPEC flight to vertical board |
| `flight_physics.py`| Drag/gravity ODE + board crossing integrator |
| `viz_throw.py`     | Matplotlib x–z trajectory + board line |
| `models.py`        | Policy (MLP+tanh), GaussianPolicy (SAC), critic Q(s,a), target |
| `replay_buffer.py` | Uniform replay + optional **prioritized** buffer (SAC) |
| `train.py`         | DDPG-style loop + checkpointing + eval/episode CSV |
| `trainSAC.py`      | SAC loop + `policy_*SAC.pt` + `*_SAC.csv` logs |
| `plot_training.py` | Plot training curves (prefers `train_*_SAC.csv` when present) |
| `evaluate.py`      | PCA, DOF, plots, optional throw viz; **DDPG + SAC** checkpoints |
| `utils.py`         | Seeds, plots, trajectory logging |
| `config.py`        | Hyperparameters |
# dartsprojecttemp
