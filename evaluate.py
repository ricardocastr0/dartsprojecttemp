"""
Evaluate learned policy + effective degrees of freedom (DOF).

1. Roll episodes: log joint angles and actions over time.
2. PCA (sklearn) on stacked [angles, velocities] along the trajectory timeline.
3. Effective DOF at 95%% cumulative variance.
4. Per-joint torque variance from action trajectories.
5. Optional random baseline; landing scatter, PCA bars, release/score histograms.

Saves plots under eval_outputs/ by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, replace

import numpy as np
import torch
from sklearn.decomposition import PCA

# Ensure sibling imports when run as script from any cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Before utils (matplotlib); avoids Qt/xcb crash on headless or WSL when saving plots / plt.show().
import matplotlib

matplotlib.use("Agg")

from config import CFG, Config
from env import DartEnv
from flight_physics import board_ring_radii_m_for_plot
from models import GaussianPolicy, PolicyNet
from train import raw_policy_to_env_action
from utils import (
    plot_histogram,
    plot_landing_hexbin,
    plot_landing_scatter,
    plot_pca_explained_bars,
    plot_pca_variance,
    plot_torque_variance,
    to_tensor,
    TrajectoryLogger,
)


def _eval_raw_actions(actor: PolicyNet | GaussianPolicy, st: torch.Tensor) -> torch.Tensor:
    """Deterministic raw actions in [-1,1]^action_dim (matches train eval)."""
    if isinstance(actor, GaussianPolicy):
        return actor.mean_action(st)
    return actor(st)


def load_actor(path: str, device: str) -> tuple[PolicyNet | GaussianPolicy, Config]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    raw_cfg = ckpt.get("cfg", {})
    if isinstance(raw_cfg, dict) and raw_cfg:
        names = {f.name for f in fields(Config)}
        overrides = {k: v for k, v in raw_cfg.items() if k in names}
        cfg = replace(CFG, **overrides)
    else:
        cfg = CFG
    # DDPG (train.py): key "actor", PolicyNet with output dim = action_dim.
    # SAC (trainSAC.py): key "policy", GaussianPolicy with output dim = 2 * action_dim (mean + log_std).
    policy_sd = ckpt.get("actor") or ckpt.get("policy")
    if policy_sd is None:
        raise KeyError(
            "Checkpoint has neither 'actor' nor 'policy' state_dict "
            "(expected train.py or trainSAC.py checkpoint)."
        )
    w = policy_sd.get("net.4.weight")
    if w is None:
        raise KeyError("Checkpoint state_dict missing net.4.weight (unexpected layout).")
    out_features = int(w.shape[0])
    sac_tag = ckpt.get("algorithm") == "SAC"
    if sac_tag or out_features == 2 * cfg.action_dim:
        if out_features != 2 * cfg.action_dim:
            raise RuntimeError(
                f"SAC/GaussianPolicy expected last layer out={2 * cfg.action_dim}, got {out_features}"
            )
        actor = GaussianPolicy(cfg.state_dim, cfg.action_dim).to(device)
    elif out_features == cfg.action_dim:
        actor = PolicyNet(cfg.state_dim, cfg.action_dim).to(device)
    else:
        raise RuntimeError(
            f"Cannot load policy: last linear out_features={out_features}, "
            f"cfg.action_dim={cfg.action_dim} (expected {cfg.action_dim} for DDPG or "
            f"{2 * cfg.action_dim} for SAC)."
        )
    actor.load_state_dict(policy_sd)
    actor.eval()
    return actor, cfg


def collect_trajectories(
    env: DartEnv,
    cfg: Config,
    device: str,
    n_episodes: int,
    *,
    actor: PolicyNet | GaussianPolicy | None,
    rng: np.random.Generator | None,
) -> tuple[list[dict], dict]:
    """Roll out episodes. If actor is None, rng must be set (uniform raw actions like train warm-up)."""
    if actor is None:
        if rng is None:
            raise ValueError("Random baseline requires rng")

    runs: list[dict] = []
    landing_xs: list[float] = []
    landing_ys: list[float] = []
    release_flags: list[float] = []
    landing_dists: list[float] = []
    scores: list[float] = []
    release_steps: list[int] = []

    with torch.no_grad():
        for _ in range(n_episodes):
            log = TrajectoryLogger()
            s = env.reset()
            done = False
            last_info = None
            last_step_out: dict | None = None
            release_first_step: int | None = None

            while not done:
                if actor is not None:
                    st = to_tensor(s, device=device).unsqueeze(0)
                    raw = _eval_raw_actions(actor, st).squeeze(0).cpu().numpy()
                else:
                    assert rng is not None
                    raw = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
                a = raw_policy_to_env_action(raw)
                s2, r, done, step_out = env.step(a)
                t = int(step_out["t"])
                if release_first_step is None and t >= env.min_release_steps and float(a[6]) >= cfg.release_threshold:
                    release_first_step = t
                log.add(s, a, r)
                s = s2
                last_info = step_out["info"]
                last_step_out = step_out

            runs.append(log.as_arrays())
            if last_info is not None:
                release_flags.append(1.0 if last_info.released else 0.0)
                if last_info.landing_xy is not None:
                    dy, dz = last_info.landing_xy
                    landing_xs.append(float(dy))
                    landing_ys.append(float(dz))
                    d = float(np.hypot(dy - env.target_x, dz - env.target_y))
                    landing_dists.append(d)
                scores.append(float(last_info.score))
                if last_info.released:
                    assert last_step_out is not None
                    rs = release_first_step if release_first_step is not None else int(last_step_out["t"])
                    release_steps.append(rs)
                else:
                    release_steps.append(int(env.max_steps))

    meta: dict = {
        "_meta_release_rate": float(np.mean(release_flags)) if release_flags else float("nan"),
        "_meta_mean_dist": float(np.mean(landing_dists)) if landing_dists else float("nan"),
        "_meta_std_dist": float(np.std(landing_dists)) if landing_dists else float("nan"),
        "_meta_landing_x": np.asarray(landing_xs, dtype=np.float32),
        "_meta_landing_y": np.asarray(landing_ys, dtype=np.float32),
        "_meta_scores": np.asarray(scores, dtype=np.float32),
        "_meta_release_steps": np.asarray(release_steps, dtype=np.int32),
    }
    return runs, meta


def make_eval_env(cfg: Config, seed: int) -> DartEnv:
    """Matches train.py / trainSAC.py eval_env: no curriculum, checkpoint-aligned thresholds."""
    return DartEnv(
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


def print_eval_config(cfg: Config, env: DartEnv, *, checkpoint_label: str) -> None:
    print("\n=== Resolved evaluation config (matches training eval_env) ===")
    print(f"  label: {checkpoint_label}")
    print(f"  dt={cfg.dt}  max_steps={cfg.max_steps}  release_threshold={cfg.release_threshold}")
    print(
        f"  k_tau={cfg.k_tau}  damping={cfg.damping}  max_vel={cfg.max_vel}  "
        f"reset_angles=[{cfg.reset_angle_low},{cfg.reset_angle_high}]  "
        f"reset_vels=[{cfg.reset_vel_low},{cfg.reset_vel_high}]"
    )
    print(f"  release_completion_bonus={cfg.release_completion_bonus}  min_release_steps={cfg.min_release_steps}")
    print(f"  curriculum_taper_episodes (eval): 0  (training eval uses no forced-release taper)")
    print(
        f"  DartEnv.timeout_penalty={env.timeout_penalty} (hardcoded in env, not in checkpoint Config)"
    )


def stack_states_from_runs(runs: list[dict]) -> np.ndarray:
    state_rows: list[np.ndarray] = []
    for d in runs:
        ang = d["angles"]
        vel = d["velocities"]
        state_rows.append(np.concatenate([ang, vel], axis=1))
    return np.vstack(state_rows).astype(np.float64) if state_rows else np.zeros((0, 12), dtype=np.float64)


def run_pca_states(X: np.ndarray) -> tuple[np.ndarray, int, PCA]:
    n_comp = min(12, max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    explained = pca.explained_variance_ratio_
    dof_95 = effective_dof_95(explained)
    return explained, dof_95, pca


def effective_dof_95(explained_ratio: np.ndarray) -> int:
    c = np.cumsum(explained_ratio)
    for i, val in enumerate(c):
        if val >= 0.95:
            return i + 1
    return len(explained_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="DOF analysis (PCA + torque variance + baselines)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy_*.pt (DDPG: actor; SAC: policy). "
        "Defaults search checkpoints/policy_best.pt, policy_bestSAC.pt, policy_final.pt, policy_finalSAC.pt",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=("random",),
        default=None,
        help="If set, ignore checkpoint and evaluate uniform-random raw actions (same mapping as train warm-up).",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument(
        "--viz-episode",
        action="store_true",
        help="After rollouts, save throw_xz.png from one successful release trajectory",
    )
    parser.add_argument(
        "--print-cfg",
        action="store_true",
        help="Print resolved DartEnv / Config used for rollouts (paper reproducibility)",
    )
    parser.add_argument(
        "--landing-heatmap",
        action="store_true",
        help="Save landing_hexbin.png (density) in addition to landing_scatter.png",
    )
    parser.add_argument(
        "--dual-pca-rollout-split",
        action="store_true",
        help="Run separate PCA on first vs second half of evaluation episodes (same env); saves extra PNGs",
    )
    args = parser.parse_args()

    if args.episodes < 100:
        print(
            f"Note: --episodes={args.episodes} is small for stable PCA/landing stats; "
            "consider 200–500 for paper runs.",
            file=sys.stderr,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = _SCRIPT_DIR
    actor: PolicyNet | GaussianPolicy | None = None
    cfg: Config
    rng: np.random.Generator | None = None
    label = ""

    if args.baseline == "random":
        cfg = CFG
        rng = np.random.default_rng(args.seed)
        label = "baseline=random"
    else:
        cand = args.checkpoint
        if cand is None:
            for name in (
                "policy_best.pt",
                "policy_bestSAC.pt",
                "policy_final.pt",
                "policy_finalSAC.pt",
            ):
                p = os.path.join(base, "checkpoints", name)
                if os.path.isfile(p):
                    cand = p
                    break
        if cand is None or not os.path.isfile(cand):
            print("No checkpoint found. Train first: python train.py  (or use --baseline random)")
            sys.exit(1)
        actor, cfg = load_actor(cand, device)
        label = f"checkpoint={cand}"

    env = make_eval_env(cfg, args.seed)
    if args.print_cfg:
        print_eval_config(cfg, env, checkpoint_label=label)

    runs, meta = collect_trajectories(
        env,
        cfg,
        device,
        args.episodes,
        actor=actor,
        rng=rng,
    )

    out_dir = os.path.join(base, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.viz_episode:
        from viz_throw import save_throw_figure

        venv = make_eval_env(cfg, args.seed + 999)
        s = venv.reset()
        done = False
        saved = False
        viz_rng = np.random.default_rng(args.seed + 1000) if rng is None else rng
        with torch.no_grad():
            while not done:
                if actor is not None:
                    st = to_tensor(s, device=device).unsqueeze(0)
                    raw = _eval_raw_actions(actor, st).squeeze(0).cpu().numpy()
                else:
                    raw = viz_rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
                a = raw_policy_to_env_action(raw)
                s, _, done, step_out = venv.step(a)
                if done and step_out["info"].released:
                    s6 = venv.get_release_state6()
                    save_throw_figure(
                        s6,
                        os.path.join(out_dir, "throw_xz.png"),
                        wind_xyz=venv.wind_xyz,
                        drag_enabled=venv.drag_enabled,
                    )
                    saved = True
        if not saved:
            print("viz-episode: no successful release in one rollout; skipped throw_xz.png")
        else:
            print(f"Saved throw visualization: {os.path.join(out_dir, 'throw_xz.png')}")

    # Concatenate all timesteps: state features for PCA (angles + velocities)
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for d in runs:
        ang = d["angles"]
        vel = d["velocities"]
        state_rows.append(np.concatenate([ang, vel], axis=1))
        action_rows.append(d["actions"])

    X = np.vstack(state_rows).astype(np.float64)
    A = np.vstack(action_rows).astype(np.float64)
    torque = A[:, :6]

    explained, dof_95, pca = run_pca_states(X)
    cumulative = np.cumsum(explained)

    print(f"Eval: {label}")
    print(f"Trajectory samples (rows): {X.shape[0]}")
    print(f"PCA components used: {pca.n_components_}")
    print("Per-component explained variance ratio:")
    for i, r_i in enumerate(explained[:12], start=1):
        print(f"  PC{i}: {r_i:.4f} (cum: {cumulative[i-1]:.4f})")
    print(f"Effective DOF (95% variance): {dof_95}")
    print(f"Release rate: {meta['_meta_release_rate']:.3f}")
    print(
        f"Mean radial miss on board (m, bull-centered Δy,Δz): "
        f"{meta['_meta_mean_dist']:.4f} (std {meta['_meta_std_dist']:.4f})"
    )
    scores_arr = meta["_meta_scores"]
    if scores_arr.size > 0:
        print(
            f"Dartboard score: mean={float(np.mean(scores_arr)):.4f} "
            f"(std {float(np.std(scores_arr)):.4f})"
        )
    else:
        print("Dartboard score: n/a (no landing rows)")
    print("\n=== Paper summary (this run) ===")
    print(f"  release_rate:          {meta['_meta_release_rate']:.4f}")
    print(
        f"  mean_landing_dist_m:   {meta['_meta_mean_dist']:.4f}  "
        f"std_landing_dist_m: {meta['_meta_std_dist']:.4f}"
    )
    if scores_arr.size > 0:
        print(
            f"  mean_score:            {float(np.mean(scores_arr)):.4f}  "
            f"std_score:            {float(np.std(scores_arr)):.4f}"
        )
    print(
        f"Release step histogram: successful releases use first step with release>threshold; "
        f"timeouts counted as {env.max_steps}."
    )

    torque_var = np.var(torque, axis=0)
    print("Per-joint torque variance:")
    for j in range(6):
        print(f"  joint {j+1}: {torque_var[j]:.6f}")

    pca_path = os.path.join(out_dir, "pca_variance.png")
    pca_bars_path = os.path.join(out_dir, "pca_explained_bars.png")
    torque_path = os.path.join(out_dir, "torque_variance.png")
    landing_path = os.path.join(out_dir, "landing_scatter.png")
    landing_heatmap_path = os.path.join(out_dir, "landing_hexbin.png")
    rel_hist_path = os.path.join(out_dir, "release_step_hist.png")
    score_hist_path = os.path.join(out_dir, "score_hist.png")

    plot_pca_variance(np.asarray(explained, dtype=np.float32), pca_path)
    plot_pca_explained_bars(np.asarray(explained, dtype=np.float32), pca_bars_path)
    plot_torque_variance(np.asarray(torque_var, dtype=np.float32), torque_path)

    extra_paths: list[str] = []

    if args.dual_pca_rollout_split:
        n_ep = len(runs)
        if n_ep < 4:
            print(
                "dual-pca-rollout-split: skipped (need at least 4 episodes)",
                file=sys.stderr,
            )
        else:
            mid = n_ep // 2
            r_first, r_second = runs[:mid], runs[mid:]
            X1 = stack_states_from_runs(r_first)
            X2 = stack_states_from_runs(r_second)
            if X1.shape[0] < 2 or X2.shape[0] < 2:
                print(
                    "dual-pca-rollout-split: skipped (not enough timesteps in a half)",
                    file=sys.stderr,
                )
            else:
                ex1, dof1, _ = run_pca_states(X1)
                ex2, dof2, _ = run_pca_states(X2)
                print("\n=== Dual PCA (first vs second half of eval episodes, same env) ===")
                print(
                    f"  first half: episodes 1..{mid}, timesteps={X1.shape[0]}, "
                    f"PC1_explained={float(ex1[0]):.4f}, eff_dof_95={dof1}"
                )
                print(
                    f"  second half: episodes {mid + 1}..{n_ep}, timesteps={X2.shape[0]}, "
                    f"PC1_explained={float(ex2[0]):.4f}, eff_dof_95={dof2}"
                )
                p1 = os.path.join(out_dir, "pca_variance_rollout_first.png")
                p2 = os.path.join(out_dir, "pca_variance_rollout_second.png")
                b1 = os.path.join(out_dir, "pca_explained_bars_rollout_first.png")
                b2 = os.path.join(out_dir, "pca_explained_bars_rollout_second.png")
                plot_pca_variance(np.asarray(ex1, dtype=np.float32), p1)
                plot_pca_explained_bars(np.asarray(ex1, dtype=np.float32), b1)
                plot_pca_variance(np.asarray(ex2, dtype=np.float32), p2)
                plot_pca_explained_bars(np.asarray(ex2, dtype=np.float32), b2)
                extra_paths.extend([p1, p2, b1, b2])

    lx, ly = meta["_meta_landing_x"], meta["_meta_landing_y"]
    if lx.size > 0:
        plot_landing_scatter(
            lx,
            ly,
            float(env.target_x),
            float(env.target_y),
            board_ring_radii_m_for_plot(),
            landing_path,
            xlabel="Δy (m) left",
            ylabel="Δz (m) relative to bull",
            title="Landings on board face (bull at origin)",
        )
        if args.landing_heatmap:
            plot_landing_hexbin(
                lx,
                ly,
                float(env.target_x),
                float(env.target_y),
                board_ring_radii_m_for_plot(),
                landing_heatmap_path,
                xlabel="Δy (m) left",
                ylabel="Δz (m) relative to bull",
                title="Landing density (eval rollouts)",
            )
            extra_paths.append(landing_heatmap_path)
    plot_histogram(
        meta["_meta_release_steps"].astype(np.float64),
        "First release step (timeouts at max_steps)",
        "Step index",
        rel_hist_path,
    )
    plot_histogram(
        meta["_meta_scores"].astype(np.float64),
        "Terminal dartboard score",
        "Score",
        score_hist_path,
    )

    print("Saved plots:")
    for p in (
        pca_path,
        pca_bars_path,
        torque_path,
        landing_path,
        rel_hist_path,
        score_hist_path,
        *extra_paths,
    ):
        print(f"  {p}")


if __name__ == "__main__":
    main()
