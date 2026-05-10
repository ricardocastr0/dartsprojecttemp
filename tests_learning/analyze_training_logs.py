"""
Analyze training CSV logs: curriculum vs post-curriculum segments, rolling returns,
eval metric trends, optional multi-panel figures.

No Torch. Run from repository root:

  python3 tests_learning/analyze_training_logs.py --log-dir logs --algorithm sac
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# bootstrap
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from config import CFG, Config


def _read_local_default_rolling() -> int:
    """Load DEFAULT_ROLLING_WINDOW from sibling analysis_params.py if present."""
    import importlib.util

    p = Path(__file__).resolve().parent / "analysis_params.py"
    if not p.is_file():
        return 50
    spec = importlib.util.spec_from_file_location("analysis_params_local", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return int(getattr(mod, "DEFAULT_ROLLING_WINDOW", 50))


_DEFAULT_ROLLING = _read_local_default_rolling()


def rolling_mean(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean; returns (end_indices, smoothed) aligned to last index of each window."""
    if x.size == 0 or window <= 1:
        return np.arange(x.size, dtype=np.int64), x.astype(np.float64)
    w = min(int(window), x.size)
    if w < 2:
        return np.arange(x.size, dtype=np.int64), x.astype(np.float64)
    cum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    sm = (cum[w:] - cum[:-w]) / float(w)
    idx = np.arange(w - 1, w - 1 + sm.size, dtype=np.int64)
    return idx, sm


def read_episodes_csv(path: str) -> dict[str, np.ndarray] | None:
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    ep_idx: list[int] = []
    env_step: list[int] = []
    ep_ret: list[float] = []
    ep_len: list[int] = []
    released: list[float] = []
    for row in rows:
        try:
            ep_idx.append(int(row["episode_index"]))
            env_step.append(int(row["env_step"]))
            ep_ret.append(float(row["episode_return"]))
            ep_len.append(int(row.get("episode_length", 0) or 0))
            released.append(float(row.get("released", 0) or 0))
        except (KeyError, ValueError):
            continue
    if not ep_idx:
        return None
    return {
        "episode_index": np.asarray(ep_idx, dtype=np.int64),
        "env_step": np.asarray(env_step, dtype=np.int64),
        "episode_return": np.asarray(ep_ret, dtype=np.float64),
        "episode_length": np.asarray(ep_len, dtype=np.int64),
        "released": np.asarray(released, dtype=np.float64),
    }


def read_metrics_csv(path: str) -> dict[str, np.ndarray] | None:
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in out:
            try:
                v = float(row[k]) if row.get(k) not in (None, "") else float("nan")
            except (TypeError, ValueError):
                v = float("nan")
            out[k].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def resolve_log_paths(log_dir: Path, algorithm: str) -> tuple[str | None, str | None, str]:
    """Returns (episodes_path, metrics_path, resolved_algo_label)."""
    sac_ep = log_dir / "train_episodes_SAC.csv"
    ddp_ep = log_dir / "train_episodes.csv"
    sac_m = log_dir / "train_metrics_SAC.csv"
    ddp_m = log_dir / "train_metrics.csv"

    def nonempty(p: Path) -> bool:
        return p.is_file() and p.stat().st_size > 0

    if algorithm == "sac":
        ep = str(sac_ep) if nonempty(sac_ep) else None
        mt = str(sac_m) if nonempty(sac_m) else None
        return ep, mt, "sac"
    if algorithm == "ddpg":
        ep = str(ddp_ep) if nonempty(ddp_ep) else None
        mt = str(ddp_m) if nonempty(ddp_m) else None
        return ep, mt, "ddpg"

    # auto
    sac_ok = nonempty(sac_ep)
    ddp_ok = nonempty(ddp_ep)
    if sac_ok and ddp_ok:
        ep, mt = str(sac_ep), str(sac_m) if nonempty(sac_m) else None
        return ep, mt, "sac"
    if sac_ok:
        ep, mt = str(sac_ep), str(sac_m) if nonempty(sac_m) else None
        return ep, mt, "sac"
    if ddp_ok:
        ep, mt = str(ddp_ep), str(ddp_m) if nonempty(ddp_m) else None
        return ep, mt, "ddpg"
    return None, None, "auto"


def segment_stats(
    ep: dict[str, np.ndarray],
    mask: np.ndarray,
    max_steps: int,
) -> dict[str, Any]:
    if not np.any(mask):
        return {"n": 0}
    ret = ep["episode_return"][mask]
    ln = ep["episode_length"][mask]
    rel = ep["released"][mask]
    timeout_flag = (rel < 0.5) | (ln >= max_steps)
    p25, p50, p75 = (float(x) for x in np.percentile(ret, [25, 50, 75]))
    return {
        "n": int(mask.sum()),
        "return_mean": float(np.mean(ret)),
        "return_std": float(np.std(ret)),
        "return_min": float(np.min(ret)),
        "return_max": float(np.max(ret)),
        "return_p25": p25,
        "return_p50": p50,
        "return_p75": p75,
        "length_mean": float(np.mean(ln)),
        "length_std": float(np.std(ln)),
        "release_rate": float(np.mean(rel >= 0.5)),
        "timeout_rate": float(np.mean(timeout_flag)),
        "timeout_count_released0": int(np.sum(rel < 0.5)),
        "timeout_count_horizon": int(np.sum(ln >= max_steps)),
    }


def post_curriculum_conditionals(ep: dict[str, np.ndarray], post_mask: np.ndarray, max_steps: int) -> dict[str, Any]:
    """Split post-curriculum rows into released vs timeout episodes (mutually exclusive timeout flag)."""
    if not np.any(post_mask):
        return {}
    rel = ep["released"][post_mask]
    ln = ep["episode_length"][post_mask]
    er = ep["episode_return"][post_mask]
    timeout = (rel < 0.5) | (ln >= max_steps)
    ok = ~timeout & (rel >= 0.5)
    to = timeout

    def pack(mask_inner: np.ndarray) -> dict[str, Any]:
        if not np.any(mask_inner):
            return {"n": 0}
        r = er[mask_inner]
        return {
            "n": int(mask_inner.sum()),
            "return_mean": float(np.mean(r)),
            "return_std": float(np.std(r)),
            "return_min": float(np.min(r)),
            "return_max": float(np.max(r)),
        }

    return {
        "released_finish": pack(ok),
        "timeout_or_horizon": pack(to),
    }


def _timeout_mask(ep: dict[str, np.ndarray], max_steps: int) -> np.ndarray:
    rel = ep["released"]
    ln = ep["episode_length"]
    return (rel < 0.5) | (ln >= int(max_steps))


def compute_pattern_analysis(
    ep: dict[str, np.ndarray],
    *,
    curriculum_taper_episodes: int,
    max_steps: int,
    bin_width: int = 100,
    tail_ns: tuple[int, ...] = (50, 100, 200, 500),
) -> dict[str, Any]:
    """Higher-level patterns: binned trends, tail collapse, streaks, correlations (post-curriculum)."""
    ei = ep["episode_index"]
    er = ep["episode_return"]
    ln = ep["episode_length"]
    tw = _timeout_mask(ep, max_steps)
    n = ei.size
    out: dict[str, Any] = {
        "n_rows": int(n),
        "episode_index_min": int(ei.min()) if n else None,
        "episode_index_max": int(ei.max()) if n else None,
        "overall_timeout_rate": float(np.mean(tw)) if n else None,
    }

    # Fixed-width bins on episode_index (inclusive ranges), full log
    bw = max(1, int(bin_width))
    bins: list[dict[str, Any]] = []
    if n:
        emin, emax = int(ei.min()), int(ei.max())
        lo = ((emin - 1) // bw) * bw + 1
        while lo <= emax:
            hi = lo + bw - 1
            m = (ei >= lo) & (ei <= hi)
            if np.any(m):
                bins.append(
                    {
                        "episode_lo": lo,
                        "episode_hi": hi,
                        "n": int(m.sum()),
                        "return_mean": float(np.mean(er[m])),
                        "timeout_rate": float(np.mean(tw[m])),
                        "release_rate": float(np.mean(ep["released"][m] >= 0.5)),
                    }
                )
            lo += bw
    out["bins_by_episode_index"] = {"bin_width": bw, "bins": bins}

    # Tail slices by CSV row order (training time order)
    tail_stats: list[dict[str, Any]] = []
    for tn in tail_ns:
        k = min(int(tn), n)
        if k <= 0:
            continue
        sl = slice(n - k, n)
        tail_stats.append(
            {
                "last_n_rows": k,
                "episode_index_range": [int(ei[sl][0]), int(ei[sl][-1])],
                "return_mean": float(np.mean(er[sl])),
                "timeout_rate": float(np.mean(tw[sl])),
                "release_rate": float(np.mean(ep["released"][sl] >= 0.5)),
            }
        )
    out["tail_row_slices"] = tail_stats

    # Longest timeout suffix run (from last row backward)
    suffix = 0
    if n:
        i = n - 1
        while i >= 0 and bool(tw[i]):
            suffix += 1
            i -= 1
    out["suffix_timeout_run_length"] = suffix
    out["last_row_timeout"] = bool(tw[-1]) if n else False

    # Compare previous block vs last 100 rows (collapse heuristic)
    collapse: dict[str, Any] = {}
    if n >= 200:
        a = slice(n - 200, n - 100)
        b = slice(n - 100, n)
        collapse = {
            "prev100_timeout_rate_rows_minus200_to_minus100": float(np.mean(tw[a])),
            "last100_timeout_rate": float(np.mean(tw[b])),
            "delta_timeout_rate_last100_minus_prev100": float(np.mean(tw[b]) - np.mean(tw[a])),
        }
    elif n >= 100:
        b = slice(n - 100, n)
        collapse = {
            "last100_timeout_rate": float(np.mean(tw[b])),
            "note": "insufficient rows for prev100 vs last100 comparison",
        }
    out["collapse_compare_prev100_vs_last100"] = collapse

    # Post-curriculum only: Pearson r(episode_index, return), r(length, return)
    post = ei > int(curriculum_taper_episodes)
    if np.sum(post) >= 3:
        ei_p = ei[post].astype(np.float64)
        er_p = er[post]
        ln_p = ln[post].astype(np.float64)
        if float(np.std(ei_p)) > 1e-12 and float(np.std(er_p)) > 1e-12:
            r_ie = float(np.corrcoef(ei_p, er_p)[0, 1])
        else:
            r_ie = float("nan")
        if float(np.std(ln_p)) > 1e-12 and float(np.std(er_p)) > 1e-12:
            r_le = float(np.corrcoef(ln_p, er_p)[0, 1])
        else:
            r_le = float("nan")
        out["post_curriculum_correlations"] = {
            "n": int(post.sum()),
            "pearson_episode_index_vs_return": r_ie,
            "pearson_episode_length_vs_return": r_le,
        }
    else:
        out["post_curriculum_correlations"] = {"n": int(post.sum()), "note": "too few post episodes"}

    # Early post vs late post (by episode index, not row count)
    ce = int(curriculum_taper_episodes)
    early = post & (ei <= ce + 100)
    if n:
        em = int(ei.max())
        late = post & (ei >= em - 99) & (ei <= em)
    else:
        late = np.zeros_like(post, dtype=bool)
    early_stats = segment_stats(ep, early, max_steps) if np.any(early) else {"n": 0}
    late_stats = segment_stats(ep, late, max_steps) if np.any(late) else {"n": 0}
    out["post_early_vs_late_episode_index"] = {
        "early_post_episodes_501_to_600_approx": early_stats,
        "late_post_last_100_episode_indices": late_stats,
    }

    return out


def analyze(
    cfg: Config,
    *,
    episodes_path: str,
    metrics_path: str | None,
    curriculum_taper_episodes: int,
    rolling_window: int,
    max_steps_effective: int,
    include_patterns: bool = False,
    pattern_bin_width: int = 100,
) -> dict[str, Any]:
    ep = read_episodes_csv(episodes_path)
    if ep is None:
        raise ValueError(f"No episode rows in {episodes_path}")

    ei = ep["episode_index"]
    curr_mask = ei <= curriculum_taper_episodes
    post_mask = ei > curriculum_taper_episodes

    max_steps = int(max_steps_effective)
    out: dict[str, Any] = {
        "config_ref": {
            "curriculum_taper_episodes": curriculum_taper_episodes,
            "max_steps_effective": max_steps,
            "max_steps_project_cfg": int(cfg.max_steps),
        },
        "episode_columns_ok": True,
        "segments": {
            "curriculum": segment_stats(ep, curr_mask, max_steps),
            "post_curriculum": segment_stats(ep, post_mask, max_steps),
            "all": segment_stats(ep, np.ones_like(curr_mask, dtype=bool), max_steps),
        },
    }

    out["segments_detail"] = {
        "post_curriculum_conditionals": post_curriculum_conditionals(ep, post_mask, max_steps),
    }

    er = ep["episode_return"]
    idx_roll, sm_roll = rolling_mean(er, rolling_window)
    last_i = int(idx_roll[-1]) if idx_roll.size else None
    first_i = int(idx_roll[0]) if idx_roll.size else None
    tail_timeout_frac = None
    if last_i is not None and rolling_window > 0:
        lo = max(0, last_i - rolling_window + 1)
        rel_w = ep["released"][lo : last_i + 1]
        ln_w = ep["episode_length"][lo : last_i + 1]
        tw = (rel_w < 0.5) | (ln_w >= max_steps)
        tail_timeout_frac = float(np.mean(tw)) if tw.size else None

    out["rolling_return"] = {
        "window": rolling_window,
        "note": "Rolling mean over CSV row order (chronological episodes); indices index episode_return rows.",
        "end_row_indices_sample": [int(i) for i in idx_roll[: min(5, len(idx_roll))]],
        "episode_index_at_last_window_end": int(ei[last_i]) if last_i is not None else None,
        "episode_index_at_first_window_end": int(ei[first_i]) if first_i is not None else None,
        "env_step_at_last_window_end": int(ep["env_step"][last_i]) if last_i is not None else None,
        "tail_window_timeout_fraction": tail_timeout_frac,
        "last_smoothed_mean": float(sm_roll[-1]) if sm_roll.size else None,
        "first_smoothed_mean": float(sm_roll[0]) if sm_roll.size else None,
    }

    # Rolling within post-curriculum only (re-index subset)
    if np.any(post_mask):
        post_ret = er[post_mask]
        w_post = min(rolling_window, max(2, post_ret.size))
        _pi, sm_post = rolling_mean(post_ret, w_post)
        out["rolling_return_post_curriculum_only"] = {
            "window_used": w_post,
            "last_smoothed_mean": float(sm_post[-1]) if sm_post.size else None,
            "first_smoothed_mean": float(sm_post[0]) if sm_post.size else None,
            "n_episodes": int(post_ret.size),
        }

    out["eval_metrics"] = {}
    if metrics_path and os.path.isfile(metrics_path):
        m = read_metrics_csv(metrics_path)
        if m and "step" in m and m["step"].size > 0:
            row_first = {k: float(m[k][0]) if k in m else float("nan") for k in m}
            row_last = {k: float(m[k][-1]) if k in m else float("nan") for k in m}
            delta_ld = None
            if "mean_landing_dist" in m and m["mean_landing_dist"].size >= 2:
                delta_ld = float(m["mean_landing_dist"][-1] - m["mean_landing_dist"][0])
            out["eval_metrics"] = {
                "n_checkpoints": int(m["step"].size),
                "first": row_first,
                "last": row_last,
                "delta_mean_landing_dist": delta_ld,
            }
            if int(m["step"].size) == 1 and "mean_landing_dist" in m:
                out["eval_metrics"]["single_checkpoint_snapshot"] = {
                    "step": float(m["step"][0]),
                    "mean_landing_dist": float(m["mean_landing_dist"][0]),
                    "mean_reward": float(m["mean_reward"][0]) if "mean_reward" in m else None,
                    "release_rate": float(m["release_rate"][0]) if "release_rate" in m else None,
                    "mean_score": float(m["mean_score"][0]) if "mean_score" in m else None,
                }

    if include_patterns:
        out["pattern_analysis"] = compute_pattern_analysis(
            ep,
            curriculum_taper_episodes=curriculum_taper_episodes,
            max_steps=max_steps,
            bin_width=pattern_bin_width,
        )

    return out


def plot_metrics_eval_overlay(metrics: dict[str, np.ndarray], out_path: Path) -> None:
    """Multi-series overlay: periodic eval metrics vs env step (paper/training-curve alignment)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "step" not in metrics:
        return
    st = metrics["step"]
    if st.size < 2:
        return

    series: list[tuple[str, str]] = []
    if "mean_reward" in metrics and metrics["mean_reward"].size == st.size:
        series.append(("mean_reward", "mean_reward"))
    if "mean_landing_dist" in metrics and metrics["mean_landing_dist"].size == st.size:
        series.append(("mean_landing_dist", "mean_landing_dist (m)"))
    if "release_rate" in metrics and metrics["release_rate"].size == st.size:
        series.append(("release_rate", "release_rate"))
    if not series:
        return

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(4.0, 2.4 * n)), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (key, ylab) in zip(axes_arr, series):
        y = metrics[key]
        ax.plot(st, y, "o-", ms=4, color="C0")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes_arr[-1].set_xlabel("env step (eval checkpoint)")
    fig.suptitle("Training eval checkpoints (train_metrics CSV)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_analysis(
    ep: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray] | None,
    *,
    curriculum_taper_episodes: int,
    rolling_window: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax00, ax01, ax10, ax11 = axes.ravel()

    es = ep["env_step"]
    er = ep["episode_return"]
    ei = ep["episode_index"]
    el = ep["episode_length"]
    rel = ep["released"]
    curr_mask = ei <= curriculum_taper_episodes
    post_mask = ei > curriculum_taper_episodes

    ax00.scatter(es, er, s=5, alpha=0.25, c="C0", label="episode_return")
    idx_r, sm_r = rolling_mean(er, rolling_window)
    if sm_r.size > 0:
        ax00.plot(es[idx_r], sm_r, "-", lw=2, color="C3", label=f"rolling mean (w={rolling_window})")
    post_i = np.flatnonzero(ei > curriculum_taper_episodes)
    if post_i.size > 0:
        ax00.axvline(float(es[post_i[0]]), color="gray", ls="--", alpha=0.6, label="post-curriculum start")
    ax00.set_xlabel("env_step")
    ax00.set_ylabel("episode_return")
    ax00.set_title("Training episode returns")
    ax00.grid(True, alpha=0.3)
    ax00.legend(loc="lower right", fontsize=8)

    ax01.scatter(ei, el, s=5, alpha=0.3, c="C2")
    ax01.axvline(x=curriculum_taper_episodes + 0.5, color="gray", ls="--", alpha=0.7, label="curriculum end")
    ax01.set_xlabel("episode_index")
    ax01.set_ylabel("episode_length")
    ax01.set_title("Episode length vs index")
    ax01.grid(True, alpha=0.3)
    ax01.legend(fontsize=8)

    if np.any(curr_mask) and np.any(post_mask):
        ax10.hist(er[curr_mask], bins=35, alpha=0.55, color="C0", label="curriculum", density=False)
        ax10.hist(er[post_mask], bins=35, alpha=0.55, color="C1", label="post-curriculum", density=False)
        ax10.legend(fontsize=8)
    else:
        ax10.hist(er, bins=40, color="C4", alpha=0.75)
    ax10.set_xlabel("episode_return")
    ax10.set_ylabel("count")
    ax10.set_title("Return distribution by phase")
    ax10.grid(True, alpha=0.3)

    if metrics is not None and "step" in metrics and "mean_landing_dist" in metrics:
        st = metrics["step"]
        ld = metrics["mean_landing_dist"]
        if np.isfinite(st).all() and np.isfinite(ld).all() and st.size >= 2:
            ax11.plot(st, ld, "o-", ms=4, color="C2")
            ax11.set_xlabel("env step (eval)")
            ax11.set_ylabel("mean_landing_dist (m)")
            ax11.set_title("Eval: mean landing distance")
            ax11.grid(True, alpha=0.3)
        elif st.size == 1:
            ax11.bar([0], [float(ld[0])], color="C2", width=0.5)
            ax11.set_xticks([0])
            ax11.set_xticklabels([f"step {int(st[0])}"])
            ax11.set_ylabel("mean_landing_dist (m)")
            ax11.set_title("Eval: single checkpoint")
            ax11.grid(True, alpha=0.3)
        else:
            ax11.scatter(ei, rel, s=8, alpha=0.35, c="C3")
            ax11.set_xlabel("episode_index")
            ax11.set_ylabel("released (0/1)")
            ax11.set_title("Released flag vs episode (fallback)")
            ax11.grid(True, alpha=0.3)
    else:
        ax11.scatter(ei, rel, s=8, alpha=0.35, c="C3")
        ax11.set_xlabel("episode_index")
        ax11.set_ylabel("released (0/1)")
        ax11.set_title("Released vs episode (no usable eval CSV)")
        ax11.grid(True, alpha=0.3)

    fig.suptitle("Training log analysis", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_pattern_analysis(pa: dict[str, Any]) -> None:
    print("\n=== Pattern scan (bins, tail, correlations) ===")
    print(
        f"n_rows={pa.get('n_rows')} | episode_index [{pa.get('episode_index_min')}, "
        f"{pa.get('episode_index_max')}] | overall_timeout_rate={pa.get('overall_timeout_rate')}"
    )
    print(f"suffix_timeout_run_from_eof={pa.get('suffix_timeout_run_length')} "
          f"(last_row_timeout={pa.get('last_row_timeout')})")

    cc = pa.get("collapse_compare_prev100_vs_last100") or {}
    if "delta_timeout_rate_last100_minus_prev100" in cc:
        print(
            "collapse prev100 vs last100 rows: "
            f"prev100_timeout={cc['prev100_timeout_rate_rows_minus200_to_minus100']:.3f} "
            f"last100_timeout={cc['last100_timeout_rate']:.3f} "
            f"delta={cc['delta_timeout_rate_last100_minus_prev100']:+.3f}"
        )
    elif cc:
        for k, v in cc.items():
            print(f"  {k}: {v}")

    print("tail_row_slices (chronological last N rows):")
    for t in pa.get("tail_row_slices") or []:
        er = t.get("episode_index_range") or [None, None]
        print(
            f"  last {t.get('last_n_rows'):4d} rows | ep_idx {er[0]}..{er[1]} | "
            f"return_mean={t.get('return_mean'):.2f} timeout_rate={t.get('timeout_rate'):.3f}"
        )

    bb = pa.get("bins_by_episode_index") or {}
    bins = bb.get("bins") or []
    if bins:
        print(f"bins_by_episode_index (width={bb.get('bin_width')}): showing first 3 + last 6 bins")
        head = bins[:3]
        tail = bins[-6:] if len(bins) > 9 else []
        for b in head:
            print(
                f"  ep {b['episode_lo']}-{b['episode_hi']}: n={b['n']} "
                f"ret_mean={b['return_mean']:.2f} timeout={b['timeout_rate']:.3f}"
            )
        if tail and len(bins) > 9:
            print("  ...")
        for b in tail:
            print(
                f"  ep {b['episode_lo']}-{b['episode_hi']}: n={b['n']} "
                f"ret_mean={b['return_mean']:.2f} timeout={b['timeout_rate']:.3f}"
            )

    pc = pa.get("post_curriculum_correlations") or {}
    if pc.get("pearson_episode_index_vs_return") is not None:
        print(
            "post-curriculum Pearson: episode_index vs return="
            f"{pc.get('pearson_episode_index_vs_return')} | "
            f"episode_length vs return={pc.get('pearson_episode_length_vs_return')} "
            f"(n={pc.get('n')})"
        )
    elif pc.get("note"):
        print(f"post-curriculum correlations: {pc.get('note')}")

    ev = pa.get("post_early_vs_late_episode_index") or {}
    early = ev.get("early_post_episodes_501_to_600_approx") or {}
    late = ev.get("late_post_last_100_episode_indices") or {}
    if early.get("n") or late.get("n"):
        em = early.get("return_mean")
        etm = early.get("timeout_rate")
        lm = late.get("return_mean")
        ltm = late.get("timeout_rate")
        es = "n/a" if em is None else f"{float(em):.2f}"
        ets = "n/a" if etm is None else f"{float(etm):.3f}"
        ls = "n/a" if lm is None else f"{float(lm):.2f}"
        lts = "n/a" if ltm is None else f"{float(ltm):.3f}"
        print(
            f"early post (~501–600): n={early.get('n')} ret_mean={es} timeout_rate={ets} | "
            f"late post (max ep_idx window): n={late.get('n')} ret_mean={ls} timeout_rate={lts}"
        )


def print_summary(result: dict[str, Any]) -> None:
    cref = result.get("config_ref", {})
    ms_eff = cref.get("max_steps_effective")
    ms_cfg = cref.get("max_steps_project_cfg")
    print("\n=== Config reference (analysis) ===")
    print(f"curriculum_taper_episodes={cref.get('curriculum_taper_episodes')} | "
          f"max_steps_effective={ms_eff} (project CFG max_steps={ms_cfg})")

    seg = result.get("segments", {})
    print("\n=== Segments (episode_index ≤ curriculum = curriculum phase) ===")
    for name in ("curriculum", "post_curriculum", "all"):
        s = seg.get(name, {})
        print(f"\n{name}: n={s.get('n', 0)}")
        if s.get("n", 0) == 0:
            continue
        print(
            f"  return: mean={s['return_mean']:.2f} std={s['return_std']:.2f} "
            f"p25={s.get('return_p25', float('nan')):.2f} p50={s.get('return_p50', float('nan')):.2f} "
            f"p75={s.get('return_p75', float('nan')):.2f} min={s['return_min']:.2f} max={s['return_max']:.2f}"
        )
        print(f"  episode_length: mean={s['length_mean']:.2f} std={s['length_std']:.2f}")
        print(f"  release_rate={s['release_rate']:.3f} timeout_rate={s['timeout_rate']:.3f}")
        print(
            f"  timeouts (released=0): {s.get('timeout_count_released0', 0)} "
            f"| horizon len>={ms_eff}: {s.get('timeout_count_horizon', 0)}"
        )

    det = result.get("segments_detail", {}).get("post_curriculum_conditionals", {})
    if det:
        print("\n=== Post-curriculum only (split outcomes) ===")
        for k in ("released_finish", "timeout_or_horizon"):
            b = det.get(k, {})
            if b.get("n", 0):
                print(f"  {k}: n={b['n']} return_mean={b['return_mean']:.2f} "
                      f"min={b['return_min']:.2f} max={b['return_max']:.2f}")

    rr = result.get("rolling_return", {})
    print("\n=== Rolling episode_return (CSV row order = time order) ===")
    print(f"window={rr.get('window')}, last_smoothed_mean={rr.get('last_smoothed_mean')}")
    print(f"  episode_index at last window end: {rr.get('episode_index_at_last_window_end')} "
          f"(env_step={rr.get('env_step_at_last_window_end')})")
    tf = rr.get("tail_window_timeout_fraction")
    if tf is not None:
        print(f"  fraction of timeouts in last rolling window: {tf:.3f} "
              "(high values make last_smoothed_mean misleading)")

    rrp = result.get("rolling_return_post_curriculum_only")
    if rrp:
        print("\n=== Rolling return (post-curriculum episodes only) ===")
        print(f"window_used={rrp.get('window_used')}, last={rrp.get('last_smoothed_mean')}, "
              f"first={rrp.get('first_smoothed_mean')}, n_episodes={rrp.get('n_episodes')}")

    em = result.get("eval_metrics", {})
    if em:
        print("\n=== Eval checkpoints ===")
        print(f"n_checkpoints={em.get('n_checkpoints')}")
        print(f"delta_mean_landing_dist (last-first): {em.get('delta_mean_landing_dist')}")
        snap = em.get("single_checkpoint_snapshot")
        if snap:
            print(f"single checkpoint: mean_landing_dist={snap.get('mean_landing_dist')} "
                  f"mean_reward={snap.get('mean_reward')} release_rate={snap.get('release_rate')}")

    pa = result.get("pattern_analysis")
    if pa:
        print_pattern_analysis(pa)


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze train_episodes / train_metrics CSV logs.")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--algorithm", choices=("auto", "sac", "ddpg"), default="auto")
    p.add_argument(
        "--curriculum-episodes",
        type=int,
        default=-1,
        help="-1 = use config.CFG.curriculum_taper_episodes",
    )
    p.add_argument("--rolling", type=int, default=_DEFAULT_ROLLING, help=f"Rolling window (default from analysis_params.py or {_DEFAULT_ROLLING})")
    p.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="Override horizon for timeout detection vs episode_length (default: project CFG max_steps)",
    )
    p.add_argument("--out-json", type=str, default=None, help="Write full JSON summary to this path")
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="If set, save analysis_training.png here (matplotlib required)",
    )
    p.add_argument(
        "--patterns",
        action="store_true",
        help="Extra pattern stats: episode-index bins, tail slices, EOF timeout run, correlations",
    )
    p.add_argument(
        "--pattern-bin-width",
        type=int,
        default=100,
        help="Episode-index bin width for --patterns (default 100)",
    )
    p.add_argument(
        "--no-metrics-overlay",
        action="store_true",
        help="Skip training_eval_overlay.png even when train_metrics CSV has eval checkpoints",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figures even if default plot-dir would apply",
    )
    args = p.parse_args()

    cfg = CFG
    ce = cfg.curriculum_taper_episodes if args.curriculum_episodes < 0 else args.curriculum_episodes
    max_steps_eff = int(args.max_steps_override) if args.max_steps_override is not None else int(cfg.max_steps)

    base = _ROOT
    log_dir = Path(args.log_dir) if os.path.isabs(args.log_dir) else base / args.log_dir

    ep_path, mt_path, algo_label = resolve_log_paths(log_dir, args.algorithm)
    if not ep_path:
        print(f"No episode CSV found under {log_dir} (train_episodes_SAC.csv or train_episodes.csv).", file=sys.stderr)
        sys.exit(1)

    result = analyze(
        cfg,
        episodes_path=ep_path,
        metrics_path=mt_path,
        curriculum_taper_episodes=ce,
        rolling_window=max(2, args.rolling),
        max_steps_effective=max_steps_eff,
        include_patterns=args.patterns,
        pattern_bin_width=max(1, args.pattern_bin_width),
    )
    result["files"] = {"episodes": ep_path, "metrics": mt_path, "algorithm": algo_label}

    print_summary(result)

    if args.out_json:
        outp = Path(args.out_json) if os.path.isabs(args.out_json) else base / args.out_json
        outp.parent.mkdir(parents=True, exist_ok=True)
        safe = _sanitize_json(result)
        with open(outp, "w") as f:
            json.dump(safe, f, indent=2)
        print(f"\nWrote JSON: {outp}")

    plot_dir = args.plot_dir
    if plot_dir is None and not args.no_plot:
        plot_dir = str(base / "tests_learning" / "figures")

    if plot_dir and not args.no_plot:
        ep_data = read_episodes_csv(ep_path)
        assert ep_data is not None
        mt_data = read_metrics_csv(mt_path) if mt_path else None
        out_png = Path(plot_dir) / "analysis_training.png"
        try:
            plot_analysis(
                ep_data,
                mt_data,
                curriculum_taper_episodes=ce,
                rolling_window=max(2, args.rolling),
                out_path=out_png,
            )
            print(f"Saved plot: {out_png}")
            if (
                not args.no_metrics_overlay
                and mt_data is not None
                and mt_data.get("step") is not None
                and mt_data["step"].size >= 2
            ):
                overlay_path = Path(plot_dir) / "training_eval_overlay.png"
                try:
                    plot_metrics_eval_overlay(mt_data, overlay_path)
                    print(f"Saved plot: {overlay_path}")
                except Exception as e:
                    print(f"metrics overlay skipped: {e}", file=sys.stderr)
        except ImportError:
            print("matplotlib not available; skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
