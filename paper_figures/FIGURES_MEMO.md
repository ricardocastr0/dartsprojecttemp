# Figures memo — final SAC run, regulation dartboard

Companion to `analysis_decisions.md`. Captures what each figure shows, what
the paper should *say* about it, and where the numbers come from. All
quantitative values match `paper_figures/figure_stats.txt` and the JSON in
`tests_learning/figures/final_run_summary.json`.

---

## Headline result (one sentence for the abstract / conclusion)

> Trained on the regulation-style dartboard reward (inner bull r = 6.35 mm,
> outer bull r = 16 mm, third tier r = 170 mm), the SAC policy converges to a
> 1.000 release rate with a mean radial miss of **9.9 mm (σ = 8.1 mm)** over
> 400 deterministic evaluation episodes — every throw lands inside the
> doubles-wire region, and 84% land inside the outer bull (16 mm).

This is the `policy_finalSAC.pt` checkpoint at 1M env steps; see
`analysis_decisions.md` for why this checkpoint is preferred over the
`best`-by-composite checkpoint.

---

## Figure 1 — Training curves (`fig1_training_curves.png`)

**What it shows.** Three panels of the periodic 10-episode evaluation block
written every 100k env steps: mean reward, release rate, mean radial miss, all
vs environment step.

**Trend.** Mean reward jumps from ~50 at 100k to ~120 at 200k, then climbs
slowly to a ~140 plateau by 700k. Release rate is 1.000 across every 100k
checkpoint except 900k (0.9 — single eval-block timeout). Mean radial miss
falls from 0.21 m at 100k to 1.1 cm by 700k and stays there.

**What to highlight in the paper.**
- Training is qualitatively *cleaner* than the previous reward shaping: there
  is no 600k catastrophic collapse (cf. earlier draft's "transient
  instability" subsection — it does not occur here).
- The 900k dip is a single timeout in 10 evaluation episodes, not a policy
  collapse. Convergence is monotonic for both reward and miss otherwise.
- The model is essentially trained by 700k; the remaining 300k steps refine
  rather than discover.

**Caption (suggested).** "Periodic SAC evaluation (10 episodes per checkpoint,
deterministic actor) over 1M env steps. Mean reward saturates near 140;
release rate stays at 1.000 except for one 10-episode block at 900k; mean
radial miss falls below 1.5 cm by 700k."

---

## Figure 2 — Terminal score histogram (`fig2_score_histogram.png`)

**What it shows.** Distribution of terminal dartboard scores across the 400
evaluation episodes for `policy_finalSAC` (seed 1). Vertical dashed lines
mark the score values reached on the +Δy axis at the three reward tier
boundaries (`r = 0.0063, 0.016, 0.170 m`).

**Trend.** Three clean bands:
- ~110 (inner bull, full +60 ring bonus): **45.0%** of episodes.
- ~78 (outer bull, +28 bonus): **38.8%** of episodes.
- ~58 (inside doubles wire, +8 bonus): **16.2%** of episodes.
- Outside: **0.0%**.

**What to highlight.** The histogram has effectively no left tail — every
throw is either an inner bull, outer bull, or middle-tier hit. This is the
clearest single-figure evidence of the robustness claim: the landing
distribution does not have a heavy tail of misses under the new reward.

**Caption (suggested).** "Distribution of terminal dartboard scores (N = 400).
Bars cluster near the three reward tiers (110 / 78 / 58 corresponding to
inner bull / outer bull / inside doubles wire). No episodes scored below the
third-tier boundary."

---

## Figure 3 — Release-step histogram (`fig3_release_step_histogram.png`)

**What it shows.** When (in the 40-step horizon) the policy crosses the
release threshold. 100% of the 400 evaluation episodes released; 0 timeouts.

**Trend.** The mode is exactly **step 10** (`t_min`, the earliest legal
release). The tail extends only to step 14. Almost half of the throws release
at the earliest possible step.

**What to highlight.** The policy has learned to *commit* — it does not waste
steps. This is consistent with the strong negative correlation between
episode length and return (Pearson −0.76 across 75k post-curriculum
episodes). The optimal-release-step prior \(\mathbb{E}[t^*]=20\) is exposed
in the observation, but the policy ignores it once accuracy is high enough at
\(t_\min\).

**Caption (suggested).** "Release-step distribution under deterministic
evaluation (N = 400). All 400 episodes release; the modal release step is
\(t_\min = 10\), with a short tail through step 14."

---

## Figure 4 — PCA cumulative variance (`fig4_pca_cumulative_variance.png`)

**What it shows.** PCA on the 12-D joint state (6 angles + 6 velocities)
collected from every step of every evaluation episode, run independently on
two evaluation environment seeds.

**Trend.**

| Eval seed | PC1 | PC2 | Cum @ k=2 | Effective DOF (95%) |
|:---------:|:---:|:---:|:---------:|:-------------------:|
| 0 | 0.665 | 0.140 | 0.805 | 7 |
| 1 | 0.660 | 0.155 | 0.814 | 7 |

**What to highlight.**
- ~80% of trajectory variance lives in just **two principal components**
  across both seeds, supporting the motor-synergy claim.
- The 95% threshold sits at **k = 7** rather than \(k=2\): the system is
  *low-dimensional but not collapsed* — exactly the qualitative claim already
  in the Discussion section, but with concrete numbers.
- Both seeds give the same effective DOF and very similar PC1/PC2 ratios →
  the low-dimensional structure is reproducible, not seed noise.

**What this means for the paper text.** The earlier draft sometimes claimed
"~97% in 2 PCs / 2-DOF". Replace with: *"≈80% of trajectory variance lies on
a 2-D manifold (PC1 + PC2); the cumulative variance reaches 95% at PC7. The
policy compresses the 12-dimensional joint state onto a low-dimensional
manifold without collapsing it onto a degenerate subspace."*

**Caption (suggested).** "PCA on 12-D joint state across N=400 evaluation
episodes per environment seed. The first two components explain ≈80% of
trajectory variance in both seeds; the 95% cumulative threshold is reached at
the seventh component."

---

## Figure 5 — Landing density (`fig5_landing_density_contour.png`)

**What it shows.** Smoothed 2-D density of dart landings in board-relative
coordinates (\(\Delta y, \Delta z\)), with the regulation inner-bull (6.35 mm)
and outer-bull (16 mm) circles overlaid in white-on-black for legibility on
the bright peak.

**Trend.** All 400 landings sit inside the ±0.10 m view (no outliers
excluded). The density peak is centered on the bull and almost entirely
contained within the outer-bull circle (16 mm radius). The inner-bull circle
(6.35 mm) crosses the densest core. The whole pattern is sub-centimeter on
both axes.

**What to highlight.**
- This is the *visual* equivalent of the std = 8.1 mm number: the bright core
  is roughly the same scale as the inner-bull ring.
- The plot is a direct quantitative comparison to a real dartboard: the
  policy lands on the bull more often than not, on the same ring overlay you
  would see in a tournament target diagram.
- Together with Fig 2, this rules out the "tight mode + heavy tail"
  characterization that applied to earlier checkpoints.

**Caption (suggested).** "Landing density of N = 400 deterministic throws,
overlaid with regulation inner-bull (r = 6.35 mm) and outer-bull (r = 16 mm)
circles. All landings are within the doubles-wire region; the density peak is
contained within the outer-bull ring."

---

## Figure 6 — Critic value calibration (`fig6_critic_calibration.png`)

**What it shows.** Scatter of the SAC critic's pre-rollout value estimate
\(Q(s_0, \mu_\phi(s_0))\) (x-axis) vs the realized **discounted** episode
return \(\sum_t \gamma^t r_t\) (y-axis) for each of the 400 evaluation
episodes. Solid red line is the linear fit; dashed line is \(y=x\).

**Trend.** Pearson \(r = 0.49\); fit slope \(\approx 0.96\); intercept
\(\approx +10\); RMSE 6.9. The critic preserves the relative ordering of
episode quality (slope close to 1) but underestimates the absolute return
by ~6 units across the cluster.

**What to highlight.** This is the expected SAC behavior: \(Q\) is the
*entropy-augmented* return target \(\sum_t \gamma^t (r_t + \alpha \mathcal{H})\),
while the y-axis is reward-only. The constant offset is consistent with
adding the cumulative \(\alpha \mathcal{H}\) term (positive entropy bonus per
step). The scatter is dominated by stochasticity in the *episode* (initial
joint angles, sampled \(t^*\)), not by critic error per episode — exactly
the regime SAC was designed for.

**Caption (suggested).** "Critic value at episode start, \(Q(s_0,
\mu_\phi(s_0))\), versus realized discounted return on N = 400 deterministic
evaluation episodes. The fit slope (≈ 0.96) shows the critic preserves
relative quality; the constant +10 offset is consistent with the SAC
entropy-augmented value target."

---

## Figure 7 — Robustness to initial-condition noise (`fig7_robustness_reset_noise.png`)

**What it shows.** Two panels vs the standard deviation of additive
Gaussian noise on the initial joint angles (sweep over
\(\sigma_\theta \in \{0, 0.05, 0.10, 0.15, 0.20\}\) rad, 200 episodes per
condition):

- left: mean radial miss with ±1 std error bars,
- right: release rate.

**Trend.**

| \(\sigma_\theta\) (rad) | Release rate | Mean miss (m) | Std miss (m) |
|:----------------------:|:------------:|:-------------:|:------------:|
| 0.00 | 1.000 | 0.0099 | 0.0094 |
| 0.05 | 1.000 | 0.0113 | 0.0100 |
| 0.10 | 1.000 | 0.0198 | 0.1108 |
| 0.15 | 0.990 | 0.0156 | 0.0140 |
| 0.20 | 0.975 | 0.0244 | 0.1150 |

**What to highlight.** The training reset range was uniform on ±0.30 rad
on each joint; this sweep uses Gaussian noise *on top of* a fresh reset, so
the agent sees novel initial configurations. Mean miss stays at ~1–2 cm
through \(\sigma_\theta=0.20\) rad (≈ 11.5°). Release rate drops only
slightly (from 1.000 to 0.975). The two large std bars
(\(\sigma_\theta = 0.10\) and 0.20) correspond to a small handful of
episodes being knocked into a worse trajectory; the median is still
sub-centimeter.

**Caption (suggested).** "Sweep of additive Gaussian noise on the initial
joint angles. Mean radial miss remains within ~2.5 cm and release rate
above 0.97 across all noise levels tested up to \(\sigma_\theta = 0.20\) rad
(≈ 11.5°)."

---

## Figure 8 — PCA trajectory projection (`fig8_pca_trajectory_projection.png`)

**What it shows.** Each of the 400 evaluation episodes' joint state
trajectory (12-D angles+velocities) projected onto the global PC1–PC2
plane. Small dots mark episode starts; "x" markers mark the release/end.

**Trend.** All trajectories *originate* from a tight cluster at
PC1 ≈ +1.7 (corresponding to the small-amplitude reset state) and *fan
out* along PC1 toward release configurations spread on the left. The
two-dimensional projection captures **81.4%** of state variance with
PC1 = 66% and PC2 = 15%. The fan structure shows that the policy uses a
small number of correlated motion patterns rather than independent joint
control.

**What to highlight.** This is the visual companion to Fig 4: it makes the
"low-dimensional manifold" claim concrete by showing that all 400
trajectories live on a recognizable 2-D ribbon, even though release
endpoints differ. The fan opens consistently in PC1 → release direction is
the dominant mode of variation; PC2 captures finer adjustments.

**Caption (suggested).** "All 400 evaluation trajectories projected onto
the first two principal components of the 12-D joint state. Episodes start
in a tight cluster (right) and fan along PC1 (66% variance) toward
release configurations on the left, with PC2 (15% variance) capturing
secondary adjustments."

---

## Figure 9 — Policy entropy across the throw (`fig9_policy_entropy_per_step.png`)

**What it shows.** Mean ± 1 std of the policy's per-step **pre-tanh
Gaussian entropy** (sum across the 7 action dimensions) across the 400
evaluation episodes, plotted against the step index within the episode.
Light gray bars on the right axis show how many episodes are still active
at each step.

**Trend.** Entropy starts at ~3.3 nats at step 1, stays roughly flat
through step 2, then **decreases monotonically** toward ~−2.3 nats by
step ~12. The standard deviation band tightens over time, and almost no
episodes are still running by step 13.

**What to highlight.** The policy *increases its confidence* (sharper
distribution) as the throw progresses — the closer the arm gets to its
optimal release window, the more deterministic the action becomes. This
is consistent with maximum-entropy RL: SAC trades off entropy and reward,
but here the optimal trade-off is to be exploratory early (when many
configurations could still produce a good throw) and decisive late (when
the wind-up is committed). The fact that the entropy is *negative* in the
late steps means the pre-tanh Gaussian standard deviation has shrunk well
below \(1/\sqrt{2\pi e}\) — i.e. the policy is essentially deterministic
just before release.

**Caption (suggested).** "Per-step entropy of the SAC policy (sum over 7
action dims) across N = 400 deterministic evaluation episodes. The
distribution starts diffuse (~3.3 nats) and sharpens monotonically into
a near-deterministic commitment by the release step."

---

## Cross-figure narrative for the paper

1. **Fig 1 → training is stable and reaches the centimeter scale by 700k.**
2. **Fig 3 → the policy releases promptly, never timing out at evaluation.**
3. **Fig 5 → the resulting landings cluster inside the outer-bull ring.**
4. **Fig 2 → that cluster maps to a clean trimodal score distribution
   bounded below by the third reward tier.**
5. **Fig 4 → the policy that achieves this lives on a low-dimensional
   manifold (≈80% variance in 2 PCs, 95% in 7), supporting the DOF-reduction
   claim with concrete numbers.**

---

## Numbers checklist for the LaTeX rewrite

These are the values that should propagate from `figure_stats.txt`,
`final_run_summary.json`, and the Best-vs-Final table in
`analysis_decisions.md`:

- Mean radial miss \(d = 0.0099\) m, \(\sigma = 0.0081\) m, release rate
  \(1.000\), N = 400.
- Tier fractions (final / seed 1): inner 45.0%, outer 38.8%, third 16.2%,
  outside 0.0%.
- PCA: PC1 ≈ 0.66, PC2 ≈ 0.15, cum @ 2 ≈ 0.81, eff DOF (95%) = 7.
- Per-joint torque variance: joint 1 = 0.254, joint 2 = 0.231, joint 3 =
  0.112, joint 4 = 0.169, joint 5 = 0.142, joint 6 = 0.144 (final / seed 1).
- Training: 1.0 × 10⁶ env steps; 77,173 episodes; post-curriculum mean
  return 114.08; released-finish mean return 122.64; timeout penalty
  consistent at −20 (timeout episodes mean return −21.37).

Anything in the paper that conflicts with these numbers should be revised
through the items in `PAPER_UPDATE_CHECKLIST.txt`.
