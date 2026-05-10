Train SAC with regulation dartboard rewards, log metrics for Fig 1:

  python3 trainSAC.py --total-env-steps 1000000 --seed 1 --log-dir paper_runs/regulation_1m

Then regenerate figures (uses checkpoints/policy_bestSAC.pt unless PAPER_CHECKPOINT is set):

  python3 paper_figures/generate_paper_figures.py

Evaluation bundle for supplementary diagnostics:

  python3 evaluate.py --checkpoint checkpoints/policy_bestSAC.pt --episodes 400 --seed 1 --output-dir eval_outputs/paper_seed1 --print-cfg --landing-heatmap --dual-pca-rollout-split
