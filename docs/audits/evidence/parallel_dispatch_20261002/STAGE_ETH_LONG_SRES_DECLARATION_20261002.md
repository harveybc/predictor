# ETH 4h long-horizon (6..36) stage: naive skill + seasonal residual arm (declared 2026-10-02, before running)

Roles only (worker_a). Pin: predictor worktree at bbb6fac6 (campaign tool incl. `_sres` cell suffix). Caps unchanged: 2470M train and verify (1.25x measured peak). Test rows [15895,18085) never read.

1. Naives and intercept-only control on the validation rows (CPU, 1G light job), objective-equivalent = mean over horizons 6..36 of per-horizon MAE in z_train (assumed same aggregation as the campaign objective). Script stage_sres_naive_skill.py; outputs naive_table_validation.json and naive_skill_validation.json (digests in STATUS line).
2. New paired arm: grouped32_huber_adam_sres = grouped32 huber/adam with model.target_residual = seasonal_naive_cumulative (period 6, target log_return_1), seeds 2021 (5090 root) and 2022 (5070 Ti root). Pairs against grouped32_huber_adam seeds 2021 (2.36931) and 2022 (2.37383) from the 5070 root. Caveat: the two seeds run on different GPU classes.
   cid seed 2021 efcc2734b1be07a8..., seed 2022 39cd386e0dd678b8...; config_id 5e0da49e351a0193...
   Decision rule declared now: the arm "helps" only if both seeds are lower than the paired non-residual cell AND mean objective is below best same-row naive (intercept median); otherwise reported as no skill. No selection by test.
