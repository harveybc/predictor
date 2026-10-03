# Matched plan: NEAT vs random proposal on ECL h24 (declared BEFORE any new cell ran)

Declared 2026-10-02 (local evening), validation only, test never read. No new quality claim is made by this file.

## Pricing (measured, v2 queue campaign_corrected_r0_v2, incumbent config ac27a399 = grouped32 + seasonal_naive_24 residual, R0, MAE, 4 verified seeds)
- Train attempts on worker_a RTX 5090: wall 105.6 - 180.7 s (4 cells, 6314-8610 updates, ~0.0132-0.0136 s/update), cgroup peak max 5127 MiB (cap 6915M, never lowered).
- Verify (exact-match rescoring) wall 12.5 - 162.6 s, cgroup peak 3524 MiB; verify cap becomes 1.25 x 3.695e9 B = 4620M (the v2 3914M is below 1.25x, raised, never lowered).
- One cell (train + verify) ~ 4.5 min => a generation of P candidates costs about (P-1) x 4.5 min per seed (generation zero reuses the verified control, nothing retrained).
- Incumbent val MAE (seed 2021/2022/2023/2024): 0.213458 / 0.213489 / 0.213018 / 0.212887; mean 0.21321; seed-to-seed range 0.000602 (declared "spread").
- Comparators on the same validation rows: seasonal-24 naive MAE 0.247966, persistence 0.851406; skill = 1 - MAE/naive.

## Arms (same data, evaluator, pinned engine ba246aca, search space digest, restricted sub-space restrict_v1.json, seed, objective, budget)
- Free genes: train.learning_rate, weight_decay, patience, loss (+huber_delta), branch.channels, grouping_size {8,32}, core.dropout. Everything else pinned to the incumbent.
- Generation zero = the incumbent control (grouped32 seasonal residual R0 MAE), imported verbatim from the v2 queue (no retraining), plus P-1 initial candidates.
- NEAT arm: modular_proposal_strategy=neat, population P=6, generations G=2 (generation 0 initial, generation 1 = first evolved), neat seed 20261003. Duplicate identities are reused for free.
- Random arm: uniform draws from the same restricted sub-space with the same validity rule, draw seed 20261003, count N = 1 (control) + number of DISTINCT new cells the NEAT arm evaluated (declared so the budgets match in evaluated cells, not in nominal slots). Run after the NEAT arm.
- Seed: 2021 only. Escalate to 2022 (paired) only for a candidate whose seed-2021 MAE beats the incumbent seed-2021 value 0.213458 by more than 0.000602 (the declared spread); a "beats the incumbent" claim needs the 2022 pair too, and the comparison is to the incumbent mean 0.21321.
- Order and hardware: NEAT arm then random arm, strictly sequential, one heavy job at a time (worker_a has ~7 GiB free, 8 GiB slice ceiling), both on the RTX 5090 (same device, so device numerics are not a confound); the RTX 5070 Ti is the fallback only if the 5090 is occupied. Queue and runner are launched on worker_a under crispdm-run; nothing batch runs on the coordinator.
- Verification: every cell is VERIFIED only on bitwise-equal rescoring (TF_DETERMINISTIC_OPS=1); a tolerance-only pass is FINDING_NOT_EXACT and excluded from fitness.
- Not claimed: this is hyperparameter optimization, not a neural NEAT head; one seed per cell, so any difference below the spread is not evidence.
