# M5PHET round 4, 2026-09-25 — Laya chooses; the framework refuses, records and measures

WP12 API + bearer token (M5PHET ebaf1df). WP18 steps 2–6: Laya chose preprocessing per feature (all seven →
default_plugin), the cut (k=2 vs the deterministic k=3), extractors (rnn — declared but missing its module — and
lstm); the branch probe found NO predictor core accepting several input branches → core NOT_AVAILABLE_MULTI_BRANCH
(wp18_predictor_branch_capability.json). WP19: Laya chose agglomerative/complete/k=2; on the sealed holdout its second
cluster is never reached (indices NOT DEFINED) while the hand baseline (average/k=3) gives silhouette 0.70; both rows
NO_NEW_MEASUREMENT for regime accuracy (no ground truth). WP20-Laya: every column called `confounder` → NO_TREATMENT,
nothing fitted. WP22 step 3: local projections with HAC under the ASSUMED clock — realized_vol: all 20 CIs contain 0;
placebo passes 1/40; ADDITIVE_HOLDS; projection loses to the naive sign-mean in 16/20 return rows; NOT_IDENTIFIED by
construction (wp22_local_projections_assumed_clock.json).

Zero-shot Laya barely separates configuration options (0.30–0.41 over 3–5 options). That is the checkpoint's fact,
recorded in ~/.local/state/m5phet/decisions/; WP23 in the plan states the only admissible way to improve it.

| repo | tip |
|---|---|
| M5PHET master | 02f76c3 (WP12, WP18, plan WP23–25) |
| feature-eng codex/m5phet-hierarchical-regimes-20260924 | 8990ffc (WP19 + WP22-3 merged) |
| causal-inference-m5phet-provider master | ac63d32 (WP20 Laya half) |
| predictor satoshi/rp132-rp134-20260923 | 2c261c64 (branch-capability probe) |

Acceptance after round 4 on 127.0.0.1:8766 (tokened): examples 9/9, prose 14/14, refusals 2/2, envelope 15/15,
execution_authorized false. M5PHET suite 450/1; feature-eng 365/2/10 pre-existing collection errors.
NO_NEW_MEASUREMENT for every area.
