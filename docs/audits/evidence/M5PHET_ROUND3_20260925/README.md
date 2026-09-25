# M5PHET round 3, 2026-09-25 — Laya as the first layer, and where the data comes from

Packages executed by parallel agents and integrated by Satoshi: WP15 (dataset resolver; Laya chooses among
candidates), WP16 (raw rows → forecast window with the bundle's own scaler, exact parity), WP17 (m5phet.decide;
first real decision against the Laya checkpoint on the worker GPU: transform for the household series, probabilities
0.3532/0.3249/0.3219 — barely separated, plumbing proven, quality not), WP18 steps 1 and 3 (feature metric sheet and
grouping by cross-metrics on the 50,400-row household slice; k=3 recommended by silhouette 0.3075), WP20 (declared
causal study space; spec-driven fit recovering 0.93/3.11/2.03 against 1.0/3.0/2.0), WP21 (observation from a declared
representation; decision-series replay in gym-fx; training reward, not profit), WP22 steps 1, 1b, 2 (event windows
and rung-1 association over 771,554 five-minute EUR/USD bars and 121,658 calendar rows: 244,251 rows, 48,890
releases, 532 types — under the DECLARED assumption ASSUMED_SCHEDULED_PUBLICATION because no calendar on disk carries
a publication clock; not identified).

| repo | tip |
|---|---|
| M5PHET master | b0b83d1 (WP15, WP17, logo, README, plan rev 2 + WP22) |
| prediction_provider musashi/m5phet-forecast-20260924 | 4306165 (WP16) |
| feature-eng codex/m5phet-hierarchical-regimes-20260924 | a1cb7b8 (WP18-1/3 + WP22 merged) |
| causal-inference-m5phet-provider master | 72aa6e7 (WP20) |
| agent-multi satoshi/m5phet-policy-provider-20260924 | e4d01fa0 (WP21) |
| gym-fx satoshi/wp21-observation-spec-20260925 | e287ea2 |

Acceptance after round 3 on 127.0.0.1:8766 (tokened, deepseek-v4-flash): examples 9/9, prose 14/14, refusals 2/2,
envelope questions 15/15, execution_authorized false. M5PHET suite 401 passed / 1 skipped.
NO_NEW_MEASUREMENT for every area.
