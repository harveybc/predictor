# Satoshi — RP66–RP73 return: corrections demonstrated, matched reference measured, Q1 and Q3 executed, Q2 budget-limited

Orders: [MUSASHI_POST_HUBER_RP66_RP73_2026_09_21](../../handoffs/MUSASHI_POST_HUBER_RP66_RP73_2026_09_21.md) (commit 4de64a1).
Base reviewed by Musashi: `4ef9f71`. Work of this return: commits `42765bd` (RP66/67) → `aa971f2` (RP68/69) →
`d39b317` (RP70/71) → `d350cb5`, `e3aa751` (runner repairs) → this document's commit. Host: omega only
(CPU, `crispdm-run` scopes; the workers run no accounted wrapper yet, so no scientific child went there —
a measured constraint, not a request). Nothing read the reserved test rows; nothing financial trained
scientifically; no service was reconfigured.

## Owner-facing table (MAE_z first; native kW as a labelled supporting view)

Task: UCI 235 household, minute active power, W60/h60, DEV slice rows 1 412 361..1 462 761 (28 d train,
7 d validation), common evaluation set 10 020 origins (identical for every row below), horizon 60 min.
Metric: MAE_z = MAE / sd_train (sd_train 0.9125164391 kW, the one evaluation sigma of every row). Naive:
persistence y(t) on identical origins. skill = 1 − MAE_z_model / MAE_z_naive. Every row is generated from
verified artifacts: [CLOSURE_TABLE_RP73.json](../evidence/d3_k5_20260917/RP66/CLOSURE_TABLE_RP73.json)
(49 rows, 49 verified, 0 problems; every new row bound at level TERMINAL_ARTIFACT to a COMPLETED warehouse
terminal with matching artifact digests).

| block / arm (n seeds) | MAE_z mean (sd) | MAE kW (supporting) | naive MAE_z (same rows) | skill | reference error | reference source & kind | comparability | scope |
|---|---|---|---|---|---|---|---|---|
| **DEV_MATCHED · gru_adapted_w60 (3)** | **0.531273 (0.001061)** | 0.484795 | 0.676560 | +0.2147 | — (this IS the reference) | Gasparin 2019 GRU-MIMO family (L=1, 50, L2 5e-4, dropout 0), ADAPTED to our target/inputs/recipe; **OUR MATCHED RE-EXECUTION**, VERIFIED_COMPARATOR (closed run e4338091 under our contract digest) | REPRODUCTION of our protocol; the paper's Table 5 (MAE 0.52 kW at 15 min/96 steps) stays PUBLISHED and NOT_COMPARABLE | DEVELOPMENT, 3 paired seeds |
| DEV_MATCHED · modular_w60 (3) | 0.568045 (0.002472) | 0.518351 | 0.676560 | +0.1604 | 0.531273 (GRU adapted) | as above | paired diff modular − GRU = +0.0329 / +0.0360 / +0.0414 (3/3 positive); no interval from n=3 | DEVELOPMENT |
| Q1 · calendar (3) | 0.486819 (0.008113) | 0.444230 | 0.676560 | +0.2804 | 0.552871 (randomised control) | internal control, same capacity (9 915 params) | declared contrast `permitted_inputs`; paired calendar − control = −0.0893 / −0.0646 / −0.0442 (3/3) | DEVELOPMENT |
| Q1 · randomised_calendar_control (3) | 0.552871 (0.013407) | 0.504504 | 0.676560 | +0.1828 | 0.568045 (modular_w60, reused from DEV under identical contract) | — | chance association with the true clock measured: |r| ≤ 0.0029 on the 4 channels | DEVELOPMENT |
| Q3 · volume_112d (3) | 0.557459 (0.003285) | 0.508690 | 0.676560 | +0.1760 | 0.568045 (28 d baseline, reused) | — | declared contrast `split_rule` (train history); COMMON scaler and sigma; paired 56 d − 112 d = +0.0064 / +0.00002 / +0.0036 (3/3, one at 1.7e-5) | DEVELOPMENT |
| Q3 · volume_56d (3) | 0.560799 (0.005421) | 0.511739 | 0.676560 | +0.1711 | 0.568045 | — | as above | DEVELOPMENT |
| baseline · persistence (no fit) | 0.676560 | 0.617372 | 0.676560 | 0 | — | — | the naive of every row | — |
| baseline · daily seasonal y(t+h−1440) (no fit) | 0.801804 | 0.731659 | 0.676560 | −0.1851 | — | — | distinct baseline, identical rows | — |
| baseline · train-only constant (no fit) | 0.779185 | 0.711019 | 0.676560 | −0.1517 | — | — | distinct baseline, identical rows | — |
| Q2 · context (5 arms) | NO_NEW_MEASUREMENT | — | — | — | — | — | **NOT_EXECUTED: BUDGET_LIMITED_BEFORE_ANY_OUTCOME** (pilot: W=1440 arms 4.88–4.98 s/update → 127 579 s at the ceiling vs 14 400) | costed successor below |
| prior rows (31: successor RP55, phase-1 RP63, Musashi's factorial) | re-verified, unchanged (e.g. mae_adamw 0.542536, core_mae 0.545492) | — | 0.676560 | — | — | — | binding RECORD_DIGEST (19) / TERMINAL_ARTIFACT (12); NOT_COMPARABLE with any published number | preserved |

Reading rules: development evidence on one DEV week, three seeds; no accuracy, profit, convergence or
all-domain claim. **Not** an exact reproduction of the article. Full precision lives in the JSON table,
the arrays and the warehouse; the numbers here are rounded for reading only.

## What Musashi demonstrated and what this return proves — PRE/POST

PRE ([REVIEW_REPRODUCED_PRE.json](../evidence/d3_k5_20260917/RP66_PRE/REVIEW_REPRODUCED_PRE.json), his
probes on 4ef9f71, 0 differences from his results.json). POST
([REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP66/REVIEW_REPRODUCED_POST.json), his probes
plus the executable-task probes on the repaired tree; his script untouched):

| probe | PRE | POST |
|---|---|---|
| metric_scale USD / native | REPRODUCTION | NOT_COMPARABLE (USD is not a scale → invalid; native → fields differ) |
| horizon_seconds 72 h | REPRODUCTION | NOT_COMPARABLE (physical time inconsistent) |
| `reexecuted_on_our_rows=True` | MATCHED | TypeError: the boolean no longer exists |
| null contract | ACCEPTED | ContractRefusal (malformed / invalid) |
| unknown vs unknown | — | NOT_COMPARABLE: placeholders never match as proof |
| real factorial, foreign target+horizon, outer digest re-digested | ACCEPTED | ContractRefusal: invalid physical time; consistently re-digested inner+outer → "does not bind to the prepared data" (target, horizon) |
| changed predictions under the same receipt | mae 2→0, no problems | CHANGED ARRAYS + warehouse artifact digest mismatch; row not verified |
| missing warehouse terminal / NaN / missing arrays | no problems | each a problem on an unverified row; the missing forecast is "missing, not absent" |
| zero residual scale with positive sigma | MEASURED with 4 zero deltas | FALLBACK_FIXED_GRID (ZERO_RESIDUAL_SCALE), 4 positive finite deltas, nothing rounded |
| Friday 23:00 "6 h" target | Monday 05:00 (54 h) | excluded: NO_BAR_AT_ORIGIN_PLUS_H; targets by elapsed time |
| clamped long window | reach 67 declared as 60 | reach 67 measured, arm renamed `local_support_67` = extra context; exact null = `long_window_crop60`, reach 60 measured, paired weights |
| volume cadence | epochs (627/1257/2517 updates) | every 200 observed updates, patience 3 events, 20 opportunities for every arm and tier |

## RP66 — closure by content, along the chain

`tools/df_closure_table.py` v2: roles from the registered design (prepare → preparation, pilots →
cost_pilot, kind ae → pretraining, controls → control_forecast, fit/arm → forecast); artifacts bound to
the terminal's predictions digest (TERMINAL_ARTIFACT) or the record's digest (RECORD_DIGEST) — NOT_BOUND
is a problem; warehouse terminal by digest, status and artifact rows; arrays finite, exact population and
order, labels = Y[origin+h], naive = persistence, scaler identity; independent float64 MAE against the
record; model/naive population, horizon and scale are produced fields; a registered forecast without
arrays or receipt is a problem whatever NO_NEW_MEASUREMENT says. History re-verified WITHOUT fitting:
[RP66/CLOSURE_TABLE.json](../evidence/d3_k5_20260917/RP66/CLOSURE_TABLE.json) (31/31), RP65's table
preserved. Tests: tests/test_df_closure_table.py (21 rules incl. Musashi's four probes red→green).

## RP67 — the typed contract bound to the runtime

`tools/df_benchmark_contract.py` v2: dataclass with typed validity (positive integers, physical-time
consistency, transform/scale domains, z-score reporting rule), canonical digest over identity fields incl.
horizon_seconds/metric_scale/permitted_inputs; `decide()` from fields only (unknown placeholders never
match; declared contrasts under one estimand; comparator_state PLANNED_REFERENCE vs VERIFIED_COMPARATOR from
a CLOSED run carrying our digest); `affine_reexpression` refuses NaN/inf/sigma≤0/unsupported transforms;
`require()` (schema, fields, digest recomputed, decision identity) and `bind()` (target by column, horizon,
window, panel bytes, train sd, evaluation population) in the four runners this round used:
df_e1_phase1.run, df_e1_huber.validate, df_e1_block (every command), df_fin_runner (prepare). The coverage
inventory says the rest honestly: 36 training entry points, 4 enforced with runtime binding, 32 legacy
routes inventoried and not launched ([ENTRY_POINT_COVERAGE_INVENTORY.json](../evidence/d3_k5_20260917/RP66/ENTRY_POINT_COVERAGE_INVENTORY.json)).
Tests: tests/test_df_benchmark_contract.py (24 rules incl. the real factorial validator refusing a stale
digest AND a consistently re-digested foreign task).

## RP68 — the ML design repaired in the runner that runs

`tools/df_e1_block.py` (new): calendar through the production path; `randomised_calendar_control`
(label + per-row hashed offset: deterministic, prefix-stable, same capacity; chance association measured
and reported, not assumed zero — the v1 "permutation" was not prefix-stable and is replaced); daily lag
y(t+h−1440) read from padded panel rows; W60/W1440 over the same origins; `long_window_crop60` as the
exact information null (same weights as W60 — hash-equal initial weights —, same rows, same padding,
reach 60 by gradient and perturbation); `long_window_local_support_67` declared extra context (reach 67
measured); `short_window_deep_core`; enumeration from row identities with one common evaluation mask
(the 10 020 of the source, byte-identical); a context block trains every arm — baseline included — on the
common train intersection (the long windows and the lag withdraw 1 380/60 origins over a non-finite
padded row: 38 700 vs 40 080; never conflated with volume); the COMMON source scaler for every arm and
tier, one sigma; validation every 200 observed optimizer updates, patience in events, restore verified,
ceiling = censored wherever the best fell; counts from identities (Q3: 56 d = 79 390 windows / 79 568
unique support rows / 79 688 train-only rows, exposure 59.9; 112 d = 156 539 / 156 839 / 157 019);
cost pilots inside train (validation = last 7 train days, purged by the arm's window; the DEV validation
is never read by a pilot); governed units with terminal artifacts; closure with three distinct baselines.
Consumer-level tests (tests/test_df_e1_block.py, 14 rules): non-finite rows, gaps and duplicated labels
withdraw exactly the windows that touch them; future values/labels never reach earlier windows, features
or labels; a delivered panel whose rows differ from the source is refused; common mask = intersection;
calendar and control prefix-stable, the shift-minus-one leak fails; crop null with paired weights; observed
updates = optimizer iterations, cadence, patience, censoring; pilot never reads DEV validation; the closure
table reads the layout; volume tiers with fixed evaluation and common scaler; common train intersection.
Successor phase-2 design v2 sealed (36 cells, [PHASE2_DESIGN_SEALED_v2.json](../evidence/d3_k5_20260917/RP66/PHASE2_DESIGN_SEALED_v2.json)),
v1 (27 cells) preserved.

## RP69 — the literature comparator, adapted and measured

`tools/df_gru_reference.py`: article facts (Table 4 GRU-MIMO IHEPC: L=1, n_H=50, λ=0.0005, dropout 0;
MIMO dense readout; MSE loss; grid search; 10 repeats), explicit adaptations (our one 60-min target →
Dense(1); the modular arm's 7 inputs; L2 on kernel, recurrent kernel and readout kernel, biases excluded;
Keras default initialization; the continuity recipe MAE+Adam with validation in observed updates), and
unknowns (optimizer, LR, batch, epochs, initialization, regularized tensors, split dates, imputation
population, seeds, code) kept apart. 8 901 parameters (analytic = built). A native-recipe (MSE) variant is
a separately costed factor, not run. TCN transcription corrected in
[GASPARIN_2019_REPRODUCTION_CONFIG.json](../evidence/d3_k5_20260917/RP66/GASPARIN_2019_REPRODUCTION_CONFIG.json)
(v2: no n_H for the TCN; M=32, k=2, L=8; Table 3 counts are samples, not an enumeration; article facts /
adaptations / unknowns classed; RP65's file preserved with the error). The original 15-min/day-ahead lane
stays a separate, declared, NOT_EXECUTED lane — not a prerequisite. Tests: tests/test_df_gru_reference.py
(4 rules: parameter graph, seed pairing and measured gradient reach — recorded, at initialization it decays
below 1e-9 beyond a few dozen rows, so "whole window" is by construction, not by measurement —, learning
and restore through the block loop, fresh-process replay).

## RP72 — what ran, in the fixed priority

| block | design | pilot (train subset) | projection with 25 % headroom | decision | fits | CPU s (fits) | closure |
|---|---|---|---|---|---|---|---|
| DEV_MATCHED | e4338091 | 48.6 s (0.096 / 0.040 s per update) | 2 231 s | EXECUTE | 6, all STOPPED_ON_VALIDATION | 373.5 | verified, 0 problems |
| Q1_CALENDAR | (blocks/e1_block_q1_calendar_v1) | 77.0 s | 3 945 s | EXECUTE | 6, all STOPPED_ON_VALIDATION | 940.4 | verified |
| Q2_CONTEXT | (blocks/e1_block_q2_context_v1) | 2 259.6 s (5 pilots; W1440 arms 1 078 s each) | 159 474 s | **BUDGET_LIMITED_BEFORE_ANY_OUTCOME** | 0 | 0 | pilots preserved (5 COMPLETED terminals) |
| Q3_VOLUME | (blocks/e1_block_q3_volume_v1) | 62.4 s | 3 085 s | EXECUTE | 6, all STOPPED_ON_VALIDATION | 533.7 | verified |

A first DEV_MATCHED root (v1) was abandoned after its first pilot: the block runner's parallel acquisition
hit the same module-preload race the huber runner had met; its `prepare` campaign was closed with a FAILED
terminal naming the reason (4ed4c87e), its one pilot terminal stays COMPLETED and preserved, the runner
was fixed (`governance_modules()` before the pool; accepted units are reused, never re-trained to repair
a receipt; the prepare unit now closes with its own terminal carrying the data digests) and the block was
re-sealed as v2 with a fresh run-id. Closures ran under a later revision of the runner in two cases
(baseline tier fallback, DATA files for the table); the report records the drift
(`closure_code_drift`), the fits ran under the sealed code.

Reuse: modular_w60 of DEV_MATCHED is the baseline of Q1 and Q3 (identical rows, inputs, scaler, recipe,
cadence, seeds; recorded in each seal's `reuse` with the accepted terminal digests). The preserved phase-1
/ factorial modular fits were NOT reused: their contract differs in the validation cadence (per epoch =
627 updates, patience 3 epochs) — the necessary difference the order asked to explain.

**A finding the cadence produced, reported, not tuned:** under validation every 200 updates with patience
3 events, every modular_w60 fit stopped at 1 000 updates with its best at 400 (MAE_z 0.568), worse than
the same recipe under the epoch cadence (phase-1 core_mae 0.5455; Musashi's mae_adam 0.5440). The GRU
stopped at 2 000–2 400 updates. Patience in events at 200 updates gives 600 updates of patience against
1 881 before. Nothing was changed after seeing this; a predeclared second tier (patience 6 events, or 400
updates per event) for the SAME cells is the costed successor, declared before any of its scores.

Q2's costed successor: the two W60 arms (daily_lag 0.096 s/update, short_window_deep_core 0.160 s/update →
≈ 3 200 s with headroom for 6 fits) as a first sub-block; the three W=1440 arms need either a cheaper
gather (the per-batch window gather at W=1440 dominates: 4.9 s/update) or a declared lower ceiling; at the
current cost one W=1440 fit at the ceiling is ≈ 20 000 s — outside any single round. Q2 stays NOT_EXECUTED,
never "no effect".

## RP70/RP71 — the financial task, executable, and its acceptance

`tools/df_fin_task.py`: targets by elapsed time (the bar labelled exactly origin + h hours; absent →
excluded and counted; a row-offset would have been wrong for the counted origins), availability
limitations declared (intrabar finality, publication delay, timezone, cutoffs: not established by the
file), weekly fold identities before the reserve with purges, Huber deltas from the admissible train pairs
at full precision with declared fallbacks (insufficient pairs → fixed grid; flat target → NONIDENTIFIABLE,
family NOT_RUN; zero median with positive sigma → first positive quantile q60..q99, else fixed grid),
exactly 12 enumerated candidates per family + 4 defaults apart (trade-off declared: equal fit counts, not
identical LR/decay coverage), receivers frozen (compact W60 OHLCV 6 229 params; larger W144 OHLCV+calendar
9 857), min_delta/float64-floor resolution check. `tools/df_fin_runner.py`: governed prepare over a
bounded-range delivery (CUT before the reserve), cells with real loss/optimizer components (Huber(δ),
AdamW with biases excluded from decay), validation in observed updates, validation AND test arrays at full
precision, selection by validation only with every test score retained, moving-block bootstrap over folds
with multiplicity declared, closure by content. FL01–FL08 now call these entry points on synthetic bars
with negative controls (13 rules, 0 xfail): FL08 runs the whole route on a disposable data-gov + warehouse
serving a timed synthetic resource under a holdout after all rows, and a tampered artifact digest is caught
at closure; FL04's 1e-6 survives arrays, json, the terminal and the warehouse bit-for-bit.
FIN-LOSS-OPT design v2 sealed ([FIN_LOSS_OPT_DESIGN_SEALED_v2.json](../evidence/d3_k5_20260917/RP66/FIN_LOSS_OPT_DESIGN_SEALED_v2.json)),
state DESIGNED_NOT_STARTED. The real EURUSD resource's columns were not read (no scientific campaign is
authorised); the declared OHLCV columns are verified by the runner at delivery and the design is amended
before any financial fit if the served schema differs.

## Tests, CPU, objects

Full suite on the clean checkout d0237f0 (37 min 24 s): **2 442 passed, 39 skipped (stated reasons), 0 xfailed**;
3 failed + 8 collection errors are the repository's documented stale legacy suite (the same set as RP65);
one further failure was a stale constant in the phase-2 acceptance test (33 cells before Q2 gained its own
baseline arm; 36 now) — fixed in the closing commit and re-run green (8 passed):
[RP73_FULL_SUITE_SUMMARY.txt](../evidence/d3_k5_20260917/RP66/RP73_FULL_SUITE_SUMMARY.txt). New or rewritten
this round: benchmark contract 24, closure table 21, block 14, GRU 4, phase-2 acceptance 8, FL 13 — 84 rules,
no xfail left. CPU (omega, systemd scopes, parent+child, 38 scopes since 05:00):
[RP66_RP73_CPU_LEDGER_omega.json](../evidence/d3_k5_20260917/RP66/RP66_RP73_CPU_LEDGER_omega.json) —
**9 033 s of 14 400**, within the ceiling with the 2 000 s closure reserve; the Q2 pilots (2 260 s) and the two
full-suite runs (≈2 850 s, the first one aborted at 26 % because I wrote documents into the tree while it ran,
which the governed-route rules refuse) were the largest costs. Push and worker sync were DENIED by the
auto-mode classifier at the end of this session: the branch is committed locally at the closing commit and
not pushed; the workers are not synced — the owner's action, stated, not worked around. Consumed sources: the article (arXiv 1907.09207) Table 4/5 and Sections 6–7 as before; the
lake's coverage endpoint (re-read: 129 873 rows, 2005-01-03 01:00 .. 2025-12-31 16:00, `datetime`).
Remaining objects: Q2 (budget successor above); the cadence tier; the original-protocol lane; the financial
scientific campaign (authorisation pending); the workers' accounted wrapper.

## Corrections carried (dated 2026-09-21)

- RP65 GASPARIN config: TCN n_H=50 was a transcription error (Table 4 has no n_H for the TCN) — preserved, corrected in RP66/.
- RP65 phase-2 v1: "clamped depth = null" was wrong (reach 67 measured); "volume_counts" called the 50 400-row slice a train span; per-epoch validation; refit scalers — all replaced in v2 and the runner.
- RP65 FIN design v1: row-step horizons, rounded/zero deltas, unenumerated candidates, receivers "at pilot" — replaced by df_fin_task/df_fin_runner; v1 sealed file preserved.
- The phase-2 v1 "permuted calendar" control was not prefix-stable; replaced by the hashed-offset control.
