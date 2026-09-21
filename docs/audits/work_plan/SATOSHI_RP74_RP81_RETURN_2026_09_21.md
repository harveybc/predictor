# Satoshi — RP74–RP81 return: custody and comparator gaps closed, financial population repaired, architecture × calendar measured on three accounted hosts

Orders: [MUSASHI_RP74_RP81_2026_09_21](../../handoffs/MUSASHI_RP74_RP81_2026_09_21.md) (538d0ac, review of 1880caa).
Commits of this return: `26a1e07` (RP74–RP77 + worker probe tool) → `RP78 evidence` → `RP79/RP80 code` → this
document's closing commit. Hosts: omega (coordinator, seed 1), dragon (seed 2), gamma (seed 3) — every child
inside `crispdm-run` scopes with journal accounting. No service restarted, no reserve read, no financial
campaign, no historical artifact edited; the abandoned units of RP72 keep their FAILED terminal.

## Owner-facing table — MAE_z first

Task: UCI 235 household, minute active power, W60/h60, DEV slice (28 d train / 7 d validation), **10 020
evaluation origins identical for every row**. MAE_z = MAE / sd_train, **sd_train = 0.9125164391265214 kW**
(the one denominator of every row). Naive = persistence y(t) on the same origins: **MAE_z 0.676560 (0.617372 kW)**.
skill = 1 − MAE_z_model / MAE_z_naive. SD = sample SD over the three seeds, **ddof = 1** (optimization variation
across seeds, not sampling uncertainty over weeks). Every new row is bound by the ACCEPTED ARTIFACT CHAIN
(warehouse canonical payload → predictions and record digests → the bytes read → metric recomputed):
[CLOSURE_TABLE_RP81.json](../evidence/d3_k5_20260917/RP74/CLOSURE_TABLE_RP81.json) — 64 rows: 45 verified
(ACCEPTED_ARTIFACT_CHAIN), 19 historical rows PRESERVED as METRIC_ANCHORED (score equal to the metric the accepted
terminal carries; array bytes not independently anchored), 0 problems.

**New measurement — tier 2 (patience 10 events), block ARCH_X_CALENDAR, 15 cells, 3 hosts, design e1_block_arch_x_calendar_v1:**

| arm (3 seeds, seed→host: 1 omega, 2 dragon, 3 gamma) | MAE_z mean (SD ddof=1) | seeds | MAE kW (support) | skill | stop / censor | matched reference & class |
|---|---|---|---|---|---|---|
| modular_w60 (7 inputs) | 0.539087 (0.001331) | 0.539779 / 0.539930 / 0.537554 | 0.491926 | +0.2032 | 2 CENSORED at 4 000, 1 stopped at 3 800 | adapted GRU (below): OUR MATCHED RE-EXECUTION, VERIFIED_COMPARATOR |
| gru_adapted_w60 (7 inputs) | 0.528993 (0.002374) | 0.530476 / 0.530249 / 0.526254 | 0.482715 | +0.2181 | 1 CENSORED, 2 stopped (3 600) | literature-derived (Gasparin 2019 GRU-MIMO family, adapted); Table 5 PUBLISHED = NOT_COMPARABLE, notes only |
| calendar (modular + 4 real calendar) | 0.492654 (0.008862) | 0.482480 / 0.496785 / 0.498696 | 0.449555 | +0.2718 | 2 CENSORED, 1 stopped | — |
| gru_calendar_w60 (GRU + 4 real calendar) | **0.479838 (0.003816)** | 0.480267 / 0.483422 / 0.475825 | 0.437860 | +0.2908 | 1 CENSORED, 2 stopped | — |
| randomised_calendar_control (modular + 4 hashed-offset channels, same capacity) | 0.544614 (0.005810) | 0.538882 / 0.544461 / 0.550499 | 0.496969 | +0.1950 | 1 CENSORED, 2 stopped | secondary control |
| persistence / daily seasonal / train constant (no fit) | 0.676560 / 0.801804 / 0.779185 | — | 0.617372 / 0.731659 / 0.711019 | 0 / −0.185 / −0.152 | — | three distinct baselines on identical rows |

Paired seed-level contrasts (MAE_z, seed 1 / 2 / 3; mean; SD ddof=1):

| contrast | s1 | s2 | s3 | mean | SD | signs |
|---|---|---|---|---|---|---|
| GRU − modular, original inputs | −0.009303 | −0.009681 | −0.011299 | −0.010094 | 0.001060 | 3/3 − |
| GRU − modular, calendar inputs | −0.002213 | −0.013363 | −0.022871 | −0.012816 | 0.010340 | 3/3 − |
| calendar − original, modular | −0.057299 | −0.043145 | −0.038857 | −0.046434 | 0.009650 | 3/3 − |
| calendar − original, GRU | −0.050209 | −0.046827 | −0.050429 | −0.049155 | 0.002019 | 3/3 − |
| calendar − randomised control, modular | −0.056402 | −0.047676 | −0.051803 | −0.051960 | 0.004365 | 3/3 − |
| randomised control − original, modular | −0.000897 | +0.004531 | +0.012945 | +0.005527 | 0.006975 | mixed |
| interaction (GRU_cal − GRU) − (cal − mod) | +0.007090 | −0.003682 | −0.011572 | −0.002721 | 0.009368 | mixed |

Reading (development evidence, n = 3 seeds on one DEV week; informed by prior DEV results, NOT confirmatory):
the calendar input reduces error in both architectures by about the same amount (−0.046 modular, −0.049 GRU),
the equal-capacity control does not (+0.006, mixed signs), and the architecture difference at equal inputs is
about −0.010 in favour of the adapted GRU (3/3 seeds, both input sets); no interaction is resolvable at this n.
7 of 15 fits reached the 4 000-update ceiling and are CENSORED (what more budget would reach is unknown);
none is a convergence claim. A calendar result is not a finance result; a lower forecast error is not a return.

Tier 1 (RP72, patience 3 events) versus tier 2 (this block, patience 10), same cadence, ceiling and rows —
historical rows reproduced here from the RP81 table, not re-run: modular 0.568045 → 0.539087; GRU 0.531273 →
0.528993; calendar 0.486819 → 0.492654 (within its seed SD); control 0.552871 → 0.544614. The larger effective
patience moved the modular arm most; it does not prove that longer patience must improve error.

## RP74 — reproduce first; custody and comparator

PRE: Musashi's `reproduce.py` on the base — [RP74_PRE/REVIEW_REPRODUCED_PRE.json](../evidence/d3_k5_20260917/RP74_PRE/REVIEW_REPRODUCED_PRE.json),
**0 differences** from his results.json (59 leaves). POST on the repaired tree —
[RP74/REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP74/REVIEW_REPRODUCED_POST.json):

| probe | PRE | POST |
|---|---|---|
| A2 contract hash + `prepare: COMPLETED` | VERIFIED_COMPARATOR | PLANNED_REFERENCE ("no sealed design digest") |
| A1 rewritten arrays + rewritten record, warehouse without artifact rows | MAE 0.0805→0, verified both | verified False both; problem "a local record is not custody" |
| A3 one NaN in `volume` | 60 affected train windows retained, NaN tensor | 0 retained, tensor finite |
| A4 two folds, block 2 | interval [0.1, 0.1] | INSUFFICIENT_RESAMPLING_SUPPORT, interval None, 1 complete block |
| A4 two-seed population | selects seed 1 (mae_s1) | configuration-level: family mean over seeds, seeds retained as pairs |
| A5 constant model, patience expiring at update 4/4 | STOPPED_ON_VALIDATION | `UPDATE_BUDGET+EARLY_STOPPING`, CENSORED_BY_BUDGET |
| A6 Q2 projection | 127 579 s | recost 111 000.13 s (identical to his arithmetic); decision unchanged |

Custody classes (`tools/df_closure_table.py`): a row is **verified** only when its arrays are clean AND the
warehouse's canonical payload carries the predictions AND record digests that equal the bytes read; a
historical row whose recomputed MAE equals the metric the accepted terminal carries is **METRIC_ANCHORED**
(preserved, qualified, never promoted); UNANCHORED/UNCHECKED is a problem for new-result schemas and nothing
local certifies itself. `reference_evidence()` now needs a sealed design digest, a NAMED reference arm, every
declared seed with an accepted terminal, and every reference forecast verified through the closure table with
the accepted chain; without a warehouse read it returns LOCALLY_CHECKED_REFERENCE. The same verifier serves the
table CLI, the block closure and the financial closure (no side path). Read-only reconciliation by content of
every campaign of this round: [RECONCILIATION_BY_CONTENT.json](../evidence/d3_k5_20260917/RP74/RECONCILIATION_BY_CONTENT.json)
— 37 campaigns (15 cells + coordinator prepare + 5 pilots + 2 worker prepares + 2 RP78 probes + the abandoned
RP72 prepare), 0 problems, the FAILED terminal preserved. Successor tables published without refitting:
[CLOSURE_TABLE_RP74](../evidence/d3_k5_20260917/RP74/CLOSURE_TABLE_RP74.json) (49 rows: 30 verified, 19 preserved)
and RP81 above; the RP66/RP73 tables are kept as they were.

## RP75/RP76 — the financial population is the consumed tensors; selection and uncertainty

`tools/df_fin_task.admissible_pairs`: a pair is admissible only when every input role is finite on every one of
its W window rows, its origin and label are finite, and every window row is AVAILABLE at the origin; exclusions
are counted by reason; one shared population per fold and split for every candidate; the scaler, the target
sigma and the Huber residual scale come from the same admissible train pairs; a split below the declared
minimum makes the fold produce NO score; `PairBatches` refuses a non-finite tensor as the last line. Timestamps:
a UTC-offset-aware column is converted only under the declared rule, never stripped; availability is bound to the
delivery metadata (availability_use etc.) and the record states the basis (archive label ⇒ no point-in-time or
live inference); an available-time column, when the producer contract provides one, withdraws the origins a late
arrival would leak into. Tests through `prepare` → `PairBatches` → `run_cell`: NaN and inf in each of the five
roles exclude exactly the windows that consume them; missing labels; duplicate/disordered and tz-aware labels;
split boundaries with purges; stale bytes; late arrival; an insufficient fold. Selection is at configuration
level (validation MAE_z averaged over the declared seeds; incomplete configurations never selected) with the seeds
retained as paired replicates; the moving-block bootstrap runs over calendar positions (gaps stay gaps), declares
a minimum support (5 complete blocks, 2 × block length folds) and is otherwise DESCRIPTIVE; its coverage was
measured on dependent AR(1) controls: adequate at 26 folds/block 4/ρ 0.3 and at 104/8/0.6, **anti-conservative
(≈0.69) at 26 folds/block 4/ρ 0.6** — declared in the design, not hidden. 1e-5/1e-6: through prediction,
inversion, arrays, json, terminal and the disposable warehouse bit-for-bit (FL04/FL08); the float64 floor is a
representation bound only; seed variation and fold uncertainty are reported apart.

## RP77 — stopping and accounting

Both triggers recorded (`triggers.budget_reached`, `triggers.patience_expired`; `stop_reason` names both when
both hold; the ceiling always censors). Scan of every retained cell's saved events (RP72 blocks and this one):
**no cell had both triggers**; no metadata needed correction ([STOP_CORRECTIONS.json] per block). CPU split per
cell: train-update / validation / restore / setup / final replay / wall; `seconds_per_update` is train-update CPU
only; projections count validation once plus restore and setup. Retained pilots recosted (old reports preserved):
DEV 1 785 → 1 214 s, Q1 3 156 → 2 216 s, Q3 2 468 → 1 737 s, **Q2 127 579 → 111 000 s** (decisions unchanged; Q2
stays budget-limited on retained evidence). Profile (5 batches, warm): at W=1440 the forward/backward step costs
2.66 s per batch and the window gather 0.0016 s (0.06 %); at W=60 gather is 0.7–8.5 % of an update. The
"cheaper gather" prescription of RP73 is withdrawn: the long-sequence network itself is the cost.

## RP78 — accounted workers

The wrapper and the batch slice are installed at user level on both workers (no service touched). One bounded
synthetic governed child per worker under its **own actor** (`worker.key`) against production data-gov through a
temporary reverse ssh tunnel opened from this session and closed after: delivery VERIFIED_TRANSFER, a real 200-update
fit, terminal with CPU (self + children) and peak RSS, reconciliation clean, journal scope accounting (dragon
5.866 s CPU / 960.9 M peak; gamma 4.758 s / 476.1 M) — [RP74/workers/](../evidence/d3_k5_20260917/RP74/workers/).
Then the scientific host blocks by seed: identical environment (TF 2.21.0, Keras 3.15.0, numpy 2.5.1),
prepared BLOCK_DATA byte-identical on the three hosts (the merge refuses otherwise), every worker cell verified
against its terminal artifacts at merge and by content at closure. Per-host update cost differs (omega ran 3
children in parallel): omega 870.7 s, dragon 420.0 s, gamma 475.8 s for five cells each.

## RP79 — the block

Sealed before scoring (15 cells: {modular, adapted GRU} × {7 inputs, 7 + calendar} × 3 seeds, plus the modular
randomised-calendar control × 3), tier 2 = patience 10 events (2 000 non-improving updates), same cadence
(every 200 observed updates), ceiling 4 000, min_delta 0, restore best, common scaler and rows, MAE + Adam
3e-3, batch 64. Parameters measured at seal: modular 8 127 → 9 915 with calendar (+1 788: one more ARCH-A
branch and 4 head/skip rows); GRU 8 901 → 9 501 (+600: 3 × 50 × 4 kernel rows); the control has the calendar
arm's 9 915. Initialization: arms of a seed are paired by seed, not weight-identical in shared parts (declared;
initial-weight digests recorded per cell). Train-only cost pilots (5 arms, 132.7 s): projection with the corrected
accounting 3 459 s at the ceiling, 4 324 s with headroom → EXECUTE. Fits cost 1 766.5 s across the hosts.
Persisted per cell: curves per validation event, optimizer iterations, feature support, best checkpoint, both
triggers, in-process reload predictions (fresh model instance; a separate-process replay was exercised for the
GRU builder in tests, not per cell). Q3, the pretraining factorial and the full Q2 were not repeated; the cheap
context contrasts (daily lag, exact crop) stay next in queue.

## RP80 — financial handoff

[FIN_LOSS_OPT_DESIGN_SEALED_v3.json](../evidence/d3_k5_20260917/RP74/FIN_LOSS_OPT_DESIGN_SEALED_v3.json):
consumed-tensor missingness policy, shared populations, timestamp/availability binding, what the archive can and
cannot support, weekly identities before the untouched reserve, receivers frozen (compact W60 OHLCV 6 229 params;
larger W144 + calendar 9 857), populations A (fixed-default 2×2) / B (equal-budget LR: the same 3 learning rates for
EVERY loss × optimizer cell — Musashi's 3/9 vs 6/6 asymmetry removed) / C (decay factor) / D (delta factor) = 26
fits, configuration-level selection inside B with paired seeds, stop rule with both triggers, coverage-validated
interval with its 26-week caveat, provenance (Adam, AdamW, Huber; adaptations marked), cost envelope from the
synthetic acceptance with a pilot-first rule. No universal claim; electricity chooses nothing financial. Real-data
execution awaits this design's review.

## Verification, tests, CPU, pending

Independently verified in this round: the accepted-chain custody of all 45 new/huber rows against the live
warehouse; content reconciliation of 37 campaigns; PRE/POST of the seven probes; coverage of the block interval on
synthetic controls; the workers' governed route and accounting. NOT independently verified: a fresh-process weight
replay per scientific cell (in-process reload only); the 19 historical rows' array bytes (metric-anchored only);
any financial measurement (none exists). Tests (collected rules): contract 33, closure table 31, block 16, GRU 4, phase-2 8, FL 24
(0 xfail; FL08 and the governed route on the disposable stack green on the clean tree); full-suite command and
counts in [RP81_FULL_SUITE_SUMMARY.txt](../evidence/d3_k5_20260917/RP74/RP81_FULL_SUITE_SUMMARY.txt). CPU:
[RP74_RP81_CPU_LEDGER_multihost.json](../evidence/d3_k5_20260917/RP74/RP74_RP81_CPU_LEDGER_multihost.json) —
aggregate **6,108.7 s of 14 400** including the closing full suite (omega 5,193.3, dragon 430.1, gamma 485.3); the
full suite on the clean checkout 47693d7: 2 460 passed, 39 skipped, 0 xfailed, 3 failed + 8 collection errors = the documented legacy set. Pending independent work: the cheap context contrasts under tier 2; a
separate-process replay per cell at closure; the financial design review and its cost pilot; the RP72 abandoned
root's cache cleanup (no data in use).

## Corrections carried (dated 2026-09-21)

- RP73 return: "cheaper gather" as Q2's bottleneck — withdrawn (profile: forward/backward dominates).
- RP73 return: SDs were ddof=0 without saying so; this return and the tables use ddof=1 and say so.
- RP73 return: VERIFIED_COMPARATOR was claimed from an API that accepted preparation; the adapted GRU is now verified through the strict path (named arm, complete seeds, accepted chain).
- RP73 return: rows called "verified" without the accepted artifact chain (successor/phase-1) are now PRESERVED as METRIC_ANCHORED.
