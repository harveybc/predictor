# Satoshi — RP82–RP89 return: one verification authority, financial selection as declared, a cheap context measurement on three hosts, and one external refusal recorded

Orders: [MUSASHI_RP82_RP89_2026_09_21](../../handoffs/MUSASHI_RP82_RP89_2026_09_21.md) (48a7710, review of bd6fcf4).
Commits of this return after 48a7710: 447ac80 221e764 d526f0f 8383fc9 26777f0 19d8aca (newest first) plus this document's closing commit.
Hosts: omega (coordinator, seed 1), dragon (seed 2), gamma (seed 3), every child inside `crispdm-run` scopes. No service
was restarted or reconfigured, no reserve or DEV week was read by the financial work, no whole campaign was rerun, no
historical artifact edited; Musashi's 15 fresh-process replays were ADOPTED, not repeated.

## Owner-facing table — MAE_z first

Task: UCI 235 household, minute active power, W60/h60, DEV slice; **10 020 evaluation origins identical for every row**;
MAE_z = MAE / sd_train with **sd_train = 0.9125164391265214 kW** (checked against the contract for every row); naive =
persistence y(t) on the same origins, **MAE_z 0.676560**; skill = 1 − MAE_z_model / MAE_z_naive; SD over three seed/host
blocks, **ddof = 1** (it includes host effects, not only initialization). "Verified" = clean arrays AND accepted artifact
chain AND accepted preparation artifact AND contract denominator:
[CLOSURE_TABLE_RP89.json](../evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json) — 70 rows: **39 verified** (the five
blocks with accepted preparation evidence), **31 preserved with a qualified scope** (12 factorial rows: arrays anchored,
preparation local only; 19 successor/phase-1 rows: metric-anchored), 0 problems.

**New measurement — RP87 context block (tier 2: patience 10 events, cadence 200, ceiling 4 000), 6 cells + one inherited control:**

| arm (3 seeds; seed→host 1 omega, 2 dragon, 3 gamma) | MAE_z mean (SD ddof=1) | seeds | MAE kW | skill | stop / censor | comparability |
|---|---|---|---|---|---|---|
| modular_w60 (7 inputs), common train population 40 020 | 0.541641 (0.004551) | 0.546517 / 0.540898 / 0.537507 | 0.494258 | +0.1995 | 3 CENSORED at 4 000 | declared contrast `permitted_inputs`; the ARCH baseline (40 080 origins) is NOT reused: populations differ |
| daily_lag (7 inputs + y(t+h−1440)) | **0.533894 (0.004278)** | 0.538072 / 0.534088 / 0.529522 | 0.487188 | +0.2109 | 2 CENSORED, 1 stopped at 3 800 | paired lag − baseline = −0.008445 / −0.006811 / −0.007985 (3/3 −), mean −0.007747 (SD 0.000837) |
| long_window_crop60 (exact-crop control) | inherits modular_w60 | — | — | — | not fitted | proved training-equivalent on the real rows: 30 updates, max batch-loss difference 0.0, identical weights ([EQUIVALENCE.json](../evidence/d3_k5_20260917/RP82/blocks/e1_block_context_daily_lag_v1/EQUIVALENCE.json)) |
| persistence / daily seasonal / train constant (no fit) | 0.676560 / 0.801804 / 0.779185 | — | — | 0 / −0.185 / −0.152 | — | three distinct baselines on identical rows |

Reading: on this task the causal daily-lag channel (the target's value 1 380 minutes before the decision, available at
every origin by construction, no interpolation or future fill; +81 parameters) lowers the error by about 0.008 MAE_z in
all three paired blocks — a small effect of THIS feature addition, scoped as such; five of six fits are budget-censored,
no convergence claim. The unexecuted Q2 long-context questions are preserved as they were.

Retained measurements (RP79 block, re-closed under the authority, 15/15 replays adopted from Musashi): GRU + calendar
0.479838, modular + calendar 0.492654, GRU 0.528993, modular 0.539087, control 0.544614 (unchanged). Cross-tier
interpretation corrected per finding 6: only seed 1 shares training prefixes with the tier-1 cells; seeds 2/3 moved to
workers, so the tier-1 → tier-2 change is not attributed to patience alone and no hardware cause is inferred; within-seed
host blocking is kept for every contrast above.

## RP82/RP83 — PRE/POST of Musashi's probes

PRE ([RP82_PRE](../evidence/d3_k5_20260917/RP82_PRE/REVIEW_REPRODUCED_PRE.json), 107 leaves, 0 differences from his
results.json) → POST ([RP82/REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP82/REVIEW_REPRODUCED_POST.json),
his steps repeated by [reproduce_post.py](../evidence/d3_k5_20260917/RP82/reproduce_post.py), his script untouched):

| probe | PRE | POST |
|---|---|---|
| block closure on a copy with predictions replaced by truth + record rewritten | verified before and after, MAE_z 0 | `BlockRefusal: closure failed` (CHANGED ARRAYS / CHANGED RECORD); REPORT: verified False, summary/paired/baselines None |
| scaler_sd × 10 in the closure-time DATA.npz | verified, MAE_z 0.0540 | unread: the preparation's own evidence (BLOCK_DATA) is the denominator; MAE_z 0.539779, PREPARATION_ACCEPTED_ARTIFACT |
| scaler_sd × 10 in BLOCK_DATA.npz itself | — | PREPARATION_CHANGED + SCALE problems; nothing verified |
| reference arm relabeled UNTRAINED_REFERENCE, old design digest | VERIFIED_COMPARATOR | PLANNED_REFERENCE: the design digest does not recompute |
| the same, consistently rehashed | — | PLANNED_REFERENCE: accepted terminals name configuration f4e616… and arm `gru_adapted_w60` (IDENTITY); preparation belongs to another design |
| the genuine reference | — | VERIFIED_COMPARATOR (accepted chain, all seeds, 10 020 rows) |
| financial selector with all 26 candidates, A/D fabricated best | A default and D arm selected across 10 vs 16 | B strata only (3 LRs each, 4 strata), A reported, 10 C/D paired contrasts; consumed A4/B12/C4/D6 |
| bootstrap [-100, NaN, 1×10], block 2 | mean −8.18, interval [1, 1] | INSUFFICIENT_SUPPORT_ISOLATED_WEEKS: mean over all observed weeks, no interval, position 0 named |

One authority, `tools/df_closure_table.verify_run` / `verify_fin_run`: preparation custody from the ACCEPTED prepare
terminal's data artifact (BLOCK_DATA / FIN_DATA read by digest), the denominator checked against the contract's sd_train,
each accepted terminal's `config_sha256` bound to the design digest and its tags (arm, seed / candidate, fold, seed) to the
design cell, the design digest recomputed from content. The block closure, the financial closure and the table CLI
consume it; a failed verification emits no summary, no paired contrast, no baselines, no selection. History without an
accepted preparation artifact keeps an explicit weaker scope (`ARRAYS_ANCHORED_PREPARATION_NOT_ACCEPTED`,
`METRIC_ANCHORED`) — never invented custody. `reference_evidence` consumes the same authority (named arm, complete seeds,
accepted chain, recomputed digest, accepted tags). Tests: closure table 35, contract 33, block 19 (incl. the substituted-
prediction closure refusal and the fresh-process replay), FL 25 (v4 selector with all 26 candidates and rejections,
isolated weeks, coverage certification, FL08 negative control by substituted bytes).

## RP84/RP85 — financial selection and uncertainty

`select` v4: A reported; B searched within EACH loss × optimizer stratum over the same three LRs at configuration level
(validation MAE_z averaged over paired seeds; incomplete configurations never selected); C/D as paired sensitivity
contrasts against their default-LR B anchor; identity from records; duplicate, foreign, non-finite, incomplete and
custody-unverified records rejected and counted. `block_bootstrap`: estimand = mean paired effect over the usable week
population; an observed week that can enter no complete block is a NAMED support limitation (descriptive result, no
interval); minimum support 5 complete blocks; an interval is CONFIRMATORY only under a predeclared coverage certificate
for (positions, block length), assessed against nominal 0.95 with Monte Carlo error (200 sims) — at 26 positions / block 2
the certificate is NOT granted (measured coverage below 0.95 − 2·SE on dependent controls), so financial intervals at
that call stay descriptive. FIN-LOSS-OPT v4 sealed:
[FIN_LOSS_OPT_DESIGN_SEALED_v4.json](../evidence/d3_k5_20260917/RP82/FIN_LOSS_OPT_DESIGN_SEALED_v4.json).

## RP86 — replay and scope

A read-bound fresh-process checkpoint replay is part of every block closure for NEW artifacts (rebuild, load weights,
predict, `allclose(atol=rtol=1e-6)`); replays are keyed by the checkpoint digest (REPLAYS.json) and not repeated for
unchanged bytes; adopted independent replays are recorded as such. Context block: 6/6 replays pass (max native
difference 1.414e-6 under the combined rule). RP72 blocks (18 cells) were re-closed with in-process reload parity only
(stated scope: no fresh-process replay was ever run for them). Per cell the records carry initialization digest, host,
thread settings, batch order seed, observed updates, checkpoint event, cadence and both stop triggers.

## RP87 — what ran

Common training AND evaluation populations enumerated first (intersection 40 020 train origins; 10 020 evaluation);
the ARCH baseline could not be reused (population differs) → new baseline cells. Pilot 59.6 s; projection with headroom
2 041 s → EXECUTE; fits 883 s across hosts (omega 419, dragon 217, gamma 247). Gamma's first execute attempt was
REFUSED by the wrapper's memory admission (1.9 G free above the reserve) after its prepare unit had been accepted; it was
re-run at a 1 500 M cap (peak 726 M) — the refusal is recorded, no unit was left open. Merge verified artifacts and
byte-identical prepared data across hosts; closure verified 6/6 with replays; reconciliation by content of this round's
15 campaigns: 0 problems ([RECONCILIATION_BY_CONTENT.json](../evidence/d3_k5_20260917/RP82/RECONCILIATION_BY_CONTENT.json)).

## RP88 — financial cost pilot: sealed, then REFUSED by the service (recorded, not circumvented)

Sealed before any byte ([fin_cost_pilot/DESIGN.json](../evidence/d3_k5_20260917/RP82/fin_cost_pilot/DESIGN.json)):
the 52-week pre-DEV slice 2023-06-26..2024-06-23 (DEV starts 2024-06-24; reserve 2025-01-01 untouched), 2 receivers × 2
horizons × 4 fixed defaults = 16 configurations × 200 observed updates, internal purged validation, costs measured apart
(the implementation and its synthetic acceptance pass: `tools/df_fin_runner.py cost-pilot`, test RP88).
**Outcome: NOT_EXECUTED — external refusal.** Exact operation: governed ranged download of
`financial_files/market_data/forex/g10/eurusd/1h.parquet` (2023-06-26 → 2024-06-23), unit `prepare`, run-id
`satoshi-fin-cost-pilot-20260921` → **HTTP 422 "resource availability contract required"**: the deployed lake declares no
resource contract (event/available time columns, timezone, frequency, optional availability block) for that resource
and refuses a timed delivery without it. That contract is operator metadata; the runner does not invent it. The campaign
`satoshi-fin-cost-pilot-20260921-prepare-data` was registered before the download and received no delivery; the client can
close a unit only with a delivery, so it stays open and is listed. Bytes read: 0. Next step (operator): declare the
resource contract; then the sealed pilot re-runs under a fresh run-id, unchanged
([RP88_COST_PILOT_OUTCOME.json](../evidence/d3_k5_20260917/RP82/fin_cost_pilot/RP88_COST_PILOT_OUTCOME.json)).
The scientific allocation therefore stays a projection rule (26 candidates × 3 seeds × folds per receiver/horizon; history
52/104/208 weeks as estimates from the measured slice) awaiting the pilot's numbers; no search, no test scoring.

## Verification, tests, CPU, pending

Independently verified: accepted-chain custody of 51 rows and accepted preparation of 39 against the live warehouse;
reconciliation of 15 campaigns; PRE/POST of six probes; crop-control training equivalence on real rows; coverage of the
block interval on dependent controls (with the negative certification at 26/2). Not independently verified: the 18 RP72
cells' fresh-process replay (in-process reload only); the 31 preserved rows' preparation bytes; anything financial on real
data (nothing ran). Full suite on the clean checkout: [RP89_FULL_SUITE_SUMMARY.txt](../evidence/d3_k5_20260917/RP82/RP89_FULL_SUITE_SUMMARY.txt).
CPU: [RP82_RP89_CPU_LEDGER_multihost.json](../evidence/d3_k5_20260917/RP82/RP82_RP89_CPU_LEDGER_multihost.json)
(aggregate **5,659.8 s of 14 400** including the closing full suite: omega 5,191.0, dragon 219.6, gamma 249.2). Full suite on the clean
checkout 64c6205: 2 469 passed, 39 skipped, 0 xfailed, 3 failed + 8 collection errors = the documented legacy set.
Pending independent work: the financial cost pilot once the resource contract exists; a fresh-process replay of the RP72
cells if their scope must be raised; the coverage-certified regime for financial intervals (more DEV weeks or a block
length with a granted certificate); Q2 long-context questions (unchanged, budget-limited).

## Corrections carried (dated 2026-09-21)

- RP81 return: rows called "verified" whose preparation had no accepted artifact (the 12 factorial rows) are now PRESERVED with the qualified scope; "verified" requires accepted preparation and the contract denominator.
- RP81 return: the tier-1 → tier-2 change was described as an effect of patience; seeds 2/3 also changed host and training prefixes — the attribution is withdrawn.
- RP81 return: the block-level "verified comparator" claim relied on a resolver that accepted preparation; the adapted GRU is now verified through the strict authority (accepted tags, recomputed digest).
- RP81 financial: the v3 selector pooled populations (10 vs 16) and the bootstrap could drop an observed week from the interval; both replaced (v4, usable-population estimand).
