# Lane D return: corrected modular DOIN campaign (closed)

Satoshi, successor technical lead, 2026-10-01.
Branch `satoshi/d-corrected-queue-20261001`.
Evidence: `docs/audits/evidence/lane_d_20261001/`.

## New result

The campaign `d_ecl_l24_h24_corrected_r0_v1` is closed. All 36 cells are
terminal: 32 VERIFIED and 4 REFUSED_BY_ENGINE (never run). Every verified cell
was rescored from its saved checkpoint in a separate process, with exact
bitwise agreement, under `TF_DETERMINISTIC_OPS=1`.

**Final incumbent: seq 6, `c950ec17`.** Configuration: corrected default
per-feature model, branch and core donors frozen (R1), MAE loss.

| Measure | Value |
|---|---|
| Mean validation MAE (z_train) | 0.37509125759139983 |
| Seed 2021 / seed 2022 | 0.3750808655 / 0.3751016497 |
| Same-row persistence MAE | 0.851406 |
| Aggregate skill | 0.559 |

This is a **strict-minimum selection, not a demonstrated advantage**:

- The mean paired gap to the R0 reference is −0.0084737.
- That gap lies within the R0 two-seed spread of 0.00976.

Validity limits:

- **Split and task.** Validation split only. Task is ECL L24/H1..24, all 321
  channels in and out, train-only StandardScaler.
- **Short- and long-horizon weakness.** Skill against persistence is negative
  at h1, h23 and h24 in 32 of 32 cells.
- **Seasonal naive.** A 24 h seasonal naive scores about 0.248 at every
  horizon. No cell beats it at any horizon (0/32).
- **Comparability.** NOT_COMPARABLE to the published ECL rows (see the closure
  table).

## Closure table (owner closure rule)

Common to every row:

- **Scale.** z_train (train-only StandardScaler). Validation split, mean over
  2609 windows × 24 horizons × 321 channels.
- **Naive.** Persistence on the same rows: 0.851406.
- **Literature.** NOT_AVAILABLE. No published ECL row exists at L24/H1..24 on a
  validation split. Source: `docs/audits/evidence/ECL_TIMEFILTER_MATCH_20260930.md`.
- **Comparability.** NOT_COMPARABLE. The published rows use L96 with
  H96..720, on the test split.

| Configuration | Model MAE | Skill |
|---|---|---|
| c950ec17 branch+core R1 MAE | 0.375091 | 0.5594 |
| c09f3034 default per-feature R0 MAE | 0.383565 | 0.5495 |
| branch R1 MAE | 0.390565 | 0.5413 |
| 9cc0a1f4 draw3 R0 MAE | 0.391058 | 0.5407 |
| 554ff1d6 draw1 R0 Huber | 0.391705 | 0.5399 |
| 7db1feb6 draw1 R0 MAE | 0.397433 | 0.5332 |
| branch R2 MAE | 0.397631 | 0.5330 |
| e7508efd grouped32 R0 MAE | 0.397583 | 0.5330 |
| de77438d default per-feature R0 Huber | 0.399209 | 0.5311 |
| 409dedf5 grouped32 R0 Huber | 0.400195 | 0.5300 |
| branch+core R2 MAE | 0.401144 | 0.5288 |
| ff759506 draw3 R0 Huber | 0.407179 | 0.5218 |
| 0e5b677a draw2 R0 Huber | 0.421245 | 0.5052 |
| 345d300d draw2 R0 MAE | 0.428812 | 0.4964 |
| 35df52f1 draw0 R0 Huber | 0.730339 | 0.1422 |
| 66108c9c draw0 R0 MAE | 0.733589 | 0.1384 |

The machine-readable source is `R1R2_AND_CLOSURE_corrected_r0_v1.json`.
Per-horizon detail, with persistence and the seasonal naive, is in
`PER_HORIZON_corrected_r0_all24.{json,csv}`.

## R1/R2 table (paired by seed)

**R0 reference.** seq 5 `c09f3034`, mean 0.38356499199182625:

- seed 2021: 0.3884455575
- seed 2022: 0.3786844265
- spread: 0.00976

**Superseded reference.** `de77438d` (Huber δ 1.0), mean 0.3992086764. It is
recorded in queue meta `r1r2_reference`.

**Pretraining cost.** M02 donors, recorded beside each row but not part of the
selection rule: CPU 7,875.48 s, wall 6,636.5 s, peak 6.77 GB.

| Config | Seed 2021 | Seed 2022 | Mean | Δ by seed | Mean Δ | Own spread |
|---|---|---|---|---|---|---|
| branch+core R1 | 0.3750808655 | 0.3751016497 | 0.3750912576 | −0.0133647 / −0.0035828 | −0.0084737 | 0.00002 |
| branch R1 | 0.3891324749 | 0.3919979973 | 0.3905652361 | +0.0006869 / +0.0133136 | +0.0070002 | 0.00287 |
| branch R2 | 0.4040159429 | 0.3912466285 | 0.3976312857 | +0.0155704 / +0.0125622 | +0.0140663 | 0.01277 |
| branch+core R2 | 0.3991280691 | 0.4031605753 | 0.4011443222 | +0.0106825 / +0.0244761 | +0.0175793 | 0.00403 |

Reading of each row:

- **branch+core R1:** strict-minimum improvement, within the R0 spread.
- **branch R1:** worse than R0, within the spread.
- **branch R2:** worse than R0 on both seeds, by more than both spreads.
- **branch+core R2:** worse than R0 on both seeds, by more than both spreads.

Per-horizon skill for the eight R1/R2 cells:

- Against persistence: negative at h1 (−0.41 to −0.55), h23 (−0.04 to −0.12)
  and h24 (−0.50 to −0.65) in all eight.
- Against the seasonal naive: no cell beats it at any horizon.

### Reading rules used

1. **Incumbent.** Strict-minimum on the mean of all declared paired seeds, at
   full precision, with no tolerance band. Only cells verified by exact match
   count.
2. **Pairing.** Contrasts are paired by seed.
3. **"Advantage".** The word is never used unless the gap exceeds both two-seed
   spreads. Even then it is only stated with the numbers, and not as proof.
4. **Pretraining cost.** Shown beside the candidate. It does not enter the
   incumbent rule.

### The 4 refusals (REFUSED_BY_ENGINE, never run)

Cells refused:

- `corrected_default_core_R1_mae_ref_seq5`, seeds 2021 and 2022
- `corrected_default_core_R2_mae_ref_seq5`, seeds 2021 and 2022

Refusal text: "Donor manifest mismatch (features/config/grid/upstream)". A core
donor binds its upstream branch weights, and these rows use fresh R0 branches.
The engine refused them identically at both pins (IDENTITY_PROOF). They count
inside the 36-cell denominator as named refusals.

## Amendments and campaign declaration hashes

| Amendment | CAMPAIGN sha256 (prefix) | Change |
|---|---|---|
| 1 | 12f6030c | Re-pin to df9ae31c (merge of lane A 3ecdb256). Caps from the grouped32 pin pilot. Grouped and draw R0 rows released. |
| 2 | 09f6913e | Candidate budget model with caps and measured calibration. |
| 3 | e979d6c3 | Budget known gaps recorded verbatim. |
| 4 | a534456b | Per-feature pilot measured. Train cap 6580M, verify cap 4950M. worker_a exclusion removed. Per-feature rows released. |
| 4b | 1ccb932b | Verify cap set to 3914M from the measured per-feature verify peak. GPU materialization labelled not measured. |
| 5 | 7ac0eaab | Re-pin to aaee6f94 (engine 6c15af13, identity proof IDENTICAL). Donors bound. 12 Huber-referenced R1/R2 rows superseded. 4 core-only rows refused. |
| 5b | f1af672d | b0eb7ed0 moved to worker_a after the queued request on worker_b was withdrawn. Cap unchanged. |

The final CAMPAIGN sha256 is f1af672ddbf5bdea…. The queue and declaration are
in `~/.local/state/crispdm-data-foundation/m04_doin_20260930/campaign_corrected_r0_v1/`
on the orchestrating host.

## Incidents and deviations

- **D-IDPROOF-STOP-01** (`amendment5/D-IDPROOF-STOP-01.json`, incident
  d-identity-proof-1790833325-1576524-d54eba).
  - The first proof cap was derived from a single-config measurement. The real
    workload thrashed at that cap and was pressure-stopped.
  - The stop also hit M01's child as a bystander.
  - The rerun at 4G measured a peak of 517,517,312 B.
- **ADM-PROC-01** (`amendment5/ADM-PROC-01.json`).
  - Two queued requests, holding no lease, were withdrawn by signalling the
    acquirer, because the admission tool has no cancel verb.
  - Not to be repeated.
- **Q18 backlog** (`amendment5/ADMISSION_BACKLOG_Q18_worker_b.json`).
  - On worker_b, the old admission gate refused 6580M for more than 50 minutes
    while about 17 GB was available, because it charges dead cache against the
    budget.
- **D-OPS-01** (`FINDING_D-OPS-01.json`).
  - The pin worktree was missing on worker_b.
  - The runner now runs a preflight check for it.
- **Bystander runner stop.** The coordinator runner was pressure-stopped as a
  bystander. Since 1dd13e5d the runner adopts the finished remote outcome, so
  nothing was retrained.

## Cost totals for this campaign

Seconds are launcher wall time and include admission waits.

**worker_a (RTX 5090)**

| Kind | Attempts | Seconds | Status |
|---|---|---|---|
| Train | 31 | 7,750 | All completed |
| Verify | 31 | 617 | — |

**worker_b (RTX 4090)**

| Kind | Attempts | Seconds | Status |
|---|---|---|---|
| Train | 1 | 496 | Completed |
| Train | 1 | 2,018 | Queue wait only; withdrawn, nothing ran |
| Verify | 1 | 64 | — |

**Pilots (GPU)**

| Pilot | Host | Seconds |
|---|---|---|
| grouped32, pre-integration | worker_a | 27 |
| grouped32, pin | worker_a | 28 |
| per-feature | worker_a | 169 |

**CPU work, by role**

| Role | Work | Seconds |
|---|---|---|
| worker_b | Staged profiles | about 140 |
| worker_b | Identity proof (all attempts) | about 300 |
| worker_b | Pin replay | about 120 |
| worker_a | Regeneration, diagnostic, tables and unit suites | about 900 |
| Orchestrating host | 512M runner scopes only (no compute) | — |

Earlier lane spend, now old-design evidence: about 6,000 GPU-s (v2 and v3).

## Not done

- **Ablations.** The two horizon ablations are declared in
  `ABLATION_PROPOSALS_HORIZON_2026_10_01.json` but not run.
- **Test split.** Never read. No test numbers exist.
- **Financial evidence.** No financial evidence record exists. ECL is not a
  trading series. The naive-gate format was delivered to M05 but no financial
  record was produced.
- **R2 learning rate.** No learning-rate sweep was run for R2. Fine-tuning at
  lr 1e-3 is the only R2 setting measured.
- **Seeds and intervals.** Two seeds per configuration, so there are no
  uncertainty intervals.
- **Warehouse.** No gov_* warehouse write.

All receipts and attempt directories stay on both workers. Nothing was deleted.
The lane is closed and released from compute until the owner rules on the next
step.

Satoshi, successor technical lead, 2026-10-01.
