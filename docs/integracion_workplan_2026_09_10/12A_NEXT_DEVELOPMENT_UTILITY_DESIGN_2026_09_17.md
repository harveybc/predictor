# 12A — Next DEVELOPMENT utility design, per variable (design only)

> **Superseded by 12B** (`12B_DEVELOPMENT_UTILITY_DESIGN_SUCCESSOR_2026_09_17.md`) after Musashi's O1–O4 review: H_A had no calibration path of its own, the inner protocols were not validated recursively, and the purge ignored the wider history. Instance `386fe033…` is preserved as history and was never executed.

Order O3 (`MUSASHI_D3_N1_N5_REVIEW_AND_O1_O4_2026_09_17.md`). Built by `tools/df_utility_next_design.py`,
validated by `tests/test_df_utility_next_design.py`; a sealed instance over the synthetic bank is
`docs/audits/evidence/d3_k5_20260917/O3_NEXT_DEV_DESIGN.json` (`386fe033…`). **Nothing here runs**:
a governed run needs its own order; no reserve is opened; the pilot is not widened.

## What the pilot said, and did not say

On three development units (`bumps/sinusoid/steps`, white noise SNR 10, n = 2048, seed 11) the
transformed-only branch did not beat the raw lags at horizon 1 under a ridge probe (`DOES_NOT_ADVANCE`
×6 with a derived null bound ≤ α/9; ×3 descriptive for `mad_extremes_trailing`, whose derived bound
exceeds α/9). `DOES_NOT_ADVANCE` is **not** equivalence and says nothing about other domains,
horizons, models or the augmented branch. The nine contrasts share operators and series and are
correlated; they were never nine independent experiments.

## Hypotheses per variable, per operator R

| id | branch A | branch B | question |
|---|---|---|---|
| `H_T` | `raw` (window lags) | `transformed` (R lags) | does R alone predict at least as well as raw lags? |
| `H_A` | `raw_wide` (2·window raw lags) | `augmented` (window raw + window R) | does raw+R beat raw lags of the **same total width**? |

`raw_wide` is the **capacity control**: the augmented branch is never compared against raw alone,
so a gain cannot be a dimensionality gain. `augmented` without `raw_wide` is refused by the validator.

## Frozen, inherited, sealed

* Target `return`, horizon 1, model ridge (λ = 1), window 4, 4 blocks, margin 0, α = 0.05,
  `min_rows_per_block` 30 — **inherited from the sealed pilot protocol** and checked against it:
  a design whose thresholds differ from the inherited ones is refused. No threshold is chosen from
  results seen.
* One **family per (unit, variable)**: 3 operators × 2 hypotheses = 6 contrasts, α_adj = α/6;
  the calibration plan per family is derived: `white_null`, n = 2048, 0.95 bound confidence,
  `n_sims = sims_required_for_zero(α/6, 0.95) = 358`. A family that pools several units is refused.
* **Independent replication**: replication units are disjoint from selection and pilot units
  (bank seeds 12 for selection, 13 for replication, per unit family); each replicate carries its
  own identical family and is **counted** (rule: an operator is proposed for confirmation on a
  variable only if the same hypothesis ADVANCES in the selection family **and** in every
  replicate), never pooled as more tests.
* Eligibility from the verified cells of `d3mech-v3` (`MATRIX.verified.v5.cells.json`).
* Design digest seals the document; any edit after sealing is detected.

## Stages, separated by name

| stage | this design | what it needs |
|---|---|---|
| `FLOW_DIAGNOSTIC` | done (`utilreh-v7`) | — |
| `DEVELOPMENT_SELECTION` | **this** | its own execution order, governed run, successor re-verification |
| `PUBLIC_CONFIRMATION` | out of scope | a reserve opened by order, `adjudicate_holdout` once per identity |
| `FINANCIAL_REVALIDATION` | out of scope | owner's order, financial eligibility, separate design |

The module refuses to build a design for any stage but `DEVELOPMENT_SELECTION`.

## Sealed instance

6 families (3 selection + 3 replication), 6 contrasts each, 358 simulations per family;
estimated cost from the pilot: ~1 s CPU per contrast, calibration ≈ 358/538 of the pilot's
(≈ 65–200 s CPU per operator per family) — bounded, CPU only, one machine.
