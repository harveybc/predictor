# 12C — What the utility work has tested, what it has not, and the next bounded experiment

Order Q3 (`MUSASHI_UTILITY_P1_P4_REVIEW_AND_Q1_Q4_2026_09_18.md`). Updated 2026-09-18. This map
enumerates evidence; negative results are not a licence for arbitrary further search, and none of
the results below diagnoses the data as wrong or the preprocessing as useless.

## Tested (evidence in `docs/audits/evidence/d3_k5_20260917/`)

| area | what was measured | evidence | reading |
|---|---|---|---|
| Mechanics of 9 operators on the synthetic bank (D3) | causal contract, probe responses, twin sensitivity, resources — per (unit, variable, operator) cell | `d3mech-v3` verified matrix (`MATRIX.verified.v5.cells.json`, 6 390 cells) | eligibility only; says nothing about utility |
| Utility, 3 operators (`cusum_causal`, `delta_run_length`, `mad_extremes_trailing`), H_T raw vs transformed | descriptive pilot, 3 bank units seed 11, h = 1, ridge, window 4, 4 blocks, MAE on return, white null 538 sims | `utilpilot-v2` (`P4_…`, `O1_UTILPILOT_V2_REVERIFY.json`) | 6 DOES_NOT_ADVANCE, 3 descriptive |
| Utility, same 3 operators, H_T and H_A (raw+R vs raw_wide) | development selection + mapped replication, 6 families (seeds 12/13), 36 contracts × 358 sims, 36 contrasts | `utildev-v1` (`Q1_UTILDEV_V1_CLOSE_2.json`, `P4_utildev_v1_families/*`) | 18 DOES_NOT_ADVANCE under a supporting contract, 18 descriptive; no pair proposed |
| Instrument controls | null contrast, information loss, future leak: separated 6/6; positive H_T 4/6 advances (deltas positive 6/6, interval crosses 0 twice), positive H_A undecidable under a 28-sim fixture calibration | `Q3_CONTROLS.json` | the instrument's sensitivity at n = 1 200 / 4 blocks is below the predeclared 5/6; see next experiment |
| Calibration reuse | canonical computation key, verified cache: miss then hit with zero simulations and identical bytes; 36 conserved records = 6 computations × 6 | `Q2_*` | saving is measured only in the rehearsal (≈ 6.4 s CPU on 12 sims); the campaign's avoidable CPU (≈ 4 801 s) is a replay projection |

Scope of everything above: synthetic bank, family `bumps/sinusoid/steps`, white perturbation, SNR 10,
no missingness, n = 2 048, one variable, target `return`, horizon 1, ridge λ = 1, window 4,
4 walk-forward blocks, margin 0, α = 0.05 per family.

## Not tested (pending; each needs its own bounded design and order)

* The other 6 mechanically accepted operator kinds (`uniform_decile_quantizer`, `sax_paa_trailing`,
  `stft_trailing`, `wavelet_trailing`, `butterworth_causal`, `variance_regime_trailing`) and any
  operator parameters other than the sealed defaults.
* Other bank regimes: SNR 0 and −5, missingness, other perturbations, other lengths, multivariate units.
* Horizons > 1, the `direction` target with the logistic probe, other windows/blocks, other losses.
* Feature engineering of the lake's variables, per-feature denoising, compression, learned
  representations, feature selection over the lake — none of it is touched by this front.
* Financial data of any kind (needs the owner's order and financial eligibility), public
  confirmation (reserve), financial revalidation.

## Next bounded experiment — proposal only, not launched

1. **Instrument first** (fabricated fixtures, ≤ 30 min CPU): re-run the positive controls with a
   predeclared power design instead of a tuned one — n = 2 048 (the bank's length), 4 blocks, 12
   replicates, criterion ≥ 10/12 — and calibrate the H_A pair with the same predeclared plan as
   the campaign (358 sims at 0.95) so the control is decidable; report power as measured. If the
   instrument still misses the criterion, the next selection is not launched.
2. **Then** one selection + mapped replication over the 6 remaining operators on the same 6
   families (12B design, 72 contracts, 72 contracts of calibration = 12 computations under the
   cache; projected ≈ 2 500 s CPU with the cache, measured by a cost pilot before launch), same
   inherited thresholds, same map, same closure.
3. Nothing financial, no reserve, no GPU.
