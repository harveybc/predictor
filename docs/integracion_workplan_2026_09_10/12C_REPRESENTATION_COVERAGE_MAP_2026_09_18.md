# 12C — What the utility work has tested, what it has not, and the next bounded experiment

Orders Q3 and R3/R4 (`MUSASHI_UTILITY_Q1_Q4_REVIEW_AND_R1_R4_2026_09_18.md`). Updated 2026-09-18 (R4). This map
enumerates evidence; negative results are not a licence for arbitrary further search, and none of
the results below diagnoses the data as wrong or the preprocessing as useless.

## Tested (evidence in `docs/audits/evidence/d3_k5_20260917/`)

| area | what was measured | evidence | reading |
|---|---|---|---|
| Mechanics of 9 operators on the synthetic bank (D3) | causal contract, probe responses, twin sensitivity, resources — per (unit, variable, operator) cell | `d3mech-v3` verified matrix (`MATRIX.verified.v5.cells.json`, 6 390 cells) | eligibility only; says nothing about utility |
| Utility, 3 operators (`cusum_causal`, `delta_run_length`, `mad_extremes_trailing`), H_T raw vs transformed | descriptive pilot, 3 bank units seed 11, h = 1, ridge, window 4, 4 blocks, MAE on return, white null 538 sims | `utilpilot-v2` (`P4_…`, `O1_UTILPILOT_V2_REVERIFY.json`) | 6 DOES_NOT_ADVANCE, 3 descriptive |
| Utility, same 3 operators, H_T and H_A (raw+R vs raw_wide) | development selection + mapped replication, 6 families (seeds 12/13), 36 contracts × 358 sims, 36 contrasts | `utildev-v1` (`Q1_UTILDEV_V1_CLOSE_2.json`, `P4_utildev_v1_families/*`) | 18 DOES_NOT_ADVANCE under a supporting contract, 18 descriptive; no pair proposed |
| Instrument controls, fixture (Q3) | null, information loss, future leak separated 6/6; positive H_T 4/6; positive H_A undecidable under a 28-sim fixture calibration | `Q3_CONTROLS.json` | superseded by the governed validation below |
| **Instrument validation, governed (R3, `utilinst-v1`)** | sealed design (`R3_UTILINST_V1_DESIGN.json`): n = 2 048, 12 fresh seeds (200–211), α/2 = 0.025 at 0.95 → 119 simulations per pair, criteria ≥ 10/12 positives, ≤ 1 advance in 12 under the null, ≥ 11/12 loss, 12/12 refusals. Observed: **positive H_T 10/12** (CI95 0.52–0.98) MET; **null 12/12** MET (0 advances); **information loss 12/12** MET (deltas −0.06 … −0.14); **future leak 12/12** refused MET; **positive H_A INCONCLUSIVE**: its real calibration gave 1/119 advances (derived bound 0.039 > 0.025) so no decision was possible — deltas +0.08 … +0.13 in 12/12, descriptive only | `R3_UTILINST_V1_{REPORT,DESIGN,REVERIFY,CONTENT_CHECK,CALIBRATION_CONTENT_CHECK}.json` | **instrument outcome: INCONCLUSIVE** (H_T side PASS-like, H_A side undecided by calibration, not by the controls); 203 s CPU; envelope `722ab3bf…`; nothing tuned, no re-run |
| Calibration reuse | canonical computation key **bound to the executing code** (operator code objects, defining and helper modules, harness, numeric environment; v4 records), verified cache with typed recovery, conflict disposition and idempotent accounting; miss → deliberate incomplete entry → recovery + hits, content-equal in the cube; 36 conserved records = 6 computations × 6 (legacy v2, never re-keyed) | `Q2_*`, `R2_*`, `R1_R2_PRE_POST.txt` | savings are projections from producers' recorded cost; verification cost is measured; numeric portability scope: same environment only |

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

## Instrument status (R3) and what it means for the next stage

The H_T side of the instrument met every predeclared criterion at n = 2 048 with 12 replicates
(10/12 positives, 0/12 null advances, 12/12 loss, 12/12 refusals). The H_A side is
**INCONCLUSIVE by calibration**: one false advance in 119 null simulations leaves the derived
bound above α/2, so no H_A control could be decided; the 12 positive deltas are descriptive.
Diagnosis from completed evidence, not a re-run: a plan that only supports at zero advances
(119 simulations) has no slack — a plan tolerating one advance needs `CP(1, n, 0.95) ≤ 0.025`,
i.e. n ≥ 190 simulations, sealed before any new outcome. Scientific progression to the six
remaining operators is **not** opened by this order; their design is delivered below.

## Next bounded experiment — proposal only, not launched

1. **Instrument successor** (design only): same generators, effects, n = 2 048, 12 fresh seeds
   (≥ 300), both pairs, criteria unchanged; calibration plan with slack for one advance
   (n_sims = 190 at 0.95, α/2) so a single false advance under the null does not void the H_A
   decision; verified equivalent records reused only through the code-bound cache. To be run
   under its own order; an H_A side still INCONCLUSIVE would stop progression again.
2. **Six remaining operators, executable bounded design** (`R3_NEXT_STAGE_DESIGN_6OPS.json`,
   `366880…`): `uniform_decile_quantizer`, `sax_paa_trailing`, `stft_trailing`, `wavelet_trailing`,
   `butterworth_causal`, `variance_regime_trailing` on the same 6 families with the same map,
   thresholds and closure; eligibility per operator taken from the verified cells of `d3mech-v3`
   (every cell MECHANICALLY_ACCEPTED, or the design refuses); 12 contrasts per family → α/12,
   718 simulations per contract; 72 contracts of calibration = **12 computations** under the
   cache (one per operator × pair). **Limits**: resource — a cost pilot before launch, projection
   under an explicit aggregate ceiling (indicative: 12 × 718 × 0.4–1.0 s ≈ 3 400–8 600 s CPU for
   the calibrations plus 72 contrasts × ~2 s; does not fit a 2 h ceiling without the cache and
   needs ≈ 2.5 h with it); sensitivity — the positive generator validates none of these six
   representations, so each needs its own aligned positive control before its results decide
   anything; a `wavelet_trailing` twin without emissions is INSUFFICIENT by design (07C).
   **Not launched by this order.**
3. Nothing financial, no reserve, no GPU.
