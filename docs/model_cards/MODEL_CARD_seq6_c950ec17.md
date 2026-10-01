# Model card: seq 6, `c950ec17` (ECL L24/H1..24, branch+core R1, MAE)

Satoshi, successor technical lead · 2026-10-01 · built from lane D's closed campaign evidence. This card adds no new measurement.

**Status: STRICT_MINIMUM_INCUMBENT_OF_A_CLOSED_VALIDATION_CAMPAIGN.** It is not a demonstrated advantage, not a test result, and not eligible as a strategy input (see Limits).

## Identity

| Item | Value |
|---|---|
| Configuration id | `c950ec17cbfa0f50e9af00a8694f1436d69588a6106de23ab85db99f0651cdbe` |
| Label | `corrected_default_branch_core_R1_mae_ref_seq5` (queue seq 6) |
| Campaign | `d_ecl_l24_h24_corrected_r0_v1`, closed 36/36 terminal (32 VERIFIED, 4 REFUSED_BY_ENGINE). Branch `satoshi/d-corrected-queue-20261001`, return `docs/audits/work_plan/LANE_D_RETURN.md` |
| Engine | Lane A integrated engine (3ecdb256 lineage; campaign re-pinned by amendment 1 to df9ae31c, identity proof in `lane_d_20261001/amendment5/IDENTITY_PROOF.json`) |
| Verification | Each seed rescored from its saved checkpoint in a separate process, with exact bitwise agreement under `TF_DETERMINISTIC_OPS=1` (lane D). The checkpoint hashes are in lane D's queue receipts. |

## Architecture (approved default)

- **Input:** 321 ECL channels, window 24, hourly.
- **Branches:** one per feature, (B,24,1)→(B,24,16) causal Conv1D, no time reduction.
- **Fusion:** concatenated to (B,24,5136).
- **Core:**
  - positional encoding, then a projection to 64;
  - 2 causal Transformer blocks with 4 heads;
  - residual Conv1D stages 24→12→6→6 with channels 32→16→8.
- **Head:** flattens the (6,8) latent to H=24 horizons × 321 targets.
- **Budget** (lane A measurement on the same configuration family): 493,056 bytes per fused row.

**Regimes: branch R1 + core R1.** Both are frozen donors; only the head trains.

## Donors and provenance

- **Donors:** M02's ECL v2 donors, `DONORS_FOR_R1_R2_ecl_v2_s7`: 321 branch AEs and 1 core AE. `DONOR_INDEX.json` sha256 `8a6bb203…`.
- **Provenance amendment 1** (sha256 `2aaba33d…`): every donor declares
  - OPERATIONAL;
  - TRAIN_ONLY on the governed ECL TRAIN resource (input manifest `eca31ec1`, admissible declaration `ca1098ed`, source config `4d5402a0` = R0);
  - reconstruction MEASURED (train_validation).
  
  The derivation follows rule R-AE-TRAINONLY-1 from M02's records. It is countersigned by M04 (byte consistency) and M02 (re-derivation, 869afc6d).
- **Pretraining cost** (M02, shown beside the candidate, not part of its selection): CPU 7,875.48 s, wall 6,636.5 s, peak 6.77 GB.
- **Data:** ECL NPZ built by M04 (`DATA_MANIFEST_ecl_l24_h24_v1.json`): source `7e45845d…`, train `9215b099…`, validation `e3712565…`. The scaler is fit on train only. Validation has 2,609 windows. The test was never read.

## Result (validation, z_train scale; closure-rule table)

| Measure | Value |
|---|---|
| Model MAE, mean of seeds 2021/2022 | 0.375091 (0.3750809 / 0.3751016) |
| Persistence MAE, same rows | 0.851406 |
| Aggregate skill vs persistence | 0.5594 |
| R0 reference (seq 5 `c09f3034`) | 0.383565, two-seed spread 0.00976 |
| Paired gap to R0 | −0.0084737 (−0.01336 / −0.00358). **Within R0's spread: no advantage claimed.** |
| 24 h seasonal naive, same rows | ≈ 0.248 at every horizon |
| Literature | NOT_AVAILABLE / NOT_COMPARABLE (published ECL rows use L96, H96..720, on the test split) |

## Per-horizon skill (mean of the two seeds)

| h | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 20 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vs persistence | **−0.407** | 0.136 | 0.349 | 0.470 | 0.597 | 0.652 | 0.685 | 0.656 | 0.500 | 0.247 | **−0.041** | **−0.504** |
| vs seasonal naive | −0.491 | −0.507 | −0.528 | −0.529 | −0.513 | −0.519 | −0.510 | −0.518 | −0.514 | −0.511 | −0.514 | −0.504 |

Every horizon is in `R1R2_AND_CLOSURE_corrected_r0_v1.json`, under `contrasts[label=…seq5].per_horizon`.

## Controls and limits

- **Seasonal naive.** The model loses to the 24 h seasonal naive at **every** horizon, and so does every cell of the campaign (0/32). The seasonal-residual option at lane A tip 9223391d (`target_residual`) is the design built to address this; it has not been measured yet.
- **Persistence.** Skill vs persistence is negative at h1, h23 and h24. Under the owner's forecast eligibility gate this candidate is **SKIPPED_NOT_BETTER_THAN_NAIVE** for any strategy that consumes those horizons.
- **Selection.** Strict-minimum on two seeds. The gap to R0 is within R0's spread.
- **Scope.** Validation split only; L24/H1..24; all channels in and out. NOT_COMPARABLE to published rows.
- **Representation evidence.** The PS3-R pilot on ETH 4h (a different task) found no default gain from pretrained branch objectives: the raw window is the reference (coordinator ruling). That is consistent with treating R1 here as unproven.
- **Use.** No real-money or paper use is implied. Promotion would need a confirmatory paired comparison that exceeds both spreads, plus a declared test.
