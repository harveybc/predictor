# FS4 weekly campaign, stage-2 rule (predeclared, sealed in the plan before any VALIDATION week is read)

Authority: plan section 4 of `FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md` and the coordinator decision of
2026-10-07. This file is hashed (SHA-256 of its bytes) and the digest is written into `WEEKLY_PLAN.json`
(`stage2_rule_sha256`) when `tools/fs4_weekly_campaign.py init` seals the plan. The controller applies the rule
mechanically in `tools/fs4_weekly_wrapper.py::stage2_set_ids`; nobody edits the list by hand.

## Stages

1. **Stage 1** runs the RAW input mode (the R0 temporal predictor on raw standardised 24-row windows) for EVERY set of
   the sealed frontier (`FRONTIER_SEAL_EXT.json`) on EVERY eligible VALIDATION-2024 week, under
   `BUSINESS_WEEKLY_WALK_FORWARD` / `FULL_RETRAIN_ROLLING_4Y`. The winner per population and target is chosen only from
   stage 1, by the aggregate rule and tie rule in the plan.
2. **Stage 2** runs the encoder input modes `RANDOM_ENCODER` (control) and `TRAINED_ENCODER` on a subset of the stage-1
   sets, on the same weeks, with the same predictor, budget and seed.

## The stage-2 list (computed by the controller once every stage-1 task is terminal)

Inputs: the stage-1 RAW weekly aggregates exactly as written to `WEEKLY_SELECTION` aggregates (mean weekly
`skill_mae` over ALL sealed weeks), nothing else.

For every population and target, and for every method family `m` in
{SPEARMAN_CLUSTER, MRMR, JMI, MRMR_CAUSAL, JMI_CAUSAL, UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K, ALL_ADMISSIBLE}:

1. the candidates are the frontier sets of that target whose `methods` contain `m` and that are ELIGIBLE at stage 1
   (every sealed week COMPLETED);
2. order them by the stage-1 order: mean weekly skill descending, then fewer features, then lower total fit seconds,
   then `set_id`;
3. keep the first 3.

Then add every `ALL_ADMISSIBLE` set of the frontier, eligible or not (it is the reference arm). A control-bearing set
(a set whose methods include UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K or ALL_ADMISSIBLE) is therefore in stage 2
exactly when it is in the top 3 of one of its families or is ALL_ADMISSIBLE. The stage-2 list is the union by `set_id`;
it is written to `STAGE2_LIST.json` with its digest and the digest of the stage-1 aggregates it came from.
A set that fails stage 1 on any week is not eligible for a top-3 place and stays in the stage-1 denominator.

## Encoder alignment (certified before stage 2 may run; owner order 2026-10-07)

What each alignment reads, derived from the layers and checked by perturbing the real Keras models
(`tools/fs4_encoder_alignment.py`, `tests/test_fs4_temporal_predictor.py`). A window has 24 hourly grid rows, position 23 is the
origin row. The runner (feature-extractor @deffa53, pinned and unchanged) trains the strided causal encoder 24 -> 12 -> 6 at the
EVEN phase (output j of a stride-2 layer reads rows 2j-2 .. 2j):

| latent step | 0 | 1 | 2 | 3 | 4 | 5 (last) |
|---|---|---|---|---|---|---|
| EVEN_AS_TRAINED, last row read | 0 | 4 | 8 | 12 | 16 | **20** |
| ODD_PHASE_ORIGIN_COVERING, last row read | 3 | 7 | 11 | 15 | 19 | 23 |

- **Chosen: `EVEN_AS_TRAINED`.** The frozen encoder is the runner's own network applied to a window that ends at the origin, so
  the weights are used at exactly the phase they were trained at. No latent step reads a row after the origin (no future leak), and
  the LAST latent step reads rows up to origin-3: a lag of **3 hours** on the hourly grid, for every population (ETH bars are
  4 hours apart but the encoder grid is hourly, so the lag is 3 grid hours there as well). The origin row, origin-1 and origin-2 do
  not reach the last latent step (checked through the bank end to end).
- **Not chosen: `ODD_PHASE_ORIGIN_COVERING`.** Its last step reads the origin row (lag 0), but it applies the same weights at an
  untrained phase. It is never certified, whatever its reconstruction score. A TRAINED origin-covering encoder needs retraining and
  is a separate versioned arm (`docs/fs4/ENCODER_ORIGIN_COVERING_ARM_V2.md`) that the owner decides.
- **The predictor's own branches** (RAW input) are trained from scratch inside each weekly fit, with their strided layers sampled so
  the last step covers the origin; no pretrained weights are applied at another phase there.
- **Comparison caveat that every encoder report carries:** RAW sees the origin row, encoder arms see rows <= origin-3. The
  TRAINED-minus-RAW difference is therefore conservative for the encoder, most of all at the 1-6 hour targets.
- **Gate:** the plan carries `encoder_alignment` (certificate digest, alignment name, lag). `stage2` and the encoder `run-task` refuse
  the encoder arms unless the status is CERTIFIED for this alignment and the certificate was issued for the digest of THIS file.
  RAW is unaffected.

## Guarantees

- The encoder arms CANNOT change the stage-1 winner choice. `close` selects winners from the RAW arm only; encoder
  aggregates are reported as the raw-versus-encoder comparison (TRAINED minus RAW and TRAINED minus RANDOM, per set,
  on identical weekly rows), with RANDOM_ENCODER as the control. Neither a reconstruction score nor an encoder skill
  chooses a feature set (FS4-10).
- The encoder weights are the phase-4 runner's retained terminals for TRAIN fold `inner_2023` (fitted on TRAIN rows
  before 2023 only), frozen for all weeks; they are verified against the digests the runner recorded. A member with
  no such terminal (for example NOT_AVAILABLE_FOR_TRAIN) makes that set/week a FAILED disposition with the reason; it
  is never dropped from the denominator and never replaced.
- All stage-2 tasks must be terminal before `close`. TEST stays sealed until `freeze` and is opened once.
