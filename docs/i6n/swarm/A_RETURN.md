# Lane A return: trajectories, model and training

Branch `satoshi/i6n-swarm-A-trajectory-20261011`, base `5c383d17`. Commits:
- ACK `ab85c6ee`
- INTERFACE A.1 `b9eb6929`
- code `298619e4`
- INTERFACE A.2 `abddeb73`
- this return: the commit that adds this file

## Narrative

The trajectory rejection was removed only after real targets existed. A
trajectory candidate now trains heads [dy, 2dy, ..., h], with dx and dy
independent and in exact UTC elapsed hours. Each head has:
- its own C1 mask, from weekly partitions built with all offsets;
- its own TRAIN-only target center and scale;
- an equal loss weight;
- its own C2-scored naive.

Three things are projected onto the endpoint column, so an endpoint-only
candidate and a trajectory candidate of the same fold bind byte-identical
support digests, rows, counts and naive sums:
- the governing support;
- the feature scaler rows;
- the fitness rows.

Rows whose only valid labels are intermediate still train those heads, but
they never become endpoint rows. A missing intermediate label never drops an
endpoint row. Labels are looked up exactly and selected before residuals. A
True mask without an exact bar raises an error. A head without TRAIN support
gets `NO_TRAIN_SUPPORT`, receives no gradient and is never scored as zero.

Other changes:
- **Model:** a reduction plan rejects any cascade that loses t or empties the
  sequence, and odd cascades keep t.
- **Placement:** `verify_placement` enforces strict device placement
  (variables, attention variables and output) before any op runs. This is
  tested on CPU only, by contract.
- **Freeze:** requires plan section 2 stopping. The context carries the
  experiment and fold labels.
- **Attempts:** a recovery attempt is linked to its parent's terminal status
  digest, with at most three physical attempts. A COMPLETED attempt is always
  adopted.

At Satoshi's and C's request, `freeze --score-outer` seals the outer
weekly-refit week before any fit. It then reports VALIDATION rows (S_h per
head, with its own naive), but never feeds them into fitness or stopping.

Each receipt carries `metadata.candidate_receipt` (`i6n.candidate_receipt.v1`)
with:
- genotype, config and cell digests (E.1 definitions confirmed);
- dx, dy, L, n and final_steps;
- heads, stopping, cost (ru_maxrss of the process), attempt and placement;
- rows per head and split, with sums and counts, naive sums and counts,
  `support_sha256` and C2 receipt digests.

`role` uses C1's MANAGEMENT/ENTRY, with `c2_role` as an alias field.

## Requirement -> test -> result

| Requirement | Test (file::test) | Result |
| --- | --- | --- |
| dx=3, dy=6, h=12 | fit_h12::test_endpoint_and_trajectory_share_endpoint_rows_and_naive; contract::test_independent_dx_dy_map_to_exact_heads | pass |
| dx=36, dy=24, h=72 | fit_h72::test_dx36_dy24_h72_trajectory_fit; contract::test_independent_dx_dy... | pass |
| Identical endpoint pairs (support/rows/naive) | fit_h12::test_endpoint_and_trajectory...; contract::test_endpoint_support_rows_and_naive_identical_and_missing_intermediate | pass |
| Horizon not a divisor | contract::test_non_divisor_horizons_rejected, test_forged_trajectory_fields_rejected | pass |
| DST | contract::test_dst_shifted_weeks_use_exact_utc_elapsed_hours | pass |
| Missing intermediate | contract::test_endpoint_support_rows_and_naive_identical_and_missing_intermediate; fit_h72 | pass |
| Future perturbation | contract::test_future_perturbation_changes_only_the_affected_labels; model::test_future_slots_never_change_earlier_core_steps_multihead | pass |
| Checkpoint/reload, reuse without refit | fit_h12 (reload digest, all-head replay, adoption, tamper) | pass |
| Odd cascade keeps t; t reaches every head | model::test_odd_cascades_keep_origin, test_t_reaches_every_trajectory_head_after_all_reductions, test_reductions_that_lose_t_or_empty_rejected | pass |
| Mask without support | contract::test_head_without_train_support_is_excluded_not_zero_error, test_endpoint_without_train_support_refused; model::test_unsupported_head_gets_no_gradient_no_nan_equal_weights | pass |
| No Flatten; sequence LSTM | model::test_no_flatten_sequence_core_multihead | pass |
| Strict placement (CPU contract) | model::test_strict_placement_contract_on_cpu | pass |
| n=433 buildable | contract (433 mapping); model::test_build_433_samples_trajectory_without_truncation | pass |
| Plan stopping 50/5/1e-6 and restore | contract::test_freeze_requires_plan_section_2_stopping; model::test_plan_patience_five_stops_and_restores_best | pass |
| Attempt lineage | contract::test_attempt_lineage_linked_counted_and_bounded | pass |
| Outer week never in stopping/fitness | contract::test_outer_week_never_reaches_stopping_or_fitness; fit_h72 | pass |

PRE: before implementation the contract suite gave 15 failed, 2 errors and
5 passed; the model suite failed at import.

POST (CPU, `crispdm-run -m 2G -t 120s`, one at a time):

| Suite | Result |
| --- | --- |
| contract | 23 passed (22.8 s) |
| model + temporal_ops | 21 + 52 passed |
| fit_h12 | 1 passed (62 s) |
| fit_h72 | 1 passed (21.5 s) |
| historical_evaluator + search_publish | 18 passed |
| model, attention_mask, temporal_ops | 160 passed, 15 GPU-gated skipped |
| controller, optimizer, search, cell, training | 249 passed, 2 failed (below) |

## Commands

See README_A.md. All runs used:
`CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider <files>`

## Risks and open items

- **Controller test break:**
  `tests/test_i6n_search_controller.py::test_horizon_parameterized_config_and_contexts[12,72]`
  still calls `freeze(base, out)`. It needs `experiment=`/`fold=`. That is not
  lane A's file, and main has been notified.
- **No reuse of the 48 endpoint terminals:** the identity includes code
  digests, and those terminals have v1 contexts and different (2024-week)
  folds. New evaluations can never adopt them. They remain verifiable
  read-only, and `evaluate` refuses to fit v1 contexts.
- **Placement not proven on a GPU:** placement is tested on CPU only. A real
  GPU attention probe remains gated (`I6N_GPU_MASK_PROBE=1`) and was not run.
- **Cost evidence:** `process_peak_rss_bytes` is the process ru_maxrss, not a
  cgroup peak. `materialized_bytes` is an estimate.
- **Domain:** any dy = h/n is accepted; E restricts k to {2, 3, 4}.
- **Feature scaling:** trajectory candidates fit the feature scaler on the
  endpoint TRAIN rows, deliberately, so that their inputs are identical.
- **INNER_FOLD configs:** the structural evaluation week must have eligible
  labels (cell `NO_ELIGIBLE_EVAL`), even though it is never scored.
- **Not done:** no historical fit, GPU training, deployment or TEST access.

## Next artifact

B integrates the evaluator through INTERFACE_A A.2. Then the first real
TRAIN-pilot receipt per class follows, under admission.
