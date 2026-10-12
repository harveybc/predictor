# Lane E return (search space, plan, identities)

Branch `satoshi/i6n-swarm-E-search-20261011`, base `5c383d17`. Plan
`af0c3570` (sha `da51723a...`), orders sha `ac77366a...`. Sealed plan
`docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json`, `plan_sha256 606a355567257ff76810cec2dba3cd2952b0b6c3fe4ad592ea568402d8feda3c` (v1.2.0; supersedes v1.0.0 `8d8865f1` and v1.1.0 `e7de41f0`; no fit ran under either).
No fits, no data reads beyond the private legacy controller STATE files, no
GPU, no services, no TEST rows.

## Requirement -> test -> result

| Requirement (orders 2.E / plan 3-4) | Test(s) | Result |
|---|---|---|
| 12 horizons exactly | `test_exactly_twelve_business_horizons` | PASS |
| L = max(24, m*h), m in {1,2,4,6}, dedup | `test_history_is_max_24_m_h_deduplicated`, `test_history_examples` | PASS |
| dx = common integer divisors of h and L incl. 1 | `test_delta_x_is_every_common_integer_divisor_including_one`, typed DELTA_X test | PASS |
| n = 1 + L/dx incl. t; up to 433, no 241 cap | `test_samples_include_t`, `test_max_samples_is_433_not_truncated_to_241` | PASS |
| input identity/difference | `test_input_transform_domain` | PASS |
| endpoint or trajectory dy=h/k, k in {2,3,4}; endpoint alias null | `test_trajectory_*`, `test_endpoint_alias_k1_is_null_not_another_fit` | PASS |
| Conv/Transformer/Conv+Transformer, widths, blocks 0 vs 1-2 | `test_core_entry_and_width_domains`, `test_transformer_blocks_*` | PASS |
| reducers 1-3 non-increasing in {4..128} | `test_reducers_one_to_three_nonincreasing_in_domain` | PASS |
| strides free 1/2 per layer (v1.1 ruling), right-anchored; named patterns as labels; t kept; no emptying | `test_plan_v1_1_free_per_layer_strides_with_named_labels`, `test_reducer_strides_outside_domain_typed`, `test_mixed_strides_that_empty_the_core_still_rejected`, `test_right_anchored_steps_and_t_survives`, `test_reduction_that_empties_temporal_core_is_typed` | PASS |
| adapter none/causal Conv/LSTM width {8..64} | `test_adapter_domain_and_inactive_width` | PASS |
| inactive genes canonicalized; identity covers all active genes | `test_identity_is_stable_and_covers_every_active_gene`, `test_active_and_inactive_genes_recorded_in_effective_config` | PASS |
| infeasible typed with all reasons, never repaired | `test_all_reasons_reported_and_input_never_mutated` + per-gene typed tests | PASS |
| seed Conv8->Conv4 REALLY in initial population + small alternative; Transformer not forced | `test_minimal_seed_conv8_conv4`, `test_initial_population_really_contains_minimal_seed`, `test_create_population_contains_seed_and_alternative`, `test_sampling_does_not_force_transformer` | PASS |
| staged classes S1/S2/S3, earlier genes free, promotion after cost pilot | `test_size_classes_staged`, `test_sampling_reaches_larger_classes_and_keeps_small_genes_free`, `test_open_classes`, `test_full_search_three_generations_budget_and_classes` | PASS |
| model contract + exact evaluator boundary (INTERFACE_A) | `test_model_core_accepted_by_real_model_validator`, `test_every_v2_genotype_maps_to_a_valid_v1_evaluator_genome_exactly` | PASS |
| single-sourced identity; B.1 fit_id derived from E | `test_fit_id_is_b_formula_derived_from_e_identity`, `test_effective_config_sha256_*` | PASS |
| sealed versioned plan, budgets 24/h, 1152 fits, per-class caps, 3 attempts | `test_plan_is_sealed_and_verifiable`, `test_budgets`, `test_search_respects_budgets_and_class_caps`, `test_committed_plan_and_legacy_artifact_match_builder` | PASS |
| four distinct-period folds complete before ranking; sums/counts fitness; fold ids from lane C (C.2 `*_DCA`) one per quarter | `test_fold_fitness_*`, `test_folds_with_identical_support_digests_are_rejected_as_replicas`, `test_evaluate_runs_four_folds_and_averages`, `test_resolve_fold_ids_*`, `test_score_folds_with_c2_ids_and_completeness`, `test_v2_plugin_accepts_c2_fold_ids` | PASS |
| receipts bound to candidate, fold, h, frozen support and C origins (lane A review E-A1) | `test_e_a1_reproducer_mixed_candidates_and_rotated_fold_labels_refused`, `test_e_a1_each_binding_field_typed`, `test_e_a1_missing_candidate_receipt_and_support_digest_typed`, `test_e_a1_endpoint_row_origins_bound_to_lane_c_support`, `test_e_a1_expected_identity_is_required`, `test_v2_plugin_refuses_receipt_of_another_candidate` | PASS (RED first) |
| cache by full identity; failures have dispositions; no duplicate filler | `test_cache_hit_by_full_identity_is_not_a_new_fit`, `test_failed_candidates_have_disposition_not_fitness`, `test_mutation_exhausted_returns_none_not_duplicate` | PASS |
| tolerance 1e-4, patience 3 | `test_patience_and_tolerance` | PASS |
| finalists best + minimal control / small alternative; not "optimal" | `test_finalists_*` | PASS |
| 48 legacy candidates mapped or typed | `test_legacy_*`, `E_LEGACY_MAPPING_v1.json` | 48/48 v1 ids recomputed; 48 MAPPED with their own identity (v1.1); 0 fold-score reuse; 0 refits |
| legacy v1 path unchanged | `tests/test_i6n_search.py`, `test_i6n_optimizer.py`, `test_i6n_search_controller.py` | 163 PASS |

## Commands and evidence

- PRE (RED, modules absent): `docs/i6n/swarm/evidence_E/PRE_pytest.txt` (3 collection errors).
- POST: `CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_search_space_v2.py tests/test_i6n_search_plan.py tests/test_i6n_optimizer_v2.py` -> 177 passed (`evidence_E/POST_pytest_E.txt`); legacy regression 163 passed (`evidence_E/POST_pytest_legacy_regression.txt`).
- v1.1 round: PRE `evidence_E/PRE_pytest_v1_1.txt` (39 failed / 134 passed before the change), POST `evidence_E/POST_pytest_v1_1.txt` (179 passed, same three suites, run once). Legacy suites not re-run: `tools/i6n_search.py` and the v1 plugin path were not touched in this round.
- C.2 fold ids + E-A1 round: PRE `evidence_E/PRE_pytest_c2_folds.txt` (9 failed), intermediate `POST_pytest_c2_folds.txt` (54 passed under an unpublished intermediate seal `7a1abbe3`, superseded within the same round); E-A1 PRE `evidence_E/PRE_pytest_e_a1.txt` (13 failed, incl. lane A's reproducer); POST `evidence_E/POST_pytest_e_a1_c2.txt` (195 passed, the three E suites, once).
- Generation-0 configs generated for all 12 horizons (8 S1 candidates each, seed first), private scratch only.

## Risks / open interpretations

1. **INT-STRIDE RULED** (Satoshi, 2026-10-11): free per-layer 1|2 strides. Plan re-sealed as v1.1 (strict superset; every v1.0.0 identity unchanged); the 18 mixed-stride legacy genotypes now map with their own identity.
2. Accepted as recorded: S1 requires dx>=2; class width counts adapter width; reducers non-increasing from entrance width.
3. A must emit `effective_config_sha256` exactly as E defines (verified by lane A's review); C role vocabulary is settled by A.2/C.2, not by E's plan.
4. Lane A review (`REVIEW_A_OF_E.md` @172caca7): E-A1 FIXED (typed receipt binding in `score_folds`); E-A2 (min 2 final steps is stricter than A's model rule) and E-A3 (h=5 seed L=30) documented in INT-MIN-STEPS / INT-SEED-WINDOW; E-A4 dead branch in `class_of` now an explicit unreachable assertion. `origins_binding` is NOT_CHECKED unless B passes C's origins digests.
5. Measured class caps are not in the sealed plan; they come from B's pilot receipts.

## Review of B and C (interface level only)

B.1 and C.1 are consistent with E on identities, fold ids, fold aggregation,
attempt limits and head offsets. B should call `score_folds` (it enforces
distinct-period fold supports) rather than re-implement. Full code review of
B and C is pending their implementation commits.

## Next artifact

E code review of B/C deliverables when their code lands.
