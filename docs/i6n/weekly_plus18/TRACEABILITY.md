# Requirement To Evidence

| Requirement | Verification | Evidence |
|---|---|---|
| W18-01 | Source equality, horizon-only model delta, original stopper | test_scientific_sources_unchanged_except_catalogue; test_model_delta_only_horizons; test_original_monitor_is_not_any_head |
| W18-02 | Projection equality, six mutation failures, real original metadata and scored arrays | test_append_changes_only_target_and_mask_and_new_naive; test_projection_detects_mutation; prepare; preflight |
| W18-03 | Exact timestamp, missing target, strict force-flat and maturity | test_h18_force_flat_missing_target_and_train_maturity; label18 |
| W18-04 | Separate masked naive and pooled sums/counts | test_naive_per_head_masks_and_scale; test_pool_uses_sums_not_mean_weekly_percent; annual |
| W18-05 | Parent receipt, immutable preparation, weights and source custody | source_identity; initialization_proof; CONTRAST_PROOF.json; CONFIG.json; SCALES.json; TRAINING.json |
| W18-06 | Original GPU class/runtime, actual free memory, single shared-host heavy job, admitted cap | preflight; run; launcher service |
| W18-07 | Altered artifact refusal, interrupted fit terminal, outbox/readback and bounded hash-verified mirrors | test_terminal_rejects_changed_artifact; run; tick; mirror |
| W18-08 | Finite 53-period queue, status and measured ETA, own-timer shutdown | launch; tick; status |

PRE: missing weekly18 module caused collection failure before implementation.
POST: 92 focused tests passed (15 adapter tests plus existing continuation and
model tests). Real preparation: 53 periods, 157 hashed files, 74.8 MiB.
GPU initialization and first-fit acceptance are separate prospective gates.
No fit or TEST read is part of the CPU verification.
