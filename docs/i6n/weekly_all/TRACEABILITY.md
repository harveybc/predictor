# Traceability

| Requirement | Retained executable evidence |
|---|---|
| WA01 | test_trainer_fit_and_layers_are_original_bytes; test_shared_worker_science_unchanged; source_identity |
| WA02 | test_preserve_seven_columns_without_global_intersection; test_real_completed_parent_extension_preserves_seven_heads |
| WA03 | test_only_config_delta_is_five_new_heads; prospective initialization_proof / GPU preflight |
| WA04 | unchanged train.py and original MeanMonitorStopper tests inherited from weekly18 |
| WA05 | five parameterized exact-target/force-flat/maturity tests; retained real-period exact target checks |
| WA06 | original per-head naive/pooling tests; real seven-naive equality; annual supports unsupported per-period heads |
| WA07 | inherited corruption/interruption/resource/mirror gates; successor resource wait does not release |
| WA08 | successor wait, three rejection mutations, dispatch-once and WAIT_GPU tests |

PRE: test collection failed before weekly_all existed. POST: 107 focused
adapter/continuation tests, including fit-free actual retained data, passed;
the additional successor resource-wait test also passed in its focused rerun.
No scientific fit was run by the verification. GPU preflight is intentionally
deferred to the automatic handoff, after the parent has released the GPU.

The existing seven-head campaign's code, timers, models and terminals were not
modified. Only the new worktree's shared operational adapter has configurable
module/report-title constants; its worker, aggregate and metrics functions
are AST-identical to bd21fc2f. The actual trainer and builders are byte-exact.
