# Lane B return: DOIN swarm, durable recovery, campaign coordinator

Date 2026-10-11. Orders `ac77366a…` section 2.B and 3; plan `da51723a…` sections 3-5, 7.

## Pins

| Repo | Branch | Base | Tip |
| --- | --- | --- | --- |
| predictor | `satoshi/i6n-swarm-B-doin-20261011` | `5c383d17` | this commit (see `git log`) |
| doin-plugins | `satoshi/i6n-swarm-B-doin-20261011` | origin/master `2654ecf` | `34239c8` |
| doin-node | `satoshi/i6n-swarm-B-runtime-20261011` | origin/master `8bfc64f` | `c411440` |

Interfaces consumed: A.2, C.2, D.3, E.3 (relayed by Satoshi). Published: `INTERFACE_B.json` B.2.

## What was built

A ledger (SQLite, one authority) gives every physical fit a plan-independent
`fit_id`, leases with globally monotonic fencing tokens, an attempt policy and a
durable outbox. Workers reach it through an authenticated HTTP service. The
shared wrapper gained an `i6n_swarm` mode: no model/data imports, no
`default_optimizer` fallback, seed 6102026 and minimization enforced in config,
role hook, bootstrap and reproduction; evaluation goes to a bridge that fans a
candidate out over the plan folds, adopts verified terminals, and runs A's
evaluator once per fold inside `crispdm-run`, renewing the lease, self-fencing
before expiry and guarded against orphaning. Ranking goes only through E's
`score_folds` signature with C's origins digests. The node runtime now has an
absolute improvement tolerance, fail-closed reproduction, identity-bound
results, a role check and D's reference-spool ingestion. The campaign CLI
seals a DAG from pinned A/C/E inputs, installs only its own user units, starts
idempotently, activates successors in a timer tick, reports state/ETA read-only
and publishes through D (candidates without `won`, decisions as events).

## Requirement → test → result

| Requirement | Test(s) | Result |
| --- | --- | --- |
| Real shared plugin path, no fallback, no TF | `test_swarm_mode_loads_only_the_real_i6n_plugin_without_tensorflow`, `test_no_silent_default_optimizer_fallback` (doin-plugins) | PASS |
| Seed 6102026 + minimization in wrapper/role/DEAP/publication | `test_seed_and_direction_must_agree_in_config`, `test_role_validation_hook`, `test_population_requires_campaign_seed…`, `test_reproduction_seed_is_campaign_derived…`, `test_role_refusal_comes_from_the_plugin_hook`, `test_outbox_delivery…` (seed/direction in payload), `test_plan_is_sealed…` | PASS |
| Tolerance 1e-4, patience 3, pop ≤ 8, 3 generations, 80/20, 24 per h | `test_improvement_needs_more_than_the_absolute_tolerance`, plan `_check_search` via `test_plan_is_sealed…`, `test_budget_of_new_candidates_is_enforced_by_identity` | PASS |
| Admitted subprocess, no SIGALRM in node thread, no CPU fallback | `test_launcher_argv_carries_caps_and_label`, `test_runs_inside_a_non_main_thread`, `test_runner_module_does_not_use_alarm_signals`, `test_gpu_device_env_refuses_cpu_fallback`, `test_gpu_node_without_device_refuses…`, `test_evaluation_in_node_thread…` | PASS |
| Disconnect during claim | `test_disconnect_during_claim_fences_old_token` | PASS |
| After fit before ACK | `test_after_fit_before_ack_late_receipt_adopted_and_supersedes`, `test_ack_lost_after_ledger_commit_is_adopted` | PASS |
| During reproduction | `test_node_loss_during_reproduction_recomputes_identically_without_refits`, `test_fail_closed_reproduction_keeps_results_and_stops` | PASS |
| Never repeat a finished fit / never publish another candidate's score | `test_completed_fit_is_adopted_never_repeated`, `test_reevaluation_adopts_every_fold…`, `test_receipt_of_another_candidate_is_rejected`, `test_receipt_for_a_different_config…`, `test_peer_result_for_another_candidate_is_not_recorded`, `test_two_nodes_over_the_lease_service_fit_each_fold_exactly_once` | PASS |
| Recovery: adopt / resume full checkpoint / one linked recovery / ≤3 attempts / OOM no retry | `test_one_recovery_without_checkpoint_then_failed`, `test_complete_checkpoint_resumes…`, `test_incomplete_checkpoint_is_not_a_resume`, `test_terminal_failures_never_retry[*]`, `test_recovery_after_holder_death_links_the_evaluator_status` | PASS |
| Self-fencing and orphan stop | `test_lease_loss_self_fences_child_before_expiry`, `test_unreachable_ledger_self_fences_before_ttl`, `test_guard_stops_child_when_supervisor_dies` | PASS |
| Resume after boot, not from zero | `test_boot_recovery_marks_previous_boot_interrupted`, `test_state_and_fencing_survive_restart`, `test_boot_recovery_interrupts_previous_boot_of_this_host_only`, `test_shared_results_survive_full_process_restart` (node, kept) | PASS |
| Placement (5090 / 4090 health / 4070 reserve / 5070 Ti aggregate, one GPU per host) | `tests/test_i6n_swarm_placement.py` (14), `test_one_gpu_job_per_host_unless_admitted` | PASS |
| CLI plan/preflight/install/start/status/collect | `test_plan_*`, `test_preflight_never_trains…`, `test_install_*`, `test_start_is_idempotent`, `test_status_*`, `test_collect_*`; `evidence_B/CLI_FIXTURE_SMOKE.txt` | PASS |
| Status levels/states/ETA, tests never counted | `test_status_reads_only_and_reports_every_level`, `test_status_cli_does_not_import_tensorflow`, `test_eta_measured_projection_excludes_fixture` | PASS |
| Successor activation, localized failure | `test_successors_activate_automatically_and_failures_stay_local`, `test_annual_completion_activates_multi_output`, `test_unfrozen_folds_become_freeze_units…`, `test_freeze_completion_builds_swarm_context` | PASS |
| Reuse of the 48 by identity | `test_registered_prior_is_adopted_at_first_claim…`, `test_legacy_mapping_with_no_fold_reuse_adopts_nothing` | PASS (E maps all 48 `fold_score_reuse: NONE`: nothing adopted, nothing refit as the same fit) |
| C DCA folds, origins to E, D decisions | `test_declared_clock_assumption_folds…`, `test_c_origins_travel_to_the_aggregator`, `test_production_plan_requires_c_plan_for_origins`, `test_collect_publishes_once…` | PASS |

## Commands and evidence

All CPU, `CUDA_VISIBLE_DEVICES=""`, `crispdm-run -m 2G -t 120s`, one job at a time.
PRE (collection errors before code): `evidence_B/PRE_predictor.txt`, `PRE_doin_plugins.txt`, `PRE_doin_node.txt`.
POST: predictor 59 + 38 passed, 1 skipped (E module absent); doin-plugins 20 passed;
doin-node 76 passed (`evidence_B/POST_*.txt`). CLI smoke on generated fixtures:
`evidence_B/CLI_FIXTURE_SMOKE.txt` (it exposed and I fixed an empty-outbox
`COMPLETE`; the file records both).

## Not done / risks

1. Nothing installed, started or fitted; no real DOIN node or two-node smoke was run (the two-node test uses two bridges over the real HTTP ledger, not two `UnifiedNode`s).
2. E's `score_folds`, C's rollup and D's publisher are not merged here; B tests use a same-signature reference (`origins_binding NOT_CHECKED`) and typed `WAIT_DEPENDENCY`. Integration must rerun with them.
3. Cross-host recovery: A requires the parent STATUS digest, which only the lost host holds; a recovery on another host will be refused by A until A accepts ledger lineage.
4. C's `AnnualAccumulator` takes one `candidate_identity_sha256`, but A's identity includes the week cell: annual rows may be refused (`ROLLUP_REFUSED`, reported).
5. Bindings carry absolute paths: the campaign data root must be identical on all hosts.
6. `multi_output` stays `WAIT_DEPENDENCY` until an arms manifest exists; E's v2 plugin path (`i6n_plan`) is not wired (v1 wire transport).
7. NV_ERR counting needs kernel-log read access; without it the 4090 is refused (conservative).

Next artifact: integration branch merge of A/C/D/E/B, rerun of these suites with
E/C/D modules present, then the two-role smoke (`install`/`start`) on fixtures.
