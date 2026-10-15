# I6-DMO acceptance and CLI implementation contract

Design gates only. No assertion below is a passing test until evidence exists.
Implement unit tests before governing runtime changes. Use the portable
test-led/data-centric method because this checkout lacks its canonical method
document. Keep the new initiative's method state separate from historic campaigns.

| ID | Required falsification / positive control |
|---|---|
| DMO01 | Six named outputs are 2/3/4/5/6/12 h, with independent weights and one shared backbone. |
| DMO02 | 24-hour span gives 13 exact timestamps; t is present; t+1 perturbation changes no input. |
| DMO03 | Input spacing 2 allows exact h3/h5 targets; hourly origins are not thinned. |
| DMO04 | Per-head mask retains other valid heads; invalid NaN cannot enter loss. |
| DMO05 | Weekend, force-flat, Thursday entry versus management, DST and split boundaries use C1. |
| DMO06 | Future/outer data perturbation leaves preprocessing, early stopping and trained state unchanged. |
| DMO07 | Exactly one validation fit; weekly reports never call fit; two output families are separate units. |
| DMO08 | Each original-price MAE and naive use identical row IDs and physical h; independent manual recomputation agrees. |
| DMO09 | Unequal weekly counts aggregate via sums/counts; absent weeks/horizons are explicit. |
| DMO10 | Same-weight fit/inner monitoring arithmetic, patience, best-weight restore and save/load predictions agree. |
| DMO11 | Completed identity is reused; changed config/data/code cannot adopt it; interrupted units remain visible. |
| DMO12 | Outbox retry/restart duplicates no OLAP facts and never retrains; delivery closes only after exact readback. |
| DMO13 | TEST cannot be opened by short init/run/deploy; final freeze/provenance/pair receipts are required. |
| DMO14 | Masks and chronological TP/SL consumption distinguish reversed hit order; no current-price head filling. |
| DMO15 | Force-flat obtains an actual fill before cutoff; stop() invents none; one position per asset. |
| DMO16 | Trial corpus uses four calendar years and all annual validation origins; no silent sample/epoch shortcuts. |
| DMO17 | No new scientific GPU child on the preferred host while another heavy child holds its host lease. |
| DMO18 | TRAIN-only normalization/loss scales; 19-feature identity; STLFSI4 aliases and unavailable revisions rejected. |
| DMO19 | Status gives real task progress/age, measured ETA or UNMEASURED; stale/failed is not COMPLETE. |
| DMO20 | Static, quarterly, monthly, weekly have distinct identities; full TEST reads cannot influence selection. |

## Required scripts (to implement, not existing runnable commands)

Use package `tools.i6dmo` with small modules `contracts`, `data`, `model`,
`training`, `metrics`, `campaign`, `deployment`, `publication`, `reporting`.
Reuse existing helpers rather than a new distributed optimizer or warehouse.

Required public CLI, all commands supporting `--json` where output is structured:

```bash
python -m tools.i6dmo validate-design --design docs/i6n/direct_multi_output_static/EXPERIMENT.short.json
python -m tools.i6dmo preflight --design <design> --deployment <private-spec>
python -m tools.i6dmo init --design <design> --deployment <private-spec> --root <new-root>
python -m tools.i6dmo deploy --root <new-root> --deployment <private-spec>
python -m tools.i6dmo tick --root <new-root>
python -m tools.i6dmo worker --root <new-root> --host-role <role> --once
python -m tools.i6dmo status --root <new-root> --level campaign --json
python -m tools.i6dmo status --root <new-root> --level unit --unit <id> --json
python -m tools.i6dmo status --root <new-root> --level publication --json
python -m tools.i6dmo close --root <new-root>
```

Contract details:

- `validate-design` is pure: no fit, network, TEST, or credential read.
- `preflight` resolves pinned paths, event/availability clocks, real dataset
  spans, GPU visibility, host/slice memory and disk, units and stopped old
  timers; emits PRECHECK.json, not a fabricated successful launch.
- A private deployment spec maps role -> SSH alias/GPU UUID/data roots/venv/
  warehouse EnvironmentFile. Write neither its values nor secrets into Git.
- `init` resolves design to a sealed effective config and enumerates all
  expected work. No 2025 TEST array is materialized here.
- `deploy` installs bounded user services/timers with exclusive claims and
  reboot recovery; refuse old campaign resurrection. It is idempotent and
  reports actual service and child start. No interactive approval per stage.
- `tick` is finite, atomically advances state and schedules the next eligible
  stage. No waiting chat/agent or endless foreground polling.
- `worker --once` runs exactly one claimed subprocess and persists terminal,
  best checkpoint, timings, costs, per-head masks/counts and prediction lineage.
- `status` is a bounded read of generated state, not new training or a log tail.
- `close` checks complete expected populations, scientific and delivery gates,
  immutable result identities, and verified physical mirrors before closure.

Design long and strategy interfaces during this work, but do not fit the long
model until short closure. The top-level pipeline advances to long without
another chat approval. Hyperparameter search and cadence experiments are not
implicit stages of this fixed-model baseline.

## Traceability and closure

For each ID record code commit, test name, PRE/POST result and evidence digest
in TRACEABILITY.json. A separate reviewer runs temporal negative controls and
manual MAE/naive replay, with no repeated scientific training.

Publish `STATUS.json`, `PRECHECK.json`, `EFFECTIVE_CONFIG.json`, `RESULT.json`,
`PUBLICATION.json`, `TRACEABILITY.json`, `RESULTS_INDEX.md`, `RETURN.md` and
generated M1-M8 plot. Failures and exclusions are first-class metrics. Unit
progress may count epochs/batches; stage progress counts declared terminals.
Keep performance estimates separate from engineering completion percentages.


## Market-time pair addendum (order 50d20d01, ops scope)

The market-time campaign (`mode = MARKET_TIME_PAIR`) runs one finite graph and has **no TEST
stage**: `short.fit -> short.report -> long.fit -> long.report -> pair.eligibility ->
strategy.validation (only on a qualified pair) -> publication/mirror closure`.

| ID | Required falsification / positive control | Test |
|---|---|---|
| MTP01 | The graph is exactly those six units with the declared dependencies, both designs sealed at init, and no unit is TEST-open. | `test_market_time_graph_is_finite_and_has_no_test_unit` |
| MTP02 | The long family is fitted and reported even when the short family is scientifically NOT_ELIGIBLE. | `test_long_family_runs_even_when_short_is_not_eligible` |
| MTP03 | A disqualified pair closes `PAIR_NOT_ELIGIBLE` with reasons; the strategy never runs and burns no attempt. | `test_disqualified_pair_closes_with_a_typed_disposition_and_no_strategy_run` |
| MTP04 | A qualified pair runs the validation strategy. | `test_qualified_pair_runs_the_validation_strategy` |
| MTP05 | No TEST unit is enumerable (`TEST_NOT_ENUMERABLE_IN_MARKET_TIME_PAIR`) or claimable even if injected into the queue. | `test_no_test_unit_can_be_enumerated_or_claimed` |
| MTP06 | An elapsed-grid (DS2/ASOF) terminal is refused for a market-time identity even with matching shapes and horizons. | `test_an_elapsed_grid_terminal_is_never_adopted_by_shape_or_horizon_match` |
| MTP07 | A technical failure is a typed terminal with no retry and no false COMPLETE. | `test_a_technical_failure_is_a_typed_terminal_without_retries` |

All of these are in `tests/i6dmo/test_ops_market_time_pair.py`, and since the market-time review
(`f07305a2`, B.1/K2) they run against the **real sealed designs** `EXPERIMENT.mg2.json` and
`EXPERIMENT.mg12.json` through the real model-lane validator; fakes stand in only for the trainers.

| ID | Required falsification / positive control | Test |
|---|---|---|
| MTP08 | `init` seals exactly the designs the model lane produced (schema, seal digests, rules, no elapsed fields) and refuses an elapsed `i6dmo.design.v1` design for a market-time campaign. | `test_init_accepts_the_real_sealed_designs_and_refuses_an_elapsed_one` |
| MTP09 | A contradictory market-time design (elapsed block, or `elapsed_span_fixed=True`) is a typed refusal at whichever layer sees it first. | `test_a_contradictory_market_time_design_is_refused` |
| MTP10 | The ops-side design gate alone rejects every contradictory or incomplete market-time design, and leaves the elapsed path unchanged. | `test_ops_market_time_design_gate_is_independent_of_the_upstream_validator` |

### Lane coverage this graph depends on (plan section 2)

These rows point at the owning lanes' own tests; the ops suite does not re-certify them.

| Area | Property | Owner | Test |
|---|---|---|---|
| DATA | Typed market design; contradictory elapsed fields refused | data | `test_market_bundle.py::test_market_design_is_typed_and_rejects_contradictory_fields`, `::test_frozen_scope_fields_are_enforced` |
| DATA | Per-example offsets, closure flags, first/last conventions | data | `test_market_bundle.py::test_per_example_offsets_closure_flags_and_first_last_conventions` |
| DATA | Weekend/DST patterns distinct; unplanned gaps not compressed | data | `test_market_grid.py::test_monday_origin_crosses_weekend_with_exact_friday_slots`, `::test_dst_closures_change_elapsed_span`, `test_market_bundle.py::test_unplanned_gaps_are_not_compressed` |
| DATA | Current-`t` inclusion and `t+1` perturbation change nothing | data | `test_market_grid.py::test_current_t_and_t_plus_1_perturbations`, `test_market_bundle.py::test_future_perturbation_changes_nothing` |
| DATA | Exact elapsed targets, C1 masks, split/purge boundaries | data | `test_market_bundle.py::test_targets_are_exact_elapsed_hours_and_masks_are_c1`, `::test_split_and_purge_boundaries` |
| DATA | Same-row per-horizon naive matches an independent replay | data | `test_market_bundle.py::test_same_row_per_horizon_naive_matches_independent_replay` |
| DATA | Sealed identities/exclusions reconcile with support study `d2b6a238` | data | `test_market_bundle.py::test_bundle_seals_identities_timestamps_exclusions_and_support_hashes`, `::test_reconciliation_against_the_support_study_is_retained` |
| DATA | A market bundle is not adoptable as a DS2 bundle; TEST rows refused | data | `test_market_bundle.py::test_market_bundle_is_not_adoptable_as_a_ds2_bundle`, `::test_test_rows_are_refused` |
| MODEL | MG2/MG12 designs sealed with the frozen scope and every bound identity | model | `test_dmo_market_time.py::test_mg2_and_mg12_designs_are_sealed_with_the_frozen_scope`, `::test_seal_binds_every_required_identity` |
| MODEL | Fixed elapsed fields beside `market_time` refused; typed refusals | model | `test_dmo_market_time.py::test_fixed_elapsed_fields_alongside_market_time_are_refused`, `::test_market_time_typed_refusals` |
| MODEL | Encoder uses actual offsets, not slot indices; two context channels | model | `test_dmo_market_time.py::test_encoder_uses_actual_offsets_not_slot_indices_and_appends_two_context_channels` |
| MODEL | Fixed per-batch grid substitution refused | model | `test_dmo_market_time.py::test_encoder_rejects_fixed_grid_substitution_and_bad_shapes` |
| MODEL | Controlled weights prove the real time inputs reach the model | model | `test_dmo_market_time.py::test_controlled_weights_prove_actual_time_inputs_reach_the_model` |
| MODEL | Serialization reloads all inputs including time and closure | model | `test_dmo_market_time.py::test_serialization_reloads_all_inputs_including_time_and_closure` |
| MODEL | No donor compatible by shape alone; DS2 families untouched | model | `test_dmo_market_time.py::test_no_donor_is_compatible_by_shape_alone`, `::test_ds2_and_other_elapsed_families_are_untouched` |
| MODEL | The SEALED effective config builds its network (review B-1 / RB04) | model | `test_market_time_fit_path.py::test_the_sealed_effective_config_builds_its_model` |
| MODEL | The trainer consumes the real market bundle and assembles the three inputs (B-2 / RB03 J1) | model | `test_market_time_fit_path.py::test_trainer_loads_the_real_market_bundle_and_assembles_three_inputs` |
| MODEL | The REAL trainer CLI completes on both sealed designs against real bundles (B-2) | model | `test_market_time_fit_path.py::test_real_cli_end_to_end_on_the_sealed_design_and_real_bundle` |
| MODEL | Ordinal, future or compressed time ladders are CAUGHT typed at the encoder (B-4 / RB02 E2) | model | `test_market_time_fit_path.py::test_encoder_refuses_ordinal_or_impossible_time_inputs`, `::test_encoder_accepts_the_valid_convention_and_stays_numerically_unchanged` |
| MODEL | The same refusal fires at the trainer's input assembly (B-4) | model | `test_market_time_fit_path.py::test_trainer_input_assembly_refuses_ordinal_offsets` |
| MODEL | The spacing bound comes from the design at BOTH seams, so a compressed MG12 ladder is refused at the trainer (RB05 item 1) | model | `test_market_time_fit_path.py::test_trainer_bound_comes_from_the_design_spacing_not_a_fixed_floor`, `::test_an_mg2_valid_ladder_is_refused_under_the_mg12_design` |
| MODEL | Market and elapsed bundles are never adopted across designs; the DS2 fit path is unchanged | model | `test_market_time_fit_path.py::test_a_ds2_bundle_is_refused_under_a_market_design_and_vice_versa`, `::test_ds2_elapsed_path_is_untouched` |

Independent review `f07305a2` reported FIX_REQUIRED with blockers B-1/B-2/B-4 in the model lane and
B-3 (K2) in ops. The ops blocker was fixed at `0884cf9a` (MTP08-MTP10 are its regressions) and the
model lane landed B-1/B-2/B-4 at `3cfcb59d`, which RESEALED both designs; their regressions are the
seven MODEL rows citing `tests/i6dmo/test_market_time_fit_path.py` above, and RB05 item 1 (the
design-derived spacing bound at both seams) left the seals unchanged. The integrated identities
are pinned in `market_time/INTEGRATION_PINS.json` and asserted by MTP08, so a later reseal fails
the suite until the integration pin is updated deliberately.
