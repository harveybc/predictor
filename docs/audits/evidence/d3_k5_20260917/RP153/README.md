# PRE and POST for Musashi's integration-review counterexamples

`PRE_probe_output.json` is the review's own probe, run unchanged against predictor `5c2dc332` and M5PHET `adba176`. All twelve
scenarios reproduced.

`POST_probe_refusal.txt` is the same probe, unchanged, against the repaired code. It stops at its first scenario with

```
ContractError: run 'local' was opened under identity d3742fb3217e and the request asks for f36d68a59bfe:
a changed task, code, model, data or calibration identity requires a NEW run
```

That is the repair working: the probe was written to characterise the old behaviour and records a value where the API now
refuses. As the review says, an exception after a correct rejection is not a reason to weaken the expectation, so the probe was
not edited and the scenarios were translated into regressions instead.

| review scenario | repaired behaviour | regression |
|---|---|---|
| `resume_changed_identity` | a changed identity requires a NEW run | `test_resume_with_a_changed_identity_is_refused` |
| `resume_loses_pending_attempt` | open attempts are rebuilt from durable events | `test_resume_rebuilds_the_attempts_that_were_left_open` |
| `recovery_erases_interior_corruption` | interior corruption is quarantined and refused; only a torn tail is dropped | `test_interior_corruption_is_rejected_and_preserved_not_rewritten`, `test_only_an_identified_torn_trailing_append_is_recovered` |
| `denied_receipt_accepted` | an accepted delivery is a validated contract | `test_a_denied_receipt_is_not_an_accepted_delivery`, `test_an_incomplete_receipt_is_refused` |
| `governed_resume_without_resolver` | a profile is not changed by resuming, in either direction | `test_a_governed_run_cannot_be_resumed_as_a_local_one`, `test_a_local_run_cannot_be_resumed_into_governed_authority` |
| `invalid_duplicate_metrics` | unknown attempts, non-finite or boolean values, boolean populations and contradictory finishes all refuse; events have stable identities and are idempotent | `test_finishing_an_unknown_attempt_is_refused`, `test_an_attempt_cannot_be_finished_twice_with_a_different_status`, `test_an_invalid_metric_value_or_population_is_refused`, `test_a_metric_event_has_a_stable_identity_and_is_idempotent`, `test_a_metric_for_a_foreign_attempt_is_refused` |
| `empty_payload_success` | an OK answer without a payload is invalid | `test_an_ok_answer_without_a_payload_is_invalid` |
| `invalid_request_does_work` | an unanswerable request refuses before load and infer | `test_a_request_without_questions_refuses_before_load_and_infer` |
| `foreign_future_calibration_bound` | a foreign task, a foreign state or a future clock does not bind | `test_a_foreign_or_future_calibration_does_not_bind` |
| `class_entrypoint` | a class entry point is instantiated | `test_a_class_entry_point_is_instantiated` |
| `missing_R2_evidence` | a missing measurement is None, never a positive finding | `test_F5_missing_R2_change_evidence_is_never_a_positive_finding`, `test_F5_an_untyped_detector_flag_is_not_evidence` |
| `extra_regime_cell` | a cell nobody asked for is an error, not extra support | `test_F5_an_unexpected_cell_cannot_expand_the_denominator` |

One behaviour the review did not name but the repair adds: a provider exception keeps its own class rather than being renamed
to resource exhaustion, which is `test_a_provider_exception_keeps_its_class`.
