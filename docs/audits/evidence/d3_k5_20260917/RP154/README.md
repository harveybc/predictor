# PRE and POST for the 7385937 specialized-provider review

`PRE_probe.json` is Musashi's probe run unchanged against `7385937`. Every counterexample reproduced.

`POST_probe_refusal.txt` is the same probe, unedited, against the repaired code. It stops inside its case loop at
`wrong_loaded_task` with `KeyError: 'outputs'`, because that scenario now refuses **before inference** and therefore returns no
outputs for the probe to index. The probe was written to characterise the old behaviour and cannot express a refusal.

Checked separately so the positive case is not lost: the probe's own valid baseline still returns `OK` with its `tone` output;
`wrong_loaded_task` returns `INVALID_INPUT` naming the contradiction; the NaN payload returns `INVALID_INPUT` naming
`payload.probability` as non-finite.

| review finding | repaired behaviour | regressions |
|---|---|---|
| 1, calibration binds without matching evidence | every binding field must be PRESENT and equal, including the state's own digest, and clocks are parsed to UTC before comparison | `test_a_calibration_missing_a_binding_field_does_not_bind` (3 cases), `test_a_calibration_with_another_state_digest_does_not_bind`, `test_a_future_calibration_in_another_timezone_does_not_bind`, `test_a_past_calibration_in_another_timezone_still_binds`, `test_an_unparseable_clock_does_not_bind` |
| 2, task and population guarantees | the requested task is never overwritten, a state fitted elsewhere is a contradiction unless it DECLARES compatibility, and the returned rows must be the requested rows | `test_a_state_declaring_another_task_is_a_contradiction`, `test_a_state_that_declares_cross_task_compatibility_is_accepted_explicitly`, `test_a_returned_population_that_is_not_the_requested_one_refuses`, `test_the_population_identity_is_carried_and_moves_with_the_rows` |
| 3, typed output validation stops at non-null | payloads are validated per output kind: no non-finite number anywhere, a typed question needs a label, quantiles must be the declared levels and monotonic, and an unknown kind gets no contract | `test_a_non_finite_number_anywhere_in_a_payload_is_invalid` (3 cases), `test_a_typed_question_payload_must_carry_a_label`, `test_quantiles_must_be_keyed_and_monotonic` |
| 4, resume integrity and bypassable recovery | the stored digest is RECOMPUTED from the stored identity, an authority that contradicts its profile refuses, and interior corruption is refused when the run is OPENED | `test_a_modified_manifest_is_detected_even_when_the_old_digest_is_kept`, `test_a_manifest_whose_authority_contradicts_its_profile_is_refused`, `test_interior_corruption_is_refused_on_the_normal_path_without_calling_recover` |
| 5, questions as a global gate | the required output schema follows the output kind | `test_a_declared_forecast_without_questions_is_not_rejected_for_lacking_them`, `test_a_classification_request_still_needs_its_questions`, `test_a_forecast_without_its_own_declared_schema_refuses_before_loading` |

The accepted dispatch behaviour from that review was not touched: the Cartesian-product refusal, named selection with no hidden
fallback and the blocked-language-model case keep their own tests.
