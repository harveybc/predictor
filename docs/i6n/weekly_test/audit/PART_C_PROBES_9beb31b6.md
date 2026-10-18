# Part C: completed probes (Retsu, 2026-10-17)

Order: `ORDEN_SATOSHI_OPERACION.md` part C, after `MUSASHI_REVIEW_5c04fa42.md`. Base `9beb31b6`.
Write set: `tests/i6dmo/test_retsu_prequential_audit.py` and this directory. `tools/`,
`predictor_plugins/` and `PROCEDIMIENTO_CANONICO.md` are byte-identical to the base
(`git diff --stat 9beb31b6 -- tools predictor_plugins` is empty).

Limits honoured: synthetic CPU tests only, `CUDA_VISIBLE_DEVICES=""`, `crispdm-run -q -m 2G -t 120s`,
no pytest cache; no fit, GPU, TEST data, deploy, ssh, systemctl, warehouse or monitoring; the live
worktrees and campaign root were not touched. Every test that substitutes an external boundary
carries `simulated_boundary` in its id and docstring; none is GPU or end-to-end evidence.

## Requirement → test → result

| Req. | Test (`tests/i6dmo/test_retsu_prequential_audit.py`) | What is real / what is simulated | Result |
|---|---|---|---|
| C1 PS06: the REAL initial-weights guard runs; a wrong hash is refused typed; the right one comes from the checkpoint | `test_ps06_real_initial_weights_guard_accepts_only_the_hash_computed_from_the_loaded_checkpoint_simulated_boundary` | Real: `GpuBackend.load` (`keras.models.load_model` of a tiny saved `.keras`, `DONOR_WEIGHTS_CHANGED`), `GpuBackend.update` (`validate_update_span`, `transform`, `WARM_START_WEIGHTS_CHANGED`, save/reload, `SAVED_CHECKPOINT_RELOAD`, `monitor_forward`, `measured_metrics`, `checkpoint_files`). Simulated: the TF/Keras version and GPU-presence checks (patched attributes), and `train.fit_dmo`, replaced by a stub that COMPUTES the initial hash with the real `weights_hash` on the model it is given (never copied from the donor record). Cases: (1) hash computed from the loaded model equals the donor's and equals an independent reload of the archive → accepted, next checkpoint carries the restored hash; (2) stub computes from a freshly built model → `WARM_START_WEIGHTS_CHANGED`, no receipt; (3) donor record declares a hash the archive bytes do not have → `DONOR_WEIGHTS_CHANGED` at load; (4) archive bytes swapped while a stub merely echoes the declared value → `ARTIFACT_DIGEST`, no fit | PASS |
| C2 PS10: pending readback then idempotent retry | `test_ps10_pending_readback_then_idempotent_retry_simulated_boundary` | Real: `weekly_test.tick` (verify, terminal collection, status, outbox SQLite in the temp root, `annual`, `PUBLICATION.json`/`READBACK.json`). Simulated: `run_cmd` (rsync = local copy of the fake gpu_root; ssh/systemctl recorded), `storage_guard`, `W.mirror`, `HttpWarehouse` → `FakeWarehouse(fail_query=5)`. First tick: 5 items POSTED, `pending 5`, no PUBLICATION; second tick: `delivered 5, pending 0, mismatch 0`, warehouse POST count still 5 (readback only), PUBLICATION and READBACK written | PASS |
| C2 PS10: a failed mirror prevents CLOSE | `test_ps10_failed_mirror_prevents_close_and_touches_no_timer_simulated_boundary` | Mirror stub returns `MISMATCH`: state stays `COMPUTE_COMPLETE`, no `CLOSE.json`, zero systemctl calls; a retry tick gives the same answer | PASS |
| C2 PS10: a later closure stops ONLY the campaign's own timers | `test_ps10_later_closure_stops_only_this_campaigns_own_timers_simulated_boundary` | Mirror stub flips to `VERIFIED`: `CLOSE.json{state: CLOSED}`; the injected recorder sees exactly two calls, `ssh <alias> systemctl --user disable --now <NAME>-gpu.timer` and `systemctl --user disable --now <NAME>-tick.timer`, both carrying this campaign's batch name; a further tick on the closed campaign makes no call | PASS |
| C3 PS03/PS04: fake predictor depends on the donor's state; positive control; NPZ bytes vs clock-stripped JSON named explicitly | `test_ps03_current_week_outcomes_never_change_its_sealed_prediction_positive_control[StatefulFake]`, `test_ps04_future_week_outcomes_never_change_an_earlier_refit_or_sealed_prediction[StatefulFake,StatelessFake]`, `test_ps07_inference_scaler_is_the_donors_and_refit_scaler_is_the_periods_train_scaler` | `StatefulFake.predict` adds a term derived from `donor['weights_sha256']`; `update` derives the next checkpoint's weights digest from the exact (donor, FIT, observed) inputs. Helpers `npz_bytes_equal` (sha256 of file bytes) and `json_without_clock_equal` / `tree_json_without_clock` (receipts minus `created_at`/`at`). Positive control: changing week-1 outcomes leaves week-1 `PREDICTED.npz` bytes and `PREDICTION.json` (without clock) equal, changes week-1's score, refit inputs and next checkpoint, and therefore changes week-2's sealed NPZ bytes. Negative: changing week-2 outcomes leaves every week-1 file (bytes / clock-stripped JSON), the week-1 refit inputs and checkpoint equal, and also week-2's own seal equal | PASS |
| C3 PRE/POST | `test_ps03_...positive_control[StatelessFake]` (`xfail(strict=True)`) | PRE: against the old state-free fake the positive-control assertion `not npz_bytes_equal(a/week2, b/week2)` fails (week-2 was invariant by construction of that fake). POST: with `StatefulFake` it passes. Both are kept in the same run so the distinction is executable | XFAIL (as designed) |
| C4 naming | all `*_simulated_boundary` ids + module docstring | `test_ps09_stale_update_receipt..._simulated_boundary`, `test_edge_no_pairs_week..._simulated_boundary`, PS06 and the four PS10 tests | done |

Unchanged from the audit (still passing): PS02 order, PS05 window/maturity refusals, PS07 identity
binds scalers, PS09 crash-after-seal resume, PS01 initial donor selection and refusals, no-pairs week
carried forward and the `FAILED`→`BLOCKED_DEPENDENCY` observation (O1).

## Defect found while building C2 (OPS part B scope; not fixed here, `tools/` untouched)

**`tools/i6dmo/weekly_test.py:183` — the closure tick crashes when any head has zero annual pairs.**
`annual()` pools a head with no pairs over the whole family as `n 0, mae None, naive_mae None`
(`weekly18.pool_metrics`), and the `RESULTS.md` renderer formats `{m['mae']:.12f}` → `TypeError:
unsupported format string passed to NoneType.__format__`. The exception escapes `tick` after
publication succeeded and before `CLOSE.json`/timer stop; `main` would record
`WAIT_OPERATIONAL_RETRY` and every later tick would fail the same way. Reproduction:
`test_ps10_closure_with_a_head_lacking_annual_pairs_must_still_close_simulated_boundary`
(`xfail(strict=True)`, states the required behaviour) and
`test_ps10_observation_head_without_annual_pairs_crashes_the_closure_tick_simulated_boundary`
(records today's behaviour: PUBLICATION delivered, no CLOSE, no timer call). Severity: a real
closure blocker only if a head has no pairs for the entire year (the live campaign's long head 72
already has weekly pairs, so it is unlikely to trigger, but part B's coverage reporting must handle
`None` per head). Suggested fix (part B): render `n/a` for `None` and keep `n 0` explicit.

## Commands (all exit 0 unless stated)

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 120s -n preq-c -- python -m pytest -q -p no:cacheprovider -rxXf tests/i6dmo/test_retsu_prequential_audit.py
  -> 18 passed, 2 xfailed (the PRE control and the part-B defect), 1.49 s
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 120s -n preq-c -- python -m pytest -q -p no:cacheprovider tests/i6dmo/test_prequential_test.py tests/i6dmo/test_weekly_test.py tests/i6dmo/test_weekly_test_data.py tests/i6dmo/test_weekly_test_source.py
  -> 24 passed, 1.90 s (unchanged procedure tests)
```

Iterations before the final run (fixture fixes only, no scope change): run 1 — the PS10 fixture
produced a zero naive (target equal to origin) and the one-head Keras model broke
`monitor_forward`'s list-output protocol; run 2 — the report builder needs `cell.slots`; run 3
exposed the part-B defect above, which was then isolated into its own strict-xfail test.

Not covered here, by order: real GPU fits, real runtime versions, real rsync/ssh/systemctl, the
real warehouse and mirrors, and the live annual close. PS06/PS10 therefore remain "CPU seam tests
with mocked external boundaries", as the review required, not GPU/E2E evidence.
