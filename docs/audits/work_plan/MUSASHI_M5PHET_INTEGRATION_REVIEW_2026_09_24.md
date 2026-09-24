# Candidate integration review: changes required, no global experiment hold

Reviewed predictor `5c2dc332` and M5PHET `adba176`. Reviewer: Musashi.
Scope: source inspection, actual local/runtime APIs in temporary directories,
exact extracted predictor AST function, recording provider through real runtime
and entry-point discovery. No training, live service inspection, broker request,
original prediction reduction or production mutation in this review.

## Findings

1. **High: resume can misattribute new work and lose unfinished attempts.**
   M5PHET `src/m5phet/evidence.py:167` discards the requested identity on resume.
   A request for task/code B reopens A with A's digest. `_open_attempts` starts
   empty (`:55`) and `recover()` never rebuilds it: a started, unfinished attempt
   becomes `open_attempts_at_close: []` after restart. `recover()` (`:99`) also
   rewrites away an invalid INTERIOR record as if it were a torn final append.
   All three behaviors reproduced. This is not durable, identity-safe recovery.

2. **High: governed authority is inferred from truthiness, including on resume.**
   `evidence.py:144` accepts `{"status": "DENIED"}` as a receipt and creates a
   GOVERNED run. Reopening that run without governed mode or a resolver keeps
   GOVERNED authority. No real data-gov exploit is claimed: INT02/04 are explicitly
   pending. The candidate's public boundary nevertheless contradicts its claimed
   deny-before-work behavior. Define a validated accepted-delivery contract and
   exact resume/profile checks before a real adapter depends on it.

3. **High: the typed runtime certifies outputs and calibration it has not checked.**
   `runtime.py:190` accepts an OK answer with `payload: null`; `run()` returns OK
   and `schema_valid: true`. At `:277`, calibration for another task/state and a
   future calibration end returns OK with `calibration_bound: true`. A request
   without output_schema invokes load AND infer before refusal (`:252`). These
   are behavioral counterexamples through the real entrypoint, not text tests.
   Application eligibility remains false; no trading authorization is inferred.

4. **High: metric/attempt records are not typed or idempotent as claimed.**
   `evidence.py:72,82` accepts completion of an unknown attempt, two identical
   metric appends for that attempt, NaN with status OK and a boolean population.
   Required key presence is not value/domain validation; the two appends have no
   stable metric-event identity for safe replay. Local evidence can therefore
   contain invalid numbers or double-counted observations before any DOIN/store
   integration. Reproduced on temporary records only.

5. **High: predictor still turns missing R2 evidence into a positive finding.**
   `tools/df_ecl_modular.py:654` computes `not v.get(...)` for R2. Exact function
   extraction with all expected cell names, but no R2 change measurement, returns
   `COMPLETE` and `R2_detector_changed_by_its_fit: true`. Adding R0 from an
   unrequested seed still returns COMPLETE. This is a closure defect; it does
   not demonstrate missing evidence in the live run. F2 is only partially fixed.

6. **Medium: conventional class entry points cannot load.**
   `runtime.py:133` treats a provider class with capabilities() as an instance.
   Actual discovery with a class entrypoint returns the missing-self TypeError.
   Isolate discovery transport with a fake EntryPoint, not a replacement loader;
   then prove the repaired path from an installed fixture distribution.

## Additional source-level gaps, not runtime experiments

- `analytics.py:56` groups across split, target, definition/version, status and
  overlapping populations, then exposes sum(population) as population. Preserve
  full metric grain and distinguish summed counts from unique row coverage.
  Its loader also silently skips malformed interior lines. DuckDB was unavailable
  in this review interpreter, so the projection was not executed here.
- Predictor downstream checkpoint persistence and observed optimizer counters
  are now implemented. But `:817-836` only records a failed restored-loss check;
  selected detector identity is not measured after restoration. Require a closure
  gate over the selected checkpoint, finite losses and complete scoring population.
  Do not infer that the real restored checkpoints failed.
- Predictor budget admission still measures wall seconds/step then spends a
  combined CPU/wall allowance (`:732-757`); probes fit before its "nothing was
  fitted" refusal and validation/restoration cost is not separately projected.
  `:840` checks limits after a whole regime. External scope enforcement was not
  audited here. F4's counters improved, but strict pre-work budget compliance has
  not been established by this function. Do not hot-patch the running successor.
- M5PHET candidate predates the optional-governance design extension: merge base
  with main `fd437d6` is `4508b98`. Missing newer docs are branch divergence, not
  evidence that Satoshi deliberately removed them. Reconcile additively.

## Experimental progress, separately scoped

The retained `docs/audits/evidence/d3_k5_20260917/RP145/SOTA_TABLE.L512.md`
at predictor `5c2dc332` reports all 12 cells verified
with exact replays. Normalized author float32 MSE/MAE means:

| Horizon | Replicated | Published in retained table | Same-row persistence |
|---|---|---|---|
| 96 | 0.1259 / 0.2210 | 0.126 / 0.220 | 1.5878 / 0.9455 |
| 192 | 0.1437 / 0.2380 | 0.143 / 0.237 | 1.5962 / 0.9507 |
| 336 | 0.1527 / 0.2513 | 0.153 / 0.252 | 1.6178 / 0.9613 |
| 720 | 0.1790 / 0.2753 | 0.177 / 0.275 | 1.6468 / 0.9754 |
| Four-horizon mean | 0.1503 / 0.2464 | 0.150 / 0.246 | not pooled here |

These are Satoshi's artifact-backed reported measurements, NOT fresh independent
array verification by this review. Our run uses L=512; the paper table searches
L in {192,336,512,720}. Keep that limitation next to any operational agreement;
do not relabel it exact reproduction of the paper's selection protocol. This
is ECL forecasting, not a financial result or evidence for detector pretraining.
The corrected R0/R1/R2 successor was reported running; not inspected live here.

## Evidence and disposition

`../evidence/M5PHET_INTEGRATION_REVIEW_2026_09_24/probe.py` and `results.json`
freeze the counterexamples. Run from the predictor documentation worktree:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<m5phet-candidate>/src python \
  docs/audits/evidence/M5PHET_INTEGRATION_REVIEW_2026_09_24/probe.py \
  --predictor <predictor-checkout> --revision 5c2dc332
```

The probe characterizes PRE behavior; translate each scenario into a rejection or
recovery regression test before changing implementation. An exception after a
correct rejection is not a reason to weaken the regression's expected outcome.
Candidate suite independently rerun: **74 passed, 1 skipped**, skip is missing
DuckDB (`tests/test_evidence.py:146`). No claim of 75 executed tests here.

Disposition: candidate code exists, integration acceptance remains pending.
Keep established classification consumers pinned until integration tests pass.
Correct the scoped code and closure in parallel with valid independent work.
Implementation remains Satoshi's, no new compute budget or production migration.
See [continuation orders](../../handoffs/SATOSHI_M5PHET_INTEGRATION_CONTINUATION_2026_09_24.md).
