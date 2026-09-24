# Specialized-provider review: dispatch improved, acceptance partial

Candidate: M5PHET `7385937`, including `8b78114`. Scope: source, two CPU suite
runs, recording providers through the real dispatcher and temporary evidence.
No real model, broker, live governance or predictor campaign was exercised.
Counterexamples: `../evidence/M5PHET_SPECIALIZED_7385937/probe.py` and results.json.

## Findings, highest priority first

1. **High: calibration can still claim binding without matching evidence.**
   `src/m5phet/runtime.py:309` permits absent task/state identity; a different
   state_digest is ignored. Both return OK/calibration_bound=true. At `:314`
   clocks are compared as strings: calibration ending 2026-09-23T23:30:00-05:00
   is accepted for as_of 2026-09-24T00:00:00Z, although it is 4.5 hours FUTURE.
   These are three separate one-factor cases. Require actual binding fields and
   parsed aware timestamps normalized to UTC, including availability of calibration.

2. **High: provider-swap population and task guarantees are not established.**
   `runtime.py:274-292` copies the requested task into the output binding without
   checking an incompatible task explicitly declared by the loaded state. A
   foreign returned population is ignored. Each isolated mutation still returns
   OK/schema_valid=true. `tests/test_specialized_providers.py` equates the same
   question names with the evaluation population; that does not check rows,
   timestamps, targets or horizons. Separate immutable task/population identity
   from provider/model identity, and validate the returned coverage. A reusable
   foundation model need not be fitted to a single task id, but any supported
   cross-task compatibility must be explicit, not an ignored contradiction.

3. **High: typed output validation still stops at non-null payload.**
   `runtime.py:203-211` accepts a payload containing NaN with OK/schema_valid=true.
   This probe changes only the numeric value from the positive baseline. Complete
   structural and semantic validation is required per output kind; an unsupported
   output schema must not get a successful result. Do not copy Laya's schema
   compiler: use the reviewed upstream subset plus our strict result boundary.

4. **High: resume does not rederive manifest integrity and recovery is bypassable.**
   `evidence.py:262` compares stored digest to requested digest but does not hash
   stored identity. Changing the stored task A -> B and authority to GOVERNED,
   leaving the old digest and governed=false, is accepted by local resume.
   `_rebuild_state()` (`:74`) silently skips malformed lines. With an interior
   broken record, open/close still succeed unless the caller happens to invoke
   recover(). No bytes were erased in this probe, but recovery validation is not
   enforced on the normal path. Recompute integrity and validate profile/authority
   consistency and the event log before accepting appends or a closure.

5. **Medium: classification-specific questions remain a global inference gate.**
   `runtime.py:258` rejects a DECLARED forecasting combination with target,
   horizons and quantile axes because it lacks questions. Do not encode forecasts
   as dummy classification questions. Dispatch to operation/output-kind validation;
   unimplemented families should remain explicitly unsupported. This does not
   claim that the future forecasting engine is already implemented.

## What is accepted at the tested scope

- Explicit supported triples prevent the Cartesian-product counterexample before
  model load. Independently replayed: UNSUPPORTED_TASK, zero provider calls.
- Existing suite demonstrates named selection/no hidden fallback and execution
  of a recording local provider without the blocked LLM packages. This establishes
  runtime dependency/dispatch behavior, not a functioning trained classifier.
- Previous repair tests cover changed requested identity, unknown attempts,
  invalid metric numbers, duplicate metric event ids, class entrypoint loading
  and explicit null output rejection. Keep these improvements and regressions.
- Real Laya/Jev, other engines and installed-provider integration remain pending
  exactly as Satoshi stated. A supported declaration alone is not measured capability.

Independent suites: **115 passed, 1 skipped** both in base Python and trading-stack,
CUDA hidden. The skipped test needs DuckDB. Satoshi's reported 116 may reflect an
environment with it installed; no all-environment pass is claimed by this review.
No new ML measurement here. The user's report says the independent contrast is
at the second seed's AE, 5,257/14,400 CPU seconds; that is a reported snapshot,
not a live observation by Musashi. Do not restart it for these runtime changes.

## Scoped continuation for Satoshi

1. Retain the positive dispatch cases. Freeze these additional cases before code
   changes and test each changed factor separately. Add offset-equivalent clocks,
   missing timezone, future calibration availability, wrong state digest, altered
   sample identity with the same row count, and nonfinite nested output values.
2. Implement per-kind validators in the existing registry path. Bind immutable
   task/population independently of provider, including exact expected coverage.
   A change of provider may change model/output values, not the comparison set.
   Preserve the distinction between request validity and valid returned outputs.
3. Make validated recovery part of open/resume and all writable state transitions;
   compare canonical stored identity/profile and fail on interior corruption.
   Do not confer remote authority through local strings; real governance adapter
   acceptance remains required and optional. Keep the previously assigned
   idempotency, durability and real DOIN round trip work in scope.
4. In PARALLEL, complete one actual Laya vertical slice through the public provider
   API, reusing the upstream engine and reviewed schema subset. Use a pinned SDK
   and checkpoint, direct-SDK versus wrapper comparison and real consumer tests.
   Keep that engine work separate from the small contract repairs, then integrate.
   No requirement to implement all five engines before the first useful slice.
5. Report implemented/engine-integrated/independently-accepted separately. Update
   the method state and return exact failing/passing scenarios, dependencies and
   costs. Finish existing assigned independent work without asking permission at
   each step. No new compute allocation, real-capital orders or blanket retraining.

This is a targeted acceptance continuation, not a new product redesign or a
reason to leave admitted experimental work idle. Existing Laya-reuse, optional
governance/DOIN and external5090-first instructions remain applicable.
