# M5PHET d9ffc4d: repaired cases accepted, three boundary gaps remain

Candidate: `d9ffc4deaf7948d9432597e8c68bcd48e6ceb41c`. Independent review of
source, CPU suite and public APIs with recording providers and temporary logs.
No trained model, live governance, broker or running contrast was exercised.

## Findings

1. **High: a multi-horizon forecast can silently omit a requested horizon.**
   `src/m5phet/runtime.py:153` checks membership of the one returned horizon,
   while `:386` enumerates only targets. In the probe, requesting price at
   horizons `[6, 72]` and returning only price at 6 gives `OK/schema_valid=true`,
   just like the complete single-horizon positive control. Requested row identity
   matches in both cases. Coverage is not just rows or target names: omitting the
   long-horizon forecast must not look like a complete short/long pair. Either
   implement exact declared target/horizon coverage or explicitly reject the
   unsupported multi-horizon shape before loading. Never fabricate the second value.

2. **High: resuming a torn final append accepts writes that corrupt new evidence.**
   `src/m5phet/evidence.py:289` rejects existing interior corruption but `:307`
   permits a malformed last line. `_rebuild_state()` skips it and `_append()`
   (`:103`) appends directly. The probe leaves a truncated final JSON record,
   reopens normally, then starts AND finishes a new attempt successfully. Its
   start is concatenated onto the torn bytes and is no longer a parseable event;
   its finish is parseable. The next open rejects interior corruption. This is
   not merely a missing diagnostic: the supported resume path damages a newly
   acknowledged record. Recover/quarantine the tail under the writer boundary
   before allowing writes, or return a non-writable recovery-required state.

3. **Medium: the previous population counterexample still passes in its original shape.**
   `src/m5phet/runtime.py:380` reads only top-level `population`. The frozen PRE
   used `input_schema.population`; the new tests changed that location. With that
   original shape, a foreign returned row still gives `OK`, no population binding
   and no warning. The top-level equivalent correctly refuses. Select and document
   a canonical location; explicitly migrate or reject the other spelling and
   reject conflicting declarations. No requirement to keep two permanent APIs,
   but silently dropping the declared population does not fix the original case.

These are reproduced API cases, not claims that any stored experimental scores
or running predictor jobs were corrupted by them.

## Accepted at the tested scope

The independent continuation passes the baseline and confirms refusals for an
incompatible loaded task (before inference), foreign top-level rows, missing or
wrong calibration state digest, future calibration after UTC normalization,
nonfinite output, changed manifest identity and interior log corruption on open.
A genuinely earlier offset calibration still binds. An undeclared combination
still refuses before any provider call. A declared single-horizon forecast now
works without classification questions.

The original PRE is unchanged. Its KeyError after the new early refusal is a
limitation of that probe's continuation, not an implementation regression and
not evidence for the unexecuted later cases. The new independent probe isolates
all 15 cases, including the three open defects above.

Evidence: [probe](../evidence/M5PHET_D9FFC4D/probe.py),
[results](../evidence/M5PHET_D9FFC4D/results.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=<candidate>/src python \
  docs/audits/evidence/M5PHET_D9FFC4D/probe.py --output /tmp/m5phet-d9-post.json
# In the candidate checkout:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q -rs
```

Suite independently measured: **138 passed, 1 skipped** (DuckDB absent from this
interpreter). This does not contradict 139 passes in a DuckDB-enabled environment.
The probe freezes observed behavior, including defects; its successful exit is
NOT a declaration that those defects are repaired. Real provider integration
and scientific performance remain separate acceptance layers.

## Continuation for Satoshi

1. Freeze these three cases with independent expected outcomes before editing.
   Keep the previous PRE intact; make the POST runner continue after expected
   refusals and report each case. Preserve the accepted positive cases.
2. Repair exact forecast coverage and canonical population handling. Tests must
   include two targets/two horizons, omission, duplicate/foreign coordinates,
   conflicting population locations, and an explicitly supported complete result.
   If multi-horizon inference is not implemented yet, refuse it explicitly rather
   than claiming it works. Do not redesign the entire provider system for this.
3. Repair interrupted-tail recovery on the normal writable path. Prove a normal
   reopen/start/finish/reopen round trip after an interrupted final append, without
   requiring the application to remember a hidden call. Cover attempts and metrics;
   preserve existing good events and record the recovery disposition.
4. In a separate worktree/lane, CONTINUE the already assigned real Laya slice:
   pinned upstream SDK and checkpoint, direct-SDK versus provider parity, real
   structured output and local trace in a consumer. Do not substitute more mock
   tests for that delivery, and do not reinvent Laya's engine. Integrate after the
   narrow boundary fixes; no need to wait for all five model families.
5. Keep the independent contrast running under its existing admission and budget.
   These M5PHET defects do not invalidate it or justify a restart. Respect external
   5090-first placement without preemption. Continue the existing optional
   governance/DOIN work in dependency order, not as a mandatory local-app stack.
6. Return software acceptance and real-engine acceptance separately, plus actual
   measured experimental progress if available. No invented new metric for this
   CPU audit. The reported second-seed progress at 6,138/14,400 CPU seconds is an
   owner-supplied snapshot, not a live observation from this review.

No additional owner decision, compute allocation or live-capital authorization
is requested by this continuation. Implementation belongs to Satoshi. Existing
orders remain in effect; this is a scoped repair list, not a reset of the plan.
