# RAW Pair Hook

Code-only integration contract, 2026-10-15. Independent actual-path review is
required before deployment. No fits, GPU jobs, services, TEST access or campaign
launches were performed for this change. Existing files were not edited.

## API

`tools.i6dmo.raw_pair.run_unit(spec, out_dir, progress_path)` returns an
`i6dmo.terminal.v1` child terminal. The parent routes only `raw.pair`, an
`evaluation` unit after the `MATCHED_LONG_CONTRAST` comparison, to this hook.
The hook makes one finite pass and never polls or starts another agent.

Required inputs:

- `spec.upstream.comparison`: stored-terminal `{path, sha256}`.
- `spec.upstream.mg1l.report` and `mg12m.report`: stored-terminal references.
- `spec.effective_config.raw_pair.short_report`: private immutable pin
  `{path: <absolute TERMINAL.json>, sha256: <file digest>, unit_id: mg2m.report}`.
- `spec.effective_config.raw_pair.policy`: frozen
  `{min_valid_long_heads: 2, min_valid_short_heads: 1}`.
- The long effective config's `code_commit` and
  `data_pins.common_population.population_sha256`.
- `spec.test_open` must be false.

For an eligible pair, also freeze these before launch:

- `raw_pair.strategy_bars` and `raw_pair.strategy_calendar`, each
  `{path: <absolute file>, sha256}` or `{relpath, sha256}` under `spec.data_root`.
- `raw_pair.strategy_execution`: the existing versioned configuration with
  `instrument`, or explicit `instrument_file`. No default file is substituted.
- `effective_config.design.splits.validation`: explicit `start` and `end`, with
  `start < end <= 2025-01-01T00:00:00Z`.
- The existing strategy adapter's pinned policy blob must be locally resolvable.

The retained recovery effective config inspected during implementation had
`strategy_execution.instrument` and `source`, but **no** `strategy_bars` or
`strategy_calendar` pins. The parent must resolve these private objects before
freezing the new effective identity. This hook never infers their paths.

## Authentication And Decisions

The stored-terminal and CHILD envelopes are checked with the existing strategy
verifier. Identity components must reproduce their original folder digest;
long reports/comparison must match the current code pin. The retained short's
original code, fit identity, matched metadata and lineage remain unchanged.
`matched_runner._authenticated_report` authenticates each METRICS file.

Each METRICS head must agree with its child eligibility row, declared counts,
row-support digest, retained manual pooled replay and own naive error. The
short horizon set is exactly 2/3/4/5/6/12, and the long set 18/24/36/48/60/72.
The official comparison must authenticate both long input records, complete
scores and per-head support, with no contradictory selection or schema. Its
selected arm, not a new ranking or fit, is used for the pair.

The existing `strategy_adapter.pair_eligibility` is the head gate. Requiring six
reported fixed horizons is not requiring six eligible heads: the frozen policy
requires only two long and one short. No short-vs-long naive substitution is
made, and no relative error is compared across their different populations.

For a negative pair, `ELIGIBILITY.json` retains all per-head decisions and
original inputs. The terminal includes a governed OLAP decision report and
returns `COMPLETE / PAIR_NOT_ELIGIBLE`, with matching `gates.pair_status`.
No OHLC, calendar, prediction arrays, strategy policy blob or TEST is read.
Prediction digest declarations are checked across METRICS, child artifacts
and lineage; **prediction bytes are not authenticated on this negative path**.

For an eligible pair, prediction bytes are hashed and replayed using the
existing manual scorer; per-head row digests are also checked. The existing
adapter validates time bounds, loads bars/calendar, derives the versioned
execution receipt and calls `run_backtest(..., allow_test=False)`. No costs,
strategy, instrument, calendar or entry requirements are invented.
`BACKTEST.json` retains source pins, original report identities, byte-verification
results and execution receipt identity. A successful terminal has
`gates.pair_status = PASS` and `strategy_status = PASS`; execution failures are
reported explicitly. The eligibility evidence remains a separate pre-execution
decision with prediction bytes not yet verified.

Missing objects return **`FAILED / RAW_PAIR_MISSING_ARTIFACT`**, with exact
configuration object names in `gates.required_objects`. The worker does not
accept a child `WAIT_ARTIFACT` status; this hook does not pretend it does, and
does not install a retry loop. Scientific/schema/tampering errors remain typed
failures. Parent pre-claim handling and any retry policy remain parent-owned.

## Evidence And Traceability

Test design preceded implementation: first run failed at the expected missing
`raw_pair` import. The post-implementation command was run with two CPU affinity
slots, `ulimit -v 2097152`, a 60-second timeout, hidden CUDA and one BLAS/OMP
thread. `--noconftest` avoids the existing parent fixture's TensorFlow import:

```bash
python -m pytest --noconftest tests/i6dmo/test_raw_pair.py \
  tests/i6dmo/test_raw_long_operations.py -q
```

Observed: **32 passed** (25 hook tests, 7 long-operation tests), 0.77 seconds.

| Requirement | Executable evidence in `test_raw_pair.py` |
| --- | --- |
| Immutable short/long/comparison lineage | forged stored terminal, CHILD tampering, repinned forged identity |
| Official complete long choice only | absent/contradictory/schema/digest/support comparison cases |
| Own naive on identical head population | changed naive/replay, wrong horizons, positive row-digest replay |
| Frozen 2-long/1-short minimum | eligible minimum with exact missing-pin objects |
| Negative is complete, published, data-free | negative gate with NPZ/policy/calendar/backtest read tripwires |
| Existing positive strategy path | actual NPZ/calendar/versioned instrument and pinned policy; closed trades |
| No TEST or stale source substitution | TEST-open/window refusals, prediction-byte/source-pin tampering |

A separate bounded read-only probe used the closed short PLAN's `mg2m.report`
identity to call `_report`, with an NPZ read tripwire. It authenticated original
identity `dda1e2270c8eaff01ad5469cb0c64f47d659dd85a8640dfc4338bc3d93bdd1e0`
and code `796ec8f48b0b1621fb9fa9e60e12c1c37079e51e`, exact short heads, and eligible
heads 4/6/12. `prediction_bytes_verified` was false. No database connection or
prediction array was opened, and no retained artifact was modified.

Synthetic accepted tests validate plumbing and backtest execution, not scientific
performance. Deployment review, actual long evidence and the parent's frozen
private execution pins remain prerequisites; this note does not grant release.
