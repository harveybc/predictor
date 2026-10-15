# I6-DMO return: market-time MG2/MG12 pair — integration, storage policy and launch

Order `50d20d01` (+ the storage addendum), plan `MARKET_TIME_PAIR_PLAN.md` sections 4 and 6.
Release pin `24f5a6cc` on `satoshi/i6dmo-static-20261014`. **This is a launch receipt, not a
completion claim**: the first fit is running and no result exists yet.

## 1. Integrated pins and seals

| Package | Commit |
|---|---|
| ops / integration (this branch) | `24f5a6cc` |
| model (encoder, sealed designs, fit path, review items) | `afdfdc92` |
| data (MG2/MG12 bundles, support study) | `00c27fed` |
| strategy (instrument configuration, pair hook) | `8870a853` |
| reviewer (ACCEPT_FOR_SCIENTIFIC_EXECUTION) | `84360909` |
| storage policy | `128e7a1c` |

Seals, unchanged by this integration and asserted by `market_time/INTEGRATION_PINS.json`:

| Family | Design | Seal | Effective config |
|---|---|---|---|
| Short MG2 | `1fa48222…` | `4b64b4ee…` | `920332ae…` |
| Long MG12 | `4da1d3e0…` | `97541764…` | `0853f6d9…` |

Bundles: MG2 `21400c51…` (`BUNDLE.json` `471ea311…`), MG12 `c3867b5c…` (`e6595c0e…`), input
`ef22ddb5…`, manifest `2ff34fa2…`, calendar `782603fd…` (ASSUMED, not broker). Instrument
configuration `f2443141…`, `NOT_BROKER_VERIFIED`.

## 2. Reviewer result

`ACCEPT_FOR_SCIENTIFIC_EXECUTION` (RB05) on `116bbef4`, after `FIX_REQUIRED` (`f07305a2`, B.1).
The reviewer ran the real CLI end to end on both sealed designs, verified the pin discipline and
the scoped reseal, and did not reopen DMO10. Blockers closed: B-1/B-2/B-4 by the model lane
(sealed effective config builds 3-input/78-channel networks; the trainer consumes
`i6dmo.market_bundle.v1` end to end; ordinal and impossible offset ladders are typed refusals at
both seams) and **B-3/K2 by ops** — the campaign hard-required `i6dmo.design.v1`, so it could not
seal either market design, and my own tests passed only because the fixture improvised an
elapsed-schema design. The fixtures now use the real sealed designs through the real validator.
Two non-blocking items were closed at `afdfdc92` (design-derived spacing bound at both seams;
seven MODEL acceptance rows cite `test_market_time_fit_path.py`).

**The reviewer's missing-coverage list stands as stated by him**: all GPU paths, the real private
bundles, and deploy/preflight/init/publication were outside his bounded CPU scope. Nothing in this
return claims he verified them.

## 3. Tests actually run

- `tests/i6dmo`: **404 passed** at the release pin (bounded CPU job, CUDA hidden).
- Ops market-time regressions MTP01–MTP11 (finite graph, long progresses after a NOT_ELIGIBLE
  short, typed `PAIR_NOT_ELIGIBLE` with no strategy run, qualified pair runs the strategy, no TEST
  unit enumerable or claimable, elapsed-grid terminal never adopted, typed failure without retry,
  real sealed designs accepted and elapsed refused, contradictory designs refused, the ops gate
  independent of the upstream validator, swapped bundle caught before the claim). PRE for the
  first seven was a collection error; PRE for the design-seam tests was the `Refused: SCHEMA`
  blocker the reviewer found.
- Storage adapter: 14 regressions, PRE 12 failed (module absent), POST all pass.

## 4. Storage policy integration

Live state stays on ext4 — proven on the running campaign by `findmnt` per path:
`queue.sqlite3`, `outbox.sqlite3` (+ `-wal`/`-shm`), the three `worker-*.lock` files, the private
`deployment.json` and the campaign root itself are all **ext4**; the experiment-storage target is
`fuseblk`. No SQLite, PostgreSQL or DuckDB database is ever placed there.

The versioned adapter (`tools/i6dmo/storage.py`, `i6dmo.storage_adapter.v1`) archives only large
immutable artifacts of an authenticated terminal: digest computed before the copy, re-verified
after `fsync`, copies landing on a temporary name and renamed, so nothing is half-published. The
terminal keeps its relative-path identity and only gains an `archive` block, so identity never
depends on the mount; the archival path and digest publish as a governed report to the outbox and
both mirrors. Restart adopts a digest-matching archive and replaces a partial one.

Real round-trip on the mount (six artifacts, 2,949,607 bytes): every source digest equals the
archived digest and the re-read digest; the second pass reported `ADOPTED` for all six; terminal
identity unchanged and all artifact paths still relative.

PRE storage checks (literal): `experiment_storage_preflight.sh <root> 64424509440 5368709120` →
exit 0, `STORAGE_READY … available_bytes=763253284864`; the same on the `i6dmo/` subdirectory.
Refusal controls, all from the real script and **without touching the mount**:
`PATH_OUTSIDE_EXPERIMENT_STORAGE` (exit 2), `OUTPUT_DIRECTORY_MISSING` (exit 2),
`INSUFFICIENT_SPACE` (exit 5). Read-only and missing-mount failures are injected **at the seam**
(an injectable probe), not by remounting anything, and are stated as such.

Gates run before init, before deploy and when a job claims an artifact-writing unit; each failure
is a typed refusal or `WAIT_RESOURCE`, never a fallback to the home filesystem. Estimate used:
60 GiB with a 5 GiB reserve. Disk is reported **separately** for the coordinator and each worker,
with `UNAVAILABLE` for a worker that cannot report; the coordinator's experiment-storage volume is
never presented as a worker's free space, and it is not mounted on the workers.

## 5. Launch receipt

Campaign `i6dmo-72809b6c88a6`, root on the local ext4 scratch filesystem.

- `validate-design` PASS for both designs at the expected identities; `preflight` PASS with no
  launch blockers (coordinator, preferred and secondary disk all pass; experiment storage PASS).
- `init` sealed both designs and the execution-policy receipt; the plan has the six units and
  `final_units: NOT_ENUMERABLE_IN_MARKET_TIME_PAIR`.
- `deploy` installed 10 units; tick, publish and the three role-worker timers are enabled/active.
- **Single verification:** `short.fit` RUNNING on the **preferred role's 5090**, one GPU compute
  process on that host and no second heavy child; durable lease held and fresh; effective-config
  seal verified; cap **3,314,114,560 B** = 1.25 × the measured 2,651,291,648 B peak of the
  previous exact-DS2 fit; bundle `21400c51`. The 5070 Ti is never a parallel slot; the 4090 was
  used only for admitted checks.

Two earlier roots were retired before any unit was claimed and are preserved with their reasons:
one aborted at init (the instrument file is an envelope needing `instrument_from_config`), one
retired for an ops defect (the claim-time data-contract check called the elapsed function on a
market-time design). **No fit ran in either; nothing was published.**

## 6. Status commands (read-only, finite)

```bash
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign    --json
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level unit --unit short.fit --json
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level publication --json
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level fleet       --json
```

Expected validation units: 6. **ETA: UNMEASURED** — no market-time fit has completed, and the
exact-DS2 short timing is not this treatment's ETA (different population and a new encoder); the
in-flight unit reports only an upper bound at its 200-epoch ceiling. No calendar-week projection
is offered.

## 7. Scope and limitations

- **2024 is DEVELOPMENT**, not an untouched holdout: the 19-feature set was selected using it.
- FRED publication clocks: `PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD` — 339 of
  1,587 observations covered, with the 2020-08-25 back-fill; **TEST stays closed**.
- All four strategy costs are `ASSUMED_NOT_BROKER_VERIFIED`; the MT5 evidence is
  `REFUTED_NOT_FOUND`. Any backtest would be development-only.
- Calendar `ASSUMED_NOT_BROKER`; elapsed spans vary across closures and are never labelled as
  fixed 24/144 elapsed hours.
- DS2 versus MG2 is descriptive, not an isolated downsampling effect.
- M1–M8 remain **66/78/78/48/22/24/28/15**; only M6's descriptive text changed.
