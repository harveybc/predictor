# Return: temporal semantics corrected, and the four consumers through the new host

Orders: `docs/handoffs/MUSASHI_TO_SATOSHI_TEMPORAL_SEMANTICS_AND_REAL_HOST_ADOPTION_2026_09_14.md`
(N1–N8), over `docs/audits/work_plan/MUSASHI_REVIEW_A1_A5_CONTRACT_AND_ADOPTION_2026_09_14.md`.
Every finding in that review is accepted; none is argued with.

Scope of everything below: transport, mechanics and temporal semantics. No scientific
approval, no D2 change, no restart of a healthy service, no branch checkout in a live data
root, and no financial resource consumed without a contract.

## N1–N2. The clocks separated, the zero-lag claim withdrawn

**Withdrawn:** `available_time_column: close_time` with `completion_lag_max: "0s"`. A window's
close is when the window ended, not when the data could be known — and the same submission
declared delivery latency UNOBSERVED. The measurements stand; the contract they justified
does not. The old file is kept with a supersession header.

| clock | status | evidence |
|---|---|---|
| window start | MEASURED | `open_time`, tz-aware **UTC** in the physical schema |
| window end | MEASURED | `close_time`; modal span 4 h − 1 ms |
| finalization | **NOT DEMONSTRATED** | no field says whether a row is final |
| publication | **UNOBSERVED** | nothing records when the provider published a row |
| reception | **UNOBSERVED** | one file-level acquisition timestamp bounds the file, not a row |
| revision | **UNKNOWN** | one acquisition cannot show restatements |

The 21 non-nominal bars are classified from the bytes instead of assumed final: **1 empty
placeholder** (span 0, no volume, no trades), **12 shorter nominal intervals** (exactly
2 h − 1 ms or 3 h − 1 ms), **8 partial aggregates**; 8 of the 21 sit immediately before one of
the file's 8 gaps. None may be used where finality is required; the gaps stay gaps.

Eligibility is now computed, with typed refusals: `PUBLICATION_TIME_UNOBSERVED`,
`FINALITY_NOT_DEMONSTRATED` → kind `ARCHIVE_RETROSPECTIVE`, point-in-time **REFUSED**, live
**REFUSED**. A failed measurement, a non-monotonic clock, duplicates, an end before its start
or bytes that are not the producer's each stop the contract instead of producing one.

**Minimal additive extension** so the honest statement is representable:
`use_class: ARCHIVE_RETROSPECTIVE` with `completion_lag_max: "UNKNOWN"` — the only class
allowed, and required, to leave the lag unknown. It has teeth: whole delivery or nothing, a
ranged delivery refused with its reason, and refused entirely under a holdout. Existing
classes are untouched and no installed contract declares it.

**Tests corrected, as the review required:** the live case now asserts the refusal that
applies to *this* resource (the old one called the validator without expecting a refusal and
then asserted a different one); the time zone must be **exactly** UTC; the intrabar case
requires the **typed** refusal. Added: next-day reception, late revision, non-final bar, and
one case per invalidating measurement. 19 tests, plus the auditor's four counterexamples
frozen under `store/tests/frozen/`.

Delivered as **financial-data PR #2** (`cfedbddd6`), superseding PR #1. Nothing installed.

## N3. Fixtures with the schemas the consumers actually read

`tools/make_consumer_fixtures.py` (generator digest `1ca00f81…`, seed `20260914`) writes seven
deterministic files: three 4 h price series, three 27-column feature series, one hourly OHLC
series. The manifest records generator, seed, schema, roles, frequency, first and last event
and the digest of every file.

Two things this exercise taught, both recorded in the generator:

* **no availability column inside the data.** The consumers read every column as a feature;
  feature-extractor put an ISO timestamp into a float tensor on the first attempt. The
  generator is the producer, so it declares the rule it applied — a row of a step-long grid
  is complete one step after its timestamp — and the contracts encode that as `WINDOW_START`
  with `completion_lag_max = step`;
* **schemas come from the code that reads them.** feature-eng indexes on `DATE_TIME` with
  `OPEN/HIGH/LOW/CLOSE`; feature-extractor's CVAE targets must exist in the fixture's columns.

The fixtures are served from a root separate from the financial lake, and the two-row fixture
`panel.csv` keeps its identity (`5b4cabe4…`) with the catalogue extended around it.

## N4. The route, proven and controlled

`consumer -> data-gov -> data-lake host + external provider (http) -> data-warehouse host`,
on a disposable stack (`tools/disposable_route_stack.py`: free ports, throwaway SQLite cube,
`PG*` stripped so it cannot reach the real one).

| consumer | governed delivery | pipeline output | stale refused | failure with cost | retry |
|---|---|---|---|---|---|
| feature-eng | VERIFIED_TRANSFER | **produced** | REFUSED | FAILED | sends nothing |
| feature-extractor | VERIFIED_TRANSFER | **produced** | REFUSED | FAILED | sends nothing |
| preprocessor | VERIFIED_TRANSFER | **produced** | REFUSED | FAILED | sends nothing |
| predictor | VERIFIED (transfer + cache) | **produced** | REFUSED | FAILED | sends nothing |

The harness takes lake, root, manifest and governance URL from the command line, and records
the route data-gov reports for that lake (`kind=lake, engine=files_inventory, transport=http`)
— the previous round's `predictor_examples` was a local `files_lake`, which is exactly the gap
the review named.

**Negative control:** with the entry host stopped, the same run fails with
`503 lake unreachable: [Errno 111] Connection refused`, at the delivery. The route depends on
the host; a local adapter would have kept working.

**Production:** the fixtures are copied into the live synthetic root (additive; `panel.csv`
untouched) and the extended catalogue is written as the **pending** configuration at
`5058.pending.json`. The active configuration is unchanged and still serves one resource.
Activation is one restart of the synthetic service, which this agent may not perform
(`BLOCKED(process control of a running service, the owner or an operator,
`systemctl --user restart crispdm-data-lake-synthetic` after promoting the pending file)`).
Until then, the production route for the four consumers stays at what the previous round
proved: data-gov's local lake for inputs, the new warehouse host for terminals.

## N5. The retry that was only asserted before

The review was right: flushing an empty outbox proves the flush wrote nothing. This creates
the condition for real — the terminal destination is killed **during** the run, in the window
between the governed delivery and the terminal send:

| step | result |
|---|---|
| run with the destination falling mid-run | `terminal_pending: true`; outbox holds 1 envelope, class `TRANSIENT`, with its 503 |
| recovery through the wrapper's outbox | `sent: 1`; the cube holds **exactly one** row for that campaign |
| second flush | `sent: 0`; the cube is unchanged |
| second run over the same input | `VERIFIED_CACHE` receipt, delivery not repeated |

Evidence: `docs/audits/evidence/repro_runs/route_20260914/n5_receipt.json` and the script
next to it. Classification: 8 tests — default GOVERNING, `NON_GOVERNING` in **both** the
campaign and the receipt, invalid values refused by the CLI, the recorded runs declare what
they were, and an `ARCHIVAL_REPLAY_NON_AUTHORITATIVE` replay cannot be cited as governing
because the campaign that carried it is not.

## N6. Requirement → test → result, per consumer

| requirement | test | preprocessor | feature-eng | feature-extractor | predictor |
|---|---|---|---|---|---|
| input identity reaches the run | receipt records resource, sha256, verification state, contract | ✔ | ✔ | ✔ | ✔ |
| the entry host is really used | route recorded as `http`; negative control fails at delivery | ✔ | ✔ | ✔ | ✔ |
| a stale output cannot pass as fresh | stale case | REFUSED | REFUSED | REFUSED | REFUSED |
| a failure is recorded with its cost | failure case | FAILED | FAILED | FAILED | FAILED |
| a retry does not duplicate | real pending + recovery + second flush | ✔ (measured) | flush after run | flush after run | flush after run |
| a cached input is declared as cached | second run | ✔ | — | — | ✔ |
| future rows do not change an earlier delivery | `test_appending_future_bars…` (financial-data) | ✔ | ✔ | ✔ | ✔ |
| partitions disjoint | fixtures declare train/validation/test; contract test | ✔ | n/a | ✔ | ✔ |
| fit only on training | **not covered here** — belongs to each consumer's own suite | open | open | open | open |

Applicability, declared: feature-eng and preprocessor are batch transforms with no incremental
API to fake; their temporal boundary is the fixture's grid and the contract's
`WINDOW_START + step`. What this evidence does **not** cover: that any transformation is
useful, that any model is free of leakage, or that a centred transformation is causal — that
control is named below as the next step, not claimed.

## N7. Rights and revisions

Not resolved, and not assigned to the owner. What is recorded: the producer declaration names
the source and states no publication time, no revision policy and no usage rights; a second
acquisition would compare two snapshots without establishing a policy. The next action is
primary-source research on the provider's terms and on the acquisition script's own
provenance, which does not block N3–N6 and has not been used to justify any claim.

## N8. DOIN

Not started in this round; it remains in its owning repositories, offline replay only.

## Blockers

| blocker | owner | next action |
|---|---|---|
| the extended synthetic catalogue is pending, not active | owner or operator | promote `5058.pending.json` and restart `crispdm-data-lake-synthetic` (that unit only) |
| the four consumers in **production** through the new lake host | follows the above | re-run `tools/verify_consumer_adoption.py --lake governance_smoke` after activation |
| the archive extension is proposed, not deployed | review | financial-data PR #2, then a deliberate window |
| usage rights / revision policy of the ETH resource | Satoshi first (primary sources), then the data owner | research before asking anyone for a decision |

Services at the end: `:5055/:5056/:5057/:5058` healthy, four units active with zero restarts,
loader advancing, cube history intact, no production configuration modified.

Stop: `TEMPORAL_SEMANTICS_CORRECTED_FOUR_CONSUMERS_PROVEN_THROUGH_THE_NEW_HOST_ACTIVATION_PENDING`.
