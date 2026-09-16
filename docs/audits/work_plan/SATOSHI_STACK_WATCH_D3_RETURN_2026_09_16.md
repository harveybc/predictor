# Return: stack identity, bounded watch, and the D3 preparation

Order: `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md` (`e3403f5`),
over the acceptance of `18ffa4e`. Three independent blocks, executed without pausing between
them. The orphan stop is CLOSED by the reviewer and was not repeated.

## 1 — the STACK.json overwrite cannot recur

The defect, stated exactly: a second stack started in the same `--work` directory wrote its own
`STACK.json` over the first one's. The first run's children were then recorded nowhere, and its
own teardown answered `ALREADY_GONE` for three PIDs that belonged to the newer run. Two holes
sat beside it: the record was written only after all three children were healthy, so a launch
that failed part-way left running children and **no record at all**; and the marker that
authorises a signal was the `--work` directory the caller named, which two runs share by
definition.

| | before | now |
|---|---|---|
| identity | none | `run_id`, 32 hex from `os.urandom(16)`, immutable once written |
| directory | shared `--work` | `work/run-<run_id>/`, created with `exist_ok=False` |
| marker | the caller's directory | the run's own directory, which carries the identity |
| record | `work/STACK.json`, overwritten | `work/runs/<run_id>.json`, one file per run, never written over |
| written | after all three were healthy | **before any child exists**, updated as each starts |
| teardown by directory | impossible | `--teardown <WORKDIR>` walks every run ever opened there |

`STACK.json` stays where it was, as an advisory pointer for whoever reads it. It is no longer
what teardown depends on.

**Tested** — `tests/test_disposable_stack_run_identity.py`, **19 rules**, none against
production and none starting a store host:

* two starts at the same destination: both records survive, and the first one's PIDs are still
  its own;
* concurrent starts: four threads, four identities, four directories, no collision;
* partial launch failure: the first `wait_for` raises and the manifest holds exactly the two
  children that were started, with the argv each was given, and `complete: false`;
* teardown after the parent exits: from the reloaded record alone, with no handle and no
  parent;
* tearing down one run leaves the other running;
* a PID reused by a stranger is still `PID_REUSED_REFUSED`;
* **a child whose argv merely mentions the work directory is refused** — the directory name
  alone never authorises a signal;
* no `pkill`, `killall`, `pgrep`, `-f data_` or `process_iter` anywhere in the harness.

Failing before: **18 red** (`K_FAILING_BEFORE.txt`).

## 2 — a bounded watch that cannot repair anything

`tools/olap_consistency_watch.py`. It watches for the two faults that were found by hand:
surplus multiplicity, and a filtered read disagreeing with a scan.

**Read-only by construction, not by intention.** There is one `urlopen` in the module and it is
a `GET`; the class that reaches the service has no write method; and a test asserts that the
source, with its docstrings stripped, contains no `DELETE`, `UPDATE`, `INSERT`, `DROP`,
`CREATE INDEX`, `ALTER`, `CHECKPOINT`, `--repair` or `reindex_relation`. A finding is evidence
for an operator, never a trigger: `actions_taken` is empty in every report and the unit treats
a finding exit as success so nothing downstream can react to it.

**Cadence from measured cost.** Every run times its own queries and derives the interval from a
duty-cycle budget, floored at 300 s and capped at a day. Measured against the live cube on
2026-09-16: **16 queries, 0.055 s wall, no alert, 55 terminals**. At one percent that earns
5.5 s, so the floor governs — a check this cheap still should not be continuous. The query
count is **constant in the number of terminals** (two aggregates per relation, not two per
terminal), which a test holds at 202 terminals.

Deployed: `crispdm-olap-consistency-watch.timer`, `OnUnitActiveSec=300s`, enabled and run once
through systemd (`Result=success`). Observations append to
`~/.local/state/crispdm-duckdb/watch/observations.jsonl`; a discrepancy stays in the log after
it clears, so the record is not only the latest report. `MemoryMax=512M`, `ProtectSystem=strict`
with one writable path.

**Tested** — `tests/test_olap_consistency_watch.py`, **14 rules**: both faults detected
separately and together, a clean cube raising nothing, the cost stated, the cadence following
the cost, the cube byte-identical after being watched, and the token never an argument.

## 3 — D3, prepared and not executed

**The step.** The binding sequence is `D0 -> D1 -> D2 -> D3 -> ...`. D2 has candidate evidence
awaiting review; the next **unexecuted** step is **D3** — quantization/compression, entropy,
time-frequency and detectors (STEP 04-07), all of which the status table lists as written but
not common and without evidence. Its design is sealed in
`07_DISENO_D3_..._2026_09_14.md` and was not touched.

**The actual prerequisite, probed rather than read from a status table**
(`tools/df_d3_prerequisites.py`, `D3_PREREQUISITES.json`, against the live cube):

| prerequisite (§4 of the design) | measured | |
|---|---|---|
| R2 repaired adjudicator | `PRESENT` | `tools/df_d2_adjudicate.py` |
| N3 productive micro-run | `PRESENT` | the reviewer's restart act |
| **R6 current coverage view** | **`NOT_APPLIED` — blocking** | `df_coverage_current` and `df_coverage_history` do not exist in the cube |

The block is concrete. `df_fact_coverage` holds **440,694 rows**, comprising **one** `run_id`
and **two** code digests; `df_fact_coverage_v2` holds **633,189**. With no selection view, a
coverage figure read today cannot say which run and which code digest it counted — and the
design requires a current coverage view *for every measurement that grounds a decision*.
Measuring D3 before R6 is applied would produce numbers nobody can attribute.

R6 is applied through the adoption route, never by editing the cube, and that is an operation
with its own authorisation. It was not done here.

**Contracts and tests prepared:**

* `tools/df_d3_contract.py` — the eleven mandatory declaration fields of §1, made refusable.
  No defaults anywhere. `output_availability` and `delay_samples` are the same fact stated
  twice and a disagreement is refused, because that disagreement is exactly how a claim of zero
  delay gets made in silence. An available output inside a declared warm-up is refused as a
  fabricated observation. The raw branch is part of the contract rather than a convention.
* `tools/df_d3_acceptance.py` — the ten acceptance tests of §3, runnable against any operator
  that carries the contract, with `MECHANICALLY_ACCEPTED` / `MECHANICALLY_REFUSED` /
  `INCONCLUSIVE`. It scores nothing, opens no database and starts no governed run.
* `tests/test_d3_contract_and_acceptance.py` — **40 rules**. A causal operator declared
  honestly passes; its centred twin, the design's own non-causal control, fails; and there is a
  fixture per test that lies about one field and fails only that test. `UNKNOWN` availability
  makes the verdict `INCONCLUSIVE`, never accepted.
* `docs/integracion_workplan_2026_09_10/11_PLAN_EJECUCION_GOBERNADA_D3_2026_09_16.md` — the
  governed execution plan: inputs, Flow v3 route, terminal shape, `NON_GOVERNING`
  classification while the battery is mechanical, one-thread CPU with the design's 2 GiB
  ceiling, cost piloted before any sweep, and abstention as a valid outcome.

### A measured limit of the sealed design, for the reviewer

§3.1 compares `transform(X[:n])` against `transform(X)` only up to `n - lookback - delay`. An
operator that looks **four** samples ahead while declaring **seven** samples of lookback is
therefore invisible to test 1: every index it compares is one both calls can compute. Measured,
with the centred control: **test 1 passes, test 2 fails.**

Test 2 catches it and the pair does its job. Recorded, **not corrected**: widening test 1 to
`n - delay` would close the gap and is a design decision on a sealed design, not mine to take.

## Suites

| suite | scope | result |
|---|---|---|
| migration + reconciler + watch + stack + D3 + `olap/store/tests` | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **327 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, with the store environment | **1465 passed, 12 skipped**, 0 failed, in 7m29s |

The disposable PostgreSQL database was dropped. `K_FAILING_BEFORE.txt` records 18 red for
block 1; it notes its own scope, since `git stash` restores tracked files only and the D3
modules are new — their "before" is that they did not exist, which is not evidence of a fix.

## Open, with owners

| item | owner |
|---|---|
| **R6 through the adoption route** — the only measured block on D3 | owner; needs operational authorisation |
| review of the D3 contract and battery before any operator exists, so results stay comparable | Musashi |
| what made `gov_terminal_metric_sha_idx` lose four entries — four mechanisms reproduced and excluded | Satoshi; unknown, not named |
| Metabase driver decision, historical terms research | Satoshi; nonblocking |
