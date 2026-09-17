# Return: D3 L1–L6 — verifier closed, twin sensitivity demonstrated and replayed, dispositions, warehouse boundary, utility protocol

Order: `docs/handoffs/MUSASHI_D3_K1_K5_REVIEW_AND_L1_L6_2026_09_17.md` (`72d6a63`), over the
review of `a37d495`. Executed without pausing between blocks. `NON_GOVERNING` throughout: no
scientific utility, no promotion, no financial selection, no GPU, no live. v1, v2, every attempt,
refusal and receipt are untouched; the replay is `d3mech-v3` under its own freeze.

## L1 — the verifier, closed without another campaign

The reviewer's three cases were frozen as failing rules first, then fixed:

| omission | now |
|---|---|
| a missing `UNIT.json` became a `None` binding and the contract check was skipped | `CONTRACT_UNBOUND`: every unit record is required; toys declared in the freeze require `TOYS.json` and each `TOY.json` with variables and contract (`TOYS_UNBOUND`) |
| the freeze's digest was read, never recomputed | `check_freeze`: digest recomputed over the body (`FREEZE_IDENTITY`), schema and cardinalities validated (`FREEZE_SCHEMA`, `FREEZE_CARDINALITY`), and the freeze cross-checked against the campaign record the report conserved — `synthetic_spec.freeze_sha256/design_sha256` and `config_sha256` recomputed (`CAMPAIGN_RECORD_MISMATCH` / `_MISSING`). A self-digest alone proves nothing. |
| a copied terminal under another worker/shard was accepted; the sort order chose | every location of every attempt is judged against the shards' member lists and the dispatch receipts: outside its shard or a role that shard was launched on → `UNASSIGNED_LOCATION`; a byte-identical duplicate (terminal and rows) is a recorded **transport copy**, never counted twice; a differing record for the same unit and attempt is `CONTRADICTORY_ATTEMPT`; unknown statuses (`UNKNOWN_STATUS`), discordant ids (`ID_DISCORDANT`) and terminals outside the population refuse. Nothing is resolved by path order. |

`tests/test_d3_matrix_verify.py` — **28 rules** (10 new, including the reviewer's reproducer's
cases; `reproduce_d3_k_review_2026_09_17.py` now prints `CONTRACT_UNBOUND`, `FREEZE_IDENTITY`,
`UNASSIGNED_LOCATION`). **v1 and v2 re-verified from their preserved bytes with the successor
verifier**: VERIFIED, 0 refusals, 0 transport copies, every figure identical to the earlier
verified matrices (`MATRIX.verified.v2.json` beside `MATRIX.verified.json`, nothing overwritten,
nothing re-measured). Delta between verifier generations: none.

## L2 — twin sensitivity, demonstrated per unit, never relaxed causality

**Diagnostic** (`d3_mechanics_v2/L2_TWIN_SENSITIVITY_DIAGNOSTIC.json`): for the 15 MCAR wavelet
variables — cuts (51 prefix, 21 perturbation), twin emissions, comparisons, and for each whether
the centred twin's window (w = 50, half 25, **right reach 24**) crosses the cut. The five `FAILED`
variables of v2 have **0 sensitive comparisons** in both tests (nearest output to any cut 47–256
samples); the `PASSED` ones have ≥ 1 sensitive comparison or an effective detection. The
hypothesis is a fact per unit.

**Amendment 07C** (`07C_ENMIENDA_SENSIBILIDAD_GEMELOS_D3_2026_09_17.md`,
`D3_TWIN_SENSITIVITY_AMENDMENT_V1`, sealed before measuring, names 07B): every twin declares
`reach_right` from its geometry (centred `w − w//2 − 1`; `filtfilt` the whole series;
`cusum_lookahead` its `ahead`; a twin with reach 0 is refused as a control); a comparison of
output *i* at cut *c* is **sensitive iff `i + reach > c`** — necessary, not sufficient: zero
coefficients, saturation or missing data can still leave no effect. Policy: **any observed
violation** (value, availability mask or emission instant) is a detection, never discarded by a
support declaration; **zero sensitive comparisons and no violation = `INSUFFICIENT_TEST`**;
sensitive comparisons without effect = the declaration's `FAILED`. Recorded per control:
emissions, comparisons, sensitive comparisons, detections, nearest output to a cut, reach,
first detection. The candidate's own causal tests are not restricted to the twin's mask. Support
50, imputation, thresholds, parameters, the complete-data control and the three warm-up refusals:
unchanged.

`tests/test_d3_twin_coverage.py` — **20 rules**: known centre (sensitive detection), a future
impulse just after the cut (reach 1, detected), extreme zero future weights (crossing without
effect is a failed declaration, never a pass by geometry), missingness (isolated, blocks, MCAR),
support boundary, restart, deliberately non-causal controls failing **by mask** and **by emission
time** apart (the emission depends on future *data*: an emission later than the cut is excluded
by 07A's delayed-output rule, so a time-stamp leak is invisible by design and the control
depends on values), a causal pretender with sensitive comparisons and no effect, a reachless
pretender refused.

**Replay scope, justified:** the sensitivity facts (availability masks per cut) are not in v2's
conserved rows, so `non_causal_twin` had to be measured again — for the **seven** operators that
declare a twin, over the **whole** frozen population including the cases accepted before. The
other eleven tests were **inherited from `d3mech-v2` by digest**, verifiably: the freeze of v3
declares `inherits` (source run, freeze and design digests, receipt, inherited/measured tests,
measured operators, justification); the worker measures the subset (`--tests --operators`, rows
carry `scope`); the verifier re-verifies the source from its own bytes, takes only inherited
tests from it and only measured tests from v3, each bound to its own freeze, demands the union
cover every cell and recomputes the verdict; a source that no longer verifies, a mismatched
source freeze or a missing measured cell refuses (`INHERITED_SOURCE_UNVERIFIED/_MISMATCH`,
4 composite rules). The two twin-less operators take all twelve tests from the source. No
operator was repeated for convenience.

**Run `d3mech-v3`** (root `~/.local/state/crispdm-data-foundation/d3_mechanics_v3/`): freeze
`67f2961a…` under design 07C `5ffe25f9…`, `inherits` from `d3mech-v2` (freeze `e9154ad5…`,
design `f198340c…`, 11 tests) and measures `non_causal_twin` on 7 operators; toys re-delivered
under seven campaigns; inputs synced to both workers before dispatch (the K5 lesson); 43 shards
COMPLETED on WORKER_A (6) and WORKER_B (2), coordinator excluded; collect 511/511 verified, 0
mismatches, 9,940 measured rows. Pilot: the freeze's own cost pilot (twin test only).

The composite verifier first refused with `CAMPAIGN_RECORD_MISSING` — the report had not run yet,
and the verifier will not seal without the conserved campaign record; after the governed report,
sealed as `MATRIX.verified.sealed.json` (the first, unsealed file is kept beside it):

Run `d3mech-v3` — **VERIFIED** (0 refusals) over receipt `COLLECT.json`: population 511 units × 710 variables × 9 operators × 12 tests; 511 completed, 0 failed (recorded), 0 missing; 9,940 rows re-read and bound.

| operator | group | units × vars | verdicts | causal tests failed | restart | availability | probe onset | cost s/1k (median, max) |
|---|---|---|---:|---|---|---|---|---|
| `butterworth_causal` | time_frequency | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0059, 0.0078 |
| `cusum_causal` | detectors | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0063, 0.0122 |
| `delta_run_length` | quantization_compression | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0005, 0.0011 |
| `mad_extremes_trailing` | detectors | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0005, 0.002 |
| `sax_paa_trailing` | quantization_compression | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0005, 0.002 |
| `stft_trailing` | time_frequency | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 1.0 697 | 0.0005, 0.002 |
| `uniform_decile_quantizer` | quantization_compression | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0005, 0.002 |
| `variance_regime_trailing` | detectors | 511 × 710 | INCONCLUSIVE 13 / MECHANICALLY_ACCEPTED 697 | none | PASSED 710 | PASSED 710 | 0.0 697 | 0.0005, 0.002 |
| `wavelet_trailing` | time_frequency | 511 × 710 | INCONCLUSIVE 24 / MECHANICALLY_ACCEPTED 683 / MECHANICALLY_REFUSED 3 | warm_up_edge 3 | INSUFFICIENT_TEST 7 / PASSED 703 | INSUFFICIENT_TEST 3 / PASSED 707 | 0.0 697 | 0.001, 0.002 |

**Delta v2 → v3** (`DELTA_v2_v3.json`, all attributed):

Delta `d3mech-v2` → `d3mech-v3` (design changed: True, population changed: False, all attributed: **True**)

| operator | verdict movements | test movements | causes |
|---|---|---|---|
| `wavelet_trailing` | INCONCLUSIVE 19→24; MECHANICALLY_REFUSED 8→3 | non_causal_twin: FAILED 5→0, INSUFFICIENT_TEST 3→8 | 07B: a twin without observable comparisons is INSUFFICIENT_TEST, never a wrong twin |

Exactly what the diagnostic said: the five `FAILED` twin outcomes are `INSUFFICIENT_TEST`, no
detection was lost (the `PASSED` cases keep their detections, now recorded with sensitive
counts), wavelet under MCAR ends at 24 `INCONCLUSIVE` and 3 `MECHANICALLY_REFUSED` — the three
warm-up refusals, kept apart. No other operator moved; nothing was tuned.

**Governance.** 504 SYNTHETIC terminals (`d3mech-v3-synthetic`, `dc7f741a…`) and 7 toy terminals
under their own campaigns, generation 1, all accepted at the first flush, reconciled
`missing_units: []`; envelope `90500297…` loaded (4,970 measured cells, 551 consumption rows;
campaign key now `d3-mechanics-d3mech-v3`). Content reconciliation against the independent
accounting (`docs/audits/evidence/d3_k5_20260917/L6_CONTENT_RECONCILE.json`):
**NO_LOSS_FOR_THE_COMPARED_POPULATION** — 1,588 terminals, 0 missing, 0 cube rows without an
accepted record.

## L3 — dispositions and campaign names

* **Envelope `475b93fb…`** (accidental DEVELOPMENT ingestion, first adoption post-check): kept
  with its history; its `envelope_sha256` **rederived** from the retained bytes with the loader's
  own `_sha` and equal to the cube's; children verified (`dim_campaign_run` 1, `fact_campaign_unit`
  1 — `run_wall_seconds`, consumption 0). Disposition published through the recorded procedure
  (warehouse stopped → `olap_duckdb_migrate publish-selection` → started; manifest
  `docs/audits/evidence/d3_k5_20260917/L3_DISPOSITIONS.json`, receipt `L3_PUBLISH.json`):
  `gov_campaign_disposition` row `ACCIDENTAL_OPERATIONAL_INGESTION`, admissibility
  `NOT_ADMISSIBLE_SCIENTIFIC`, status `RETAINED_AS_LOADED`. Effect verified: absent from
  `gov_scientific_evidence`, present in `gov_mechanical_evidence`; unit rows unchanged (12,782).
* **`d3mech-v2` under `campaign_key d3-mechanics-v1`**: kept; a second disposition row
  `CORRECTIVE_RELATION` (`MECHANICAL_OPERATIONAL`) records the logical campaign and the rule:
  select by `run_id` / `design_sha256` / `envelope_sha256`, never by prefix. Nothing re-emitted.
  `tools/df_d3_cube_select.py` resolves one run identity to exactly one envelope (refuses zero
  or several) and pages its cells under the service's `LIMIT` cap; proved on the real disposable
  warehouse that two runs sharing a key select disjoint populations while the key alone mixes
  them.
* **Idempotency probe**: `idempotency_probe()` posts only an envelope the cube already holds
  (chosen by `envelope_sha256` present in `fact_campaign_unit`), compares content (units before
  = after = declared) and runs before = after; with no held envelope it posts nothing. Proved on
  the real service that the probe introduces no run.

## L4 — error handling completed in `data-warehouse`

In the owner repository (`~/Documents/GitHub/data-warehouse`, branch
`satoshi/s2-availability-contract-20260915`, commits `96a3178`, `4198232`, `6f5adb8`, `1cbfba9`,
pushed): `errors.classify` maps `ValueError`/`SystemExit` and the client's own SQL errors
(parser, binder, catalog, conversion) to **400 `INVALID_INPUT`**, Unsupported/BackendRefusal/
Holdout/Permission to **422**, `StorageUnreachable` and driver connection/lock/timeout errors to
**503 `STORE_UNAVAILABLE`**, and everything else to **500 `INTERNAL_DEFECT`**, logged; every
answer is JSON with its class and a bounded, redacted message. The six catch-alls that answered
"503 database error" to any exception are gone; the query route no longer returns an HTML 500.
`DW_INDUCE_INTERNAL_DEFECT` (a marker file; off unless set) lets a disposable stack induce an
internal defect. `SystemExit` — the loader's typed refusal — is caught explicitly (it is not an
`Exception`; without that the connection dropped). 8 rules in that repository (37 pass).

Predictor side: `tests/test_olap_ingest_diagnostics.py` — **14 rules**, run with the candidate on
the real service's path (`K4_EXTRA_PYTHONPATH`): invalid document 400, induced internal defect
500 kept pending as `RETRYABLE_SERVER_ERROR`, transient outage 503/transport, recovery, second
drain without duplicates, selection by identity, probe without runs, wheel parity.

**Adoption** (procedure with rehearsal, backup, post-check; receipts
`satoshi-warehouse-adoption-20260917T07{5133Z,5433Z-2,5753Z-3}`):

* First attempt **crash-looped the production warehouse** for its restart window (`NRestarts`
  reached 18): `build/` is **tracked in git** in `data-warehouse` and setuptools packaged its
  stale `backends.py` (missing capabilities) instead of the source. Restored at once from the
  adoption backup (byte copy of what ran); the service came back `active`. Root cause proved
  (`wheel == build/lib`, `wheel != src`).
* The adoption tool now builds only from an **exported commit with `build/` and egg-info
  removed** and **refuses a wheel whose modules are not byte-identical to the exported source**
  (rule `test_a_wheel_whose_module_differs_from_the_exported_source_is_refused`). Second attempt
  `6f5adb8` and third `1cbfba9`: adopted, `active`, `NRestarts 0`, post-check green (malformed →
  400 with class, idempotency probe ok, table counts unchanged). In production now: a malformed
  envelope is `400 INVALID_INPUT`, invalid SQL is `400 INVALID_INPUT` JSON (was HTML 500).
* No other service touched; no WAL removed.

## L5 — the utility design, corrected before any freeze

`12_PREPARACION_UTILIDAD_REPRESENTACIONES_2026_09_17.md` rewritten as a protocol and
`tools/df_utility_harness.py` implemented: a sealed `Protocol` (target, horizon, probe model,
window, walk-forward blocks, **purge = horizon + reach + window**, margin, seeds, multiplicity
family, abstentions); MAE/log-loss named as **a difference of predictive losses of the probe
model**, never information; eligibility per (dataset, variable, regime, representation) from the
verified matrix, development / public confirmation / financial revalidation apart; windows by
observation identity and decision instant; normalisation and fits inside the training block;
paired scoring on the same emittable rows with coverage, missingness, cost and failed attempts
reported apart; raw / transformed / augmented branches with declared capacity control and what
each contrast identifies; block as statistical unit, Bonferroni over the predeclared family;
`INSUFFICIENT_ROWS`, `BUDGET_EXHAUSTED`, `NOT_COMPARABLE`, `REFUSED` as recorded outcomes; the
reserved holdout adjudicated once, write-once. `tests/test_df_utility_harness.py` — **11 rules**
on fabricated truth: positive control advances, noise control does not, unaccepted
representation refused before scoring, paired rows, purge, budget, insufficient rows, augmented
contrast, one-time holdout, log-loss for direction. **Not executed on project data; no reserve
scored; this order opens nothing.**

## L6 — closure

**Suites** (trading-stack, `crispdm-run`, with the data-warehouse candidate on the real
service's path): D3 contract 40 · operators 37 · pipeline 15 · matrix 1 · verify 28 · probe
amendment 17 · twin coverage 20 · delta 3 · ingestion diagnostics 14 · utility harness 11 ·
dispatch · outbox disposition · `olap/store/tests` — **272 passed, 1 skipped** (the store suite's
own skip), 34.1 s; `data-warehouse` tests: **37 passed**. Exclusions: none. Digests read: every
freeze, receipt and terminal named above, re-hashed by the verifier.

**Commits (predictor, `satoshi/r1-r6-20260914`)**: `fa732d8` L1 · `1fd903b` L2 amendment ·
`8840d54` composite replay · `bc50256` L3–L4 · `c424059` adoption hardening · `acee660`
`0dac0c2` adoption tool · `5092be9` L5 · `a564ce8` · this return. **data-warehouse**: `96a3178`
`4198232` `6f5adb8` `1cbfba9`.

Backlog, measured not assumed: terminal outbox 0 pending / 0 awaiting adjudication; OLAP outbox
0 pending, dead letters adjudicated, `attention_required: false`; loader heartbeat healthy.

Open, with owners: `build/` tracked in `data-warehouse` (the packaging trap; its owner's call to
untrack); the accidental envelope's disposition is recorded and awaits Musashi's ruling on
anything further; index-loss root cause, Metabase, terms: separate fronts, unchanged.

Ending: **`D3_MECHANICS_VERIFIER_CLOSED_AND_UTILITY_DESIGN_READY_FOR_REVIEW`**.
