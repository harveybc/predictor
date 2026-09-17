# Return: D3 K1–K5 — verified matrix, amended probe and twins, observable ingestion, successor mechanics

Order: `docs/handoffs/MUSASHI_D3_J1_J3_REVIEW_AND_K1_K5_2026_09_16.md` (`80643a6`), over the
review of `8364714`. Executed without pausing between blocks. Nothing here is scientific utility;
every run is `NON_GOVERNING` mechanical evidence. v1 (`d3mech-v1`), its attempts, its refusals
and its receipts are untouched; the successor run is `d3mech-v2` under its own freeze.

## K1 — verify before summarising

`tools/df_d3_matrix.py` now has two things that were one: `verify(root, receipt)` seals the
**verified matrix** and `aggregate(root, receipt)` is the **exploratory summary** that says so
(`verified: false`, "not a verification") and no longer repeats any count of the receipt.

`verify` derives the expected population from the frozen manifest — units and their variable
counts from `FREEZE.json` and the bank's own unit records, toys from `TOYS.json`/`TOY.json`,
operators with their `spec_sha256`, the twelve required tests, the amendment digest — resolves
each unit's attempt from the collected **ledger** (the terminals, highest attempt, never by
score), re-hashes every `rows.jsonl` against the terminal that produced it (digest and line
count), binds every row to the manifest (design, spec, code, contract digests), demands exactly
one row per unit × variable × operator × test and exactly one verdict, recomputes each verdict
from its own tests under the battery's state policy, and seals only with zero refusals. A unit
recorded `FAILED`/`RESOURCE_EXCEEDED` stays in the denominator; a unit with no terminal is
missing scope; `output_verified` is never taken as sufficient.

`tests/test_d3_matrix_verify.py` — **15 rules**: the reviewer's two counterexamples (a lone
favourable verdict row → `MISSING_TESTS`; a missing file → `MISSING_FILE`, and the summary
refuses too), a file modified after COLLECT (`DIGEST_MISMATCH`), a failed test under a
favourable verdict (`VERDICT_CONTRADICTION` with recorded/recomputed), a row of a variable the
unit does not have (`UNEXPECTED_VARIABLE`), a duplicated receipt (`DUPLICATE_RECEIPT`), a
duplicated row, the seal on a consistent run, a recorded failed unit in the denominator, a
frozen unit with no terminal, attempts from the ledger and a receipt disagreeing with it, a unit
outside the population, rows bound to another spec, and the summary declaring itself.

**v1 re-aggregated from its preserved bytes, not re-measured**: `MATRIX.verified.json` beside
the original `MATRIX.json` (never overwritten) — **VERIFIED, 0 refusals**, population 511 units
× 710 variables × 9 operators × 12 tests, 511 completed, 0 failed, 0 missing, 83,070 rows re-read
and bound. Every per-operator figure equals the original summary: **no difference**, so no cause
to report. The reviewer's reproducer now stops on the summary's missing `mismatched` key (it no
longer exists: the summary repeats nothing from the receipt); its two cases are the first two
rules above.

## K2 — probe amendment, not result adjustment

`07B_ENMIENDA_SONDA_Y_GEMELOS_D3_2026_09_17.md`, sealed as `D3_PROBE_AMENDMENT_V1` in
`tools/df_d3_design.py` (own `design_sha256`, names 07A by digest; `D3_DESIGN_CURRENT`), **before
any measurement**; contract `d3_operator_spec.v3` (`response_probe.scale = TRAIN_FIT`).

The excitation is built from the training fit only: baseline = train quantile 0.10, scale = q90 −
q10 (zero → `UNIDENTIFIED`), noise 0.01·scale identical in both branches, amplitude = the
operator's declared `probe_resolution(state, baseline, scale, sigma)` — the smallest excitation
it guarantees moves its impact-sample output, from its own fit, or `UNIDENTIFIED` with a reason.
Nothing searched, nothing from validation/test, no operator parameter or threshold changed
(`test_operator_parameters_are_the_declared_defaults_unchanged`).

Found on the way and frozen: the baseline of a decile codec fitted on the same train **is** an
edge — without clearing the noise band the measurement would have been an artefact in the other
direction. The codecs' resolutions clear the band (`b + 3σ`) and the first edge/breakpoint above
it; CUSUM reaches `mean + k`; variance regime declares a gain of 8; linear/unbounded operators
answer with the scale.

Three facts recorded apart in every probe outcome: `identifiable` (declared), `first_change_observed`
(measured), `matches_declared`. Policy: declared identifiable and nothing moved → **FAILED**
(contradicted, not abstained); first change ≠ declared → **FAILED** (J2's shifted control still
measures 2 and fails; a deliberately delayed quantizer measures 1 and fails); `UNIDENTIFIED`
only by declaration and it makes the verdict **INCONCLUSIVE**, never accepted.

`tests/test_d3_probe_amendment.py` — **17 rules**: the seal; the v3 refusal; the decile quantizer
at five scales (`1`, `300` at 5000, `1e5` at −2·10⁶, `1e−9`, `1e9`) shows onset 0 on its own
domain; SAX at two scales; the three facts; STFT declares 1, measures 1; constant train
`UNIDENTIFIED`; saturation `UNIDENTIFIED` with reason; declared-unidentified → INCONCLUSIVE;
delayed control fails; contradicting control fails; the J2 control; parameters unchanged.

## K3 — twins and scarce availability

`check_non_causal_twin`: a demonstrated causality failure of the twin is the detection
(`PASSED`); **no observable comparison** is `INSUFFICIENT_TEST` (undecided → `INCONCLUSIVE`),
never a detection and never "the twin passed causality"; compared many times and never failing
is `FAILED` (wrong twin or tests not measuring causality). `twin_emissions` and
`twin_comparisons` recorded for every control. `run_battery` reports `coverage = {n, emitted,
inputs_available}` apart from causality and from `applicability`. No future interpolation;
support 50 unchanged.

`tests/test_d3_twin_coverage.py` — **9 rules**: wavelet under MCAR 10 % is insufficient, not a
wrong twin; every twin control records emissions and comparisons; a centred twin over complete
data is detected (wavelet, STFT, Butterworth, SAX); a twin compared many times that never fails
is a failed declaration; coverage apart from causality; an isolated NaN blanks exactly one
support and blocks reduce coverage while causality and restart hold; the support boundary emits
exactly one output and a NaN inside it is never filled; restart under missingness is consistent
or insufficient, never wrong.

## K4 — observable ingestion errors

* **Boundary.** `validate_envelope` (repository copy and the packaged copy, byte-identical, parity
  green, `predictor-olap-store` **0.1.1**, digest re-pinned) refuses `data_consumed` items that
  are not `{id, digest, eligibility_state}` objects and units that are not objects, as the typed
  `EnvelopeRefusal` the store's web maps to **400**. The internal `AttributeError` the malformed
  d3 envelope caused inside the store is no longer reachable by that input; the store's own
  catch-all (`except Exception → 503 "database error"`) lives in the sibling `data-warehouse`
  repository and is recorded for its owner, not patched here.
* **Loader/outbox.** `olap_loader_duckdb` treats 400/422 as permanent with the typed reason; any
  other non-201 answer keeps the entry pending with a `.retry.json` beside it — HTTP status, class
  (`RETRYABLE_TRANSPORT` / `RETRYABLE_AUTH` / `RETRYABLE_SERVER_ERROR`), bounded and redacted
  reason, attempts, first/last seen — moved to `receipts/` when the entry leaves pending.
  `health` reports `retrying` and raises `attention_required` when the same bytes are answered
  5xx three times. Sidecars are never counted as entries.
* `tests/test_olap_ingest_diagnostics.py` — **10 rules**, with a stand-in store and with the
  **real** `data_warehouse_service` on a disposable DuckDB cube carrying the candidate package:
  boundary refusals; permanent 400 → `failed/` with reason; 503 → pending with status/class/reason
  /attempts, secrets redacted, sidecar not an entry; attention after three 5xx; transport and
  401 with their classes; recovery loads, diagnosis to receipts, second drain posts nothing; the
  real service answers the malformed envelope **400 not 503** and a good one 201; real service +
  loader: permanent, outage, recovery, no duplicate.
* **Adoption** (`tools/store_package_adopt.py`, the recorded procedure): inventory, backup of
  0.1.0 (`backup/` + manifest), rehearsal (the K4 real-service rules as the rehearsal),
  wheel built by the test interpreter and installed `--no-deps` into the host venv (the first
  attempt failed before touching the service: `--no-build-isolation` needs a backend the host
  venv does not carry), restart of `crispdm-data-warehouse-olap.service` (active, `NRestarts 0`),
  post-check, receipts in `satoshi-store-adoption-20260917T05{3708Z,3728Z-adopt,3759Z-adopt2}`.
  **0.1.1 is live: the malformed d3 envelope answers 400 in production.** The loader unit was
  restarted to run the diagnostics; heartbeat healthy, `attention_required: false` after the
  malformed envelope's dead letter was adjudicated `DEAD_LETTER_SUPERSEDED` by the corrected one.
* **An error of mine in the first post-check, recorded, not undone.** Its idempotency probe
  posted "the last file of `loaded/`" by name — `envelope-fadabebd…`, campaign
  `predictor::governed_config`, `result_class DEVELOPMENT`, `envelope_sha256 475b93fb…`, a
  route smoke of an earlier session that the Postgres-era cube had loaded and this DuckDB cube
  had never seen — and the store loaded it: `dim_campaign` 2→3, `dim_campaign_run` 2→3,
  `fact_campaign_unit` 6391→6392, consumption unchanged. A legitimate governed DEVELOPMENT
  envelope, no terminal, no MECHANICAL row touched; an unplanned write to production all the
  same. The probe now posts only an envelope the cube already holds (`dim_campaign_run`), else
  skips. Disposition of that row is Musashi's to rule; nothing is deleted.

## K5 — measure only what is affected, and close

Scope decision, stated so it can be checked: the amended probe changes `response_probe` for all
nine operators and the amended twin rule changes `non_causal_twin` for the seven with twins;
both live in `df_d3_acceptance.py`, whose digest every row carries, so no per-test reuse of v1
evidence is verifiable at row level. The whole predeclared scope is re-measured under the new
freeze: **504 bank units + 7 toys × 9 operators × 12 tests**, ~3 h wall on two workers; v1
stays as it is.

K5_PLACEHOLDER

## Prepared, not executed

`12_PREPARACION_UTILIDAD_REPRESENTACIONES_2026_09_17.md`: the per-variable representation-utility
experiment — raw branch mandatory, one predeclared objective and margin, walk-forward blocks with
purge, comparable budgets, train-only fit, `DEVELOPMENT` envelopes, no ranking. Passing mechanics
means nothing about prediction, information or trading.

## Open, with owners

| item | owner |
|---|---|
| store: `except Exception → 503 "database error"` disguises an internal defect as unavailability (`data-warehouse` `web.py`) | data-warehouse owner; recorded, not patched here |
| the DEVELOPMENT envelope `475b93fb…` my post-check loaded into the production cube | Musashi to rule; nothing deleted |
| index-loss root cause, Metabase driver, terms research | separate fronts, unchanged |
