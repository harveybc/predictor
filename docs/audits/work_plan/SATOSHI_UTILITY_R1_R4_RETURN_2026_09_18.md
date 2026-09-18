# Return: utility R1–R4 — code-bound reuse, cache recovery, governed instrument validation, closure

Order: `docs/handoffs/MUSASHI_UTILITY_Q1_Q4_REVIEW_AND_R1_R4_2026_09_18.md`, over the review of
`0352328`. Executed without pausing between blocks. No GPU, financial data, live, reserve, broad
sweep or service restart; `utildev-v1` not repeated; the six remaining operators not launched.
CPU of this order's rehearsals and instrument work: **34 s (R2 rehearsals) + 203 s (instrument)
= 237 s of the 7 200 s ceiling**, failed attempts included (none). Test suites are reported apart.
No owner action.

**PRE** (reviewer's reproducer at `0352328`, `docs/audits/evidence/d3_k5_20260917/R1_R2_PRE_POST.txt`):
`operator_behavior_changed true, same_key true, stale_cache_hit true`; `MISSING_RECORD_LOOKUP
(None, 'absent')`; `RECOVERY_EXCEPTION OSError errno 39`. **POST** (reproducer unchanged):
`same_key false, stale_cache_hit false`; `(None, 'incomplete entry (record absent)')`;
`RECOVERY {'status': 'RECOVERED_INCOMPLETE_ENTRY', …, 'quarantined': …}`.

## R1 — reusable computations bound to the executing code

Regression rules declared first (`tests/test_df_utility_calibration_cache.py`, R1: behaviour
changed without declaration; helper and module change in a **fresh process**; unchanged code
under other labels; legacy records), then `tools/df_utility_harness.py`:

* `scientific_code_identity(operator)`: digest of the **code objects** of every function on the
  operator's class hierarchy (bytecode, constants with nested code, names, variables, arity,
  flags — independent of file location and line numbers), the defining module's bytes, the
  helper modules it inherits (`df_d3_contract`, `df_d3_operators`), the harness, and the numeric
  environment (python/numpy/scipy) with the **declared scope** `SAME_ENVIRONMENT_ONLY` (version
  strings identify the environment; they do not demonstrate cross-CPU bit identity). No
  whole-repository label; no administrative label.
* The computation key is `df_utility_calibration_computation.v2` and carries `code_identity`;
  records are **`df_utility_calibration.v4`**. A method replaced in memory or edited on disk
  changes the key and never hits; a copy of the same code elsewhere keeps it; a family relabeled
  keeps it. **Schema transition**: v1 (pilot), v2 (utildev-v1), v3 (Q2 rehearsals, key without
  code) remain valid historical records, compared on their own terms (`_as_legacy_key`), and a
  v3 entry never hits a code-bound key; nothing regenerated — the 36 historical calibrations stay
  as they were.

## R2 — cache miss recovered end to end

Reproduced first (PRE above), then `tools/df_utility_calibration_cache.py`:

* **Typed recovery**: an entry without record or META, malformed, or left by an interrupted
  publication is a typed miss (`incomplete entry (…)`) and, at the next store, is moved to
  `quarantine/<key>.<ns>.<pid>/` with its reason; the new record becomes usable
  (`RECOVERED_INCOMPLETE_ENTRY`). Exercised with the **real isolated child**: incomplete entry →
  `MISS_PRODUCED` with `why_miss` → recovery → the next child `CACHE_HIT` with identical bytes.
* **Scientific equality apart from cost/provenance**: a later record with the same computation,
  simulations, counts and bounds but another cost is a `DUPLICATE_PRODUCER`; a scientifically
  different record for the same key is `CONFLICT_RECORDED` with a `CONFLICT.json` disposition
  `PENDING` and **nothing served** until resolved. Concurrent independent producers store once
  (8 threads → 1 STORED, 7 duplicates).
* **Accounting idempotent by attempt**: repeated reports of one hit count once; measured
  verification seconds are kept apart from projected avoided CPU (`avoided_cpu_seconds_projected`,
  from the producers' recorded cost).
* **Governed demonstration** (`R2_UTILCACHE_V2{A,B}_*`, `R2_CACHE_STATUS.json`): `utilcache-v2a`
  (3 misses under v4 keys, stored) → one entry's `record.json` removed on purpose →
  `utilcache-v2b`: cusum `MISS_PRODUCED` / `RECOVERED_INCOMPLETE_ENTRY`, delta and mad
  `CACHE_HIT`; both runs reconciled `[]`, content-equal in the cube (live query at run time; the
  content checks are separate warehouse queries), re-verified with the shared records bound by
  computation (`source` declared). Cache: 6 unique computations, 24 unique simulations, 5 hits,
  1 quarantined incomplete entry, 0 conflicts.

## R3 — instrument validated before any extension

`tools/df_utility_instrument_run.py` (tests `tests/test_df_utility_instrument_run.py`, 4 rules):
design **sealed before any outcome** (`DESIGN.json`, `R3_UTILINST_V1_DESIGN.json`, `160b84de…`): generators and
effects from Q3, **fresh seeds 200–211** (disjoint from Q3's 100–105), n = 2 048, 12 replicates,
4 blocks, both pairs, MAE, margin 0, one family of two contrast ids → **α/2 = 0.025 at 0.95 →
119 simulations derived** (not 358), criteria fixed with their finite-sample justification
(positives ≥ 10/12; null ≥ 11/12 DOES_NOT_ADVANCE with ≤ 1 advance — expected 0.3, P(≥ 2) =
3.7 %; loss ≥ 11/12 with delta < 0; leak 12/12 refused). Budget checked before calibration and
before every control through the isolated runner. Governed: cost pilot (6 sims per pair) →
projection 372 s ≤ 7 200 − 34 s → calibration campaign (2 units; both `MISS_PRODUCED` and
offered to the cache) → controls campaign (60 units) → terminals with the children's instants,
costs, control, seed, pair → reconciliation `[]` for both → DEVELOPMENT envelope `722ab3bf…`.

| control | criterion | observed | CI95 | status |
|---|---|---|---|---|
| positive H_T (raw vs transformed) | ADVANCES ≥ 10/12 | **10/12** (deltas +0.08 … +0.14) | 0.52–0.98 | MET |
| positive H_A (raw_wide vs augmented) | ADVANCES ≥ 10/12 | 0/12 decidable; 12/12 INCONCLUSIVE_UNCALIBRATED (deltas +0.08 … +0.13) | — | **INCONCLUSIVE_CALIBRATION** |
| null contrast | DOES_NOT_ADVANCE ≥ 11/12, ≤ 1 advance | 12/12, 0 advances (deltas within ±0.005) | 0.74–1.00 | MET |
| information loss | DOES_NOT_ADVANCE with delta < 0 ≥ 11/12 | 12/12 (deltas −0.06 … −0.14) | 0.74–1.00 | MET |
| future leak | REFUSED "not causal" 12/12 | 12/12 | 0.74–1.00 | MET |

Calibration support, apart: H_T 0/119, derived bound 0.0249 ≤ 0.025 → supports; H_A **1/119**,
derived bound 0.0392 > 0.025 → **no support** (a failed calibration is INCONCLUSIVE, not
negative utility). **Instrument outcome: `INCONCLUSIVE`.** Diagnosis from completed evidence:
119 simulations support only at zero advances; a plan tolerating one advance needs n ≥ 190 —
sealed into the successor design, **not** re-run with the same seeds, nothing tuned. Twelve
replicates estimate detection at these effect sizes only; the positive generator is aligned to
this operator and validates none of the six other representations. CPU 203 s.

**Next-stage design for the six remaining operators, executable and bounded, not launched**
(`R3_NEXT_STAGE_DESIGN_6OPS.json`, `366880…`): same 6 families and map, eligibility per operator
from the verified cells (all accepted), α/12 per family, 718 simulations per contract, 72
contracts = 12 computations under the cache; resource and sensitivity limits in 12C.

## R4 — evidence closed, plan updated

* `utildev-v1` **reclosed from conserved bytes only** (`CLOSE.3.json`, `R4_UTILDEV_V1_CLOSE_3.json`):
  TOTAL, four checks true, 0 proposals, 0 unverified candidates, 0 verdict changes.
  **Compatibility boundary**: its 36 records are v2 (no computation key); they are verified on
  their own terms (protocol base and family) and are not re-keyed; no new historical scores.
* New results verified files → parent → accounting → warehouse by content: `utilcache-v2a/b`
  (6 units each), `utilinst-v1` (62 contrast units + 2 calibration units, all equal;
  `R3_UTILINST_V1_*CONTENT_CHECK.json`), re-verified (`R3_UTILINST_V1_REVERIFY.json`, decision
  delta 0; the verifier now binds an attempt named by control and replicate to its job). Retained
  receipts vs live queries are labelled in every report.
* 12C updated with the instrument status and what remains untested (feature engineering,
  variable-specific denoising, learned representations, lake-wide selection, other regimes,
  horizons, targets, financial data); 09_ADOPCION updated.

**Suites** (trading-stack, `crispdm-run`, CPU): `tests/test_d3_*.py tests/test_df_*.py
tests/test_olap_*.py olap/store/tests` — **1270 passed, 6 skipped**, 631 s, run at the tree before the verifier's control/replicate identity rule; after that rule the reverify suite was re-run: 6 passed. Targeted: calibration cache 13 · harness 41 · reverify 6 · instrument 4 · dev close 7 · dev run 4 · next design 6 · controls 4 · run order 8. Skips: the store suite's own skip and `systemd-run`-gated children (present here).
**Commits**: `9fdc3c7` (merge) · `b7dba70` (R1–R2) · `f4f4a3b` (R2 evidence) · `7f8e6c9` (R3) ·
the closing commit that names this return. Workers synced to the final commit. Pending: the
empty-envelope disposition (next needed maintenance window); index/Metabase/terms separate.

Ending: **`UTILITY_CODE_BOUND_REUSE_AND_INSTRUMENT_SUCCESSOR_REVIEW`** — instrument outcome
**INCONCLUSIVE** (H_T side met every criterion; H_A undecided by its calibration).
