# Return: utility Q1–Q4 — closure bound to the population, verified calibration reuse, instrument controls, coverage map

Order: `docs/handoffs/MUSASHI_UTILITY_P1_P4_REVIEW_AND_Q1_Q4_2026_09_18.md` (`a8ba857`… review of
`fb1833c`). Executed without pausing between blocks; no repetition of `utildev-v1`, no reserve,
no GPU, no financial data, no service restart, no index/Metabase. Bounded rehearsals of this
order: ≈ 45 s CPU (cache) + 19 s (controls) + tests, far under the 2 h ceiling. Owner remote; no
owner action.

**PRE** (reviewer's reproducer at `fb1833c`, `docs/audits/evidence/d3_k5_20260917/Q1_PRE_POST.txt`):
`EMPTY_DESIGN {four checks True}`; `UNVERIFIED_PROPOSALS {files_verified False …} 12`. **POST**
(reproducer unchanged): the empty design is `REFUSED: the design is not a sealed valid one …`
(typed `ClosureRefusal`); with the verifier forced to fail, `proposed 0`, `unverified_candidates
12`, every row `UNVERIFIED_PAIR`.

## Q1 — closure bound to the population, proposals conditioned

Red first (`tests/test_df_utility_dev_close.py`, 7 rules incl. API and CLI), then
`tools/df_utility_dev_close.py` (`df_utility_dev_close.v2`):

* The sealed design is validated (recursively, against the pilot's inherited protocol) and the
  population is **derived** from it: exact families, members, calibration contracts, pairs and
  replication map. The campaign report must name the design's identity. Omitted family, family
  not in the design, duplicate terminal, terminal not a member, calibration unit not a contract,
  family report of another run → **typed refusal**.
* Receipts bound to identity and to the whole population: the contrasts/calibration campaign keys
  must be this run's; `missing_units == []` alone does not reconcile; every member needs a
  terminal and every contract a calibration unit; the content check must name this run, cover
  every member and contract, and be equal on each — equality of what is present is not enough.
* **Closure policy** (documented in the artefact): PARTIAL closure allowed; a pair yields a
  verdict only when **both** families passed every mandatory check over their whole population;
  otherwise `UNVERIFIED_PAIR`, and any advancing candidate is listed apart as an *unverified
  candidate*, never under `PROPOSED_FOR_REVIEW`. The CLI writes the artefact with zero proposals
  on failure and exits non-zero; an incoherent design writes nothing.
* **Successor closure of `utildev-v1`** without re-measuring (`CLOSE.2.json`, evidence
  `Q1_UTILDEV_V1_CLOSE_2.json`, `Q1_UTILDEV_V1_TABLE_2.md`, delta `Q1_UTILDEV_V1_CLOSE_DELTA.json`):
  population {6 families, 36 members, 36 contracts, 3 pairs}, closure TOTAL, four checks True,
  **0 verdict changes**, 0 proposals, 0 unverified candidates. `CLOSE.json` (v1) preserved.

## Q2 — calibration cache: equivalence first, then saving

* **Canonical computation key** (`df_utility_harness.computation_key`): harness code, numeric
  dependencies (python/numpy/scipy), generator and parameters, seed, n, n_sims, confidence,
  operator kind/spec/params, branch pair and widths, target, horizon, model, window, blocks, min
  rows, ridge λ, logistic steps, prefix checks, blocks policy, inference, margin, **effective
  alpha**, failure policy, rows policy — **no family or unit label**. Records are now
  `df_utility_calibration.v3`, sealed with the key; the verifier recomputes the key from the
  record's fields; support and job binding use the key (labels apart) for v3 and keep the
  base/family checks for v1/v2. Tests: two families differing only in labels share the key;
  seventeen scientific fields each break it; H_T is never shared with H_A nor across operators.
* **Cache** (`tools/df_utility_calibration_cache.py`): a hit only when the stored bytes match
  their digest, the record passes the **current** verifier (every simulation recounted, bound
  re-derived), and its key equals the consumer's field by field; absent, corrupt, altered or
  incomplete → miss. Atomic store; a race stores once (8 threads → 1 STORED, 7 duplicate
  producers); a different record for the same key is quarantined, never served. Accounting keeps
  unique computations, producers, consumers, reads, verification time and saved CPU apart.
* **The child declares its source** (`MEASURED_HERE` / `MISS_PRODUCED` / `CACHE_HIT`) in
  `result.json`; the terminal tag `calibration_source` carries it; a hit runs **zero**
  simulations and writes the shared bytes; the consumer keeps its own campaign binding.
* **Governed rehearsal** (bounded, real child, real data-gov): `utilcache-v1a` (miss ×3, stored)
  then `utilcache-v1b` (hit ×3, `simulations_run_here 0`, identical `calibration.json` bytes and
  digests, same bounds), both reconciled `[]`, both content-equal in the cube (6 units each),
  both re-verified (`Q2_UTILCACHE_V1{A,B}_*`); cache status: 3 unique computations, 12 unique
  simulations, 3 hits, saved 6.4 s CPU, verification 2.4 s (`Q2_CACHE_STATUS.json`).
* **Replay over the 36 conserved records** (`Q2_UTILDEV_V1_CALIBRATION_REPLAY.json`): 6 unique
  computations, each with 6 records of one `per_sim` digest, 5 761 s CPU in total, 4 801 s
  avoidable — a replay projection, not a measured saving. Nothing deduplicated in the cube.

## Q3 — controls of known utility, coverage map

`tools/df_utility_controls.py` (tests `tests/test_df_utility_controls.py`): five controls, each
binding requirement, estimand, generator, pair, loss and a **predeclared** criterion and budget;
positive controls and calibration kept as distinct roles. Positive generator: next increment
= 1.0 × the unsigned extremeness score `|x − median16| / MAD16` (the score the operator emits;
4 or 8 raw lags cannot compute a 16-row median/MAD) — the differential advantage is by
construction, verified on the generators (correlation of the score with the increment > 0.3;
the signed-momentum generator's increment is uncorrelated with the unsigned score). Run at
n = 1 200, 6 replicates, seeds 100–105, 1 800 s budget (used 19 s), `Q3_CONTROLS.json`:

| control | expected | observed | met |
|---|---|---|---|
| null contrast | DOES_NOT_ADVANCE ≥ 5/6, ≤ 1 advance | 6/6, 0 advances, deltas within ±0.02 | yes |
| information loss | DOES_NOT_ADVANCE with delta < 0 ≥ 5/6 | 6/6 (deltas −0.07 … −0.11) | yes |
| future leak | REFUSED "not causal" 6/6 | 6/6 | yes |
| positive H_T | ADVANCES ≥ 5/6 | **4/6** (deltas +0.05 … +0.16 in 6/6; lower bound below 0 twice) | **no** |
| positive H_A | ADVANCES ≥ 5/6 | **0/6 decidable**: the fixture calibration of the pair (28 sims at 0.5, 1 advance) gives no support → INCONCLUSIVE (deltas +0.05 … +0.15 in 6/6) | **no** |

Instrument verdict: **`DOES_NOT_SEPARATE_OR_INCOMPLETE`** at this predeclared power design.
Nothing was tuned to pass: the sensitivity of a 4-block t-interval at n = 1 200 is below the
5/6 criterion for this effect size, and a 28-simulation fixture calibration is too fragile to
decide H_A. Both go to the next design, not to a re-run.

**Coverage map** (`12C_REPRESENTATION_COVERAGE_MAP_2026_09_18.md`): tested — 3 operators, 3
synthetic families at SNR 10 / white / no missingness / n = 2 048, h = 1, ridge, H_T and H_A;
not tested — the other 6 accepted operators, other regimes, horizons, the direction target,
feature engineering, per-feature denoising, compression, learned representations, feature
selection over the lake, anything financial. **Next bounded experiment, proposal only**: the
instrument first (positive controls at n = 2 048, 12 replicates, criterion ≥ 10/12, H_A
calibrated with the campaign's plan), then the 6 remaining operators on the same 6 families
under the cache (projected ≈ 2 500 s CPU, cost pilot before launch). Not launched. The doctoral
document was not touched.

## Q4 — closure

**Suites** (trading-stack, `crispdm-run`): `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py
olap/store/tests` — **1259 passed, 6 skipped**, 633 s (utility harness 41 · run order 8 · reverify 5 · next design
6 · dev run 4 · dev close 7 · calibration cache 6 · controls 4 among them). Exclusions: no
repetition of `utildev-v1`, no wide power study (its design is delivered instead), no dispatch
(no independent heavy task), pending empty-envelope disposition unchanged (next needed window).
**Commits**: `97107a3` (merge) · `6ee97ee` (Q1) · `03106e9` `5743721` (Q2) · `72bdd51` (Q3) · the
closing commit that names this return. Workers synced to the final commit.

Ending: **`DEVELOPMENT_CLOSURE_VERIFIED_CALIBRATION_REUSE_TESTED`**.
