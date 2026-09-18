# Return: D3 O1–O4 — calibration re-derived, one truthful resume outcome, pilot revalidated, next design

Order: `docs/handoffs/MUSASHI_D3_N1_N5_REVIEW_AND_O1_O4_2026_09_17.md` (`0b18cad`), over the review of
`9880f45`. Executed without pausing between commits; one machine (cheap re-reading, no dispatch);
no GPU, live, finance, reserve, historical deletion or new D3 campaign; service and foreign loads
untouched. `NON_GOVERNING` throughout.

**PRE** (reviewer's reproducer at `9880f45`, `docs/audits/evidence/d3_k5_20260917/O1_O2_PRE_POST.txt`):
a copy of the real MAD record with `upper_bound` set to 0 → `calibration_supports (True, None)`;
`advances=0/rate=0` with the advance still in `per_sim` → `calibration_record_problems []`; a resumed
attempt with `ADVANCES` in the summary and no `contrast.json` → `outcome=ADVANCES, score=null,
resumed=true, refusal=SCORE_UNVERIFIED`. **POST** (same reproducer, unchanged): the copy cannot be
sealed (`upper_bound 0.0 is not the derived 0.008786971227311154`); through the successor API:
support `False` with the derived bound, counts refused (`advances 0 is not the derived 1`,
rate and bound likewise), the resumed attempt returns **`outcome=SCORE_UNVERIFIED`**, `score=null`,
typed refusal, the old summary under `history`; an attempt without a recorded job is not reused.

## O1 — calibration recounted from the simulations

Red first (`tests/test_df_utility_harness.py`, six O1 rules), then closed in
`tools/df_utility_harness.py`:

* `derive_calibration(rec)`: every simulation validated — a scored one carries exactly
  `index, seed, outcome, delta_mean, delta_lower`, a failed one `index, seed, outcome, why`;
  indices are `0..n_sims−1` in order and unique, seeds unique and non-negative, deltas finite,
  the label **consistent with `delta_lower` and the margin** (`ADVANCES` iff `delta_lower > margin`);
  `scored`, `failed`, `advances`, the rate and the Clopper–Pearson bound are **derived** and
  compared with the record's summaries; any disagreement is a problem. A partial denominator,
  duplicate index/seed, NaN, unknown label, missing field, bound 0, counts changed with
  `per_sim` intact (even with a valid digest) — all refuse to seal (`ProtocolRefusal`).
* **Failure policy, declared before use** (`FAILURE_POLICY = WORST_CASE_FAILED_COUNTED_AS_ADVANCES`):
  the rate stays `advances / scored` with failures counted apart, but the **decision bound** is
  Clopper–Pearson over `advances + failed` of `n_sims`: selective failures can only enlarge the
  bound. Counting failures apart never proved the scored-only estimate valid; this policy does
  not need it to be.
* `calibration_supports(protocol, operator, n, record=, harness_sha256=)`: operator declaration,
  protocol base, family, length, multiplicity, then the **sealed plan** (generator, simulations,
  length, confidence) against the protocol's, margin/blocks/window/target/model, and the **harness
  code that applied** (record's `harness_sha256` vs the file at the run's frozen commit); the
  decision is gated on the **derived** decision bound. The record's own plan/confidence must
  agree with itself. No extra or missing field becomes an authorising default (exact key set,
  exact plan set).
* **Re-verification from conserved results**, no simulation repeated, margin/confidence
  untouched (`tools/df_utility_reverify.py`, receipts `utility_rehearsal_v7/REVERIFY.3.json`,
  `utility_pilot_v2/REVERIFY.json`, copies under `docs/audits/evidence/d3_k5_20260917/O1_*`):

| run | operator | stated | derived (advances/scored +failed) | derived bound | decision bound | supports |
|---|---|---|---|---:|---:|---|
| `utilreh-v7` | `cusum_causal` | 1/239 | 1/239 +0 | 0.01969 | 0.01969 | no (> 0.0125) |
| `utilreh-v7` | `delta_run_length` | 0/239 | 0/239 +0 | 0.01246 | 0.01246 | yes |
| `utilreh-v7` | `mad_extremes_trailing` | 0/239 | 0/239 +0 | 0.01246 | 0.01246 | yes |
| `utilpilot-v2` | `cusum_causal` | 0/538 | 0/538 +0 | 0.00555 | 0.00555 | yes |
| `utilpilot-v2` | `delta_run_length` | 0/538 | 0/538 +0 | 0.00555 | 0.00555 | yes |
| `utilpilot-v2` | `mad_extremes_trailing` | 1/538 | 1/538 +0 | 0.00879 | 0.00879 | no (> 0.00556) |

  Every stated value equals its derived value; **decision delta: 0 contrasts** in both runs.
  `INCONCLUSIVE` where evidence is missing (the slow control of v7 ended `RESOURCE_EXCEEDED` and
  is carried as such, not rebuilt).

## O2 — a resumed attempt has one truthful outcome

* `run_isolated` on an existing attempt: the job the caller brings is bound to the **recorded**
  `job.json` (any difference, or no recorded job → nothing reused); then the evidence is re-verified
  (bytes vs declared and runner-verified digests, JSON, schema, contrast and protocol identity,
  finiteness, summary/file agreement; a preparatory calibration record is re-validated with O1 and
  bound to the job's operator, protocol base and plan). Failure → **`outcome = SCORE_UNVERIFIED`**,
  `score = None`, typed `refusal`, the old summary only under `history`. The reporter builds no
  metric from it and the envelope item carries `terminal_state = SCORE_UNVERIFIED` and
  `UNAVAILABLE` values (`envelope_items`; `COMPLETE` requires a verified score). No automatic re-run.
* Tests: absent, altered bytes, discordant id, discordant protocol, unparseable, positive identical;
  job binding (changed operator, no job); preparatory record (other operator, tampered bound);
  and the **entry point** resumed over an attempt whose evidence is gone
  (`tests/test_df_utility_run_order.py::test_O2_the_entry_point…`): `SCORE_UNVERIFIED` in the
  outcome, the receipt and the envelope item, no second terminal, no `ADVANCES` anywhere.
* **The pilot's resume, checked** (`REVERIFY.json → resume_diff`): frozen `be1f170`, resumed
  `85246fa`; files changed: three evidence JSONs, the N return, `tests/test_df_utility_run_order.py`,
  `tools/df_utility_run.py` (the id map). `df_utility_harness.py`, `df_d3_operators.py`,
  `df_d3_contract.py`, `df_d3_acceptance.py`, `df_d3_design.py` — **identical digests at both
  commits**; same population, data, protocol, operators, calibration and score logic. Nothing
  scientific changed; the recorded difference is operational. No new identity was needed.

## O3 — the pilot closed and its limits

Twelve contrast units and three calibrations compared **metric by metric** with the cube
(`CONTENT_CHECK.O3.json`, both runs): all equal (`utilpilot-v2` 12 units; `utilreh-v7` 6 units, the
slow control carried as not completed). No campaign started. Negative and inconclusive results
kept. Readable table (`O3_UTILPILOT_V2_TABLE.md`; losses are MAE on the return target, 4 blocks):

| unit | operator | raw | transformed | Δ | lower (1−α/9) | paired rows | cpu s | outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bumps | cusum | 0.18053 | 0.20386 | −0.02333 | −0.05163 | 2044/2048 | 1.04 | DOES_NOT_ADVANCE |
| bumps | delta_run_length | 0.18053 | 0.18337 | −0.00283 | −0.00810 | 2043/2048 | 0.64 | DOES_NOT_ADVANCE |
| bumps | mad_extremes | 0.17999 | 0.20476 | −0.02477 | −0.05323 | 2029/2048 | 0.66 | INCONCLUSIVE (bound 0.00879 > α/9) |
| sinusoid | cusum | 0.34573 | 0.40464 | −0.05891 | −0.07944 | 2044/2048 | 1.07 | DOES_NOT_ADVANCE |
| sinusoid | delta_run_length | 0.34573 | 0.34798 | −0.00225 | −0.00580 | 2043/2048 | 0.65 | DOES_NOT_ADVANCE |
| sinusoid | mad_extremes | 0.34647 | 0.40581 | −0.05934 | −0.07504 | 2029/2048 | 0.65 | INCONCLUSIVE |
| steps | cusum | 0.45855 | 0.56744 | −0.10889 | −0.14785 | 2044/2048 | 1.04 | DOES_NOT_ADVANCE |
| steps | delta_run_length | 0.45855 | 0.45310 | +0.00545 | −0.01003 | 2043/2048 | 0.65 | DOES_NOT_ADVANCE |
| steps | mad_extremes | 0.46035 | 0.57065 | −0.11030 | −0.16021 | 2029/2048 | 0.66 | INCONCLUSIVE |

Null scope for every row: `white_null` × 538 at n = 2048, 0.95 bound, 0 failed. **What this does
not say**: `DOES_NOT_ADVANCE` is not equivalence and does not show an operator useless on other
domains, horizons or models; the nine contrasts are correlated (same operators, similar series)
and were never nine independent experiments; the augmented branch was not tested.

**Next DEVELOPMENT design, per variable** (`12A_NEXT_DEVELOPMENT_UTILITY_DESIGN_2026_09_17.md`,
`tools/df_utility_next_design.py`, 5 tests; sealed instance `O3_NEXT_DEV_DESIGN.json` `386fe033…`):
hypotheses `H_T` (raw vs transformed) and `H_A` (**`raw_wide` vs augmented** — the capacity control,
a new harness branch of 2·window raw lags so raw+R is never compared against raw alone); target,
horizon, model, window, blocks, margin and α **inherited from the sealed pilot protocol and checked**
(a differing threshold refuses); one family per (unit, variable) with α/6 and a derived plan
(358 simulations); **independent replication** (bank seeds 12 selection / 13 replication, disjoint
from the pilot's seed 11), counted not pooled; stages separated by name (flow diagnostic done,
development selection = this, public confirmation and financial revalidation out of scope and
refused by the builder). **Nothing executed**; no reserve opened; pilot not widened.

## O4 — operation, disposition, continuous closure

The empty corrective envelope `2e68209e…` (`utility-rehearsal-utilreh-v4-recovery`, four
`UNAVAILABLE` units) is **retained, NOT admissible scientifically**, and the pending manifest now
names its successor (`utility-rehearsal-utilreh-v4-recovery-2`, `57e5f959…`) and the approval
scope. Proved excluded today (`O4_EMPTY_ENVELOPE_EXCLUSION.json`): 4 fact rows and 1 run row
exist for it; **0 rows** in `gov_scientific_evidence` (view = dispositions `INCLUDED_CURRENT` ∧
`CURRENT_SCIENTIFIC` only) and **0** in `gov_mechanical_evidence`; disposition pending. It will be
published in the next window the warehouse already needs; no restart for it.

## Closure

**Suites** (trading-stack, `crispdm-run`): `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py
olap/store/tests` — **1231 passed, 6 skipped**, 406 s (utility harness 36 · run order 7 ·
reverify 4 · next design 5 among them). **Commits**: `1a5dab8` (merge of the order) · `3050195`
(O1–O2) · `8826263` (verifier, O4 evidence) · this return (design, docs). Not repeated: the 538 and
239 simulations, the D3 mechanics, the warehouse publication window. Omitted by design: any
governed run of 12A. Terminal outbox and OLAP outbox untouched (no new evidence persisted;
receipts and checks are files under the run roots and the evidence folder). The owner works
remotely; sync of workers to the final commit via the alias map (Tailscale route if the home
route fails).

Ending: **`UTILITY_PILOT_REVALIDATED_AND_NEXT_DEVELOPMENT_DESIGN_READY`**.
