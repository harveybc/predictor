# Return: utility P1–P4 — hypotheses calibrated exactly, one experimental contract, bounded development, reproducible closure

Order: `docs/handoffs/MUSASHI_UTILITY_O1_O4_REVIEW_AND_P1_P4_2026_09_17.md` (`a8ba857`), over the
review of `d402532`. Executed without pausing between blocks; one machine (the projection fit the
ceiling with room, no dispatch); no GPU, live, finance, reserve, historical deletion, new
population or new D3 campaign. `NON_GOVERNING` throughout. Owner remote; no owner action.

**PRE** (reviewer's reproducer at `d402532`, `docs/audits/evidence/d3_k5_20260917/P1_P2_PRE_POST.txt`):
`CHILD_MARGIN_CHANGED_ACCEPTED` (a family protocol with margin 99 under a correct outer digest
passed) and `CALIBRATED_BRANCHES [['raw', 'transformed']]` for a protocol declaring
`raw_wide/augmented`. **POST** (reproducer unchanged): `DesignRefusal: … protocol digest does not
seal the protocol; protocol base digest differs; protocol margin 99.0 is not the inherited 0.0`;
a calibration for H_A observes `[['raw_wide', 'augmented']]`, the record names the pair and widths
`{a: 8, b: 8}`, and transferring it to H_T is refused (`calibrated for another branch pair`).

## P1 — each hypothesis calibrated with exactly its pair

Red first (`tests/test_df_utility_harness.py`, P1 rules), then `tools/df_utility_harness.py`:

* `calibrate(protocol, operator, plan, branch_a, branch_b)` simulates through the **same
  `contrast` callable** the experiment uses, with the same pair; a spy on the callable sees
  `raw_wide/augmented` for H_A and `raw/transformed` for H_T, in simulation and in the score.
* The record is **`df_utility_calibration.v2`**: pair, `widths` (columns per branch at this window),
  `rows_policy` (`PAIRED_EMITTABLE_ROWS_ALL_BLOCKS_OR_INSUFFICIENT`), plus everything v1 carried
  (operator declaration, protocol base, family, length, plan, margin, blocks, window, target,
  model, every simulation). A v1 record (the pilot's) is the raw/transformed pair by construction.
* `calibration_supports(…, branch_a, branch_b)` refuses another pair, other widths, another rows
  policy — on top of O1's derived bound, plan, margin and harness-code checks. Records are **never
  shared** between hypotheses; no cache was used (see the observation under P3).
* Generator, margin, confidence and plan stay predeclared; 358 simulations are **per contract**:
  36 contracts (6 families × 3 operators × 2 hypotheses), declared in the design and counted in
  the projection. No unfavourable bound licensed more simulations (none was re-run).

## P2 — one experimental contract, explicit replication, derived boundary

* **Recursive validation** (`tools/df_utility_next_design.py`, `df_utility_dev_design.v2`; tests
  `tests/test_df_utility_next_design.py`, 6 rules): every family protocol is rebuilt from its
  sealed document and checked — protocol digest, base digest, **every inherited field** (target,
  horizon, model, window, blocks, margin, alpha, ridge λ, min rows), family = member list, members
  = operators × hypotheses (duplicate, missing, swapped H_T/H_A all refused), pair per hypothesis,
  contract per member (widths, policy, plan), α/|family|, simulations derived from α_adj and the
  **bound resource's length** (`UNIT.json`, never `--n`), eligibility from the bound cells record
  (a refused cell refuses the design), execution `NONE`. Re-hashing the outer digest is not enough.
* **Replication as an explicit map**: `bumps→bumps`, `sinusoid→sinusoid`, `steps→steps` (seed 12 →
  seed 13), each pair checked from the bank: same generator family, perturbation, SNR,
  missingness, length and variable; different seed **and** different data digest; disjoint from
  the pilot's seed-11 units. A map to another regime or to a copy with the same data refuses.
  α/6 per family is stated **not** to be a global control; nothing is presented as confirmed.
* **Boundary from consumed support** (`tests/test_df_utility_harness.py`, P2 rules): `features`
  returns, per row, the earliest row it consumed (widths, gaps, emission delays, availability);
  `blocks(rows, protocol, support_start, train_reach)` takes training rows whose consumption
  (label to t+h, representation to t+reach) ends strictly before the earliest row the validation
  slice consumes; each block records `boundary = {max_train_consumed_row, min_validation_support_row}`
  by row identity, re-derived after the block's own refit, and a violation refuses the contrast.
  Eight raw lags widen the purge; a gap at the validation front pushes the training end back.
  The paired comparison and its denominators are unchanged.
* H_T reworded to **improvement** (superiority at the sealed margin), margin and inference
  untouched; equal width is stated to equalise the number of inputs only.
* Successor design **12B sealed before measuring**: `P2_NEXT_DEV_DESIGN_12B.json` (`55ef8321…`),
  predecessor `386fe033…` preserved as history, never executed. Population, h = 1, ridge λ = 1,
  margin 0, window 4, blocks 4, α = 0.05: unchanged.

## P3 — bounded development execution

`tools/df_utility_dev_run.py` (tests `tests/test_df_utility_dev_run.py`, 4 rules): design validated
against the pilot's inherited protocol and the bank; **governed cost pilot** (campaign
`utildev-v1-cost-pilot-utility-calibration`, one child per contract type, 6 simulations each);
projection contract by contract; families in the design's order through the governed entry point
(freeze-pre → calibration campaign with one unit per contract → before_run → child → seal one
protocol per contract → contrasts campaign → before_run → child with its pair → terminal with
the child's instants, cost, hypothesis and pair → reconcile → DEVELOPMENT envelope); aggregate
ceiling over calibrations, contrasts, failures and retries read from the attempts (plan instead
of launch when the projection exceeds it; incompletes kept when exhausted mid-way — both tested).

| measured | value |
|---|---|
| cost pilot per simulation (cusum H_T/H_A, delta H_T/H_A, mad H_T/H_A) | 0.97 / 0.94, 0.38 / 0.42, 0.86 / 0.69 s |
| projection (36 contracts × 358 + 36 contrasts × 2 s + pilot) | 9 232 s ≤ 14 400 s ceiling |
| spent (78 attempts, read from `outcome.json`) | **5 852 s** CPU; ~2 h 20 min wall, sequential |
| families | 6 of 6 complete; 12 campaigns registered before any child; every reconciliation `missing_units: []` |
| envelopes | 6 DEVELOPMENT (`97b0a969…`, `ecaba971…`, `b2ef4163…`, `553adecd…`, `71f4d674…`, `ad0402a1…`), each 6 units, loaded |
| outboxes | terminal 0 pending (1 657 sent); OLAP loader took all six |

One correction on the way, recorded: the first launch refused at the cost pilot
(`the run's family is not the sealed design's family`) because a sealed design file is key-sorted
and the family configuration took the hypotheses in key order; fixed (`35799b7`, regression), the
empty root removed, relaunched. No child had started.

**Observation for review**: the calibration seed is `protocol.seed + 1000 + contract index` and
the design inherits `seed = 3` for every family, so the 36 contracts simulated **six distinct
null draws, each six times** (identical advances/scored per contract type across families:
cusum H_A 0/358, cusum H_T 1/358, delta H_A 3/358, delta H_T 0/358, mad H_A 0/358, mad H_T
1/358). Each record is still bound to its own family, pair and protocol and was measured, not
cached; validity is unaffected (the null does not depend on the unit). A verified cache with
exact contract equivalence — or per-family seeds — would have avoided ~4 900 s of the 5 852 s.
Left as a finding, not changed mid-campaign.

## P4 — reproducible closure

`tools/df_utility_dev_close.py` (tests `tests/test_df_utility_dev_close.py`) over the six family
roots: **files** (bytes vs declared and runner-verified digests, job bound to the freeze, record
pair bound to the job, derived bounds, re-decision) → **parent** (recorded outcome per contrast
equals the file's) → **accounting** (both campaigns reconciled per family) → **warehouse**
(`CONTENT_CHECK.json` per family: every calibration and contrast metric the cube holds equals the
verified file; 12 units × 6). All four checks **True**; decision delta 0 in every family. Evidence:
`P4_UTILDEV_V1_{CLOSE,REPORT}.json`, `P4_UTILDEV_V1_TABLE.md`, `P4_utildev_v1_families/*`.

Per contract (selection deltas A−B for bumps / sinusoid / steps with the 1−α/6 lower bound;
derived null bound; mapped replica deltas). Losses are MAE on the return target, 4 blocks,
≈ 2 029–2 044 paired rows of 2 048:

| operator | hyp | selection Δ (lower) | bound | outcome | replica Δ | outcome |
|---|---|---|---:|---|---|---|
| `cusum_causal` | H_A | −0.0013 (−0.0069) / −0.0349 (−0.0653) / −0.0254 (−0.0767) | 0.0083 | DOES_NOT_ADVANCE | +0.0028 / −0.0037 / −0.0063 | DOES_NOT_ADVANCE |
| `cusum_causal` | H_T | −0.0080 (−0.0184) / −0.0331 (−0.0695) / −0.1295 (−0.2257) | 0.0132 | INCONCLUSIVE_UNCALIBRATED | −0.0076 / −0.0274 / −0.0708 | INCONCLUSIVE_UNCALIBRATED |
| `delta_run_length` | H_A | −0.0009 (−0.0028) / −0.0240 (−0.0375) / −0.0101 (−0.0201) | 0.0215 | INCONCLUSIVE_UNCALIBRATED | +0.0026 / −0.0037 / −0.0032 | INCONCLUSIVE_UNCALIBRATED |
| `delta_run_length` | H_T | −0.0018 (−0.0043) / +0.0027 (−0.0174) / +0.0039 (−0.0085) | 0.0083 | DOES_NOT_ADVANCE | −0.0015 / −0.0018 / +0.0018 | DOES_NOT_ADVANCE |
| `mad_extremes_trailing` | H_A | +0.0009 (−0.0129) / −0.0361 (−0.0603) / −0.0179 (−0.0329) | 0.0083 | DOES_NOT_ADVANCE | +0.0028 / −0.0043 / −0.0059 | DOES_NOT_ADVANCE |
| `mad_extremes_trailing` | H_T | −0.0071 (−0.0212) / −0.0352 (−0.0628) / −0.1297 (−0.2194) | 0.0132 | INCONCLUSIVE_UNCALIBRATED | −0.0052 / −0.0279 / −0.0735 | INCONCLUSIVE_UNCALIBRATED |

**Proposed for review: none.** No pair advanced anywhere (every lower bound is below the margin);
18 contrasts are decided `DOES_NOT_ADVANCE` under a supporting contract (H_A for cusum and
mad, H_T for delta), 18 are descriptive `INCONCLUSIVE_UNCALIBRATED` (their contract's derived
bound — 1/358 or 3/358 at 95 % — exceeds α/6 = 0.0083; not converted, not re-run). What this
does **not** say: `DOES_NOT_ADVANCE` is not equivalence and does not show these operators
useless on other domains, horizons, models or widths; the augmented deltas near zero say the
capacity control did its job (raw+R against raw of equal width), not that R carries nothing; a
favourable bound covers only the white null at n = 2048 under this probe. Nothing is publicly
eligible or licensed.

## Closure

**Suites** (trading-stack, `crispdm-run`): `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py
olap/store/tests` — **1244 passed, 6 skipped**, 616 s (utility harness 41 · run order 8 · reverify 5 · next
design 6 · dev run 4 · dev close 2 among them). Not repeated: the pilot, D3 mechanics, reserves.
Exclusions: no dispatch to workers (single host, projection within the ceiling); no cache of
calibration records (observation above); the pending empty-envelope disposition remains for the
next window the warehouse needs. **Commits**: `f91b422` (merge) · `2aa29c9` (P1–P2 harness) ·
`2fd5fc5` (12B) · `0a7d2dd` `35799b7` (P3) · `e1fad68` (P4 tools) · `e51b0f3` (12B methodology) ·
the closing commit that names this return. Workers synced to the final commit.

Ending: **`PER_VARIABLE_DEVELOPMENT_SELECTION_AND_REPLICATION_READY_FOR_REVIEW`**.
