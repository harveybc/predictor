# Return packet — C106–C121

**Date:** 2026-09-12
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_C106_C121_ORDER_2026_09_12.md`
**Audit it answers:** `MUSASHI_AUDIT_ROUND7_C87_C105_2026_09_12.md`
**Licence in force:** `READ_ONLY_RECOMPUTATION_AND_CPU_MECHANICS`
**Stopping at:** `T2_V4_SUBMITTED_FOR_EXTERNAL_REVIEW / B4_CLOSED /
PER_VARIABLE_V5_BANK_INSUFFICIENT_FOR_MUSASHI_REVIEW`

Zero GPU. Zero training. Zero scores. Zero confirmation. Zero live. Zero venue.

---

## 1. Confessions

1. **The T2 submission v4 does not itself show 1.3125 as an invalid,
   superseded value.** The reviewed builder at `4bf38b6f` applied the
   sign-test supersession to the *recomputed* screen, so its own block
   reads `published_value 1.0`, `published_value_valid true`. The code may
   not change under your record and C108 allows one attempt, so I did not
   rebuild it. The history is bound beside it in an addendum. The POST
   reports this item as NOT CORRECTED.
2. **The round-7 template flaw you found was mine:** the template hashed
   the historical `screen_adjudication` with 1.3125 while computing the
   correction beside it. The PRE freezes it through the real CLI and API.
3. **My first PRE run failed one item because of my harness, not your
   reproducer.** I copied it into a shallow temporary directory, and its
   argparse default `parents[4]` is evaluated even when `--predictor-root`
   is given. I fixed the depth and re-ran at the same bases before anything
   was edited. The committed PRE is that re-run.
4. **In round 7 my fixture replay ran over a private copy of the T2 root
   (`crispdm-pre-copies/t2_successor`, inventory `c61ea446`), not the root
   your record binds (`71c8ae7e`).** I noticed it this round by hashing both
   before C108, so the single attempt used the bound root.
5. **My PRE fact for C111 looked only at the outbox.** The real cube already
   held a B4 row, `b4_campaign_generation_v7_20260908`, loaded earlier from a
   translated summary. I found it before emitting. It is a first-observation
   stub (`NON_GOVERNING`, identity `UNAVAILABLE`, no run, no units), not a
   terminal closure unit, so C111's condition to emit held. It stays in the
   cube beside the new campaign.
6. **My prompt to the agent that built the successor census named the row
   keys but not the key of the row list.** The census uses `rows`; v5 expected
   `variables`. I closed it in v5 A2 by reading that one declared schema with
   its exact row keys, not by a loose fallback.
7. **One of my own v5 tests was wrong** (a row filter by column alone). The
   run caught it; the code was right.
8. **C114–C116 and the C120 inventory were produced by agents under my
   direction.** I reviewed them before use:
   - I re-ran the 119 financial-data tests after their results commit;
   - I checked that only files were added;
   - I checked every join field against the real files;
   - I spot-checked the licence claims on disk.
   The C120 draft had a stale successor entry, a wall-clock field, and a
   shortlist described as "record metadata on disk" when only community
   listings are on disk. The published wrapper corrects all three and binds
   the listings by digest.
9. **The C116 pre-ledger carries a `sealed_at` timestamp,** so re-running the
   characterization would not reproduce its ledger bytes.

## 2. PRE and POST

**PRE** `c106_c121_pre_2026_09_12.py` (`ab8134e5`), frozen at the ordered
bases before any edit: **37 defects, 37 reproduced**, every fact holds,
31 preserved identities unchanged. Your reproducer gives its captured
output here.

**POST** `c106_c121_post_2026_09_12.py` (`a44607e`), at the final tips:
**64 checks, 63 corrected, 1 NOT corrected.**

| not corrected | why |
|---|---|
| `C109.submission_itself_shows_1_3125` | the submission's own block reads 1.0 / valid; 1.3125 is visible only in the bound addendum (confession 1) |

- **Preserved identities:** all 31 the PRE captured are equal to their PRE
  values and did not change during the POST.
- **New since the PRE:** the T2 submission v4 and its addendum, and design
  v5 and its population. None of these is a preserved item.
- **Focal batteries at the final tips:**

| repository | tests |
|---|---|
| predictor | 99 passed |
| B4 | 27 passed (one on a throwaway database) |
| T2 | 57 passed |
| financial-data | 119 passed |

**Mutations per guard.** Each PRE defect is attacked again in the POST with
the PRE's own bytes (input digests compared):

| guard | attacks |
|---|---|
| member-by-member join (C113, C117) | the six-by-five control stays sufficient; your counterexample refuses its unkeyed census row; duplicate column refuses; terminal of another variable, census of another dataset, distinct source digest, license or role absent, temporal contract of another digest, mask of another dataset, member only in aggregates, recomputed terminal with role or license UNKNOWN, semantically unresolved terminal, date stored as an integer: each insufficient by its exact reason |
| design validator (C119) | boolean, NaN, infinite and out-of-domain margins; float minimum; duplicate seed, operator and contrast; LOPO declared in prose |
| Holm and contrasts (C119) | p above one, NaN, incomplete family; boolean panel values; a contrast without a frozen sample digest |
| strict parser (C119) | duplicate key, NaN constant through the `--validate` CLI |

## 3. T2 — chronology and submission v4

| when | what |
|---|---|
| 03:20 | two-sided formula committed in `5fb2849e` |
| 09:53 | public design declaring 1.3125 superseded by 1.0 (`86ca8ca8`) |
| 18:25 | round-7 hardened replay code `4d0b3159` |
| round 7 | replay stopped: one field, 1.3125 vs 1.0 |
| this round | your record `c75d8d6e` installed; single replay from a clean checkout of `4bf38b6f` |

The PRE executes the committed function as committed: `p(3, 6) = 1.0`,
table `1/32, 7/32, 11/16, 1, 11/16, 7/32, 1/32`, the same function at the
replay commit.

**C107.** Your record was read in place: bytes `c75d8d6e`, mode 0600, never
copied, regenerated or chmod-ed. The committed gate verified schema,
reviewer, decision, date, scope, commit and tree, the eleven-file surface,
the historical record, the preserved root and the corrected expectation
before any evidence opened.

**C108.** One attempt. Checkout, record and root inventory (`71c8ae7e`, 736
entries) were identical before and after. No divergence report was written.

**C109.** Commit B `a18ca3b2` carries three evidence files and no code:

| file | content |
|---|---|
| `T2_READJUDICATION_SUBMISSION_V4_2026_09_12.json` | as the builder wrote it; file `b3bab71d`, self digest `1cf89d93` |
| `T2_C108_HARDENED_READJUDICATION_CLOSURE_2026_09_12.json` | the replay closure: `signs_positive 3`, `sign_test_exact_p_two_sided 1.0` |
| `T2_READJUDICATION_SUBMISSION_V4_HISTORY_ADDENDUM_2026_09_12.json` | binds both files and the preserved evidence; 1.3125 `INVALID_SUPERSEDED_NOT_EDITED` |

Declared and checked: `242 COMPLETED_VERIFIED / 0`, `DOES_NOT_ADVANCE`,
estimand `-0.001048443391358884`, the six panel effects equal to the
historical evidence, scientific digest `1be80a0a` equal to the record,
`grants_nothing`, zero promotion, no retraining, downloads or model
execution.

## 4. B4 — final disposition

**C110.** `B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE` recorded with
the identities you named, in a disposition file and an append-only register
note (`2042af83`). 2 `COMPLETED_VERIFIED`, 1 `QUARANTINED_PARTIAL`, 9
`NOT_STARTED`, `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`, campaign closed. No
consumer, no runner.

**C111.** One envelope, built only from the accepted v4 submission after
re-checking its file digest, self digest, counts, verdict and commit A
(`1b4c22a5`, `432599be`):

- campaign `b4::screen_b_v7_campaign`, `DEVELOPMENT`, `CLOSED`,
  `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`;
- five campaign units: the three counts, 35,682.4 s declared for the
  completed cells, 124,993.6 s declared only as a lower bound for the
  partial cell;
- `units_failed` 0; no per-cell scientific result.

Loaded into a throwaway database first (twice; the second load added
nothing). Emitted once to the real outbox (`230c049a`) and drained by the
running loader. A second emission was a no-op.

## 5. Terminals and the successor

**C112.** `TERMINALS_V4_ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE`
(`TERMINALS_V4_DISPOSITION.v1`, bound to verification `be73a592`,
supersession `f7072351`, census file `818063ba`):

| scope | variables |
|---|---|
| numeric descriptors recomputed from their bound bytes | 1,501 |
| physical storage type known | 1,505 |
| semantics, role, unit, license and missing policy declared | 0 |
| producer authority only | 460 |
| semantically unresolved (no numeric value eligible) | 4 |

`terminal_evidence_scope` refuses any wording that calls these counts
verified without their scope; it caught one such phrase in this packet's own
first draft.

**C113.** The rule lives in the join, the last point of use: a recomputed
terminal with role or license UNKNOWN, a semantically unresolved terminal
and a date stored as an integer never become members.

**C114 — temporal contract v3** (`ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3`,
mask `ETH_H4_SUCCESSOR_SAMPLE_ELIGIBILITY_MASK.v1`), from the successor
bytes:

- identity: dataset `…successor_stage22_rerun.v1`, sha `427a754b…`; no
  historical id or digest in it;
- FEATURE_DAG.v4 and binding manifest v2 bound and verified;
- 18,085 rows; 20 truncated bars inside, 1 before the first row; 8 gaps; no
  filling;
- 10,282 eligible, 7,803 ineligible;
- against v2: geometry equal on all 14 aspects, `identity_equal: false`.

**C115 — semantic census** (`ETH_H4_SUCCESSOR_SEMANTIC_CENSUS.v1`):

- 89 rows keyed by (dataset_id, dataset_sha256, column), stable variable ids;
- role `input_feature`: 89;
- unit `1`: 59, each with the producer formula as evidence; unit `UNKNOWN`: 30;
- **license `UNKNOWN`: 89.** No licence document exists for these bytes, and
  none was inferred from the provider;
- missing policy declared: 89; sentinel policy declared: 89, in 8 distinct
  forms.

**C116 — own characterization:** 89 terminals recomputed from the bound
parquet, all `INDEPENDENTLY_RECOMPUTED` and `NUMERIC_MEASURABLE`, with a
pre-ledger written first (`139c825e`) and write-once outputs. No terminal v4
was copied.

## 6. Population and bank

**C118 — membership ledger** (`PER_VARIABLE_DESIGN_V5_POPULATION.v1`),
every input digest bound:

| | |
|---|---|
| candidates | 89 |
| eligible variables | **0** |
| panels | 0 |
| verdict | `BANK_INSUFFICIENT` |
| exclusions | 59 `UNDECLARED_LICENSE_LICENSE_SOURCE`, 30 `UNDECLARED_UNIT_LICENSE_LICENSE_SOURCE` |
| population / ledger digest | `4f53cda1` / `71854f32` |

Every successor column satisfies the other seven conditions. Nothing was
relaxed.

**C120 — panel inventory** (`PANEL_INVENTORY.v1`): 38 candidates, 0
qualify.

- Binance feed: one family, whatever the number of instruments.
- financial-data families: license `UNKNOWN`; the repository README grants
  no reuse permission.
- Monash datasets already on disk: CC-BY-4.0 by their Zenodo records, but
  no join artifact exists for any of them.
- Synthetic data and univariate sets are excluded.

**Deficit: 6 panels and 30 eligible variables.**

**The 2 GiB download authorization was not used.** A download adds bytes,
not members: a downloaded panel counts only after its own terminals,
licensed semantic census, lineage and temporal contract exist and the join
admits five of its variables. The shortlist of official sources is recorded
for that work, with licenses read from the listings on disk and flagged for
upstream-terms review:

| record | dataset | license | bytes |
|---|---|---|---|
| 4656132 | Traffic Hourly | cc-by-4.0 | 22,868,806 |
| 4656140 | Electricity Hourly | cc-by-4.0 | 11,823,931 |
| 4656719 | KDD Cup (with missing values) | cc-by-4.0 | 2,456,948 |
| 5184708 | Oikolab Weather | cc-by-4.0 | 1,326,101 |
| 4659727 | Australian Electricity Demand | cc-by-4.0 | 5,770,526 |
| 4654909 | Wind Farms (with missing values) | cc-by-4.0 | 71,383,130 |
| 4656072 | London Smart Meters (with missing values) | cc-by-4.0 | 219,673,439 |

**C119 and C121 — design v5** (`PER_VARIABLE_PREPROCESSING_DESIGN.v5`,
`2936b6dc`): supersedes v4 by file (`109cdc50`) and design digest,
scientific change `NONE` enforced block by block; strict parser; typed
validator; complete Holm family; one frozen sample per contrast; LOPO
derived from panel rows. Scoring refuses
`EXTERNAL_V5_DESIGN_REVIEW_AND_LICENSE_REQUIRED`.

## 7. OLAP, backlog and counts

| table | before C111 | after |
|---|---|---|
| dim_campaign | 9 | 10 |
| dim_campaign_run | 9 | 10 |
| fact_campaign_unit | 183 | 188 |
| fact_campaign_consumption | 264 | 265 |
| fact_variable_characterization | 40,749 | 40,749 |
| fact_terminal_verification_variable_v2 | 1,965 | 1,965 |

At the final tip:

- **Outbox:** 0 pending, 23 loaded, 16 failed, and all 16 dead letters
  adjudicated.
- **Loader:** active with zero restarts. It was never restarted, and neither
  were PostgreSQL and Metabase.
- **Earlier B4 stub:** `b4_campaign_generation_v7_20260908` remains
  (confession 5).
- **Backlog:** none introduced by this order.

**Full suites at the final tips:**

| repository | result |
|---|---|
| predictor | 588 passed, 19 skipped, 3 failed, 8 collection errors |
| financial-data | 784 passed; one file not collected |

- **predictor:** the 3 failures (`test_configuration_handling.py`) and the
  8 legacy collection errors are the same set as the previous round. The
  passed count rose with this round's tests.
- **financial-data:** the file not collected needs `yaml`, which is not
  installed. It was excluded as in the previous round.

## 8. Branches and digests

| repository | branch | commits |
|---|---|---|
| predictor | `satoshi/c106-c121-20260912` | PRE `ab8134e5` · A `134aecc` · B `2281bf5` · A2 `5f067790` · ledger `ca3c73a` · C120 A `f1904ff1` · C120 B `38d57ad` · POST `a44607e` · this packet |
| financial-data | `satoshi/c106-c121-20260912` | A `5d21fb5a3` · B `fba552195` |
| agent-multi (B4) | `satoshi/b4-c110-c111-20260912` | C110 `2042af83` · C111 A `1b4c22a5` · A2 `432599be` |
| agent-multi (T2) | `satoshi/t2-c106-c109-20260912` | B `a18ca3b2` on `4bf38b6f` |

## 9. What needs you

1. **T2 submission v4** (`b3bab71d`) with its closure and addendum, and the
   confessed gap: the submission's own block does not show 1.3125.
2. **Terminal v4 disposition** as recorded.
3. **Successor artifacts:** temporal contract v3, semantic census and
   characterization, including the unit and sentinel judgment calls in the
   census.
4. **Design v5 and the empty population.** No scoring license is requested:
   the bank is insufficient.
5. **The licence question for the ETH successor bytes,** which alone
   excludes all 89 variables, and whether to build join artifacts for the
   shortlisted public panels under the download authorization.

**Zero line:** no GPU compute process at the POST; training 0, scores 0, confirmation 0, promotion 0, live 0, venue 0, account actions 0, downloads 0, records issued in Musashi's name 0.
