# Return packet — B4 R23–R26, T2 R22–R27, CRISP-DM C87–C105

**Date:** 2026-09-12
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_B4_R23_R26_T2_R22_R27_CRISPDM_C87_C105_ORDER_2026_09_12.md`
**Licence in force:** `MECHANICS_ONLY_CPU_NO_SCORES`
**Closing state:** `B4_V4_READY_FOR_REVIEW /
T2_HARDENED_READJUDICATION_STOPPED_ON_ONE_FIELD_DIVERGENCE /
TERMINALS_V4_AND_FEATURE_DAG_V4_READY_FOR_REVIEW /
PER_VARIABLE_SCREEN_V4_STILL_UNLICENSED`

Zero GPU. Zero scores. Zero confirmation. Zero live.

---

## 1. My own defects first

1. **I claimed that nothing was resolved by name again after the
   photograph, and it was false.** The audit's counterexample
   (`ACCEPTED_REPLACED_CHILD_DIR`) reproduced in all four copies.
2. **My first enumeration guard relied on two `fstat`s alone.** A file
   created inside the same timestamp tick left `mtime` and `ctime`
   unchanged, and my own battery caught it. A second listing is now
   compared name by name.
3. **My first version of that battery counted descriptors after
   patching `listdir`.** The count itself mutated the directory before
   the attack started.
4. **Binding the root directory into B4's identity broke the
   deduplication of unchanged closures.** The closure's own log moves
   the root's timestamps. The root binding moved to the volatile
   measurement block, and an existing test caught it.
5. **Last round's verifier never recomputed the census digest.** It
   compared the expected digest with the document's own text and
   filename, so a re-pointed appearance verified.
6. **It parsed JSON leniently, and validated appearances and variables
   without exact keys** while the order demanded exact schemas.
7. **Neither the producer nor my verifier established what a column
   meant before measuring it.** An int64 datetime sentinel was averaged
   to −9.22e18.
8. **I shipped a readjudicator that kept a verifier weakness on
   purpose**, and its entry point imported modules before demanding
   the record.
9. **The code I ported onto the hardened line still carried the
   historical gate's loader**, and its post-import check hashed only
   five modules. Both were removed and extended before commit A.
10. **My first v4 population derivation read the temporal contract's
    dataset id from keys neither contract uses.** The temporal
    requirement was never evaluated until corrected.

11. **I built the T2 candidate from historical evidence that still carried
    the impossible p = 1.3125**, while the hardened line computes the
    corrected 1.0. The replay caught it, as it should, and publication
    stopped.
12. **My first hardened readjudicator discarded the recomputed result on
    a divergence stop**, so the stop could not say what differed. A2
    `47be2d20` writes the field-level report before refusing.

## 2. PRE and POST

PRE `crispdm_c87_c105_pre_2026_09_12.py`, frozen at the bases before any
edit: 0 not reproduced; identities unchanged.

POST `crispdm_c87_c105_post_2026_09_12.py` (`4324cef`): **25 checks,
23 corrected, 2 NOT corrected.**

| not corrected | why |
|---|---|
| `T2-R26.divergence` | hardened replay stopped: one field diverges |
| `T2-R26-R27.submission` | submission v4 absent, because of that stop |

Focal batteries at the final tips: B4 98 passed; predictor 54 passed and
4 skipped; financial-data 277 passed; T2 readjudicator 57 passed; design
v4 18 passed. All 16 preserved identities are unchanged.

**Full suites at the final tips:**

| repository | result |
|---|---|
| predictor | 526 passed, 3 failed, 8 collection errors |
| financial-data | 750 passed, 1 collection error |

- **predictor:** the 3 failures (`test_configuration_handling.py`) and
  the 8 legacy collection errors are the same set shown to occur
  identically at the previous round's base `9bb90fa` in a throwaway
  worktree.
- **financial-data:** the collection error is `yaml`, which is not
  installed in the environment.

## 3. P0-A — directory components (C87–C88, B4 R23–R26, T2 R22–R23)

Custody contract v2 (`agent_multi.descriptor_custody.component_binding.v2`,
digest `a299cf78`, byte-identical in predictor, B4 and T2):

- the inventory keeps facts for files, directories and other objects;
- `subdir()` compares the opened directory with its parent's inventory;
- every listing is bracketed by two `fstat`s and a second identical
  listing;
- strict JSON refuses duplicate keys and non-finite constants;
- descriptors close on every refusal;
- every snapshot publishes its directory binding.

A shared battery covers each attack, with a mutant per guard. The
historical reproducer `6fe6c1ea` stays at its round-6 custody by
design, as a preserved artifact.

**B4-R25** re-ran from a clean checkout of A `4c842dd` over a fresh
private copy of the preserved root. The result is unchanged: 2/1/9,
`SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`. All 40 reads are leaf-bound and
7 directories are bound. B is `ff52ca7d`.

## 4. P0-B — canonical census, types and nulls (C89–C92)

**Verifier v3** (v2 kept as history):

- recomputes the census digest with the producer's `_self_sha`
  (`49a8813d…`) and requires it to equal the reviewer-supplied
  expectation, the declared digest and the filename; the raw file digest
  `818063ba…` is kept separate;
- parses strictly;
- applies exact key-and-type schemas.

**Semantic contract** per column before any statistic. A census
declaring UNKNOWN for type, unit, role and license makes "measurable"
rest on the physical type alone, and the report says so.

**Real run** (`be73a592…`):

| | |
|---|---|
| population | VERIFIED, 0 divergences |
| columns swept | 3,345 |
| NUMERIC_MEASURABLE | 1,501 |
| NON_NUMERIC | 1,840 |
| SEMANTIC_TYPE_UNRESOLVED | 4 (the four announcement datetimes) |
| layers | 1,501 INDEPENDENTLY_RECOMPUTED · 460 PRODUCER_DECLARED · 4 SEMANTICALLY_UNRESOLVED · 0 DIVERGES |

Of the 56 zero-dominated columns found, none was declared null: they
are dividends, splits, volumes and recession flags. 1,965 terminals v4
were written beside v1–v3, guarded through custody.

**Cube, additive.** Six-layer tables and view:

| layer | rows |
|---|---|
| INDEPENDENTLY_RECOMPUTED | 25,434 |
| PHYSICALLY_TYPED | 11,767 |
| PRODUCER_DECLARED | 3,448 |
| SEMANTICALLY_UNRESOLVED | 100 (86 with a numeric value withdrawn) |

The historical 40,749 rows are unchanged, a second load writes nothing,
and the outbox stays healthy with 16/16 dead letters adjudicated.

## 5. P0-C — hardened T2 readjudication (R24–R27)

**Result: STOPPED. No submission v4 is published.**

| commit | content |
|---|---|
| `305d2edd` | C88 custody contract v2 |
| A `4d0b3159` | stdlib entry gate before any import; hardened closure; 92 focal tests |
| A2 `47be2d20` | divergence stops write a field-level report first |
| B `4bf38b6f` | `T2_HARDENED_READJUDICATION_DIVERGENCE_2026_09_12.json` only |

- **Template** (R24): candidate `a989d02d…`, built from the versioned
  2026-09-10 evidence. No real record was installed; the replay used an
  `ISOLATED_FIXTURE_NOT_EXTERNAL_REVIEW` record in a private directory.
- **Stop without a record:** `HARDENED_READJUDICATION_REVIEW_RECORD_REQUIRED`.
- **Replay** (R26), from a clean detached checkout, twice (under A and A2):
  both refused `CANDIDATE_ADJUDICATION_DIVERGES`, and the worktree stayed
  clean both times.

**The one difference:**

| field | candidate | hardened recomputation |
|---|---|---|
| `screen_adjudication.sign_test_exact_p_two_sided` | 1.3125 | 1.0 |

Rebuilding the candidate reproduces `a989d02d`; the recomputation digests
to `1be80a0a`. Counts 242/0, `DOES_NOT_ADVANCE`, the estimand
`-0.001048443391358884` and all six panel effects are equal.

The historical 1.3125 was superseded beside the evidence, never inside it;
the hardened code computes the corrected value directly. **I did not
rebuild the candidate to match**: that would be choosing the expectation
after seeing the result. Whether a v4 candidate with this field superseded
may be declared is the reviewer's decision.

**Completion battery at the hardened tip** (`test_t2_completion.py`):
**10/10 passed** (29 min, CPU), including the omitted MLP seed as a
typed refusal. The run started at A `4d0b3159`; A2 and B landed while it
ran, and they touch only the closure, the entry point and one evidence
file.

## 6. P1-A — prospective producer provenance (C93–C97)

v3 is recorded as `STATIC_CANDIDATE_INVENTORY_UNBOUND`.

**Prospective rerun.** The current committed Stage 2.2 producer ran
from a clean checkout of `6ee88a43` into a new write-once root, in
5.4 s. The pre-run manifest was sealed first.

| comparison | columns equal | rows |
|---|---|---|
| technical | 61/61 | 18,337 |
| statistical | 22/22 | 18,337 |
| historical model-ready CSV | 89/89 | 18,085 |

**All bytes differ.** The values reproduce; that does not bind the
historical run.

- The end-to-end prefix invariance passes for all 89 columns.
- The binding manifest v2 carries 89 complete bindings: 61 from
  `compute_technical`, 22 from `compute_statistical` and 6 from the
  identified assembler.
- FEATURE_DAG.v4 has **89 CAUSAL_ACTIVE** on the successor dataset
  `…successor_stage22_rerun.v1`; the historical dataset stays at 0.

## 7. P1-B — temporal quality (C98–C99)

Contract v2 separates availability, completeness, regularity and
horizon eligibility.

- **Samples:** 10,282 eligible and 7,803 ineligible out of 18,085.
- **Where the ineligible ones come from:** 20 truncated bars and 8
  gaps, each spreading through the 512-bar warm-up. Every gap
  immediately follows a truncated bar.
- **Origins:** all 5 are supported.
- **Tests:** 85, and v1 is intact.
- **Mask parquet:** force-added, because `*.parquet` is gitignored.

## 8. P1-C — design v4 (C100–C104)

- **Hypotheses:** H1 is A1 against A0; H2 is A2 against both A0 and A3;
  H3 is descriptive plus a harm rule.
- **A3:** generated from training-fold moments only.
- **Panels:** at least six independent ones.
- **Population:** requires semantics, role, unit, license and a
  missing/sentinel policy, and applies the temporal mask.
- **Inference:** a frozen common sample; Holm, bound levels and the t
  quantile are executable; LOPO is required.
- **Score:** refuses `EXTERNAL_V4_DESIGN_REVIEW_AND_LICENSE_REQUIRED`.

**Population from the real artifacts: `BANK_INSUFFICIENT`**, for three
independent reasons:

1. The 89 active columns live on the successor dataset, which has no
   temporal quality contract of its own.
2. No census variable declares semantics, role, unit and license.
3. There is one candidate panel, not six.

## 9. Identities, OLAP, records

**Identities:** all 16 are unchanged, byte for byte:

- the real B4 v7 and T2 original, and their private copies;
- lake terminals v1–v3;
- FEATURE_DAG v1–v3 and binding manifest v1;
- temporal contract v1;
- B4 and T2 submissions v2 and v3.

**OLAP:** the load was additive, and the loader, PostgreSQL and Metabase
were never restarted.

- The C73 historical rows (40,749) are unchanged.
- A second load writes nothing.
- The outbox is healthy, with 16/16 dead letters adjudicated.
- Backlog: none introduced by this order.

**Final tips:**

| repo | branch | tip |
|---|---|---|
| predictor | `satoshi/crispdm-c87-c105-20260912` | this packet, on top of POST `4324cef` |
| financial-data | `satoshi/crispdm-c87-c105-20260912` | `7d9862a33` |
| B4 | `satoshi/b4-r23-r26-20260912` | A `4c842dd`, B `ff52ca7d` |
| T2 | `satoshi/t2-r22-r27-20260912` | A `4d0b3159`, A2 `47be2d20`, B `4bf38b6f` |

**Records that need Musashi** (none created on his behalf):

1. B4 readjudication submission v4 (`ff52ca7d`).
2. **T2:** the one-field divergence. Rule whether a candidate with
   `sign_test_exact_p_two_sided` explicitly superseded may be declared, and
   whether the hardened readjudication record may then be issued from the
   template.
3. Terminals v4 and verification v3, including the 4 semantically
   unresolved datetime columns.
4. FEATURE_DAG v4 and binding manifest v2 on the successor dataset.
5. Temporal quality contract v2 and the mask.
6. Design v4, whose population is `BANK_INSUFFICIENT` for three reasons.

**Zero line:** GPU 0 · score 0 · confirmation 0 · live 0 · venue 0 · DOIN
publication 0 · cube cleanup 0.
