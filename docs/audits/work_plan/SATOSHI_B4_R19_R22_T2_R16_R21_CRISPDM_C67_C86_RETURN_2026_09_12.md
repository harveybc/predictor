# Return packet — B4 R19–R22, T2 R16–R21, CRISP-DM C67–C86

**Date:** 2026-09-12
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_B4_R19_R22_T2_R16_R21_CRISPDM_C67_C86_ORDER_2026_09_12.md`
**Authorization in force:** `MECHANICS_ONLY_CPU_NO_SCORES`
**Disposition:** `READY_FOR_MUSASHI_EXTERNAL_REVIEW`

Zero GPU. Zero scores. Zero confirmation. Zero live.

---

## 1. My own defects first

1. **An uncommitted edit of mine sits in the recoverable T2 snapshot.**
   The working tree of `f070eeb1` carries a modification to
   `tools/t2_campaign_closure.py` written at 09:24 that matches no
   commit. The snapshot commit is intact and recoverable; its working
   tree is not clean. I did not touch it under this order: the
   reproducer was built from a fresh checkout of `f070eeb1` instead.
2. **The verifier I shipped last cycle claimed a recomputation that had
   not happened.** `TERMINALS_VERIFIED_EXACT` was printed after checking
   population and filenames. It is retired.
3. **That verifier bound every variable to the census's first
   appearance.** Measured on the real terminals, all 1,505 MEASURED
   terminals declare exactly that first appearance, so the 420 files it
   digested were the right ones — but the code would have accepted a
   wrong one, and I had not checked.
4. **My first real run of the rewritten verifier bound the disposition
   row to the wrong digest.** The disposition adjudicates a census
   variable and is published against the census content digest; I
   applied the source-file rule and reported all 1,505 variables as
   diverging. The run is kept as evidence; the rule is now declared in
   code and pinned by a test.
5. **My B4 v3 patch broke the module's syntax.** It inserted an import
   line inside a multi-line import. The collection error caught it.
6. **My design v3 validator refused its own design.** Operator O6
   declared a window without `min_periods`, and a missing arm raised
   `KeyError` instead of a refusal. Both caught by the design battery.
7. **My search for the ETH H4 source parquet filtered on file names**
   and missed `trading_asset_data/ethusdt/4h.parquet`, where `eth` is in
   the directory. Found on the second search.
8. **A commit message of mine cites the wrong verification digest
   lineage** for the corrected real verification. The digests in §5 of
   this packet are the ones measured.

9. **My first readjudication template pointed at a file that no longer
   existed.** It bound the candidate to a closure JSON in a previous
   session's scratch directory. The run stopped before opening any
   evidence. The candidate is now derived from the versioned historical
   evidence file inside the reproducer (commit A2).
10. **I wired the single-checkout reproducer without first tracing every
    executor identity check.** Two isolated-fixture reproductions stopped
    inside the pinned code, before anything was scored: first at
    `verify_executor_checkout` in the gates (HEAD must equal the
    historical commit), then, after A3, at `_verify_authority_block` on
    the first unit (each claim's pinned commit and tree must equal the
    executing HEAD). Both refusals are kept as evidence; A3 and A4 answer
    them with scoped, declared replacements.

11. **I left the predictor and financial-data branches of this round
    unpushed for most of it.** Eight predictor commits (PRE, C68, C72,
    C73, C84–C85, evidence) and the C81–C83 commit existed only locally
    until a branch-state check caught it; no reviewer could have fetched
    them. Both are now on the remote, and the pre-push secret gate found
    no leak.

**Declared limits, not defects:** FEATURE_DAG.v3 resolves helpers
within their own module only, models `try` bodies coarsely, and treats
frame parameters by the name heuristic inherited from v2; its probe
sandbox patches Python-level entry points only. The descriptor
recomputation counts underspecified contracts as not independently
verifiable rather than choosing a reading for them. The ETH contract
reads DATE_TIME as UTC because the registry says so, supported but not
independently proven by the exact match.

## 2. PRE and POST

PRE `crispdm_c67_c86_pre_2026_09_12.py`, frozen at the audited bases
before any edit, reproduced every counterexample (0 not reproduced) and
found identities unchanged.

POST `crispdm_c67_c86_post_2026_09_12.py` at the final tips:
**40 checks, 0 not corrected.**

| finding | PRE (bases) | POST (final tips) |
|---|---|---|
| C67 leaf substitution, 4 copies | consumed 9.0 | refused `OPEN_VS_INVENTORY` on inode, ctime |
| C67 equal-length in-place write, 4 copies | consumed 9.0 | refused on ctime |
| C67 late NPZ substitution (T2) | consumed 999.0 | refused on inode, ctime |
| C68 divergent copies | 1 digest, no leaf binding | 1 digest `93e0fd89` with binding |
| C69.1 absent source | `TERMINALS_VERIFIED_EXACT` | `SOURCE_ABSENT` |
| C69.2 fabricated block | `TERMINALS_VERIFIED_EXACT` | `TERMINAL_SCHEMA` |
| C69.3 another variable's appearance | rebound to first appearance | `APPEARANCE_NOT_OWNED_BY_VARIABLE` |
| C69.4 absolute / traversal / symlink | outside bytes read | `SOURCE_OUTSIDE_ROOT_OR_LINK`, never read |
| C69.5 unknown outcome, extra key, wrong type | accepted | `TERMINAL_SCHEMA` |
| C69.6 substituted population | accepted | refused before any terminal |
| C74.1 last definition leaks | CAUSAL_ACTIVE | NON_CAUSAL |
| C74.2 windows 10 then 20 | lookback 20 | 29 |
| C74.3 forward callable in apply/map/transform | CAUSAL_ACTIVE | NON_CAUSAL |
| C74.4 constant positional indices | CAUSAL_ACTIVE | UNRESOLVED |
| C74.5 nested body | CAUSAL_ACTIVE | UNRESOLVED |
| C74.6 two producers | verdict flips with order | UNRESOLVED, identical in any order |
| C74.7 name-coincident producer | CAUSAL_ACTIVE | STATIC_CAUSAL_CANDIDATE_UNBOUND |
| T2 self-authorization | — | no readjudication record at the real root |
| T2 readjudication | two trees, HEAD-bound | one checkout, 242/0, same estimand and effects |

Focal batteries at the final tips: B4 78, T2 49, T2 reproducer 51,
predictor C68–C73 85, financial-data DAG v3 + v2 + temporal 133.

Full suites at the final tips:

| repository | result |
|---|---|
| predictor `satoshi/crispdm-c67-c86-20260912` | 468 passed, 3 failed, 8 collection errors. The 3 failures (`test_configuration_handling.py`: default config, save config, configure with args) and the 8 legacy collection errors all occur identically at this round's base `9bb90fa`, run in a throwaway worktree. That base run also failed `test_acceptance_c1_c16.py::test_6_availability_contracts_reach_exactly_their_variables`, which passes at the tip; the base run had no PG credentials loaded (24 skipped), so the conditions were not identical and I claim no fix for it |
| financial-data `satoshi/crispdm-c67-c86-20260912` | 606 passed, 1 collection error (`yaml` is not installed in the environment; `test_p3x_feature_redundancy_stability.py` cannot import) |
| T2 reproducer historical battery `test_t2_completion.py` | 9 passed, 1 failed in 29 min 10 s: `test_omitted_mlp_seed_refuses` raises `KeyError: 'seed11'` at historical `t2_confirmatory_executor.py:1976` — see below |

**A known weakness the reproducer carries on purpose.** In the historical
C90 battery, `test_omitted_mlp_seed_refuses` fails in the reproducer. It
exercises a hardening added on the branch tip after the execution record:
`verify_unit_record` turning an omitted `mlp_small` seed into a typed
refusal. The reproducer must carry the historical executor bytes, and in
those bytes an omitted seed raises an untyped `KeyError: 'seed11'` at `t2_confirmatory_executor.py:1976` instead; the other 9 mutations of the battery refuse as designed. So the
historical verifier is weaker here than the tip, and the readjudication
reproduces that weakness rather than hiding it. Measured on the private
copy, all 242 real records carry exactly 3 `mlp_small` seeds in all
1,936 origin-arm entries, so no real record exercises the gap and the
readjudicated result is unaffected. Whether a readjudication should
instead run under the hardened tip identity is a decision for review.

## 3. P0-A — leaf identity (C67–C68, B4 R19–R22, T2 R16–R17)

`descriptor_custody.py` now photographs, per leaf, type, device,
inode, uid, mode, size, `mtime_ns` and `ctime_ns`. A read is accepted
only if the `fstat` of the opened descriptor equals the inventory and a
second `fstat` after the last byte equals the first; otherwise
`LeafIdentityRefusal` names the stage and fields, and no byte leaves.
Every artifact publishes the three fact sets.

One implementation, one digest (`93e0fd89…`), one 16-test fixture file,
byte-identical in predictor, B4, T2 and the T2 reproducer, pinned by a
digest file. Mutants remove the open check, the after-read check,
ctime, and each of inode, size, mode and mtime with ctime, and each
readmits its attack.

**B4-R21** — re-run only on the private copy, from a clean checkout of
commit A'. Result unchanged: 2 COMPLETED_VERIFIED, 1
QUARANTINED_PARTIAL, 9 NOT_STARTED, `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`;
40 of 40 reads leaf-bound. Commits: A `996e89ea` (binding), A'
`cc4b6ada` (submission v3), B `2fb3e2b9` (submission only).

## 4. P0-B — T2 readjudication identity (R18–R21)

**History is not repinned.** `MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD`
is read and compared, never written; its commit `7bcd3f0d` and seven
digests are unchanged.

**One reproducer checkout**, branch
`satoshi/t2-readjudication-reproducer-20260912`, built from the
recoverable snapshot `f070eeb1`, whose seven executor files are the
historical bytes. The tip's later hardening of two pinned files is
deliberately absent. Every import comes from this checkout, including
the closure, custody and gate modules; anything else is `IMPORT_MIX`.

| commit | what |
|---|---|
| `fe980a34` A | readjudication record contract (candidate side), fixture mode, single-checkout loader, R21 battery |
| `ccefa161` A2 | candidate derived from the VERSIONED historical evidence, not a scratch file |
| `7de596b0` A3 | scoped readjudication checkout gate replaces the executor's HEAD gate |
| `fc3d159f` A4 | unit claims checked against the reviewed historical identity; executor write paths closed |
| `bfbad73c` A5 | submission v3 builder; leaf-binding summary in the closure |

**R18.** The contract `agent_multi.t2_readjudication_review_record.v1`
binds the historical record's raw-byte digest, commit and tree, the seven
historical digests, the reproducer commit, tree and eleven-file surface,
the preserved root's inventory digest, the candidate adjudication digest,
reviewer, canonical date, `READ_ONLY_READJUDICATION`, and retraining,
downloads and grants_execution all false. The candidate can build only
an unreviewed template, which refuses to be written inside the authority
root. With no record at the fixed path the reproducer stops with
`READJUDICATION_REVIEW_RECORD_REQUIRED` before opening any evidence —
demonstrated at A, A2, A3, A4 and A5.

The candidate adjudication digest covers the scientific result only
(counts, screen adjudication, corrected sign-test table), derived from
`docs/audits/evidence/T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json`
(sha `181c0796…`): `8b935413…`.

**R19 — two identity checks the executor owns, answered for a read-only
act, never by repinning.** The pinned gates end in
`verify_executor_checkout` (HEAD must be `7bcd3f0d`), and the pinned deep
verifier compares each unit claim's commit and tree with the executing
HEAD. Both are the executor's questions. Within one closure only, and
restored on success and exception, they are answered by:

- `readjudication_checkout_gate` — the historical record must still pin
  the reviewed commit and tree, and the executing checkout must be the
  clean reproducer commit, tree and surface the readjudication record
  reviewed;
- `historical_identity_view` — claims are compared with the historical
  commit and tree the record reviewed, re-checking on every call that the
  checkout is still the clean reviewed reproducer;
- the executor's claim-writing entry points `main` and `rehearse` refuse
  for the whole scope.

The claims' code identity is still compared with the seven physical
historical digests, unchanged. Every replacement is named in the output.

**R20 — isolated fixture, then reproduction on the private copy.** A
fixture record with reviewer `ISOLATED_TEST_FIXTURE`, in an isolated
0700 authority directory, never at the real root, never in the
reviewer's name. Under A4:

| | historical (old identity) | readjudication (reproducer identity) |
|---|---|---|
| units | 242 COMPLETED_VERIFIED / 0 | 242 COMPLETED_VERIFIED / 0 |
| verdict | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| primary estimand | −0.001048443391358884 | −0.001048443391358884 |
| screen adjudication | — | identical, field for field |
| scientific digest | candidate `8b935413…` | equal |

726 artifact reads under leaf-bound custody, 834 s CPU, no retraining,
no download, no promotion. The published p 1.3125 remains superseded by
the exact table (corrected 1.0), and the historical envelope is not
rewritten.

**Commit B `6fe6c1ea`** carries only
`T2_READJUDICATION_SUBMISSION_V3_2026_09_12.json`, generated from a
detached checkout of A5 verified clean before and after, with its own
isolated fixture authority. Same result as above, 726 of 726 reads
leaf-bound, submission digest `2a655c55…`, record kind
`ISOLATED_FIXTURE_NOT_AN_EXTERNAL_RECORD`, `requires`
`EXTERNAL_READJUDICATION_REVIEW_RECORD`. The template, the stop without a
record, both refusals and the A4 closure are published as auxiliary
evidence in predictor `docs/audits/evidence/t2_r18_r20/`.

## 5. P0-C — the 1,965 terminals (C69–C73)

`TERMINALS_VERIFIED_EXACT` is retired. The verifier now checks exact
schemas and types, requires an expected census content digest supplied
from outside, binds each MEASURED terminal to its own declared
appearance and that appearance to one file read by contained components
under leaf-bound custody, and compares every published descriptor row
with values recomputed from those bytes by a module that imports no
producer code. Specificity is declared before comparison.

Real run over the live state (report `LAKE_TERMINAL_VERIFICATION.v2`):

| | |
|---|---|
| population | `TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED`, 0 divergences |
| terminals | 1,965 (PRODUCER_DECLARED_MEASURED 1,505, PRODUCER_DECLARED_NOT_IDENTIFIABLE 460) |
| sources bound | 420 files, 1,505 variables, each to its declared appearance |
| INDEPENDENTLY_RECOMPUTED | 1,504 variables |
| PRODUCER_DECLARED | 460 (temporal axis columns: `timestamp` 420, `close_time` 40) |
| SOURCE_BOUND | 1 |

Every fully specified descriptor agrees. The single divergence is
`discrete_entropy_bits` on a constant series: the producer published
`NOT_COMPUTABLE:ValueError`; the declared reading gives 0 bits. It is
reported as found. Underspecified descriptors (p01, p99, iqr, mad,
difference-to-level dispersion, spectral centroid, window-mean
dispersion, and the conditional ones where the window holds non-finite
values) are `NOT_INDEPENDENTLY_VERIFIABLE` and are never counted.

C69 counterexamples at the final tip: absent source → `SOURCE_ABSENT`;
fabricated block → `TERMINAL_SCHEMA`; another variable's appearance →
`APPEARANCE_NOT_OWNED_BY_VARIABLE`; absolute, traversal, symlinked leaf
and symlinked directory → `SOURCE_OUTSIDE_ROOT_OR_LINK`, outside bytes
never read; unknown outcome, wrong type, undeclared key →
`TERMINAL_SCHEMA`; substituted population → refused before any terminal.

v3 terminals: 1,965 written beside v1 and v2, whose content digests were
checked unchanged before and after.

**C73 (live cube, additive):** one verification run, 1,965 variable rows
and 37,301 descriptor rows loaded; the historical 40,749
characterization rows unchanged; a second load writes nothing.
`v_characterization_evidence_layer` for attempt c49: 25,505 rows
INDEPENDENTLY_RECOMPUTED, 11,795 SOURCE_BOUND not independently
verifiable, 1 SOURCE_BOUND diverging, 460 PRODUCER_DECLARED; the 2,988
rows of earlier attempts stay PRODUCER_DECLARED. Outbox healthy before
and after, 0 backlog, 0 unadjudicated dead letters.

## 6. P0-D — FEATURE_DAG.v3 (C74–C80)

The heuristic is replaced by an analysis that respects scope, assignment
order and control flow: nested bodies never bind the outer scope, the
reaching definition governs, disagreeing branches and disagreeing
producers are UNRESOLVED in any order, callables passed to methods are
followed, every positional index not proven historical is UNRESOLVED,
windows compose inclusively (10 then 20 → 29) and shifts add. Seven
mutants, one per guard, each restore v2's wrong answer.

**No column is CAUSAL_ACTIVE, because no producer binding could be
established.** For the ETH dataset the chain is strong — recomputed
dataset digest equals the inventory and manifest, the export names
`technical.parquet` and `statistical.parquet`, all 82 feature columns
equal a parquet column exactly, the Stage 2.2 receipt names both files,
one static writer calls both producers — but the receipt records no
commit and no code digest, and the run ended about two minutes before
the worker file's first commit. The code that ran was never recorded;
that is not a binding.

| class (97 columns) | v2 | v3 |
|---|---|---|
| CAUSAL_ACTIVE | 37 | **0** |
| STATIC_CAUSAL_CANDIDATE_UNBOUND | — | 83 |
| UNRESOLVED_PRODUCER_BINDING | — | 14 |
| HISTORICAL_OR_RETIRED_PRODUCER | 5 | 0 |
| UNRESOLVED | 55 | 0 |

Statically alone: 92 causal, 5 UNRESOLVED (`typical_price`, `BC-BO`,
`BH-BL`, `BH-BO`, `BO-BL`). `log_return_1` lookback 2, `macd`
UNBOUNDED, `stoch_k` 14, `cci_14` 27, `mfi_14` 15. The prefix-invariance
probe changed no class, since nothing is bound; the two evidence-selected
candidates passed it for information. v1 `4991e368…` and v2 `8fbe787a…`
byte-identical. 68 v3 tests and the untouched 31 v2 tests pass. Commit
financial-data `b24b22719`.

One way to close the binding gap would be to re-run the committed
producer on the recorded source and compare with the parquet files. It
was not done: this order limits producer runs to synthetic data.

## 7. P1-A — temporal semantics (C81–C83)

ETH H4: 18,085/18,085 rows matched, OHLCV error 0.0, BAR_OPEN, nominal
14,400 s, information complete at each bar's actual `close_time`,
delivery latency UNOBSERVED. 21 truncated bars (20 inside the CSV) and
8 gaps (3 longer than two bars) — the audit's "four hours after" is the
nominal bar, not a universal one. E5a OPEN, E5b CLOSED; EURUSD
UNDECLARED.

## 8. P1-B — design v3 and license (C84–C85)

Design v3 `1a8227e0…`, validates field by field, refuses deltas after
review; population derived from the real artifacts today: **0 panels, 0 members, not evaluable**, emptied by P2 (no CAUSAL_ACTIVE column). The score
entry point refuses `EXTERNAL_DESIGN_REVIEW_AND_LICENSE_REQUIRED`
before importing any numeric or learning library.

## 9. Identities unchanged

Tree digests (name, size, mtime, inode) taken before and after both
the PRE and the POST: B4 v7 real root, T2 original root, the B4 and T2
private copies, and the content digests of lake terminals v1 and v2 —
**all unchanged**. The populated cube's historical characterization rows
stay at 40,749.

## 10. OLAP and backlog

| | |
|---|---|
| historical characterization rows | 40,749, unchanged |
| verification runs loaded | 1 (1,504 variables recomputed) |
| evidence layers, all rows | INDEPENDENTLY_RECOMPUTED 25,505 · SOURCE_BOUND 11,796 · PRODUCER_DECLARED 3,448 |
| batch membership | 14 batches, all EXACT |
| outbox | healthy, backlog 0, 16 dead letters all adjudicated, heartbeat fresh |
| loader service | active, 0 restarts |

PostgreSQL and Metabase were not restarted. Every new table and view is
additive, and the schema was proven on throwaway databases first.

## 11. Records Musashi must author afterwards

1. `MUSASHI_T2_READJUDICATION_REVIEW_RECORD.json`
   (`agent_multi.t2_readjudication_review_record.v1`) at the reviewer-
   authority root — only after reviewing the template and this packet.
   It would bind reproducer commit `bfbad73c`, tree and surface as in
   `T2_READJUDICATION_REVIEW_TEMPLATE_A5.json`, and candidate `8b935413…`.
2. External acceptance, or rejection, of the B4 re-adjudication
   submission v3 (commits `cc4b6ada` / `2fb3e2b9`).
3. A review of `FEATURE_DAG.v3` and `PRODUCER_BINDING_MANIFEST.v1`, and a
   decision on whether re-running the committed producer on the recorded
   source is authorized as a way to establish bindings.
4. A review of the per-variable design v3 (`1a8227e0…`) and, only if
   accepted, a scoring license. Today its population is empty.

5. A decision on the readjudication identity itself: whether a read-only
   readjudication should keep running the historical executor bytes,
   reproducing their weaker handling of an omitted `mlp_small` seed, or
   run under the tip's hardened verifier with its own record.

No record was created in Musashi's name by this work. The isolated
fixture records carry reviewer `ISOLATED_TEST_FIXTURE` and never sat at
the reviewer-authority root.

## 12. Zero line

Zero GPU. Zero scores. Zero confirmation. Zero live. No record created
in Musashi's name.
