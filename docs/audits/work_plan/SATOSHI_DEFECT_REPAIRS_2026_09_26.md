# Two reported defects, repaired: the dirty-tree trap, and a preparation that holds origins its design never declared

Satoshi, successor technical lead. 2026-09-26. Acting on the owner's grant of 2026-09-26.
Worktree `predictor-defrep-20260926`, branch `satoshi/defect-repairs-and-fred-20260926`, from `7d2b1c83`.
Evidence: [`docs/audits/evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/`](../evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/).
The third item of the same order, the eleven FRED siblings, is in `data-gov`:
`docs/audits/work_plan/SATOSHI_FRED_DATA_GOV_REGISTRATION_2026_09_26.md` on the branch of the same name.

> ## READ THIS HEADER OR READ NOTHING
>
> **The dirty-tree trap cannot fire any more.** A `__pycache__` (or `.pyc`, `.pytest_cache`, `.mypy_cache`,
> `.ruff_cache`, `.ipynb_checkpoints`) under `docs/audits/evidence/` is re-ignored, so importing a retained
> reproducer can no longer make `git status --untracked-files=all` non-empty and can no longer make
> `strict_code_identity()` refuse the checkout. The fourteen tests now pass **and the tree is still byte-clean
> after they run** — which is the part that was broken before.
>
> **`strict_code_identity()` is exactly as strict as it was.** Any non-empty `git status` still refuses to name a
> governing commit. Nothing under evidence except interpreter and tool caches is re-ignored: a `.py`, `.json`,
> `.csv`, `.log` or `.out` there still dirties the tree until it is committed, and the refusal now names what is
> uncommitted instead of leaving an engineer to find it. Both directions are asserted by tests.
>
> **Q2_CONTEXT v1's preparation is refused, not rewritten.** The sealed bytes are untouched. A validator computes
> the contradiction from them and refuses at `prepare()` and again at `load_data()`, so that preparation can
> neither seal nor be fitted. **No published number moves: no cell of `e1_block_q2_context_v1` was ever fitted.**
>
> **One correction to the earlier report of that defect.** It said v1's design *declares*
> `train_population: COMMON_INTERSECTION`. It does not: v1's sealed `DESIGN.json` carries **no
> `train_population` key at all**. That text is the *block catalogue's*, and the catalogue is where Q2_CONTEXT
> declares `COMMON_INTERSECTION` today. The defect is unchanged in substance and slightly worse in kind — a
> reader of the catalogue would believe the sealed preparation honours a policy the sealed bytes never mention.

## 1. Item 1 — fourteen tests went red whenever the checkout was not byte-clean

### 1.1 The cause, verified rather than reasoned

`git check-ignore -v` blames the exact line:

```
.gitignore:197:!docs/audits/evidence/**   docs/audits/evidence/RP40_REVIEW_2026_09_19/__pycache__/reproduce.cpython-312.pyc
```

That negation exists so the LaTeX `*.log` / `*.out` rules above it cannot swallow a retained PRE/POST
reproduction output. It also un-ignored the interpreter's own cache, so importing any of the 20+ retained
reproducers under evidence wrote an untracked `__pycache__`, `git status --untracked-files=all` surfaced it, and
`tools/governed_run.py` `strict_code_identity()` refused the tree as dirty. The primary checkout was carrying
exactly such an untracked directory (`docs/audits/evidence/RP139_REVIEW_2026_09_23/__pycache__/`) when this
round started.

### 1.2 The repair is in the ignore rules, not in the refusal

`.gitignore`, immediately after line 197, re-ignores only generated caches:

```
docs/audits/evidence/**/__pycache__/
docs/audits/evidence/**/*.py[cod]
docs/audits/evidence/**/*$py.class
docs/audits/evidence/**/.pytest_cache/
docs/audits/evidence/**/.ipynb_checkpoints/
docs/audits/evidence/**/.mypy_cache/
docs/audits/evidence/**/.ruff_cache/
```

No tracked file is covered by any of them (`git ls-files docs/audits/evidence | grep -E …` → empty), so nothing
that was visible becomes invisible.

`strict_code_identity()` keeps its semantics exactly. The only change is that the refusal names the offending
paths:

```
governed_run.GovernedRunError: governing run requires a clean checkout; uncommitted:
 M .gitignore;  M tools/governed_run.py; ?? tests/test_strict_code_identity_evidence_cache.py
```

That message is itself the proof the guarantee survived: the fourteen tests failed on **my own** uncommitted
changes mid-round, and named them.

### 1.3 The tests

`tests/test_strict_code_identity_evidence_cache.py`, nine tests, against a throwaway repository carrying **this
checkout's real `.gitignore`** rather than a paraphrase of its rules:

| test | what makes it fail |
|---|---|
| `…a_generated_bytecode_directory_under_evidence_cannot_dirty_the_tree` | the trap returning: eight generated artifacts under evidence, and `git status` must stay empty and the commit must still be named |
| `…the_committed_gitignore_ignores_those_paths_in_this_very_checkout` | the rules being present in a copy but not in the live repository |
| `…still_refuses_a_genuinely_uncommitted_source_change` | an edit to committed source being allowed to govern, or a refusal that does not name it |
| `…uncommitted_evidence_of_its_own_still_refuses_a_governing_commit` (×5) | a new reproducer, a PRE/POST output, a measurements file or a cell table under evidence buying silence too |
| `…the_evidence_negation_is_still_in_force` | the repair undoing what line 197 is for — a `POST.log` under evidence swallowed by the LaTeX rule |

### 1.4 Measured, in the anaconda env `trading-stack`, CPU only

| run | result |
|---|---|
| `tests/test_strict_code_identity_evidence_cache.py` | **9 passed** in 0.47 s |
| the fourteen + the two that always passed + the nine new, on the committed tree | **25 passed** in 5.25 s |
| `git status --porcelain --untracked-files=all` **after** that run | **empty** |
| `tests/test_df_e1_block.py` (the suite the item-2 change touches) | **21 passed** in 155 s |

## 2. Item 2 — Q2_CONTEXT v1 holds per-arm train origins

### 2.1 What the retained bytes say

| | v1, sealed | v2, the 2026-09-26 bounded block |
|---|---|---|
| design | `6d1aaecaf27c581c…` (file `96ebcb1c55744f1b…`) | `47a270eec01f203c…` |
| `train_population` | **the key is absent** | `COMMON_INTERSECTION` |
| train origins held, per arm | 40 020 / 38 700 / 38 700 / 38 700 / **40 080** | 38 700 / 38 700 / 38 700 / 38 700 |
| the same origin *sets*? | **no** (compared on the arrays) | **yes** (element-wise identical) |
| `train_admissible_before_intersection` | absent for every arm | present for every arm |
| `binding_to_source.common_train_origins` | absent | **38 700** |
| state | `SEALED_NOT_EXECUTED` / `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`, zero cells fitted | fitted: 12 cells |

The block catalogue (`tools/df_e1_block.py` `BLOCKS["Q2_CONTEXT"]`) declares `train_population:
COMMON_INTERSECTION` with *"every arm of this block, the baseline included, trains on the SAME origins so
context is never conflated with volume"*. `git log -S` dates that: commit `ac40e0db` added both the declaration
and the intersection in `prepare()`. Q2_CONTEXT v1 was sealed **before** it, which is why its design has no such
key and its preparation has no intersection. Had a cell been fitted on it, the input contrast would have been
confounded with the very train volume the block says it controls.

### 2.2 The record

[`docs/audits/evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/FINDING.json`](../evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/FINDING.json),
label **`Q2V1-ORIGIN-POLICY`**, generated by `probe.py` beside it from the retained bytes — every count, digest
and verdict in it is read, none is typed. It records the digests it read (including the two `BLOCK_DATA.npz`
files from the local working state, `e895239741e7dc0b…` for v1 and `43c93db381de4e42…` for v2, the latter the
digest the seal document already pinned), the verdict on each, and:

- `published_numbers_affected: 0`, with the reason;
- `usability.e1_block_q2_context_v1: UNUSABLE_FOR_ANY_COMPARISON_THAT_ASSUMES_COMMON_TRAIN_ORIGINS`, and what it
  *is* still usable for — its own retained cost pilots, which measure one arm's resource cost and compare arms
  to nothing;
- `not_repaired_in_place`: the sealed preparation is retained exactly as it is.

**No number from the program's finding allocator is claimed.** That allocator is fragmented across git refs and
only Musashi rules on it; this is a labelled finding against one preparation.

### 2.3 The validator — the durable deliverable

`tools/df_e1_block.py`:

- `TRAIN_POPULATIONS = ("COMMON_INTERSECTION", "PER_ARM_ADMISSIBLE")`, and `"UNDECLARED"` is not a member: it is
  the absence of a policy.
- `train_population_report(design, rec, origins=None)` — the origins a preparation actually **holds**, read from
  `counts_from_identities[arm]["labels"]` (the post-intersection count), and, when the arrays are available,
  whether the origin *sets* are identical, because two arms can hold the same number of different origins.
- `validate_train_population(...)` refuses, naming the disagreement, when: an arm has no held count; the policy
  is outside the vocabulary; the policy is `COMMON_INTERSECTION` and the counts differ, or the counts agree and
  the sets do not, or no `train_admissible_before_intersection` is recorded, or no `common_train_origins` is
  recorded, or the recorded common count is not the one the arms hold; or the policy is **undeclared and the
  arms hold different origins** — which is v1.

It is called from `prepare()` (on the arrays, before the record is written: a preparation that contradicts its
own design never seals) **and** from `load_data()` (before anything is fitted: a retained preparation that
contradicts its design cannot be consumed, whenever it was sealed).

An undeclared policy whose arms *do* hold the same origins is allowed and reported as `UNDECLARED`, not refused:
it is readable, and refusing it would break preparations that are factually fine.

### 2.4 The tests

`tests/test_df_e1_train_population_validator.py`, **14 passed** in 0.18 s. On the retained bytes: the v1 record
is refused and the refusal names each arm's count; the v1 *arrays* are refused too (skipped where the working
state is absent); the v2 record is accepted with 38 700 and the intersection evidence present; the v2 arrays are
element-wise identical at 38 700 across all four arms; and the finding artifact must keep naming the
preparation, the digest and the verdict. Then every refusal branch on synthetic records, plus
`PER_ARM_ADMISSIBLE` legitimately differing, plus a check that both `prepare()` and `load_data()` call the
validator.

## 3. What is NOT done, refused, or not measured

- **v1's retained preparation is not corrected and not deleted.** It is refused at consumption. What to do with
  the bytes is the owner's call; rewriting a retained artifact is not mine.
- **The seal gap `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` is untouched** and stays `UNMET`. Nothing here
  fits a cell or moves a number.
- **The smaller sibling defect is still there, by choice.** After the intersection is applied,
  `feasibility[arm]["train_admissible"]` still reports the **pre**-intersection count under a name that reads as
  the count used — in v2's bytes too (`modular_w60`: `train_admissible` 40 080, origins held 38 700). Renaming
  or re-defining it would change the facts digest of a sealed preparation. The validator reads
  `counts_from_identities`, never that field, and the report names both, so nothing depends on the misleading
  name; the rename belongs with whoever may re-seal.
- **`PYTHONDONTWRITEBYTECODE=1` was not adopted as the fix.** It would work only for a process that remembers
  to set it. The ignore rules work for every process.
- **No governed run, no fit, no training of any kind was executed in this round**, and no service was started,
  stopped or restarted.
- **The stale legacy `tests/` suite is still stale.** It was not touched and was not run.

## 4. Report

```
DEFREP — items 1 and 2 of the 2026-09-26 defect order (item 3 is in data-gov)
repo/branch/tip: predictor / satoshi/defect-repairs-and-fred-20260926 / b24f71e4
files: .gitignore (re-ignore caches under evidence)
       tools/governed_run.py (strict_code_identity names what is uncommitted; semantics unchanged)
       tools/df_e1_block.py (TRAIN_POPULATIONS, train_population_report, validate_train_population;
                             called from prepare() and load_data())
       tests/test_strict_code_identity_evidence_cache.py (9 new)
       tests/test_df_e1_train_population_validator.py (14 new)
       docs/audits/evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/{probe.py,FINDING.json}
       docs/audits/work_plan/SATOSHI_DEFECT_REPAIRS_2026_09_26.md
suites (anaconda env trading-stack, CUDA_VISIBLE_DEVICES='', under crispdm-run -m 3G):
       test_strict_code_identity_evidence_cache 9/0 · test_df_e1_train_population_validator 14/0
       test_df_e1_governed_route + test_df_sota_lake_adopt + the 9 new 25/0 (the fourteen were the red ones)
       test_df_e1_block 21/0
acceptance: the dirty-tree trap CANNOT still fire — git check-ignore covers __pycache__/.pyc/.pytest_cache/
       .mypy_cache/.ruff_cache/.ipynb_checkpoints under evidence, the 25-test run left
       `git status --porcelain -uall` EMPTY, and an uncommitted .py/.log/.out/.json/.csv under evidence still
       refuses to govern · validate_train_population REFUSES the v1 bytes (per-arm 40020/38700/38700/38700/40080,
       arrays not identical) and ACCEPTS the v2 bytes (38700 identical counts AND identical arrays,
       common_train_origins 38700)
what is NOT done / refused / not measured: v1's sealed preparation refused, never rewritten · no cell fitted,
       no published number moved · the pre-intersection `train_admissible` misnomer left in place on purpose
       (renaming it would change a sealed preparation's digest) · ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR
       stays UNMET · no finding number claimed from the fragmented allocator · legacy tests/ suite still stale,
       not run · no service touched
```
