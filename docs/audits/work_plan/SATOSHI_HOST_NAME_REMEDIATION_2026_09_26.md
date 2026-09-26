# Host-name remediation — today's breach repaired, the pre-existing one measured

**Date:** 2026-09-26
**By:** Satoshi, successor technical lead
**Rule at issue (owner's, standing):** never write a machine host name, IP address, token or account identifier
into any repository. The worker host exists only as an environment variable read from the operator's config
outside the checkouts. `AGENTS.md` carries the same rule: *"Never write account identifiers, broker credentials,
private IP addresses or machine host names into files here. Use placeholders such as `<your-host>`."*

**The name itself appears nowhere in this document.** It is referred to as *the coordinator host name*. Every count
below was produced by a pattern match, never by an echo of the value. The reproduction commands in §3.7 are written
with the value as a placeholder the reader substitutes.

**The two halves are kept apart.** §1–§2 are today's breach, repaired. §3 is the pre-existing breach, measured only.
No remediation of any kind was executed on `master` or `main`.

---

## 1. Headline

1. **The remote branch no longer serves the coordinator host name — at any commit.**
   `origin/satoshi/q2-context-deep-arms-20260926` was force-published after its five own commits were rebuilt
   (`b1a13928`…`d8e9386b` -> `dabd8172`…`00a2b48f`, map in §2.4) with this remediation committed on top. A pattern match over every commit of the branch returns nothing the branch introduced. The 212 files it
   inherits from `origin/master` still carry it; that is half two and is not this branch's to fix.
2. **The breach was three committed files, not two.** The lane reported `DESIGN.json` and `EXTENSION_SEAL.json`.
   It missed `tools/df_e1_block.py`, the generator whose `BLOCKS` registry literal is the *source* of the string both
   JSON files copy — and, in the branch's history, a fourth: `FOOTPRINT_BASIS.json` carried it at commit `ced14654`
   and was scrubbed only at the tip, so the pushed history served it.
3. **Inventory headline (half two, measurement only).** On `origin/master`, the public default branch:
   **212 files** carry the coordinator host name, in **705** word-boundary occurrences; **86** of those 212 cannot be
   redacted without breaking a digest and **126** can; **68 file paths** on `master` carry a fleet host name in the
   *filename*. On `origin/main`: **1 file** — a committed LibreOffice lock file, which carries `user@host` and is the
   single most disclosive item found. **13 tools, 40 call sites** still write the host name into artifacts, so the
   count grows on its own.

---

## 2. Half one — today's breach, repaired

### 2.1 What was wrong, and why the previous reasoning was wrong

The lane reported the breach instead of repairing it, on the ground that the name sits inside the body the design's
`sha256` covers, so editing it would make the design refuse validation. Three reasons were given. Measured:

| reason given | verdict | measurement |
|---|---|---|
| (1) editing the published copy breaks the digest that makes the seal checkable | **true** — and it is why v1 was *not* edited in place. It does not reach the remedy it excludes: withdraw v1, seal v2 | — |
| (2) editing `tools/df_e1_block.py` changes a sha the design pins in `source_code`, so `validate(strict_code=True)` would refuse the design, and the block's two unstarted W1440 arms must stay runnable | **false in substance** | Of the **fourteen** committed design records in this repository that pin `source_code.df_e1_block.py` (thirteen sealed block designs plus `RP82/fin_cost_pilot/DESIGN.json`), **thirteen already pinned a digest the pre-edit file did not have** — that file is a living tool and has drifted many times in ordinary practice, so `strict_code` was already refusing all thirteen. v1 was the only design still matching it. The edit therefore cost exactly one strict validation: the withdrawn design's. v2 pins the repaired file and validates with `strict_code=True`, so the two unstarted arms remain runnable |
| (3) re-sealing is not available: cells already have scores, and a design is never re-sealed after a score | **true as stated, and not what was done** | v1 is not re-sealed. It is **withdrawn by digest**, and the twelve landed fits are **not** re-bound to v2 |

The priority was inverted. A seal exists to prove a design predated its scores. With **zero verified rows**, custody
`UNCHECKED` on all eighteen registered cells, a **FAILED** closure and disposition `HISTORICAL_DEV_ONLY`, v1's seal
guarded nothing admissible. The owner's rule against publishing a machine name outranks the checkability of a digest
that guards nothing.

### 2.2 The v2 seal

Sealed by the committed command, not by hand:

```
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python tools/df_q2_context_deep_seal.py \
  --out docs/audits/evidence/E1_Q2_CONTEXT_DEEP_20260926/DESIGN.json \
  --crossing-out docs/audits/evidence/E1_Q2_CONTEXT_DEEP_20260926/EXTENSION_SEAL.json
```

| | v1 (withdrawn) | v2 (standing) |
|---|---|---|
| `design_sha256` | `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318` | `27ed712d88e0b1a84a3a7d48d5a8dcd899274389f8cc7341df196705dbb0300d` |
| `source_code.df_e1_block.py` | `2e956b9f0ad8bf934ca63398f95c4eb0767c3fff2250c676c7dc92c86e7387e6` | `65330dc1fc437eb4a509d85ff5a7f915cc0202ea8bc124dd88acafcdde248093` |
| state | withdrawn for carrying a machine host name | `SEALED_NOT_EXECUTED` |
| digest recomputes over its own body | — | **yes**, verified |
| `validate(strict_code=True)` | refuses (its pinned code no longer exists) | **passes** |
| `EXTENSION_SEAL.json` re-derived from it | — | **byte-identical** to the committed copy |

**v1 → v2 differs in exactly three leaf values of the design body**, verified leaf by leaf over the flattened
documents: `factorial.informed_by` (one clause: *"on this same host (`<name>`)"* → *"on the coordinator host"*),
`source_code.df_e1_block.py`, and `design_sha256` itself. Every arm, seed, crop, dilation list, depth, parameter
count, recipe field, update ceiling, resource limit, scaler rule, common-evaluation rule, train population, declared
contrast, reading rule and benchmark contract is byte-identical to v1. **No science changed.**

The supersession is recorded, not silent, in
`docs/audits/evidence/E1_Q2_CONTEXT_DEEP_20260926/SEAL_WITHDRAWAL.json`: v1's digest, the reason
`WITHDRAWN_FOR_CARRYING_A_MACHINE_HOST_NAME`, v2's digest, the leaf-by-leaf difference, the three refusal reasons
answered, and the binding of the twelve fits. `REDACTIONS.json` of the same directory keeps its original reasoning
and carries the correction of it beside it. v1 was **not** edited in place and is **not** pretended away.

### 2.3 The twelve landed fits

**Binding: `NOT_BOUND_TO_A_SEAL`.** Not re-bound to v2. Every cell record, and `REPORT.json`, `TABLES.md`,
`CLOSURE_TABLE.json` and `UNANCHORED_MEASUREMENT_TABLE.json`, carry `design_sha256 = 7d0bf921…` — v1's digest. That
is what they were actually run against and it is the truth of the artifacts. v1 is withdrawn, so their seal is
withdrawn with it; they stand under no seal. Re-binding them to a seal created *after* their scores existed would be
exactly the back-dating a seal exists to prevent.

**Their numbers are kept, unaltered**: twelve rows of MAE / MAE_z / naive / skill on 10 020 common evaluation rows,
all `CENSORED_BY_BUDGET` at 600 observed updates, six cells never fitted (the two W1440 FULL-DEPTH arms × three
seeds). **Custody is not upgraded**: `UNCHECKED` for all twelve, 0 verified rows, closure FAILED,
`HISTORICAL_DEV_ONLY`. A fit bound to no seal is *weaker* evidence than it was this morning, not stronger. Nothing
may promote, select, rank or recommend on them — as was already the case. If the block is to yield evidence, it is
refitted under v2 with the two W1440 arms when shared admission control is repaired.

### 2.4 What was removed from the remote branch, and how

The bytes had to stop being served, and the branch is a few hours old with nothing downstream depending on it, so the
branch's history was rewritten — **only this branch**, never `master`, never `main`, never the pushed parent branch.

- **Scope.** Exactly the five commits above `7d2b1c83`, the pushed tip of the parent branch
  `satoshi/e1-seal-q2-context-20260926`. That parent tip and everything below it were left untouched, so no other
  ref moved. No commit message on the branch carried the name (checked: zero matches).
- **Method.** Not `filter-branch` and not `filter-repo` — this checkout carries 37 worktrees and 46 remote branches
  and a whole-repo rewriter is the wrong instrument. The five commits were rebuilt one at a time with git plumbing
  (`read-tree` → `update-index` → `write-tree` → `commit-tree`), preserving each commit's message, author, committer
  and both dates exactly. Only four paths were altered in any tree.
- **What each rebuilt tree says.** `DESIGN.json` and `EXTENSION_SEAL.json` become a
  `withdrawn_artifact_placeholder.v1` record naming v1's digest, the reason, v2's digest and where to read the
  withdrawal — rather than a hole, so the supersession is visible at every commit and no intermediate commit serves
  either the name or a digest that no longer recomputes. `tools/df_e1_block.py` and (at one commit)
  `FOOTPRINT_BASIS.json` have the one offending clause replaced by the same wording the tip uses.
- **Commit map.**

  | old | new |
  |---|---|
  | `b1a13928` Seal the six-arm Q2_CONTEXT_DEEP block before any cell of it has a score | `dabd8172` |
  | `82c633e8` Add the memory gate and the crossing table generator this block needs | `5b6e2043` |
  | `91ad13a5` Declare how the context contrast must be read before its long-window term exists | `f3aa314b` |
  | `ced14654` Stand down, record every termination by mechanism, and correct the footprint basis | `9c4796fc` |
  | `d8e9386b` Deliver Q2_CONTEXT_DEEP as it landed: the seal, twelve fits, and the question still UNMET | `00a2b48f` |

- **Then one new commit on top** installs the v2 seal, the withdrawal record, the corrected `REDACTIONS.json` and
  `FOOTPRINT_BASIS.json`, the code repairs of §2.5, and this document. The branch was force-published to that tip.
- **Verified after the rewrite.** For each of the five rebuilt commits and for the tip, the set of pattern-carrying
  files equals the set inherited from `origin/master` — the branch introduces **none**. The trees differ from the
  originals in exactly the four paths named above and nowhere else.
- **I did not judge a rewrite unsafe.** The one caveat the owner should hear: a force-push does not retract anything
  already fetched, and GitHub keeps unreachable objects addressable by sha for a time, so anyone who fetched this
  branch in the last few hours still holds v1's bytes locally. Nothing about this branch was public beyond this
  fleet, and no fork or clone of it is known.

### 2.5 The cost-basis correction, carried forward

**The correction.** Every footprint figure this round used or published came from
`resource.getrusage(RUSAGE_SELF).ru_maxrss` — **main-process RSS only**. The correct basis is the peak charged to the
scope's cgroup. For the same arm (`long_window_own_depth`, 2026-09-26 pilot) that is **7.4 GiB cgroup peak against a
declared 9 GiB cap**, where the main-process figure was **7.88 GiB**. Figures are kept labelled by basis, per figure,
in `FOOTPRINT_BASIS.json`; nothing is restated as if it had been a tree peak, and the unresolved case recorded there
(a main-process high-water mark of 7.05 GiB *above* the 6.7 G cgroup peak of the same scope) stays unresolved and
labelled, not explained away.

**The defect is real, and is now repaired.** `run_cell` recorded *only* `getrusage(RUSAGE_SELF).ru_maxrss` as
`cost.peak_rss_bytes`, with no cgroup or tree peak anywhere in the cell record: every cost pilot that ever justified
a placement from this runner justified it from the wrong basis. `tools/df_e1_block.run_cell` now records

- `cost.cgroup_memory_peak_bytes` — `memory.peak` of its own cgroup, via a new `cgroup_memory_peak_bytes()` helper
  that returns `None` when the cgroup is unreadable, so a missing tree peak *says* the cell has none rather than
  falling back to the wrong basis; and
- `cost.peak_rss_bytes_basis` — the literal label `MAIN_PROCESS_RSS_ONLY …` beside the old figure.

Both bases are reported side by side in the closure rows.

**Why the repair was admissible here, against the owner's condition** (*repair only if it does not touch a file pinned
by a seal that must stay checkable*): it touches `tools/df_e1_block.py`, pinned by fourteen committed design records. **Thirteen of
the fourteen already pinned a stale digest of that file** and are already outside `strict_code` through ordinary drift, not
through anything done here. **The fourteenth is v1**, which this record withdraws and which must *not* stay checkable. So
**no seal that must stay checkable is harmed**, and v2 pins the repaired file.

**A second defect found in the same line, and repaired.** `run_cell` also wrote
`cost.host = os.uname().nodename` into **every** cell record. That single line is the machine source of the
pre-existing breach: **45 committed `cell.json` files on `origin/master` carry the coordinator host name because a
tool put it there**, and **44 of those 45** have their blob digest pinned in a `TERMINALS` receipt or a
`CLOSURE_TABLE`, so they can no longer be redacted without breaking a pin. `run_cell` now records
`cost.host_identity` — a salted 16-hex opaque per-host id, stable across runs on one host and different across hosts,
with `host_name_recorded: false` — and `cost.host` is `None`. The question a reader actually asks of that field
(*same host or not?*) is still answerable; the machine is not named. The one remaining status print that echoed the
nodename now prints the same opaque identity. Three consumers of the old key were updated to read `host_id`.

### 2.6 Verification of half one

| check | result |
|---|---|
| `tests/test_df_e1_block.py` + `tests/test_df_closure_table.py` | **58 passed**, 0 failed (CPU only, `CUDA_VISIBLE_DEVICES=''`, under the memory guard) |
| v2 `design_sha256` recomputes over its own body | yes |
| `EXTENSION_SEAL.json` quotes v2's digest | yes |
| v2's pinned `df_e1_block.py` digest == the file on disk | yes |
| `validate(strict_code=True)` on the committed v2 design | passes |
| `EXTENSION_SEAL.json` re-derived from the committed design | byte-identical |
| pattern match over the evidence directory and `tools/df_e1_block.py` | no match |
| pattern match over every commit the branch introduces | no match |

---

## 3. Half two — the pre-existing breach, measured only

**Nothing here was executed.** No file on `master` or `main` was changed, no history was rewritten, no branch was
touched. This is the measured basis for the owner's decision.

### 3.1 Method, so the owner can reproduce and so the value is never echoed

Counts are over **tracked blobs of `origin/master` and `origin/main`** — not the working tree, not the 37 worktrees,
not untracked files. Two measures are reported and they differ:

- *files with the token* — a case-insensitive substring match anywhere in the blob;
- *word-boundary occurrences* — the token not adjacent to `[A-Za-z0-9_]`, which is what a host name looks like.

A third measure, *files with host context*, requires at least one word-boundary occurrence on a line that also
mentions a host, hostname, nodename, ssh, `@`, worker, machine, node, `.local`, coordinator, fleet or a sweep-script
name. It exists because two of the four fleet names are also ordinary technical words.

### 3.2 The coordinator host name on the public default branch

| measure | `origin/master` | `origin/main` |
|---|---|---|
| files carrying it | **212** | **1** |
| word-boundary occurrences | **705** | 1 |
| file **paths** carrying it in the filename | 18 | 0 |

`origin/master` is the repository's default branch (`origin/HEAD → origin/master`) on a public GitHub remote.
`origin/main` is a long-diverged sibling (1 002 commits behind, 1 721 ahead-of-nothing: the two histories have
diverged, not one being a prefix of the other). The single `main` file is a committed **LibreOffice lock file** under
`examples/results/phase_2_daily/`, which carries `user@host` — a user account identifier *and* a machine name, in a
build artifact that should never have been committed at all. It is the most disclosive single item found and it also
exists on `master`.

### 3.3 Reconciling the owner's ≈349

My exact figure for the coordinator name alone on `origin/master` is **212**, not ≈349. The gap is accounted for and
the owner can pick the definition he meant:

| what is counted, on `origin/master` | files |
|---|---|
| the coordinator host name only | **212** |
| ∪ the three other fleet host names, token match | **429** |
| ∪ the three others, restricted to a host context | **264** |

≈349 sits inside that range, so the owner's grep almost certainly spanned more than one machine name (two of the
other three tokens are also ordinary technical words, which inflates a bare token match), and/or ran over the working
tree rather than a single ref. Per-name, token match / host-context, on `origin/master`: coordinator 212 / 179,
second host 208 / 91, third 98 / 71, fourth 42 / 4.

### 3.4 The 212 files grouped by kind, with the digest question answered per group

"Digest-affected" means one of two measured things: the file carries a **self-covering** digest (a top-level
`*_sha256` that recomputes over the rest of its own body, so redaction stops it recomputing), or the file's **own
blob sha256 is pinned in some other file on `master`** (a `TERMINALS` receipt, a `CLOSURE_TABLE`, a
`METADATA_BACKUP_MANIFEST`, a `PILOT_FREEZE`, `E1_SEAL.json`, a design record), so redaction breaks that pin.

| group | files | digest-affected | occurrences | would redaction break a digest? |
|---|---|---|---|---|
| **A. sealed designs whose own digest covers the text** | 3 | 3 | 5 | **Yes, structurally.** `RP65/PHASE2_DESIGN_SEALED.json`, `RP74/FIN_LOSS_OPT_DESIGN_SEALED_v3.json`, `RP82/FIN_LOSS_OPT_DESIGN_SEALED_v4.json`. One is also pinned in `E1_SEAL_20260926/E1_SEAL.json` |
| **B. machine-generated cell records** (`cost.host` from `os.uname().nodename`) | 45 | 44 | 46 | **Yes for 44.** Their blob digests are pinned in `TERMINALS/<unit>.json` and in `CLOSURE_TABLE_RP74/RP81/RP89.json`. One is free |
| **C. other evidence JSON** — `DELIVERIES.json` ×12, `REPORT.json` ×7 (+6 per-host variants), `SOTA_TABLE.<host>.json` ×4, `RUN_LEDGER.json` ×4, `TERMINAL_RECEIPTS.json` ×3, deletion receipts, per-host CPU ledgers, RP30 `TERMINALS/*` ×14 | 105 | 33 | 483 | **Yes for 33** (pinned in a manifest, a freeze, a closure table or `E1_SEAL.json`); **no for 72** |
| **D. ordinary documents** — return packets, READMEs, suite-summary `.txt`, the three `STEP_*_FINAL.md` methodology chapters | 44 | 6 | 153 | **No for 38.** **Yes for 6**: three `SATOSHI_PROGRAM_RP*_RETURN` packets pinned in `E1_SEAL.json` and three `STEP_*_FINAL.md` pinned in two `D3/D4 …DESIGN.v1.json` records and their Retsu letters |
| **E. code** — 2 per-host sweep scripts at the repo root, 5 test files with the name as a **string fixture**, 4 evidence repro scripts, `analyze_candidate_history.py`, `tools/df_fin_loss_opt_design.py` | 13 | 0 | 16 | **No** for the content itself. But editing a **tool** changes a sha that sealed designs pin in `source_code`; that must be measured per tool before each edit (for `df_e1_block.py` it was measured today: 13 of 14 already stale) |
| **F. example config, and the committed lock file** | 2 | 0 | 2 | **No** |
| **total** | **212** | **86** | **705** | **86 break something; 126 are free to redact** |

By directory: `docs/audits/evidence` 168 · `docs/audits/work_plan` 21 · `docs/tres_temas_entrevista` 9 · `tests` 5 ·
repo root 3 · `docs` 2 · `examples` 2 · `docs/integracion_workplan_2026_09_10` 1 · `tools` 1.

### 3.5 The name is also in 68 file paths, which no content redaction reaches

`origin/master` has **68 tracked paths** whose *filename* carries a fleet host name (coordinator 18, second host 30,
third 18, fourth 2) — per-host CPU ledgers, per-host `REPORT.<host>.json`, `SOTA_TABLE.<host>.json`, deletion
receipts, the per-host sweep scripts. Renaming a file does not change its blob digest, so a rename breaks no content
pin — but it breaks every *path* reference, including the path strings recorded inside pinned manifests and closure
tables. This is a separate, smaller job and must be costed separately.

### 3.6 Why the count grows on its own

**13 tools, 40 call sites** on `origin/master` read `os.uname().nodename`, `socket.gethostname()` or
`platform.node()` and write the result into records: `df_sota_repro.py` (23), `df_e1_block.py` (2, repaired today),
`df_e1_governed.py` (2), `df_e1_pilot.py` (2), `df_ecl_modular.py` (2), `df_public_lake_adopt.py` (2), and one each
in `df_cpu_ledger.py`, `df_e1_full_rehearsal.py`, `df_e1_innovation.py`, `df_e1_phase1.py`, `df_fin_runner.py`,
`df_worker_probe.py`, `worker_delivery_probe.py`. Group B above — 45 committed cell records — is entirely
machine-written. **Any remediation that does not change the producers is undone by the next run.**

### 3.7 Reproduction commands (substitute the value; never commit them with it filled in)

```
H='<the coordinator host name>'
git grep -liE "(^|[^a-z0-9])$H([^a-z0-9]|$)" origin/master -- . | wc -l     # expect 212
git grep -liE "(^|[^a-z0-9])$H([^a-z0-9]|$)" origin/main   -- . | wc -l     # expect 1
git ls-tree -r --name-only origin/master | grep -ciE "(^|[^a-z0-9])$H([^a-z0-9]|$)"   # expect 18
git grep -nE 'uname\(\)\.nodename|socket\.gethostname|platform\.node\(' origin/master -- '*.py' | wc -l  # expect 40
```

### 3.8 The options, with their real costs

**Option 1 — leave as is.**
*Cost:* the owner's standing rule stays broken on a public default branch in 212 files and 705 occurrences, and the
number grows every run because 13 tools still write the name. *Benefit:* no digest breaks, no rewrite, no effort.
*Assessment:* not acceptable as an end state; it is acceptable only as the state of the 86 files Option 3 cannot
touch, and only with those 86 named in a record.

**Option 2 — redact going forward only (fix the producers).**
*What:* change the 40 call sites to an opaque per-host id of the kind installed in `df_e1_block.py` today, and add a
repository pattern guard (a pre-commit hook or a CI check) that refuses a commit introducing a fleet host name.
*Cost:* ~13 code files; each tool edit changes a sha that sealed designs may pin in `source_code`, so each needs the
same per-tool measurement done today (`df_e1_block.py`: 13 of 14 already stale, so the edit cost one
validation) — cheap but not free, and it must be measured rather than assumed. The existing 212 files stay.
*Benefit:* the breach stops growing; the exposure becomes a finite, closed set the owner can then decide about once.

**Option 3 — redact documents but not sealed artifacts.**
*What:* one commit redacting the **126** free files (group E, group F, 38 of the 44 documents, 72 of the 105 other
evidence JSON, 1 cell record), plus a `REDACTION_REFUSED` ledger naming each of the **86** that cannot be redacted
without forging a digest, with the per-file reason and the digest that would break.
*Cost:* one large, mechanical, reviewable commit; a permanent and documented public residue of 86 files, among them
3 sealed designs and 44 machine-written cell records; plus the 68 paths of §3.5 if the owner wants those too.
*Benefit:* removes about **60%** of the public exposure (126 of 212 files; by occurrences, the free files carry the
bulk of the 705) without forging a single digest and without rewriting one byte of published history.

**Option 4 — rewrite history on `master` / `main`.**
*Cost:* `master` is the published default branch. It carries 46 remote branches and, in this checkout alone, 37
worktrees, all of which would need rebasing; every commit sha an audit packet names would change, and this corpus
names commit shas as identity throughout; and it **does not actually un-publish** — any existing clone or fork keeps
the old objects, and the forge keeps unreachable objects addressable by sha for a period. *Benefit:* the only option
that removes the bytes from the default branch's reachable history.
*Assessment:* the cost is very high, the benefit is partial by construction, and the risk of silently invalidating
audit identities across the corpus is the kind of damage that cannot be walked back.

### 3.9 Recommendation

**Do Option 2 now, then Option 3 on the owner's word. Do not do Option 4.**

1. **Now, needs no decision:** finish the producer fix — the remaining 39 call sites in 12 tools, each with its own
   `source_code`-pin measurement first — and add the repository pattern guard so the count cannot grow. One tool was
   done today as the worked example. This is the only step that changes the trajectory rather than the snapshot.
2. **Then, one commit, owner's word:** Option 3's 126 redactions plus the `REDACTION_REFUSED` ledger for the 86.
   Decide the 68 paths of §3.5 separately, after that.
3. **Separately and first, because it is the worst single item:** the committed LibreOffice lock file carrying
   `user@host`, present on both `master` and `main`. Deleting a build artifact at the tip is an ordinary commit, not
   a history rewrite. I did not do it — it is on `master` and `main` and that scope is the owner's — and I recommend
   it be the first thing done. **Related live risk, not acted on:** the owner's own `predictor` checkout currently
   holds an untracked LibreOffice lock file of the same kind beside a presentation draft. It is untracked and it is
   his, so it was not touched; a single `git add -A` would publish a `user@host` string. This is one more reason the
   pattern guard of Option 2 should refuse the commit rather than rely on anyone noticing. `.gitignore` should also
   carry `.~lock.*#`, which it does not.
4. **Never Option 4** on `master` or `main`. It does not un-publish, it breaks every open branch and every commit
   identity the audit corpus depends on, and it buys nothing that steps 1–3 do not.

---

## 4. What is NOT done, refused, or not measured

- **No remediation was executed on `master` or `main`.** Not one file, not one byte, not one ref. §3 is measurement.
- The **86 digest-affected files** are named by group and by pin in §3.4, and the full per-file list with the
  breaking digest is reproducible with the commands in §3.7; it is not transcribed here because it would have to
  carry paths that themselves contain host names.
- **The other three fleet host names are counted but not classified.** §3.3 gives their file counts and host-context
  counts; the group-by-kind and digest analysis of §3.4 was run for the coordinator name only.
- **Sibling repositories were not measured.** The owner's rule spans every repository; this inventory is `predictor`
  alone, on two refs.
- **The 39 remaining producer call sites are not repaired.** Only `df_e1_block.py` was, because that is the file
  today's breach was in. Each of the others needs its own `source_code`-pin measurement before it is touched.
- **Nothing about the twelve fits was upgraded.** Custody stays `UNCHECKED`, 0 verified rows, closure FAILED,
  `HISTORICAL_DEV_ONLY`, and they are bound to no seal.
- **No new measurement of the block was taken.** No fit, no pilot, no GPU, no heavy compute: the host's shared
  admission is broken and under repair. Everything run here was CPU-only under
  `$HOME/.local/bin/crispdm-run -m … -t … -n hostfix --`, and the guard refused three oversized requests, which were
  re-issued smaller rather than bypassed.
- **The owner's untracked files were not touched**, in this checkout or any other. The lock file named in §3.9(3) is
  reported, not removed, and `.gitignore` was not edited: both are on `master`'s side of the line.
- **A force-push does not retract what was already fetched.** Anyone who fetched this branch in the hours before the
  rewrite still holds v1's bytes locally, and the forge keeps the unreachable objects addressable for a period. No
  fork or clone of this branch is known outside the fleet.

---

## 5. Correction to this document and to the commit that carried it

The first version of this document, and the message of the commit
`Withdraw the seal that named a machine, seal v2, and repair the cost basis`, said **ten** committed designs pin
`source_code.df_e1_block.py` and **nine** were already stale. Re-measured over every record on this branch that
carries that field: **fourteen** records pin it (thirteen sealed block designs plus `RP82/fin_cost_pilot/DESIGN.json`),
and **thirteen** of the fourteen already pinned a digest the pre-edit file did not have. The one that matched was the
withdrawn v1 design. The conclusion is unchanged and in fact stronger: the edit cost exactly one strict validation,
the withdrawn design's. The number in that commit message is wrong and is corrected here rather than by rewriting it.
