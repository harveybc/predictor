# Return packet — B4 R15–R18, T2 R11–R15, CRISP-DM C53–C66

**Date:** 2026-09-12
**Order:** `predictor@ec55d8fa`
**Disposition:** `READY_FOR_MUSASHI_EXTERNAL_REVIEW`
**Authorizes:** nothing. No GPU was opened, no training run, no
confirmation touched, no venue contacted, no cube truncated, no DOIN
publication, and no record of Musashi's was created.

---

## 1. My own defects first

Twelve, all found by my own batteries or by the POST reporting a green
suite as a failure. Each is stated before any result it touches.

1. **A method is not a module function.** The DAG walker looked up
   `series.mean()` as a function named `mean`, found three of them, and
   declared `macd` and `cci_14` UNRESOLVED — for a reason about *my
   code*, not theirs.
2. **Fixing that broke the other direction.** With module attributes
   now resolvable, `np.log(...)` began resolving against the seventeen
   local functions named `log`, and every logarithmic return became
   UNRESOLVED. Only a **bare-name** call may be a helper.
3. **My own fixtures were indexed as producers.** The C55 test file
   deliberately defines ambiguous helpers; indexing it turned six real
   columns UNRESOLVED. A test is not a producer.
4. **`np.roll(a, -1)` carries its amount SECOND.** I read `args[0]`,
   which is the array, so every `np.roll` leak came out UNRESOLVED
   instead of the detectable `NON_CAUSAL` it is.
5. **I conflated two different digests.** The census's `census_sha256`
   is a digest of its CONTENT; the sha256 of its bytes is a different
   value, and the census receipt records them as separate fields. My
   verifier demanded they be equal and refused a perfectly intact
   census.
6. **I guessed the terminal naming rule.** My first verifier assumed
   md5 and declared all 1,965 terminals misfiled. A verifier does not
   guess: it now tests seven candidate rules against every
   `(variable_id, filename)` pair and publishes the ones that survive.
7. **A duplicate erased its own evidence.** `seen[vid] = doc`
   overwrote the earlier file, destroying exactly the misfiling
   evidence the naming check exists to find.
8. **An absent file was counted once per variable**, inflating one
   gone file into hundreds of findings.
9. **My POST passed two test paths inside ONE argv string**, so pytest
   looked for a file literally named `"tests/a.py tests/b.py"` and
   reported a 61-test passing suite as "no tests ran".
10. **My POST read B4's commit A from a `publication` block that does
    not exist** — it lives under `code_identity.commit` — and so read
    an empty string.
11. **A custody defect, not a code defect.** The census fixing the
    denominator of all 1,965 terminals existed ONLY inside a session
    scratchpad. A terminal set whose denominator lives in a temporary
    directory is anchored to nothing. It is now committed under its
    own content address, verified to hash to the name it already
    carried.
12. **I rewrote my own intermediate.** The 1,965 v2 terminals were
    first written by a verifier I then corrected, leaving their
    recorded `verifier_sha256` unreachable. I removed and rewrote
    **my own uncommitted output** — v1 was never touched, and the
    inode census before and after is identical.

---

## 2. B4 R15–R18 — descriptor-first custody

A photograph now retains the directory it photographed. `DirSnapshot`
holds its own fd plus `(device, inode, uid, mode)`, so renaming or
replacing a photographed directory cannot change what is consumed, and
a restore under the same name is visible because the inode moves.

* 61 focal tests, including the directory swap, the intermediate swap,
  restore-after-swap, a slashed name, a file outside the inventory, a
  read after close, descriptor-leak checks on success **and** on
  exception, world- and group-writable directories, a foreign-owned
  directory, and three guard mutants.
* The submission binds commit A `c578fef3`, reachable from a remote
  ref; commit B is `0eab2705`. Two-phase publication holds.

## 3. T2 R11–R15 — one verified snapshot, reproduced

Each of the 726 artifacts is read **once**, through one retained
descriptor, and both the pinned verifier and the screen consume that
same instance. Reproduced from the single audited snapshot:

| | |
|---|---|
| inventory | exact, 242 sealed units, 726 artifacts |
| adjudication | 242 `COMPLETED_VERIFIED`, 0 `TERMINAL_FAILED` |
| verdict | `DOES_NOT_ADVANCE` |
| primary estimand | **−0.001048443391358884** |
| retraining / downloads | none |

The published sign-test p-value **1.3125 is not a probability**. It
stands `SUPERSEDED` by the exact two-sided table
`0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125`; the
corrected value for 3 of 6 is **1.0**. It changes no panel effect, no
estimand and no verdict, and the historical envelope is **not
rewritten**.

Two things are published as **blocked**, not as done:

* the audit snapshot is missing three replay files, so its identity
  assertion is a **refusal** and appears as one;
* **R12's single recoverable checkout remains impossible.** The pinned
  gate requires `HEAD == 7bcd3f0d`, and a tree that also carries the
  replay code is necessarily a different commit. Pointing the gate's
  `repo_root` at a pristine checkout while running code from elsewhere
  would satisfy the check without satisfying its meaning, so it is not
  done. The two-tree arrangement actually used is declared, both trees
  are recoverable, and the resolution — re-pin the execution record, or
  declare digest equality sufficient — **is yours**.

The submission publishes a logical id and `physical_paths: WITHHELD`.
Commit A `ee1d7904` was pushed before the submission existed; the
submission was generated from a detached, verified-clean checkout of
that commit; commit B `26b4214b` carries the submission alone.

## 4. C53 — the labels the numbers did not earn

Recorded **additively**: no prior file edited, no prior number changed.

| Old label | Value | New label |
|---|---|---|
| `ACTIVE_EXECUTION_DEMAND.executable_configs` | 137 | `INPUT_FILES_PRESENT_CONFIG_CANDIDATES` |
| `…unreachable_configs` | 87 | `INPUT_FILES_ABSENT_CONFIGS` |
| `active_x_subjects` | 162 | `RAW_CONFIG_DERIVED_PENDING_EFFECTIVE_CONFIG` |
| `active_y_subjects` | 162 | `RAW_CONFIG_DERIVED_PENDING_EFFECTIVE_CONFIG` |
| `targets` | 2 | `RAW_CONFIG_DERIVED_PENDING_EFFECTIVE_CONFIG` |
| `CAUSAL` (v1 DAG) | 52 | `PRODUCER_ASSIGNMENTS_LOCATED_PENDING_TRANSITIVE_RESOLUTION` |

`UNRESOLVED_PRODUCER` (45) **stands**: it claimed exactly what it
measured.

## 5. C54 — three levels, never mixed

* `OBSERVED_EXECUTION_DEMAND` — **9 runs** that reached a terminal
  state consumed **248 subjects**, read from the cube's own campaign
  facts.
* `VALIDATED_RUNNABLE_CONFIGS` — **20 validated runnable, 204 refused**
  with reasons. The effective configuration is built exactly as
  `app/main.py` does, every entry point resolved, the target validated,
  in a **subprocess** whose caller asserts from OUTSIDE that no
  framework was imported and no file was created.
* `INPUT_FILES_PRESENT_CONFIG_CANDIDATES` — the old 137/87 census under
  its honest name.

No level is promoted to the one above it.

**A finding the pre-push secret gate surfaced.** The cube stores each
run's identity as its producer wrote it, and **four of the nine
terminal runs are identified by a FILESYSTEM PATH** —
`./prediction.csv`, `examples/results/…`. The public record now carries
logical ids bound to a digest of the original, with each original's
**shape** declared as `PATH_SHAPED_IDENTITY` or `OPAQUE_IDENTITY`,
because "this run is identified by a path" is a fact about the cube and
not something to hide. The defect belongs to how those producers name
their runs; it is **reported, not corrected**, since rewriting a
historical identity would rewrite history. The gate itself fired on a
campaign key it read as a high-entropy token — that one was a false
positive; the paths beside it were not.

## 6. C55–C58 — a transitive lineage, or `UNRESOLVED`

252 producer files, 2,383 symbols. Locals and intermediates are
resolved inside the defining symbol and helpers are followed by bare
name; a centred window, a negative shift or roll, a `bfill` or
positional indexing **anywhere** on a path makes the output
`NON_CAUSAL`; one unfollowed call, one ambiguous helper or one cycle
leaves the whole output `UNRESOLVED`.

**37 `CAUSAL_ACTIVE`, 5 `HISTORICAL_OR_RETIRED_PRODUCER`,
55 `UNRESOLVED`.**

The acceptance you named is met:

| column | v1 | v2 |
|---|---|---|
| `log_return_1` | CAUSAL, lb 1, no inputs | `CAUSAL_ACTIVE`, lb 1, `close` |
| `macd` | CAUSAL, lb 1, no inputs | `UNRESOLVED` — its window arrives as a parameter |
| `stoch_k` | CAUSAL, lb 1, no inputs | `CAUSAL_ACTIVE`, lb 14, `high/low/close` |
| `cci_14` | CAUSAL, lb 1, no inputs | `CAUSAL_ACTIVE`, lb 14, `high/low/close` |
| `mfi_14` | CAUSAL, lb 1, no inputs | `CAUSAL_ACTIVE`, lb 14, `high/low/close/volume` |

**Not one column** is left causal with lookback 1 and empty inputs.

**C56** — availability is never `event_time + one bar` by default. The
dataset must declare what its timestamp MEANS (open, close or
publication) and its provider latency. **No dataset declares either**,
so every column's earliest availability is `UNAVAILABLE` with the
reason named. 31 fixtures, each written so the v1 reader fails it.

## 7. C59–C61 — verified from the outside, superseded additively

The verifier imports **nothing** from `characterize_lake` — not its
census reader, its normalization, its digest helper or its path
builder. A verifier that borrows the producer's code verifies only that
the code agrees with itself.

| | |
|---|---|
| declared / read | 1,965 / 1,965 |
| missing / extra / unreadable / duplicates | 0 / 0 / 0 / 0 |
| outcomes | 1,505 `MEASURED`, 460 `NOT_IDENTIFIABLE` |
| source files re-digested | 420, all present |
| naming rules consistent with every file | exactly one: `sha256[:32]` |
| custody | 2 retained descriptors, 1,966 reads |
| verdict | **`TERMINALS_VERIFIED_EXACT`** |

The v1 terminal carried an outcome and a count. The v2 terminal adds
the **schema**, the **source digest** re-computed by the independent
verifier, the **window contract and its digest**, and a **digest of
itself**. 1,965 written **beside** v1; the run captures every v1 inode,
size and mtime before and after and refuses outright if one moves.
1,965 unchanged.

## 8. C62–C63 — exact membership, and an honest current view

The cube recorded *which run* produced a row and nothing about *which
batch* a variable belonged to. "Which variables were in `batch_00007`,
exactly?" had no answer.

* `bridge_batch_variable` answers it with **rows**: 14 batches, 1,965
  variables, every batch `EXACT`.
* A membership digest over the sorted id list makes a second load a
  no-op and a **changed** membership a **refusal**, never a merge.
* `v_variable_characterization_current_v2` supersedes the v1 view
  without dropping it: same deterministic row, plus `currency_state`,
  where `AMBIGUOUS_TIE` means two observations share the newest
  `measured_at` and the tie was broken by digest order. Both views
  return 40,341 rows and no tie exists today, so the guard is proven on
  **throwaway databases** created and dropped by the tests.
* Outbox, read live: **0 pending, 16 of 16 dead letters adjudicated, 0
  unadjudicated, heartbeat 0.4 s old, healthy**. A dead letter is
  adjudicated or superseded, never deleted.

PostgreSQL and Metabase were never started, stopped or restarted; the
populated cube was never truncated.

## 9. C64–C65 — the design stops calling itself sealed

v1 read `SEALED_DESIGN_NO_SCORES_COMPUTED`. It was not sealed and could
not have been: a seal asserts that an **external** party fixed the
document before any score could influence it, and no external party had
seen it. Hashing my own design establishes only that I have not changed
it. The status is now **`DRAFT_CANDIDATE_NO_SCORES_COMPUTED`**, and v1
is **not rewritten** — a label that overclaimed is itself evidence.

The machine-readable companion binds each of five eligibility
conditions to a named artifact. Its eligible set is **EMPTY**: 37
columns reach `CAUSAL_ACTIVE`, and condition E5 — availability is not
`UNAVAILABLE` — removes every one, because no dataset declares its
timestamp semantics. The condition is **enforced, not dropped** to
manufacture a population. That is why this is a draft candidate and not
a runnable screen.

## 10. P2 — E2 Alpaca, offline only

27 tests classify the Alpaca wrapper's noise **without contacting the
venue**: no socket, no credential, no order. Transport failures wrapped
in `AlpacaPaperError` via an explicit `from exc` classify transient and
the cause is **named**; an invalid-JSON `ValueError` under the *same*
wrapper stays fatal; a refusal whose text says "could not connect
account: ConnectionError in config" stays fatal, because message
sniffing is what this taxonomy refuses to do; an implicit `__context__`
is never followed. The venue-dependent half — noise measured against
live Alpaca data — is **`DEFERRED`**: it cannot be done without
contacting the venue.

## 11. Evidence

* **PRE**: 17 items, frozen at `05000ab`-equivalent base tips **before
  any correction**, 17/17 reproduced, exit 0.
* **POST**: 17/17 **CORRECTED**, exit 0.
* Preserved roots `b4_v7` and `t2_successor`: mode `0o700`, never
  opened for writing. The 1,965 original terminals: untouched.
* Suites — predictor **393 passed**, 3 failed, 8 legacy collection
  errors; the 3 failures fail **identically** at the pre-cycle tip
  `16fbbbf`, verified in a throwaway worktree, so they are
  pre-existing. financial-data **504 passed**, 1 collection error
  (`yaml` absent from the environment, pre-existing). agent-multi B4
  **61**, T2 **30**. lts **27**.

## 12. Still yours to decide

1. **R12's checkout conflict** — re-pin the execution record to the
   audit snapshot commit, or declare digest equality sufficient.
2. **Dataset timestamp semantics and provider latency.** Until a
   dataset declares what its timestamp means, every availability is
   `UNAVAILABLE` and the C65 eligible set is empty by rule.
3. **External review of `FEATURE_DAG.v2`**, on which eligibility
   depends.
4. **The development licence** for the per-variable design. It does not
   carry one and does not ask the owner for one.

The owner's only pending operational action remains the host reboot to
realign the NVIDIA driver; `uptime -s` still reads 2026-09-11 22:53:03.
