# QRM01 — the cell-scope instrument: one cell, one fresh exclusive scope, one reservation

Satoshi, successor technical lead. 2026-09-29.

Lane A of four. This lane owns the shared runner; the other three pin the revision named at the
end of this document.

**No new forecast accuracy was measured.** Nothing was trained, scored or re-scored, no cell of
any block was run, and no model error, naive comparison or literature comparison is reported —
there is no scored task here. What follows is a measured *resource* and *mechanism* result.

---

## 1. The three answers the order asks for first

| Question | Answer | Where it is proved |
|---|---|---|
| Is a reused scope now rejected? | **YES, by name, in two places.** A cell child whose cgroup is the driver's own cgroup refuses `REUSED_DRIVER_SCOPE`; a second cell claiming a scope another cell already holds refuses `SCOPE_ALREADY_CLAIMED_BY_ANOTHER_CELL`. A cgroup that is not a scope refuses `NOT_A_SCOPE`; a scope with `memory.max = max` refuses `NO_KERNEL_LIMIT`; a child started without the external supervisor refuses `CELL_NOT_LAUNCHED_BY_THE_SUPERVISOR`. A refused cell measures nothing and leaves no claim. | `tests/test_df_cell_scope.py` — five refusal tests, each asserting the code by name |
| Are two concurrent children correctly isolated? | **YES, and measured against the deployed launcher and the real kernel, not only in a fixture.** Two children launched concurrently through `crispdm-run` took two distinct scope cgroups, two distinct scope **inodes** and two distinct admission **lease ids**, each with its own `memory.max`. The two cgroup peaks came out **72 626 176 B** and **198 443 008 B** against known fixture allocations of 62 914 560 B and 188 743 680 B — two different charges, which a shared driver scope cannot produce, because in a shared scope both children read the same charge. | `docs/audits/evidence/QRM01_CELL_SCOPE_20260929/ISOLATION_EVIDENCE.json` |
| What do the other lanes pin? | The tip of `satoshi/qrm01-cell-scope-instrument-20260929` named in §7. | §7 |

---

## 2. The four corrections carried in, and what each changed in the code

1. **7.4 G is a killed multi-child wrapper scope's cgroup peak — not RSS and not a cell peak; 8 458 399 744 B is child RSS.** Two different quantities. The instrument therefore records them as two labelled fields — `host_ram.cgroup_peak` and `host_ram.process_rss_peak` — with a `comparability` field stating `NOT_INTERCHANGEABLE` on every record, and it refuses to offer a resident set to admission as a footprint (`peak_evidence()` emits `peak_scope: "cgroup"` only for a measured cgroup peak).
2. **1 463 877 632 B is one 20 000-window data materialization, not a lower bound and not a training cap.** Every record carries a mandatory `stage`, and the stage is carried into the peak evidence, so a `DATA_MATERIALIZATION` floor cannot be read as a `CELL_TRAIN_AND_SCORE` cap. `cell_cap_bytes()` **refuses** rather than defaulting: no retained Q2 figure can become an allocation by omission. QRM01 declares no cap.
3. **RSS and cgroup accounting measure different quantities; one observed ordering is not an invariant.** The instrument reads cgroup **identity, ancestry and lifetime** — the relative cgroup path, its parent, whether it is a `.scope`, its directory **inode** (a scope *name* repeats; a fresh scope's inode does not), its boot id and its kernel limit — and never infers freshness from a memory reading. Two further observations from this lane's own bounded fixture: RSS exceeded the cgroup peak by 7 536 640 B and by 7 700 480 B. That is consistent with the earlier pilot's 9 158 656 B, and it is reported as **three observations, not an invariant**; neither quantity bounds the other in either direction.
4. **Reproducible reachability from the selected runner commit, not presence on master.** The instrument at `84bcd605` and the v2 seal are integrated into the published lineage with explicit commit provenance, and the seal is explicitly **not** used to reinterpret old cells — see §3.

---

## 3. The integration, and what it deliberately does not do

Branched from **`6de538e2`** (`satoshi/q2-household-governed-20260929`), the Q2 return tip: the
actual published lineage of the governed household chain, and the lineage whose
`tools/crispdm-run` and `tools/crispdm_admission.py` are byte-identical to the deployed copies.

- **The instrument** came from `84bcd605`'s own diff against its parent, applied as a patch to
  `tools/df_e1_block.py`. Not a branch merge: that branch adds 981 files and 412 046 lines of
  unrelated lineage, which is not a reviewable runner change. What it adds:
  `cost.cgroup_memory_peak_bytes` with its basis labelled, `cost.peak_rss_bytes_basis =
  MAIN_PROCESS_RSS_ONLY` stated on the record, and `host_identity()` — an opaque salted per-host
  id in place of `os.uname().nodename`, which the lineage at `6de538e2` was still writing into
  every cell record.
- **The v2 seal** (`27ed712d…`) came across **byte-exact** in five files — `DESIGN.json`,
  `EXTENSION_SEAL.json`, `SEAL_WITHDRAWAL.json` (which records v1 `7d0bf921…` WITHDRAWN),
  `FOOTPRINT_BASIS.json`, `REDACTIONS.json` — each verified equal to its `84bcd605` blob.
- **The thirteen result files of the twelve historical cells were deliberately NOT carried.**
  Carrying `CLOSURE_TABLE`, `REPORT`, `TABLES`, `UNANCHORED_MEASUREMENT_TABLE`, `UNGOVERNED_RUN`,
  `MEMORY_GATE` and the rest into the branch that carries v2 is exactly the move the order
  forbids: it would let old cells be read under a seal made after their scores existed.
- **Stated on the face, not implied:** the twelve landed fits remain `NOT_BOUND_TO_A_SEAL`,
  `HISTORICAL_DEV_ONLY`, custody `UNCHECKED`, zero verified rows, closure `FAILED`, and this
  integration changes **nothing** about them. The six missing W1440 cells are **STILL MISSING**;
  none was relaunched. v2 remains `SEALED_NOT_EXECUTED`; only its reachability changed.

All of this is machine-readable in
`docs/audits/evidence/QRM01_CELL_SCOPE_20260929/PROVENANCE.json`.

---

## 4. What was built

`tools/df_cell_scope.py` (new) and the launch path of `tools/df_e1_block.py`.

**The launch.** `run_units` used a bare `subprocess.run([sys.executable, __file__, "child", …])`,
three at a time. A cell child therefore took no scope, inherited whatever cgroup the driver sat
in, and shared it with its siblings — so a `memory.peak` read there is the charge of a shared
driver scope over a *batch*, and no Q2 cell ever had a per-cell figure. It now launches through
the **existing** launcher and its atomic admission module — `crispdm-run`, never a second
scheduler — which gives each cell a fresh transient scope inside `crispdm-batch.slice` enclosing
its complete process tree, that scope's own `MemoryMax`, and **its own reservation** held until
the whole tree has finished. Admission is taken **fresh per child** against every other live
reservation on the host. A refusal is terminal: the cap is asked for once, at the declared size,
and never re-asked smaller — the launcher itself makes a lowered cap terminal, and the supervisor
does not attempt it (asserted, by reading back what the launcher was actually asked for).

**Recorded per cell, separately** (`df_cell_scope_record.v1`, plus a `CELL_SCOPE.jsonl` row per
cell at the block root): host RAM **cgroup** peak · GPU allocated and reserved · scope identity
(cgroup, unit, inode, parent, boot id) · the kernel limit (`memory.max`) · optimizer updates ·
CPU time · wall time · stage. GPU is never added to or compared with host RAM; a cell that
selected no device reports **absence with its reason**, not zero.

**The peak is read before the scope is removed**, from *inside* the scope by the child —
`memory.peak` is a kernel high-watermark, so it is a measurement even for a child shorter than
any sampler's interval and it survives a release that never happens. The **external** supervisor
also reads it while the scope still exists, and independently identifies the scope rather than
trusting the child: a disagreement between the two sets `usable_for_costing: false`.

**A failed child.** The supervisor is outside the child's scope and depends on the child writing
nothing. It retains `COMPLETED` / `FAILED` (with the signal, from both `128+N` and a negative
return code) / `WALL_TIMEOUT` / `REFUSED_BY_ADMISSION`, durably and atomically, and distinguishes
a launcher refusal — where **nothing was started** and no limit or kernel setting was changed —
from a cell that ran and failed.

**A missing peak is `UNKNOWN`.** Never `0`, never a success, and never substituted from the
resident set. `usable_for_costing` is false, the reason is on the record, and the admission module
refuses such evidence outright rather than sizing a cap on a silence.

**The deployed runner's identity is verified**, not only the tracked bytes: both halves of the
runner are read from the paths that will actually start a cell and digested.
`RUNNER_IDENTITY.json` — verdict **`DEPLOYED_MATCHES_TRACKED`**; launcher and admission module
both sha256-equal to `tools/crispdm-run` and `tools/crispdm_admission.py` at this integration
commit.

### Two real defects this lane found in its own first implementation

Both were found by the tests, and both are now regression-tested:

1. `admission_module()` executed the admission module without publishing it in `sys.modules`.
   `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so the first dataclass
   raised `AttributeError`.
2. Worse: `run_units` supervises a batch on threads, and two threads loading that module at once
   raced — one published it and began executing its body while the other read a half-built module.

The visible symptom of both was **`lease_id: null` on every cell of a parallel batch whose
reservations the launcher had genuinely taken** — a real reservation reported as absent. The load
is now serialised, and an unconfirmed lease carries its reason, because *could not confirm* and
*there is none* are different statements.

---

## 5. The report shape

```
QRM01 — the cell-scope instrument: one cell, one fresh exclusive scope, one reservation
repo/branch/tip: predictor / satoshi/qrm01-cell-scope-instrument-20260929 / see §7
worktree: /home/harveybc/Documents/GitHub/.worktrees/predictor-qrm01-20260929
branched from: 6de538e2 (Q2 return tip) · instrument carried from 84bcd605 · v2 seal 27ed712d byte-exact
files: tools/df_cell_scope.py (new) · tools/df_e1_block.py (launch path + the 84bcd605 instrument)
       tests/test_df_cell_scope.py (new, 27 tests, committed RED before the implementation)
       docs/audits/evidence/QRM01_CELL_SCOPE_20260929/{PROVENANCE,RUNNER_IDENTITY,ISOLATION_EVIDENCE}.json
       docs/audits/evidence/E1_Q2_CONTEXT_DEEP_20260926/{DESIGN,EXTENSION_SEAL,SEAL_WITHDRAWAL,FOOTPRINT_BASIS,REDACTIONS}.json
suites: test_df_cell_scope 27/0 · test_df_e1_block + test_df_closure_table + test_crispdm_admission
        + test_df_admission_guard 108/0 (167 s, unchanged by this integration)
acceptance: reused scope REJECTED by name (5 refusal codes) · two concurrent children under the
        DEPLOYED launcher -> distinct cgroups TRUE, distinct inodes TRUE, distinct leases TRUE,
        supervisor agreed with each child TRUE, own kernel limit each 402 653 184 B, peaks
        72 626 176 B vs 198 443 008 B · RUNNER_IDENTITY verdict DEPLOYED_MATCHES_TRACKED
budget: bounded fixtures only. Every run launched under crispdm-run with a declared cap: tests
        2-6 GiB, isolation fixture 1 GiB driver + 384 MiB per child, longest run 167 s. Total
        elapsed measured compute this lane: under 6 minutes. No GPU. No displacement.
what is NOT done / refused / not measured:
  - NO cap is declared. cell_cap_bytes() REFUSES without an explicit declaration; QRM02 declares it.
  - NO accuracy, NO fit, NO score, NO cell of any block run. NO closure table: no measurement.
  - The twelve historical cells are NOT re-bound, NOT re-read and NOT re-costed under v2.
  - The six missing W1440 cells are STILL MISSING; none relaunched.
  - NO training-path footprint exists yet: no model, gradients or optimizer slots were built here.
  - The preferred RTX 5090 host was NOT used and its host-memory restriction is NOT lifted.
  - No host rebooted, no service started or stopped, no broker touched, no worktree removed.
```

---

## 6. Resources actually used, and what was refused

Everything ran on the coordinator, under `crispdm-run`, with a declared cap each time — the
hook that forbids uncapped heavy compute was never bypassed. The heaviest single run was the
existing 108-test suite at a 6 GiB declared cap for 167 s; the isolation fixture was a 1 GiB
driver with two 384 MiB children for 4.1 s. **No GPU was used.** The **external RTX 5090 host was
not used and its reported host-memory restriction is not lifted by this lane.** Nothing was
displaced: every admission was taken fresh through the shared atomic gate, and no reservation was
held while waiting for anything.

**Idle time: none.** No eligible unit of this lane was blocked at any point.

---

## 7. What the other lanes pin, and what is unblocked

- **Pin:** `satoshi/qrm01-cell-scope-instrument-20260929`, the tip recorded in the commit that
  carries this document. Lanes B, C and D take the runner from that revision; this lane owns it.
- **Unblocked: QRM02.** The mechanism a per-cell training footprint needs now exists and is
  measured. QRM02 still has to *declare* its stage budgets, host and device caps and remaining
  allocation **before** dispatch, and `cell_cap_bytes()` will refuse it otherwise. It must not
  reduce a cap to pass admission, and it must not take any figure in this document as a training
  cap: **no training-path footprint has been measured anywhere yet**, because no model, gradient
  or optimizer slot has been built in any instrumented cell.
- **Still open, unchanged by this lane:** the six missing W1440 cells, the twelve cells' custody,
  the matched-budget estimand and the 600-update contract — all of which QRM03 must cost.

Signed, Satoshi, successor technical lead, 2026-09-29.
