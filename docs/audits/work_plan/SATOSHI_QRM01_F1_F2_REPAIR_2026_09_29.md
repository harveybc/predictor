# QRM01 F1/F2 — the gate now certifies what it claims, and producer and consumer speak one schema

Satoshi, successor technical lead. 2026-09-29. Lane A of the parallel orders in
`docs/audits/work_plan/MUSASHI_AUDIT_23B2EFA3_2026_09_29.md`.

I own the shared runner and both findings against it were real. I read the dictamen first, ran its
reproduction against the unmodified instrument, and only then repaired anything. F3 is lane B's and
I did not touch the PyTorch telemetry or QRM02.

---

## 1. The three counterexamples, by name

| Auditor's case | Before (c1033dc6) | Now | Refused by |
|---|---|---|---|
| `stale_foreign_record` — another cell, clock 1, no scope seen, no lease | `usable_for_costing=true` | **refused** | `ENVELOPE_MISSING`; with the envelope supplied so the fields are judged: `CELL_IDENTITY_MISMATCH`, `ATTEMPT_IDENTITY_MISSING`, `BOOT_IDENTITY_MISSING`, `NOT_A_SCOPE`, `CLOCK_OUTSIDE_THIS_ATTEMPT`, `KERNEL_LIMIT_MISSING`, `RESERVATION_MISSING`, `PEAK_BASIS_IS_NOT_THE_TREE_PEAK` |
| `negative_peak_record` — status `MEASURED`, bytes `-1` | `usable_for_costing=true` | **refused** | `PEAK_BYTES_NEGATIVE` |
| `production_nested_record` — the real producer shape | `UNKNOWN`, `usable_for_costing=false` **while containing the peak** | **accepted and costable** | — (read through the declared envelope) |

The third is the opposite kind of defect and is repaired in the opposite direction: it was a false
negative at a schema boundary, so the test asserts acceptance.

This is not narration. `tests/test_qrm01_auditor_counterexamples.py` loads the instrument **from
the git object at c1033dc6** and the instrument as it stands now, drives the auditor's own three
record shapes through `supervise` over his own process and observer doubles, and asserts both
sides in one run: old accepts / new refuses. If anyone loosens the gate again, that file fails.

## 2. F1 — the fresh-attempt contract

The old gate read any pre-existing JSON at `record_path` and took `MEASURED` plus exit 0 as a cost.
It is replaced by `verify_fresh_attempt()`, which asks whether the document describes **this attempt
of this cell**, completely and in domain. Every clause refuses by its own name and every clause
refuses on absence:

- **Attempt identity.** `supervise` now mints an opaque token *before the child exists*
  (`new_attempt_id()`, passed as `CRISPDM_CELL_SCOPE_ATTEMPT`) and the record must echo it. A file
  already lying on disk cannot carry a token created after it was written, so freshness stops being
  an assumption and becomes a comparison. `ATTEMPT_IDENTITY_MISSING` / `ATTEMPT_IDENTITY_MISMATCH`.
- **Full identity.** Cell, stage and boot must all be present and match:
  `CELL_IDENTITY_MISMATCH`, `STAGE_MISMATCH`, `BOOT_IDENTITY_MISMATCH`.
- **Scope with its inode.** `SCOPE_CGROUP_MISSING`, `SCOPE_INODE_MISSING`,
  `SCOPE_INODE_OUT_OF_DOMAIN`, `NOT_A_SCOPE`. A missing inode is now a refusal of its own; before,
  it merely skipped the comparison.
- **A clock that belongs to this attempt.** `recorded_at` must be a number inside
  `[started_at, finished_at]` ±2 s: `CLOCK_MISSING`, `CLOCK_OUT_OF_DOMAIN`,
  `CLOCK_OUTSIDE_THIS_ATTEMPT`.
- **The kernel limit in force.** `memory.max` must be MEASURED, a positive integer, and the
  declared cap within one page: `KERNEL_LIMIT_NOT_MEASURED`,
  `KERNEL_LIMIT_BELOW_THE_DECLARED_CAP`, `KERNEL_LIMIT_ABOVE_THE_DECLARED_CAP`. A smaller limit is
  refused, never quietly adopted.
- **Typed peak, checked domain.** `PEAK_BYTES_NEGATIVE`, `PEAK_BYTES_ZERO`,
  `PEAK_BYTES_OUT_OF_DOMAIN` (strings, floats and `bool` all rejected), `PEAK_BYTES_MISSING`,
  `PEAK_ABOVE_THE_KERNEL_LIMIT`, `PEAK_BASIS_IS_NOT_THE_TREE_PEAK`.
- **Evidence of a confirmed reservation.** The record must carry a reservation confirmed in the
  launcher's own admission store, and the supervisor then **re-verifies the named lease against
  that store itself** — the store's copy must bind that lease to this cgroup and this cap:
  `RESERVATION_MISSING`, `RESERVATION_NOT_CONFIRMED`, `RESERVATION_BOUND_TO_ANOTHER_SCOPE`,
  `RESERVATION_CAP_IS_NOT_THE_DECLARED_CAP`, `RESERVATION_NOT_VERIFIABLE_IN_THE_STORE`.

**A path that also works for short-lived children**, which is where the old gate fell open. Nothing
above needs the supervisor to have seen the scope. A child shorter than one observation interval
still confirms its own reservation from inside its scope while it is alive, and after it ends the
store keeps the lease **body** (RR02 `retained/`), so the supervisor can verify the binding when the
scope is already gone. `test_a_short_child_whose_scope_is_already_gone_is_still_costable_through_the_retained_lease`
deletes the scope directory and the attempt is still certified — on evidence written by the
launcher, not on the child's word.

**Absence is not coincidence**, carried into the code as `ABSENCE_IS_NOT_COINCIDENCE`, stamped on
every record and every verdict. Where the supervisor observed no scope, the verdict records
`scope_identity_agrees_with_the_supervisor: null` with the reason and says in the record itself that
this is *not* read as agreement. When an observation does exist it must agree:
`SCOPE_IS_NOT_THE_SCOPE_THE_SUPERVISOR_OBSERVED`.

## 3. F2 — one schema, declared by version

`df_e1_block.py` wrote the record nested under `cell_scope`; `supervise` read `host_ram` at the
root. Repaired explicitly, not by trying both places:

- the producer calls `CSC.embed_record(record, …)`, which writes `cell_scope_envelope`:
  `{schema: "df_cell_scope_envelope.v1", record_schema: "df_cell_scope_record.v2", record_at: "cell_scope"}`;
- the consumer calls `extract_record(document)` and reads **that declaration alone**;
- the record schema is now `df_cell_scope_record.v2`; `df_cell_scope_record.v1` is refused **by its
  version** (`RECORD_SCHEMA_SUPERSEDED`), so no reader has to decide which v1 fields to trust;
- a document with a root `host_ram` that looks right but declares nothing is refused
  (`ENVELOPE_MISSING`). **No field is accepted because it shares a name** — that is asserted as a
  test, since accepting one would replace a silent mismatch with another.

The producer also honours the stage the supervisor declared (`CRISPDM_CELL_SCOPE_STAGE`), so the two
sides agree about *which measurement this is*, not only about where it sits. Each `CELL_SCOPE.jsonl`
row now carries `attempt_id`, `fresh_attempt_accepted` and `refused_by`.

## 4. The descriptor mechanism you and I both had wrong

The kernel documents the reset of `memory.peak` as applying to reads **through the same open file
descriptor** (<https://docs.kernel.org/admin-guide/cgroup-v2.html>). A `write_text` followed by a
`read_text` opens two descriptors and establishes nothing; our earlier report asserted that
experiment and it was not the experiment performed.

- `stage_peak_one_descriptor()` does the read and the reset on **one `os.open`**, via `os.pread` /
  `os.pwrite`, and records the discipline in the result.
- `lifetime_peak()` keeps the lifetime watermark **separately**, is never reset, and is carried in
  every record as `host_ram.cgroup_lifetime_peak`.
- `test_a_write_text_then_read_text_does_not_establish_the_reset_experiment` counts the descriptors
  opened on `memory.peak` and asserts exactly one, with `O_RDWR`, and that the source performs no
  `write_text` on that file. It **pins the discipline, not the kernel**: a regular file cannot show
  descriptor-scoped reset semantics, and the test says so rather than pretending otherwise.

No production path resets a peak today. The helper exists so that when a per-stage figure is wanted
it is taken correctly, and the lifetime watermark survives either way.

## 5. The inference I have stopped using

Two different peaks do not by themselves prove two distinct scopes: one scope can show different
maxima at two instants. Nothing in this delivery argues from a difference in magnitude. The
distinctness assertions are now inode identity, cgroup membership, scope lifetime and lease
binding — see the rewritten
`test_two_concurrent_children_take_two_distinct_scopes_and_two_distinct_reservations`, which asserts
two inodes and **two leases**, one per cgroup.

**And my own real workload produced the counter-observation.** On the end-to-end run the
supervisor's sampled value came out **above** the child's in-scope read — 559,276,032 B against
557,481,984 B — the opposite of the 9.5× undercount the classification lane measured, because
`run_cell` reads its own `memory.peak` before the child has finished and the watermark keeps rising.
So the old text calling a sampled observation "a LOWER BOUND" was wrong as stated. It is a lower
bound on the scope's **lifetime** watermark and on nothing else; it bounds the in-scope read in
neither direction. `floor_against_in_scope_peak` now carries `direction` and says the comparison is
**unsigned**, and names the lifetime watermark as the figure that is at least as large as both.

## 6. Report

```
QRM01-A — F1 fresh-attempt contract + F2 producer/consumer schema
repo/branch/tip: predictor · satoshi/qrm01-f1-f2-repair-20260929 · off c1033dc6
worktree: .worktrees/predictor-qrm01-f1f2-20260929
files:
  tools/df_cell_scope.py            fresh-attempt contract, declared envelope, record v2,
                                    reservation confirmation + independent re-verification,
                                    one-descriptor stage peak, separate lifetime watermark
  tools/df_e1_block.py              producer embeds through the envelope; honours the declared
                                    stage; CELL_SCOPE row carries attempt + verdict + refusals
  tests/test_qrm01_auditor_counterexamples.py     the auditor's probe frozen: old accepts / new refuses
  tests/test_df_cell_scope_fresh_attempt.py       30 rules: identity, freshness, clock, limit,
                                                  reservation, typed domains, descriptor, short child
  tests/test_qrm01_producer_to_supervisor_e2e.py  11 rules on a REAL workload
  tests/test_df_cell_scope.py                     4 tests updated to SUPPLY what the contract demands
  docs/audits/qrm_scope_probe_20260929.py         the auditor's reproduction, committed as received
  docs/audits/evidence/QRM01_F1_F2_20260929/      the real producer document, the real supervision
                                                  record (paths under /home redacted) and a summary
suites: test_df_cell_scope 29 passed + 2 real_launcher passed · fresh_attempt 30 passed ·
        auditor_counterexamples 7 passed · producer_to_supervisor_e2e 11 passed ·
        test_df_e1_block 20 passed (154 s).  Total 99 passed, 0 failed.
acceptance (the real workload, docs/audits/evidence/QRM01_F1_F2_20260929/SUMMARY.json):
  run_units' launch path -> deployed crispdm-run -> fresh transient scope in crispdm-batch.slice
  -> real df_e1_block.run_cell (6 optimizer updates, 13.0 CPU s) -> cell.json -> supervise
  producer document schema df_e1_block_cell.v1, root carries host_ram: false
  envelope df_cell_scope_envelope.v1 -> df_cell_scope_record.v2 at "cell_scope"
  accepted: true · refused_by: [] · usable_for_costing: true · attempt token echoed: true
  in-scope peak 557,481,984 B MEASURED · kernel limit 4,294,967,296 B = declared cap
  reservation confirmed LIVE_ADMISSION_STORE, cap agrees, re-verified by the supervisor: true
  scope crispdm-<cell>-<epoch>-<pid>.scope under crispdm-batch.slice, supervisor and child
  inodes agree · GPU UNKNOWN (absent, not zero) · cap asked once at the declared size
what is NOT done / refused / not measured:
  - F3 and QRM02 untouched: they are lane B's, against this corrected runner.
  - The e2e does NOT exercise the governed delivery check at the top of df_e1_block.child, nor
    governance terminal reporting; the panel is SYNTHETIC and the model tiny. It is a mechanism
    proof, not an experiment, and no model-quality claim follows from it.
  - No GPU, coordinator only, CUDA_VISIBLE_DEVICES empty throughout. The 5090 host was ineligible
    and was not used. No service, ceiling, reclaim or /dev/shm change; no cap shrunk, no
    reservation reduced, no shared memory cleared. Everything ran through the already-deployed
    crispdm-run (3 GiB test wrappers, one 4 GiB child) inside a slice whose 14 GiB ceiling I did
    not touch, against ~20-22 GiB MemAvailable. No OOM was provoked.
  - The descriptor test pins the CODE's discipline, not kernel semantics; a regular file cannot
    demonstrate descriptor-scoped reset.
  - Historical Q2 peaks are NOT retrospectively validated or invalidated by this repair. Every
    record written before it is v1 and is therefore refused by version: nothing already retained
    becomes costable, and nothing already retained is shown to have been false.
  - No cap, allocation or admission is requested or granted here. The 18 GiB request stays
    UNAPPROVED and QRM02's 4800 CPU s / 4800 s wall is lane B's to re-ask against this state.
```

## 7. What I would attack first if I were reviewing this

1. **The reservation is verified against the store, not against the kernel.** The lease body is
   written by the launcher, so it is third-party evidence — but a lease body is a file, and a
   process that can write the store can forge one. The contract is only as strong as the store's
   own integrity, which I did not harden here.
2. **The clock window is ±2 s on an unsynchronised wall clock.** A record written by a child whose
   clock jumped inside its own run would be refused for the wrong reason. I chose refusal over
   tolerance deliberately, but the tolerance is a number I picked, not one I measured.
3. **`PEAK_ABOVE_THE_KERNEL_LIMIT` assumes the kernel never charges a scope above `memory.max`.**
   With swap accounting or `memory.high` behaviour I have not tested, that assumption could refuse
   a genuine reading. It is a domain check I reasoned about rather than measured.
