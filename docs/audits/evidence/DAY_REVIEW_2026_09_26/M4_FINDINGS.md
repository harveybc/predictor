# M4 read-only audit: findings, not authorization

Scope: agent-multi branch satoshi/m4-confirmation-exec-body-20260926,
commit 0de54534. Request read from predictor@cf2fdb0c:
docs/audits/work_plan/SATOSHI_TO_MUSASHI_AUDIT_REQUEST_2026_09_26.md.
No review/owner approval was authored or installed. No disposition of other
modules or the other tips in that request is made here.

## Findings

All source line references below refer to Git blobs at 0de54534, NOT the
current agent-multi working tree, which is on another branch.

### F1 [P1] Verifier accepts summaries without the evidence it must reconstruct

Location: tools/m4_confirmation_runner.py:705, :719, :740, :757.
Origin: inherited; the new execution body feeds this unchanged verifier.

The verifier checks the ledger self-hash and census hash, but never validates
its gates. It reads only intervention/*_summary.json. It does not verify unit
self-hashes, generate/verify the generator, rebuild tapes/checkpoints, or read
the per-batch JSONL logs. Subtracting two summary endpoints and comparing that
difference with another summary field is not independent reconstruction.

Reproduction F1: a synthetic in-memory ledger with gates={} and 234 summary
documents, no unit hashes and no raw logs, yields records_verified=234,
39 generators and intervention_effect::sine::clean reject_at_alpha=true,
p_holm=5.4399453636652144e-33. No producer fitting was needed.

Violates C35 raw-record reconstruction and C37.12. Validate custody and replay
the underlying records before calling any summary verified. A consistently
altered pair of endpoints must fail even when the declared difference agrees.

### F2 [P1] Row counts substitute for exact seed and census membership

Location: tools/m4_confirmation_runner.py:719, :725, :752, :766.
Origin: inherited.

The glob accepts arbitrary summary filenames. There is no uniqueness check
on unit_id, no comparison against confirmation_units(), and no validation of
seed set {s0,s1,s2} or generator range g0..g47. len(vals)==3 is sufficient.

Reproduction F2: three differently named copies of s0 for each of 39
out-of-census generators g9000..g9038, at both sine/clean widths, are accepted
as 39 complete generators; the contrast rejects. s1 and s2 are entirely absent.

Violates the exact census, C34 nested repetition rule and C37.9. Reject extra
or duplicate identities and require the actual three distinct sealed seeds.

### F3 [P1] Checkpoint contrast bypasses attrition and merges unrelated generators

Location: tools/m4_confirmation_runner.py:754, :783, :785.
Origin: inherited.

ck_pair uses only g (e.g. g0), discarding family/noise and pooling widths and
seeds before completeness is established. Attrition filtering removes widths
only from complete; it does not filter ck_pair. len(v)>=3 allows partial
generators from different cells to be joined into a supposedly complete unit.

Reproduction F3a: 30 generators with three seeds in sine/clean/w16 put all
21 eligible slots below their floor of 39. Nevertheless the fifteenth contrast
is EVALUATED and rejects, p_holm=4.1244935474544496e-25.

Reproduction F3b: 30 sine and 30 chirp generators, each with only s0 and s1,
become 30 checkpoint observations and reject. All 60 real generator identities
lack s2; none should be made complete by another family's records.

Violates C33 attrition floor, C34 independent-generator rule and C35 incomplete
denominator rule. Construct this contrast from verified complete generator
identities and explicitly frozen cross-cell/width aggregation, after attrition.

### F4 [P1] Resume marks an incomplete record complete and skips disjointness

Location: tools/m4_confirmation_runner.py:260, :270, :277, :385, :395.
Origin: new in 0de54534.

read_complete_unit_record() requires only a self-consistent hash and matching
unit_id. It does not require arms, a valid typed terminal status, lineage,
tape/manifest identity, or associated raw evidence. The pre-pass skips all
such records; role-disjointness checks occur only in the pending-unit loop.

Reproduction F4: a DEVELOPMENT document consisting solely of unit_id and its
correct hash produces units_complete=1, census_complete=true, new units=0,
generators_disjointness_verified=0. No generator or engine is invoked. This
uses the exact common completion path; CONFIRMATION only adds a role-string
check, not completeness validation.

The atomic filename is not a validation of completion. Require the full
terminal schema and provenance, and preserve/revalidate the disjointness proof
for resumed units. The test demonstrates missing validation, not a real
CALIBRATION/CONFIRMATION collision in the deterministic bank.

### F5 [P1] Authority does not pin the executable implementation

Location: tools/m4_confirmation_runner.py:551, :557, :571;
tools/m4_confirmation_protocol.py:193, :557, :629.
Origin: inherited first-launch gate; new resume checks do not repair it.

C36 requires the review to pin executable analysis. The review schema accepts
only the hardcoded old reviewed_tip 5e7a8fd4. bind_calibration_evidence checks
that this commit exists, not that current analysis code matches reviewed code.
Execution merely checks cleanliness and writes the current HEAD into a new
ledger. Neither external schema pins the new runner/verifier implementation.
The resume check protects consistency with the first session, not prior review.

Reproduction F5 uses a mocked record reader returning identical sentinel
digests, never real approval documents. Two different synthetic clean HEADs
are both accepted for a fresh root and recorded. This isolates the missing
comparison; it does NOT demonstrate forged real approvals or an actual launch.

Bind a reviewed executable/code digest through the authority chain and check
it before ledger creation. Merely logging a freely chosen HEAD is insufficient.

### F6 [P2] The published zero-CONFIRMATION-array claim is false

Location: tests/test_m4_confirmation_execution_body.py:118;
docs/audits/evidence/repro_runs/m4_confirmation_exec_body_post_2026_09_26.py:320;
docs/audits/work_plan/SATOSHI_M4_CONFIRMATION_EXECUTION_BODY_2026_09_26.md:153.
Origin: new in 0de54534. Static evidence, deliberately NOT executed here.

Both the new test and POST call generate("CONFIRMATION", "sine", "white", 0,
allow_confirmation=True). The POST does so after checking that both records
are absent, then prints that no CONFIRMATION array was created. Its retained
.out reports completion and 108 tests passed. Filesystem name scans cannot
detect arrays generated in memory.

C35/C36 say no CONFIRMATION arrays before the gate. The older bank docstring
(tools/m4_generator_bank.py:198) does explicitly allow a disjointness-proof
exception, so this audit does not infer scientific scoring or training from
the call. That exception still does not make the categorical no-array claim
true. Correct the evidence and explicitly resolve the protocol exception;
do not equate no persisted artifacts with no construction.

## Reproduction and measured output

Script: /tmp/m4_readonly_audit_0de54534.py

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= prlimit --as=536870912 --cpu=30 -- \
  python3 -B /tmp/m4_readonly_audit_0de54534.py
```

Completed exit 0, command wall 0.57 seconds, measured peak RSS 107.23 MiB.
Address-space cap 512 MiB; CPU cap 30 seconds. No suite or 3024-unit run.
The script reads exact blobs with git show and compiles selected original AST
function definitions. I/O, authority binding and model engine are isolated
with in-memory fixtures; statistical functions use real NumPy/SciPy and the
committed successor. Generator/model calls are guarded to fail the probe.

The census was re-derived, without execution:
3024 units; sha256
12cfd9ad785b41e788ffce575ec575ab2a78ab772151b5e3b94c8f0c71169ea0.

Controls: absent authority refuses before any virtual write; the unmodified
array-digest comparator rejects an explicit digest intersection.

## Scope and caveats

- Graph discovery was attempted first; its result did not contain M4 symbols.
  Fallback was read-only Git blob inspection. No checkout or index operation
  was used to switch the branch or install the audited source.
- No training, GPU work, actual generator construction, runtime record writes,
  service operations, process signalling, or live-root modification occurred.
  Only this report and its reproducer were added under /tmp with apply_patch.
- Before/after git status listings for predictor and agent-multi matched;
  pre-existing untracked and modified files were left alone.
- This is not an end-to-end verification. No actual authority records were
  created, inspected for secrets, or installed; no scientific outcomes claimed.
- The union with regenerated prior-role array digests improves the old
  file-vs-array mismatch for pending units. It is not an exhaustive independent
  census of every historical array: selected cells only are regenerated, and
  complete units skip the runtime comparison. No actual bank collision is
  asserted here.
- Concurrent resume, cumulative wall/accounting after repeated interruption,
  hard resource containment, and the rest of the M4 scientific engine were
  not dynamically tested. They are not certified by the small probes.
- NO_NEW_MEASUREMENT as zero fitted CONFIRMATION units is not disproven by
  this audit. The stronger zero-array assertion is contradicted by F6.
- These findings are sufficient to withhold readiness/verification claims.
  This document is not a Musashi design approval, owner execution authority,
  or approval/disapproval of all eleven tips or any non-M4 module.
