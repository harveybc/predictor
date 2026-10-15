# R1: donor trust boundary (2026-10-15)

A prior campaign root may donate terminals to a new campaign only when ONE verifier accepts it.
That verifier is `tools.i6dmo.donor_trust.verify_donor`. Preparation
(`matched_runner._check_donor`) and actual adoption (`worker._adoptable`, through
`campaign.donor_problem`) both call it with the same target descriptor, so they give the same
verdict. `tests/i6dmo/test_r1_donor_trust.py` proves this with a spy on every case.

Any problem refuses the whole root. Refusing a root adopts nothing and fits nothing.

## What is enforced

The checks below are for a scientific matched target. Elapsed, market-time and engineering
targets check mode and fit procedure only.

1. **The donor's own declaration matches.** Each of these must hold, or the donor is refused:
   - its sealed effective config verifies;
   - its mode and its fit procedure are the same: entry points, fit device, engineering flag;
   - it is not engineering or CPU;
   - its sealed arm designs (seal and design digest), runtime pins, frozen population digest
     and retained input digest equal the target's.

   A self-described scientific root is not trusted on that basis alone. Everything below is
   also required.
2. **Every COMPLETE fit terminal binds the experiment.** For each one:
   - the stored terminal verifies: identity components, child digest and artifact bytes;
   - it has a `RECEIPT` artifact (`DONOR_FIT_RECEIPT_MISSING` otherwise);
   - the receipt is `COMPLETED`, `scientific: true`, on `gpu`, without overrides, and carries
     no `fake` or `fixture` marker;
   - its trainer identity equals the terminal's, and binds:
     - the expected population digest and the arm (in both the identity and the matched block);
     - the scientific design and effective-config digests;
     - the actual runtime (TensorFlow from the receipt, Keras from the saved archive's
       `metadata.json`), which must equal the donor's own sealed pin and the target's;
     - the capacity record: declared counts, a recomputed `capacity_sha256`, equal to the
       identity's;
     - the weights and archive: every receipt-listed artifact's bytes, the archive digest
       equal to the identity's, and `MODEL.json` weights equal to the identity's.

   A runtime that differs from the donor's own seal is `DONOR_RUNTIME_MISMATCH`, an
   incompatible donor. Its seal is never rewritten. The closed MG2/MG12 campaign (declared
   2.18, actual 2.21) can therefore never be a donor.
3. **Independently retained evidence. The root's own hashes are not enough.**
   - **SOURCE.** The queue backup records the unit's terminal as produced LOCALLY, by a
     COMPLETE attempt, with exactly this child-terminal digest. The backup itself must pass
     the custody check below. A terminal copied in by hand, or adopted from elsewhere, is
     `DONOR_SOURCE_TERMINAL_UNTRUSTED`.
   - **PUBLICATION.** Every report the fit published must be `DELIVERED`, with the exact
     readback digest, in the custody-verified outbox backup. The fit's cost report must bind
     the same unit identity and `weights_sha256`. A rewritten receipt whose weights differ
     from what was published is `DONOR_PUBLICATION_MISMATCH`.
   - **CUSTODY.** Every file used above must be byte-identical in the donor's two physical
     mirrors, the second on another host. That covers each fit terminal's files and the queue
     and outbox backups. A rewrite of the root alone is `DONOR_CUSTODY_MISMATCH`.
4. **Live warehouse confirmation (optional).**
   `donor_trust.warehouse_readback(root, units, client)` re-queries the warehouse (OLAP) for
   each fit report's digest and compares the exact rows. It needs the configured publisher's
   token, so it is not part of the automatic adoption gate. It was run for the receipt below.

## What it does NOT protect against

- **No authenticity from unkeyed hashes.** Every digest here is unkeyed. A party who can
  rewrite the donor root, both mirror copies (including the copy on the second host), and the
  queue and outbox backups coherently can construct a donor this verifier accepts. Custody
  raises the bar from "write access to one directory" to "write access to two hosts". It is
  not a signature.
- **The automatic gate does not consult the warehouse.** The outbox delivery record is checked
  through custody. A forger who could also POST to the warehouse would pass the live readback
  too.
- **No retraining check.** Weights are bound by archive bytes, `MODEL.json` and the published
  `weights_sha256`. They are not recomputed by reloading the model; the worker never imports
  TensorFlow.
- **SQLite backups are read as files.** They are opened with `immutable=1`. A backup in WAL
  mode whose WAL is not empty would be read without its WAL. Mirror passes write
  checkpointed backups, so this does not arise in practice.
- **Code identity is not re-proved.** Code-file digests in the receipt are not re-verified
  against a commit here. Code identity enters through the unit identity: `code_commit` is an
  identity component, so a different code pin is a different identity and is never adopted.
- **Absent or disagreeing evidence means refusal.** If the mirrors are unreachable, or the
  evidence is missing or disagrees, the donor is refused. Nothing repeats a costly fit
  automatically to repair documentation.

No PKI, signature scheme or ledger was introduced.

## The genuine donor, verified read-only

`R1_DONOR_READONLY_RECEIPT.json` records the check of the closed matched campaign
`i6dmo-6004b711281c` (CLOSED / MG2_SELECTED). Its `mg2m.fit` is the R0 reference the R2
contrast intends to reuse.

| Check | Result |
|---|---|
| Common verifier | ACCEPTED; `mg1.fit` and `mg2m.fit` both verified |
| Preparation interface | ACCEPTED |
| Custody | 30 files byte-identical on the local mirror and on the second host |
| Live warehouse readback | Exact for both fits' cost reports (`3c5414fc…` for mg2m, `3e267cec…` for mg1; 5 rows each) |
| Changes | No adoption, no fit, and the root's file tree was unchanged by this run |

One earlier run in the same session opened the SQLite backups in plain read-only mode. That
created two empty sidecar files in the closed root's `backup/` (`outbox.sqlite3-shm`, `-wal`)
and touched the existing `queue.sqlite3-shm`. The two new sidecars were removed. The verifier
now opens backups with `immutable=1`, which never writes.
