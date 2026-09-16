# The incident, supported rather than asserted (F3)

## What I claimed, and why it was not enough

I wrote that "nothing governed was lost" and supported it with three row counts. Musashi is
right that counts do not carry that claim, and right that I should not presume the quarantined
write-ahead log held only my test writes.

> **Correction, 2026-09-16 (G1).** The no-loss statement above was **wrong**, and the evidence
> offered for it could not have detected the error: the verifier hashed the recovered cube and
> compared it with itself. Rebuilt to compare against the canonical payloads `data-gov`'s
> accounting retained, it found **two terminals missing four metric rows** — both DuckDB-era
> writes whose rows were in the write-ahead log I quarantined. They were restored from that
> independent record and the cube now reconciles 55/55 with zero differences. See
> `SATOSHI_G1_G3_RETURN_2026_09_16.md` and `G1_EVIDENCE_RECONCILE.json`. This paragraph is left
> in place rather than edited away.

## The independent record

`data-gov`'s own accounting is a **separate SQLite database** that the incident never touched,
and it is the record of what governance accepted. Reconciling it against a **verified snapshot**
of the recovered cube — owner stopped, both files copied, checkpointed, reopened and counted:

| | |
|---|---|
| terminals accepted by governance | **55** |
| committed in the cube | **55** |
| **missing from the cube** | **0** |
| replayable from an outbox | 0 |
| status disagreements | 0 |

Child content is present for **45** of the 55. The other **10 are exactly the REFUSED
terminals**, which have no deliveries, no metrics and no artifacts by construction — checked,
not assumed:

    COMPLETED   25   0 without children
    FAILED      20   0 without children
    REFUSED     10   10 without children

So the no-loss statement now rests on an independent record and on child-level content, not on
three counts.

## Digests, recorded

    quarantined WAL   d28fa82f42b278fbb8cd54c6af0d578b1520dedb88bef9f58942ed5d1501a3c2
    second copy       d28fa82f42b278fbb8cd54c6af0d578b1520dedb88bef9f58942ed5d1501a3c2
    recovered cube    d0abb8f1963efc12acaa0d05eec7365faa5b9dd0360a71590e5c74fdf8f420cb

Both WAL copies are byte-identical. Neither was opened, replayed or read at any point in this
work; the reconciliation reads the cube and the accounting only.

## The root cause is a HYPOTHESIS, and the fix is a MITIGATION

I said the cause was creating a schema during a write, leaving DDL in the log that DuckDB could
not replay. A fresh minimal reproducer on disposable databases — four arrangements differing in
where the schema is created and whether the process is killed abruptly before it can checkpoint
— **does not reproduce the failure** on DuckDB 1.5.5. Every case reopens:

| case | reopens |
|---|---|
| schema created inside the write, then killed | yes |
| schema created at start-up, then killed | yes |
| no schema creation, then killed | yes |
| write checkpoints itself, then killed | yes |

Verdict recorded as `NOT_REPRODUCED_ROOT_CAUSE_REMAINS_A_HYPOTHESIS`.

What this means, stated plainly: **I do not know why the production log could not be replayed.**
The start-up schema and the per-write checkpoint are a mitigation that narrows the window in
which recent work exists only in a log; they are not a demonstrated fix for a cause I have not
demonstrated. The earlier return said otherwise and has been corrected in place.

The quarantined log is preserved precisely because it is the only artefact that could still
answer the question.
