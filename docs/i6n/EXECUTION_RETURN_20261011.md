# I6-N integration return

Code: `68d1d6acb3f28a8537f1154b7f48fcc1e687ba07` on
`codex/i6n-integration-20261011`, published. The merge preserves reviewed C1
and C2 as ancestors. Scope: candidate engineering; no historical campaign.

## Measured engineering evidence

- Finite CPU verifier: 5/5 stages, 503 tests, no failures or skips, 42.989 s.
  Breakdown: C1/C2 284; business adapter 59; temporal operators 52; Keras
  factory 102; automation 6. Read-only status returns 0 against current code.
- Verification identity:
  `797e5acef838076646604e16e9086d78ea4420dc50c5e9ad7c60252999901cd8`.
  Private retained STATUS SHA-256:
  `6b9e46cd7a7f0a0114efef3d04de20bc12046ff5cb03f1d3761624af23ae4f22`.
- RTX 5090, TensorFlow 2.21: two synthetic models, six weight updates each,
  55 input positions, 20 feature branches, 12 outputs. Conv-only 18,168
  parameters; Transformer + reduced sequence 10,152. Both executed on GPU,
  weights changed, reload output difference exactly zero. 9.129 s probe body.
  No market data, no TEST, no production cost or predictive-quality claim.
- GPU result SHA-256:
  `45a2bbadc04edd5249b47db3de4d3b07dd2cc107c196e22eac87a7bfb15653e9`.
  Result and models copied back from the worker and model hashes verified.

CPU tests were written before their new implementations; initial collection
failed on absent modules. Final post-implementation evidence is the automated
sequence above. Existing model/audit history was not rewritten.

## Corrections during integration

The initial adapter requirement wrongly demanded sealing historical S support
before historical outcomes. It was removed: record the actual artifact seal
time and freeze support before candidate scoring, without pretending to have
created it years earlier. Pre-outcome U decision receipts remain mandatory.
The adapter verifies U operational eligibility and rescoring from supplied
forecast rows, but neither the code nor a local hash authenticates the source
rows against an independent custodian.

## Resources and operations

All three hosts were inspected once; the preferred 5090 had no existing compute
process and passed the bounded probe. Its sibling GPU was not used. The
secondary had only about 3.64 GiB admissible under the desktop reserve, below
the configured 4 GiB historical pilot. Kernel unreclaimable slab was 12.45 GiB;
its owner was not determined. No speculative cache flush or reboot was done.

Retired the secondary's completed `fs4-preliminary@cpu-raw`, `cpu-random`,
`gpu-4090` dispatchers, five completed weekly dispatchers and their weekly
activator, timers first. Weekly queue: 6240/6240 completed. Preliminary pinned
scope: 589 complete + 81 failed per arm, zero pending/claims; out-of-scope
pending rows were not cancelled. No live fit was stopped. These small
dispatchers were NOT blamed for the kernel slab.

Coordinator home had only about 9 GiB free (99% used). New local integration
and evidence are on the root-backed temporary filesystem, not RAM tmpfs.
Primary dirty/conflicted checkout was left untouched. Old campaign remains
paused: 220 complete, 46 failed, 3762 queued, no running; 266/266 published.

## Why historical fitting has not started

The plan audit did not accept a runnable I6-N campaign. Before this integration
there was no I6-N runner. The old runner lacks the new operational masks,
closure-time weekly schedule, flexible core and working joint DOIN bridge.
The current delivery addresses components, not all those integration steps.

There is also a source-clock contract conflict: retained ingestion marks
source text as UTC, while subsequent code strips timezone and interprets it
as America/New_York. The producer documents fixed EST without DST:
[HistData FAQ](https://www.histdata.com/f-a-q/). The archive-level manifest
and source exports used in ingestion were not retained in the inspected
provenance; the recorded download directory is absent on all three hosts.
The audit's year-dependent discrepancy is not enough to choose a repair.
Do not blindly add one hour only to DST-gap weeks or assert old scores changed
until the byte-bound clock is reconciled. No raw data was altered.

Next: resolve that provenance/calendar, wire weekly masks into the trainer,
complete joint DEAP/DOIN evaluation and S/U OLAP publication, then run the
successor configs automatically. These are technical objects to implement,
not another request for owner authorization. TEST remains closed.
