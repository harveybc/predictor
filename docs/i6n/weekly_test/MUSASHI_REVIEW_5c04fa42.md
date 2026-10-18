# Review Of The Independent Audit

Reviewed audit 5c04fa42 against deployed fa47d2b2 and documentation 811a7544.
The three audit additions were integrated without changing deployed tool/model
source bytes. A fresh authenticated comparison against the live spec passed.
The independent probes and original procedure tests were rerun: 21 passed,
zero failed, under a 2 GiB/120-second CPU-only admission. No training, service
change, data re-selection or scientific runtime change was performed.

## Disposition

Accept the six protocol findings within the stated code/receipt/synthetic-test
scope. This is not a completed annual evaluation or an end-to-end certification
of future mirroring/closure. PS06 has actual initial-weight receipt evidence;
PS10 still awaits real annual close. Do not label unexecuted paths GPU-tested
or production-close-tested.

The retained audit snapshot contains 13 terminals: 11 COMPLETE and two explicit
SKIPPED_NO_EVAL_ROWS, not 13 COMPLETE. Its status/publication observation belongs
to that earlier snapshot. These counts must not be mixed with later live status.

The synthetic PS04 tree comparison removes wall-clock fields before comparing
JSON; NPZ byte hashes are compared directly. It proves equality of normalized
receipt payloads, not literal equality of timestamps across independent runs.
The fake predictor does not depend on the updated weight identity; the test's
week-2 invariance after changing week-1 outcomes is a property of that fake,
not a required or proven property of the real adapting model. Strengthen the
positive control in the next test revision; the real required invariance is
that current/future outcomes cannot change already-sealed earlier predictions.

## Follow-Up

- O1 is not a demonstrated error in current scientific results, but remains a
  real gap against fully unattended recovery. Prepare bounded, explicitly
  classified transient retries without changing a saved prediction or lowering
  memory caps. Permanent integrity/scientific failures must remain terminal.
- O2 needs explicit annual coverage reporting: declared, scored, no-pair,
  failed and pending weeks plus per-head pair counts. No fake zero MAE.
- O3 needs a glossary, not a machine-state rename or historical rewrite.
- PS06/PS10 can gain CPU seam tests with mocked external boundaries, separately
  labelled from an actual GPU/remote annual-close test.

The follow-up order is [ORDEN_SATOSHI_OPERACION.md](ORDEN_SATOSHI_OPERACION.md).
It authorizes isolated candidate code/tests only, not a live deployment or a
restart of the scientific campaign.
