# Release Evidence And Scope

Parent focused release: 129 tests passed in 63.19 seconds, CPU-only under
3 GiB. Includes full TRAIN/no INNER, TRAIN-only scales, frozen calendar
periods, boundary/availability perturbations, actual two-epoch trainer cases,
private heads, absence of additive encoding, reload/replay, conditional
strategy and the prior batch's regression tests. Two protobuf deprecation
warnings remain. After orchestration corrections, 23 control/transition tests
passed in 0.30 seconds under 2 GiB. No full legacy repository-suite claim.

Independent bounded static re-review: ACCEPT, no remaining P1 in reviewed
fixes. Reviewed bytes:

- continuation.py: 0dd4b60e6b05a56a14f300e275b3a495bafa64dd859587c575e7f691cab0722d
- resolution_batch.py: 239dc52651c5e96ff1bc2b10b61d567e7e53aa82fda87107d86fcd61f3857ebf

Corrections: technical strategy errors cannot close successfully; verified
bars/calendar pins are bound before arming; the joint grid is independent of
a failed head pilot; only fully completed representative subgrids are ranked.
The private launch supplies the retained strategy-source manifest. No new
credential action, TEST read, fit or service action was part of the review.

The continuation uses 2024 for checkpoint and recipe selection. Its validation
and strategy results are retrospective DEVELOPMENT, not independently
out-of-sample or prospective deployment evidence. All older fits retain their
original protocol and identity. Real population, GPU engineering, actual
launch and closure are separate receipts, not conferred by this document.
