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

Real population preflight PASS for both groups: semantic digest
8dd12ec51e62efd7505cc8d13f3d7b1ea7ef27e97c6ca3bc26cb0a677e940c35.
Static ranges are TRAIN [2020-01-01,2024-01-01), VALIDATION
[2024-01-01,2025-01-01); no INNER. Calendar periods: 4 quarterly, 12 monthly,
53 weekly. The frozen graph therefore has 407 logical cells, of which 138
are conditional cadence cells and some static identities may be adopted.

5090 synthetic engineering under 4 GiB: Conv/Dense/LSTM private six-head
models each completed two epochs with TRAIN/VALIDATION monitor self-check
PASS and no additive sinusoid. Peak cgroup 2,625,724,416 bytes; parameter
counts at the largest 24-tick test were 7,306 / 6,994 / 3,442 respectively.
Code: 3d0dae41, whose subsequent changes before deployment are milestone
rendering/docs only. This is not a full real-data per-case memory estimate.
