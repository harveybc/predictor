# Historical pilot return

Measured 2026-10-11, code `ecc94a49`. This is one retrospective development
week, not annual validation, TEST, a selected champion, or trading utility.

## Result

EURUSD, horizon 2 hours, development week 2024-01-07: 94 paired predictions
in original USD/EUR have MAE **0.0008196087246374458**, against the **naive for
2 hours of 0.000818191489361712**. Owner skill `(MAE-naive)/naive` is
**+0.0017321559734622643**, or **0.1732156% worse**. The model does not beat
this week's naive. No annual or other-horizon conclusion follows.

Twenty RAW features; 13 input samples spaced two elapsed hours, including t,
cover 24 hours. Per-feature Conv branches, fusion and Conv core have 18,113
parameters. The four-year history starts 2020-01-05 22:00 UTC and ends at
the preceding market close, 2024-01-05 22:00 UTC. There are 18,346 TRAIN
and 350 inner rows. TRAIN-only scaling and per-output business masks apply.

One fit, seed 6102026: 23 epochs, 1,656 optimizer updates, 295.599 seconds.
Mean TRAIN/inner early stopping restored epoch 18; checkpoint reload and
prediction parity passed. The admission monitor retained a sampled cgroup
peak of 1,310,998,528 bytes under a 6 GiB ceiling. This is not an exact
internally retained final peak or a bound for larger model families.

## Custody

- Input: `56dbc22e182e5e720d4415f6998206065a506081941c3c2e76d5f8ed6c076031`.
- Scope certificate: `376f0001be4b129a5a2d247a521ae379569deb03ed451ee62249f9a58260becf`.
- Paired support: `b8b21c249e6f6aac34c66649a3d1451e10c93e88fa2b3d0210ca2d396f843f5e`.
- Result: `00fc4e017b2ca40a6310d97bf304b188d2e28a078b0263d5af4447e305b22e28`.
- Selected weights: `3fe0fb81635886438b6628db076cd489e800ea20938d3a1eefdf492ac3170d6e`.

The complete result/checkpoint is retained on the worker and coordinator.
Four OLAP reports passed exact metric readback. S/ALL_ROLES and the explicit
management role describe the same population, not independent observations.
U is empty, with no invented error or business performance. Raw evidence and
large arrays are not committed to Git.

## Source limitations

The old source clock was corrected against annual raw quotes and independent
UTC observations. The full 23-case audit remains INCONCLUSIVE because one
October 2019 download failed. A separate scope certificate admits only the
required December 2019-2024 data: 20/20 checks from 2020-2024 passed, and
December warmup alignment remains an explicitly declared winter-clock
assumption. Of 30,854 retained rows, 1,968 moved one hour later; none moved
earlier. Original artifacts remain intact.

The NY calendar is nominal, not broker-certified. Feature selection used
2024 development; historical exogenous release times and revision vintages
are not certified. These are conditional retrospective results, not evidence
of prospective deployability. No TEST data was read.

## Continuation

Two additional cost pilots completed on the same development week and code,
without repeating h2. Each result was copied and delivered automatically:

| Horizon | Inputs (samples / spacing / span) | TRAIN / inner / evaluation | MAE | Naive for this horizon | Owner skill |
| --- | --- | --- | --- | --- | --- |
| 3 h | 9 / 3 h / 24 h | 17,873 / 346 / 93 | 0.0009720544473832989 | 0.0009698924731182897 | +0.2229087% |
| 72 h | 5 / 36 h / 144 h | 4,629 / 70 / 24 | 0.0024206369962948746 | 0.002183750000000003 | +10.8477159% |

H3 ran on the secondary GPU: 16 epochs, best 11, 1,120 updates, 464.702 s;
admission sampled peak 1,177,051,136 bytes under 3 GiB. H72 ran on the
preferred GPU: 8 epochs, best 3, 152 updates, 33.282 s; sampled peak
1,405,149,184 bytes under 6 GiB. These are sampled whole-scope observations,
not exact final peaks or guarantees for unmeasured architectures. Both used
seed 6102026, the same small 18,113-parameter family, and four calendar years.

Result digests:
- H3: `44e745e2b15833032d5897a62a0bfc339f50cdc6cd985ea5461ce0c0e562792b`.
- H72: `b27d74c17cdac79cb59cd8640c85316359469cbe600049ec44cbe2ac94808ad0`.

There are now 12 OLAP reports with readback across three pilots. None beats
its horizon's naive in this week. Different horizons have different targets,
support and naive values; do not average these into one common naive.

Continue the joint context/resolution/core search on frozen TRAIN/inner
support, then evaluate the fixed procedure across all eligible development
weeks. Do not reuse this outer week as the candidate fitness or refit this
identity. The endpoint evaluator exists; trajectory supervision and the full
distributed DOIN deployment are not yet operational. Retain these distinctions
in status until their actual acceptance and execution.
