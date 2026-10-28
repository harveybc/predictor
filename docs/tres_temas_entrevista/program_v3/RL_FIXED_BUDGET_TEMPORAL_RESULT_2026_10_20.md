# Fixed-Budget Temporal SAC Result

Completed2026-10-20T08:36:51UTC,about70min,100000 transitions,four monitor
pairs,zero worker restarts. Runtimeec4855ce,contractfc9c0c53,final100000
selected without patience or validation-driven restoration. Same15 selected
features,13 observations,stride2,Conv1D branches2/core4/hidden[8],seed7,
business mask,costs and leverage50. TRAIN2020-23 net-1454.77USD(-14.5477%),
maxDD37.0186%,827 trades,44.8609% wins,PF0.9340,daily nonannualized
Sharpe-0.02002;VALIDATION2024 net-727.83USD(-7.2783%),maxDD10.7319%,
203 trades,43.3498% wins,PF0.8326,Sharpe-0.06103. Terminal inference
matches the final monitor. Unprofitable development result,not a TEST result.

| Transitions | TRAIN return | VALIDATION return |
| --- | --- | --- |
| 25000 | -52.6425% | -16.8084% |
| 50000 | -41.2313% | -32.0757% |
| 75000 | -30.4627% | -22.6425% |
| 100000 | -14.5477% | -7.2783% |

Final improves both periods versus earlier observations. This does not support
simply blaming increasing training duration alone,does not establish adequate
learning and does not exclude other forms of overfit or distribution shift.
No previous intermediate checkpoint is substituted for the declared final one.

Closure:10/10 reports with readback,zero pending/mismatch;574 files,including
weights,curves and metrics,checksum-verified on a second physical machine.
Close digest810364a7c01b5dc4b6bd02b4de1cf972f64a80e28b648557b85d1d4ec50a3761.
An operational collector defect initially refused this temporal arm because
the paired branch required SAC_FLAT. Corrected with four test-first failures
and60 passing collector tests; exact contract/task arm binding and digest
checks remain mandatory. No closed scientific bytes were changed.
Root STATUS is retained as a pre-publication snapshot;CLOSE is authoritative.
No further scientific fit under this identity is active. TEST intact.
