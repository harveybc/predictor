# Post-power interruption check

Observed 2026-10-11, 21:51-21:54 UTC. No historical fits were repeated and
TEST was not opened. This supersedes the running counts in
`SEARCH_EXECUTION_RETURN_20261011.md`, which remains a historical observation.

## Completion and custody

Both bounded endpoint searches completed before the reported interruption:
h12 finished at 19:10:21 UTC and h72 at 18:44:42 UTC. Each has 24 completed
candidates, zero failures, and terminal reason `generation_budget`. Neither
has an interrupted candidate awaiting recovery. Both GPU search queues are
finished, not currently training.

The coordinator's delivery timer and warehouse service recovered and are
active. At 21:54:25 UTC, a fresh read-only warehouse comparison verified all
396 reports / 3,015 metric rows belonging to 51 terminals (48 candidates plus
three cost pilots) against the retained report bodies. No metrics were changed.
The physical receipt is `POST_POWER_READBACK.json` under the private campaign
root. Controller receipts were additionally copied from workers to the
coordinator without deleting either copy.

## Search results

These are the candidates selected by the prespecified mean TRAIN/inner
normalized-error fitness, not by retrospectively switching to inner alone.
All MAEs and horizon-specific naives below are in original price units.
Relative error is `(MAE - naive) / naive`; negative is better.

| Horizon | TRAIN n | TRAIN MAE | TRAIN naive | Inner n | Inner MAE | Inner naive | Inner relative error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 h | 16158 | 0.0027109665753556086 | 0.002755766802822132 | 310 | 0.002875319672532508 | 0.00286241935483871 | +0.4506788% |
| 72 h | 4629 | 0.007286911667513896 | 0.007443208036292938 | 70 | 0.011061611850654176 | 0.011874714285714282 | -6.8473431% |

H12's selected candidate uses 48 hours of history, spacing 3 hours (17
samples including t), differenced inputs, one Transformer block of width 16,
reducers [16,16,8] with strides [1,2,1], and a causal Conv adapter of width 8.
Its combined fitness is 0.9941249436999559. Six of the 24 h12 candidates beat
the inner naive; that does not change the prespecified selection rule.

H72's selected candidate uses 72 hours of history, spacing 72 hours (two
samples including t), raw inputs, Conv plus two Transformer blocks of width
16, reducer [16] with stride [1], and a causal Conv adapter of width 32.
Its combined fitness is 0.9552640231845004. It is the only h72 candidate to
beat the inner naive. Seventy overlapping inner observations, used during
search, do not establish out-of-sample or annual improvement.

## Host health

The preferred host restarted and has approximately 11 GiB available, no
current memory pressure, and no matching recent kernel OOM/GPU-fault records.
Both devices are visible; only its preferred GPU should receive training.
The coordinator has approximately 24 GiB available and no current memory
pressure. Its warehouse and delivery timer are active. Network wait-online
and crash-report services show failed boot units, but networking and warehouse
requests work; these are not evidence of a required reboot.

The secondary host did not restart: uptime is almost 13 days. Its GPU passes
a real strict-placement TensorFlow matrix operation, as does the preferred
GPU, with no fits and bounded memory. However, the secondary has only about
6 GiB available and 12.7 GiB SUnreclaim. In the last two hours its NVIDIA
driver recorded 117 NV_ERR_NO_MEMORY allocation errors, most recently at
21:53:01 UTC. There were no kernel OOM kills or Xid records in that interval.
The slab ownership is not proven; do not attribute all slab to NVIDIA.

Recommendation: reboot the secondary before new heavy training, then repeat
the health check. No additional reboot is indicated for the preferred host
or coordinator. Passing a small GPU kernel is not a certificate of memory
health for large fits. No machine was rebooted by this check.

## Pending work

1. Complete independent output-trajectory supervision and its masked,
   current-origin-preserving tests; the finished searches are endpoint-only.
2. Complete full DOIN placement and durable successor-stage orchestration.
   These transient local search units do not constitute that implementation.
3. Run the fixed development procedure over all eligible weeks with weekly
   refits and per-horizon paired naives, before any annual performance claim.
4. Keep TEST and trading promotion closed. Calendar, source-vintage and
   retrospective 2024 feature-selection limitations remain as documented.

There is no remaining ETA for these two completed searches. The subsequent
campaign has not started, so the earlier 44-minute estimate must not be
presented as its ETA. The lack of a successor launch is an implementation
gap, not a request for another permission or a hardware diagnosis.
