# I6-A: selected-input long-horizon successor

Status: `Y_l_24h` COMPLETE and warehouse-published; `Y_s_2h` and `Y_l_48h`
running independently on the 4090 and 5090. The first matched weekly control was
EURUSD `Y_l_24h` on the four RAW features frozen for that target. It uses the
same four architecture arms, 24-hour input window, four-year rolling weekly
retraining, 2024 validation weeks, one seed and same-row zero-return naive as
the 1-hour campaign. The target's 24-hour availability support is purged; it
is not borrowed from the old 1-hour task. TEST remains closed.

`tools/i6a_weekly_arch_pilot.py` now parses only declared short (1-6 h) and
long (24-144 h, 24-hour steps) regression targets. Barrier targets and unknown
horizons reject. `tools/i6a_campaign.py --target` partitions complete weeks;
the preferred worker takes two of every three and a second worker takes the
third. Every worker checks its expected GPU UUID against the host inventory
and the device name TensorFlow sees before fitting; result receipts carry the
observed name and UUID. This matters because CUDA ordinal 1 resolved to the
other GPU despite the `nvidia-smi` index on the preferred worker. That first
one-week probe is retained as a device diagnostic, not mixed into the annual
5090 campaign. The four matching week-0 cells were run on the measured 5090
and are reused by digest rather than trained again.

`tools/i6a_close.py --target Y_l_24h` requires all 208 cells, paired row
populations, a single input identity and a single core-code identity before
publishing any architecture means. It reports both the predeclared equal-week
summary and an additional row-weighted pooled MAE and paired naive MAE. While
the workers run, their own `STATUS`
files report completed cells, current cell and ETA. Do not interpret the
one-week pilot or open TEST before the full closure.

## Complete 24-hour validation result

Closure `4b90854e5877c934fccccb907cdc43c18b95b645eb7c791f0f4d9f303bad14e6`
accepted all 208 cells and 52 paired weeks, with no TEST access. The compact
result is indexed at [`Y_l_24h`](../../results/I6A/EURUSD/2024_validation/Y_l_24h.md).
ARCH-B has mean weekly MAE 0.002568483 versus same-row naive 0.002516863;
pooled MAE is 0.002570567 versus pooled naive 0.002519108 over 6,196 rows.
It beats naive in 16/52 weeks but fails the full-year gate. ARCH_0/A/C have
mean weekly MAE 0.010082880/0.002596691/0.002780527 respectively. This is
development evidence conditioned on 2024 feature selection, not a test result
or trading eligibility.

The first warehouse publication incorrectly stamped every 24-hour metric with
`horizon=1`. Its experiment set `i6a:2024:EURUSD:Y_l_24h` is retained as
**superseded for horizon analysis**. Without retraining or deleting history,
the same 208 authenticated cells were republished under
`i6a:2024:EURUSD:Y_l_24h:metric-horizon-v2` with `horizon=24`; readback
verified 208 reports and 1,664 metric rows. Queries comparing horizons must
use the v2 set. Future non-1-hour targets publish only with the v2 binding.

`tools/i6a_result_catalog.py` produces a digest-bound one-paragraph result
per target plus [`docs/results/INDEX.md`](../../results/INDEX.md). It refuses
partial closures, TEST access, mismatched paired naives and missing warehouse
readback. `tools/i6a_collect.py` can update this index automatically after
publication. `tools/i6a_status.py --target` checks verified cells and ETA for
any declared horizon without reading TEST.

Coordinator status for the running campaigns, without watching logs:

```bash
python -m tools.i6a_fleet_status \
  --campaign Y_s_2h=$HOME/.local/state/canonical_20261003/i6a/horizon2_collected \
  --campaign Y_l_48h=$HOME/.local/state/canonical_20261003/i6a/horizon48_collected
```

Each worker status carries completed/total cells and an ETA derived from its
observed fits. The collector status carries the synchronized view and only
reports `PUBLISHED` after full closure and warehouse readback. These two
targets use separate GPUs; a failure in one does not erase or relabel the other.

The remaining declared regression horizons now have unattended, per-host
sequences: dragon takes `Y_s_3h` through `Y_s_6h` after `Y_s_2h`; gamma takes
`Y_l_72h`, `Y_l_96h`, `Y_l_120h`, `Y_l_144h` after `Y_l_48h`. The sequence
script waits for the predecessor's exact terminal status, runs one complete
target at a time and resumes authenticated cells. Separate coordinator
sequences then collect, reject partial closures, publish with the correct
horizon dimension, read back the OLAP count and update the result index.
No sequence opens TEST or turns a negative naive comparison into a strategy
result. Inspect `SHORT_SEQUENCE_STATUS.json` on dragon,
`LONG_SEQUENCE_STATUS.json` on gamma, and the corresponding
`SHORT_COLLECTOR_SEQUENCE_STATUS.json` / `LONG_COLLECTOR_SEQUENCE_STATUS.json`
under the coordinator's retained I6-A state. A failed target stops its own
sequence with its cell receipt and error; it does not affect the other host.
