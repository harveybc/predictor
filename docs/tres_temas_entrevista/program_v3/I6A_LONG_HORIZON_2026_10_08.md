# I6-A: selected-input long-horizon successor

Status: RUNNING, not a scientific result. The next matched weekly control is
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
