# Weekly Integration Return

Date: 2026-10-11. Code: `3a9c575ee3a7b522e5ea2024c0207a8be5fb6c10`.
Branch: `codex/i6n-integration-20261011`, published.
Scope: **software integration with synthetic training**, not financial results.

## Completed

Three independent delegated scopes delivered masked training, bounded joint
DEAP search and source-clock recovery tooling. The coordinator implemented
weekly partitioning, the end-to-end test and finite automation. Subsequent
cross-reviews found and corrected: absent inner labels treated as zero loss;
unverified calendar/origin coverage; evaluation support differing from C1;
known closures rejected as missing calendar; stale RUNNING status after process
death; and a device probe being insufficient proof of training placement.

Tests preceded new implementations. The complete pinned CPU verification
finished all ten stages in **71.334 seconds**, **721 passed**, no failed/skipped
tests: contracts 284, population bridge 59, temporal operations 52, model 102,
weekly data 45, trainer 64, search 97, source recovery 9, weekly integration 3,
automation 6. Source retrieval tests use mocks, not recovered prices.

The preferred 5090 then executed the synthetic weekly integration under a
6 GiB / 180 s admission after 9,777,618,944 available host bytes were observed.
It completed in **8.967 seconds**, with **3 epochs, 12 optimizer updates,
3,158 parameters**, best epoch 3 restored and model reload checked. Placement
used hard GPU assignment, GPU-resident trainable variables and output, not
only a standalone device probe. The other GPU on that host was not used.

The fixture has three generated features and two heads (2/72 h) for cheap
contract verification. It explicitly records `coverage_complete=false` under
`synthetic_sparse`; it is neither the 20-feature market experiment nor a
four-year fit, annual validation, historical cost estimate or scientific
replicate. Original-price forecasts and separate horizon naives pass the
C1/C2 bridge. A subsequent status query recomputed the metrics from the
retained inputs and returned COMPLETED without another fit.

## Retained Evidence

CPU directory: `/var/tmp/i6n-verification-20261011-v2`.
GPU directory: `/var/tmp/i6n-weekly-gpu-3a9c575e`, copied back to the coordinator.
Large artifacts and source data are not committed.

| Artifact | SHA-256 |
| --- | --- |
| CPU STATUS | `29072e0651f76fd9fe6b518431d0e4239ff3d3829bd38565582195ba56d33a05` |
| GPU STATUS | `48544257b78e52c3278f70358cbb44353e66ba6912dc9fa496bbc595d64f104f` |
| GPU RESULT | `ffbf7283c02c9ebef8f3e0d21da598fca3e5dc7e963c19c89643415671983be3` |
| GPU checkpoint | `1ad1a97c7b1772ce2c71788d8f65e06a30811542c0e3185fcb551707dfec72a4` |

The evidence is checksummed, not cryptographically authenticated against an
attacker rewriting every file. CPU and GPU environment identities differ;
the GPU status was checked with its original interpreter, not falsely accepted
against the coordinator's different TensorFlow environment.

## Not Completed

- Original-source clock reconciliation: the public March 2024 download
  returned an empty response; **zero ZIPs** recovered. No timestamp repair
  was inferred from that failed retrieval. See `SOURCE_CLOCK_RECOVERY.md`.
- Historical input adapter, full weekly development campaign and live OLAP
  publication for this successor. No new market MAE/naive comparison exists.
- Actual shared DEAP/DOIN bridge. The local tested search is mutation plus
  immigrants with stub scoring, not deployed distributed optimization.
- Strategy chronological TP/SL integration and actual-use population scoring.

No TEST data, live orders, VM, warehouse restart, cache drop or machine reboot
was used. The legacy campaign stayed paused. The new finite verification and
GPU integration run are complete, not still running. There is no unattended
historical I6N campaign to report. Next engineering work is the historical
clock/calendar/data adapter and the real shared optimization interface; these
are technical implementation tasks, not renewed permission requests.
