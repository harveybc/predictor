# FINDING PS3R-MEM-01: synthetic-noise timing did not predict the real pilot's memory

Recorded by lane A (Satoshi, successor technical lead) on 2026-10-01, on the coordinator's order.

**What happened.** Two PS3-R pilot children were each capped at 2G with crispdm-run on worker_b. The 2G cap had been approved on the strength of timings measured on synthetic noise. At 03:16:58Z the admission monitor stopped both children:

- incident `ps3r-pilot-s0-1790824388-1394310-90baca`
- incident `ps3r-pilot-s1-1790824388-1394314-7efb0e`

Each child sat at about 2.06 GB against its 2G cap, with own-cgroup PSI of 79–88 % (reclaim thrash). Host PSI reached 49, and M02's donor run was stopped as a bystander.

**Cause in my design.**
- The design (§5) estimated peak RSS from a different workload: TF plus one fusion forward pass. No pilot record had been run under measurement before the cap was declared.
- The cost section priced time only.
- The runner did not record per-fit cgroup peaks, and had no self-stop.

**Status of results.** Both runs are INTERRUPTED_RESULT_NOT_RETAINED. The only exception is the per-(input, fold, arm, seed) records that completed atomically. Those are reused only where they bind the same data sha256 and the same objective identity (`_reusable`).

**Correction (runner after this finding).**
- One child only, capped at 4G.
- The cgroup `memory.current`/`memory.peak` and the RSS peak are written per record to `memory_peaks_shard*.json` and to the heartbeat.
- The child stops itself at a record boundary if a peak reaches 90 % of its declared cap.
- The production cap is declared as 1.25 × the measured peak, and the remaining work resumes one child at a time.
- No two PS3-R children run concurrently while M02's 7G donor run is live.

Note: 2.06 GB is a LOWER bound on demand, because the children were held at their cap.

## Measurement under the 4G single child (one process for all records)

Run as crispdm scope `ps3r-pilot-measure`, with one process fitting record after record. The scope's cgroup peak grew with every record instead of settling to a stable per-fit peak:

1.61 GB → 2.02 GB → 2.48 GB → 2.78 GB → 3.73 GB, over 9 records. The last process RSS peak was 4.10 GB.

So the memory is **accumulation across fits inside one process**: TF graphs and traced functions are not released by `clear_session`. It is not a per-fit requirement, which means no fixed cap would have been correct for the old runner. I stopped the child at a record boundary (3.73 GB against 4G). The per-record table is in `memory_peaks_single_process_measure.json`.

**Fix.** Every record now runs in its own spawned process (`_child`), so its memory is returned when the record ends. The parent imports no TensorFlow. The production cap will be 1.25 × the measured per-record peak under this runner.

**Second correction found in the same run.** One contrastive fit (close_sma_ratio_100, inner_3, seed 2021) never beat its initial validation loss, and version 1.0.0 then failed the fit. Version 1.0.1 of `ts2vec_contrastive` instead restores the initial state as the best checkpoint, with stop reason `no_improvement_over_initial`. Δ_probe is then exactly 0, which is a result, not a failure. Because the identity changed, the earlier contrastive records are not reused: they are re-fitted under 1.0.1.

## Planned stop for M04's headroom (coordinator sequencing ruling)

At 03:53:27Z I stopped child `ps3r-pilot-measure2` (4G) at a record boundary. The stop came right after record 30 was written atomically, and before the next record had been written. My queued suite request (`laneA-suite`, 4G) was cancelled at the same time, so that M04's per-feature GPU pilot can be admitted on worker_b.

- **Records retained:** 30 atomic records.
- **Reuse rule:** they are reused by data sha256 and objective-identity sha when the pilot resumes.
- **Production cap:** 3G, per the coordinator's ruling (1.25 × 2.04 GB, rounded up).
- **Resume conditions:** one child only, and only when the coordinator says M04's pilot is complete.

The runner now also honours a `STOP` file in its state directory. That ends the run cleanly at the next record boundary, so future stops do not need the wrapper to be signalled.
