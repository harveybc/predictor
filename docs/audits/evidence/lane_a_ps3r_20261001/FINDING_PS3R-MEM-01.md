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
