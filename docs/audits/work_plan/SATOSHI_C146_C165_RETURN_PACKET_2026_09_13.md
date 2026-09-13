# Return packet — C146–C165, memory and causality

**Date:** 2026-09-13
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_C146_C165_MEMORY_AND_CAUSALITY_ORDER_2026_09_13.md`
**Audit answered:** `MUSASHI_AUDIT_C122_C145_RUNTIME_CAUSALITY_2026_09_13.md` (`REVISE_P0_BEFORE_RELAUNCH`)
**Stopping at:** `D0_D2_MEMORY_BOUNDED_AND_CAUSALITY_AUDITED_READY_FOR_MUSASHI_REVIEW`

Zero GPU, training, feature selection, models, D3–D5 execution, RL, DOIN, live or venue. No eligibility granted. The conclusions are mechanical and bounded.

---

## 1. My own faults, first

1. **My uncapped profile run exhausted the coordinator's memory twice on 2026-09-12.**
   - I launched it with 8 workers and no memory estimate.
   - The kernel killed a runner worker at 22:13 (12.63 GiB) and at 22:29 (21.83 GiB), and systemd-oomd killed the owner's browser at 22:12:57.
   - Each kill ended the editor session scope, which stopped the SNR calibration. That is why it never wrote an exit line.
2. **I relaunched after the first kill without reading the kernel log.**
   - I took it for a session restart, removed the two empty roots, and reused their names.
   - The kept roots belong to the second attempt; the first attempt's launch time is not on disk.
3. **My status message after the relaunch said the runs were progressing** without checking memory. The owner found the OOM before I did.
4. **The runner wrote no job-start record,** so neither killed worker can be tied to its dataset. The PRE names only projection-consistent candidates.
5. **When I resumed work on 2026-09-13, one agent's deliberate 200 M probe ran in the desktop app.slice.** Its capped kill (00:48) showed the owner an "app closed because of low memory" warning; no application was affected. I stopped both agents and installed user-level guards on all three hosts:
   - `crispdm-batch.slice`: one hard ceiling for all batch work (14 G, 14 G, 8 G), no swap;
   - `crispdm-run`: a launcher with a per-job MemoryMax and a refusal when free memory is short;
   - `crispdm-memguard`: a watchdog that signals only the batch slice;
   - a Claude Code hook that refuses uncapped compute.

   The hook's first version blocked a read-only `pgrep` and briefly let `env -u PYTHONPATH python tools/df_*` through. I rewrote it and tested it on 15 cases.
6. **My PRE dry run had three harness defects:**
   - a literal base comparison with a stripped porcelain line;
   - a list-membership test for `read_table`;
   - word matching for the reference and controls.

   I corrected them before commit and noted them in the script.
7. **`host_preflight_c161_v1` recorded the coordinator as 1 CPU,** because `nproc` honoured `OMP_NUM_THREADS=1` inside the launcher. The field is not judged. I fixed it with `nproc --all`, and v2 records 16/32/32.
8. **Faults in merged agent work, found in my review:**
   - **Peak field:** the runner recorded maxrss as the observed peak. In the C151(b) mutant, maxrss read above the cgroup limit that held. It now records the cgroup's own memory.peak (`3fc923c`).
   - **Constant blocks:** the first worst-case smoke recorded 20 ADF/KPSS block rows as FAILED ValueError. I checked on the data that every one of those blocks is all zeros, and they are now INCONCLUSIVE ZERO_VARIANCE_BLOCK (`58a4420`). `c151_worst_case_smoke_v1` is kept as the record.
9. **Behaviour changes to the frozen PRE and past lab results:**
   - **PRE:** it calls the old API, so re-running it now refuses. The POST reproduces each defect against the new API instead.
   - **Sums:** batch mean and FIR sums now accumulate oldest-first, so they match the independent reference bitwise. They may differ from the committed C137 lab outputs at the 1e-16 level.
   - **FROZEN operators:** they now transform only from the end of train.
   - **C137 lab root:** not re-run; it stays historical.
10. **My first POST run had three harness defects,** although all 17 checks and the battery (807 tests) had passed:
    - it crashed at the identity comparison, reading `PRE.HERE`, which the PRE module lacks;
    - its health probe labelled every kernel kill as outside the batch slice, when both kills it counted were the deliberate C151(b) mutants inside it;
    - it counted the pre-existing OLAP outbox loader as batch work.

    The crashed output is kept, redacted, as `c146_c165_post_2026_09_13.first_run_harness_defect.out`. The committed POST is the re-run.
11. **Two defects the loads exposed,** open for the next round. Neither changes a result:
    - **Estimate rows:** in block mode, the child's memory-estimate rows for ADF/KPSS do not carry the block offset, so 2,098 rows are byte-identical and the cube keeps one.
    - **Coverage grid:** it counts metrics that do not apply as NOT_RUN, and records REFUSED lab runs as FAILED. See section 7.

## 2. A correction to the audit's arithmetic

- **The arithmetic reproduces:** 22.71 GiB for ADF at n = 13,253,761, with lag 228.
- **That n never reached ADF:** the runner passes one partition's longest finite run. The worst cell is FIN job 547 train, n = 7,952,256, lag 201, a 12.03 GiB design.
- **The audit's conclusion still stands:** adfuller peaks at 5.04× its design (measured at n = 100k–400k in capped children), so that cell projected to 60.7 GiB, above every host's RAM.
- **The dataset:** it is a monthly 0/1 recession indicator materialized at 5 minutes, so its unit-root rows describe a step signal created by that materialization.

## 3. PRE and POST

**PRE** `c146_c165_pre_2026_09_13.py` (`ad30cf3`, output `dcf5710b`), frozen before any edit under an 8 GiB cgroup cap:
- 14 of 14 defects reproduced;
- every fact holds;
- identities unchanged.

**POST** `c146_c165_post_2026_09_13.py` (`3ed9816`), at the final tips, under a 6 G capped scope:
- 17 of 17 corrected;
- preserved identities unchanged;
- no host names or home paths in the output.

| check | what the POST read or executed | result |
|---|---|---|
| C146.record | incident record `0343da9f` re-derives, is bound to the PRE, and both incomplete roots are unchanged, read-only and unpromoted | CORRECTED |
| C151.worst_case_smoke | smoke v2 terminal and output re-hash; 0 FAILED; v1 defect record present | CORRECTED |
| C148.finite_adf | policy `dcea5b87`; the largest ADF (block 200,000) projects to a small fraction of the smallest host | CORRECTED |
| C150.isolation | no process pool; hard cgroup limit, heartbeat, stop file, resume, six terminal states | CORRECTED |
| C152.fit_snapshot | bare confirmation array and confirmation rows as TRAIN: both refused | CORRECTED |
| C153.transform_snapshot | unordered bare rows and a frozen fit on its own rows: refused; reorder control detected | CORRECTED |
| C154.fit_modes | 28 FIT_MODE rows PASS, including the decomposition under both modes | CORRECTED |
| C155.name | old name refused; `T06_CAUSAL_SWT` DESIGN_ONLY_NOT_IMPLEMENTED | CORRECTED |
| C156–C159.battery | 1,079 rows; 10 controls and 17 guard mutants detected; seven-cut helper removed | CORRECTED |
| C160.wavelet_mad | a bare-array call refused; contract state offline; no public kernel | CORRECTED |
| C161.preflight | all roles dispatchable at `cc1c1f7`; evidence by role only | CORRECTED |
| C162.physical_counts | 715 datasets, one terminal each, outputs re-hash, peaks under limits, one code digest | CORRECTED |
| C162.by_bank | per-bank terminal states; no RESOURCE_EXCEEDED or UNCERTAIN | CORRECTED |
| C163.snr | 513 units, rows re-hash, wavelet MAD offline, real label model-conditional | CORRECTED |
| C164.olap | throwaway idempotent and dropped; real load zero refused; history unchanged; loader never restarted | CORRECTED |
| batteries.focal | 27 files, 807 passed, including the C151(b) preflight-bypass mutant | CORRECTED |
| health.roles | see section 9 | CORRECTED |

## 4. Runtime bounded by memory (C147–C151)

- **Planner (C147):** estimates peak bytes per dataset × variable × partition × metric group from metadata only.
  - 90 calibration cases, every estimate ≥ measured.
  - Decisions are RUN_EXACT, RUN_BOUNDED or NOT_RUN_RESOURCE_BOUND. The library is never invoked past the budget.
- **ADF/KPSS (C148):** exact up to 200,000 observations. Beyond that, three contiguous 200,000-observation blocks (start, middle, end), each with its temporal universe, plus the spread.
  - Policy hash `dcea5b87`.
  - The 5% decision agrees with the exact test in 59 of 60 comparisons.
  - A constant block is INCONCLUSIVE.
  - ADF and KPSS are descriptors, never gates.
- **Columnar reads (C149):** only needed columns, row group by row group, variable by variable; rows written incrementally with fsync and atomic rename; multivariate matrices bounded and declared.
- **Isolation (C150):** one process per dataset in `crispdm-batch.slice`, with MemoryMax and MemorySwapMax=0, RLIMIT_CPU, a wall limit, a heartbeat, a stop file, resume by contract and code digest, and six durable terminal states.
- **Worst case (C151):**

| run | dataset | status | observed cgroup peak | planned | limit | wall |
|---|---|---|---|---|---|---|
| smoke v1 | FIN 547 (13.25M rows) | COMPLETED; 20 constant blocks FAILED (defect) | 1.56 GiB | 3.02 GiB | 3.77 GiB | 80 s |
| smoke v2 | same, fixed code | COMPLETED, 0 FAILED | 1.557 GiB | 3.017 GiB | 3.772 GiB | 80 s |
| C151(b) mutant | preflight bypassed, 300 M limit | RESOURCE_EXCEEDED, CGROUP_OOM_KILL inside its scope; host unaffected | — | — | 0.3 GiB | 1 s |

## 5. Executable causal boundary (C152–C160)

- **Snapshots (C152, C153):**
  - `FitSnapshot` and `TransformSnapshot` are built only from a contract and its verified bytes.
  - At the last point of use the consumer re-derives the contract, matrix, source, snapshot digest, monotonic timestamps, range inside the partition, role, later-partition exclusion and availability.
  - A bare array or a label is refused; the numeric kernels are private.
- **Fit modes (C154):**
  - FROZEN_PREVIOUS_PARTITION refuses in-sample rows.
  - EXPANDING_PREFIX updates state after emitting t; it is implemented for the seasonal decomposition.
  - OFFLINE_ANALYSIS_ONLY_NON_CAUSAL never transforms.
  - Wavelet thresholds are FROZEN only.
- **Name (C155):** `wavelet_haar_atrous` is now `trailing_haar_threshold`, with its recurrence, edge rule and warm-up documented; `T06_CAUSAL_SWT` stays DESIGN_ONLY_NOT_IMPLEMENTED.
- **Battery (C156–C159):** `causal_battery_c156_v1`, summary `71d26a0b`; 1,079 rows, `all_pass`, 157 s, 120 MiB.

| class | outcome | rows |
|---|---|---|
| PREFIX_ALL_T | PASS | 252 |
| BATCH_STEP_CHUNK_RESTART | PASS | 252 |
| SUFFIX_ADVERSARIAL | PASS | 252 |
| REFERENCE_EQUALITY (bitwise, independent prefix-only reference) | PASS | 252 |
| NEGATIVE_CONTROL (10 classes) | DETECTED | 10 |
| GUARD_MUTATION (17 guards, including seven cuts vs every t) | DETECTED | 17 |
| SNAPSHOT_REFUSAL | PASS | 16 |
| FIT_MODE | PASS | 28 |

- **Wavelet MAD (C160):**
  - Its contract state is `OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL`, and it has no public kernel.
  - It yields one figure per whole TRAIN snapshot.
  - A call-graph check finds no consumer, and a bare array is refused.
  - SNR calibration is identical to `ad30cf3`.

## 6. Distributed execution (C161–C163)

**Hosts by role (C161).**
- **Preflight:** `host_preflight_c161_v2` (`5f9147aa`) at `cc1c1f7` found all three roles dispatchable: same commit and code digest, 3,804 data files verified by sha256 (1.71 GiB), memory cgroup delegated, zero GPU compute.
- **Isolation:** workers ran from a detached worktree; their main checkouts were not touched.

**Profile campaign (C162).** 715 datasets split by planned memory (`campaign_plan_c162_v1`, `f77b2ca1`).

| role | datasets | COMPLETED | other terminals | max cgroup peak | receipt |
|---|---|---|---|---|---|
| COORDINATOR | 118 (72 synthetic, 46 financial) | 118 | 0 | 0.87 GiB | `d51e32e0`, 113,757 rows, 0.33 h |
| WORKER_A | 392 (293 synthetic, 96 financial, 3 public) | 392 | 0 | 1.91 GiB (planned 6.71, limit 8.38) | `8e5d0432`, 554,141 rows, 3.72 h |
| WORKER_B | 205 (148 synthetic, 56 financial, 1 public) | 205 | 0 | 1.25 GiB (planned 3.05, limit 3.82) | `d32ca999`, 198,763 rows, 0.70 h |

- **Totals:** 715 datasets and 866,661 profile rows. No observed peak exceeded its limit.
- **Collection:** worker roots were copied to the coordinator, and every output re-hashes to its terminal and receipt.
- **Scheduling:** memory admission kept each host at one dataset at a time, because the planned peaks are conservative. That cost throughput, not safety.

**SNR (C163)** `snr_calibration_c163_v1` (JSON `7f571fac`, rows `8c7f8499`):
- **Run:** 513 units, 3,798 records, 37,980 rows; 24 min, 141 MB peak.
- **Not identifiable:** 39 to 130 per estimator.
- **Colored noise:** under AR(1) and 1/f noise, even the least-biased estimator is off by about 3 dB. The least-biased table is descriptive only.
- **Real data:** stays `MODEL_CONDITIONAL_SNR_ESTIMATE`.

## 7. OLAP and coverage (C164)

**Throwaway rehearsal** (`throwaway_d0d2_c164_v1.json`, `37c540d3`), 743 s under a 10 G cap:
- **Rows:** the first load inserted 1,786,383 rows with 0 refused; the second inserted 0; the database was dropped.

**Real additive load** (`real_d0d2_c164_v1.json`, `c97f5cab`), 389 s:
- **Rows:** 1,786,383 inserted, 0 refused.
- **History:** unchanged over the 34 pre-existing base tables, discovered at load time.
- **Loader service:** active, NRestarts=0 before and after.
- **Duplicates:** 2,098 offered rows were already present, all identical estimate rows (confession 11).

C164 grains in the cube:

| table | rows |
|---|---|
| df_fact_dataset_terminal | 715 |
| df_fact_resource_estimate | 255,786 distinct (257,884 offered, parent preflight plus child runtime) |
| df_fact_causal_test | 1,079 |
| df_fact_naming_isolation_decision | 3 |
| df_fact_host_receipt | 54 (preflight v1 and v2) |
| df_fact_incident_attempt | 4 |

Also loaded:
- profile rows: 866,661 across five tables;
- SNR rows: 37,980;
- the C137 lab tables with the corrected delay/cost table;
- 715 datasets and 2,487 variables.

**Exact coverage deficit.** 220,347 cells, 158,138 RESULT, 0 undeclared. The rest:

| state | cells | what they are |
|---|---|---|
| NOT_RUN | 35,328 | block ADF/KPSS metrics expected for runs tested exactly (at most 200,000 observations): not applicable |
| NOT_RUN | 15,540 | 259 variables × 60 metrics never profiled: 198 financial `timestamp` columns (excluded by design), 49 financial and 12 public text columns (non-numeric, listed in the receipts) |
| NOT_RUN | 2,349 | PCA matrix shares for variables outside the declared matrix cap |
| NOT_RUN | 1,544 | exact ADF/KPSS cells for runs tested by blocks (NOT_RUN by policy) |
| INCONCLUSIVE | 6,764 | typed per row: aliasing without a control, no ACF crossing, short samples, zero variance/MAD/denominator, constant blocks |
| FAILED | 675 | the historical C137 lab's REFUSED operator runs, all typed abstentions: Kalman MLE at bound (200), zero wavelet detail MAD (49), zero Hampel scale (42), LLT MLE not identified (36), too-short train stretches. The ledger has no REFUSED state. |
| UNAVAILABLE | 9 | one financial variable whose symbol-based information metrics are unavailable |

The backlog is not zero, and the deficit above is exact.

## 8. Branches, tips, digests

| repository | branch | tip |
|---|---|---|
| predictor | `satoshi/c146-c165-20260913` | `3ed9816` (pushed) |
| financial-data | `satoshi/c122-c145-20260912` | `7464ea79` (pushed) |

**Predictor commits this round:**
- `ad30cf3` PRE
- `760ff32` incident record, host preflight, C164 grains
- `ff89716` C147–C151
- `3fc923c` peak field
- `58a4420` constant blocks
- `ded6939` C152–C160
- `cc1c1f7` campaign plan and jobs file
- `5003550` loader for role roots
- `3ed9816` POST

**Other commits on these branches:**
- The owner committed an additive OLAP lake service on the predictor branch (`e894d63`, `426405c`). It is recorded and not touched.
- The financial-data tip moved past my C127 commits through the owner's lake commits (`d5386b998` and later). None of those commits is mine.

**Worker worktrees:** detached at `cc1c1f7` on both workers.

**State roots,** all read-only:

| root | digest |
|---|---|
| `c146_incident_v1` | — |
| `c151_worst_case_smoke_v1` | — |
| `c151_worst_case_smoke_v2` | terminal `540326f0` |
| `causal_battery_c156_v1` | — |
| `host_preflight_c161_v1` | — |
| `host_preflight_c161_v2` | — |
| `campaign_plan_c162_v1` | — |
| `profiles_c162_v1_coordinator` | — |
| `profiles_c162_v1_worker_a` | — |
| `profiles_c162_v1_worker_b` | — |
| `snr_calibration_c163_v1` | — |
| `load_receipts/*_c164_v1.json` | — |

The non-governing roots `profiles_c130_v1` and `snr_calibration_c134_v1` are kept unpromoted.

## 9. Final health of the three roles (POST)

| role | available | memory watchdog | campaign service | GPU compute | kernel OOM kills since campaign start | outside the batch slice | failed units | batch units running |
|---|---|---|---|---|---|---|---|---|
| COORDINATOR | 23 GiB | active | inactive/success | 0 | 2 (the C151(b) mutants) | 0 | 1 (`logrotate`, pre-existing config error) | 0 |
| WORKER_A | 19 GiB | active | inactive/success | 0 | 0 | 0 | 0 | 0 |
| WORKER_B | 10 GiB | active | inactive/success | 0 | 0 | 0 | 0 | 0 |

Still running and expected: the memory watchdog on every role, and the pre-existing OLAP outbox loader on the coordinator.

## 10. What needs you

1. **Review** of C146–C165 against the stop `D0_D2_MEMORY_BOUNDED_AND_CAUSALITY_AUDITED_READY_FOR_MUSASHI_REVIEW`.
2. **A ruling on the two open defects** (confession 11): add the block offset to estimate rows, and make coverage state-aware for inapplicable metrics and REFUSED runs. Either now or in the next order.
3. **Whether the C137 lab should be re-run** on the snapshot API. Its results stay historical, and FROZEN kinds now transform only after train.
4. **Owner-side, not mine:** the coordinator's `logrotate` fails on a duplicate `cloud-init` entry and needs sudo.
