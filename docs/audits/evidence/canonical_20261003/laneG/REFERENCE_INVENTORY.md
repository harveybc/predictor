# Lane G: inventory of the faithful published references, and the one sealed cell run

Date: 2026-10-03. Authority: consolidated master plan v3 (commit `2edc2684`), §3 rule 3, §10 and §13;
`EXPERIMENT_EXECUTION_QUEUE.json` lane `LITERATURE-CONTROLS`; `SATOSHI_CANONICAL_SELECTION_FIRST_EXECUTION_2026_10_03.md`
§3 (lane G) and the last line of §8. Hosts and GPUs appear here only as roles.

I did not change any author recipe, add any seed beyond the published three, or repeat any cell that already has
authenticated evidence. Lane G ran one cell on worker_b's RTX 4090, after a bounded pilot. Sections 2 and 3 give the
reasons and the result.

## 1. Inventory

The state words mean:

- **VERIFIED**: the cell was measured, and an independently invoked replay reproduced it.
- **MEASURED_UNVERIFIED**: the cell was measured once, with no independent replay.
- **SEALED_NOT_RUN**: the cell is in a sealed design, with the author's recipe and governed bytes, but has no score yet.
- **OPEN**: the question cannot be settled by running anything.

All metrics are in the author's normalized space and are reduced over every window × step × channel. Naive is
persistence on the same rows, proved by digest. Agreement uses the frozen rule `|mean − published| ≤ 2·std_paper + 0.0005`,
with each dataset's own Table 7 dispersion. This is an operational class, not statistical equivalence.

### 1.1 Electricity (ECL), TimeFilter (ICML 2025, arXiv:2501.13041) at pinned revision `dffde87e`

| protocol | T | ours MSE / MAE (3-seed mean) | published | naive MSE / MAE | class | state |
|---|---|---|---|---|---|---|
| A, L96, Table 8 | 96 | 0.135496 / 0.232884 | 0.133 / 0.230 | 1.5878 / 0.9455 | OPERATIONAL_AGREEMENT | VERIFIED: replay exact on the observed device; training device UNKNOWN |
| A | 192 | 0.157645 / 0.252163 | 0.154 / 0.248 | 1.5962 / 0.9507 | OPERATIONAL_AGREEMENT | VERIFIED (replay); training device UNKNOWN/INFERRED |
| A | 336 | 0.164480 / 0.262985 | 0.162 / 0.261 | 1.6178 / 0.9613 | OPERATIONAL_AGREEMENT | VERIFIED (replay); training device INFERRED |
| A | 720 | 0.190226 / 0.290617 | 0.184 / 0.284 | 1.6468 / 0.9754 | OPERATIONAL_AGREEMENT | VERIFIED (replay); training device INFERRED |
| A | avg | **0.161962 / 0.259662** | 0.158250 / 0.255750 | | | 12/12 cells (RP144 composed table `d3_k5_20260917/RP144/COMPOSED_TABLE.1790220949.json`) |
| B, released L=512 script | 96 | 0.125925 / 0.220958 | 0.126 / 0.220 (Table 9) | 1.5878 / 0.9455 | OPERATIONAL_AGREEMENT | VERIFIED: bit-exact replay, device measured |
| B | 192 | 0.143650 / 0.237970 | 0.143 / 0.237 | 1.5962 / 0.9507 | OPERATIONAL_AGREEMENT | VERIFIED |
| B | 336 | 0.152696 / 0.251345 | 0.153 / 0.252 | 1.6178 / 0.9613 | OPERATIONAL_AGREEMENT | VERIFIED |
| B | 720 | 0.178957 / 0.275313 | 0.177 / 0.275 | 1.6468 / 0.9754 | OPERATIONAL_AGREEMENT | VERIFIED |
| B | avg | **0.150307 / 0.246397** | 0.150 / 0.246 | | | 12/12 (`d3_k5_20260917/RP145/`) |
| B → Table 9 identity | all | — | — | — | — | **OPEN: the L that Table 9 used for each horizon is not published** (§4) |

The protocol A and protocol B means are a multi-parameter **recipe** contrast, not a lookback effect (RP145).

### 1.2 Weather, TimeFilter Table 8, L96 (design `dbb3e87f…`)

| T | ours MSE / MAE (3 seeds) | published | naive MSE / MAE | class |
|---|---|---|---|---|
| 96 | 0.1557 / 0.2021 | 0.153 / 0.199 | 0.2591 / 0.2542 | OPERATIONAL_AGREEMENT |
| 192 | 0.2041 / 0.2482 | 0.202 / 0.246 | 0.3092 / 0.2917 | OPERATIONAL_AGREEMENT |
| 336 | 0.2617 / 0.2906 | 0.260 / 0.289 | 0.3764 / 0.3377 | OPERATIONAL_AGREEMENT |
| 720 | 0.3452 / 0.3442 | 0.342 / 0.341 | 0.4652 / 0.3935 | OPERATIONAL_AGREEMENT |
| avg | 0.2417 / 0.2713 | 0.239 / 0.269 | | |

There are four statuses, and they are kept apart (branch `satoshi/weather-custody-and-replay-20260929` @ `f328db3a`):

- **identity**: RECONCILED for 12/12 cells;
- **replay**: BITWISE_REPRODUCED for 12/12 cells;
- **custody**: `DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT`;
- **scientific**: MEASURED, not verified as a result.

Our values are higher than the published ones on all four horizons. This one-sided gap of 1–2 % is still unexplained.
It is not a device effect: rerunning on a different device moved the metric by only about 3e-8. All 12 cells are VERIFIED
in the replay sense. **No Weather cell is SEALED_NOT_RUN.**

### 1.3 Traffic, TimeFilter Table 8, L96 (design `6cba7e20…`, protocol `322262e8…`, sealed 2026-09-29, 12 cells)

| T | seeds | ours MSE / MAE | published | naive MSE / MAE | class | state |
|---|---|---|---|---|---|---|
| 96 | 2021, 2022, 2023 | 0.375199 / 0.251143 (sd 0.00060 / 0.00028) | 0.375 / 0.251 | 2.7145 / 1.0772 | OPERATIONAL_AGREEMENT | MEASURED_UNVERIFIED: no independent replay; scored with the author's native unchunked `test()`; every seed hit the 30-epoch ceiling with its best epoch the last one |
| 192 | 2021, 2022, 2023 | 0.396509 / 0.262478 (sd 0.00015 / 0.00031) | 0.395 / 0.262 | 2.7471 / 1.0851 | **OPERATIONAL_AGREEMENT** (+0.0015 / +0.0005) | MEASURED_UNVERIFIED; run by lane G, bounded route with inherited parity (`TRAFFIC_L96_h192_CLOSURE.json`) |
| 336 | 2021, 2022, 2023 | — | 0.414 / 0.271 | — | — | SEALED_NOT_RUN; the bounded probe at h336 has not been measured |
| 720 | 2021, 2022, 2023 | — | 0.445 / 0.289 | — | — | RUNNING in lane G, one seed at a time. Bounded probe `ADMISSIBLE_PARITY_INHERITED`, 3.600 GiB. Cap 7.25 GiB = 1.25 × (h192 cell peak 5.45 − h192 probe 3.36 + h720 probe 3.60 + train-pilot delta 0.04) |

The custody class for all Traffic cells is `DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT`. The bytes are
`cb06463d…`, delivered by the adoption campaign as VERIFIED_TRANSFER and re-verified inside each child.

### 1.4 Classification

No faithful classification cell is sealed. The references are dossier-only (`satoshi/cb01-cb02-classification-references-20260928`).
Remote-code and licence review is a separate lane. Lane G has nothing to run here.

## 2. Decision: which sealed cell to run

The only faithful cells that are sealed and have never been scored are the **nine Traffic cells at h192, h336 and h720**.
They use the same sealed design as the measured h96 cells, the exact author argv, and the same governed bytes. ECL A, ECL B and
Weather have no unrun cell. Re-running any of them would be a repetition.

I chose **`traffic_L96_h192_s2021`**, the first cell in the design's own order that has no score. The order asked for one cell
at a time.

**The evaluation route is an operational patch, not a recipe change.** At h192, the author's native unchunked `test()` needs
11.25 GiB of arrays alone, by derivation. The measured native h96 cell already peaked at 10.55 GiB whole-cgroup. worker_b has
30 GiB in total, and other tenants left 8–13 GiB available. At 1.25× the native peak, the native path is not admissible. The
cell therefore runs through the existing bounded exact evaluator, `df_sota_bounded_eval.v1` with the exact float32 reducer:

- it uses the same model, weights, loader, batches, order, dtype and population;
- predictions and targets stream to disk;
- the reduction was proved bit-equal to `utils.metrics.metric` on the complete real Traffic h96 population (bounded
  evaluator return `8d461628`; `TRAINED_PARITY.traffic.h96.json`);
- at h192 its parity is **inherited**, and the record labels it that way.

The executor is the one at `codex/traffic-scored-bounded-20260930` @ `6d419f57`. Its `df_tsl_execute.py` has sha256
`b7cfb12e…`; every other module is byte-equal to the worker's sealed snapshot. Its 13 acceptance tests pass. The executor
refuses to score Traffic unless a complete-population bounded probe exists for the same design and horizon.

**Pilot, then cap.**

1. **Bounded probe at h192.** The probe was new: untrained weights, the complete 3 317 × 192 × 862 population, and no author
   witness. Its cap was 1.25 × the h720 bounded peak of 3.600 GiB = 4.5 GiB.
   - verdict `ADMISSIBLE_PARITY_INHERITED`
   - whole-cgroup peak **3.360 GiB**
   - disk high-water 4.12 GiB
   - record `3fa2e716…`
2. **Cell cap.** The cell cap is 1.25 × (TRAIN-only h192 pilot peak 1.986 GiB + probe peak 3.360 GiB) = 6.68 GiB, rounded
   up to **7 GiB**. Launch was through `crispdm-run -q`, with a 6 h wall and the GPU UUID asserted inside the child.
3. **Heartbeat.** A sidecar under its own 64 MiB `crispdm-run` scope wrote `HEARTBEAT.json` every 60 s. Each beat recorded
   the log, the epoch count, GPU utilisation, temperature and throttle state, and the scope's memory.

## 3. Result

Cell `traffic_L96_h192_s2021`, seed 2021. It ran on worker_b's RTX 4090 from 04:58:02Z to 05:55:59Z on 2026-10-03, and
the child exited 0. The full record is in `TRAFFIC_L96_h192_s2021_RESULT.json`, with record sha `86b28103…`.

| metric | ours | published (Table 8) | difference | operational tolerance | naive, same rows | skill vs naive |
|---|---|---|---|---|---|---|
| MSE | 0.396605 | 0.395 | +0.00161 | 0.0165 | 2.747078 | 0.8556 |
| MAE | 0.262610 | 0.262 | +0.00061 | 0.0085 | 1.085077 | 0.7580 |

- **Population**: 3 317 × 192 × 862 = 548 976 768 elements. It is complete and finite and matches the sealed count.
- **Naive pairing**: proved by target digest (`4ddc6290…`).
- **Metric check**: the author's float32 reduction and an independent float64 reduction agree to 9e-9.
- **Training**: all 30 sealed epochs ran, with no early stop. The best validation epoch was 28.
- **Cost**: 3 478 s wall and 3 474 CPU s. The whole-cgroup peak was 5.41 GiB against a 7 GiB cap, and GPU reserved memory
  was 5.36 GiB. The bounded probe added under 5 min. The heartbeat met the 60 s interval for the whole run.
- **Reading**: this is **one seed**. It is a per-seed difference, not an agreement class. The class belongs to the three-seed
  mean once seeds 2022 and 2023 exist.
- **Evidence state**: MEASURED_UNVERIFIED. No independent replay has been run yet.
- **Retention**: the predictions and targets (4.2 GB) are kept until a replay recomputes them.
- **Custody**: declared transport of governed bytes. This is the same class as Traffic h96.

## 4. TimeFilter Table 9, L512 mapping: why nothing was run, and the proposal

The following was checked on 2026-10-03, read-only:

- the upstream repository's HEAD is still `dffde87e`, and there is no other branch;
- `scripts/ECL.sh` has a single revision, and it releases only L=96 and L=512 for ECL;
- arXiv v1 and v2 both state that L is "searched from {192,336,512,720} for optimal horizon", without the per-horizon
  choice and without saying whether validation or test was used.

The other three L values have no published recipe, so running them would be our own search. Re-running L=512 would repeat 12
verified cells. **No faithful GPU cell can close this item.** The smallest closing step is a question to the authors, which
costs no GPU time. It is written as a sealed-design proposal in `TABLE9_MAPPING_PROPOSAL.json` (`design_sha256 62c81248…`),
together with the acceptance rule for each possible answer. Asking from a public identity is external communication by the
owner, so it is an irreducible human action. Until the authors answer, protocol B is reported as "released L=512 script
reproduced and verified; Table 9 identity NOT claimed."

## 5. Next item

1. **`traffic_L96_h192_s2022`, then `traffic_L96_h192_s2023`.** Each runs on worker_b's 4090, one at a time, and closes the
   h192 class.
   - Same design, executor snapshot and bounded probe.
   - Cap 7 GiB, which is 1.25 × the 5.41 GiB peak measured in this cell.
   - About 1 h each.
2. **h720 × 3.** The bounded probe already exists. A TRAIN-only pilot (2.03 GiB) is measured.
   - Cap = 1.25 × (pilot + probe 3.60 GiB) ≈ 7 GiB.
   - About 1.3 h each.
3. **h336 × 3.** Measure the bounded probe at h336 first. A TRAIN-only pilot exists.
4. **Independent replays.** Run them from the retained checkpoints for Traffic h96 and h192, with no retraining. Only then
   delete the 4.2 GB prediction files.
5. **Table 9.** Send the authors' query in `TABLE9_MAPPING_PROPOSAL.json`. This is owner-level external communication and
   uses no GPU time.

## 6. Continuation log (coordinator order: h192 ×2, then h720 ×3, then h336 probe and h336 ×3)

| cell | MSE / MAE | naive MSE / MAE | skill MSE / MAE | epochs (best) | wall s | peak GiB / cap |
|---|---|---|---|---|---|---|
| h192 s2021 | 0.396605 / 0.262610 | 2.747078 / 1.085077 | 0.8556 / 0.7580 | 30 (28) | 3478 | 5.41 / 7 |
| h192 s2022 | 0.396579 / 0.262699 | same | 0.8556 / 0.7579 | 30 (28) | 3483 | 5.45 / 7 |
| h192 s2023 | 0.396342 / 0.262125 | same | 0.8557 / 0.7584 | 30 (29) | 3477 | 5.43 / 7 |
| **h192, 3-seed mean** | **0.396509 / 0.262478** | 2.747078 / 1.085077 | 0.8557 / 0.7581 | | | |

On the three-seed mean, h192 is in OPERATIONAL_AGREEMENT with Table 8 (0.395 / 0.262) on both metrics. The difference is
+0.0015 MSE against a tolerance of 0.0165, and +0.0005 MAE against 0.0085. The seed SD is about 10× smaller than the
difference. Every seed is slightly worse than published, and every seed ended at or near the 30-epoch ceiling.

A supervision defect in my own wait loop caused 1 h of GPU idle time between s2022 and s2023. The liveness `pgrep` matched
its own ssh command line, so it never saw the cell end. The pattern is now fixed with a character class.
