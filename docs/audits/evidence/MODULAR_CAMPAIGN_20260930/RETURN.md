# Modular campaign 2026-09-30: campaign return

Consolidated by M06 (evidence and resources) on `satoshi/m06-evidence-resources-20260930`.
Orders: `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` at `dc72170e`. As of 2026-10-01 01:40Z all eight lanes have RETURNED. One job is still live: M02's ECL donor run on worker_b.

Hosts are named by role only: coordinator (the owner's desktop), worker_a (the 5090 host), worker_b (the 4090 host). GPUs are named by UUID. Numbers in the tables below are **generated from evidence files** by the tools named; this file links to them and does not restate them.

## 1. What was measured, what is retained, what is synthetic

| Item | Class | Generated table | Generator, tests |
|---|---|---|---|
| Traffic L96→h96, TimeFilter published recipe, seeds 2021/2022/2023 | **New measurement**, three-seed class OPERATIONAL_AGREEMENT (MSE and MAE). Orchestrator independently verified it. | `RESULTS/traffic_h96_closure_table.md` and `.json`; per-cell `RESULTS/traffic_h96_rows.json` and `.csv` | `tools/m06_closure_table.py`, `tests/test_m06_closure_table.py` (9) |
| M04 DOIN batch 1 v3: 16 verified R0 candidates, ECL L24→H1..24, validation, z_train | **New measurement**, validation only. Fresh-process checkpoint rescoring with `exact_match` on all 16. **NOT_COMPARABLE** to any published row. | `RESULTS/m04_r0_candidates.md` and `.csv` | `tools/m06_campaign_tables.py`, `tests/test_m06_campaign_tables.py` (8) |
| M04 incumbents e323775b → cc4235d7, with per-horizon skill | Same class. Per-horizon skill is negative at **h1, h23 and h24** in both seeds of both incumbents (see §4, finding R1). | `RESULTS/m04_incumbents.md`; `RESULTS/m04_campaign_tables.json` | same |
| M01 facade, M02 staged pretraining, M05 replay/smoke, M02/M04 R1/R2 plumbing | **Synthetic component checks / plumbing**, not forecasting results | lane return docs | lane suites |
| M03 coverage index and declarations, C07 code review, S07 support derivation | Inventory, review and support derivation. No model performance. | lane return docs | lane suites |

Inputs of the M04 tables: `RESULTS/m04_QUEUE_batch1_v3.from_6ced97e7.json` (the queue export, copied byte for byte from M04's evidence at `6ced97e7`) and `RESULTS/m04_verify_receipts/<cid16>.json` (the 16 verification receipts, copied read-only from the worker that verified each). Their sha256 values are in `m04_campaign_tables.json` under `inputs`.

## 2. Per lane: delivered, tip, acceptance evidence, NOT done

| Lane | Tip(s) | Return document (acceptance evidence) | NOT done, as the lane itself states |
|---|---|---|---|
| M01 model assembly | predictor `satoshi/m01-model-assembly-20260930`: tested `64a91a74`, return `1bb63f51`, remanifest tool `41955f50` | `docs/audits/work_plan/SATOSHI_M01_MODEL_ASSEMBLY_2026_09_30.md` (MS rows, legacy boundaries) | MS09–MS15 not re-audited; no forecasting measurement; `predict_with_uncertainty` returns zeros (Uncertainty/SNR columns meaningless for this plugin); pre-`c1e035d7` donors refused by design |
| M02 pretraining | predictor `satoshi/m02-pretraining-20260930`: return `310a836f`, latest `d1e2e77c` | `docs/audits/work_plan/SATOSHI_M02_PRETRAINING_2026_09_30.md`; `docs/audits/evidence/m02_pretraining_20260930/` | Synthetic results only. ECL donors **in progress** (`m02-ecl-donors-r0-resume1` LIVE on worker_b; core and receipt pending). No multi-seed or matched ablations. |
| M03 feature inventory | feature-eng `satoshi/m03-feature-inventory-20260930` `4ddcce4` | feature-eng `docs/audits/work_plan/SATOSHI_M03_FEATURE_INVENTORY_2026_09_30.md` (coverage index, 4 admissible declarations) | No blanket coverage (columns without a sealed TRAIN contract); no warehouse metrics; selection protocol specified, not executed; Weather `-9999` sentinel policy undeclared |
| M04 DOIN execution | predictor `satoshi/m04-doin-execution-20260930` `6ced97e7` (campaign pin `8f724b12`, v3 CAMPAIGN sha `53bce359`); doin-node same branch `7d207a6` | `docs/audits/work_plan/SATOSHI_M04_DOIN_EXECUTION_2026_09_30.md`; `docs/audits/evidence/m04_doin_20260930/`; tables in §1 | R1/R2 (needs v4 with real donors); per-feature default R0 (COST-01); pretraining cost not attached; two seeds give no interval; no test split; no literature comparison; no warehouse write |
| M05 paper adapter | lts `satoshi/m05-paper-adapter-20260930` `4e88c01`; predictor `a3218b78` | lts `docs/audits/work_plan/SATOSHI_M05_PAPER_ADAPTER_2026_09_30.md` (REPLAY_PASS, SMOKE_PASS_SHADOW_ONLY) | No runner consumes the modular policy; no MT5/IBKR modular smoke; candidate is plumbing, not a finalist; provenance is a local hash binding, not independent review |
| C07 classification remote code | predictor `satoshi/c07-classification-remote-code-20260930` `788082f6` | `docs/audits/work_plan/SATOSHI_C07_CLASSIFICATION_REMOTE_CODE_2026_09_30.md`; `docs/audits/evidence/C07_REMOTE_CODE_20260930/` | No `trust_remote_code=True`, no weights, no model load, no score. **One approval request open** (owner). |
| S07 strategy support | heuristic-strategy `satoshi/s07-strategy-support-20260930` `d08ae00`; predictor `53c6f47a` | `docs/audits/work_plan/SATOSHI_S07_STRATEGY_SUPPORT_2026_09_30.md` (predictor) | No scoring by order; synthetic-arm trade support predeclared, not derived; reserved window **not a clean hold-out** (owner decision) |
| M06 evidence and resources | this branch (tip in the handback); admission fix `satoshi/m06-admission-dead-cache-20260930` `0928dc06` | `docs/audits/work_plan/SATOSHI_M06_EVIDENCE_RESOURCES_2026_09_30.md`; this directory | Slab cache on worker_a not named (needs root); admission fix not deployed (owner); dead cache not reclaimed (owner) |

## 3. Incidents

| When (Z) | Where | What | Status |
|---|---|---|---|
| 09-22 → 09-30 22:53 | worker_a | Unreclaimable slab grew to 5.48 GiB; driver host-allocation failures (NV_ERR_NO_MEMORY). The owner rebooted at his own terminal. | Cleared, **not diagnosed**. Persistence-mode hypothesis UNVERIFIED. Writer samples slab every cycle (`worker_a_slab_series.jsonl`). |
| 09-30 23:31 | coordinator | Host memory PSI stayed above 37.5 for 20 s or more after queued CPU suites admitted. The monitor stopped `m06-status-writer` (a bystander) and `m01-venv`. | Resolved. CPU jobs above 1G steered to worker_b. Admission gates only at entry. |
| 10-01 00:09:56 | worker_b | M02 `m02-ecl-donors-r0` stopped by `PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND` (tree peak 1.28 GB of a 4 GiB cap, so a bystander). C07's `c07-closure-install` was stopped by the same rule 3 s later (tree peak 2.08 GB, tmpfs-charged venv). M02 resumed as `-resume1`. | **Cause UNVERIFIED.** The two stops are simultaneous in the incident records, but which load drove the host pressure is not established. |
| 09-30 23:48 | worker_b | **COST-01.** The per-feature default R0 (321 branches) exceeds 6.44 GB in first-step trace/compile with zero updates. | Its 8 candidates are held; needs an 8G solo slot. |
| 09-30 | both workers | **v2 nondeterminism.** Cross-process GPU float32 rescoring is not exact without deterministic kernels; v2 is REFUTED and retained as `superseded_nondeterministic`. | v3 reran with `TF_DETERMINISTIC_OPS=1`; 16/16 exact. |
| 09-30 | worker_a, worker_b | **ADM-DEADCACHE-01.** Admission charges dead clean page cache left by finished scopes as live use (it kept the M04 pilot queued on worker_a). | Fix accepted by the orchestrator, **not deployed** (owner, `DEPLOY.md`). |
| 09-30 | worker_b | **tmpfs scratch.** `/tmp` is tmpfs and charged to the job's cgroup. C07's throwaway venv (about 1.74 GB) was charged as job memory and one install was pressure-stopped. | C07 removed its own tmpfs files; rule: scratch goes under `~/.local/state`, not `/tmp`. |
| 09-30 19:34–19:52 host time | worker_b | **OLAP outbox pollution.** M01's legacy-CLI smokes wrote 11 envelopes into the shared `olap_outbox/pending/`. | Quarantined (moved, not deleted). Fix `342f1fc5` points `CRISPDM_OLAP_OUTBOX` at each smoke's own directory. The legacy CLI still emits by default (owner-visible finding in M01's return). |
| 09-30 | this branch | **Host-name leaks** in pushed history: `af215de3` (a staging path) and `1a80f5f6` (the coordinator clock-restore unit name). | Fixed forward. The force-push to scrub them was denied by the permission check, so it is an owner decision. |

## 4. M06 review findings on lane deliveries

- **R1 (M04 per-horizon caveat is incomplete).** M04's return names negative skill at h1 and h24. The generated incumbent table shows **h23 is also negative** in both seeds of both incumbents (−0.07 to −0.13). Persistence wins at the next hour and at the daily lag neighbourhood (h23, h24). The aggregate skill hides all three.
- **R2.** The M04 tables are VALIDATION only and NOT_COMPARABLE. No incumbent may be quoted against the published ECL rows (L96→H96, test).
- **R3.** Every verified candidate's receipt reproduces the queue's objective exactly and binds the same model digest. The generator refuses otherwise, and it passed on the real inputs.

## 5. Owner decisions still open

1. **Dead cache.** Deploy the accepted admission fix (`DEPLOY.md`, one paste, with rollback) or run a one-time reclaim on worker_a and worker_b.
2. **worker_a slab.** Enable persistence mode and add a root `/proc/slabinfo` snapshot timer, so the next growth names its cache.
3. **Force-push** of `af215de3` and `1a80f5f6` on `satoshi/m06-evidence-resources-20260930` (host-name leaks; never master).
4. **C07 approval request:** a research-only BANKING77 comparison, as specified in C07's return §6.
5. **S07 hold-out decision:** the reserved window is not a clean hold-out.

## 6. Live state

- `STATUS.json` (`modular.program.status.v1`) is written by `tools/m06_status_writer.py` as `m06-status-writer.service`, through `crispdm-run -m 512M`, restart-on-failure, with no LLM in the loop.
- All lanes are RETURNED. The live job is `m02-ecl-donors-r0-resume1` on worker_b, with heartbeat every epoch.
- `PROGRESS.png` is rendered from `STATUS.json`.
- Next, in order: the M02 donors finish, then M04 v4 (R1/R2) with M01's re-manifest; then COST-01's 8G solo slot.
