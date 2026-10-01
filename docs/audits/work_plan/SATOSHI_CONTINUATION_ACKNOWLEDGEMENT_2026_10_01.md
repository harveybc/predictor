# Satoshi acknowledgement: progressive selection and modular continuation

Written for the owner and Musashi. Effective receipt 2026-10-01 02:50Z (chat paste pointing at master
`ac125db9`); this acknowledgement is written at 03:10Z. Signed Satoshi III, successor technical lead.

## 1. Orders read

- `docs/handoffs/SATOSHI_PROGRESSIVE_SELECTION_AND_MODULAR_CONTINUATION_2026_09_30.md` (master `ac125db9`), in full.
- `docs/handoffs/SATOSHI_MODULAR_RECONCILIATION_2026_09_30.md` (`bf96bc15`), in full.
- `MODULAR_STACK_WORK_PLAN_2026_09_30.md` diff `dc72170e..ac125db9` (24-step branches, residual Conv1D core, factors `[2,2,1]`).
- Subplan `FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md`: sections 1-10 skimmed by heading, FS01-FS20 and
  PS0-PS7 tables read; full read delegated to lanes B and C and to the verification workflow.
- `FEATURE_SELECTION_METHOD_STATE.json` (stage `S3_ACCEPTANCE_TEST_DESIGN`), `CURRENT_EXECUTION_2026_09_30.md`.

Accepted as binding: the architecture correction precedes any dependent fit; no v4 R1/R2 via remanifest of
12-step donors; old weights are evidence of the old design; active old work stops at its safe checkpoint boundary.

## 2. Real state at receipt (read by me on the hosts, 02:55Z)

| Role / device | State | Evidence |
|---|---|---|
| worker_a, RTX 5090 | idle, 0 %, 10 MiB, 36 C; no leases; batch slice 2.42 GB of which 2.37 GB clean file (dead cache, ADM-DEADCACHE-01) | `nvidia-smi`, admission store, `memory.stat` |
| worker_a, RTX 5070 Ti | idle, 14 MiB, 27 C (shares host RAM with the 5090: one host, not two) | same |
| worker_b, RTX 4090 | idle, 14 MiB, 37 C; one lease: `m02-ecl-donors-r0-resume1` (CPU); slice 7.77 GB (file 4.26, shmem 0.73) | same |
| coordinator, RTX 4070 | desktop, 33 %, 1,038 MiB; only `m06-status-writer` lease; nothing heavy allowed | same |

**Old donor, real state:** M02's run `m02-ecl-donors-r0-resume1` on worker_b was alive at 181/321 branch records,
stage `branch_ae:branch_180`, heartbeat 15 s old, 4 processes, old design (window 24, branch_steps 12, core factors
`[2,1,1]`). Disposition ordered to M02 at 03:05Z: stop through its normal safe checkpoint boundary (after the
in-progress branch writes its record), retain every record/provenance/heartbeat as `OLD_DESIGN_BRANCH_STEPS_12_SUPERSEDED`,
no in-place edit, no relabel, no old core. M02's confirmation is pending.

**Owner package verified to exist:** `da4ce7b4` on `origin/codex/modular-stack-20260930`, `predictor_plugins/modular_temporal/`
split (`artifacts, assembly, common, components, config, layers` with `FeatureSelect` and `PositionalEncoding`,
`pretraining, registry`); 20 files, +1030/-585 against `556c5f3e`. Its real-Keras shapes are being verified by the
workflow below; not yet certified by me.

## 3. Lanes, agents and tasks accepted (dispatched 03:05Z; each agent's own acknowledgement is PENDING)

| Lane | Agent(s) | Branch | Immediate task | ETA basis |
|---|---|---|---|---|
| A engine + donors | M01 (`a5b79d5488abdee3a`), M02 (`a5289d234c296e0c5`) | `satoshi/a-engine-integration-20261001` off `da4ce7b4` | integrate da4ce7b4 as the single engine; port M01's component versions, effective-params identity, keras pin, facade/entry points, grammar, save/load, remanifest; M02 ports early-stop/materialization/resume after; Keras graph PNG; shape/grid/regime/parity/legacy tests in an installed isolated venv | not profiled: ETA after M01's ack |
| B data + selection | M03 (`a160efdffffffa9f2`) | `satoshi/b-selection-ps0-ps2-20261001` off `4ddcce4` | PS0 denominator reconciliation; PS1 batched metric matrix with per-cell status; PS2 reversible prioritization; FS01/02/15/16/19 tests first | not profiled |
| C causality + bibliography | new agent (id recorded in STATUS when it acknowledges) | `satoshi/c-causal-ps3c-20261001`, `satoshi/c-representations-ps3r-20261001`, `satoshi/c-contracts-20261001` | PS3-C spec (three rungs, FS10/11/12 tests), PS3-R shortlist with official references, consumer contracts | not profiled |
| D optimization | M04 (`a083979bc9a8fa1b8`) | `satoshi/d-corrected-queue-20261001` off `6ced97e7` | corrected queue/search space persisted, not executed; staged cost plan; old-arch results labelled; no fit before A | depends on A |
| E product | M05 (`a65d2f3a115a9aa25`) | continue `satoshi/m05-paper-adapter-20260930` | installed-consumer end-to-end test in shadow/paper; adapter tests against the integrated package | depends on A for the final run |
| F resources + evidence | M06 (`aa83df3b3fa713744`) | continue `satoshi/m06-evidence-resources-20260930` | reconcile leases/processes/queues with parsed argv; lane table and plan revision; ADM-DEADCACHE review against the five test cases; warehouse grain proposal | not profiled |

Parallel independent verification (workflow `wf_6286ece8-903`, four readers + adversarial refutation): da4ce7b4
shapes in real Keras on worker_b; M01 port map; old-design job inventory; FS01-FS20 to lane mapping.

## 4. Planned integration commit and candidate slot

- Integration commit: lane A, branch `satoshi/a-engine-integration-20261001`, off `da4ce7b4`; revision unknown until M01 publishes.
- Candidate slot for the corrected complete pilot: worker_a RTX 5090 (`GPU-a9f35631`), verified loader recipe
  (`LD_LIBRARY_PATH` to the cu12 wheel libraries, env `tensorflow`, Keras 3.13.2), `TF_DETERMINISTIC_OPS=1`,
  `CUDA_CACHE_MAXSIZE=2147483648`, in-child device facts. Admissible only if the measured need of the CORRECTED graph
  fits beside 2.37 GB of dead cache under the 8.59 GB slice ceiling; otherwise worker_b (RAM-limited alternative).
  The cap is declared from a staged CPU profile of the corrected graph, not from COST-01's 8 GiB.

## 5. Allocation remainder and proposed cost

- No numeric ceiling was declared for the modular campaign beyond "the existing allocation"; spend so far (measured by
  the lanes): v2+v3 DOIN attempts 27 train + 25 verify attempts of 1-4 min each on two GPUs; M02's old-design donor run
  about 3.5 h CPU on worker_b (to be reported exactly by M02 at the stop); S07's strategy ceiling remainder 2112.607 CPU s.
- Proposed cost of the next deliverables: lane A tests and smoke, CPU only, <=4G, no fit; lane B profiling in batches
  under 2G on worker_b; lane C CPU inspections under 1-2G; the corrected pilot cost is UNMEASURED until the staged
  profile exists, and its first ETA is the time of that profile, not a global date.

## 6. ETA of the next deliverable

- 03:25Z: each lane's acknowledgement with its own worktree, first command and ETA (or the explicit missing measurement).
- 03:40Z: workflow verdicts on da4ce7b4 shapes and the old-design inventory; forwarded to lanes A and D.
- Next 30-minute summary at 03:47 local cron.

## 7. Addendum 03:40Z: every lane has acknowledged with its own worktree, branch, first command and ETA

| Lane | Acknowledged | Worktree / branch | First command | ETA (planning, not profiled) |
|---|---|---|---|---|
| A M01 | 03:12Z | `predictor-a-engine-integration`, `satoshi/a-engine-integration-20261001` off `da4ce7b4` | real-Keras shape proof + graph PNGs first (ordered), then ports | ~3.5 h total |
| A M02 | 03:08Z / stop done 03:25Z | same branch, disjoint files (`pretraining.py`, `tools/modular_pretrain.py`, evaluator early-stop, materialization, resume) | old run stopped at boundary branch 189 (190 records, no partial), `OLD_DESIGN_BRANCH_STEPS_12_SUPERSEDED.json` sha `37cc105a`, CPU 8399.49 s lower bound | port tests to 24-step grids first |
| B M03 | 03:35Z | `feature-eng-b-selection`, `satoshi/b-selection-ps0-ps2-20261001` off `4ddcce4` | FS01/02/15/16/19 tests committed red, then `tools/progressive_selection.py` | tests + first code 60-90 min; PS1 minutes of CPU per dataset |
| C new agent | 03:40Z | `causal-inference-c-ps3c-20261001` off `bb11d64`; `feature-extractor-c-ps3r-20261001` off `cf39e4b`; `predictor-c-contracts-20261001` off `ac125db9` | read-only inspection of the existing local-projections estimator | spec + red FS10/11/12/17 tests + contracts ~2.5 h |
| D M04 | 03:14Z | `predictor-d-corrected-queue-20261001`, `satoshi/d-corrected-queue-20261001` off `6ced97e7` | staged CPU profile of the corrected graph against `da4ce7b4` (PRE_INTEGRATION), corrected queue draft with SUPERSEDED_OLD_ARCH test | profile 20-40 min after code; GPU pilot within 1 h after profile |
| E M05 | 03:30Z | continues `satoshi/m05-paper-adapter-20260930` (lts `4e88c01`, predictor `a3218b78`) | named route tests before any change | runner e2e + receipt 90-120 min |
| F M06 | 03:13Z | continues `satoshi/m06-evidence-resources-20260930` and `satoshi/m06-admission-dead-cache-20260930` | read-only process/lease/cgroup census with parsed argv | STATUS +40 min; ADM review next; ETA/PNG and gov_* proposal after |

Findings already surfaced by the lanes in their first minutes, recorded as they state them:

- M02: resident memory grew with branch count inside one process (about 1.2 GiB at 18 branches, 3.3 GiB at 189 of 321);
  a single-process 321-branch run would likely have breached its 4 GiB cap. The successor frees each branch graph or
  runs branches in bounded child processes, with a memory-growth test.
- M03 correction: the ETH 4h model-ready view does have an immutable TRAIN contract (manifest `14a1077f`, sha `1b447c66`);
  the earlier "no TRAIN boundary" row was wrong. Y_s is constructible only at h=4 there; Y_b is NOT_EVALUATED without a
  versioned TP/SL rule.
- Lane C: the retained event-study counterfactual is MODEL_BASED and does not preserve inferred perturbations (the FS11
  gap); every retained event study is NOT_IDENTIFIED for stated reasons (assumed clock 2011-2021; no consensus 2024-12..2026-05).
- M04: spent about 4,500 GPU-s in v2/v3 attempts and pilots; no numeric grant exists for the lane, so remaining allocation
  cannot be stated; costs are reported as spent and proposed.
