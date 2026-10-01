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

## 8. Addendum 05:15Z: independent verification outcome and open owner questions

Independent verification workflow (16 agents, four readers plus two adversarial refuters per load-bearing claim;
nine claims confirmed by code reading and re-execution on worker_b, two refuted):

- `da4ce7b4` builds the required hourly default in real Keras 3.13.2: branch `(None,24,16)`, fused `(None,24,16F)`,
  positional encoding then per-step projection to 64, two causal attention blocks with FFN and residual norms, residual
  Conv1D stages `(None,12,32) -> (None,6,16) -> (None,6,8)` with valid-padded strided skip and block projections, the only
  Flatten in the forecast head, `FeatureSelect` a weightless `tf.gather(axis=-1)`; `branch_steps != window` and
  `time_factors [2,1,1]` are rejected. Three findings forwarded to lane A: a stale docstring (`config.py:88`), one legacy
  test failing on the moved `entry_points` symbol (1 failed / 59 passed), and the three-stage grid being a default rather
  than an invariant (`[4,1,1]` is accepted; factors are candidate parameters by the plan).
- Old-design inventory: v3's 16 verified candidates split into 4 with 12-step branches and 12 with 24-step branches but
  non-residual core factors on the compress engine; the current incumbent belongs to the second class. Unstarted rows
  in the v1, v2 and v3 queues are being marked `CANCELLED_SUPERSEDED` in place by lane D; finished rows stay as old evidence.

Open owner questions raised by the FS01-FS20 mapping (object, owner, minimal action). Each is treated as a stated
assumption by the lanes until ruled:

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 1 | FS13/FS14 name an RL/SAC comparison arm that is none of lanes A-F | owner | name the RL owner or defer FS13/FS14 |
| 2 | Home of the PS3-C study code: the causal-inference repo is a legacy fork with an inference-only ATE provider | owner / lane C | confirm the provider package as the home; lane C records what of the fork is not used |
| 3 | PS3-C needs a concrete episode manifest; 6,555 financial columns lack sealed TRAIN contracts | owner / lanes B+C | lane B sends lane C the contracted financial resources; the rest are NOT_ADMISSIBLE_NO_CONTRACT |
| 4 | FS07 (small improvements preserved) versus the live strict-minimum incumbent rule | owner / lane D | decide whether incumbency keeps a tolerance band or stays strict minimum |
| 5 | FS15 denominator: rows (15,256 / 15,228 distinct) versus rows x metric families; 28 superseded d4 rows | owner / lane B | choose the acceptance denominator; lane B reports both meanwhile |
| 6 | FS03/FS17 need new contract fields in `predictor.modular.v1` (lane A) and the lts inference contract (lane E) | owner | authorize the schema field additions (OPERATIONAL/SYNTHETIC_OFFLINE, corpus provenance, reconstruction NOT_APPLICABLE) |
| 7 | FS04 oracle: literal `[2,2,1]` versus a derived default that also accepts other exact partitions | owner | confirm "default pinned by test, factors remain candidate parameters" |
| 8 | FS02 scope: Y_s/Y_l/Y_b are financial targets; TSL lanes forecast all channels | owner | confirm NOT_APPLICABLE for TSL author-protocol lanes |
| 9 | Reporting denominator "/20" includes FS08/FS09, which belong to optional PS7 | owner | state whether optional criteria count in the denominator |
| 10 | FS13 holding period source for the purge: S07's variant-E config or the target plugin; S07's reserve is not a clean hold-out | owner | rule the hold-out (new prospective window or declared development) and the holding-period source |
| 11 | FS01 fold definition: TSL 7/1/2 prefix versus 4y/1y/1y with inner chronological folds; Weather physical time UNSUPPORTED | owner / lane B | confirm the fold rule per dataset family |
| 12 | Cross-lane dependencies absent from the lane table: FS16 M03+M01, FS19 M03+M06, FS18 M04 needs lane B's PS2 batch | owner | acknowledge; lanes name the other half as a dependency, not as covered |

### 8.1 Addendum 06:30Z: question 13

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 13 | PS3-C has no contracted price source for its asset: none of the 198 sealed financial appearances covers EURUSD, and subplan 5.1 forbids swapping the asset; the only contracted episode sources are the FXMacroData announcements and the release calendar; the 2011-2021 archive, FRED proxies and uncontracted EURUSD bars are NOT_ADMISSIBLE_NO_CONTRACT | owner | seal a EURUSD price appearance through the governed route; lane B prepares the sealing request (source sha, split rule, identity fields, activation path) without activating it; PS3-C is specified and bound, NOT_EXECUTABLE until then |

### 8.2 Addendum 07:20Z: the minimal action for question 13 is prepared

Lane B wrote the EURUSD appearance sealing request (feature-eng `satoshi/b-selection-ps0-ps2-20261001` `5d12e88`,
`docs/feature_metrics/laneB/EURUSD_APPEARANCE_SEALING_REQUEST.v1.{md,json}`, JSON sha `fddb8ceb`). Nothing is sealed,
activated or written to the lake, data-gov or any config. Candidate: census appearance `app_ae9142c201e6694a3b1e1fde`
(eurusd 5m parquet, sha `c746f344`, 1,552,028 rows, 2005-01-03..2025-12-31; omitted from C127 by byte budget, not
refused; 15m/1h/4h siblings listed). Split constraint: the contract module accepts only contiguous blocks from row 0,
so the recommended option A is train 2005-2023 / calibration 2024 / confirmation 2025 with the recipe's 2020-2023 as
the modelling window inside train, older rows usable only under a pretraining declaration, three expanding inner
folds with 15 % validation and a 2,016-bar purge. Owner steps, none executed: rule on the split option; seal into a
new contract file (a calendar-cut helper does not exist yet and lane B writes it after the ruling); save the pending
resource_contracts entry; promote and restart only the financial lake service as on 2026-09-14; register in data-gov;
verify a VERIFIED_TRANSFER delivery.

### 8.3 Addendum 11:25Z: third leaked commit on lane F's branch; campaign state

- Lane F reports a third host-name leak in its own branch history: `ffed65e7` pushed a STATUS.json carrying a host name
  taken from a compiler temp-file path in a child process's command line; fixed forward at `d032d9cb` (the writer now
  scrubs every host alias and the local host name from every string; the pre-commit grep aborts instead of printing).
  The owner's force-push decision now covers three commits on that branch: `af215de3`, `1a80f5f6`, `ffed65e7`. Not on master.
- Corrected campaign (amendment 1, CAMPAIGN sha `12f6030c`, pin `df9ae31c`): 16 grouped/draw R0 cells live on worker_a
  (first verified pair pending); 8 per-feature cells await their pilot on worker_b; 12 R1/R2 await donors; the corrected
  donor run is at branch 47 of 321 under a 7G cap after a planned stop and resume (finding MEM-02). No scientific score yet.

## 9. Acknowledgement of the coverage addendum `256c61a6` (received 12:05Z, acknowledged 12:15Z)

Read in full: `docs/handoffs/SATOSHI_SOURCE_TRANSFORM_COVERAGE_ADDENDUM_2026_10_01.md` and the diffs it made to the
continuation orders and the subplan. Accepted as binding: the catalogue is exhaustive relative to a declared finite
discovery snapshot; the search is progressive; a claim of complete coverage is refused while a known source or family
is absent from the accounting; catalogue coverage, model-input count and evaluated-candidate count are reported apart.

The six findings are accepted as stated; none of my earlier coverage wording survives them: 3,538/15,228 is index
coverage, not source completeness; five datasets are not a provider-by-entitlement census; the 1,400 engineered
parquet files are outside demonstrated selection coverage; `compute_wavelet` is a multiscale rolling-filter proxy;
the spectral producers carry length-dependent windows and unit-less frequencies; the legacy regime detector's
fitting period and forward-informed labels are a provenance/leakage risk.

Assignments (dispatched 12:10Z; each agent's acknowledgement pending; running fits untouched):

| Lane | Owner | Task from the addendum | First deliverable |
|---|---|---|---|
| B | M03 | source entitlement-to-use matrix (Yahoo, Alpaca, FXMacroData, existing macro/indices/commodities/crypto/positioning/news), transform-family ledger per input type with PROXY/NATIVE, DAG extension of the existing census, old and new denominators, red tests (uncovered row, proxy cannot certify native wavelet, cache reuse vs vintage, budget overflow → named deferred, exploration survives a limited batch) | ack with branch and first command; then the two matrices |
| C | lane C agent | method semantics and temporal tests: wavelet proxy id + native producer spec, Hilbert/multitaper units and length-dependent windows (prefix and chunked-replay invariance, frozen TRAIN windows), regime detector fold-boundary test; card field method_id PROXY/NATIVE | ack; METHOD_SEMANTICS dossier and red tests in the owning repos |
| D | M04 | candidate budget model (measured raw/expanded channels, branches, fused width = sum of branch widths, time, materialization bytes, parameters, RAM/VRAM, weekly cost), caps, overflow → named deferred row, never truncation; tests | after the 24-row batch |
| A | M01 | engine exposes its measured-shape budget per config; build refuses silent collapse/truncation; cards carry recipe identity | after the PS3-R pilot |
| F | M06 | `coverage` block in STATUS with the three separate counts; plan_revision 256c61a6; no "complete coverage" wording | next STATUS write |

Nothing else changes: the corrected DOIN batch, the corrected donor run and the PS3-R pilot continue as approved.
