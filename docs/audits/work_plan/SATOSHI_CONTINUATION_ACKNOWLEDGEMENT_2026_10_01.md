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

### 9.1 Addendum 14:00Z: lane B's coverage return and four more owner questions

Lane B returned the source entitlement-to-use matrix and the transform-family ledger (feature-eng
`satoshi/b-source-transform-coverage-20261001` `8215886`; DAG v3 at financial-data `satoshi/b-feature-dag-v3-20261001`
`42bed364`, extending v2 in a worktree, live lake checkout untouched). Verified by me. Both denominators stand side by
side: the old 15,228 / 3,538 index grain, and the new discovery grain of 5,273 files (773 raw, 4,500 derived) of which
3,593 lie outside the census. Twenty-two providers; no source is point-in-time admissible and nothing is evaluated or
selected. The ledger has 77 rows: 20 deferred, 16 excluded, 15 profiled, 13 not applicable with justification, 6
materialized, 4 applicable, 3 temporally verified, 0 evaluated, 0 selected. Native wavelet is not covered (the 200
files are a proxy); Hilbert and multitaper are excluded on timing; EMD on an undeclared backend; the regime detector
until fold-bound. A claim of complete coverage is refused.

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 14 | Yahoo Finance: profiled through yfinance, but the paid product and its terms for programmatic access are undocumented | owner | state the actual product and supported access method, or mark Yahoo as research-only via the public connector |
| 15 | Alpaca: only an execution connector (paper) exists; no data plan on file, no retained bytes | owner | state whether a data plan exists and its entitlement, or mark Alpaca data NOT_PRESENT |
| 16 | CFTC positioning and News: declared sources with no attributed bytes (path names only) | owner | declare their scope and connector, or mark them deferred with reason |
| 17 | Availability activation: the eight FXMacroData derived resamples stay UNAVAILABLE because their producer is not located; raw-parent availability is superseded by the data-gov registry | owner / lane B | locate or declare the resample producer; activation stays the owner's |

## 10. Acknowledgement of the next-dispatch orders `b327b771` (received 16:05Z, acknowledged 16:15Z)

Read in full: `docs/handoffs/SATOSHI_NEXT_DISPATCH_2026_10_01.md` and
`docs/tres_temas_entrevista/program_v3/RL_TEMPORAL_COMPARISON_WORK_PLAN_2026_10_01.md`. Accepted as a delta, not a
restart; nothing completed is repeated; running fits continue.

Assignments dispatched 16:10Z (each agent's acknowledgement pending):

| Owner | Order | Immediate action |
|---|---|---|
| M06 | §2: deploy the reviewed admission repair `775c5545` on worker_a only (never `0928dc06`), retained one-paste procedure with rollback; read back hashes, accounting, reservations; bounded smoke; then ONE real CUDA operation on the 5090 UUID in the pinned environment, retaining any error verbatim; regenerate PNG/JSON with three denominators; plan revision `b327b771` | executing; if its permission check refuses the remote write it stops and I hand the owner the one-paste |
| M05 | §5: the naive eligibility gate for heuristic-strategy consumption, eight required tests red first, failure proves zero strategy invocations at the actual runner; coordinates lane G | implementing on its lts branch |
| M04 | §3: finish the finite corrected queue (8 per-feature, 12 R1/R2 as donors arrive); after the repair, place the next eligible pilot/candidate on the 5090 if its measured demand fits; §5 CPU diagnostic of the horizon pattern without retuning; frozen forecast evidence format for the gate | continuing |
| M01 | §3: representation-cost pilot with the honest 3G cap and a per-record heartbeat; §4.6 provenance fields with a versioned migration; encoder export spec for lane G | continuing; pilot resumes after M04's per-feature pilot |
| M03 + C | §3: source/availability reconciliation, native/proxy repair, regime audit, PS2/PS3; §4.3/4.5/4.8/4.10/4.11/4.12; a frozen SELECTED_FEATURE_MANIFEST for the ETH task when its point-in-time resources are readable | continuing |
| Lane G (new, RL owner, under M05) | §6 and the RL subplan: inventory of installed SAC/DQN and env plugins, RL01–RL08 red tests, adapters/configs/accounting/monitoring for RL-S0/S1/D0/D1, cost-pilot plan gated on a frozen selection and readable lake resources; no real-data fit before that | dispatched; agent id recorded when it acknowledges |

Resolved by me under §4 without an owner loop: the RL owner is lane G under M05; the causal home is the maintained
provider package; the strict-minimum incumbent rule stays with full precision and paired uncertainty; `[2,2,1]` is the
pinned default with declared variants; financial-only criteria are NOT_APPLICABLE on author-protocol TSL tasks; FS08/09
shown as deferred optional criteria; purges by target support and by versioned trade support; splits preserved per
dataset; cross-lane owners FS16 B+A, FS19 B+F, FS18 D+B.

### 10.1 The one consolidated request for genuinely external facts

1. Yahoo Finance: the actual paid product and its supported programmatic access method (today profiled only through
   the public connector).
2. Alpaca: whether a market-data plan exists and its entitlement (today an execution connector only, no retained bytes).
3. C07: execution of the pinned remote classification code under its CC BY-NC 4.0 licence, per the exact request in
   `SATOSHI_C07_CLASSIFICATION_REMOTE_CODE_2026_09_30.md` §6.
4. A new reserved confirmation period for the strategy (the read reserve is development evidence), and the holding-period source.

Nothing else in sections 8–9 requires an owner decision; the remaining items there are implementation tasks now owned.

### 10.2 Addendum 16:40Z: question 18, the secondary worker's gate

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 18 | worker_b still runs the old admission gate, which charges its dead clean page cache (about 4 GB) plus the 7G donor reservation; M04's per-feature pilot (6.74 GB) is queued on SLICE_AGGREGATE_BUDGET and M01's 4G suite and 3G PS3-R child will queue the same way until the donor run ends, hours away. The orders authorize the repair on worker_a only and forbid extending it automatically. | owner | extend the reviewed repair `775c5545` to worker_b with the same one-paste (M06 prepares it, labelled NOT executed), or accept the wait |

### 10.3 Addendum 16:55Z: question 19, the ETH 4h availability contract (blocks the RL pilots' frozen manifest)

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 19 | Lane B cannot freeze the ETH 4h selected-feature manifest (draft `SELECTED_FEATURE_MANIFEST.eth_4h.v0-DRAFT.json`, sha `22ae7730`, variant A = 83 admissible features as cost-pilot/control, variant B = 58 screen-tier features) because: B1 ETH 4h has no availability contract (candidate `998e3f80` was never installed); B2 the selected columns' lake parents are outside the census and their producer is not temporally verified; B3 the lake's C127 contract for ethusdt 4h cuts 0.6/0.2/0.2 by rows, conflicting with the task's calendar split, so a successor contract is needed; B4 (variant B only) the inner-validation freeze has not run. Lane G's real-data RL pilots start only from a frozen manifest. | owner for B1 (activation); lane B for B2–B3 (successor contract drafted, not activated; producer verification with lane C) | activate the ETH 4h availability contract; lane B prepares the successor split contract now |

### 10.4 Addendum 17:55Z: worker_a repair deployed and verified; question 18 downgraded

- The reviewed admission repair `775c5545` is deployed on worker_a only (M06 tip `c91b91ca`; verified by me on the
  host: `crispdm-run` 056e207a…, `crispdm_admission.py` 7882d20f…, rollback copies 499fdc18/8dc2c03b retained; slice
  in use 2.66 GB, charged 73 MB, host free for new work 9.0 GB, six own-scope reclaim events in the ledger). Correction
  to my earlier note: the launcher bytes did change (499fdc18 → 056e207a); only `0928dc06` had the same launcher.
- One real CUDA operation on the RTX 5090 (GPU-a9f35631) succeeded in the pinned environment under the repaired
  gate: a 1024×1024 ones matmul on /GPU:0, checksum exactly 1,073,741,824.0, device peak 8,411,136 B, compute
  capability 12.0, zero NV_ERR_NO_MEMORY lines around the probe. One small operation; not proof of health under load.
- The per-feature pilot is ordered onto worker_a; its worker_b request is cancelled through the admission tool.
- Question 18 is downgraded: M06's arithmetic shows that even with the repair worker_b could not admit the 6.74 GB
  pilot beside the 7G donor lease (154 MB short), so extending the repair there changes nothing for lane D; it would
  only let M01's 4G suite and 3G child co-exist with the donor run, which they can already do one at a time. The
  NOT-executed worker_b one-paste stays in DEPLOY.md for whenever the owner wants it; nothing critical waits on it.

### 10.5 Addendum 20:15Z: second coordinator pressure incident and question 20

- Incident COORD-PSI-02 (owner report: gnome-settings-daemon crashed): PSI some avg60 19.6, swap half used, kernel
  OOM kills inside crispdm scopes at 23:22:28 and 23:28:08 local (lane G's green run in its own 1G cgroup; a batch-slice
  kill). Lane jobs live on the coordinator at that time: lane G's Keras parity (1.0 GB RSS) and lane B's census hashing
  (about 74 GB read, 1G-capped children). The earlier "≤ 1G on the coordinator" rule was insufficient. Standing rule
  now: ZERO batch jobs on the coordinator (only M06's 512M writer and 128M watcher and the 512M campaign runners);
  all lane CPU work on worker_a (repaired gate) or worker_b. Acknowledged by every lane.
- Structural fact established by lane B: the lake's data files exist ONLY on the coordinator's disk (the live
  financial-data lake root); both workers' checkouts hold zero data files; there is no mount or rsync path. Every
  data-bound CPU job (profiles, hashing, RL pilot inputs, materializations) therefore reads the owner's desktop disk.

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 20 | Data locality: the lake lives only on the coordinator (owner's desktop); every lake read lands there | owner | approve a read-only mirror of the lake root on worker_b (680 GB free; the lake measured from metadata only by M06, tip 9cb07b8b: 5,273 catalogued files ≈ 94.0 GB, of which raw 773 files 4.51 GB and derived 4,500 files 89.50 GB; all data directories 8,499 files 94.36 GB, 91.7 GB of it under features/; a mirror would leave worker_b about 586 GB free; the coordinator itself has only 89.9 GB free of 762.6), synchronized by a bounded, ionice'd job only while the owner is away; until then, lake reads on the coordinator require Satoshi's explicit approval at 256M, one file at a time, ionice -c3 |

### 10.6 Addendum 07:10Z (2026-10-01): corrected donors bound-ready, incumbent seq 5, strategy gates, two incidents, questions 21–23

- **Corrected ECL donors (lane A).** M02's run under engine 3ecdb256 finished on worker_b without incident: 321/321
  branch donors + core (1,850,448 B, sha 443ed684…), index 8a6bb203…, receipt 946c252f…; CPU 7,875 s over three processes,
  peak 6.77 GB of a 7G cap (page cache of the 9.03 GB fused memmap charged); next run declares ≥ 8G and child-process mode.
  Reconstruction error is not forecasting skill (core relative MSE 0.441). The donors carried schema-1 sidecars and
  therefore read UNKNOWN under bundle v2; M01's versioned migration wrote alongside schema-2 sidecars (OPERATIONAL /
  TRAIN_ONLY, rule R-AE-TRAINONLY-1 from M02's own records, originals byte-unchanged, my spot check 39/39) as
  DONOR_INDEX amendment 1 (2aaba33d…), countersigned by M02 (869afc6d) and M04.
- **Engine hole found by M04 and closed by M01.** The regime path loaded donors without requiring a contract, so the
  engine never enforced OPERATIONAL for R1/R2. Fixed at 6c15af13 (config `donor_contract`, absent = OPERATIONAL,
  `UNKNOWN_ALLOWED` the only bypass, recorded in bundle.json, digest-neutral when absent). Lane A suite 167/2/0 at
  a771e37e. Because the engine package differs from the campaign pin 3ecdb256, R1/R2 open ONLY after M04's bitwise
  graph/forward identity proof at 6c15af13 (CPU, worker_b) and campaign amendment 5; the engine is frozen meanwhile.
- **Incumbent seq 5 (verified by me on worker_a).** Corrected default per-feature R0 with MAE loss, 810,072 params:
  8c2a2eab 0.3884455575 / 103303b1 0.3786844265, paired mean 0.3835649920 (spread 0.0098) vs 9cc0a1f4 0.3910575164,
  Δ −0.0075. STRICT_MINIMUM_SELECTION, not a demonstrated advantage (seed spreads of the gap's size). Unchanged: VALIDATION
  L24/H1..24, NOT_COMPARABLE to published ECL, persistence 0.851406, seasonal 24 h naive 0.247966 beats every candidate at
  all 24 horizons, negative skill at h1/h23/h24. Campaign 20/36 verified, 4 per-feature cells left; verify cap amended to
  3914M = 1.25× measured.
- **Strategy gates (owner rule b327b771 §5), all red-first, mutant-killed, zero invocations at the real entry points.**
  lts a30a2b9/f93832d/8eb1ba1 (heartbeat cycle); heuristic-strategy e6431e6 (pipeline + app/main.py) and 71fa1a4 (WFO
  entry + per fold, phases B/C/D, sweep_noise, API plugin, oracle runner flag-gated and stamped
  DIAGNOSTIC_ORACLE_NOT_A_STRATEGY_RESULT). Every live asset stays SKIPPED until M04 issues a real financial evidence record.
- **Incident S09-MUT-01.** A gate-disabling mutant, untested beyond the gate, launched real phase-D NEAT training on
  worker_b for 15 min (2G cap, wall-stopped) and modified three tracked files in that host's predictor clone; backed up,
  restored; rule issued: never checkout/reset/clean in a foreign checkout. Collateral hazard fixed: sweep_noise did an
  HTTP POST at import time.
- **Rulings by me, for the owner's veto.** (a) The lane D runner launcher (512M, ssh/poll only, peak ~19 MB) is EXEMPT
  from the zero-batch rule as orchestration, with a 64 MiB alert; reason: workers cannot reach each other and worker_a's
  RAM is scarcer. (b) M06 self-disclosed two 384M PNG renders on the coordinator after the rule; generators now run on
  worker_b.
- **Lane G** returned (agent-multi 02db0701, 42/42, Keras→torch parity receipt, parameters measured from built models),
  stopped until lane B's frozen manifest and question 19. Lane B: the ETH 4h view resolves as a git-pinned reference on
  worker_b but is not a governed lake resource; freeze still blocked by B1 (question 19) and B2.

| # | Object | Owner | Minimal action |
|---|---|---|---|
| 21 | Direction classifiers (phases B/C/D feed direction probabilities, which have no form in the forecast-vs-naive contract, so those runners always SKIP) | owner | rule whether a classification baseline (e.g. majority class / persistence of sign per horizon, strictly beaten) admits them, or keep them excluded |
| 22 | `regime_wfo` declares it consumes no learned predictions → currently allowed as NOT_APPLICABLE_NO_LEARNED_PREDICTIONS with the declaration recorded | owner | confirm or require the gate anyway |
| 23 | API-mode runs (per-tick predictions cannot bind to declared horizons) → always SKIP | owner | confirm, or define a per-tick evidence form |

### 10.7 Addendum 09:45Z (2026-10-01): corrected campaign CLOSED (36/36 terminal), final incumbent seq 6

- **Terminal state (verified by me on origin f787ad51 and on both workers' receipts):** 32 VERIFIED by exact match + 4
  REFUSED_BY_ENGINE (core-only R1/R2: a core donor binds its upstream branch weights, by design; named inside the 36,
  never run). Pin aaee6f94 for R1/R2 (engine 6c15af13 proven bitwise identical to 3ecdb256 on 18 configs + replay),
  R0 rows under df9ae31c. Donors bound from index 8a6bb203 + amendment 2aaba33d, OPERATIONAL verified before build in
  every receipt (read by me).
- **R1/R2 vs R0 (seq 5 c09f3034, 0.3835650, R0 spread 0.00976), paired by seed, validation MAE in z_train:**

| regime | mean | mean Δ vs R0 | own spread | reading |
|---|---|---|---|---|
| branch+core R1 (both frozen) c950ec17 | **0.3750913** | −0.0084737 | 0.00002 | STRICT_MINIMUM improvement, within the R0 spread → NOT an advantage |
| branch R1 (frozen branches, core R0) | 0.3905652 | +0.0070002 | 0.00287 | worse, within spread |
| branch R2 (fine-tuned) | 0.3976313 | +0.0140663 | 0.01277 | WORSE on both seeds, beyond both spreads |
| branch+core R2 (fine-tuned) | 0.4011443 | +0.0175793 | 0.00403 | WORSE on both seeds, beyond both spreads |

  Fine-tuning from these donors at lr 1e-3 hurt; freezing both gave the strict-minimum best. **Final incumbent seq 6 =
  c950ec17 at 0.37509125759139983**, strict minimum, not advantage. Pretraining cost beside the row, outside the rule:
  CPU 7,875.48 s, wall 6,636.5 s, peak 6.77 GB.
- **Closure table (owner rule), top rows; all 16 verified configurations in the evidence file.** Scale: z_train
  train-only scaler, validation mean over 2609 windows × 24 horizons × 321 channels. Same-row persistence naive
  0.851406. Literature value NOT_AVAILABLE (no published ECL L24/H1..24 validation row; published rows are L96/H96–720
  on test; ECL_TIMEFILTER_MATCH_20260930.md). Comparability NOT_COMPARABLE for that reason.

| configuration | MAE | skill vs persistence |
|---|---|---|
| c950ec17 branch+core R1 | 0.375091 | 0.5594 |
| c09f3034 default per-feature R0 MAE | 0.383565 | 0.5495 |
| branch R1 | 0.390565 | 0.5413 |
| 9cc0a1f4 draw3 R0 MAE | 0.391058 | 0.5407 |
| 554ff1d6 draw1 R0 Huber | 0.391705 | 0.5399 |

- **Controls that decide the reading:** the 24 h seasonal naive (0.247966) beats every one of the 32 verified cells at
  every horizon; skill vs persistence is negative at h1/h23/h24 in 32/32. Under the owner's gate none of these forecasts
  may reach the heuristic strategy, and none is financial evidence in any case.
- **Not done:** ablations (proposed, not run); no test split; no learning-rate sweep for R2; no financial evidence record.
- **Deviations/incidents in the closure evidence:** D-IDPROOF-STOP-01 (undersized cap, bystander stop of PS3-R),
  ADM-PROC-01 (two SIGTERMs to queued acquirers with no lease, because the admission tool has no cancel verb and
  ADM-CANCEL-01 is undeployed on the secondary worker), Q18 backlog file (old gate refused 6580M for 50+ min with
  ~17 GB available).

### 10.8 Acknowledgement 10:35Z (2026-10-01): owner's full-autonomy order and the parallel dispatch (master plan, per front)

Received: the owner's chat order (intent: no authorization waits from the owner or Musashi; never ask the owner; advance on
all branches, above all code and tests of the branch+core architecture; selection in parallel with defaults and the RL
comparison; no GPU ever idle, the 5090 first) and the written orders `docs/handoffs/SATOSHI_AUTONOMOUS_PARALLEL_EXECUTION_2026_10_01.md`
(sha256 eef2ca9b19ed5198…, copied here unchanged). Every "owner decision open" in this document is now ruled by me and recorded
as an assumption: Q13 EURUSD appearance = DEVELOPMENT; Q14–17 external facts = use what the held service key exposes,
exclude only inputs whose availability time cannot be established; Q18 = deploy the admission repairs on the secondary
worker (M06, with rollback); Q19 = DEVELOPMENT availability from the git-pinned ETH 4h view (predictor b1f8a74f, sha
1b447c66…); Q20 = a read-only lake mirror on the secondary worker is approved when a lane needs it; Q21–23 = current gate
behaviour as coordinator rulings; C07 = proceed as filler when a GPU would otherwise idle. Musashi audits in parallel; his
review is not a dispatch dependency.

**Live at 10:30Z (adopted by identity, not restarted):** 5090 (GPU-a9f35631) at 52 % with lane D v2 cell 3b5f18b4
(6580M cap); 4090 (GPU-a8bd1b2c) idle until M04 places a v2 cell there and M07's first pilot admits; 5070 Ti
(GPU-b77fc3ad) idle until lane G's first RL cell; M01's PS3-R child at 3G on the secondary worker (125/240 records
retained, resumed); M06 writer and watcher units; the LTS campaign supervisor and MT5 services untouched.

| Front | Responsible (agent id) | Branch(es) | Next action and first deliverable | ETA basis |
|---|---|---|---|---|
| A engine | M01 a5b79d5488abdee3a | predictor satoshi/a-engine-integration-20261001 (a771e37e) | seasonal-residual target option and R3 warm regime, tests first → tips to M04; PS3-R contrast; model card for seq 6 | next measurement when each suite runs (≈1 h per item) |
| B data/selection | M03 a160efdffffffa9f2 | feature-eng satoshi/b-* (be2766b, 1b22c64) | ETH 4h variant A FROZEN_DEVELOPMENT within 60 min → sha to G/C2/M07; EURUSD next; Yahoo/Alpaca/FXMacroData real inventory; ledger denominators | first sha ≤11:35Z |
| C causality/extraction | C2 a5fd5cd9bcbc7f3bd (new) | causal-inference / feature-extractor / predictor satoshi/c2-causal-eth-20261001 | per-feature incremental utility with causal-order and stability checks on M07's train split; dossier v1 with failing controls; ranking to M03/M07; PS3-R rerun on ETH | first table after M07's split sha |
| D DOIN/training | M04 a083979bc9a8fa1b8 | predictor satoshi/d-corrected-queue-20261001 (e3330b9d→v2) | campaign v2 on the 5090 (and the 4090 until M07 claims it): seeds 2023/2024 for top-3 → intervals; R2 lr sweep; R3; two ablations; seasonal-residual pair; L96 cost pilot (literature lane, separate) | 622 s per per-feature cell measured; first v2 receipt ≤11:00Z |
| D-financial | M07 afcf115024ffa1381 (new) | predictor satoshi/f2-eth-forecast-20261001 | split sha → CPU cost pilot → R0 (base control + grouped32 + per-feature × Huber/MAE × Adam/AdamW × 2 seeds) on the 4090 → donors → R1/R2 → evidence record v1 for M05 → donors to lane G | cost pilot sets the ETA; its first measurement reported when admitted |
| E trading/product | M05 a65d2f3a115a9aa25 | lts satoshi/m05-paper-adapter-20260930 (2b18e17), heuristic-strategy satoshi/s08-backtest-naive-gate-20261001 (71fa1a4) | RL shadow adapter (≈1.5 h); paired forecasting+heuristic vs RL backtest harness (≈3 h after lane G's spec); PP/LTS integration dormant until a model passes the gate; M5PHET health check | per its estimates |
| F evidence/resources | M06 aa83df3b3fa713744 | predictor satoshi/m06-evidence-resources-20260930 (1cc79478) | first dispatch report ≤20 min with GPU UUID/pid/dataset/stage; 2-min heartbeats; GPU-idle alarm; deploy 775c5545 + stale-lease/cancel on the secondary worker with rollback; STATUS/RESULTS/PROGRESS per §6 | continuous |
| G RL | lane G aa7913479a97b45c0 | agent-multi satoshi/g-rl-temporal-20261001 (02db0701) | freeze (own or M03's) → re-materialize 16 cells → C1/C2 CPU pilots → 16 cells on the 5070 Ti and the 4090 (≤3G each, one per host) | pilot per-step time sets the ETA |

Standing technical limits (not permissions): measured caps never lowered; zero batch work on the coordinator except the
two exempt ≤512M launchers; one memory-heavy job per host beside measured small ones; no desktop OOM; no artificial GPU
work; test splits untouched; DEVELOPMENT labels; paper/demo only under the existing risk mandate; roles not host names.
