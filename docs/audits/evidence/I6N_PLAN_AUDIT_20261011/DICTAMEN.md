# I6-N plan audit: verdict

Date: 2026-10-11. Auditor: Satoshi (independent of the plan author).
Object: uncommitted draft over HEAD `269c2d13` on `codex/i6d-weekly-walk-forward-20261009`.
All eight document hashes in the request section 2 matched before reading
(proposal `d4341e72...`, master `c0fc85c3...`, market-week contract `604f9480...`,
weekly contract `d64206b8...`, state `ee8a23f2...`, queue `c9f1559f...`,
CURRENT_EXECUTION `bbbb09cc...`, I6-E plan `129cea8c...`). The draft diff against
HEAD is retained by digest in `EVIDENCE_MANIFEST.json`. Nothing in the audited
object was edited; no campaign, service, GPU or broker was started or stopped;
no TEST row, credential or token was read.

**Verdict: REVISE_BEFORE_IMPLEMENTATION.** The business requirements of the
owner's amendment (eligibility by origin and horizon with t+h < B, per-output
masks, paired naive per h, separate S_h and U_h, business before optimization)
are stated correctly in the market-week contract. The experimental design that
should realize them is not yet implementable without improvised decisions and
contains one methodological contradiction (I6N-AUD-01). This verdict does not
certify any software, performance or profitability.

## Conflict of interest

Components produced under Satoshi's lanes are touched by this audit: the FS4
weekly wrapper and campaign that produced the 20-feature manifest
(I6N-AUD-01), `tools/fs4_temporal_predictor.py`, which is part of the sealed
I6-E-M code identity (`i6m_prepare.py:253`), and the FS4 encoder whose
origin-3 lag the proposal cites (I6N-AUD-06). The findings on those
components were established by executed checks of a sub-auditor and by the
repository's own statements, not by self-attestation, and they are reported
against Satoshi's lane. They are not self-signed as verified-clean.

## Method

Read-only, in parallel: A/B business, calendar, strategy and DAG; C/D/E
parameters, DEAP/DOIN and architecture; F/G resources, custody and OLAP. One
integrator. Small CPU probes under `crispdm-run` (1-2 GiB caps) on synthetic
tensors or the retained INPUT `ts` array ending 2024-12-31; their scripts and
outputs are in `probes/`. One read-only resource snapshot per host role. The
market calendar used in executed counts is an ASSUMPTION (Forex Sunday 17:00
to Friday 17:00 America/New_York, offsets 12+12, Thursday 00:00 cut), not the
broker calendar.

Status vocabulary: ESPECIFICADO, AMBIGUO, NO_ESPECIFICADO, NO_IMPLEMENTADO,
CONTRADICTION, VERIFIED, NOT_MEASURED. The absence of the new I6-N executor is
not counted as a defect by itself; missing rules needed to build it are.

## Findings

### I6N-AUD-01 [CRITICAL, CONTRADICTION] (request 3.3/A)

The 20 retained features were selected on VALIDATION 2024 and with rankings that include 2023 labels; N1-N3 screen on 2023 weeks and N4 lets VALIDATION 2024 choose recipes.

- Evidence: Executed set-equality: the 20 members equal the union of the 14 EURUSD FS4 weekly winners, each chosen by mean over all sealed 2024 VALIDATION weeks (state dir fs4/preliminary/wave2_successor/weekly/TEST_FREEZE.json aggregate_rule; WEEKLY_PLAN validation_year 2024); candidate rankings include fold inner_2023 (fs_close FINAL_SELECTION_MANIFEST folds). Repo admits it: I6A_ARCHITECTURE_CAMPAIGN_2026_10_08.md:50-51 'selected using later validation'. Proposal:21 calls them only 'retenidas'. Selection targets include dropped horizons (1,96,120,144 h) and none of 12,18,36,60 h.
- Minimal correction: Either a TRAIN-only manifest whose selection rows end before each fold cutoff (<=2022 for 2023 folds, <=2023 for VALIDATION 2024), or declare all I6-N VALIDATION results RETROSPECTIVE_DEVELOPMENT with feature_selection_used_validation=2024 in identity and make TEST the only prospective evidence.
- Acceptance test: Provenance checker refuses any manifest whose selection rows are dated at or after the cutoff of the fold that uses it; identity carries manifest digest and selection cutoff.
- Owner: Musashi (plan) / Satoshi lane (FS4 weekly manifest producer)

### I6N-AUD-02 [CRITICAL, NO_IMPLEMENTADO] (request 3.1/A)

The production strategy plugin does not implement the business rule; the plan has no DAG node to repair it, yet U_h (real-use score) depends on it.

- Evidence: heuristic-strategy checkout = origin/master 5c87a25 (no pin anywhere in predictor or lts). Early close uses min/max without time order and never looks at TP (policy:141-176); executed counterexample TP@2h/SL@12h vs SL@2h/TP@12h gives identical decisions for all seven variants, buy and sell. Positional Prediction_h_{i+1} mapping (plugin:111-117), lexicographic key sort breaks at >=10 horizons (policy:214), missing column filled with current price (plugin:302-305, forbidden by BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (604f9480):127), no availability masks, no two-long abstention, no calendar. Installed site-packages copy differs and uses short+long for entry.
- Minimal correction: Add a strategy-repair node before any U_h or strategy result: pinned plugin, index->horizon_hours->target_timestamp map, mask consumption, chronological first-crossing SL/TP rule with tie/no-crossing/missing rules, abstention, calendar flat/entry rules.
- Acceptance test: Executed unit tests: TP-first vs SL-first counterexample (buy and sell) yields different decisions; changing shorts with longs fixed leaves entry candidates unchanged; a missing head is never filled; 12 horizons keep numeric order.
- Owner: Owner of heuristic-strategy plugin

### I6N-AUD-03 [CRITICAL, NO_ESPECIFICADO] (request 3.1/C)

Joint search is incomplete: N3 (input transform, output supervision) runs after N2 fixes window/capacity with <=4 configs; L={1,2,4,6}*h ties history to h.

- Evidence: Proposal:115, 145-149. h=2,3 never reach the master's 24 h minimum window (MASTER v3:315); 168 h reachable for no h; 4-config N3 cannot hold transform x supervision cell. Contradicts request 'do not infer 2*h is a law'.
- Minimal correction: Make input transform and output mode N2 genes, or re-evaluate each at >=2 windows (N2 winner and longest); add absolute L values {24,72,168} h where support allows.
- Acceptance test: Sealed N2 domain lists transform/output genes and absolute L; INTERACTION_AUDIT IA-01/06/07/11 reported with paired contrasts.
- Owner: Musashi

### I6N-AUD-04 [CRITICAL, NO_ESPECIFICADO] (request 3.2/C)

Optimizing only h=4 and h=72 and transferring <=2 recipes is not per-horizon optimization, and the recipe rule is undefined for other h.

- Evidence: Proposal:76, 113, 163-164. Admissible delta_x: h=2 {2,1}, h=3 {3,1}, h=5 {5,1}; delta_x=h/2 is fractional for h=3,5; k=9 only for h=18,36,72; absolute L=432 is off-grid for every h<72. h=72 is the least supported horizon (I6N-AUD-08).
- Minimal correction: Define recipes as explicit rules (L rule, delta_x rule, fallback) and publish the 12 (L,delta_x,n) triples before N4; per-family representatives covering both extremes (2/12 and 18/72) and a per-h TRAIN-inner check of the top recipes with explicit budget.
- Acceptance test: Table for all 12 horizons with no undefined cell, sealed before N4; budget line per h.
- Owner: Musashi

### I6N-AUD-05 [CRITICAL, NO_ESPECIFICADO] (request 3.5/D)

DEAP is not specified: fitness formula (error vs measured cost), Pareto/lexicographic/weighted, population, generations, operators, selection, repair, tolerance, stopping, DEAP seed; single seed with no seed-noise estimate.

- Evidence: Proposal:112-113, 139-141 ('se sella' without values); line 46 single seed. h=72 space ~1.6e6 configs; 24 evaluations from <=3 N1 survivors = one or two seeded generations, i.e. local screening.
- Minimal correction: Either name N2 'seeded local screening' or give a real GA budget; seal numeric fitness, operators, repair, tolerance and stop rule; calibrate seed noise (3 seeds x 3 configs x 4 folds) and set tolerance >= 2 SD.
- Acceptance test: Sealed numeric values exist before any fit; reproducibility test of the generator given the DEAP seed.
- Owner: Musashi

### I6N-AUD-06 [CRITICAL, NO_ESPECIFICADO] (request 8/E)

Temporal reduction can drop the origin t: no rounding, padding or alignment rule.

- Evidence: Proposal:128-130 states only the requirement. Executed probe: stride-2 causal conv covers t only for odd length; 55->28 keeps t, 28->14 loses it; n=2 loses t; 5->3->2 keeps, third stage loses; existing _residual_temporal_stage raises on odd n (components.py:294-295). Same defect class as the FS4 origin-3 lag (Satoshi lane).
- Minimal correction: Origin-anchored reduction: left-pad (masked) to a multiple of the factor, then valid kernel=stride=factor; forbid stages that would leave <2 positions.
- Acceptance test: Perturbing the last slot changes the last reduced output for every grid n and every stage count; future-perturbation leaves outputs unchanged.
- Owner: Musashi

### I6N-AUD-07 [MAJOR, AMBIGUO] (request 3.2/A)

Long-horizon support collapses within the week; h=72 is a poor representative.

- Evidence: Executed (assumed NY 17:00 calendar, offsets 12+12, Thursday cut): management origins = 96-h per normal week (h=72: 24, 60: 36, 48: 48, 36: 60, 24: 72, 18: 78); entry = min(67, 96-h). All h=72 origins fall Mon 05:00-Tue 04:00 NY; consecutive targets overlap 71 of 72 h; four 2023 weeks give 96 eligible / 87 scorable h=72 rows; VALIDATION 2024 h=72 1,272 eligible vs 3,712 in the old run; holiday-aware first 2024 week has zero h=72 support.
- Minimal correction: Seal per-h/week/role support table and a declared row/power budget before any fit; choose representatives with adequate support or more screening weeks.
- Acceptance test: Generated support table sealed, digest in identity; power statement per representative.
- Owner: Musashi

### I6N-AUD-08 [MAJOR, AMBIGUO] (request 4/A)

Bar timestamp convention (open vs close label) is not fixed.

- Evidence: Proposal:36 only 'ventana incluye t'. Data look end-labelled, but the DST-start week 2024-03-10 opens with a bar labelled at the open itself. With open labels the input leaks one hour and a target labelled B-1 closes at B (BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (604f9480):131-133).
- Minimal correction: Fix the convention in the calendar artifact and test it.
- Acceptance test: Reject bars whose end label <= open; a one-hour label shift changes the identity digest.
- Owner: Musashi

### I6N-AUD-09 [MAJOR, NO_ESPECIFICADO] (request 8/A)

No rule for input slots without a bar (weekends) on the elapsed-hour grid.

- Evidence: Executed on 2024 h=72 origins: L=144/delta 36 always lacks t-36 (29.4% missing); L=432 hourly 33.7%; delta 8 33.2%. Old code: zero placeholder + presence channel (i6e_clock_windows.py:41-48, i6m_model.py:19-46); only Transformer has an attention mask, Conv1D reads zeros, LSTM integrates them.
- Minimal correction: Specify placeholder, presence channel and per-route masking.
- Acceptance test: Perturbing a placeholder leaves output bit-identical on every route.
- Owner: Musashi

### I6N-AUD-10 [MAJOR, NO_ESPECIFICADO] (request 4/A)

VALIDATION/TEST boundary rule missing.

- Evidence: Executed: last 2024 market week has B on 2025-01-03; 58/94 h=2 rows and all h>=48 rows have 2025 targets. BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (604f9480):77 'pertenecer al split' does not say origin or target decides; old code drops them (i6m_data.py:58).
- Minimal correction: Split membership by target timestamp; remainder NO_ELIGIBLE_ROWS; same rule at TRAIN/inner cutoffs.
- Acceptance test: Boundary rows typed and counted; no 2025 price read in VALIDATION scoring.
- Owner: Musashi

### I6N-AUD-11 [MAJOR, CONTRADICTION] (request 3.8/A)

'53 segments' and the Monday 00:00 UTC cutoff inherited from I6-E contradict the market-week contract (WMC06).

- Evidence: Proposal:43; I6E plan weeks Monday 00:00 UTC with 'cutoff del lunes anterior'; BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (604f9480):14-19 requires cutoff at close C and fit during closure. Mid-week holiday closures (Dec 25, Jan 1, Good Friday closed 2023 / open 2024 in data) and a lower bound on force-flat offset unspecified.
- Minimal correction: Derive segments from a versioned instrument calendar with explicit clipping/holiday rules.
- Acceptance test: For every week fit_cutoff == C(previous) and release < O.
- Owner: Musashi

### I6N-AUD-12 [MAJOR, AMBIGUO] (request 4/A,6/C)

Inner folds, early stopping and weekly update mode are not pinned.

- Evidence: Inner weeks never named (proposal:103). 'promedio TRAIN/inner' = inherited 0.5*(running train loss + inner loss) unweighted (modular_candidate_evaluator.py:395-396); old inner = last 28 days (i6m_data.py:51-55), ~96 overlapping rows at h=72; 'inner' names two different sets. FULL_RETRAIN vs WARM_UPDATE (BWC:59-72) undeclared.
- Minimal correction: Name the four weeks; monitor inner-only loss; declare FULL_RETRAIN and init rule.
- Acceptance test: Sealed fold list and monitor definition; each fit records stop_class, selected_epoch, inner-only loss.
- Owner: Musashi

### I6N-AUD-13 [MAJOR, NO_ESPECIFICADO] (request 3.2/A)

Fitness population undefined; S_h unpaired across configs; U_h needs a 12-vector not defined for per-h models.

- Evidence: Proposal:139 'error agregado'; BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (604f9480):182-183 S_h depends on each config's verifiable rows; no intersection/coverage rule; N4 allows two recipes per h (proposal:163) so the consumed recipe is unspecified.
- Minimal correction: Fitness = S_h on the sealed common support; a forecast failure is a disqualifying typed failure (or coverage threshold); U_h is a diagnostic computed only after the strategy node with a declared recipe-per-h map.
- Acceptance test: Paired-row digest identical across compared configs.
- Owner: Musashi

### I6N-AUD-14 [MAJOR, NO_ESPECIFICADO] (request 6/C)

Fit hyperparameters absent from the proposal and fixed across capacities; monitor favours high capacity.

- Evidence: Inherited lr 1e-3, wd 1e-4, Huber delta 1.0 in per-h MAD units, min_delta 0, max_seconds 3600, max_updates 1e5 (i6m_prepare.py:63-65); dropout 0; chronological unshuffled batches; post-LN Transformer, no warmup; budget-truncated fits still scored. Contradicts FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md:8,83-88 (MAE/Huber comparison and LR/decay/delta tuning).
- Minimal correction: State every fit value; lr in {3e-4,1e-3,3e-3} check on smallest and largest configs; type budget-stopped fits RESOURCE.
- Acceptance test: Each fit records stop_class, selected_epoch, observed_updates.
- Owner: Musashi

### I6N-AUD-15 [MAJOR, NO_ESPECIFICADO] (request 6/C)

Multi-output task weights and scales unspecified.

- Evidence: Proposal:175-176 'declarar pesos' without values; h=18 ~78 eligible origins/week vs ~24 at h=72 -> ~3x gradient mass unweighted.
- Minimal correction: Fix w_h and s_h from TRAIN eligible labels before N4-M.
- Acceptance test: Report per-head gradient-norm share.
- Owner: Musashi

### I6N-AUD-16 [MAJOR, AMBIGUO] (request 8/E)

Transformer ordering and positional encoding ambiguous.

- Evidence: N1 'Transformer + Conv1D' (proposal:86) vs N2 'Conv1D+Transformer' (:119); blocks=0 repair; dropout/warmup/attention mask missing; post-LN in code. ClockView counts hours from window start (i6e_masked_model.py:40-47) so t's code varies with L and aliases for delta_x>=7 h; layers.py:79-95 uses ordinal index.
- Minimal correction: Origin-anchored lag-hour encoding; explicit order and norm placement.
- Acceptance test: t's encoding invariant to L.
- Owner: Musashi

### I6N-AUD-17 [MAJOR, AMBIGUO] (request 6/E)

Head readout is not a gene; LSTM return_sequences=True claim is inert.

- Evidence: Proposal:92 last state->Dense(1); executed probe LSTM(return_sequences=True)[:,-1] == return_sequences=False (max |diff| 0.0).
- Minimal correction: Add a readout gene or drop the claim.
- Acceptance test: Identity unchanged when the inert flag toggles.
- Owner: Musashi

### I6N-AUD-18 [MAJOR, NO_ESPECIFICADO] (request 7/D)

Nothing can consume the hierarchical genotype; no canonicalization rule for inactive genes.

- Evidence: DEAP flat bounded scalars, cxTwoPoint, selTournament(3), clamp repair (optimizer_plugins default_optimizer.py:185-201,405-419,1273-1275,1600-1605); DOIN wrapper takes {param:[low,high]} (doin-plugins predictor/optimizer.py:46,152-163); I6 identity digests the full arm dict (i6m_cell.py:58-63); factory fixed at 34,921 params, n<=241 (i6m_model.py:14,165; i6m_cell.py:66).
- Minimal correction: Specify genotype schema + canonicalizer; name missing components: I6-N model factory, conditional-operator DEAP driver with paired fitness to OLAP, DOIN plugin.
- Acceptance test: Active-gene perturbation changes config digest and model; inactive-gene perturbation leaves identity and cache key identical.
- Owner: Implementer after design revision

### I6N-AUD-19 [MAJOR, NO_ESPECIFICADO] (request 7/D,9/F)

ARIMA control underspecified and cannot produce an annual result under current failure rules.

- Evidence: Fixed order (1,1,0) (i6m_cell.py:145); resolution criterion absent (proposal:165); 46/128 cells FAILED_CONVERGENCE (h2 13/53, h3 18/53, h4 15/22); every arm has failures so none gets annual MAE. Weekend regularity claim REFUTED: inherited code uses a regular clock with NaN slots (i6m_cell.py:137-143).
- Minimal correction: Seal ex-ante order/convergence policy; TRAIN failure disqualifies a resolution; per-h coverage; NOT_COMPARABLE when incomplete.
- Acceptance test: Coverage table per h before N4.
- Owner: Musashi

### I6N-AUD-20 [MAJOR, NO_IMPLEMENTADO] (request 9/F)

Pause of I6-E-M is procedural, not durable.

- Evidence: Queue meta holds only design identity (i6m_queue.py:57-61); re-running i6m_deploy restarts the coordinator (i6m_deploy.py:183-189); claim() has no pause check; fate of 3,762 QUEUED not stated.
- Minimal correction: Sealed campaign state PAUSED/SUPERSEDED checked by claim(); type queued jobs as deferred without touching terminals.
- Acceptance test: run --once on a copy yields zero claims.
- Owner: Musashi

### I6N-AUD-21 [MAJOR, AMBIGUO] (request 9/F)

Two placement authorities disagree about the 5090 and the role map is reversed.

- Evidence: df_host_capacity.py:51-53 quarantines a GPU index that is the 5090 under the current host_roles.json, while i6m_deploy.py:157-160 picks by name and 85 fits ran on the 5090 on 2026-10-11. Snapshot today: 5090 enumerates normally, idle.
- Minimal correction: Name one placement authority; reconcile quarantine and role map.
- Acceptance test: Placement decision reproducible from one versioned file.
- Owner: Musashi/owner

### I6N-AUD-22 [MAJOR, NO_ESPECIFICADO] (request 9/F)

One-heavy-child-per-host is not enforced across campaigns.

- Evidence: GPU.lock is per campaign root (i6m_worker.py:28-30); FS4 timers live on both workers while the queue file marks that lane RETAINED_HISTORICAL_NOT_ACTIVE (EXPERIMENT_EXECUTION_QUEUE.json:102).
- Minimal correction: Host-level lease shared by all campaigns.
- Acceptance test: Second campaign blocks while the lease is held.
- Owner: Musashi/Satoshi

### I6N-AUD-23 [MAJOR, VERIFIED] (request 9/F)

The 4090 host has only ~7.3 GB available (12.94 GB unreclaimable slab, 7.54 GB vmalloc); 'when it fits' means small classes only.

- Evidence: One read-only snapshot 2026-10-11; ~4 GB admissible after 3 GiB host reserve.
- Minimal correction: Admit against live MemAvailable, never slice MemoryMax; schedule the reboot window with the owner.
- Acceptance test: Admission refuses when live headroom < 1.25*peak.
- Owner: Owner (reboot) / placement authority

### I6N-AUD-24 [MAJOR, NOT_MEASURED] (request 9/F)

Largest N2 window not buildable or costed.

- Evidence: n<=241 cap (i6m_model.py:165; i6m_deploy.py:147); windows held fully in RAM (i6m_cell.py:82); estimate 4.5-5.5 GB, cap 5.6-6.9 GB, can exceed 7 GiB MemoryHigh on the 5090 host.
- Minimal correction: Pilot before N2 admits n>241; stream windows per batch; unadmissible = RESOURCE_NOT_ADMISSIBLE, not worse fitness.
- Acceptance test: Measured peak per class in PLACEMENT_AUDIT.
- Owner: Implementer

### I6N-AUD-25 [MAJOR, NO_IMPLEMENTADO] (request 9/F)

Automation spec partly unbacked.

- Evidence: Reusable: WAL queue with exclusive claim, immutable terminals, OLAP outbox retry, pilot adoption, per-class ETA (i6m_queue.py:47-119; i6m_publish.py:267-302; i6m_worker.py:62-77). Missing: stop-drain, stale-progress watchdog (only ActiveState + 3,900 s cap), queue cannot take DEAP proposals or stage/h/config/fold levels, hard-coded stage advance, free-text failure reasons.
- Minimal correction: Specify typed failure classes (RESOURCE vs QUALITY vs DATA), watchdog, stop-drain and a DEAP-capable queue.
- Acceptance test: Fault-injection tests for each state.
- Owner: Implementer

### I6N-AUD-26 [MAJOR, NO_IMPLEMENTADO] (request 10/G)

Identity omits the protocol (WMC10).

- Evidence: fit_identity (i6m_cell.py:58-63; i6m_prepare.py:94-102) lacks contract version, calendar id, offsets, cut, bar convention, mask-function digest, S_h/U_h row digests; proposal:56-57,194-195 do not list fields. The 266 old terminals could be adopted under the new protocol.
- Minimal correction: List identity fields explicitly.
- Acceptance test: Changing any protocol field changes identity; old terminals are not reusable.
- Owner: Musashi

### I6N-AUD-27 [MAJOR, NO_ESPECIFICADO] (request 10/G)

OLAP has no place for S_h/U_h populations; custody fields missing.

- Evidence: gov_metric holds metric/value/split/horizon/unit only (olap/store query.py:552-555); publisher sends only Scored_Rows (i6m_publish.py:235-262); no dataset lineage, checkpoint/init/scaler digests local only, FAILED reason omitted, VRAM/device missing, readback checks only gov_metric (i6m_publish.py:281-283).
- Minimal correction: Population fact table (h, role, population, n, rows_sha256, mask_fn_sha256, calendar_id, contract_version, six exclusion counts, forecast failures); map custody fields to gov_terminal/_dataset/_artifact/gov_campaign_disposition.
- Acceptance test: Readback of every published field.
- Owner: Implementer

### I6N-AUD-28 [MAJOR, AMBIGUO] (request 9/F)

Budget ceilings: arithmetic correct (72+192+32=296; 1,272; 636; 212) but several costs are outside or unstated.

- Evidence: Outside/unstated: class pilots (proposal:72), annual control under new labels, ARIMA TRAIN resolution choice, recounts of old terminals, any search for non-representative h, seed calibration (none planned); R1/R2/H-CORE/donors excluded explicitly (:205). Measured basis only n=3 control (31.5 s 5090, 60.7 s 4090).
- Minimal correction: Itemize every class in the budget with a measured or NOT_MEASURED cost.
- Acceptance test: Budget table sealed with per-stage ceilings and STOP rule.
- Owner: Musashi

### I6N-AUD-29 [MINOR, VERIFIED] (request 3.2/A)

The proposed two-long minimum never binds under default offsets/cut.

- Evidence: Executed: minimum available longs over entry origins = 2; only 24 of 67 entry origins have all six longs, so entry geometry changes within the week.
- Minimal correction: Declare that entry geometry varies by origin and how TP/SL use partial longs.
- Acceptance test: Entry decision test at R=29 h with only 18 and 24 h.
- Owner: Musashi

### I6N-AUD-30 [MINOR, NO_ESPECIFICADO] (request 6/C)

Reducer/adapter kernel, residual, norm, mask channels, branch weight sharing, feature order and inner length not stated (inherited).

- Evidence: PARAMETER_AUDIT.csv rows marked AMBIGUO/NO_ESPECIFICADO (27 + 29 of 90).
- Minimal correction: State inherited values in the plan.
- Acceptance test: PARAMETER_AUDIT rows all ESPECIFICADO or justified FIXED.
- Owner: Musashi

### I6N-AUD-31 [MINOR, AMBIGUO] (request 10/G)

Physical copy: proposal asks for a local copy while CURRENT_EXECUTION requires copies verified on two machines; STATUS.json of the pause was not written by cycle() and says published 265.

- Evidence: Proposal:200-201 vs CURRENT_EXECUTION.md:106-108; STATUS.json vs i6m_coordinator.py:257-261. Authority for the pause is QUEUE.db + receipts (266/266 verified).
- Minimal correction: Align copy rule; state authority.
- Acceptance test: Second-machine digest verification.
- Owner: Musashi

### I6N-AUD-32 [INFO, VERIFIED] (request 1/G)

Pause counts and custody of I6-E-M verified; FAILED history preserved.

- Evidence: QUEUE.db read-only: NEURAL 138 COMPLETED; ARIMA 82 COMPLETED, 46 FAILED; 3,762 QUEUED; 0 RUNNING; 266/266 seals and PUBLISHED_READBACK_VERIFIED receipts; Queue.complete refuses replacement (i6m_queue.py:117-118). Live warehouse not re-queried (token-free read path missing).
- Minimal correction: None.
- Acceptance test: -
- Owner: -

## Answers to the eight doubts of request section 3

1. Joint search: incomplete (AUD-03). Input transform and output mode must be N2 genes or contrasted at >=2 windows.
2. Representatives vs per-h optimization: not per-h optimization (AUD-04, AUD-07); h=72 is the worst-supported representative.
3. Future-informed selection: confirmed; the 20 features were chosen on VALIDATION 2024 and with 2023-inclusive rankings (AUD-01).
4. Four weeks are not a year: four 2023 weeks give ~87 scorable, heavily overlapping h=72 rows; screening only, no robustness claim (AUD-07, AUD-12).
5. Open rules: weeks, tolerance, stop rule, task weights, fitness, repair, fit hyperparameters all unsealed (AUD-05, AUD-12, AUD-14, AUD-15).
6. Strategy described vs implemented: the effective plugin uses unordered min/max, ignores TP in early close, has no masks, abstention or calendar (AUD-02).
7. Horizons 18..72: calendar-compatible as candidates; support per origin collapses toward B and per-head masks are required; the 12-head index map is not implemented in the strategy (AUD-02, AUD-07, AUD-29).
8. Calendar before optimization: segments and cutoff inherited from I6-E (Monday 00:00 UTC) contradict the contract; a recount of old predictions does not certify the new labels (AUD-11, AUD-26).

## Limits

- NOT_VERIFIED: live warehouse readback (token-free read path absent); second-machine copy of I6-E-M; broker calendar (assumed); condition of software not yet written.
- Lead-auditor claims REFUTED by sub-auditors: ARIMA weekend irregularity (inherited code uses a regular clock with NaN slots); early-stopping monitor 'unknown' (it is defined; its effect is the defect); h/k step count (always integer; admissibility is the problem). Lead count for entry origins at h=18/24 corrected to 67 (entry cut binds).
- Executed tests are listed in `EVIDENCE_MANIFEST.json`; every other acceptance test is PROPOSED.

## Correction order

See `CORRECTIONS_REQUIRED.md`.
