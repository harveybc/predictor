# Shared Selection To Independent RL Consumers

Status: selection execution complete; final consumer and release audit pending.
Owners: Musashi, native SAC without expert; Satoshi, ZigZag-guided learning.
Owner clarification: the initial SAC policy has NO agent feedback, clipping,
rolling scaling or implicit extra transforms. See the measured
[profiles and input policy](FEATURE_PROFILES_ENCODINGS_AND_SAC_INPUT_POLICY_2026_10_22.md),
including the narrow ASINH fit-scope finding that must be checked before adoption.
This updates the research sequence, not an authorization to replay the held SAC
queue or a claim that a new RL campaign has started. DOIN is unchanged.

## 1. Measured State And Remaining Work

| Layer | Evidence read on 2026-10-22 | Remaining acceptance |
| --- | --- | --- |
| v3 selection execution | 3,894 units, no recorded errors; full input pool consumed | Final source-to-consumer audit, not a new selection rerun by default |
| Paired comparisons | Independent check: contrast 714, encoding 78, rescue 120 multi-arm groups, no support/seed/fold mismatches | Preserve the pair-specific rescue controls in the audit tool |
| Family decisions | 41 retained, 2 interaction rescues, 14 not retained under this evidence | Consumer usefulness, not universal exclusion claims |
| Feature candidates | Selection output strict126/shortlist174; corrected consumer lists core15, strict125, shortlist173, strict+encodings137 | Authenticate actual tensor column order, exclusions and encoding artifacts |
| Encoding | sessions, GMM one-hot, rel24, diff+rel480 selected in particular task/model cells | Downstream matched comparison; retain base columns as explicit controls |
| Expert consumer | Four v3 finalists declared in Satoshi's request c098c946 | Completed runs, TRAIN/validation tables and reproducible consumption |
| SAC consumer | Existing temporal branches/fusion/core code; separate frozen-TRAIN scaler candidate gym-fx 57d5556 | Integrate and verify scaler/selected-store interface before new fits |
| Extractor selection | Selection probes used a small temporal model and HGB | Per-family extractor, fusion and core architectures have NOT been selected |

The original evaluated pool is 260 HGB / 257 temporal columns, not thousands.
The larger engineered/encoding matrices and catalog are different populations.
Report requested, generated, eligible and consumed counts separately; do not
equate a catalog entry with an input actually evaluated by a model.

The shared worker receipts demonstrate Musashi's v3 shard ran all four stages.
Satoshi corrected the audit request attribution in b5bcef18. This affects
execution attribution, not the measured selection numbers.

### Consumer Update, Observed 2026-10-22 After 13:04 UTC

Read-only inspection confirms the sequential v3b driver and the strict-list
fit are active. The lists actually installed for the consumer contain core15,
strict125, shortlist173, strict+encodings137 and core15+relatives24. None contains
raw.log_close. That column was a synthetic selection control, absent from the
consumer store; it is not silently synthesized or substituted. Preserve the
original selection counts and the corrected consumer counts as separate fields.

The script queues the remaining development finalists, four matched scaling
arms, then four consolidated-contract finalists. All four scaling commands
carry eligible-from2020-02-01; the CLI propagates it to the loader, which records
per-period support hashes. This verifies configuration and hash instrumentation,
NOT completed equality of supports: cutoff equality alone cannot exclude later
NaNs, different windows, labels or missing encoded rows. Check completed receipts
before computing paired deltas; if mismatched, refit on explicit common support.

Rolling-z versus fixed+keep is the scale-policy-only contrast. Fixed+diff changes
the input transform; fixed+diff+relative columns also changes feature count and
model capacity. Keep these as distinct contrasts, not four one-variable copies.

The log confirms earlier large-list attempts failed before this corrected queue;
their failure is not a scientific score. Driver completion markers alone cannot
establish success: an older queue logged DONE after failures. The final consumer
table must enumerate successful, failed and missing arms and authenticate each
terminal artifact. Do not modify the peer's active driver from this handoff.

## 2. One Shared Input Release, Two Learning Decisions

Keep separate identities for: admitted observations, generated feature pool,
selected column list, encoding recipe, fitted scaler and learned representation.
Both consumers bind the same immutable source/store and calendar release, but
each records its exact input subset and support. Do not merge weights, metrics,
or expert targets into SAC observations.

The release manifest must resolve these existing artifacts, not duplicate their
large contents in Git:

- Source/store hashes, observed UTC bar-start/close meaning, availability and
  DST-aware clock verification; no shifted or synthetic weekend observations.
- Feature names in tensor order, parent/component links, exclusions, transform
  code/parameters and fit cutoffs for every learned upstream transform.
- Exact encoding implementation, output columns and train-only fitted state.
  A family decision such as USE_sessions is not executable materialization.
- TRAIN-only fixed means/scales, constant-column rule, native cyclic/binary
  treatment, clipping policy and reload identity. No silent rolling transform,
  automatic differencing, NaN-to-zero or interpolation in the adapter.
- Split boundaries, per-arm eligible origin/support hashes, window count and
  observed-bar stride, seeds, code/runtime versions and accounting settings.
- Warehouse locator/read-back receipts, checkpoints, effective configs and two
  verified physical copies. Common IDs join feature and trading metrics.

Stationarity tests can motivate a keep/diff/relative experiment; they do not
make differencing a mathematical prerequisite of a fixed z-score. Preserve the
chosen transform explicitly rather than hiding it inside scaler construction.
For learned decompositions/PCA/regimes, hash equality alone does not establish
proper fit cutoffs: include prefix/future perturbation and fold-fit evidence.

## 3. Satoshi Handoff And Minimal Corrections

1. Finish the four already queued v3 consumer arms, without a duplicate launch.
   Export features requested versus consumed, tensor order, support hashes,
   scaler/encoding artifacts, seeds and native TRAIN/validation metrics.
2. The explicitly matched rolling reference is now queued alongside fixed+keep,
   fixed+diff and fixed+diff+relatives. Verify shared support from final receipts,
   not the shared cutoff alone; no reuse of the old unmatched score.
3. The four consolidated arms are now queued separately. Label dev3y/internal-2023
   comparisons separately from the principal 4-year
   TRAIN + next-year VALIDATION + next-year TEST contract. No quiet substitution
   of one experiment's years for another's; TEST is not used for selection.
4. Produce the final audit request with completed consumer evidence. Recompute
   only stages actually affected by demonstrated defects.
5. Keep ZigZag demonstrations/labels on the training side with their realized
   label-support purge. They never become causal market observations. Identify
   supervised imitation, DAgger, and actual RL stages separately in artifacts.

Economic consensus/surprises and a multi-rate gold consumer are separately
tracked extensions, not prerequisites to run the clean eligible subset. Do not
call these missing features useless, or call the entire universe exhausted.

## 4. Musashi SAC Sequence

1. Complete the fixed-TRAIN scaler and selected-store consumer binding. Verify
   current t inclusion, no future dependency, exact names/order, reload parity,
   and invariant business/accounting behavior under feature changes.
2. Evaluate core15, strict and shortlist as candidate subsets for SAC, not as
   settled SAC winners. The selection used opener/hit and expert-related tasks;
   SAC reward, risk and exits can require a different subset. Keep deferred
   families available for bounded interaction rescue.
3. Cross these three candidate subsets with direct selected-window SAC versus
   per-feature causal Conv1D -> fusion -> temporal core -> policy/Q heads.
   Preserve input/time support, seeds, budget, preprocessing, action space and
   costs. Do not discard a subset using only the direct model before its temporal
   contrast: subset utility can depend on the representation. This is a bounded
   three-subset by two-representation design, not a broad architecture search.
   Report parameter counts; changed counts preclude a capacity-isolated
   attribution unless a further matched-capacity control is run.
4. Start small, using the existing configurable temporal implementation rather
   than inventing an encoder from a ranking table. Separate branch, fusion and
   core capacity contrasts; 8/16 filters are candidates, not measured optima.
   Keep native cyclic fields interpretable. Compare family-specific extraction
   only after the uniform temporal control; no automatic best-extractor claim.
5. Keep fixed-budget native SAC and sparse evaluation for this stage; do not
   restore costly per-update early stopping or periodic refitting implicitly.
   Refitting cadence, window/stride refinement and bounded subset interaction
   studies follow after the clean common baseline.

Initial context is 13 observed instants at stride 2, including t; this is an
explicit starting control, not a proven universal optimum. Elapsed time spans
closures and must not be misreported as always 24 wall-clock hours.

The environment retains position, SL, TP, balance and calendar state in all
arms. The initial SAC policy and critics receive no agent-state observations;
none is the explicit baseline, not an optional choice between inherited modes.
Any future minimal-feedback arm requires an explicit separate experiment, not
automatic inclusion. Record partial-observability consequences, not as a
feature-quality failure alone. Replay actions/rewards remain SAC learning data.

## 5. Shared Business And Evaluation Contract

- One concurrent order per asset; SL, TP and risk-dependent fractional volume
  on every order, with broker minima. rel_volume and leverage stay explicit;
  use the approved 1:50 baseline, never silently substitute 1:1.
- Forced flat 12 hours before weekly market close and no entries until 12 hours
  after weekly reopening. Apply entry caps, fills and costs identically. Input
  history may cross the weekend using existing observed bars only.
- Keep main TRAIN four years, VALIDATION the next complete year, TEST the next
  complete year. Selection development/confirmation panels remain declared;
  previously inspected panels are not relabeled pristine holdouts.
- Evaluate TRAIN and VALIDATION with identical deterministic policy and native
  accounting. Publish return/profit on initial cash, Sharpe convention, max DD,
  closed orders, win rate, fees, funding, turnover and activity by day/week.
  Report both splits and their gaps, seed dispersion and paired deltas.
- No final architecture is selected from TRAIN profit alone. Freeze the
  validation/risk/gap decision rule in the finite campaign spec before fitting;
  no hidden scalarization or undocumented threshold chosen after results.

## 6. Requirement To Evidence And Next Action

| ID | Required test/evidence | Current disposition |
| --- | --- | --- |
| H01 | Store digest, exact columns, support parity | v3 selection repaired; consumer evidence pending |
| H02 | Prefix/future perturbation and upstream fit cutoffs | Final delivered audit required |
| H03 | TRAIN-only scaler and checkpoint reload parity | Candidate exists; SAC integration pending |
| H04 | Business calendar/SL/TP/sizing/costs unchanged | Must run through real SAC adapter, not synthetic-only certification |
| H05 | Actual feature list -> tensor -> checkpoint -> metrics | Expert finalists pending; SAC not yet run |
| H06 | Native feature vs temporal representation, shared seeds/support | New finite SAC spec after H01-H04 |
| H07 | Durable sequential jobs/publication/restart | Existing worker pattern; new campaign identity required |
| H08 | Both split metrics, gap, seed dispersion and custody | Required at each terminal result, negative or positive |

Next: audit the completed consumer handoff and integrate the SAC input binding
in an isolated worktree. Close the real-path tests, freeze the finite SAC spec,
then deploy its sequential 5090 queue. Do not resume old held configurations,
overlap a peer's heavy job, or mark deployment done from this document alone.
