# RETSU: I6N causal traceability audit

## Mission

Audit the complete data path used by the running H4 search, from retained source
bytes to the paired model/naive score. The audit must try to falsify the current
provisional improvement. It must not train a model, read TEST, change a running
service, or alter campaign evidence.

The object under audit is `predictor` commit
`cbbe1f168c0a6cc189abbd1e8b297e6bd093f951`, the H4 scientific configuration
whose digest is
`fb70328d8b057f110006e0713d304b5c42fec5419a8dcee35527add6ca19a4cf`, and the
sealed search plan whose digest is
`1b83490b54af927033a93105e3be85b8b7323c89672b147745bbeb1d3efee983`.

This is an evidence audit, not a code review by narration. Every conclusion must
be produced by a deterministic script and tied to file digests and concrete row
identities.

## Hard constraints

1. CPU only. Set `CUDA_VISIBLE_DEVICES=""` for every executable check.
2. Do not stop, restart, signal, reconfigure, or attach a debugger to any live
   DOIN node or evaluator.
3. Do not read TEST or any row dated 2025 or later. TRAIN and the declared inner
   2023 folds are sufficient.
4. Do not run a fit. Loading retained weights for a forward comparison is
   allowed only when no optimizer step can execute.
5. Do not read or print credentials. Use retained local evidence only.
6. Do not monitor logs. Implement one bounded command, run it once, and return
   after it writes its terminal status.
7. Do not modify a historical artifact. New evidence lives only under
   `docs/audits/evidence/I6N_CAUSAL_TRACE_AUDIT_20261012/`.
8. A missing provenance edge is `UNRESOLVED`, never `SAFE`.
9. The live campaign may continue as engineering exploration. This audit may
   block scientific promotion, VALIDATION, or strategy use; it must not stop the
   live campaign itself.

## Business-time contract to prove

- An origin is `t`. Every model input may use information available at or before
  `t`, including the complete observation at `t`.
- No input may use `t+1` or any later observation, directly or through a fitted
  statistic, decomposition, imputation, resampling operation, feature selector,
  cached artifact, or pretrained weight.
- The H4 endpoint target is at `t+4h`. A trajectory candidate may additionally
  supervise `t+1h`, `t+2h`, and `t+3h`, but its ranking endpoint remains H4.
- Market eligibility, weekend exclusions, forced-flat cutoff, folds, and
  purging must be applied before scoring. The model and the H4 naive must use
  exactly the same origin identities and target identities.
- Each fold uses only its declared four-year TRAIN interval to fit mutable
  state. INNER may score and stop training as declared, but cannot fit
  preprocessors or feature selection. Outer VALIDATION and TEST remain closed.

Bidirectional attention among positions already at or before `t` is not, by
itself, future leakage. Centered transforms, smoothers, decompositions, or
statistics that consume observations after the emitted timestamp are leakage.

## First critical question: feature-selection chronology

Trace every one of the 20 retained RAW features to the selection evidence that
admitted it. Explicitly test the existing claim that their union was selected
with 2024 weekly VALIDATION results and then supplied to 2023 search folds.

If confirmed, classify it as `PROVEN_FUTURE_SELECTION_LEAK` for any claim that
the 2023 fold result is an unbiased estimate. Do not erase the result and do not
call the model fit itself invalid. Reclassify the current H4 score as
`RETROSPECTIVE_DEVELOPMENT_ONLY`, and specify the minimum correction:

- either select features using data strictly before each 2023 fold and nest the
  selector inside that fold; or
- freeze one feature manifest using a period strictly earlier than all search
  folds, then rerun the search under a new identity.

The 2025+ TEST boundary remains closed in either correction.

## Required trace graph

Build a machine-readable directed acyclic graph with one node for every retained
source, transformation, fitted state, selected feature, window, target, mask,
fold, model fit, prediction, naive value, fold score, and aggregate fitness.
Every edge must contain:

- producer path and git commit when code produced it;
- artifact path, SHA-256, schema, and row identity;
- minimum and maximum source timestamps consumed;
- fitting interval when mutable state is learned;
- output timestamp and availability timestamp;
- downstream consumer.

For every prediction origin, prove `max_information_time <= t`. For every fitted
object, prove `max_fit_time < first_scored_time` under the declared fold
contract. A filename or narrative assertion is not proof.

## Executable falsification checks

Implement `tools/i6n_causal_trace_audit.py` with `run` and `status` commands and
a JSON spec committed beside the evidence. Add focused tests before the
implementation. The `run` command performs all checks below in one bounded CPU
process and writes status atomically.

### 1. Source and transformation provenance

- Verify source bytes and all declared digests before reading values.
- Reject duplicate or non-monotonic timestamps and undocumented timezone
  conversion.
- Enumerate each transformation used by the 20 features. Prove trailing-only
  semantics for indicators, rolling statistics, resampling, and imputations.
- Reject backward fill, centered rolling windows, whole-series normalization,
  interpolation across a future endpoint, two-sided filters, and non-causal
  decomposition outputs.
- For Kalman features, distinguish forward filtering from smoothing; smoothing
  is not admissible.
- For STL, wavelet, Hilbert, multitaper, or similar features, do not infer
  causality from the name. Recompute or inspect the exact implementation and
  report the support interval used for each emitted timestamp. In particular,
  catch coefficients timestamped at the center of a window that consumed future
  samples.

### 2. Preprocessing and fitted state

- Perturb all INNER values by a large deterministic sentinel and prove that
  scaler, imputer, selector, decomposition calibration, and normalization state
  fitted on TRAIN remain byte-identical.
- Perturb later folds and prove earlier-fold fitted state and windows remain
  byte-identical.
- Prove inverse transformations use parameters fitted only on the corresponding
  TRAIN interval.
- Prove each fold starts from the intended deterministic model initialization;
  no weights, optimizer slots, callbacks, or normalization state leak from a
  different fold or candidate.

### 3. Window and target alignment

Use at least ordinary, week-open, pre-close, DST-transition, missing-bar, and
fold-boundary origins. For each case, retain explicit timestamp arrays.

- Prove the rightmost input is exactly `t`, not `t-1`.
- Prove every other input timestamp is less than `t` and follows the candidate's
  right-anchored `delta_x` grid.
- Prove H4 endpoint is exactly `t+4 elapsed hours`, not four rows and not a
  timezone-shifted label.
- For trajectory supervision, prove offsets are exactly `+1,+2,+3,+4` hours.
- Perturb only source rows after `t` but before/after the target and prove the
  input tensor at `t` is unchanged.
- Perturb the source observation at `t` and prove the rightmost input changes.
- Perturb only the target observation at `t+4h` and prove the H4 target changes
  while the input tensor remains unchanged.
- Inject an impulse strictly after `t` into every transformation family and
  prove no feature emitted at or before `t` changes.

Use exact equality where the pipeline is deterministic. Any tolerance must be
declared before seeing the result and justified by numeric serialization only.

### 4. Fold boundaries and business mask

- Recompute the four 2023 fold cutoffs from the sealed calendar and compare row
  identities and digests with the retained frozen support.
- Prove TRAIN, purge/embargo, and INNER do not overlap after accounting for the
  longest input history and H4 target.
- Prove an origin is rejected whenever its target crosses the forced-flat
  cutoff or enters a period the business never trades.
- Prove weekend gaps are not silently treated as four elapsed market hours.
- Prove DST boundaries use elapsed instants, not naive wall-clock subtraction.
- Compare the complete ordered `(origin_id, target_id)` population used by the
  model with the population used by the H4 naive. Require exact identity, not
  merely equal counts.

### 5. Metric and ranking parity

- Recompute each retained candidate's endpoint MAE and H4 naive MAE from the
  retained paired rows in original price scale.
- Recompute per-fold fitness and the unweighted four-fold aggregate using
  `(MAE_h - naive_h) / naive_h`; negative is better.
- Prove no trajectory diagnostic, TRAIN metric, failed fold, excluded row, or
  candidate from another identity enters ranking.
- Verify that early stopping reads only the declared TRAIN/INNER quantities and
  cannot inspect outer VALIDATION or TEST.

### 6. Cross-check the current provisional champion

Audit genotype
`356e4accee09059a45b7c6863f953ae8aeeedd298f5725efd489589200733535`
without fitting it again. Recompute its four fold scores and aggregate fitness.
The currently advertised value is `-0.010586766321054819`; equality alone does
not establish validity, so report metric parity and causal validity separately.

## Result vocabulary

Every audited assertion receives exactly one state:

- `PROVEN_SAFE`: executable evidence tried the relevant counterexample and it
  did not cross the boundary.
- `PROVEN_LEAK`: executable evidence demonstrates future information crossing
  the boundary.
- `RISK_UNRESOLVED`: required bytes, code identity, or provenance are absent.
- `NOT_APPLICABLE`: the operation is not present in this campaign.

Never turn `RISK_UNRESOLVED` into a pass. Never average states.

## Deliverables

Write only these new outputs:

- `RETURN.md`: findings first, ordered by severity, then the minimum correction.
- `STATUS.json`: terminal state and counts by result vocabulary.
- `TRACE_GRAPH.json`: the complete provenance DAG.
- `FEATURE_PROVENANCE.csv`: one row per retained feature and fold.
- `ALIGNMENT_CASES.jsonl`: timestamps and mutation outcomes.
- `MASK_PARITY.json`: model/naive row-identity comparison.
- `METRIC_REPLAY.json`: fold and aggregate replay for the provisional champion.
- `FINDINGS.csv`: assertion, state, evidence path, and affected claims.
- `MANIFEST.sha256`: digest of every output and executable audit input.
- focused tests for the audit runner.

The final return must state separately:

1. whether the tensor/target pipeline is causally valid;
2. whether preprocessing and transformations are causally valid;
3. whether feature selection is chronologically valid;
4. whether model and naive use exactly paired rows and scale;
5. whether the provisional H4 result may be promoted, must remain retrospective,
   or must be rejected.

## Execution and reporting

Run the audit once under a 4 GiB memory cap and a 30-minute wall cap. It must
write periodic `STATUS.json` itself; no agent polls it. If it cannot finish in
that envelope, return a measured shard plan without inventing an ETA.

Use one implementation agent and one independent reviewer only. The reviewer
reruns the command, mutates at least one future row, one mask identity, one
digest, and one scaler boundary, and verifies that the audit fails closed.

Return one message only after the command terminates. Include branch, commit,
test counts, output digests, counts in each result state, the first causal break
in chronological order, and the disposition of the current H4 result. Do not
remain active to monitor the optimization campaign.
