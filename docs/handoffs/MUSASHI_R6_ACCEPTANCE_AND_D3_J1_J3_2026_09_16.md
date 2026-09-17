# R6 accepted at queried scope; D3 temporal amendment and implementation order

Reviewer: Musashi. Executor: Satoshi. Reviewed return: predictor 54f2df0.
Date: 2026-09-16. Existing bounded CPU/governance authorization remains in force.

## Independent observations

Queried main schema through the running warehouse, read-only:
df_fact_coverage=440694; df_fact_coverage_v2=633189;
df_coverage_current=633189 with one run_id and one code_sha256;
df_coverage_history=1073883; df_coverage_current_denominator=715.
This accepts the current view's observed attribution/counts, not an independent
before/after proof of every historical row. No production changes were made.

Ran tests/test_r6_coverage_selection.py and
tests/test_d3_contract_and_acceptance.py: 57 passed in 1.27 seconds in the
separate DuckDB test interpreter. Full reported suites were not rerun.

## D3 findings reproduced before any operator implementation

Using the committed TrailingMean fixture and production df_d3_acceptance:

1. check_availability with resource lag=5, operator delay=0, lookback=7 returns
   passed=true. The code sums delay+lookback: past history does not make a late
   input available sooner. It also ignores actual output timestamps.
2. The same check refuses completion_lag_max='0s' as not a sample count. The lake
   contract expresses durations, not untyped sample counts. int(lag) is not the
   producer's parser and also risks truncating fractional values.
3. check_prefix accepts the centred noncausal control, independently reproduced.
   Removing the last lookback samples from comparison hides the exact boundary
   where future leakage appears. Test 2 catching this control does not correct
   the deficient test 1.
4. The design lists two operators without a noncausal twin. The harness returns
   undecided when no twin is supplied, making their acceptance inconclusive by
   construction rather than by measured failure.

ready_to_measure currently reports infrastructure presence, not acceptance of the
temporal battery. Preserve this distinction in machine-readable status and prose.

## J1 - Authorized pre-candidate design amendment

Issue a superseding addendum to design 07 BEFORE implementing/scoring candidates;
preserve the original design bytes and cite its digest. The reviewer authorizes
the following corrections explicitly, so no new owner permission is needed:

- Separate event index/time, input availability, output emission time and signal
  response delay. Past lookback, warm-up and group/response delay are not
  interchangeable measures of availability. Output emission cannot precede the
  latest availability of the inputs it actually consumes. Unknown stays unknown.
- Reuse the governed resource's duration/UTC semantics. Convert durations to
  sample offsets only with a declared sampling contract; no truncation, guessed
  cadence or treating holes as equally spaced observations. Test fractional
  duration, late arrivals, missing timestamps and real '0s' contracts.
- Prefix invariance compares EVERY output declared available at the prefix
  cutoff, including its value, availability mask and timestamp. No exemption for
  past lookback. A delayed retrospective representation is compared only after
  its declared emission time, without backdating it. Require nonempty tested
  available populations or an explicit insufficient-test outcome.
- Repeat prefix/future perturbation checks at multiple predetermined cuts,
  boundaries, lengths, impulses and missingness regimes. Test fitted state and
  chunk/restart separately; frozen train-only fit must not include the tested
  evaluation future. Instantiate/reload state independently across test branches
  so mutation by one transform cannot contaminate the next comparison.
- Noncausal twins are required where meaningful. For stateless pointwise codecs,
  explicitly mark that particular check NOT_APPLICABLE with a design reason,
  retaining independent temporal checks; absence is not an automatic pass.
- Distinguish impulse onset from group/phase delay. Do not claim that the first
  changed output measures group delay of every nonlinear detector or filter.
  Detector/quantizer response that an impulse cannot identify needs an appropriate
  predefined probe or an explicit unidentified diagnostic, never fabricated zero.

No scientific hypotheses, utility margins, data splits or accepted D2 outcomes
are changed by this mechanical amendment. D2's scientific review remains separate.

## J2 - Prove the contract before candidates

Freeze the reproduced defects as negative tests, plus valid causal zero-lag and
delayed-output positive controls. Prove each rule against the actual harness
callable and governed input contract, not a substitute implementation in tests.
Extend the schema where needed with truthful availability/emission fields.

Explicitly address window padding/centering, full-series reconstruction and
normalization, restart state, same prefix with changed future and changing future
fit data. Never reuse the old wavelet/STL leakage as an apparently better feature.
For wavelets derive actual filter support, boundary mode, level and latency; do
not assume 2^L alone describes every wavelet's support. For recursive filters
represent state dependencies rather than pretending filter order is finite memory.

Record code/design identity and the complete required-test matrix. Review-ready
means all mandatory tests ran and passed; a skip/unidentified case is scoped,
not silently accepted. Keep the 2 GiB per-process bound and fresh CPU cost pilot.

## J3 - Conditional implementation and governed mechanics, without another pause

After J1 is recorded and J2 mandatory tests pass, proceed within this order to
implement the nine declared operators through the existing plugin interfaces.
Use established numerical libraries for transforms/codecs where appropriate and
record exact configuration and versions. Preserve raw inputs and state identity.
Do not stop after fixing one test to ask whether to continue.

Run a bounded single-thread cost pilot, then the predefined mechanical matrix on
governed synthetic and already contracted toy resources. Distribute independent
units across the three healthy workers within measured memory budgets. Freeze
population and per-unit/aggregate time budgets before the matrix. Persist all
outcomes, including refusal and inconclusive, through data-gov into DuckDB.

Runs remain NON_GOVERNING mechanical evidence. No utility ranking, financial-data
eligibility, new scientific confirmation, GPU training or feature promotion.
An operator failing causality is recorded and excluded, not tuned against the
evaluation fixture until it passes. Legitimate code corrections get new versioned
evidence with the failed version preserved. Unknown real availability is not
replaced with synthetic truth.

Finish with per-operator applicability, causal-test coverage, resource contract,
cost, declared/measured availability, restart evidence and reconciled terminals.
Update method state/work plan. Infrastructure readiness is distinct from test
acceptance and from scientific utility. The warehouse index incident remains an
independent investigation and does not block this bounded mechanical work.
