# SOTA-first: reference reproduction before further model selection

Owner decision: 2026-09-21. Status: ACTIVE; NO_NEW_MEASUREMENT.
This overrides older next-fit instructions, not the recorded historical facts.

## Execution amendment, 2026-09-23

Retention-tool repairs and original-device replay waits are NOT global training
prerequisites. Execute the already mapped TimeFilter released L512 recipe under
RP135 on the admitted external 5090 while retaining its artifacts. This advances
SOTA-REPRO itself; it does not waive independent acceptance before downstream
model selection. Its Table 9 comparison must retain the unresolved published
lookback qualification. Keep scientific-path, governance, disk and thermal gates.
No new owner engineering authorization is needed between campaign cells.

## What changes

The primary experiment is a faithful reproduction of the strongest supported
published result for the specified benchmark. A small homemade TCN, adapted GRU
or linear regressor is not an acceptable substitute. Do not choose a model by
size or popularity: establish its reported performance under the SAME protocol.
Do not silently downgrade to an easier reference because it is cheaper.

Previous exploratory model results are HISTORICAL_DEV_ONLY. They must not select
the future architecture, optimizer, loss, pretraining regime or trading policy.
Keep their provenance, failure reports and engineering regression tests. This
is a change in intended use, not a verdict that every measurement was wrong.
Do not delete results merely because their error is high or retain only winners.
Naive/seasonal controls remain mandatory diagnostics alongside the SOTA model;
they are not proposed production architectures or a new baseline-only campaign.

## Dataset and reference identity

Electricity/ECL in the owner's cited overview is the commonly used 321-client
hourly forecasting benchmark. UCI235 is one household at minute resolution.
The raw UCI321 collection and the official processed ECL benchmark are also not
interchangeable. Our custom subset of clients is not the full benchmark.

Primary sources inspected on 2026-09-21:

- [UCI235](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
- [UCI321 source collection](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
- [PatchTST official ECL script](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST/electricity.sh):
  W336, H96/192/336/720, 321 channels, 3 layers, 16 heads, d_model128,
  d_ff256, patches16/stride8, batch32, LR1e-4, epochs100/patience10.
  Its official trainer uses Adam and MSE, not the AdamW claimed by the overview.
- [iTransformer official ECL script](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/scripts/multivariate_forecasting/ECL/iTransformer.sh):
  W96, four horizons, 321 channels, 3 layers, d_model512/d_ff512,
  batch16, LR0.0005. Defaults and paper-version parity still require the freeze.
- [TimeFilter, ICML 2025](https://proceedings.mlr.press/v267/hu25ac.html), with
  [author code](https://github.com/TROUBADOUR000/TimeFilter): a newer candidate,
  not evidence that the two older models are undisputed current SOTA.
- [NPMixer, 2026 preprint](https://arxiv.org/abs/2605.07476): include in the
  search for stronger results; resolve table/protocol and code availability
  before selecting it. A preprint claim alone is not independent verification.

The two pinned repository heads above are inspected revisions, NOT certified
original paper-release commits. No candidate is yet declared the winner.
The candidate search must include stronger recent methods; it is not restricted
to this discovery list. Record missing code/data without concealing a stronger
claim. If MAE and MSE favor different methods, retain both leaders rather than
inventing a universal winner. Compare per horizon and the paper's aggregate.

## Exactness and acceptance

Freeze paper/version/table, code revision, environment, processed dataset digest,
channel ordering, temporal boundaries, input and target construction, scaler fit
population, missing-value policy, window/horizon/stride, full architecture,
loss, optimizer, schedules, training allocation, validation/checkpoint policy,
seeds and all metric reductions BEFORE evaluation. Resolve paper/code conflicts
explicitly. A missing detail cannot be filled with our favorite default while
claiming exact reproduction.

Run the author's implementation first; wrapping transport and accounting must
not change its mathematical path. Validate train-only transformations, future
perturbations of actual inputs, split support, checkpoint restoration and metric
recomputation. Context inside an observed window is distinct from future target
access. In particular, audit wavelet/STL padding and reconstruction on the actual
forecasting path. Do not reject valid within-history processing by name alone.

Use the published metrics in their original normalized space and reduction.
Our auxiliary MAE_z cannot replace them. Cross-dataset normalized errors do not
create identical tasks. Report matched naive and seasonal baselines as context.

Freeze the numerical agreement criterion from reported precision/variability
and replication design before seeing new test results. Protocol fidelity and
numerical reproduction are separate verdicts. No hand-tuning against the public
test until the score matches. Never guarantee identical floating-point results
across hardware or silently widen tolerances. Retain every attempted seed.
The public benchmark test is evaluated under the frozen reproduction protocol;
it is not a fresh private confirmation set. Financial reserved data stay closed.

## Budget and downstream use

Plan compute to execute the full author recipe; do not reduce channels, context,
depth, training data or convergence budget to fit the previous pilot allowance.
Profile without exposing benchmark test scores for recipe selection. Inventory
available CPU/GPU and memory; do not assume three workers imply three suitable
GPUs. Parallelize independent replicas only where it preserves the protocol.
Use existing governed execution, with resource limits and observed cost.
Missing capacity is a named execution gap, not permission to report an adaptation
as an exact reproduction or to evade environment restrictions.

After independent reproduction acceptance, measure the doctoral interventions
against this strong reference under matched data, tuning and training budgets.
Electricity ranking does NOT choose the financial or RL winner. Preserve the
financial loss/optimizer comparison, weekly retraining, trading evaluation and
core-pretraining hypotheses as later mandatory stages, each with domain-specific
references. No scientific conclusion is inferred from a large parameter count.
