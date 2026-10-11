# Historical Endpoint Evaluator

`tools.i6n_historical_evaluator` connects the shared I6N optimizer adapter to
the existing historical cell's data preparation and real masked model training.
It does not launch a search, a historical pilot, or a campaign. The pinned
historical cell, its tests, and its documentation are unchanged.
This is the first **endpoint-only cost class**, not full-swarm or full-search-space
support. Every successful receipt labels `metadata.search_space_scope` as
`ENDPOINT_ONLY`.

## Freeze Before Proposals

Use a complete, authenticated `i6n.historical_cell.v1` JSON config as documented
in [HISTORICAL_CELL.md](HISTORICAL_CELL.md). The parent must supply its audited
source-clock successor and explicit complete C1 calendar. This bridge neither
infers a clock nor upgrades an unresolved source-clock attestation.

From the checkout root, under the operator's external resource admission:

```sh
python -m tools.i6n_historical_evaluator freeze \
  --config /absolute/path/cell.json --output /absolute/path/new-bridge-directory
```

This prepares data but does not fit a model. The output directory must not
already exist. `FROZEN.json` seals separate full TRAIN and inner C1/C2 opportunity
populations, their ordered masks, counts, naive absolute-error sums, and support
digests. `BINDING.json` supplies these shared-adapter configuration fields:

```json
{
  "i6n_horizon": 2,
  "i6n_support": {
    "counts": {"train": 123, "inner": 45},
    "naive_sums": {"train": 0.123, "inner": 0.045},
    "support_digest": {"train": "<sha256>", "inner": "<sha256>"}
  },
  "i6n_evaluation_context": {
    "schema": "i6n.historical_evaluator.v1",
    "base_config": "/absolute/path/cell.json",
    "base_config_sha256": "<sha256>",
    "frozen_support": "/absolute/path/new-bridge-directory/FROZEN.json",
    "frozen_support_sha256": "<sha256>",
    "output_root": "/absolute/path/new-bridge-directory"
  },
  "i6n_evaluator_command": ["/absolute/path/python", "-m", "tools.i6n_historical_evaluator", "evaluate"],
  "i6n_evaluator_timeout": 720
}
```

The numbers above illustrate the schema only. Merge the generated binding,
without substituting counts or digests, into the shared optimizer config.
The evaluator request has exactly `config` (canonical effective genome),
`generation` (nonnegative integer), `context`, and `support` keys.

The generated binding uses `sys.executable` for the command and explicitly sets
`i6n_evaluator_timeout` to the cell's `resources.max_seconds + 120` (the example
above assumes a 600-second cell budget). Pass both fields through unchanged;
do not also set `i6n_evaluator`. This subprocess binding supports DOIN's executor
thread while the evaluator's wall timer runs in the child process's main thread.
Direct callback calls from other threads, or with an existing wall timer, fail
before request loading, output directory creation, or a RUNNING status write.
The command consumes one JSON request on stdin and returns one JSON receipt on
stdout. Existing bindings using the inline callback must be updated before
launch; this change does not rewrite retained artifacts or authorize retries.
Run the worker under the appropriate hard memory/time
limits; config estimates are not measurements of whole-process peak memory.
Do not run this command to inspect status: it can initiate the requested fit.

## Supported Genome

| Effective field | Behavior |
| --- | --- |
| `horizon` | One endpoint horizon, equal to the frozen binding's horizon. |
| `multiplier`, `history_hours`, `samples` | Search's exact derived history and sample count; complete elapsed-time windows include current t. |
| `delta_x`, `input_offsets` | Exact input grid spacing, including missing-row and feature masks. |
| `core`, `final_steps` | Entire canonical core passed to the real model builder; derived reduction size verified. |
| `input_transform=identity` | TRAIN-scaled input levels. |
| `input_transform=difference` | Backward differences of TRAIN-scaled levels on the selected input grid. |
| `output_mode=endpoint`, `delta_y`, `output_offsets` | Canonical single endpoint only; no independent trajectory supervision. |

For differences, the first slot has zero values and unavailable masks. Every
later difference requires both adjacent selected-grid observations and both
feature-source flags; it does not bridge a missing selected observation. No
extra observation before the selected window is silently fetched. A larger
`delta_x` means differencing across that larger elapsed interval, not across
adjacent retained rows. Feature levels are standardized using TRAIN only before
differencing; differences are not independently standardized.

Every effective field must exactly match recomputation by
`i6n_search.effective_config`. Unknown transforms, unknown fields, altered
derived values, noncanonical inactive genes, and incompatible horizons fail.
Genuine trajectories fail with `TRAJECTORY_UNSUPPORTED` before data preparation
or fitting. Search's one-point trajectory alias is already canonical endpoint.
The general search sampler can still propose genuine trajectories: those are
rejections, not endpoint evaluations. Do not describe this as full search-space
support or silently rewrite those proposals.

Remaining trajectory work is independent `delta_y` targets/masks, TRAIN-only
per-head scales, multi-head training, and endpoint scoring against the same
frozen S. It must not change endpoint support or silently couple `delta_y` to
`delta_x`.

## Training And Receipt

The inherited preparation uses the current complete four-year history, the
28-day inner interval, and the prior market-close fitting cutoff. All eligible
TRAIN and inner rows are materialized in bounded window batches. There is no
18-hour sample truncation or candidate-selected subset. Every proposal's full
ordered support must equal the frozen support; intersection or row dropping is
an error. Feature scaling and the endpoint log-return target scale use TRAIN
only. Resource caps apply to candidate sample count, rows, materialization,
batch bytes, parameters, worst-case optimizer updates, and wall time.

`fit_masked` performs real training and restores the selected checkpoint using
the configured patience and mean TRAIN/inner objective. The checkpoint is saved,
reloaded, weight-digest checked, and both splits' predictions replayed. Explicit
CPU/GPU placement is checked; GPU requests cannot fall back to CPU.

Predictions are inverted to original prices and scored separately through
`business_metrics` C2 on the identical frozen S. A nonfinite, nonpositive, or
missing prediction fails the candidate. The shared receipt returns actual
`error_sums`, `naive_sums`, `counts`, and `support_digest`, with fitness:

```text
0.5 * (TRAIN absolute-error sum / TRAIN naive absolute-error sum
     + inner absolute-error sum / inner naive absolute-error sum)
```

Metadata includes original-price endpoint MAE/MSE and naive metrics for each
split, selected training audit, scales, coverage dates, effective genome,
checkpoint reload verification, and native C2 receipt paths/digests. This is
TRAIN/inner development scoring, not an untouched evaluation result. The base
cell still validates its evaluation partition during preparation, but the
bridge does not materialize or score that partition or use it in fitness.
TEST remains forbidden.

## Durable Identity

Each candidate lives under `output_root/candidates/<identity-digest>/`:

- `STATUS.json`: atomic RUNNING, COMPLETED, or FAILED state; update/epoch progress,
  elapsed time and epoch-based ETA (unknown before the first epoch).
- `model.keras`: selected checkpoint, saved and reloaded before scoring.
- `scores.npz`: ordered origin, current price, future price, and prediction for
  each split; original-price rows retained for independent inspection.
- `train.input.json`, `inner.input.json`: full native C2 forecast inputs.
- `train.c2.json`, `inner.c2.json`: native C2 results against sealed S.
- `RESULT.json`: shared receipt, full identity, and artifact digests.

Identity binds effective genes, config, input and clock pins, frozen support,
governing dependency code/runtime, and bridge/search code. Generation is
intentionally excluded. An identical completed request validates every artifact
and freshly rescores the retained rows through C2 without retraining. An
exclusive claim prevents concurrent fitting. FAILED, interrupted RUNNING, or
orphaned artifacts do not authorize a retry. Externally killed processes may
leave RUNNING: the external pilot supervisor owns kill/OOM receipts and retained
whole-cgroup resource peaks. This bridge is not that supervisor.

## Verification And Scope

Tests were written first, then implementation added in the new bridge only.
Coverage maps directly to canonical gene mapping; causal difference masks;
full-support invariance and drop refusal; float32-source/C2 float64 agreement;
real masked fitting; original-price C2 receipts; checkpoint reload;
shared-adapter forwarding; cross-generation reuse; pin/artifact tampering; and
terminal failure without retry.

No-fit regression coverage also checks the persisted command binding and timeout,
executor-thread command dispatch, and callback/timer preflight rejection without
creating output. To run evaluator checks without its generated-data fit test:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  crispdm-run -m 512M -t 120s -- python -B -m pytest -q -p no:cacheprovider \
  tests/test_i6n_historical_evaluator.py -k 'not tiny_real_difference_fit'
```

The command-binding correction passed this no-fit selection on 2026-10-11:
**14 passed, 1 deselected in 16.38 seconds** under the limits above. The existing
fit/reload test was deliberately not run for this correction.

On 2026-10-11, the evaluator, optimizer, and search tests passed together:
**154 passed in 27.41 seconds**, with two protobuf deprecation warnings, under
`crispdm-run -m 2G -t 120s`, CPU-only. These tests use tiny generated observations
within a complete multi-year calendar. They are not historical fit results or
external clock evidence. No historical fit, deployment, network access, or
campaign was run for this implementation.

Scientific scope remains retrospective TRAIN/inner evaluation conditional on
the parent's clock audit, declared nominal calendar, inherited current-vintage
features and availability assumptions. This is not production certification,
broker-observed calendar evidence, or point-in-time exogenous-vintage proof.
