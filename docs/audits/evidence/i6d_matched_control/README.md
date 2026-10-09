# I6-D matched control harness

Status: implementation and one real annual VALIDATION control are complete;
TEST remains unread. See `WEEKLY_RESULT_20261009.md` for its exact scope.

The executable comparison has two intentionally different representation
paths over the same selected EURUSD feature union and exact causal 24-hour
windows:

- `DENSE`: one `causal_window_dense` branch per semantic feature (value and
  observation-mask channels stay together), unordered latent concatenation,
  and a non-temporal Dense control core.
- `CONV`: one causal time-preserving Conv1D branch per semantic feature,
  sequence fusion, positional encoding, causal Transformer blocks and residual
  Conv1D temporal reduction.

Both arms reach the same vector width only at the predictive-head boundary and
then use the same head implementation, shape, parameter count and initial
weights. DENSE latent slots are never timestamps. The report exposes distinct
architecture and parameter counts instead of forcing equivalence by reshaping.

The input NPZ contract requires explicit `baseline` values on the exact target
rows. Targets therefore need not masquerade as input features. `run-arm`
opens TRAIN and VALIDATION only; no TEST path exists in the design schema.

```bash
PYTHONPATH=. python tools/i6d_matched_control.py plan \
  --config examples/config/i6d/i6d_eurusd_y_s_1h_matched_control.json

PYTHONPATH=. python tools/i6d_matched_control.py init \
  --config examples/config/i6d/i6d_eurusd_y_s_1h_matched_control.json \
  --train <train.npz> --validation <validation.npz> --output <campaign-dir>

PYTHONPATH=. python tools/i6d_matched_control.py run-arm --output <campaign-dir> --arm DENSE
PYTHONPATH=. python tools/i6d_matched_control.py run-arm --output <campaign-dir> --arm CONV
PYTHONPATH=. python tools/i6d_matched_control.py status --output <campaign-dir>
PYTHONPATH=. python tools/i6d_matched_control.py close --output <campaign-dir>
```

Each arm is protected by a process lock, writes heartbeat/status files, adopts
an existing authenticated terminal, and saves a reload-checked Keras model.
Closure refuses different design, data-row identity, fitting budget or seed.

## BUSINESS weekly successor

The additive `weekly-*` commands reuse the I6-A/FS4 weekly calendar, as-of
resolver and row contract. They do not reuse one model across the validation
year. Each complete validation week receives a fresh fit over exactly the four
calendar years ending before its cutoff, with an internal chronological tail
for early stopping. DENSE and CONV then score the same finite rows of that week.

The campaign defaults to one seed. `--seed` may explicitly seal two or three
unique seeds, never more. The closure first requires arm parity for every week,
then computes annual MAE/MSE and the zero-return persistence naive by weighting
weekly values by their scored-row counts. No partial annual summary is emitted.
There is deliberately no TEST argument or TEST path.

The optional `--target-transform ROBUST_Z_FIT` changes only the numerical
coordinate used during fitting. For every week it estimates the target median
and robust scale from that week's fitting rows, applies them to fitting and
inner-validation targets, then inverts predictions before calculating metrics.
The raw target is neither denoised nor replaced, and the same-row naive remains
in raw log-return units. The transform and its fitted values are retained in
each result. It has a distinct campaign/task identity; legacy RAW campaigns
remain readable under their original schema.

New scored cells also retain `prediction_diagnostics.v1`: mergeable moments,
prediction/target scale, Pearson correlation and directional agreement. The
vectors themselves remain ephemeral. This distinguishes a useful forecast from
a model that approaches the naive merely by shrinking its output toward zero.
Closures produced before this addition remain valid but cannot acquire these
diagnostics retrospectively without rerunning inference.

```bash
PYTHONPATH=. python tools/i6d_matched_control.py weekly-plan \
  --config examples/config/i6d/i6d_eurusd_y_s_1h_matched_control.json \
  --validation-year 2024

PYTHONPATH=. python tools/i6d_matched_control.py weekly-init \
  --config examples/config/i6d/i6d_eurusd_y_s_1h_matched_control.json \
  --validation-year 2024 --output <weekly-campaign-dir> \
  --target-transform ROBUST_Z_FIT

# Run one independently schedulable cell. Repeat over sealed weeks and both arms.
PYTHONPATH=. python tools/i6d_matched_control.py weekly-run-cell \
  --output <weekly-campaign-dir> --arm DENSE --week-ordinal 0 --seed 0 \
  --feature-parquet <train-features.parquet> --target-parquet <train-targets.parquet> \
  --validation-feature-parquet <validation-features.parquet> \
  --validation-target-parquet <validation-targets.parquet>

PYTHONPATH=. python tools/i6d_matched_control.py weekly-status --output <weekly-campaign-dir>
PYTHONPATH=. python tools/i6d_matched_control.py weekly-close --output <weekly-campaign-dir>
```

For the strict branch-only successor, add `--control-kind BRANCH_ONLY` to
`weekly-plan` or `weekly-init`. This changes only the per-feature branch plugin:
`causal_conv1d` versus `causal_dense_sequence`; both arms retain the same
fusion, temporal core, forecasting head and synchronized downstream initial
weights. It creates a distinct design identity and never adopts cells from a
`WHOLE_MODEL` campaign.

An initialized campaign can run unattended in deterministic non-overlapping
shards. Each worker loads the data once, writes atomic progress with a measured
ETA under `<weekly-campaign-dir>/shard_status/`, resumes completed cells, and
records a failed cell once without retrying it or stopping independent cells:

```bash
LD_LIBRARY_PATH="<tensorflow-cuda-library-path>" \
CUDA_VISIBLE_DEVICES=<gpu-uuid> PYTHONPATH=. \
python tools/i6d_weekly_shard_runner.py \
  --output <weekly-campaign-dir> --worker-id <stable-worker-id> \
  --shard-index <zero-based-index> --shard-count <worker-count> \
  --require-gpu \
  --feature-parquet <train-features-1.parquet> \
  --feature-parquet <train-features-2.parquet> \
  --target-parquet <train-targets.parquet> \
  --validation-feature-parquet <validation-features-1.parquet> \
  --validation-target-parquet <validation-targets.parquet>
```

`--require-gpu` rejects before loading data unless TensorFlow sees exactly one
GPU, preventing silent CPU fallback and ambiguous placement. Every host must
receive the identical sealed `WEEKLY_DESIGN.json`. Reconcile host-local results
only through the authenticated merge, which preflights every design, seal and
conflict before writing any cell:

```bash
PYTHONPATH=. python tools/i6d_weekly_shard_merge.py \
  --destination <coordinator-campaign-dir> \
  --source <worker-a-campaign-dir> \
  --source <worker-b-campaign-dir> \
  --close
```

Copying cells directly is not a scientific merge. `--close` may produce an
`INCOMPLETE_EVIDENCE` closure if any assigned cell failed; it never manufactures
or retries missing evidence.
