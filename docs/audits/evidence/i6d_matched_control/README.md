# I6-D matched control harness

Status: implementation verified; no real training or VALIDATION/TEST read in
this delivery.

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
