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
