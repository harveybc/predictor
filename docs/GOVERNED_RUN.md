# Governed predictor run

`tools/governed_run.py` is the decision-bearing predictor path for data-gov
Flow v3. It registers an immutable campaign before opening data, obtains and
confirms each input role through data-gov, runs `app/main.py` on CPU, persists
every outcome locally, reports it to the configured terminal lake, and checks
that data-gov and the lake agree.

Small mechanics tests may use other runners. Their results are non-governing
and cannot promote a model, transformation, feature or experiment. A result
used for a decision must be reproduced through this path or an equivalent
Flow-v3 client.

## Run

Prerequisites are a clean committed predictor checkout, a data-gov service
with a source lake whose resources have explicit availability contracts, a
terminal lake, and a service key for the predictor actor.

```bash
cd <predictor-checkout>
python tools/governed_run.py \
  --load_config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
  --experiment-key toy-ann-1575-1d \
  --gov-url http://127.0.0.1:5055 --api-key-file <service-key-file> \
  --lake predictor_examples --lake-root examples/data_downsampled \
  --metrics-lake olap_cube --out-dir <run-output-dir> \
  -- --epochs 2 --max_steps_train 300 --max_steps_test 300 --mc_samples 2 \
     --execution_purpose ARCHIVAL_REPLAY_NON_AUTHORITATIVE
```

Everything after `--` is passed to predictor as long-form flags. Overrides of
inputs, outputs or the config are refused. Data-gov calls occur before or after
training, never inside `fit`, `transform`, `step` or `learn`.

## Contract

1. The checkout must have an exact 40-hex commit and no tracked or untracked
   changes. The original config, requested roles/ranges and extra arguments
   form `predictor_execution_spec.v1`; its digest is fixed before data opens.
2. The wrapper registers one governing campaign with one declared unit. A
   reused campaign key with different code, config or inputs is refused.
3. Every configured input role is requested separately from `/api/v2/download`.
   The stream and any existing cache entry are both hashed. A cache path is
   never overwritten with different bytes. Confirmation records
   `VERIFIED_TRANSFER` or `VERIFIED_CACHE` only after local verification.
4. Predictor runs on CPU from a generated config whose inputs name the
   content-addressed cache and whose outputs remain under `--out-dir`. The
   output namespace must be fresh: any pre-existing scientific output causes a
   terminal `REFUSED`, without downloading data or replacing the old bytes.
5. A terminal is built for `COMPLETED`, `FAILED`, `INCONCLUSIVE`, `REFUSED` or
   `QUARANTINED`. It binds the campaign, verified deliveries, full metric
   identities, measured cost and output artifact hashes.
6. Before any network send, the terminal envelope is written with `O_EXCL`,
   file `fsync` and directory `fsync` under
   `~/.local/state/data-gov/terminal-outbox` (override with `--outbox-dir`). It
   moves to `sent/` only after the remote terminal is accepted and the
   reconciliation endpoint reports no difference between accounting and the
   terminal lake.
7. `<out-dir>/GOVERNED_RUN.json` records the campaign, exact inputs, execution
   spec, terminal and reconciliation. A pending terminal makes the command
   fail: the result exists, but is not governing.

Retry pending terminals after an outage:

```bash
python tools/flush_governed_terminals.py \
  --gov-url http://127.0.0.1:5055 --api-key-file <service-key-file>
```

The operation is idempotent. Replaying the same terminal returns the existing
receipt; a different terminal for the same campaign, unit and generation is
refused.

## Data recorded

`gov_terminal` stores one terminal per campaign/unit/generation. Metric names
are terminal keys over `[A-Za-z0-9._:-]`: every run of other characters in a
results label becomes one underscore (`Naive MAE` -> `Naive_MAE`). A refused
terminal stays in the outbox and its reason is recorded under
`outbox_flush.failures` in `GOVERNED_RUN.json` (and printed by
`flush_governed_terminals.py`).
`gov_terminal_dataset` stores each delivery id, source and delivered hashes,
role, range, available-time column, availability-contract hash and whether the
bytes came from a verified transfer or cache. `gov_terminal_metric` stores the
complete metric identity. `gov_terminal_artifact` stores content hashes for
the effective config, metrics and produced artifacts.

The legacy `/api/v1` report tables remain readable for history but do not
govern new decisions.

## Tests

```bash
python -m pytest -q \
  tests/test_governed_run_v3.py tests/test_governed_run.py \
  tests/test_governed_run_overrides.py

cd olap/lake
python -m pytest -q tests/test_terminal_v3.py tests/test_write_metrics.py \
  tests/test_metrics_api.py
```
