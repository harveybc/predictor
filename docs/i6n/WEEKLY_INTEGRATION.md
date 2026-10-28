# Weekly Integration

This is the continuation of C1/C2, not a replacement for their audited
contracts. Historical experiments remain separate from synthetic engineering
verification. The legacy I6-E-M queue must not be resumed as I6N.

## Components

| Component | Responsibility | Verification |
| --- | --- | --- |
| `tools/i6n_weekly_data.py` | Four calendar years before the previous market close, TRAIN/inner purge, per-head C1 masks, development boundary | `tests/test_i6n_weekly_data.py` |
| `tools/i6n_training.py` | Masked AdamW, exact per-head epoch sums/counts, mean TRAIN/inner stopping, best-weight restoration | `tests/test_i6n_training.py` |
| `tools/i6n_search.py` | Joint context/capacity/resolution genomes, bounded local DEAP proposals and receipt validation | `tests/test_i6n_search.py` |
| `tools/i6n_weekly_smoke.py` | Calendar, origin-containing windows, TRAIN-only scales, model fit, checkpoint replay, C2 original-price scores | `tests/test_i6n_weekly_smoke.py` |
| `tools/i6n_source_recovery.py` | Bounded public-source retrieval and clock comparison, never automatic relabelling | `tests/test_i6n_source_recovery.py` |

The partition API checks a complete calendar and hourly origin enumeration by
default, and rejects empty eligible TRAIN/inner/evaluation populations. It
cannot certify that a caller's declaration of source coverage is true: actual
source bytes, feature versions and broker calendar must be bound by the
historical adapter. Its `support_sha256` is not a hash of feature/price values.
Evaluation membership follows C1's target instant; TRAIN/inner additionally
purge shared origins at the stopping boundary. Known calendar closure rows are
excluded explicitly rather than interpreted as unknown calendar gaps.

The synthetic integration uses a small, deliberately sparse invented calendar
and three generated features. It is not four years of real training, an annual
validation, an optimization result or a prediction-quality measurement. Its
two heads exercise independent masks, not the production horizon schedule.
No synthetic score is published as EURUSD or ETH performance.
The fixture explicitly uses `coverage_mode=synthetic_sparse` and records
`coverage_complete=False`; this mode must never qualify historical fitting.

## Finite Execution

Run from this checkout in the established application environment. Each CPU
suite is admitted at 2 GiB / 120 seconds; no agent needs to tail logs:

```bash
python -m tools.i6n_verify run --output /var/tmp/i6n-verification-new
python -m tools.i6n_verify status --output /var/tmp/i6n-verification-new
```

Use a new output directory for changed code. Repeating the same completed
verification command checks retained evidence and does not rerun the tests.

Independent synthetic end-to-end run, with an explicit small epoch limit:

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  crispdm-run -m 2G -t 120s -n i6n-weekly-synthetic -- \
  python -m tools.i6n_weekly_smoke run \
    --output /var/tmp/i6n-weekly-synthetic-new --device cpu --epochs 3
python -m tools.i6n_weekly_smoke status \
  --output /var/tmp/i6n-weekly-synthetic-new
```

The GPU form uses `--device gpu`, exactly one visible GPU, its established
TensorFlow environment and fresh host admission. Do not copy a CPU cap as a
GPU memory estimate. The preferred host runs only its 5090, never both GPUs
concurrently. The secondary host's available RAM must be checked independently
of free VRAM. No reboot, service restart or cache drop is part of these scripts.

`STATUS.json` is updated after optimizer updates and epochs, with elapsed time
and an epoch-based estimate after the first completed epoch. Final artifacts
include the model, metric inputs, calendar inputs and `RESULT.json`. Failure
is terminal for that output directory, not a lower-cap retry loop. Neither
script is a daemon or an unattended historical campaign.
If a runner dies, an unlocked retained RUNNING receipt is reported as
INTERRUPTED, without changing the old evidence. Completed status rechecks
artifact digests and recomputes paired metrics from the retained inputs.

## Remaining Production Work

1. Resolve the retained source-clock discrepancy and bind a complete calendar
   and source-availability manifest. An empty vendor download did not resolve
   the discrepancy; see `SOURCE_CLOCK_RECOVERY.md`.
2. Connect the historical data reader, gap/source masks, transformations and
   TRAIN-only scaling to the weekly partitions and masked trainer. Check every
   development week and original-price naive for each horizon. TEST stays closed.
3. Complete and verify the actual shared DEAP/DOIN wire interface, persistent
   claims, resume identities, bounded costing and successor jobs. The local
   search component is not that distributed interface; see `SEARCH_COMPONENT.md`.
4. Measure an actual historical cell before sizing campaign leases or quoting
   a training ETA. Use the full approved population, not a tiny scientific proxy.
5. Connect verified C2 receipts to the live warehouse with idempotent readback.
   Local synthetic receipts are not proof of live OLAP publication.
6. Repair/test chronological TP-versus-SL strategy behavior and forced weekend
   close before reporting realized strategy use (`U_h`) or trading utility.

Implementation of independent items may proceed concurrently; unresolved
source time must not be hidden by renaming a retrospective experiment as valid.
