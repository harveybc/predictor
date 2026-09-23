# Independent verification scope

Reviewed revision: `316c67af0af317c345c6959681135ba622770584`.
Date: 2026-09-23. Execution: WORKER_B, CPU only, no production mutation.

## Executed probe

`probe.py --repo <reviewed-checkout> --output <disposable-path>/probe.json`

Executed inside the existing `crispdm-run` wrapper, 3 GiB / 240 seconds,
`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`; torch thread count set to one inside the probe.
Exit status 0. Output retained as `PROBE_RESULTS.json`.

The real test fixture trains the pinned author's tiny model for two CPU epochs,
then exercises actual closure/catalog/deletion functions on disposable copies.
Warehouse responses are immutable fixture data where the probe specifies that;
this is not a live-warehouse integrity audit. No production predictions deleted.

| probe | observed |
|---|---|
| valid catalog | pass, no refusals |
| wrong plausible ACF/quantiles and absent defined correlation | pass, no refusals |
| diagnostic artifact requested as closure | refused before registry relabel |
| same immutable terminal, local kind/role/subject relabeled | accepted |
| deletion with empty accepted-chain response, default API | REFUSED |
| same inputs, require_acceptance=False | COMPLETE, disposable array absent |

## Focused suite

The same bounded wrapper and CPU/thread environment, using the worker's
`trading-stack` Python and `PYTHONPATH=<reviewed-checkout>`:

```sh
python -m pytest <reviewed-checkout>/tests/test_df_sota_repro.py -q
```

Exit status 0. **85 passed, 1 skipped, 5 warnings in 75.65 seconds.**
Interpreter/dependencies queried on that worker: Python 3.12.13 (Anaconda),
NumPy 2.5.1, torch 2.13.0+cu130, executed with CUDA hidden. The skipped test
is `test_RP92_the_lock_is_sealed_before_any_data_and_names_the_official_file_and_paper_values`:
the benchmark store receipt is absent on this worker; its fixture-based tests
are not a substitute for that production receipt check.
Five warnings are Python multiprocessing `fork()` after threads, in the
DataLoader/trajectory parity tests. The suite output and independent probe are
different evidence: a passing regression suite did not detect the new cases.
No full repository or store suite was rerun by this reviewer. No GPU replay,
benchmark fitting, current production metric recomputation or service restart.

The recorded return's 84 tests and this run's 85 are not combined or silently
substituted: this execution is pinned to the final reviewed commit above.
