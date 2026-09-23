# RP127 reviewer execution

Reviewed revision: `a73e55b2dcad1265b7643f49fcfda810fcb7d801`.
Date: 2026-09-23. WORKER_B CPU only; no production writes or GPU work.

## Probe

Executed `probe.py --repo <reviewed-checkout> --output <temporary-output>` with
the worker's `trading-stack` Python, under the existing `crispdm-run` wrapper
(3 GiB, 240 s), CUDA hidden, OMP/OPENBLAS/MKL threads each one, torch threads one.
Exit 0; full output retained in `PROBE_RESULTS.json`. The tiny actual-author
fixture trains for two CPU epochs; it is not a new scientific benchmark cell.

| case | observed |
|---|---|
| valid numerical catalog | pass, fully_independent true |
| producer emits wrong zero ACF, reference invoked | refused, discrepancy 0.9428690469 |
| same wrong ACF, default catalog call without data path | pass, no reference, no accepted families |
| normal deletion with that report accepted and backed up | COMPLETE; disposable array removed |
| foreign-design accepted terminal, explicit expected design | refused |
| same foreign-design evidence through actual deletion | COMPLETE; disposable array removed |
| design absent in both authoritative declarations, explicit expectation | accepted |
| histogram numeric zero replaced by boolean false | full catalog passes, numeric difference 0 |

Fixture terminal identities/receipts were recomputed consistently for the
foreign/missing-design cases; this is a relationship check, not a corrupted-hash
experiment. No original benchmark artifacts were mutated or deleted.

## Focused suite

Same admitted worker wrapper (3 GiB, 240 s) and one-thread/CUDA-hidden environment;
`PYTHONPATH=<reviewed-checkout>` with its own process:

```sh
python -m pytest <reviewed-checkout>/tests/test_df_sota_repro.py -q -rs
```

Exit 0: **98 passed, 1 skipped, 5 warnings in 90.06 seconds**.
Queried worker environment: Python 3.12.13 (Anaconda), NumPy 2.5.1,
torch 2.13.0+cu130, with CUDA hidden for both verification runs.
Skip: `benchmark store receipt absent` at test file line 195.
Warnings: `multiprocessing.popen_fork` with a multi-threaded process, in the
DataLoader and training-trajectory parity tests. These warnings were not counted
as failures. The suite does not include the new review probes as regressions yet.

The full repository, warehouse-engine suites and full production numerical
reductions were not rerun by this reviewer. Retained RP127 acceptance examples
were inspected as JSON, not treated as fresh independent measurements.
Documentary coverage validator passed with `scientific_approval: false`.
