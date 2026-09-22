# RP97 external audit evidence

Audited revision: fb6ec31e28db6d7dc3434c86f18218ee3e5743f1.
Date: 2026-09-22. Executor: Musashi, cooled WORKER_A only for compute.

- `probe.py`: real tiny author-path fixture, actual closure counterexamples,
  vault numeric cases and pure table aggregation oracle. TemporaryDirectory
  removes only the fixture it creates. No production mutation.
- `PROBE_RESULTS.json`: measured output. A successful baseline precedes the
  altered-vault and contradictory-replay cases. Short-loader and correlation
  cases call the deployed metric function. The aggregation case supplies a
  mathematical fixture to the actual table function, not a real benchmark run.
- `reduce.py`: independent row enumeration and error/baseline reductions on the
  four retained benchmark cells. Shares the published StandardScaler library,
  not the author's loader/scorer or the verifier's reductions.
- `INDEPENDENT_REDUCTION.json`: measured output, hashes, numeric discrepancies,
  scope, CPU/wall time and RSS. No live warehouse read or extended-vault approval.

Worker commands (replace paths; scripts and environment already installed):

```sh
crispdm-run -m 3G -t 300 -n musashi-rp97-audit -- env CUDA_VISIBLE_DEVICES= \
  <python> probe.py --repo <reviewed-checkout> --output <audit-output>/PROBE_RESULTS.json
crispdm-run -m 4G -t 600 -n musashi-rp97-reduce -- env CUDA_VISIBLE_DEVICES= \
  <python> reduce.py --root <retained-run-root> --output <audit-output>/INDEPENDENT_REDUCTION.json
crispdm-run -m 3G -t 300 -n musashi-rp97-tests -- env CUDA_VISIBLE_DEVICES= \
  <python> -m pytest <reviewed-checkout>/tests/test_df_sota_repro.py -q
```

Focal suite independently measured: 20 passed, 1 skipped in 7.67 s. No claim
that the full mixed-framework suite passes. One first audit launch targeted an
older checkout lacking this test module and failed before the fixture ran;
the correct synchronized checkout's HEAD was verified before the successful run.
The probe was then rerun with the aggregation oracle added, with the same first
four outputs. Benchmark reduction ran once, without training or model replay.

Lightweight documentary checks after updating the plan: check_plan.py PASS
(`scientific_approval=false`), 31 unittest rules passed, git diff --check clean.
No prediction deletion, compression, new benchmark fit or service restart.
