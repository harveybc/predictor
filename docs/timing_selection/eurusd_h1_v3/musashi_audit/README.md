# Independent v3 Audit Return

Decision: **REVISE_BEFORE_BUSINESS_USE**. Start with AUDIT_RESULT.md and the
bounded SATOSHI_CORRECTION_ORDER.md. DECISION.json binds the original manifest;
it intentionally does not grant ACCEPT_SELECTED_SET for operational consumption.
PROBE_RESULT.json is generated evidence, not a new model experiment.

Reproduce the finite read-only probes with1GiB RAM/one CPU and no GPU:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=<trading-contracts>/src <analysis-python> probe_v3.py \
  --study <private-v3-study> --predictor <audited-predictor-checkout> \
  --strategy <audited-strategy-checkout> --admitted <admitted-eurusd.parquet> \
  --lake <original-lake.parquet> --output <new-output>/PROBE_RESULT.json \
  --replay-labels --replay-strategy
```

The negative freeze probe uses a temporary directory, changes cap4 to cap3 and
stops at model loading via a sentinel. It never fits a model, publishes a fake
scientific result or modifies the source study. The predictor argument points
to the exact audited tools; the strategy checkout is the pinned evaluator code.

Independent tests: predictor tools/test_timing_selection.py20 passed; strategy
test_short_horizon_timing_eval.py plus test_short_horizon_timing_adapter_contract.py,
19 passed with3 tiny-fit cases excluded. Configure the real expert module and
digest as documented in the audit request. No duplicate scientific fits needed.

The warehouse connection contract passed to publish_probe.py supplies only the
existing publisher settings. Dataset identity is recorded from audit hashes;
no RL training dataset is falsely attached to this audit. Credentials remain in
the configured private environment. Local resource remediation is documented
separately in HOST_MEMORY_RECOVERY.md.
