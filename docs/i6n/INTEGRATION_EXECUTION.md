# I6-N integration and execution

Date: 2026-10-11. Owner authorized autonomous implementation and execution.
The C1/C2 component acceptance is NOT acceptance of the historical campaign.
No further owner approval is required for ordinary engineering steps below.
Do not resume the paused I6-E-M queue as a substitute for I6-N.

## Delivered code and traceability

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Per-head calendar E/M, exact endpoints | audited C1, unchanged | `test_i6n_eligibility.py` |
| Naive for each horizon, original-price pairing, coverage | audited C2, unchanged | `test_i6n_paired_metrics*.py` |
| Exact C1 population reaches C2; stored metrics are rescored | `business_metrics.py` | `test_i6n_business_metrics.py` |
| Current tick survives each reduction and cascade | `temporal_ops.py` | `test_i6n_temporal_ops.py` |
| Invalid labels removed before residual/gradient calculation | `masked_huber` | NaN and gradient tests in same suite |
| Configurable small branches, fusion, core, temporal head interface | `i6n_model.py` | `test_i6n_model.py` |
| Finite unattended verification and read-only status | `i6n_verify.py` | `test_i6n_verify.py` |
| Real GPU execution, weight updates, reload parity | `i6n_device_probe.py` | synthetic device probe; not historical cost evidence |
| Weekly partitions and purged per-head labels | `i6n_weekly_data.py` | `test_i6n_weekly_data.py` |
| Exact masked epoch metrics and owner early stopping | `i6n_training.py` | `test_i6n_training.py` |
| Calendar -> model fit -> original-price C2 scores | `i6n_weekly_smoke.py` | `test_i6n_weekly_smoke.py`, synthetic only |
| Joint hierarchical candidate search | `i6n_search.py` | `test_i6n_search.py`, local stub scoring only |

The core factory is additive: existing factories and audited modules remain
unchanged. It accepts Conv-only, Transformer and combined entrances, 1-3
Conv reducers with independent filters and temporal factors, and an optional
sequence-valued LSTM or Conv final adapter. Positional encoding follows fusion.
Reduction is right-anchored, not a left-anchored stride that can omit t.

`masked_huber` is now consumed by the additive `fit_masked` trainer. Empty-label
batches skip optimizer updates, including weight decay. Epoch metrics aggregate
per-head sums and counts. The owner's TRAIN/inner average controls stopping and
the best weights are restored; outer validation is not passed to this helper.
See [Weekly Integration](WEEKLY_INTEGRATION.md) for the finite commands and
remaining historical-data/runtime connections. The local DEAP component is
not a deployed shared DOIN optimizer.

## Repeatable finite verification

From this checkout in the configured Python environment:

```bash
python -m tools.i6n_verify run --output /var/tmp/i6n-verification
python -m tools.i6n_verify status --output /var/tmp/i6n-verification
```

Each suite obtains its own existing resource admission at 2 GiB / 120 seconds.
The sequence terminates, records logs/JUnit/digests and reuses verified completed
steps only under the same code/environment identity. Changed code requires a
new output directory. No agent needs to inspect logs periodically. `status`
is read-only and never means campaign readiness. Local hashes do not establish
authenticity against a fully rewritten directory; keep independent receipts.

GPU smoke, only after resolving a device UUID and fresh resource admission:

```bash
CUDA_VISIBLE_DEVICES=<one-approved-GPU-UUID> \
crispdm-run -m 6G -t 180s -n i6n-gpu-integration -- \
python -m tools.i6n_device_probe --output <new-artifact-directory>
```

This performs six updates of each of two small models on synthetic tensors.
It is NOT a market experiment, an early-stopping experiment, an estimate of
full TRAIN memory, or evidence that the model beats any naive. The configured
TensorFlow environment/library paths must be provided by the host inventory.

## Required successor sequence

1. Resolve source-clock provenance and the versioned instrument calendar.
   The audit addendum reports a one-hour discrepancy in US/EU DST-gap weeks
   after 2019. Do not guess a timestamp shift, infer closures solely from gaps,
   or reuse the old Monday-cutoff schedule. Preserve original data/evidence.
2. Materialize per-week TRAIN/inner/development masks using C1, four preceding
   calendar years, previous market close cutoff, current-origin input, exact
   target endpoint and per-output masking. Freeze comparison support before
   model scoring; retain actual artifact creation times, not backdated seals.
3. Integrate the new factory and masked loss with the existing best-weight
   training loop, C1/C2 adapter, model identity and OLAP publication/readback.
   Run a full-workload TRAIN cost pilot before deriving the 1.25x memory cap.
4. Implement the DOIN evaluator bridge and canonical hierarchical DEAP genome.
   Jointly optimize context, input resolution, core family/depth/filters,
   temporal reduction, final adapter and genuinely distinct output supervision.
   A scalar endpoint renamed into coarser ticks is not another experiment.
5. Generate and freeze bounded per-horizon configs, sequential stage barriers,
   cost-class identities, exclusive claims and successor-only queues. Evaluate
   short 2/3/4/5/6/12 h and long 18/24/36/48/60/72 h, with each horizon's naive.
   Honor minimum 24 h input history and compare identical support per horizon.
6. Run small downsampled candidates first, then larger hourly contexts. Use one
   seed by default; no duplicate fits by effective identity. Expand through
   all weekly development segments only after support/cost checks. TEST stays
   sealed until the procedure is frozen. Feature selection informed by 2024
   makes development results retrospective; it must remain labelled as such.
7. Repair and verify the strategy's chronological SL/TP consumption and forced
   weekly liquidation before any claim about U_h or trading performance.
   Forecasting opportunities S_h do not require live trading to be measured.

These are implementation dependencies, not requests for repeated approval.
The previous draft's incomplete search and transfer budget is not silently
promoted to a runnable config. Historical fitting remains unstarted until
the missing data-clock and historical-adapter/shared-search objects exist.

## Resource policy

- Preferred worker: 5090 only; no concurrent 5070 Ti job on its shared RAM.
- Secondary: 4090 only when fresh host and slice admission fits the measured
  workload plus its margin. Do not shrink a failed request to force admission.
- Coordinator: small CPU work, orchestration, OLAP; keep desktop reserve.
- Large artifacts on workers with disk room. Coordinator home was at 99%
  during this session; integration and small receipts use the root filesystem.
- Retire only identified completed-campaign dispatch timers. Kernel slab is
  not attributed to a service from process RSS; no blind cache drop or reboot.
- Git holds code/configs/summaries, not large datasets or database snapshots.

## Current scope

C1/C2 are merged from their reviewed branches. New integration primitives and
factory are candidates with tests; the finite verifier generates their latest
evidence. No DOIN swarm or historical I6-N fitting is claimed operational yet.
The old queue remains paused, with 220 completed, 46 failed and 3762 queued
dispositions at the observed snapshot. Those are historical results, not new
I6-N measurements.
