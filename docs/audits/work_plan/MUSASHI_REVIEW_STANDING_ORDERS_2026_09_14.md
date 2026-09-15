# Review of the standing-orders return

Date: 2026-09-14. Reviewer: Musashi.
Reviewed predictor: `90d4fcf5ade5060a3f82b2c4fa8f10231521fe94`.
Disposition: operational transport accepted with limited scope; causal processing
and reproducible replay require the corrections below. No scientific promotion.

## Findings, in priority order

### 1. The causal battery does not exercise the claimed production operators

[test_causal_processing.py](https://github.com/harveybc/predictor/blob/90d4fcf5ade5060a3f82b2c4fa8f10231521fe94/tests/test_causal_processing.py)
defines its own `zscore_fit`, `windows`, `centred_mean`, `trailing_mean` and
`response_delay`. Its symmetric three-tap convolution is not the deployed wavelet
implementation. These are useful educational controls, not evidence that the
consumer plugins meet P3. The real scaler could regress without these tests
noticing. First response to a step also does not measure frequency-dependent
filter delay or prove general prefix invariance.

The window test compares target VALUES with feature VALUES. Repeated values are
legitimate; leakage must instead be checked using observation identity, timestamp,
availability and horizon.

### 2. Column-role assertions have two executable counterexamples

[app/column_roles.py:99](https://github.com/harveybc/predictor/blob/90d4fcf5ade5060a3f82b2c4fa8f10231521fe94/app/column_roles.py#L99)
defaults `allow_target_as_feature` to True. The report promises explicit opt-in.
Also, a numeric column can be both metadata and a feature and reach the model.
Both were reproduced through `resolve` and `select_features`:

```python
import pandas as pd
from app.column_roles import resolve, select_features

cases = [
    {"features": ["x", "y"], "targets": ["y"]},
    {"features": ["x", "available_time"], "metadata": ["available_time"]},
]
for contract in cases:
    names = contract["features"]
    frame = pd.DataFrame({name: [1., 2.] for name in names})
    plan = resolve({"column_roles": contract}, frame.columns)
    print(select_features(frame, plan).columns.tolist())
# Accepted: ['x', 'y']; ['x', 'available_time'].
```

Historical values of a variable can legitimately forecast its future values.
Explicit permission alone, however, does not establish that temporal alignment.
This finding is about the contract, not a claim that every shared variable leaks.

### 3. Replay custody and metric meaning are incomplete

The committed [production receipt](https://github.com/harveybc/predictor/blob/90d4fcf5ade5060a3f82b2c4fa8f10231521fe94/docs/audits/evidence/repro_runs/doin_offline_20260914/GOVERNED_OFFLINE_REPLAY_PRODUCTION.json)
names `e28c51870d2d420b81edfdb23ce0f542102d1e93-dirty`, not a complete source
identity. The flattening/hash fix is still an uncommitted diff in the inspected
agent-multi tree. Preserve it; do not revert unrelated files to obtain a clean tree.

The inspected `tools/governed_offline_replay.py:83-91` accepts a pre-existing
`summary.json` and otherwise reads repository-global `config_out.json` as metrics.
It does not establish that either belongs to this attempt. A configured step budget
is not automatically a measured step count. The terminal really contains 64, but
this audit does not certify 64 actual executed steps. The receipt also refers to
external template/overlay paths without separately binding both source files.

The runner explicitly excludes DOIN network execution. Its success establishes
governed agent-multi offline plumbing, not distributed DOIN integration.

### 4. The published test fixture is not self-contained

The default focal invocation below produced **24 passed, 9 errors**: the committed
manifest exists while `causal_train.csv` is absent, so the fixture does not regenerate.
Selecting a fresh, absent `CAUSAL_BENCH` path produced **33 passed**.

```bash
python -m pytest tests/test_column_roles.py tests/test_causal_processing.py \
  tests/test_governed_run_classification.py -q --tb=short
```

Both runs used the existing Python 3.12 trading-stack environment. This is a focal
check, not a full-suite or clean-install claim. Passing the 33 tests does not cure
findings 1 and 2.

## Independently checked operational facts

Read-only PostgreSQL queries found **35 terminals**. The three production replay
attempts are COMPLETE/FAILED/COMPLETE (stored status uses `COMPLETED` for successes).
The final replay has `wall_seconds=7.113479464984266` as a metric,
`7.163661383005092` as outer execution cost and `total_timesteps=64` as reported.
The distinction between inner and outer wall time is legitimate and must be named.

All four store/governance services and the loader are active/running with
`NRestarts=0`. This review did not restart services or alter the cube. Service
activity is a point-in-time observation, not a new throughput/heartbeat proof.

## Scope and continuation

Keep the four-consumer production acceptance and all historical receipts. The
first replay with no metrics is history, not something to overwrite. Per-wrapper
pending recovery and archive end-to-end remain genuinely unfinished.

File-grain reception, if supported by the producer log and matching artifact,
does not demonstrate row-level availability or publication. A second download
can measure a difference between snapshots, not a provider's revision policy.
Primary-source terms research belongs to the agent first; only an actual required
agreement or missing entitlement belongs to the owner. No rights are inferred here.

Execute the [next order](../../handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md).
