# Live integration and TRAIN pilots

2026-10-12 UTC. No annual validation or TEST result is claimed here.

## Exact fixes to adopt

- `5bff3b21`: the real admitted launcher drops inherited stdin. Reproducer
  returned an empty string with exit 0. The new inner request relay reads a
  private SHA-256-checked file after admission and execs the original evaluator.
  The same real launcher then returned the exact request. No resource control
  was bypassed. 26 runner/bridge tests passed; 232 pilot/trajectory/search tests
  passed. The script records PRE failure and POST success in the retained tool
  outputs; this is not model-quality evidence.
- `0f1f08de`: deploy distinct, bounded TRAIN cost pilots through that launcher.
- `e242c846`: generated production node configs explicitly select E's v2 joint
  optimizer, bind all C folds and origins, and carry admitted classes. A missing
  joint plan is refused. Both regressions failed before; 39 integrated
  optimizer/config/agent tests passed after. This addresses E-B1, not every
  finding in REVIEW_E_OF_B.md.

Use cherry-picks or equivalent reviewed edits in the active integration branch;
do not deploy an older copy of runner.py over the transport repair. Runtime
code of the pilots is pinned to 0f1f08de and must not be overwritten mid-run.

## New completed measurements

Both used the 20 RAW features and C's 2023Q1_DCA fold, full rolling TRAIN,
inner stopping, seed 6102026, maximum 50 epochs and patience 5. No outer scores.

| Workload | Device class | Updates | Best epoch | Fit wall seconds | Cgroup peak bytes |
| --- | --- | --- | --- | --- | --- |
| h12, history 48 h, dx2, 25 samples, trajectory 3/6/9/12, width32, two Transformer blocks, three Conv32 reducers, LSTM32 adapter | rtx5090 | 666 | 4 | 297.800 | 2410315776 |
| h2 minimal Conv8 -> Conv4, history24 h, dx2, 13 samples, endpoint | rtx4090 | 525 | 2 | 107.604 | 1944051712 |

The minimum model was measured on the secondary and the larger S1 workload on
the preferred. These are not a two-device speed comparison. Peaks certify the
exact workloads, not an unmeasured universal envelope. Retain 1.25 safety
factor and fresh host/slice admission. Neither pilot may be rerun under a new
label just to obtain another score.

Private manifest and consolidated status: `/var/tmp/i6n-launch-cost-20261011/`.
Completed model, prediction, C2 and measurement artifacts are mirrored under
`mirror/preferred/` and `mirror/secondary/`, with RESULT hashes verified.
The collector timer is `i6n-launch-cost-collect.timer`; no agent watches logs.
The two pilot services are `i6n-cost-{preferred,secondary}-0f1f08de.service`.

## Remaining actual launch dependencies

E's review names E-B2 through E-B7: reproduction seed, passing the real plan
to Bridge, required origins binding, ledger receipt binding, typed collection
refusals and isolating failures. B is actively correcting them. Integrate and
test these with real modules; then use the two-real-node smoke and production
start already authorized. Do not wait for another user message.

Also exercise cross-host artifact visibility (same paths are not shared bytes),
class promotion, and C annual rollup against A's per-week identities. Keep the
missing multi-output arms explicit; do not report them as delivered.

The entire swarm is NOT running merely because the two pilots completed.
Publication in the chain and OLAP also remains separate from local completion.

## Publication and next pilot

The first two pilots are now OLAP_READBACK_VERIFIED: 40 reports / 310 metric
rows, including all intermediate trajectory heads and their own paired naive.
They carry TRAIN_COST_PILOT, not annual validation or search-winner claims.
`chain_finalized=false` is explicit. The old search verifier incorrectly called
single-head preparation for a v2 trajectory; a failing regression and the real
retained trajectory exposed it. Preparation is now trajectory-aware for v2;
the four search-publication tests pass.

The next measured workload was started on the preferred GPU with pin
`da32dee6`: h36, history216 h, dx4, 55 samples, trajectory9/18/27/36,
two Transformer64 blocks, three Conv64 reducers, LSTM64 adapter. Its private
root is `/var/tmp/i6n-launch-cost-S2-20261011/`. No S1 fit was repeated.
The same timer collects both roots and publishes completed pilot metrics
automatically, under a bounded CPU scope. S2 subsequently COMPLETED with 490
updates, restored best epoch 2, fit wall303.274 s and cgroup peak2390552576 B.
The S3 pilot is next, pinned `b3aa8421`: h72 / L432 / dx1 / n433 / width128 /
two Transformer blocks / three Conv128 reducers / LSTM64 / four heads. Its
base cell changes only max_samples145 ->433 (data and stopping unchanged),
and retains the 7936 MiB declared pilot cap. A refusal or OOM is terminal,
not a license to trim the model or memory margin. Root:
`/var/tmp/i6n-launch-cost-S3-20261011/`. Collector and publisher cover all roots.

Correction: the first S3 request failed BEFORE TensorFlow/updates with
`materialized bytes exceed resource limit`. Its max_samples was extended but
the inherited 400 MB materialization /20 MB batch guards were not. This was
a configuration defect, not an OOM or a measured resource refusal. The failed
attempt is preserved, no scientific score. Successor pin `2df243fc` first
preflights the full candidate on CPU and explicitly declares 2 GiB materialized
and64 MiB batch bounds; the actual admission cap remains7936 MiB, with the
same data, architecture, batch size, seed and stopping. New root:
`/var/tmp/i6n-launch-cost-S3-corrected-20261011/`. No failed fit was retrained:
the predecessor had zero updates. This correction is not permission for a
retry after OOM. The timer retains both dispositions separately.
