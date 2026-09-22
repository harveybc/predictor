# RP90-RP97 independent review

Reviewed `fb6ec31e28db6d7dc3434c86f18218ee3e5743f1`. Disposition:
**CHANGES_REQUIRED; FOUR MEASURED CELLS PRESERVED; REPRODUCTION PARTIAL.**

## Findings, ordered by impact

1. **High: altered metric vault accepted by the real closure.**
   `tools/df_sota_repro.py:1111` reuses a vault on schema and prediction digest
   alone, then hashes its current contents as if that established their validity.
   In a disposable copy of the real author's tiny training fixture, changing
   global MAE and the first per-step MAE to 999 leaves the unit VERIFIED, with
   no problems. The closure reports the digest of that altered file.
   This blocks the current prediction-deletion gate: it does NOT invalidate the
   real scores independently recomputed below. Bind accepted evidence and verify
   contents; newly hashing the candidate itself is not independent verification.

2. **High: replay cache trusts a contradictory success flag.**
   `tools/df_sota_repro.py:1149` trusts local REPLAYS.json when input identity
   matches. Retaining that identity and `allclose_rule=true`, but changing the
   maximum prediction difference to 999, still yields VERIFIED with no problems.
   The report is neither anchored to independent acceptance nor internally
   validated. Input hashes alone cannot authenticate an output report. The
   example is executed through `verify_sota_run`, not an invented helper.

3. **High for deletion: a short loader silently halves the error.**
   `metrics_vault`, lines 902-928, divides by prediction shape, with no final
   check that the loader consumed the same population. Ten predictions, but five
   supplied target windows, yield MAE 0.5 instead of 1.0 and still claim ten
   windows. The unfilled temporal residual buffers also participate in ACF.
   Require exact identities/cardinalities, dimensions, ordering and finite values
   before publishing any metric, with a typed refusal on incomplete input.

4. **Medium: undefined per-channel correlation is reported as zero.**
   `tools/df_sota_repro.py:961` clamps the variance product instead of requiring
   positive variance in both series. Constant predictions and constant targets
   produce global correlation null but per-channel correlation 0.0. This is not
   an observed zero association. Fix both constant-prediction and constant-target
   cases and supply reasoned undefined states. Also rename `mase_vs_naive`: its
   denominator is test persistence MAE, not the training difference scale of MASE.

5. **Medium: published uncertainty and replication grain are conflated.**
   Lines 83-89 explicitly borrow Table 7's four-horizon-average dispersion for
   each horizon. That is a predeclared operational margin, not a per-horizon
   error bar from the paper. Preserve the original rule/result with that scope;
   do not retroactively call it published statistical equivalence. The table
   average path, lines 1249-1250, would pass four horizon means to `agreement`,
   which labels their dispersion as seed dispersion. Calculate a four-horizon
   average within each matched seed, then dispersion across the three seeds.
   Executed table oracle: three identical seed averages of 2.5 (true seed SD
   zero) are reported with SD 1.290994 and `n_seeds=4`, the four horizons.
   No complete four-horizon average exists in the real campaign yet.

Executable PRE and measured results:
[probe.py](../evidence/RP97_MUSASHI_REVIEW_2026_09_22/probe.py),
[PROBE_RESULTS.json](../evidence/RP97_MUSASHI_REVIEW_2026_09_22/PROBE_RESULTS.json).
The first four findings and the aggregation oracle were executed on the reviewed worker checkout in a
3 GiB, no-swap, 300-second CPU scope. Only disposable fixtures were modified.
The unchanged focal suite also ran there: **20 passed, 1 skipped in 7.67 s**.
That green suite did not cover these counterexamples.

## Independent ML result, not just receipts

[reduce.py](../evidence/RP97_MUSASHI_REVIEW_2026_09_22/reduce.py) reads the delivered
official CSV and predictions on the worker. It independently constructs the
chronological row indices, train-only scaling, test windows, persistence and
seasonal-24 predictions. It uses neither the author's loader nor Satoshi's
metric function. StandardScaler remains the published scaler implementation.
Input file, prediction bytes, target bytes and cell records match their recorded
digests; all four vault digests also match the published REPORT.json.

| Horizon | n seeds | Published MSE / MAE | Independent mean MSE / MAE | Persistence MSE / MAE | Seasonal-24 MSE / MAE |
|---|---:|---|---|---|---|
| 96 | 3 | 0.133 / 0.230 | 0.135496 / 0.232884 | 1.587837 / 0.945456 | 0.321095 / 0.325769 |
| 192 | 1 | 0.154 / 0.248 | 0.160857 / 0.255988 | 1.596174 / 0.950709 | 0.304320 / 0.323741 |
| 336 | 0 | 0.162 / 0.261 | NOT_EXECUTED | NOT_MEASURED | NOT_MEASURED |
| 720 | 0 | 0.184 / 0.284 | NOT_EXECUTED | NOT_MEASURED | NOT_MEASURED |

All errors use the paper's train-standardized, all-window/step/channel space.
H96 sample SD across seeds: MSE 0.003973, MAE 0.004238. H96 MAE is 28.51% below
seasonal-24 and differs from the rounded paper value by +0.002884. This is a
substantive benchmark result, not a trading return or evidence about UCI235.
The public test has been evaluated by the author's training loop; it is not an
untouched private financial confirmation set. Do not tune on its scores.

The four cells differ from the author's recorded float32 MAE by at most
2.06e-8; their independent per-channel MAE differs from the vault by at most
1.07e-8. This does NOT certify ACF, MI, quantiles, tails or every extended metric.
Full output and scope:
[INDEPENDENT_REDUCTION.json](../evidence/RP97_MUSASHI_REVIEW_2026_09_22/INDEPENDENT_REDUCTION.json).
Measured audit: 23.34 s wall, 23.63 s CPU, 2,167,872 KiB peak RSS under a 4 GiB
scope. No fit/replay of a benchmark model was launched by this audit.

No new live warehouse query was made by Musashi this turn. Accepted-chain
statements remain Satoshi's published evidence, not a claimed independent live
reconciliation. No predictions, checkpoints or production records were deleted.

## Replay and resource disposition

Keep the original CPU failures. Same-device GPU equality and near-equal aggregate
metrics are separate observations, not permission to erase those failures.
Top-p routing as the cause remains a hypothesis until route-level comparisons
locate it. Metric equality alone cannot certify individual predictions.

The author evaluation retains lists and concatenates full inputs/predictions/
targets; the audit implementation itself also constructs full target tensors.
The reported 20+ GiB need therefore describes these paths, not a demonstrated
minimum memory requirement of TimeFilter. Investigate a disk-backed, bounded
evaluation/verification adapter, with actual numerical parity before adoption.
Do not change model, input window, batch size, precision, optimizer or training.

WORKER_A observation: about 11.1 GiB available, a 14 GiB batch slice with about
4.0 GiB already charged; GPU 31 C at sampling. Those values are not reservations.
No VM shutdown, no GPU reset, no change to slice limits. The coordinator's travel
hold persists until cooling is explicitly restored, not automatically on Sep 24.

## Primary sources

[TimeFilter, appendix A.3/B.1/B.2](https://arxiv.org/html/2501.13041v2): Table 6
specifies Electricity patch32, two graph blocks, lr1e-3, d_model/d_ff512 and
15 epochs; the text specifies Adam and batch16. Table 7's Electricity row is
the aggregate 0.158/0.256, not H96's 0.133/0.230. Tables 8 and 9 distinguish
fixed L96 from searched lookback; do not identify all Table 9 results with L512
without establishing the per-cell author recipe. This is the selected published
reference, not an independently established best model across all 2026 papers.

## Next action

Execute [RP98-RP105](../../handoffs/MUSASHI_SOTA_RP98_RP105_2026_09_22.md).
No owner decision is needed to correct these defects or pursue bounded execution
within already authorized workers and storage. Report a measured capacity deficit
only after testing the bounded path, without substituting a smaller experiment.
