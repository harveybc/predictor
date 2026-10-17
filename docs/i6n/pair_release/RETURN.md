# Separate Pair Retention And Business CSV Handoff

2026-10-17. Fit-free retention and interface work, not a new forecast experiment
or an economic-performance result. TEST was not read. The secondary window/core
campaign was not changed, stopped or redeployed.

## Retained References

Pair identity: `0072c750018f3a8efe7966a71b084f845164325c8c7a71bbbd28698aaee9ec83`.

| Family | Reference | Output eligibility | Retained checkpoints | CSV rows |
|---|---|---:|---:|---:|
| Short | `weekly-window-core:ANNUAL_window13` | 7/7 actual outputs beat own naive | 51 | 4456 |
| Long | `weekly-long-only:ANNUAL_long_only_weekly` | 6/6 actual outputs beat own naive | 51 | 4514 |

The short reference is the existing seven-output model, not a silently retrained
six-output variant: its 18h head is preserved in the model, but the short CSV
contains only 2/3/4/5/6/12h. The long CSV contains 18/24/36/48/60/72h from the
long-only reference. Distinct original populations are preserved; each MAE is
paired to its horizon's naive on its exact own rows. Common-support ranking and
native metrics remain separate in the registry. All-output eligibility is a
strict prerequisite; averages can only rank models that already passed it.

| Horizon | Short native relative error | Long native relative error |
|---|---:|---:|
| 2h | -0.213308% | -- |
| 3h | -0.712522% | -- |
| 4h | -0.867257% | -- |
| 5h | -0.797215% | -- |
| 6h | -1.555002% | -- |
| 12h | -3.267697% | -- |
| 18h | -4.571488% (retained extra head) | -1.822553% |
| 24h | -- | -3.220871% |
| 36h | -- | -8.381802% |
| 48h | -- | -15.674639% |
| 60h | -- | -23.735451% |
| 72h | -- | -24.564042% |

These are complete recipes, not a best-head mosaic. The per-horizon historical
leaderboard is retained independently. Neither recipe is guaranteed Pareto-best
at every horizon. The original full twelve-output result remains immutable.

## Saved Models

All 102 original `model.weights.h5` checkpoints were retained with config,
scales, training history, seed, periods and source digests. They were converted
to full `forecast_model.keras` archives without a fit. Every archive was loaded
with safe deserialization; its tensor hash matched the original restored hash,
and a synthetic CPU forward matched the reconstructed model exactly. Actual
TF2.21/Keras3.13 runtime, source identity and hashes are recorded per archive.
Conversion took 90.3s total; governed cgroup peak was 929.3MiB, swap zero.

Canonical scientific scoring remains rebuild+load_weights; archive compatibility
does not claim bitwise GPU cross-process forward equivalence. Future fits using
the new continuation code also save full .keras archives. Running immutable
scientific runtimes were not patched to add that artifact retroactively.

Two physical copies, on the artifact volume and the separate archive worker,
were verified. Large files remain outside git. Two governed family-retention
reports reached OLAP with exact metric readback: delivered2/expected2,
pending0/mismatch0. Their tags bind profiles, pair manifest and CSV hashes.

## Strategy Interface

`heuristic-strategy` commit `3a9c55b59ec314e1e3e14dbccac1d30a7cd657f9`
adds a strict named CSV loader. `run_csv_backtest` pins that blob and feeds the
existing business executor/policy, not the positional legacy plugin. Original
UTC timestamps and close prices aligned exactly for both real CSVs. Availability
is calculated from the calendar and finite predictions, not future-label masks.
Neither prices nor missing forecasts are forward-filled. The base OHLC stream
must keep bars without forecasts so risk controls and force-flat still execute.

Synthetic integration verified forced closure without later forecasts, no open
position at week end and no execution failures. Policy regression checks cover
long-only entry, chronological SL/TP management, named masks and fill deadlines.
The operational window remains market-open+12h to market-close-12h, with the
earlier entry cutoff preserved. The supplied calendar remains explicitly
assumed rather than broker-certified; no claim of verified brokerage costs or
realistic positive profit is made. No real-data strategy simulation ran here.

## Automation And Status

`i6dmo-best-registry.timer` now emits strictly eligible family events under a
new rule-bound cohort identity; historical average-based records are not erased.
The last long-only failed period has a supplemental zero-label proof, authenticated
against prepared arrays. Its original FAILED receipt remains untouched. No
completed fit was repeated merely to change the disposition.

`i6dmo-pair-retention.timer` performs bounded transitions: retain the eligible
pair, adopt or launch fit-free archive conversion, verify two physical copies,
verify CSV alignment, publish/read back and expose generated status. It uses
768MiB coordinator memory, zero swap and no GPU; conversion uses a separate
bounded CPU service. Source credentials are inherited through EnvironmentFile,
not read or printed. Agents do not poll logs.

```bash
python -B -m tools.i6dmo.pair_release status --root "$ROOT"
python -B -m tools.i6dmo.pair_release follow --spec "$SPEC"
python -B -m tools.i6dmo.pair_release status --root "$PAIR_ARTIFACTS"
```

The second `follow` adopts verified archives and retains the same two reports,
without fitting. Its status distinguishes retained assets from TEST readiness.

## Verification And Remaining Work

Predictor final focused suite: 129 passed, including follower refusal/adoption
and bounded timer cases. Heuristic CSV/policy: 26 passed, 1 skipped. Synthetic
and real-file acceptance are not an independent review of profitability.

TEST remains blocked on the selected inputs' point-in-time evidence and a
frozen prospective weekly evaluation procedure that NEVER uses the scored TEST
week for checkpoint selection. Prior 2024 weekly checkpoints can reproduce 2024
development, but are not a prospective sequence of 2025 weekly fits. New as-of
fits for such a sequence are business retraining, not recovery of lost weights.
Resolve those technical requirements before TEST; do not ask for per-week chat
approval or weaken the original business mask to manufacture support.
