# S07 — Strategy support corrections before scoring (2026-09-30)

Satoshi III, successor technical lead, lane S07. Order: `docs/handoffs/SATOSHI_POST_CONSOLIDATION_2026_09_30.md`
§"Agent assignments" item 3 and §"Strategy design correction before scoring", plus two coordinator amendments
(nothing above 1 GiB on the coordinator; pytest and full-CSV derivations on worker_b; adopt the prior lane's
tested work and report every divergence).

Code lives in `heuristic-strategy`, branch `satoshi/s07-strategy-support-20260930`, tip **`d08ae00`**
(base `origin/master` `5c87a25`). This report and its evidence live in `predictor`, branch
`satoshi/s07-strategy-support-20260930` (base `dc72170e`). Evidence directory:
`docs/audits/evidence/S07_STRATEGY_SUPPORT_20260930/` (shas in `EVIDENCE_SHA256.json`).

**Nothing was scored.** No PnL, pips, equity, metric or ranking was computed anywhere in this lane. No sweep,
no broker call, no horizon search, no B0. **NO_NEW_MEASUREMENT of financial or model performance.**

---

## 1. Implementation (what exists on the branch)

| Commit | Content |
| --- | --- |
| `92ebccf` | Cherry-pick of the prior lane's `1b25d31` (Retsu, `satoshi/strategy-support-20260930`): variant E resolved in `app/config.py`, `Plugin.plugin_params` and the `HeuristicStrategy.__init__` signature; `app/strategy_support.py` with `create_elapsed_hour_predictions` (unit **hours**, origins lacking an exact target bar excluded), legacy generators annotated `offset_unit = "rows"`, `calibration_set` (rejects timestamps ≥ cut), `cpu_reconciliation`; its 9 tests. Adopted as ordered. |
| `16bbe35` | `app/target_support.py` (derivation library), `tools/s07_target_support_purge.py` (purge table generator), `tools/s07_prior_access_scan.py` (access-accounting scanner), `config_replication_baseline_20260930.json` (baseline config resolving variant E), 11 new tests; `sweep_241_exit_variant` upgraded from `NOT_CHECKED` to `E` after reading the retained manifest. |
| `d08ae00` | Test fixture fixes found by the first worker_b run (missing subdirectories; spike-bar short entries), WFO fold key in the scanner. |

### 1.1 The derivation (`app/target_support.py`)

For each real arm of design `4bb763d` §2 (R1 = ANN short + ANN long on `phase_1_base_d3`; R2 = LSTM short +
ANN long on the same base; R3 = CNN short + CNN long on `phase_2_3_base_d3`, the shipped default config):

- **Origins** are the `process_data` alignment: base ∩ short ∩ long timestamps, strictly before the cut.
  The design's admission rule (H6 and H144 elapsed targets exist) is reported alongside as
  `design_admissible_h6_h144_in_dev`.
- **Elapsed latest target** = origin + 144 h; crosses when ≥ 2019-05-16 00:00. **Legacy row-offset latest
  target** = base row + 144 rows (`create_daily_predictions` steps `i + d*24` rows); crosses when that row
  is not a development row. Both are reported; either crossing purges.
- **Trade-exit / cost support per exit variant A–G**: an independent unit trade opened at the origin is replayed
  with the plugin's own `calculate_entry_geometry` (close-only decision at the origin close) and a
  vectorised transcription of `should_early_close` (parity with the scalar rule is a test); market fill at the
  next open with the launcher's fixed slippage `(2+1)·1e-5/2` per side, clipped to the fill bar as backtrader
  does; TP/SL checked at each later close (TP first, then SL, then the early exit, as in `next()`); exit fill at
  the bar after the signal. Cost support equals trade support (commission/slippage are paid at the two fills;
  swap accrues over the plugin's `duration = exit_fill_row − origin_row`). **Replay runs on development bars
  only**: a trade still open at the last development bar is `CENSORED_AT_CUT` and counts as crossing. Reserved
  rows are dropped at load; `DevelopmentBars.row_of()` raises `ReservedRowsAccessError` for any timestamp ≥ cut.
- The unit-trade reading replays every origin whose geometry fires. It is a **superset** of the sequential
  backtest's entries (which also blocks entries while a position is open and caps entries per five days), so the
  purge is conservative with respect to the sequential run.

### 1.2 Baseline resolution of variant E (`config_replication_baseline_20260930.json`)

The three readings the source contradicted: (1) `Plugin.plugin_params['exit_variant'] = 'E'`; (2) the nested
`HeuristicStrategy.__init__` signature default `'D'`; (3) the comment block "D = both must agree (DEFAULT …)".
**Fixed reading: (1).** Every launcher in `app/` (`evaluate_candidate`) and the retained native adapter read
`plugin_params`; the signature default is never consumed (every caller passes the parameter) and the comment is
prose. So E is what every retained result actually executed; the retained 241-cell manifest
(`run_out/native_conditional_20260927_full/manifest.json`, sha256
`958726415c09ced866c7d8164dfe4e49b99b4ccb8e0e5e7059fa26effdbba94f`) records `execution.exit_variant = "E"`
and `plugin_params.exit_variant = "E"` (read, not rerun). Readings (2) and (3) are aligned to E so that the
three agree; aligning them changes no retained result. `historical_run_recovered: false`; this is **not** a
recovered historical run that used D. Fill semantics stay `close_only_decision_next_open_market_fill`.
`use_protective_broker_orders: false`; `protective_orders_experiment:
PROTECTIVE_BROKER_ORDERS_EXPERIMENT_NOT_RUN` — protective orders remain a separately named experiment, not a
silent correction.

### 1.3 Timestamps

New predictions "1–6 hours" and "24–144 hours" come from `create_elapsed_hour_predictions` (unit hours; an origin
without an exact target bar is excluded). The legacy `create_hourly_predictions` / `create_daily_predictions`
remain, unit **rows**, still what `process_data` auto-generation calls, so the replication baseline is unchanged.
The purge table carries both bounds per origin.

### 1.4 Development-only calibration

`development_residual_statistics()` computes per-horizon residual scale and the residual correlation matrix from
origins whose every target is strictly before the cut and has an exact bar; it looks prices up only through
`DevelopmentBars.row_of`, which cannot return a reserved price. Encoded as a test (poison invariance plus
exclusion counting). It was **not** run on any committed file: no calibration was measured.

### 1.5 Divergences from the prior lane branch `satoshi/strategy-support-20260930` (tip `7a50f62`)

| Prior commit | Disposition | Reason |
| --- | --- | --- |
| `1b25d31` Resolve variant E and separate elapsed-hour targets | **Adopted** (cherry-pick `92ebccf`) with one change: `sweep_241_exit_variant` `NOT_CHECKED → E` (manifest read, sha above); its test updated. | Tested, consistent with the order. Its `derive_development_support` marks every origin `NOT_SEPARATED_BY_ORIGIN_CUT` and refuses to derive trade support; S07 derives it on development rows (§2). Both functions coexist; the purge table is S07's. |
| `781022a` Reject reserved target support and measure a synthetic elapsed-hour run | **Not adopted.** | Redeclares the cut tz-aware (`RESERVED_START_UTC`), incompatible with the naive timestamps of every committed file and with `1b25d31`; executes a synthetic micro-run recording trades, costs and a pending position (a synthetic measurement outside this order). Its calibration-admission idea is covered by §1.4 with a test. |
| `e7966f3` Corrige la convención de margen, costos y caja del plugin | **Not adopted.** | Rewrites `plugin_long_short_predictions.py` accounting (swap debited to cash on a timestamp clock, symmetric collateral, `COMM_PERC`). That changes the replication baseline the order says to preserve. It is a candidate for a separately named accounting experiment, decided by the owner, not merged silently. |
| `7a50f62` Añade el ejecutor sintético del barrido | **Not adopted.** | Executes a 44-cell synthetic sweep with PnL; this order forbids a sweep. Retained on its branch. |

---

## 2. Synthetic checks (tests)

`tests/unit_tests/test_s07_target_support_20260930.py` (11 tests) and the adopted
`tests/unit_tests/test_strategy_support_20260930.py` (9 tests). **20 passed, 0 failed** on worker_b under
`crispdm-run -q -W 900 -m 2G -t 20m -n s07-tests`, CPU-only, from a worktree of the pushed branch at `d08ae00`
(`pytest.log`). The first run at `16bbe35` failed 3 of 20 on my own fixture defects (subdirectories not created;
entry count that ignored the plugin shorting a bar whose close sits 30 pips above a flat long forecast); fixed in
`d08ae00`, no derivation code changed.

Ordered tests, by name:

- **Purge derivation** — `test_purge_removes_origin_whose_trade_is_still_open_at_the_cut` (a short whose
  stop/target are never reached on flat prices is censored at the last development bar and purged although its
  144 h target is before the cut, `purge_reasons == "TRADE_SUPPORT"`);
  `test_purge_keeps_origin_whose_trade_and_targets_end_before_the_cut` (TP exit at bar 3, fill at bar 4, all seven
  variants, kept); `test_purge_removes_origin_whose_elapsed_target_crosses_even_if_trade_exits`;
  `test_no_entry_origins_are_purged_only_by_target_support`.
- **Reserved rows never read** — `test_reserved_rows_are_never_read_poison_invariance` (every reserved price set
  to NaN/9.9 and 500 reserved rows added: the per-origin table and the summary are byte-identical; only the file
  sha differs); `test_reserved_lookup_raises_and_bars_hold_no_reserved_price`.
- **Variant E** — `test_variant_e_resolution_is_explicit_and_consistent_in_config_file` (config file, defaults,
  plugin_params, signature and the removed comment agree; `historical_run_recovered` false; protective orders
  off); `test_variant_e_rule_is_the_weighted_minimum_with_empty_family_fallback`.
- **Development-only calibration** — `test_calibration_uses_development_rows_only` (poison invariance; the six
  origins whose ≤ 6 h targets reach the cut are excluded and counted).
- Parity — `test_unit_trade_early_exit_masks_match_should_early_close`; population —
  `test_population_target_support_counts_reserved_origins_already_present`.

---

## 3. Measured results of the derivation (support only; no financial quantity)

Generated by `tools/s07_target_support_purge.py` on worker_b under `crispdm-run -q -W 900 -m 2G -t 20m -n
s07-purge` (`purge.log`; 19.9 CPU s, 200.6 MiB peak RSS). Files: `purge_summary.json`, `purge_table.csv.gz`
(18,207 origin rows), `purged_origins.csv` (the 266 purged rows), `population_retained_sweep_241.csv.gz`.
Input identities (also inside `purge_summary.json → files`):

| File | sha256 | rows | reserved rows dropped at load |
| --- | --- | ---: | ---: |
| `tests/data/phase_1_base_d3.csv` | `c5a8d9f3d0ade7956974b59cd85ba37505053cf849f8f944a3081a19058923bf` | 11,378 | 5,211 |
| `tests/data/phase_2_3_base_d3.csv` | `4b57164f54d03fe25de4796b75c367a15e2b1d90c89b0a40d831e5d65cd4b6e1` | 18,475 | 5,211 |
| `tests/data/ann_predictions_daily_d3.csv` | `c01341e8ebb1f1cf3128616b6eee2d7c32fd828ba5ed859101f20fae66ad34e5` | 6,156 | 0 |
| `tests/data/ann_predictions_hourly_d3.csv` | `612fd3170086a6d14bc138340f26d19c06790bb6cc97e179d079224aef9f7493` | 6,294 | 127 |
| `tests/data/lstm_predictions_hourly_d3.csv` | `4f4ed2d99706b64fa1fbaeb82169eeeb873c18596a80557ee1058b0c32bb9d2a` | 6,288 | 121 |
| `tests/data/phase_2_3_cnn_1h_prediction_d3.csv`, `..._1d_...` | in `purge_summary.json` | 6,033 / 5,895 | 0 / 0 |
| retained sweep `origins.csv` | `7b7ad5edbe54583f001ee8750988dfa44685e976623dc798333adb5aca153c81` | 13,590 | — |

### 3.1 Purge table, real arms

| Arm | Aligned dev origins | Design-admissible (H6 & H144 in dev bars) | With entry signal | **Purged** | Kept | by elapsed target | by legacy row offset | by trade support (any variant) | First purged | Last kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| R1 | 6,156 | 4,491 | 6,156 | **133** | 6,023 | 85 | 133 | 1 | 2019-05-08 00:00 | 2019-05-07 23:00 |
| R2 | 6,156 | 4,491 | 6,156 | **133** | 6,023 | 85 | 133 | 1 | 2019-05-08 00:00 | 2019-05-07 23:00 |
| R3 | 5,895 | 4,393 | 5,895 | **0** | 5,895 | 0 | 0 | 0 | — | 2018-03-21 19:00 |

Derivation of the counts: origins ≥ 2019-05-10 00:00 have origin + 144 h ≥ cut (85 origins in the ANN daily
file). The legacy bound is wider because rows are not hours: with the 11–12 May 2019 weekend gap, origin
2019-05-08 00:00 + 144 rows already lands past the last development row, so 133 origins cross under the
row-offset reading, and the purge is the union. Every one of the 6,156 R1/R2 origins fires an entry: at
`profit_threshold = 5` (half a conventional pip at `pip_cost = 1e-5`) one long-family extremum always clears the
threshold, which is the tenfold-overstated-threshold fact of the design, now observed on the real inputs.

The design's admissible counts (4,546 for R1/R2) were computed against the **full** base file: 4,491 remain when
targets must be development bars, so **55 of the design's R1/R2 admissible origins had their H144 target inside
the reserved window**. R3's 4,393 reproduces exactly.

### 3.2 Trade-exit support and censoring, per variant (derived, not assumed)

| Arm | Variant | TP / SL / early exits | Censored at cut | Max observed plugin duration | Max elapsed fill-to-fill | Censored lower bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R1 | A | 1,375 / 4,394 / 386 | 1 | 57 bars | 104 h | 12 bars |
| R1 | E | 1,452 / 4,614 / 89 | 1 | 57 bars | 104 h | 12 bars |
| R1 | G | 1,470 / 4,685 / 0 | 1 | 57 bars | 104 h | 12 bars |
| R2 | E | 1,470 / 4,682 / 3 | 1 | 57 bars | 104 h | 12 bars |
| R3 | A | 564 / 4,268 / 1,063 | 0 | **546 bars** | **771 h** | — |
| R3 | E | 752 / 4,597 / 546 | 0 | 546 bars | 771 h | — |
| R3 | G | 923 / 4,972 / 0 | 0 | 546 bars | 771 h | — |

All seven variants per arm are in `purge_summary.json → arms → per_variant`. Reading:

- **A six-day purge does not bound trade duration.** On R3 a unit trade lasts up to 546 bars (771 elapsed
  hours, about 32 days) before a close-only barrier; on R1/R2 up to 57 bars (104 h). The bound is an observed
  maximum on development rows for these inputs, **not a cap**; the plugin has none (`stop()` closes only because
  the feed ended).
- **Censoring, R1/R2:** exactly one origin per arm, the last one (2019-05-15 12:00), is still open at the last
  development bar under every variant, with at least 12 bars held. It is purged by all three criteria anyway.
  Every origin ≤ 2019-05-07 23:00 exits before the cut under every variant, so for the real arms the target
  bounds are binding and trade support adds nothing beyond them.
- **Censoring, R3:** none; its window ends 2018-03-21 and its longest trade closes within development rows.

### 3.3 Retained 241-cell sweep population (synthetic arms of design (b))

Design §6.1 states this population is "all ≤ 2019-05-15". **Refuted by the population itself**: 13,590 origins,
`origin_first` 2017-03-22 18:00, `origin_last` 2020-03-13 06:00; **3,738 origins are reserved** (≥ cut). Of the
9,852 development origins, 66 cross by elapsed target and 114 by legacy row offset; **9,738 are kept**, last kept
2019-05-07 23:00. Trade support for synthetic arms is **NOT_DERIVED** (their predictions are generated per cell) and
is therefore **predeclared**: any successor run truncates the price feed at the last development bar and reports
every position open at that bar as censored, per cell, alongside the 12 naive denominators.

### 3.4 Boundary: keep the cut, move the development end

The reserved cut **2019-05-16 00:00 is not moved**. The derived last development origin with a clean prefix under
every criterion is **2019-05-07 23:00** (R1, R2 and the sweep population agree; R3 is unaffected). Under the
elapsed-hour reading alone it would be 2019-05-09 23:00; the row-offset legacy reproduction needs the earlier
value, and the union is what the purge table applies. Purged development origins: **133 (R1) + 133 (R2) + 0 (R3)
= 266** real-arm rows, **114** synthetic-population rows.

---

## 4. Prior-access audit of the reserved rows

Claim in `4bb763d` §6.2: the reserved window 2019-05-16 00:00 … 2020-04-29 22:00 holds "no rows currently in this
set … no result from it exists". **REFUTED.** Evidence: `prior_access_scan.json` (generated by
`tools/s07_prior_access_scan.py` on the coordinator, streaming first fields only, under `crispdm-run -m 1G`),
cross-checked by an independent read-only sweep of the repositories.

| Where | What was read | Evidence |
| --- | --- | --- |
| Retained noise sweeps, 12 manifests under the owner's `heuristic-strategy/run_out/` (untracked), 2026-09-27 | `phase_2_3_base_d3.csv` (sha `4b57164f…`) as `input.csv`, 18,475 bars, **origin_last 2020-03-13 06:00**; the sweep the design cites (`native_conditional_20260927_full`, manifest sha `9587264…94f`) scored **3,738 reserved origins in every one of its 241 cells; 237 of 241 cell `trades.csv` files carry reserved timestamps** (240/241 in `conditional_noise_…_full`, 63/64 in the `price_noise_*` runs) | `prior_access_scan.json → retained_manifests` |
| The design's own §2 characterization (`4bb763d`, 2026-09-29) | R1/R2 admission by H144 target existence against the full base file: 55 admitted origins had H144 targets in the reserved window (§3.1 above); the `ideal_predictions_*_d3.csv` "verified exactly" against CLOSE contain 5,067 reserved rows each | `purge_summary.json`, `prior_access_scan.json → root_files` |
| Committed year-loop experiments `run_oracle_ceiling.py`, `run_phase_b_cnn.py`, `run_phase_c_ensemble.py`, `run_phase_d_neat.py` (loop `2006..2019`) | `eurusd_hour_2005_2020.csv` (5,923 reserved rows, sha `72b8271d…`); the four `*_results.json` carry a `"year": 2019` fold; trades in the window: 148 / 134 / 67 / 21 | scan `year_loop_results_2019`, `root_files` |
| Walk-forward `run_wfo.py` / `app/walk_forward_optimizer.py` (calendar-year folds) | `wfo_results.json` and `wfo_results_2017_2019.json` each hold a **2019 fold with 6,188 test bars** (the whole year, cut included); the OOS trade files show none in the window, but the strategy ran over the bars | scan `wfo_results_2019_fold` |
| Historical root `trades.csv` versions (2026-04-06 … 04-21, incl. "re-optimize" commits `6af6a3e`, `3cb5cd1`) | parameters selected on data running to 2020-04; current `trades.csv` has 1 reserved row (2019-07-02) | independent sweep, `git show <c>:trades.csv` |
| Presentation archive `musashi/presentation-sources-20260927` (`f2d3922`) | `latest_native/origins.csv` with 3,738 reserved origins; figures rendered from it; predictor `docs/presentacion_revision/PNG_DELIVERY_VERIFICATION.json` records `data_sha256 4b57164f…` | independent sweep |
| `cnn_predictions_15yr.csv`, `ensemble_predictions_15yr.csv` | 1,504 reserved rows each, to 2020-04-29 21:00 | scan `root_files` |
| ANN/LSTM hourly fixtures | end 2019-05-23 (127 / 121 reserved rows): the predictor consumed its test split past the cut to produce them | scan `root_files` |
| predictor `examples/results/` | about 83 prediction CSVs end inside the window (phase_1b_binary, phase_1c_direction, phase_2_5, phase_3_*, phase_4_*); `docs/audits/evidence/characterization_v2_ledger_2026_09_12.json` profiles `base_d6.csv` (2018-12-14 … 2020-04-29) as MEASURED | independent sweep |
| `max_steps` | `app/config.py` sets 6300 but nothing in `app/` applies it: `process_data` loads whole files and keeps `base_df_full`; a default run holds the reserved rows in memory even when it scores none | code read |

Not found: any hash record of `eurusd_hour_2005_2020.csv`, `phase_1_base_d3.csv` or `phase_2_3_base_d3_last_year.csv`
before this lane (the scan now records them). No evidence in gym-fx or agent-multi; feature-eng and
prediction_provider reference the full files as code defaults only (PRIOR_READ_LIKELY, not verified as executed).

**Consequence.** The rows from 2019-05-16 to 2020-04-29 have been backtested, scored, optimized on and published.
Labelling them reserved on 2026-09-29 does not create non-use. They can be treated only as previously used
development data. A genuinely unread hold-out would have to be data after 2020-04-29 that no artifact in these
repositories has touched; none exists in the checkout. This is reported for the owner's decision; the cut and the
purge above remain correct as separation of *future* development scoring from those rows, which is all they can
achieve.

---

## 5. CPU-seconds reconciliation

Historic ceiling **3,600 CPU s** (the prior sweep's bound, not a fresh allowance). Retained 241-cell sweep:
**1,461.693 CPU s**. Prior lane's measured pytest spend on its branch: 2.98 + 6.86 = 9.84 CPU s (not merged).

This lane, measured with `/usr/bin/time` inside each `crispdm-run` (user + sys):

| Run | Host | CPU s | Peak RSS |
| --- | --- | ---: | ---: |
| pytest at `16bbe35` (3 failures, mine) | worker_b | 2.41 | 175 MiB |
| pytest at `d08ae00` (20 passed) | worker_b | 2.62 | 155 MiB |
| purge derivation, R1+R2+R3 + population | worker_b | 20.11 | 201 MiB |
| prior-access scan, two runs | coordinator | 0.56 | 18 MiB |
| **Total measured** | | **25.70** | |

Unmeasured, small: local syntax/import checks, `awk`/`csv` counts on `origins.csv`, gzip of the evidence
(each well under a second of CPU and under 100 MiB). `3600 − 1461.693 − 25.70 = 2112.607` CPU s remain under
the historic ceiling; the remainder is **not permission to spend**. **B0 is not proposed and not started.**
No run on the coordinator exceeded 1 GiB (`-m 1G` for the scan); the 2 GiB cap on worker_b was never lowered
and no run queued.

---

## 6. What is NOT done

- **No scoring, sweep, B0, broker call or horizon search** — as ordered.
- **Trade support of the synthetic arms is not derived**, only predeclared (feed truncation + per-cell censoring);
  the per-cell predictions do not exist outside a run.
- **The sequential backtest's entry set was not replayed**: the unit-trade reading is its superset, which is
  conservative for the purge but is not the sequential run's own duration distribution.
- **`e7966f3`'s accounting corrections and `7a50f62`'s executor are not merged**; whether to run the accounting
  change as a named experiment is an owner decision.
- **The reserved window is not a clean hold-out** (§4); this lane cannot create one. No replacement data was
  fetched.
- **The elapsed-hour generator is not wired into `process_data`** (doing so would change the replication baseline).
- The prior-access audit's predictor-side counts (about 83 prediction CSVs) come from an independent read-only
  sweep and are not reproduced by a committed tool; the heuristic-strategy side is tool-generated.

Signed: Satoshi III, lane S07, 2026-09-30.
