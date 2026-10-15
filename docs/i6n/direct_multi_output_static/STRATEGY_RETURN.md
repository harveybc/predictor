# I6-DMO strategy lane return

Status: adapter PREPARED, not run. No strategy fit, no TEST row, no trading, no
service or GPU use. All prices and calendars in the tests are synthetic.

## Commits

| Repository | Branch | Commit | Content |
|---|---|---|---|
| heuristic-strategy | satoshi/i6dmo-strategy-20261014 (base origin/master 5c87a25) | d40fd3b054e4f952054a86ea203e8f74e6b295e1 | `app/policies/business_pair_policy.py` (blob 79b6df7a, sha256 6b6c5d96...) and its tests |
| predictor | satoshi/i6dmo-strategy-20261014 (base 2a440265) | see `git log` (adapter 9630b424; docs on top) | `tools/i6dmo/strategy_adapter.py`, `tests/i6dmo/test_strategy_adapter.py`, interface, plugin map |

Legacy blobs are unchanged: plugin 3a19b3f, policy 0940589. They are not used by I6-DMO.

## Plugin actually imported

`STRATEGY_PLUGIN_MAP.json` (generated once by `python -m tools.i6dmo.strategy_adapter map-plugin --json`):
in the default interpreter `heuristic_strategy.plugins:ls_pred_strategy` resolves to an
installed 2025 copy (sha256 3e1ebec1...), not the audited checkout. The I6-DMO executor never
uses that route. It loads one blob from the pinned commit through the git object store, checks
blob id and sha256, and runs it as an isolated module.

## DMO -> test -> PRE/POST

PRE was recorded before any implementation existed. heuristic-strategy: collection error
`FileNotFoundError ... business_pair_policy.py`. predictor: `ModuleNotFoundError: No module named 'tools.i6dmo'`, 1 error.

| ID | Test(s) | PRE | POST |
|---|---|---|---|
| DMO14 | hs `test_reversed_hit_order_changes_the_decision[long/short]`, `test_order_is_numeric_horizon_order_across_both_families`, `test_missing_head_is_skipped_never_filled`, `test_entry_uses_long_family_only_and_refuses_short_heads`, `test_entry_minimum_valid_long_heads_abstains_and_never_fills`, `test_entry_geometry_is_bitwise_equal_to_the_legacy_policy` (500 cases) and 9 others | ERROR (module absent) | 32 passed (with the existing policy and replay tests), trading-stack env |
| DMO14 | predictor `test_dmo14_*` (14): pin and plugin map, masks near B, missing/NaN/validation-ineligible reasons, positional input refused, arrays mapped by horizons_hours, reversed order for buy and sell plus a min/max negative control, unavailable crossing head not consumed, long-only entry with minimum 2, backtest-level reversed order | ERROR (module absent) | passed |
| DMO15 | predictor `test_dmo15_*` (9): force-flat fill before B, missing fill bar retained as failure, end of data invents no fill, one position per asset, entry cancelled after cutoff, SL first when both barriers touch in one bar, 10000 equity and costs recomputed by hand, zero-trade metrics undefined with reasons, TEST refused | ERROR | passed |
| gate | `test_pair_gate_requires_strict_improvement_and_enough_heads` | ERROR | passed |

predictor POST: `25 passed in 0.74s`. The first POST run had 23 passed and 2 failed: two tests
wrongly assumed one trade per week, but re-entry after a close is correct. I fixed the tests, not
the code. Mutation check (12+1 mutants, file restored after each): 12 killed. One survived
(`pending` guard) because it is equivalent: a pending order is always consumed at the next bar
open before the decision step.

## Commands

```bash
# heuristic-strategy worktree
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-strategy -- <trading-stack python> -m pytest -q tests/unit_tests/test_business_pair_policy.py tests/unit_tests/test_prediction_entry_exit_policy.py tests/unit_tests/test_prediction_entry_exit_backtrader_replay.py
# predictor worktree
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-strategy -- python -m pytest -q tests/i6dmo/test_strategy_adapter.py
python -m tools.i6dmo.strategy_adapter map-plugin --json
```

## Owner rulings needed (declared defaults are implemented)

1. Barrier equality: a forecast exactly at a barrier touches it (`INCLUSIVE_TOUCH`).
2. Tie: one value cannot touch both barriers because they are validated to lie on opposite sides. A duplicate physical hour is refused.
3. No sampled touch means HOLD. No available head also means HOLD. Real SL/TP and force-flat stay active in both cases.
4. Entry needs at least 2 distinct available long heads. This is the market-week contract's proposal "para auditar" and has not been ratified.
5. The pair gate needs at least 1 eligible short head and 2 eligible long heads. A head that is ineligible in validation is masked in the strategy.
6. Fill model: market orders fill at the next bar open. The force-close trigger is t >= B-1h, and the fill bar must close by B. An entry fill needs bar close <= min(K, B-1h). When both barriers are touched inside one bar, SL is taken first.
7. Costs are the legacy effective values: spread 2 and slippage 1 times pip_cost 0.00001, which is about 0.2/0.1 real pips. Commission is 7 per lot round trip. Swap is a 10 per lot per day cost, charged pro rata by hours. These magnitudes need ratification.
8. Sharpe and Sortino use a UTC-day mark-to-market return clock with 260 periods per year. The Sortino target is 0.
9. Passive reference: one long per week at the first entry origin, at the maximum admissible size, closed at force-flat.

## Risks and limits

- The bar-label convention (close_time) and the hourly OHLC source belong to the data lane and are not delivered yet. The calendar used is the C1 synthetic or assumed one, not a verified broker calendar.
- The ops pair `run-unit` entry is not implemented here. Integration must wire `pair_eligibility` and `run_backtest` once both families have terminals.
- Every host that runs a pair unit needs the pinned heuristic-strategy commit. Otherwise the load fails with the typed error `STRATEGY_PIN_UNRESOLVED`.
- Sampled endpoints do not reveal continuous first passage.

## Addendum: pair-unit hook for the ops runner

The ops branch at ae70a6a9 is merged so the real `unit_runner`, `campaign.gate_problem` and
`worker._validate_child` are under test. `run_unit(spec, out_dir, progress_path)` is the
`pair` hook. While the STRATEGY_OWNER_RULINGS gate is not PASS, it returns a typed FAILED
terminal (`STRATEGY_OWNER_RULINGS_OPEN:<problem>`, `pair_status BLOCKED_OWNER_RULINGS`) and
computes nothing. When the gate passes, it runs with the owner-ruled values, never with the
declared defaults. It rejects a ruling the code does not implement. `final_test` is refused
(`FINAL_TEST_HOOK_NOT_PREPARED`).

PRE: `tests/i6dmo/test_strategy_pair_unit.py` gave 13 failed (no `run_unit`). POST: 38 passed.
After a one-line typed-code rendering fix, the full `tests/i6dmo` run gave 76 passed (ops,
adapter and pair).

Open contract points for integration: the bars and calendar pins
(`data_pins.strategy_bars`/`strategy_calendar`) and the report-terminal artifact
`validation_predictions` must be produced by the data and reporting owners. The worker should
check those pins before it claims the unit, because the hook's FAILED terminal is final.

## Addendum 2: AUD-DMO-06 correction (sealed execution policy, no owner gate)

Merged audit branch codex/i6dmo-strict-audit-20261014 @98d83fcf (ordinary merge; my prior
commit was already an ancestor of the audited integration). The WAIT_OWNER_RULING dependency
in the pair hook is removed: `run_unit` now derives a SEALED execution-policy receipt
(`i6dmo.execution_policy.v1`, sha-sealed canonical body) automatically from the versioned
instrument configuration in the sealed effective config plus the business contract. No
approval prompt exists on this path. Fixed values: minimum longs 2 (the pinned policy's own
default), inclusive barrier equality, same-bar ambiguity SL-first, next-bar execution recorded
as a SIMULATION_ASSUMPTION statement, initial equity 10000, return clock UTC day with 260
periods per year recorded in the receipt. Costs come from the configured price-unit
spread/slippage/commission/swap and are labelled NOT_BROKER_VERIFIED (BROKER_VERIFIED needs
non-empty evidence). Tampered sealed receipts are refused (EXECUTION_POLICY_TAMPERED), a
sealed receipt differing from the configuration is EXECUTION_POLICY_MISMATCH, and a missing
instrument configuration is the typed config failure INSTRUMENT_CONFIG_MISSING, never a wait.

Point/pip/lot audit (tests at prices 1.10 and 0.95 against the ACTUAL pinned policy blob):
the legacy `pip_cost` 0.00001 is a POINT (0.1 real pip); geometry distances and the
profit_threshold are in that unit and independent of price level; sizing is base-currency
units = cash*rel_volume*leverage with no price conversion, so notional varies with price
(22000 at 1.10 vs 19000 at 0.95 for the same 20000 units); commission and swap scale with
units/lots, never price. All recorded in the receipt's pip_semantics and sizing assumptions.

PRE: tests/i6dmo/test_strategy_execution_policy.py + rewritten test_strategy_pair_unit.py gave
19 failed, 4 errors (no receipt API). POST: 54 strategy-lane tests pass; full tests/i6dmo run
recorded separately. Hook export names unchanged (`run_unit`; final_test still refuses
FINAL_TEST_HOOK_NOT_PREPARED for lane 2 to replace). The ops worker guard and
STRATEGY_OWNER_RULINGS gate entry in worker.py/campaign.py are ops-lane files; my hook no
longer reads them.

## Addendum 3: run_final_backtest export for lane 2

Merged satoshi/i6dmo-model-20261014 @e8968d67. New stable export
`run_final_backtest(spec, predictions, execution_params, out_dir)`: same semantics as
run_backtest on the frozen final models' TEST-window predictions, consuming ONLY the sealed
execution-policy parameters passed in (no re-derivation), with identity-bound receipts
(weights/design/readiness/predictions digests of both families, policy pin, costs seal,
NOT_BROKER_VERIFIED label) and typed refusals. No fit. PRE: 8 failed (export absent).
POST: 8 passed; strategy+final subset 79 passed; full tests/i6dmo green run recorded.
One test fix during POST: an early close may be followed by a legitimate re-entry in the
same week, so the counterexample asserts the first trade, not a single trade.

## Addendum 4: versioned EURUSD instrument/cost configuration (order 85ea134e section 3)

Recovered from retained evidence only (pinned plugin blob 3a19b3f with its 'worst-case retail'
comments and commit 310e285; lts backtrader_simulation_broker blob ddeef86c @12bce5f with
pip_value 0.0001; lts config_simulation.json / REFERENCE_plugins.md / heuristic_strategy copy;
MT5 bridge lab snapshot schema; prior strategy audits). File
tools/i6dmo/instrument_eurusd.v1.json, sealed config_sha256 f2443141..., every field with
sources and status; all four costs are ASSUMED_NOT_BROKER_VERIFIED (no broker statement or fee
schedule with numeric EURUSD costs is retained anywhere). Recorded conflict: the plugin's POINT
arithmetic gives spread/slippage 0.00002/0.00001 while the lts broker's PIP arithmetic gives
0.0002/0.0001 for the same nominal 2/1 pips; chosen 0.0002/0.0001 as the conservative larger
cost matching the retained comment's intent. pip_cost_parameter stays 0.00001 (the pinned
policy's POINT threshold identity). The execution-policy receipt now derives from this file
(receipt carries instrument_config_sha256) and the pair unit's typed INSTRUMENT_CONFIG_MISSING
wait clears exactly when the sealed file exists; unsourced fields, unknown statuses, missing
conflict reasons and broker-verification overclaims are typed refusals. PRE: 12 failed/erred
(file and API absent). POST: full tests/i6dmo 265 passed. MT5 lab live-spread rows remain
untranscribed retained evidence (noted in the file).

## Addendum 5: retained-MT5-rows claim refuted (reviewer correction)

The reviewer (retsu/i6dmo-market-time-review-20261015 @3508e149,
market_time/provenance/MT5_COST_EVIDENCE.json) refuted my note that the MT5 bridge lab retains
transcribable live spread snapshots: no mt5-bridge.sqlite exists on any searched root, and the
only surviving MT5 artifacts are five distinct 2026-08-16 snapshots (six retained files, two
byte-identical, 4964 bytes each) from an ETHUSD demo session with zero symbol_snapshots rows,
no spread field and no EURUSD. Corroborated by lts mt5_unknown_outcome_20260926/NOTE.md.

Corrections in tools/i6dmo/instrument_eurusd.v1.json: the false note is replaced by the
finding (snapshot digests, content, and that EURUSD cost evidence does not exist locally); a
typed `refuted_evidence` entry records the locator as REFUTED_NOT_FOUND with the reviewer,
finding artifact, search roots and the empty promotable_cost_fields list; the two EVIDENCED
fields that cited the bridge lab now cite only its SCHEMA (which does exist) and assert nothing
about rows, so the five non-cost fields stay EVIDENCED and all four costs stay
ASSUMED_NOT_BROKER_VERIFIED. `_refuse_refuted_citations` makes any future field citing a
refuted locator a typed INSTRUMENT_CITES_REFUTED_EVIDENCE refusal. The seal moved
f2443141 -> b5d37996 and the receipt re-derives (binding test green).

PRE: 4 failed. POST: full tests/i6dmo 270 passed (CPU only).
