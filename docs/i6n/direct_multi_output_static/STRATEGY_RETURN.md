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
