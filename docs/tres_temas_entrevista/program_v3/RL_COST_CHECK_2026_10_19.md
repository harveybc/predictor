# Selected SAC Cost Check

Owner question: unusually high TRAIN profit could indicate omitted fees.
Read-only accounting/config/source check; no training or campaign edits.
Scope: early-stop contract `ac792153`, checkpoint90000, gym-fx `df1b602`.

| Quantity | TRAIN2020-2023 | VALIDATION2024 |
| --- | --- | --- |
| Initial equity USD | 10000 | 10000 |
| Final equity USD | 36649.96314722616 | 8921.557583096403 |
| Net profit USD | 26649.96314722616 | -1078.4424169035974 |
| Sum trace pnl USD | 26649.96314722616 | -1078.4424169035974 |
| Reported trade costs USD | 7049.861051490074 | 774.7633014657854 |
| Sum trace trade_cost USD | 7049.861051489143 | 774.7633014656393 |
| Final cumulative commission_paid USD | 7049.861051489133 | 774.7633014656399 |

The equity difference equals reported net profit and cumulative trace PnL;
trade-cost totals reconcile within floating-point rounding. commission_paid
is CUMULATIVE, so summing that column over rows double-counts it repeatedly.
Under native broker accounting the cost accumulator can include financing
assigned to a closing execution; do not label its full total pure commission.

Effective config: commission5e-5 of notional per execution, slippage_perc5e-5,
leverage50, financing enabled with5% annual notional daily-interest scenario
on long and short positions. This is not a broker-certified FX swap schedule.
The legacy slippage alias is0, but the broker explicitly prefers the nonzero
slippage_perc key; it does NOT silently disable that slippage.

Instantiating the pinned broker on CPU confirms commission fraction0.00005,
interest0.05/interest_long=True, leverage50 and slip_perc0.00005. For100000
base units at price1.1 the commission is USD5.50 per execution, not zero.
This arithmetic example is not an assertion of a particular broker's tariff.
Sixteen existing financing/ordinary-and-direct-accounting tests passed on CPU.

## Limits And Interpretation

ASSUMED_SPREAD is constant0.0001 in these inputs. Its session_spread_column
path controls reopen/session admission, NOT a separate spread debit. The run
has execution_cost_profile=null; no independently verified bid/ask spread
schedule was supplied. Do not present the configured adverse slippage as an
extra verified spread charge, or count the same friction twice.

The slippage_paid trace column is empty, not evidence of zero price impact:
slippage affects fills/PnL, but is not separately itemized by this trace. The
summary and existing tests do not constitute a full independent broker-fill
audit, including every limit/gap/direct-settlement path. No source-specific
commission/swap/spread realism certificate is claimed by this check.

TRAIN +266.50% is cumulative over four years, approximately38.36% annualized
geometrically; it is not a one-year return. Position sizing permits up to
roughly5x initial notional/equity before action scaling/headroom under
rel_volume0.1 and leverage50. Reinvestment and repeated exposure to TRAIN can
contribute to its scale; that explanation is not proof of absence of other
simulation or learning defects. VALIDATION remains negative.

Current regularizer ablations keep this cost scenario fixed for comparability;
this check does not alter their sealed identities or automatic dispatch.
Broker-calibrated costs and independent fill-level reconciliation remain
separate validation work, not silently fulfilled by nonzero config settings.
