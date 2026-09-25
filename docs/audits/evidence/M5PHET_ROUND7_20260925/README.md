# M5PHET round 7, 2026-09-25 — the six open packages closed, and what the measurements said

## WP06 stages 3–4: no designed representation beat the hand window
One seal for every stage (`33820b552ddf`, protocol `d0ebd9a4bc75`, 9,824 origins); every candidate fitted with
everything but the representation held at `baseline_hand`'s configuration, so the difference is the representation.

| rank | stage | MAE (kW, h+60) | naive | skill |
|---|---|---|---|---|
| 1 | quantile_hand_95 | 0.526294 | 0.599327 | 0.121858 |
| 2 | baseline_hand | 0.537108 | 0.599327 | 0.103813 |
| 3 | quantile_hand | 0.538670 | 0.599327 | 0.101208 |
| 4 | candidate_seasonal_lag_74 | 0.545436 | 0.599327 | 0.089919 |
| 5 | laya_chosen | 0.557190 | 0.599327 | 0.070307 |
| 6 | candidate_short_memory | 0.578034 | 0.599327 | 0.035528 |
| — | candidate_seasonal_lag_1443 | refused `FEATURE_NOT_IN_DATA` + window > sealing window | | |
| — | candidate_seasonal_lag_2892 | refused: window 2892 > sealing window 197 | | |

The two refused candidates were NOT scored on other rows: a probe showed sealing at their windows gives 8,578 and
7,129 origins — different populations, hence `NOT_COMPARABLE`. `compare_stages` gained `--not-measured` so an
attempted-and-refused stage is a row with its reason instead of being invisible.

## WP07: the range answers on the owner's instance
A five-quantile bundle (`quantile_hand_95`) answers `interval` at 0.95; measured coverage **0.925998 against a nominal
0.95**, mean width 2.890 kW — below nominal, recorded as such. Verified on the owner's own 8765 through question mode:
point 0.5412255525588989 kW and interval [0.031671, 2.822953] kW at 0.95 (quantiles 0.025/0.975), both with units.
The single-sentence path still needs the horizon named; the two-question envelope is the path that answers both.

Bundle ambiguity is now resolved by what the request declares (fitted state → the question's `bundle` field → the kind
of answer), and only a request declaring none of those is refused, naming the candidates and the three ways out.

## WP23: the calibration report stops shrugging
56 human records (`chosen_by: HUMAN`, no probabilities, a required `why`), 50 outcomes linked. `NO_BEST_RANKED_OPTION`
is gone for the two questions the rank-1 stage can answer; for the other two the report names the reason
(`COMPARABLE_BUT_NO_DECISION_RECORD`). Two of Laya's question sets can never be answered by the hand stages and the
report says why: there is no `k=1` cut among the declared cuts, and `tcn` is an inline family of the fusing core that
feature-extractor does not declare — `human_choice` refused those keys 14 times rather than invent them.

## Two changes that deserve the owner's eye
1. **The server now re-expresses one window into another engine's scale**, and only under strict conditions: another
   configured bundle of the same series identified by its `scaler_digest` (never guessed), same columns, window, step,
   unit and scale; the answer carries `input_restandardized_from`. It is arithmetic, disclosed, and refused in every
   other case — but it is the one place where the server transforms the caller's data.
2. **An acceptance harness was changed**: `verify_envelopes.py` hard-coded `REFUSED:NOT_ESTIMABLE` for the interval
   question, which made installing a quantile bundle look like a product failure. It now derives the expectation from
   the catalog the running instance publishes, so it asserts the rule rather than one configuration.

Acceptance after the round on 8766: examples 11/11, prose 14/14, refusals 2/2, envelope questions 15/15,
`execution_authorized` false. M5PHET suite 545/1.

## WP21(b) — Laya as a first-layer decision evaluator: it abstained on all 256 bars

`first_layer.py` asks Laya one `choice` per bar over `[long, flat, short]`, with the state built from the window's
**summary** (last close, log returns over declared lookbacks, realized volatilities, the position held, the fitted
policy's own column names — ~20 numbers against the window's 2,656, asserted by size and content). Two operational
facts: the provider budgets head+options+state against one 512-token limit and refuses `TOKEN_BUDGET_EXCEEDED` rather
than truncate, so the state shows 12 of the 83 declared column names and says so; and each call takes ~20 s, so 256
bars took ~85 minutes.

**Result: 256 asked, 0 above the measured 0.8 threshold, 256 abstentions.** Top probability 0.575–0.705, mean 0.634 —
entirely inside the band where WP09 measured this checkpoint at chance (0.327 correct below 0.8 against a 0.333 chance
rate; 0.873 at or above). And the gate is not what flattened it: recovered from the abstention records, the
checkpoint's **own argmax was `flat` on all 256 bars**, so ungated the series would have been identical.

| stage | environment training reward (total) | naive `flat`, same rows | comparability | rank |
|---|---|---|---|---|
| laya_first_layer | 0.000000 | 0.000000 | COMPARABLE | 1 |
| flat | 0.000000 | itself | COMPARABLE | 1 |
| fitted_sac | −0.003888 | 0.000000 | COMPARABLE | 3 |

`compare_stages` refuses to rank them at all: `policy_profitability` is refused by name — "a proposed action is not a
realised return: no order was placed, no fill, slippage, financing or timing exists, and the market did not respond to
it" — and every row is `NO_NEW_MEASUREMENT`, flagged `UNDERPOWERED 256/500` against a minimum declared before the run.
`decide.outcome` refused all 256 links with `ABSTENTION_HAS_NO_OUTCOME`: an abstention is not a choice, and a row
cannot rank a choice that was not made.

A training-reward total on 256 development bars of one instrument is not evidence about any market. The 0.8 threshold
was measured on a classification corpus, not on a trading question; carrying it here is an assumption, named as one in
every record.
