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
