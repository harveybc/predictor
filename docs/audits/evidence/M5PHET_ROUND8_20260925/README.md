# M5PHET round 8, 2026-09-25 — the rule the bank measured becomes the rule the bank applies, and the interpreter is measured for the first time

## Abstention as configuration, published in the catalog
`interpreter.min_confidence` and `interpreter.abstention_source` are now configuration (`m5phet.json`,
schema-validated), read by `decide.ask` when a caller passes none — so every chooser is gated, not only the call whose
author remembered. A threshold with no citation is still refused (`UNCITED_THRESHOLD`); a citation with no threshold is
refused by its own name. `/api/catalog` publishes the rule — `{min_confidence, source:{report sha256, stage, protocol,
seal, bins at or above}}`, `NOT_CONFIGURED`, or the refusal a bad declaration resolves to — so a web, MCP or Telegram
consumer can see the rule before trusting an answer. The report's local path is never published.

**The honest asymmetry, declared rather than faked:** `openai_compatible` can report a per-value confidence (read back
from the endpoint's logprobs, `exp(Σ logprob)` over the tokens spelling the value); `command` and `ollama` cannot, and
the catalog says `CONFIDENCE_NOT_REPORTED` for them. With a threshold in force, a question they cannot score is
**refused**, naming the unresolved parameter and its declared values — not silently passed. Nothing is configured on
the owner's installation, so every path behaves exactly as before; declaring a threshold there today would refuse
every sentence the words do not settle, which is the operator's decision.

## The interpreter measured for the first time (`tools/measure_interpreter.py`)
`command` / deepseek-v4-flash (OpenCode Go), 12 parameter-bearing sentences × 5 runs = 60:

| aggregate | value |
|---|---|
| resolved correctly | **50/60 = 0.8333** |
| correct when it did choose a declared value | 50/53 = **0.9434** |
| verdicts | CORRECT 50 · DECLINED 7 · WRONG_VALUE 3 · OUTSIDE_DECLARED 0 · FAILED 0 |
| stable across all 5 runs | 8 of 12 sentences; none never-correct |

Weakest: `¿Qué acción propone la política para estas barras?` 1/5 (declined 4); the three unsupervised sentences 3/5.
All four forecasting, both causal and the named-policy sentence: 5/5.

**The finding that matters more than the number:** on this corpus the interpreter is **never consulted for a single
scored field**. All 12 sentences resolve entirely through the deterministic word pass (`sources: QUESTION_TEXT`), and
the two classification sentences declare no slots at all. So the harness's standing `prose 14/14` has always been a
true statement about the product and **never** a statement about the language model. The model is consulted only for
the optional `bundle` field, which no sentence names and no expectation scores.

Published as `/api/catalog.interpreter.reliability` with its protocol, n and digest when an installation cites the
report (the owner's now does); `NOT_MEASURED` otherwise; refused if the report measured another plugin or model.

## Not measured
No closure table for this: there is no naive baseline for "reading a sentence" and no literature value matched to this
protocol, so `NOT_COMPARABLE` and no table was produced. `orchestrate.route` — where the model writes a whole envelope
instead of choosing a declared value — remains ungated and unmeasured. `openai_compatible`'s confidences are proven
against a fake loopback server only; no cloud consent exists.

M5PHET master 6e8509d, suite 578 passed / 1 skipped; acceptance 11/11 · 14/14 · 2/2 and 15/15.

## WP06 stage 5 — the search, and the hand window did NOT survive it

Objective = the closure table's own number (held-out MAE on the identical seal `33820b552ddf`, 9,824 origins,
protocol `d0ebd9a4bc75`); everything but the representation held at `baseline_hand`'s configuration. Optimiser:
predictor's own DEAP operator set (integer genes over declared bounds, `cxTwoPoint`, per-gene redraw,
`selTournament(3)`, elitism 1, resume ledger) — **not** DOIN, because DOIN's predictor domain re-verifies on
synthetic rows, which would have put a second scoring path beside the seal. No third optimiser was written.

| rank | stage | window | MAE (kW) | skill |
|---|---|---|---|---|
| 1 | searched_b373275495e2 | **21** | **0.512925** | 0.144164 |
| 2 | searched_0588f5747b62 | 21 | 0.512970 | 0.144089 |
| 3 | searched_b367cba9b237 | 34 | 0.521704 | 0.129516 |
| 4 | quantile_hand_95 | 60 | 0.526294 | 0.121858 |
| 5 | searched_77a56b176915 | 31 | 0.526542 | 0.121444 |
| 6 | **baseline_hand** | 60 | 0.537108 | 0.103813 |
| 6 | control_baseline_hand_refit | 60 | 0.537108 | 0.103813 |
| 14 | laya_chosen | 60 | 0.557190 | 0.070307 |
| 72 | searched_facdd5812c0f | 184 | 0.697646 | −0.164049 |

**Four searched representations beat the hand window, and every one is SHORTER than it** (21–34 against 60) — the
design job's four candidates had all been longer, which is why none of them won. The effect is the window, not the
column subset: rank 3 keeps all seven columns. Best spec beats `baseline_hand` by 0.024183 kW (4.50 %). One fit per
point, one seed, no interval on the difference — stated as the caveat it is.

**The control is what makes this readable:** `baseline_hand` refitted through the modified harness reproduces
0.5371084250158522 at 18 epochs exactly, so the new transform/differencing code left the level path untouched.

531 of 596 sampled points were refused and counted, never repaired (`LAG_EXCEEDS_WINDOW` 407, `NO_LAG_DECLARED` 84,
`AMBIGUOUS_DIFFERENCING` 37 raised by `validate_spec` itself, `NO_FEATURE_SELECTED` 3). A second round was declared
because the first exposed a bias in the space — window and lags were coupled, so no measured point had a window below
74 and the region where the winners live had never been tested.

`m5phet.decide` gained the third chooser: `search_choice` writes `chosen_by: SEARCH`, no probabilities, no backend, and
a `why` that must name the objective and the budget. Because a searched stage is now rank 1, three calibration groups
lost their label to `NO_BEST_RANKED_OPTION` — the winning representation is not one of the five options that question
declares, and **a search may not add an option to a question Laya was asked**. Calibration stays
`NO_NEW_MEASUREMENT` (best group 7 scorable of 30).

Harness limitation found and recorded: `lags` decides nothing the window does not — two specs differing only in lags
produced byte-identical configs and identical MAE.
