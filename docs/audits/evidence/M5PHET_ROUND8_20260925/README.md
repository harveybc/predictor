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
