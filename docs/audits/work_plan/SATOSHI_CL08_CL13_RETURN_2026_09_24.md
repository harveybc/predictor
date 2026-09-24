# CL08-CL13 return: the four counterexamples are closed and the question is now the user's

Review: `MUSASHI_CL_FIRST_REVIEW_2026_09_24.md`. Heads returned: news-signal
`e40098b`, M5PHET `3f4a833`, feature-eng `1a0402e`, predictor this commit.

All four reproduced counterexamples were reproduced HERE first, frozen as tests
that failed, and then repaired. The fifth finding -- a product gap, not a defect --
is closed: a user can now ask their own question.

## What a person can do today that they could not yesterday

```bash
news-signal ask --input news.json \
  --question "Does this news make a near-term euro-area rate cut more likely?" \
  --option "more_likely=the facts reported point towards easier policy" \
  --option "less_likely=the facts reported point towards tighter policy" \
  --option "no_bearing=the news does not bear on the euro-area policy path" \
  --store ./shadow
```

Their words go to the pinned Laya SDK through the installed registry, and the
answer comes back typed, persisted and replayable. `--print-question` shows exactly
what the model will be given before anything runs.

## CL08-CL12, one row each

| ID | Evidence | Status | Next |
|---|---|---|---|
| CL08 | `evidence/CL08_CL12_20260924/` (PRE, POST, install) | **Closed.** Opaque keys, contained writes, split identity, validated reuse, correct pin, clean-venv install verified | keep the frozen cases as regression |
| CL09 | `evidence/CL09_CL10_20260924/question_parity.json` | **Closed.** Two user questions, 2 identities, **12/12 exact parity each** against the independent witness | more answer schemas than `choice` |
| CL10 | `evidence/CL09_CL10_20260924/wrapper/questions_pilot.json` | **Diagnosed, not tuned.** 7 errors tabulated with probabilities and token accounting; **truncation ruled out** | a held-out, source-identified relevance set before any prompt selection |
| CL11 | `feature-eng@1a0402e` | **Partly advanced.** CAL01-CAL12 implemented on fixtures, 25 tests | collector, broker adapters, other domain providers: NOT started |
| CL12 | `evidence/CL08_CL12_20260924/results_post.json` | **Closed.** Population derived from the accepted run record; authorities labelled | custody over child records is still absent and is named as such |
| CL13 | this document | delivered | — |

## The four counterexamples, before and after

| Case | Before | After |
|---|---|---|
| `../escaped` as `event_id` | SHADOW_ONLY, file written **outside** the store | written only under `store/<shard>/<key>.json`; the identifier is retained inside the record |
| refusal then success | DUPLICATE, only the refusal persisted | REEVALUATION, both retained, the successful one retrievable |
| another task | DUPLICATE, the old task's record returned | REEVALUATION, both tasks retained, each receipt points at its own result |
| tampered record | DUPLICATE, still counted actionable | REPLACED_INVALID, quarantined, reported in every replay, never counted, never served |
| rewritten population | COMPLETE with summaries | refused: the accepted run record derives 2537/1802 |

The probe's `store._path(event_id, input_sha256)` is gone on purpose: a path
derived from an external identifier was the defect. The POST probe uses `find`,
`records_for` and `find_record` instead.

## Measured, on the external 5090 under fresh admission

| Quantity | Value |
|---|---|
| CL09 parity | `rate_cut_risk` 12/12, `who_is_affected` 12/12, every decision field, no tolerance |
| Distinct task identities | 2, from 2 questions over the same 13 items |
| Warm inference | 10.9 ms median (min 10.6, max 0.69 s cold) |
| Peak VRAM / RSS | 2.47 GB of 33.7 / 3.39 GB |
| Device | `GPU-a9f35631-…`, MEASURED in process |
| Cost | one admission, `-m 6G -t 900`; the run finished in ~4 minutes |

## CL10: what the diagnosis actually shows

Token accounting for the preset question: head 47 tokens kept 47, options 98,
longest state 70 against 363 of room, **all fit**. So the seven errors are not a
truncation artefact, which was the most likely mechanical explanation and is now
excluded.

| Case | Gold | Predicted | Probabilities |
|---|---|---|---|
| us-cpi-003 | related | unclear | related 0.0998, unclear 0.7638 |
| ea-unemployment-004 | related | unclear | related 0.3522, unclear 0.4162 |
| sports-006 | unrelated | unclear | unclear 0.7477 |
| tech-007 | unrelated | unclear | unclear 0.6584 |
| passing-mention-008 | unrelated | related | related 0.7898 |
| fragment-009 | unclear | related | related 0.6834 |
| negated-010 | unrelated | related | related 0.7296 |

Two shapes: `unclear` used where a reader would commit, and `related` given to a
passing currency mention, a contentless fragment and a negation. The `confidence`
field runs 0.025-0.43 and the checkpoint warns at load that its temperatures are
invalid, so it is not a probability of being right.

A second observation, kept apart from both: asked *whose* economy a story is about,
the model answers sensibly (US CPI → united_states, ECB → euro_area). Asked whether
the same story makes a rate cut more likely, it answers `no_bearing` for eleven of
thirteen. Factual attribution and inferential judgement are different capabilities
and parity establishes neither.

**This remains a smoke set.** 13 author-written rows, 12 distinct inputs. No prompt
or variant has been selected on it and none will be until a held-out,
source-identified set with annotation rules exists.

## CAL01-CAL12

Implemented in `feature-eng` on deterministic fixtures: arrivals in, as-of views
out, nothing overwritten, and refusals instead of invention for ambiguous clocks,
mixed units, missing consensus, zero scale, unknown availability and tied arrivals
with no observed order. There is deliberately no path from a market move to a
surprise -- the exact mistake the review found in the existing NFP prototype.

## Not started, and not claimed

Prospective news capture (needs the documented feed entitlement), MT5 demo and
Alpaca paper adapters, the forecasting, hierarchy, causal and policy providers, and
the governed sample for the calendar. None of these is advanced by a contract test.

## Costs and the dispatchable queue

CPU this block: the three suites plus two clean-environment installs, all under
`crispdm-run`, none above 6G. GPU: one admission on the external 5090, ~4 minutes,
nothing preempted; the device is idle at 39 °C. Doctoral numbers untouched: R0
0.371174, R1 0.374584, R2 0.368596.

Next dispatchable without new authorization: the held-out relevance set (CPU,
annotation first), the remaining domain adapters (CPU), and the calendar's governed
sample once a source is named.
