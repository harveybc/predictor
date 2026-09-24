# CL20-CL24 return: five families answering, and the instrument to judge them

Orders: `MUSASHI_CL14_CL19_REVIEW_2026_09_24.md` (CL20-CL24) and M5PHET
`docs/CL14_CL19_REVIEW_CONTINUATION.md`. Evidence:
`docs/audits/evidence/CL20_CL24_20260924/`.

The reviewer's model became unavailable mid-round. His agents' unpushed work was
preserved before anything else was attempted, and the round was then run the way
he ran it: one agent per area in its own repository, one integrator.

## What a person can do now

`http://127.0.0.1:8765` — type a question, attach a dataset, see a typed result.
Five families, each with its own installed provider, selectable per conversation
along with its input reader, output contract, fitted state and parameters.

| Family | Provider | Engine runs in | Verified answer |
|---|---|---|---|
| Classification | `laya_news` | private worker, real Laya on the external 5090 | `euro_area` 0.9666 |
| Forecasting | `predictor_forecast` | its own TensorFlow interpreter | 0.5412255525588989 kW, 60 min |
| Hierarchical regimes | `feature-eng-hierarchical-regimes` | the interface environment | cluster paths with novelty |
| Causal | `causal_inference` | inference only over a fitted study | ATE with interval and assumptions |
| Policy | `trading_policy` | its own Stable-Baselines3 interpreter | SAC action −0.0199989 |

Ordinary words reach the engines. Against the real TensorFlow bundle: the
canonical phrasing, an English paraphrase, a Spanish one and a vague question all
return the bundle's recorded value; an unsupported horizon and an untrained target
are refused by name.

## CL20-CL23, each reproduced then repaired

| ID | Owner | What changed | Tests |
|---|---|---|---|
| CL20 | news-signal | a retry no longer re-signs corrupt content; a failed exclusive write is never reported as accepted; a record under the wrong key is detected; two sources sharing an identifier are two events | 117 → 125 |
| CL21 | feature-eng | the release boundary orders by PUBLICATION; an arrival with no publication clock is excluded and named; the original release is the earliest published, whatever order things arrived | 37 → 48 |
| CL22 | prediction_provider | paraphrase, ambiguity and refusal cases against the real bundle, with the native value reproduced exactly | 52 → 69 |
| CL23 | news-signal | recorded source → queue → installed provider → persisted result → acknowledgement bound to that exact evaluation, with crash and retry | 125 → 140 |

Each was frozen as a failing test before its repair. I re-ran every suite myself
rather than accepting a report, and confirmed the CL21 fixture independently:
release surprise **0.2** against the latest published consensus, available **0.4**.
For CL23 I disabled the recovery lookup in the source: exactly one test went red
and green again on restore, so the test bites.

## What integration found that no unit test could

**An interpreter asked to choose will choose.** Asked to forecast `Voltage` when
the bundle holds only `Global_active_power`, the deterministic pass left the slot
open, the question went to the language model, and the model — asked to pick among
the allowed values — picked the only one. A confident forecast of a different
series came back, and because a language model is not deterministic it did so only
sometimes. A provider that can enumerate what it does NOT serve now declares it,
and naming one is refused before any interpreter is consulted.

**A provider shipped an example its own slots refused.** The first thing a person
clicks would have failed. Every family's own example is now checked on every run.

**An oversize upload explained itself badly.** The limit was right; the message was
the parser's. It now names the limit and the size sent.

## The instrument, built without taking the measurement

No family has a measured quality. `evaluation/` in M5PHET is the machinery so that
the moment real labels exist a score can be computed without inventing a protocol
under time pressure: a frozen protocol with its digest, per-family metrics, a seal
taken before scoring, and refusals where there is nothing to measure —
`causal_accuracy` (a counterfactual is never observed), `policy_profitability`
(a proposed action is not a realised return) and `regime_accuracy` (cluster
identities are arbitrary). A corpus with no independent label source can only
carry `AUTHOR_WRITTEN_SMOKE`, and every report built on it says so. 38 tests.

**No score was computed for any model.** This is the instrument, not the reading.

## Costs and state

CPU only. No GPU was used for any repair; the classification answers come from the
already-running worker on the external 5090 under its existing lease. Browser
acceptance passed at three viewports with zero page errors. The adversarial pass
over the public surface ran nineteen hostile inputs and found one message worth
fixing. Verification now runs on a second instance with its own state, so the
owner's workbench stays clean. Doctoral numbers untouched: R0 0.371174,
R1 0.374584, R2 0.368596.

| Repository | Branch | Head |
|---|---|---|
| M5PHET | `musashi/chat-workbench-20260924` | `ab65ceb` |
| news-signal | `satoshi/classification-first-20260924` | `567c610` |
| feature-eng | `codex/m5phet-hierarchical-regimes-20260924` | `eaa3d03` |
| prediction_provider | `musashi/m5phet-forecast-20260924` | `9e9914e` |
| causal-inference | `feat/m5phet-causal-provider` | `38e9b86` |
| agent-multi | `satoshi/m5phet-policy-provider-20260924` | `ec385c96` |

## Not started, with what each waits on

Prospective news capture (a documented feed entitlement), MT5 demo and Alpaca
paper adapters (documented account access and the existing risk mandate), and a
labelled corpus for any family (independent annotators and a frozen split). None
is blocked by code.
