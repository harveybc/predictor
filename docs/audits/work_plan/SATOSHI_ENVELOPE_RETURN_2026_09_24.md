# The envelope: one shape for every area, in the interface and over MCP

Owner's specification: five documents (causal, forecasting, RL, unsupervised, NLP
orchestration) giving the target request/response shape with invented values.
Evidence: `docs/audits/evidence/ENVELOPE_20260924/`.

## What was built

One envelope for every area: an `area`, a `state`, and named typed `questions`
in; named typed `answers` out, each answered on its own or refused by name with
its reason. Providers declare `question_types()` and implement
`answer_questions()`; the runtime validates before the provider is reached.

Around it, what the owner's orchestration document asks for: the interpreter
(DeepSeek via OpenCode Go, through Hermes) reads the sentence and the SHAPE of the
attached data -- columns, types, count, never a row -- and proposes an envelope;
the contract validates it against the catalog, the providers' declared values and
fitted combinations, and the data's columns; the person reviews and may edit it;
the engines run; the interpreter narrates the answers and the answers check the
narration -- every number in the sentence must be one the engines produced, or the
sentence is discarded for a deterministic rendering. An MCP server exposes the
catalog, envelope execution and proposal as tools over stdio, going through the
same registry and the same refusals.

## The rule that overrode the owner's shape

His values were invented and he said so. Where an engine cannot compute a question
type, the answer is a typed refusal with its reason -- never a number.

| Area | Answered from the real engine | Declared and refused by name |
|---|---|---|
| forecasting | `point_forecast` 0.5412255525588989 kW; direction_long 0.6035 | `interval`, `anomaly_risk`: no predictive distribution in these bundles |
| causal | `ate` 2.0294, CI [1.9313, 2.1275], assumptions carried; no fabricated p-value | `cate`: the study was fitted with no effect modifier and is not refit in chat |
| rl | `next_action` 0.0591 with the actor's real distribution (mean, log-std); `value_estimation` from the twin Q critics, 3.7996 / 4.0067 | `confidence`: absent, because a SAC actor emits no such thing |
| unsupervised | `clustering` with real silhouette in the fitted space; `cluster_description` centroids in original units | a k the reference was not fitted with |
| classification | two `choice` questions in one SDK call, euro_area 0.9666, tone neutral 0.7511 | a duplicate option, a wrong language, a truncating option |

Verified through the running product: **11 of 11 questions as expected** across
the five envelopes, nothing authorized. The classification envelope reaches the
sealed checkpoint on the external 5090 through the worker on the same contract.

## What driving it in a real browser found that the harness could not

The harness writes envelopes by hand. In the browser the interpreter writes them,
and it was being validated against the wrong vocabulary.

First it chose `values` -- a key of the attached JSON -- as the forecast target,
because the only vocabulary it saw was the data's columns; the engine refused
honestly and the person got two refusals for an answerable question. The catalog
now carries each provider's declared parameter values, the router shows them, and
a data column proposed as a target is refused before anything runs.

Then, with two bundles, it paired `Global_active_power` with horizon 1 -- each
value admissible, the pair nobody's. Providers now declare fitted combinations
and the phrasings each value stands for ("una hora" is 60), and a question whose
fields match no combination is refused as not fitted.

After both: the Spanish sentence "pronostica la potencia a una hora y dame también
un rango de incertidumbre" becomes the right envelope, returns 0.5412255525588989
kW, refuses the interval with its reason, and is narrated faithfully in Spanish
with the interpreter named. Screenshots retained.

## State

| Repository | Branch | Head |
|---|---|---|
| M5PHET | `musashi/chat-workbench-20260924` | `2f1bb77` |
| news-signal | `master` | `30f3a89` |
| prediction_provider | `musashi/m5phet-forecast-20260924` | `9f2108b` |
| causal-inference | `feat/m5phet-causal-provider` | `42e83d5` |
| agent-multi | `satoshi/m5phet-policy-provider-20260924` | `982042f0` |
| feature-eng | `codex/m5phet-hierarchical-regimes-20260924` | `6296447` |

Tests: M5PHET 237, news-signal 155, prediction_provider 89 native + 14 contract,
causal 19 contract, agent-multi 69, feature-eng 96. All CPU except the classification
answers, which come from the already-running worker under its existing lease.

Quality of every model remains unmeasured; the machinery to measure it exists and
waits on independent labels.
