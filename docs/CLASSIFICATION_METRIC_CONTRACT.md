# Classification receipts in the existing warehouse

## What this is, and what it is not

`classification_metrics.v1` is a **producer** convention for classification evaluations, the way
[`tsl_literature_metrics.v1`](contracts/tsl_literature_metrics.v1.json) is one for regression
benchmarks. It is implemented in
[`docs/contracts/classification_metrics.v1.json`](contracts/classification_metrics.v1.json) and
enforced by `app/classification_receipt.py`.

The TSL MSE/MAE contract is **not** a classification contract. None of its metric names, units or
reductions apply here, and neither contract is a global validator: the general-purpose warehouse
stays generic, no historical row is rewritten, and no migration is required. New producers populate
the context fields explicitly; installing this contract rewrites nothing.

No schema change was needed. Receipts travel the existing campaign → delivery → terminal → warehouse
route as `governed_terminal.v1` terminals, with the contract context in terminal `tags` and the
existing nine-field metric rows in `metrics`.

## The rule this contract exists for

MAP, accuracy and macro-F1 all lie between zero and one and are three different metrics. Each
carries its own metric key, its own unit token and its own `metric_identity_sha256`, so a stored
value cannot be read as a family it did not come from:

| family | metric key | unit |
|---|---|---|
| MAP | `classification.map` | `map_ranking_fraction` |
| ACCURACY | `classification.accuracy` | `accuracy_fraction` |
| MACRO_F1 | `classification.macro_f1` | `macro_f1_fraction` |
| WEIGHTED_F1 | `classification.weighted_f1` | `weighted_f1_fraction` |
| MCC | `classification.mcc` | `mcc_coefficient` |

`read_metric(receipt, family)` refuses a family the receipt does not carry, naming both.
`compare(left, right)` refuses across families, naming both, and refuses within a family across a
different task, population or metric identity. A matching metric name alone is not evidence of
comparability, and here a mismatched one is a refusal rather than a coincidence.

## What every receipt carries

Each of these is its own field. None is omitted and recomputed at read time, and a missing one is
refused by name:

- **the author's primary metric**, under the author's own name for it, with its denominator policy;
- **a paired naive of the same family**, fitted on train labels and scored on exactly the same
  evaluation rows (`MAJORITY_CLASS_FROM_TRAIN`, `STRATIFIED_PRIOR_FROM_TRAIN`,
  `UNIFORM_OVER_VOCABULARY` or a `DEVELOPMENT_FIXED_KEYWORD_RULE` frozen before the evaluation
  split was opened);
- **the class vocabulary**, ordered, with `label_order_sha256`. Truncating, reordering or merging
  labels is a new task and changes the digest;
- **the per-class confusion**: reference rows × predicted columns plus a final `ABSTAINED` column,
  closing over the population. The full matrix is bound by `confusion_sha256`, reaches the warehouse
  as `classification.{support,predicted,correct,abstained}.class_<i>` rows, and is carried in the
  `confusion_matrix_json` tag while its canonical form is at most 8192 bytes (a 77-class matrix stays
  in the receipt, with the digest binding it);
- **the probability semantics** and a separate **`calibrated`** boolean. A contradiction between the
  two is refused. An entropy-derived confidence is not P(correct) and may not carry NLL, Brier or
  ECE; ECE requires declared bins;
- **abstention coverage** with its denominator policy, because an abstention is not a prediction.
  `ANSWERED_ONLY` keeps refusals out of the denominator and states the coverage so the reader sees
  what was dropped; `FULL_POPULATION_ABSTENTION_WRONG` counts them as errors. Neither removes a
  refusal silently;
- **the calibration split**, distinct from the evaluation split by both id and population digest.

Consistency between fields is checked: a headline accuracy or macro-F1 that contradicts the receipt's
own confusion is refused, naming both values and the policy under which they were compared.

## Where the evidence came from, and what it may become

Added 2026-09-29. Every receipt carries an **`answering_path`** block, and it is checked rather than
recorded:

| field | what it says |
|---|---|
| `path_id` | which code path produced the answers |
| `kind` | what that path is, from a closed vocabulary (`MODEL_CHECKPOINT_LOADED`, `MODEL_IN_PROCESS_NOT_CHECKPOINTED`, `MODEL_WEIGHTS_ABSENT`, `NON_MODEL_RULE`, `NON_MODEL_CONSTANT`, `AUTHORED_VALUES_NOT_EXECUTED`, `THIRD_PARTY_PATH_NOT_RUN_HERE`) |
| `weights_present` | whether learned parameters were loaded. The kind determines it, so the two cannot drift apart |
| `served_checkpoint` / `served_checkpoint_sha256` | which checkpoint served — a digest, or `CHECKPOINT_NOT_DIGESTED`, or `NO_CHECKPOINT_SERVED` |
| `attestation` | how that was established: `OBSERVED_FROM_ANSWERING_PATH`, `DECLARED_BY_CONFIGURATION`, `NOT_ESTABLISHED` |

Four rules over it, each a named refusal:

- **a path without weights may not carry a checkpoint digest.** `checkpoint_sha256` was an
  unconditional hex64 before, so a path that had loaded nothing had to invent one — and an invented
  digest is shaped exactly like a real one.
- **a receipt may not quote a checkpoint the answering path did not serve.** That is this
  warehouse's form of the withdrawal the product made the same day, where a real checkpoint's
  macro-F1 was published beside a declared non-model path's answers.
- **a promotion is refused by reason.** `evidence_class: MEASUREMENT` requires a path with weights
  and an attestation other than `NOT_ESTABLISHED`. A path without weights is a **declared test**: it
  is built, projected and stored, and `evidence_role` on the receipt and on the terminal tag says
  which of the two a row is, so a query for model results does not select it.
- **a badge may not rest on a declared test.** It additionally requires the answering path to have
  been `OBSERVED_FROM_ANSWERING_PATH`, and refuses a record whose checkpoint is not the served one.

**No word is banned.** The gate never matches a provider, checkpoint or corpus name against a
vocabulary of suspicious words, and `tools/test_classification_provenance.py` fails if such a match
is reintroduced. Two counterexamples are kept: a declared test whose every field is free of the
withdrawn word is still refused promotion, and a real measurement that carries that word in its own
`limitations` text is admitted, stored and badge-eligible.

### The store boundary

The producer-side gate binds only producers that call it. Measured against the deployed provider on
a disposable cube, the store took a terminal whose tags declared a non-model answering path and
claimed a model result, under any actor name — so the boundary carries its own check:
`app.classification_provenance.admit_classification_terminal`, also available as
`tools/admit_classification_terminal.py`. It reads the **tags only**, so a terminal hand-written by
anybody is held to the same contract; a terminal of another `metric_contract` is returned
`NOT_THIS_CONTRACT` and admitted, because the general-purpose warehouse stays generic.

## Provenance, and what a number may not become

`evidence_class` is one of `MEASUREMENT`, `PUBLISHED_REFERENCE`, `DECLARATION`,
`RECOUNT_OF_STORED_VERDICTS`, `TRANSPORT_TEST_NOT_SCIENCE`. `declared_fields` names, field by field,
what is a declaration rather than a measurement. A published value is its own receipt with its source
and table; it is never labelled as our measured result.

`supervision_regime` is explicit, and if labelled rows fit a downstream head the regime is not
zero-shot — that is refused, even on the same test rows.

Two prohibitions are enforced rather than documented:

- **the router corpus may not carry a classification metric.** 19 prompts × 5 repeats = 95 stored
  verdicts, 19 independent units. `build_receipt` refuses the corpus class and the corpus id; router
  reliability has its own schema, `router_reliability.v1`, with no metric family and no warehouse
  projection; and `provider_quality_badge` refuses any evidence set containing one.
- **no badge from a declaration alone.** A badge requires a `MEASUREMENT` on a `BUSINESS_HELD_OUT`
  corpus with a same-row train-derived naive. Public benchmark accuracy alone is refused.

Nothing carries execution authority. `authorises_broker_deployment` is `false` and
`execution_authority` is `NONE`, in the contract, every receipt, every badge and the corpus
manifest. No broker deployment follows from any score.

## Query template (DuckDB, read-only)

```sql
SELECT t.campaign_key, t.unit_id, t.generation,
       json_extract_string(t.tags_json, '$.task_id')                      AS task,
       json_extract_string(t.tags_json, '$.corpus_class')                 AS corpus_class,
       json_extract_string(t.tags_json, '$.evidence_class')               AS evidence_class,
       json_extract_string(t.tags_json, '$.supervision_regime')           AS regime,
       json_extract_string(t.tags_json, '$.provider')                     AS provider,
       json_extract_string(t.tags_json, '$.author_primary_metric_family') AS family,
       json_extract_string(t.tags_json, '$.author_primary_metric_name')   AS author_name,
       json_extract_string(t.tags_json, '$.metric_identity_sha256')       AS metric_identity,
       json_extract_string(t.tags_json, '$.denominator_policy')           AS denominator,
       json_extract_string(t.tags_json, '$.independent_units')            AS independent_units,
       m.metric, m.value, m.unit, m.split
FROM gov_terminal t
JOIN gov_terminal_metric m USING (terminal_sha256)
WHERE json_extract_string(t.tags_json, '$.metric_contract') = 'classification_metrics.v1'
  AND t.status = 'COMPLETED'
ORDER BY task, provider, m.metric
LIMIT 1000
```

Select model results with `evidence_role = 'MODEL_RESULT'`; every other row of this contract is a
declared test, kept on purpose. Group by `metric_identity_sha256`, never by `value` and never by
`metric` alone across tasks. Two
rows agree only when their identity digests agree. This query is an inventory, not a scientific
closure: select accepted runs and adjudicated generations before aggregating, and it currently
returns nothing, because no classification benchmark run has been launched for this contract.

## Verification performed

Against a temporary DuckDB file only, using the store host's installed provider:
`tools/test_classification_warehouse_receipt.py`, 13 tests — exact tag and metric-row roundtrip,
idempotency under resubmission, persistence across reopening, refusal of NaN and ±infinity with zero
rows left, abstention closing from stored rows alone, both badge refusals, and the three-family
one-value separation. Nothing is mocked, and one test fails if a double is introduced.
`tools/test_classification_receipt_schema.py` 56 tests, `tools/test_cb04_business_corpus.py` 22,
`tools/test_binary_pipeline_classification_receipt.py` 12 (macro-F1 and accuracy checked against
scikit-learn as an independent oracle). All fixture values are fabricated transport or schema
fixtures, not model scores. No live warehouse row was written or read.

## The held-out business corpus

`docs/audits/evidence/cb04_business_corpus_20260928/` — 45 items, three news-use questions
(relevance, novelty, window), built after this contract with the contract's digest pinned in the
manifest, items and labels sealed separately, labels handed over only against an appended use-ledger
entry. Its items are fabricated and its labels are the author's; see the manifest's
`what_it_is_not` and its named refusal
`INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT`. Sentiment or topic accuracy does
not establish news relevance, causal effect or profitability.
