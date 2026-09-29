# Satoshi — CB04 return: the classification metric contract, enforced in the existing evaluation and provider path

Orders: [CB04 in SATOSHI_CLASSIFICATION_REFERENCE_ORDERS_2026_09_28](../../handoffs/SATOSHI_CLASSIFICATION_REFERENCE_ORDERS_2026_09_28.md),
plan [CLASSIFICATION_REFERENCE_PLAN_2026_09_28](../../tres_temas_entrevista/program_v3/CLASSIFICATION_REFERENCE_PLAN_2026_09_28.md).
Branch `satoshi/cb04-classification-metric-contract-20260928`, own worktree, based on
`musashi/tsl-lake-extension-20260928` @ `a931d152` so the TSL receipt contract this one must not
be confused with is present and still passing. Commits `02a34033`, `0ab20c80`, `a0731156` plus the
closing commit of this document.

Independent of the data lanes by construction: this work package is schema, code and tests, and it
downloaded nothing, registered no dataset and ran no model. Coordinator only, everything through
`crispdm-run` at 512 MB – 2 GB per job. The admitted forecasting execution on the secondary worker
was not touched and nothing was placed beside it; the preferred external 5090 host was not used.
No service was started, stopped or restarted, no live warehouse was read or written, no
`git add -A`, and no host name, address, token or account identifier appears in any file added here.

**Satoshi, successor technical lead. 2026-09-28.** Nothing in this return is under Musashi's name.

---

## 1. Can one metric still be read as another? No — three separations and two refusals

This is the question the order asked to lead with, so it is answered first, and mechanically.

MAP, accuracy and macro-F1 all live in `[0, 1]`. The defence is not a sentence in a document; it is
that the three do not share a single storable representation anywhere in the path:

| separation | MAP | accuracy | macro-F1 |
|---|---|---|---|
| metric key | `classification.map` | `classification.accuracy` | `classification.macro_f1` |
| unit token | `map_ranking_fraction` | `accuracy_fraction` | `macro_f1_fraction` |
| metric identity digest | over family, the author's own name, definition, unit, denominator policy and label order — different in all three even when the value is bit-identical |

On top of the three separations, two refusals:

- `read_metric(receipt, family)` raises `MetricNotCarried` naming **both** families. A MAP receipt
  asked for accuracy returns no float, no nearest match and no default.
- `compare(left, right)` raises `IncomparableMetrics` naming **both** families and both kinds
  (`RANKING_OVER_LABEL_SCORES` against `TOP_ONE_AGREEMENT`, and so on). Within one family it still
  raises `IncomparableProtocol` when the task, the evaluation population or the metric identity
  differ.

The sharpest case is stored end to end, through the deployed provider, in
`test_three_families_with_one_value_stay_three_rows_in_the_warehouse`: the number `8/9` is written
as MAP, as accuracy and as macro-F1 — from a symmetric confusion whose accuracy and macro-F1 are
genuinely the same number — and a SQL query selecting `WHERE m.value = 8/9` returns **three rows
with three metric keys, three units and three identities**. The companion test then shows that
`WHERE metric = 'classification.accuracy'` cannot return the MAP row although the two values are
equal to the last bit.

One further mechanism was not in the order and is worth stating, because it caught a real mistake
while this was being written: a receipt whose headline accuracy or macro-F1 **contradicts its own
per-class confusion** is refused, naming the value carried, the value the confusion gives, and the
denominator policy under which they were compared. The first fixture written for the schema suite
declared `0.6` beside a confusion worth `0.888889`, and the check refused it. Both fields are still
carried in full; the check refuses only the case where two present fields disagree, which is
precisely the case a later reader cannot detect.

What this does **not** do: it does not stop somebody writing a MAP value into a receipt that
declares accuracy. Nothing at the schema layer can. What it does is make that a single explicit
false declaration, recorded under a name, a unit and an identity digest, instead of an ambiguity.

## 2. The frozen business corpus: what is in it, and how it was built

`docs/audits/evidence/cb04_business_corpus_20260928/` — corpus id `cb04_business_news.v1`, 45 items.

**Built after the protocol, with the evidence pinned.** The manifest records the sha256 of
`docs/contracts/classification_metrics.v1.json` and of `app/classification_receipt.py` as they
stood when the corpus was written, and both files precede the corpus in this branch's history
(`02a34033`, `0ab20c80`, then `a0731156`). A test re-computes both digests and fails if either file
changes without the pin being rebuilt deliberately. A corpus written first and a protocol fitted to
it afterwards has already been tuned against; this order is checkable, not asserted.

**It tests news use, not sentiment.** Three questions, none of which a public sentiment or intent
benchmark asks:

| question | vocabulary | counts |
|---|---|---|
| `relevance` — does this bear directly on the named asset's economics | related / unrelated / unclear | 26 / 12 / 7 |
| `novelty` — new information, a restatement of an already published item, or a correction of one | new / restatement / correction | 33 / 9 / 3 |
| `window` — does it matter immediately, within the session, or not at all | immediate / session / none | 12 / 12 / 21 |

**The cases it was built around**, one construction note per item in `labels.jsonl`:

- positive-tone irrelevance (a record load factor, a store-opening announcement) — tone is not
  relevance, and a sentiment model that scores well on a public set has not been asked this;
- items that name the pair while being about something else: a price report of the pair itself, an
  analyst note arguing the pair is mispriced — opinion about the pair is not news about it;
- three restatement chains of one underlying release, including a duplicate two minutes after the
  first print and a final reading confirming a flash estimate with no revision;
- a correction that reverses a magnitude (25 basis points → 50), and a separate item that *looks*
  like a correction and is a restatement (a republication after a transmission error, figure
  unchanged) — the two labels must not collapse;
- genuine abstention cases: an unattributed and unquantified remark, a survey split 20–21 on a
  non-comparable base, a leaked draft of unconfirmed authenticity, a cancelled appearance, and a
  headline with no body;
- out-of-domain rows that must be answered `unrelated` rather than abstained, so abstention cannot
  be used as a catch-all;
- both interface languages: 5 Spanish items mirroring English ones, so the same decision must
  answer the same way.

**Held apart, and opened on the record.** Items and labels are separate files, sealed separately,
and both seals are verified on every read. `app.business_corpus.load_items()` reads items freely and
records nothing. `open_labels()` requires an actor, a provider and a stated reason of at least
twelve characters, appends a numbered entry to `USE_LEDGER.jsonl` **before** returning the labels,
and does not prevent a second use — it makes it visible. That is the point: a corpus scored once is
validation; the same corpus scored eleven times with the prompt adjusted between attempts is
development data, and only the ledger tells them apart afterwards. `refuse_prior_fitted_on_held_out`
refuses a naive prior equal to the corpus's own label distribution, because a baseline that has seen
the answers is not a baseline. The ledger is at zero uses: no model has been run over this corpus.

**Marked as a declaration where it is one.** The items are fabricated and each carries
`synthetic: true`. No real organisation, person, publication or record is named, quoted or imitated:
institutional actors appear only as generic roles ("the euro-area central bank", "a national
statistics office"), and the manifest states that nothing here may be read as evidence about any
real institution. **The labels are the author's** — that is in `what_it_is_not`, not buried. A corpus
independent in the full sense needs third-party labels over licensed feed text, and this programme
holds no such entitlement; the refusal is recorded by name as
`INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT`. **A named refusal is not a
completed family**, so the business-usefulness family is not complete: what exists is a sealed,
held-apart screen of 45 rows, and 45 rows do not rank providers.

## 3. The two prohibitions with teeth

**No badge from the router score.** The router corpus is declared for what it is: 19 prompts × 5
repeats = 95 stored verdicts, 19 independent units, measuring
`ROUTER_ENVELOPE_SELECTION_NOT_CLASSIFIER_QUALITY`. Three mechanisms, not one:

1. `build_receipt` refuses `corpus_class: ROUTER_PROMPT_CORPUS`, and refuses the router corpus id
   under any other corpus class. A router evaluation cannot be expressed as a classification
   receipt at all.
2. Router reliability has its own schema, `router_reliability.v1`, produced by
   `build_router_reliability_record`. It carries no metric family, no class vocabulary and no
   provider quality, and both warehouse projections (`terminal_tags`, `terminal_metrics`) refuse it
   by name — so there is no route by which a router number is written in the shape a classifier's
   score is written in. A test asserts no metric family name appears anywhere in its JSON.
3. `provider_quality_badge` refuses the whole evidence set with
   `BADGE_REFUSED_ROUTER_SCORE_IS_NOT_CLASSIFIER_QUALITY` if a router record is present — including
   when it is mixed in beside otherwise good business evidence.

The record also refuses to misstate its own population: `95` never appears as an example count.
`prompts × repeats == stored_verdicts` is enforced, `independent_units` is `prompts`, and
`clustered_by` is `prompt`. On the classification side, `repeats > 1` requires
`independent_units × repeats == total` and a named `clustered_by`.

**No badge from a declaration alone.** `provider_quality_badge` refuses `DECLARATION`,
`RECOUNT_OF_STORED_VERDICTS` and `PUBLISHED_REFERENCE` with
`BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT`, a transport fixture with
`BADGE_REFUSED_TRANSPORT_TEST_IS_NOT_SCIENCE`, an evidence set with no held-out business
measurement with `BADGE_REFUSED_NO_BUSINESS_CORPUS_MEASUREMENT` — public benchmark accuracy alone
does not earn a badge — and a business measurement without a same-row train-derived or
development-fixed naive with `BADGE_REFUSED_NO_PAIRED_NAIVE`.

**Nothing grants execution.** `authorises_broker_deployment` is the constant `false` and
`execution_authority` the constant `NONE`, in the contract, in every receipt, in every badge and in
the corpus manifest, and it is stored as a terminal tag. A test asserts a badge carries no
deployment-shaped key at any value. No automatic broker deployment follows from any score here, and
this document creates no execution authority.

## 4. What was built, and where

| file | what it is |
|---|---|
| `docs/contracts/classification_metrics.v1.json` | the contract: metric families, paired naive policies, confusion orientation, probability semantics, calibration split rule, abstention policies, corpus and evidence classes, router corpus declaration, badge refusals, required context tags |
| `app/classification_receipt.py` | the enforcement: `build_receipt`, `read_metric`, `compare`, `terminal_tags`, `terminal_metrics`, `build_router_reliability_record`, `provider_quality_badge` |
| `app/business_corpus.py` | the held-out corpus loader: seal verification, use ledger, prior guard, population digest |
| `pipeline_plugins/binary_metrics.py` | `build_binary_classification_receipt` / `save_binary_classification_receipt` in the existing evaluation path |
| `pipeline_plugins/binary_pipeline.py` | calls them in its own run, step 8b, beside the plots |
| `tools/build_cb04_business_corpus.py` | the corpus builder and its recorded procedure; run once |
| `docs/audits/evidence/cb04_business_corpus_20260928/` | `MANIFEST.json`, `items.jsonl`, `labels.jsonl`, `USE_LEDGER.jsonl` |
| `docs/CLASSIFICATION_METRIC_CONTRACT.md` | the producer-facing note and the read-only query template |

The seven facts CB04 names are each their own field, and `build_receipt` refuses each by name when
absent: `author_primary_metric` (family, the author's own name, value, denominator policy),
`paired_naive` (same family, train-fitted, same evaluation rows), `class_vocabulary` (ordered, with
`label_order_sha256`), `per_class_confusion` (reference rows × predicted columns plus an `ABSTAINED`
column, closing over the population), `probability_semantics` **and** a separate `calibrated`
boolean, `abstention` (abstained, answered, coverage, denominator policy, rule), and
`calibration_split` kept distinct from the evaluation split by both id and population digest.

Not a parallel system: the receipt's warehouse projection is the existing `governed_terminal.v1`
with the existing nine-field metric rows and context in `tags`, exactly the route the TSL contract
uses. No table was added, no column was added, no historical row was rewritten or reinterpreted, and
no migration is required. `docs/contracts/tsl_literature_metrics.v1.json` remains a regression
MSE/MAE contract; this contract states in its own second key that it is not that one.

## 5. Suites — schema first, then end to end through the real store

Schema tests were written and committed **before** the implementation, at `02a34033`, and observed
red at that commit with `ImportError: cannot import name 'classification_receipt' from 'app'`. The
implementation followed at `0ab20c80`.

| suite | result | what it runs against |
|---|---|---|
| `tools/test_classification_receipt_schema.py` | 56/56 | the contract document on disk and the real receipt module |
| `tools/test_classification_warehouse_receipt.py` | 13/13 | the **installed** `PredictorDuckdbStore`, the distribution the warehouse host runs, on a temporary DuckDB file |
| `tools/test_cb04_business_corpus.py` | 22/22 | the real sealed corpus files and the real loader |
| `tools/test_binary_pipeline_classification_receipt.py` | 12/12 | the real pipeline builder, with scikit-learn as an independent oracle |
| `tools/test_tsl_warehouse_receipt.py` (pre-existing) | 2/2 | unchanged, re-run to show the TSL contract still passes |

**On the defect that stayed invisible because its test substituted the real function away.** The
end-to-end suite patches nothing, and `test_nothing_under_test_was_substituted` fails if a later
edit introduces a double: it asserts the provider class resolves into `site-packages`, that its
module is `predictor_duckdb_store.provider`, that `self.store` is that exact type, that
`write_terminal` is the provider's own function and not a rebound attribute, and that every receipt
function resolves to `app/classification_receipt.py` on disk. In the pipeline suite the same
concern is answered differently: `test_the_binary_pipeline_actually_calls_the_builder` reads
`BinaryPipelinePlugin.run_prediction_pipeline`'s source and fails if the call is removed, because a
receipt builder nothing calls would pass every other test in this return.

The end-to-end suite also proves what the plan asked of a temporary warehouse: exact roundtrip of
tags and metric rows, idempotency under triple resubmission, persistence across disposing and
reopening the connection, refusal of NaN and ±infinity with zero rows left behind, abstention
coverage and the confusion closing from the stored rows alone, and both badge refusals evaluated
from what the warehouse actually holds.

Independent-oracle check in the pipeline suite: the contract's macro-F1 and accuracy are compared
against `sklearn.metrics.f1_score(average="macro")` and `accuracy_score` on the same rows, both with
no abstention rule and with `abstain_band = 0.08` where the comparison is restricted to the answered
rows. Agreement to ten decimal places. That is a different implementation, not the same arithmetic
run twice.

## 6. Report, in the §6 shape

```
CB04 — classification metric contract and business validation
repo/branch/tip: predictor / satoshi/cb04-classification-metric-contract-20260928 / a0731156 + this document
files: docs/contracts/classification_metrics.v1.json · app/classification_receipt.py ·
       app/business_corpus.py · pipeline_plugins/binary_metrics.py · pipeline_plugins/binary_pipeline.py ·
       tools/build_cb04_business_corpus.py · tools/test_classification_receipt_schema.py ·
       tools/test_classification_warehouse_receipt.py · tools/test_cb04_business_corpus.py ·
       tools/test_binary_pipeline_classification_receipt.py ·
       docs/audits/evidence/cb04_business_corpus_20260928/{MANIFEST.json,items.jsonl,labels.jsonl,USE_LEDGER.jsonl} ·
       docs/CLASSIFICATION_METRIC_CONTRACT.md
suites: schema 56/56 · warehouse end-to-end 13/13 (installed provider, disposable cube) ·
        business corpus 22/22 · pipeline wiring 12/12 · pre-existing TSL receipt 2/2
acceptance: one metric cannot be read as another — distinct key, distinct unit, distinct identity
        digest, and 8/9 stored as MAP/accuracy/macro-F1 stays three rows through the deployed
        provider; read_metric and compare refuse by name; router corpus barred from carrying any
        classification metric and from any badge; badge refuses declaration, recount, published
        reference, transport fixture, no-business-measurement and no-paired-naive; every badge and
        receipt carries execution_authority NONE
what is NOT done / refused / not measured:
  - NO_NEW_MODEL_MEASUREMENT. No classifier was run, no dataset downloaded, no score produced. Every
    number in every test is a fabricated fixture, labelled TRANSPORT_TEST_NOT_SCIENCE or a schema
    fixture. Nothing here is a model result.
  - INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT. The business corpus is sealed,
    held apart and built after the protocol, but its labels are the author's and its items are
    fabricated. Business usefulness is therefore NOT a completed family.
  - The corpus is 45 rows: a screen, not a ranking. Its use ledger is at zero uses.
  - MAP is not implemented as a scorer here. The contract gives it a key, a unit, an identity and a
    refusal; computing it is the benchmark evaluator's job under CB02/CB03, and FinMTEB's
    paper/code discrepancy is unresolved and out of this package.
  - No live warehouse row was written or read, no service touched, no lake or data-gov registration
    performed, and the provider quality badge has never been issued: only its refusals have run.
  - M5PHET's own quality block (news_signal.quality.v1) was NOT changed. Aligning it to this
    contract is a separate change in that repository.
  - No provider was measured, so no provider is claimed better than any other, and no state of the
    art is claimed for anything.
```

## 7. Review request

Review, in this order: (1) whether the three separations plus two refusals actually close the
MAP/accuracy/macro-F1 confusion, or whether a reader can still line two of them up — attack
`compare` and `read_metric` first; (2) whether the router record's separate schema is a real barrier
or a convention I could route around; (3) whether the business corpus's declared limitations are
stated strongly enough, given that I wrote both its items and its labels; (4) whether
`test_nothing_under_test_was_substituted` and the pipeline source assertion would actually catch the
substitution class of defect, or whether they are theatre.
