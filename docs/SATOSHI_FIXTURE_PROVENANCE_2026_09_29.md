# Provenance as a checked field: a declared test stays a declared test

**Author:** Satoshi, successor technical lead
**Date:** 2026-09-29
**Repository / branch / base:** `predictor` · `satoshi/fixture-provenance-contract-20260929` · base
`40224a3a` (the CB04 metric contract named in the order)
**Worktree:** `$HOME/Documents/GitHub/.worktrees/predictor-fixture-provenance-20260929`
**Order:** order 4 of `MUSASHI_AUDIT_23B2EFA3_2026_09_29.md` — *asumir el caso fixture en contrato
productor/almacén: conservar pruebas declaradas como tales, impedir promoción como modelo real o
badge. No prohibir una palabra aislada como sustituto de validar procedencia.*

> **`NO_NEW_MODEL_MEASUREMENT`.** No classifier, forecaster or policy ran. The real Laya weights were
> not contacted, so **nothing here measures a checkpoint**. Every number in every test is a
> fabricated transport or schema value, declared as one. `NO_NEW_MEASUREMENT` in the owner's standing
> sense, so **no closure table is offered**.

---

## 1. Leading with the question the order asked to lead with

**Can a declared test still be promoted into a model result, or into a quality badge?**

In this repository's producer-to-store contract, **no — on both paths, by reason, and the refusal
names the reason rather than a word.** Four gates, each with a named refusal and a red-first test:

| the promotion | where it is refused | the reason it is refused by |
|---|---|---|
| a path with no weights labelled `MEASUREMENT` | `build_receipt`, and again at the store boundary | `PROMOTION_REFUSED_A_PATH_WITH_NO_WEIGHTS_IS_NOT_A_MODEL_RESULT` |
| a path with no weights carrying a checkpoint digest | `build_answering_path` | `PROVENANCE_REFUSED_CHECKPOINT_DIGEST_ON_A_PATH_THAT_SERVED_NONE` |
| a receipt quoting a checkpoint that path did not serve | `build_receipt`, the badge, the boundary | `PROVENANCE_REFUSED_RECORD_IS_NOT_OF_THE_ANSWERING_PATH` |
| a badge resting on a declared test, however the record is labelled | `provider_quality_badge` | `BADGE_REFUSED_NON_MODEL_ANSWERING_PATH` |

And **what a declared test can still do is everything it was for**: it is built, projected into the
warehouse, stored, and stays legible as a test. `evidence_role` is its own receipt field and its own
terminal tag, so a reader who joins nothing sees which of the two a row is, and a query for
`evidence_role = 'MODEL_RESULT'` does not select it (proved against the deployed store, not asserted).

**Three residuals, named rather than implied away.** (i) The gate binds the producers that call it
and the boundary command a loader puts in front of a write; the **installed generic store was not
changed** and still accepts any well-shaped terminal from anyone, which is why the boundary check
exists and why its refusals are printed with names. (ii) `MODEL_IN_PROCESS_NOT_CHECKPOINTED` allows
a real measurement with no digestible checkpoint — deliberately, because refusing it would hide
honest work — so for that kind the served digest is a sentinel and cannot be checked against bytes.
(iii) A producer that declares its provenance *falsely* is not detected here; the contract makes the
declaration exist, explicit and digest-bound, and makes a false one a lie on the record instead of
an absence.

## 2. How provenance is validated **without** relying on a banned word

The obvious repair is the one the order forbids, and my own lane had already written it: the
product's first withdrawal read `backend != "fixture"`. That is a spell check. A declared test may
be named anything at all, and an honest measurement's own prose may legitimately contain that word.

What is checked instead is **where the evidence came from** — one block, six fields, each its own:

```
answering_path: { path_id, kind, weights_present,
                  served_checkpoint, served_checkpoint_sha256, attestation }
```

* **`kind`** comes from a closed vocabulary in the contract document, and **the kind determines
  `weights_present`**, so the two cannot drift apart: `MODEL_CHECKPOINT_LOADED`,
  `MODEL_IN_PROCESS_NOT_CHECKPOINTED`, `MODEL_WEIGHTS_ABSENT`, `NON_MODEL_RULE`,
  `NON_MODEL_CONSTANT`, `AUTHORED_VALUES_NOT_EXECUTED`, `THIRD_PARTY_PATH_NOT_RUN_HERE`.
* **the digest axis is explicit**: a path with weights carries a sha256 or `CHECKPOINT_NOT_DIGESTED`;
  a path without weights carries `NO_CHECKPOINT_SERVED`, and a digest there is refused. Before this
  round `checkpoint_sha256` was an unconditional hex64, so a path that had loaded nothing *had to
  invent one* — and an invented digest is shaped exactly like a real one. That shape was the defect.
* **`attestation`** separates observation from configuration: `OBSERVED_FROM_ANSWERING_PATH`,
  `DECLARED_BY_CONFIGURATION`, `NOT_ESTABLISHED`. A `MEASUREMENT` needs one of the first two; a
  **badge needs the first**. Absence is not coincidence: an undeclared block is refused, not read as
  a model path.
* **`provenance_sha256`** is a tag over all six, so an editor cannot change one provenance tag and
  leave the rest agreeing with each other.

**The gate never matches the withdrawn word.** `app/classification_provenance.py` names it once, in
its own docstring, to say what it does not do; no executable literal in it carries the word, and it never
matches a provider, checkpoint, actor or corpus name against a vocabulary of suspicious words, and
`test_the_gate_never_matches_a_word` parses the module with `ast` and fails if any non-docstring
string literal in it contains that word. Two counterexamples are retained and both run:

- **a declared test that carries the word nowhere is still refused promotion.** The test asserts
  `WITHDRAWN_WORD not in json.dumps(document).lower()` *before* building, so the counterexample is
  only meaningful if the gate is not reading text — then asserts the refusal, and asserts the refusal
  message itself does not contain the word.
- **a real model measurement that carries the word in its own prose is not blocked.** `task_id`
  `news_relevance_fixture_window.v1` and a `limitations` sentence about a harness that replaced a
  fixture with the real corpus: built, `evidence_role: MODEL_RESULT`, admitted at the boundary,
  stored, and badge-eligible.

Two more counterexamples guard against over-refusing: an in-process model with no digestible
checkpoint is still measurable, and a terminal of another `metric_contract` is returned
`NOT_THIS_CONTRACT` and admitted untouched — the general-purpose warehouse stays generic.

## 3. The two live facts, measured rather than asserted

### 3.1 What the contract accepted at the base tip

`docs/audits/evidence/fixture_provenance_20260929/probe_contract.py`, run at `40224a3a`
(`00_red_today.json`). A constant answer table, `provider` and `checkpoint` both
`canned_answer_table.v1`, `evidence_class: MEASUREMENT`, a fabricated 64-hex `checkpoint_sha256`:

```json
{"receipt_built": true,
 "stored_checkpoint_sha256": "be4e4dc1f1e4907ebc4040e2a6c2ebcba6bf79cc8211367a3aceedb760503840",
 "declares_the_answering_path": false,
 "tags_name_the_answering_path": [],
 "badge_issued": true, "badge_value": 0.8888888888888888}
```

A canned table earned a **provider quality badge of 0.8889**, and no field anywhere named what had
answered. The same script at this tip (`05_same_document_now.json`) returns
`ReceiptRefused: answering_path is required and was not given`; declaring the path honestly
(`--declared`, `06_declared_honestly.json`) builds the receipt, stores
`checkpoint_sha256: NO_CHECKPOINT_SERVED`, `evidence_role: DECLARED_NON_MODEL_TEST`, and the badge
refuses by name.

### 3.2 The store alone is not a defence

`test_the_store_alone_is_not_a_defence` writes, through the **deployed** `PredictorDuckdbStore` from
site-packages onto a temporary DuckDB file, a terminal whose tags declare a non-model answering path
and claim `evidence_role: MODEL_RESULT`. **It is stored, and the test asserts it is** — if that ever
fails, the store grew a gate of its own and this test says so. The boundary check then refuses the
same bytes by reason. A second case rewrites the provenance digest consistently to show the rule is
the path and not the seal; a third forges only `evidence_role`; a fourth offers a declared test as a
`GOVERNING` terminal.

The actor's name is incidental throughout: the retained declared-test case is stored under
`actor = "fixture"` and admitted, and the retained measurement is stored under `actor = "nightly-eval"`.

### 3.3 The product's side, carried forward and not re-measured

The withdrawal of the real checkpoint's macro-F1 from beside a declared path's answers was delivered
on M5PHET at `d6e5f7c` (§3.4) and **is not repeated here**. What this round adds is the same rule in
the warehouse, where the product's rule had no counterpart: a record may not travel with answers it
was not measured on. The product's own rule still reads the word as one of two disjuncts, which this
order names as the thing not to rely on; that repair is **not done in this delivery** (§7).

## 4. A real producer stopped claiming a checkpoint it never read

`pipeline_plugins/binary_metrics.py` wrote `checkpoint_sha256` as a digest of four
hyperparameters — plugin name, epochs, window size, learning rate — a 64-hex value in the exact shape
of a checkpoint digest, for a checkpoint nothing had read. It now declares
`MODEL_IN_PROCESS_NOT_CHECKPOINTED`, `weights_present: true`,
`served_checkpoint_sha256: CHECKPOINT_NOT_DIGESTED`, attestation `OBSERVED_FROM_ANSWERING_PATH` —
because the model is fitted and scored in that same process, which is an observation, and the array
the metrics came from was the in-process model's and not that file's. Its 12 tests stay green,
including the two that check macro-F1 and accuracy against scikit-learn as an independent oracle.

## 5. The sealed corpus was not opened

The CB04 held-out business corpus pins the contract digest it was sealed against, and changing the
contract invalidated that pin. The pin is **appended to, never rewritten**: the original
`contract_sha256` stays exactly as written (a new test asserts its literal value), and an amendment
entry records the new digests, the date, why, and that the corpus bytes did not move — with
`items_sha256` and `labels_sha256` re-stated and re-verified. `USE_LEDGER.jsonl` is still **0 lines**:
no labels were opened, nothing was read, and nothing was tuned, selected or fixed against it. The
frozen held-apart router paraphrase set was not touched and its digest is unchanged; the development
router corpus (19 prompts × 5 repeats, 19 independent units) was not used to tune anything and still
may not carry a classification metric at all.

## 6. Report block

```
D — the fixture case in the producer-to-store contract
repo/branch/base: predictor · satoshi/fixture-provenance-contract-20260929 · base 40224a3a
files: app/classification_provenance.py (new, the gate) · app/classification_receipt.py ·
       docs/contracts/classification_metrics.v1.json (answering_path, promotion_rule,
       store_admission, 4 badge refusals, 8 required tags) · pipeline_plugins/binary_metrics.py ·
       tools/admit_classification_terminal.py (new, the boundary command) ·
       tools/test_classification_provenance.py (new, 31) ·
       tools/test_classification_provenance_store.py (new, 10) ·
       tools/test_classification_receipt_schema.py · tools/test_classification_warehouse_receipt.py ·
       tools/test_cb04_business_corpus.py · docs/CLASSIFICATION_METRIC_CONTRACT.md ·
       docs/audits/evidence/cb04_business_corpus_20260928/MANIFEST.json (amendment appended) ·
       docs/audits/evidence/fixture_provenance_20260929/ (probe, boundary cases, 7 logs)
suites: provenance 31/31 · provenance-at-the-store 10/10 (deployed provider) · schema 56/56 ·
        warehouse end-to-end 13/13 (deployed provider) · cb04 corpus 24/24 · binary pipeline 12/12 ·
        pre-existing TSL receipt 2/2 — 148 tests, 0 failures
acceptance: red first (01_red_producer.txt ImportError; 00_red_today.json badge issued at 0.8889 with
        a fabricated checkpoint digest and no field naming the path) → green
        (02_producer.txt, 03_store.txt, 07_suites.txt); boundary command on two terminals
        (04_boundary_cli.txt): declared test admitted DECLARED_NON_MODEL_TEST, the same terminal
        promoted refused PROMOTION_REFUSED_A_PATH_WITH_NO_WEIGHTS_IS_NOT_A_MODEL_RESULT, exit 2
the three outcomes, still distinct: a declared non-model path answering and being reported as one
        (DECLARED_NON_MODEL_TEST, NO_CHECKPOINT_SERVED, stored, queryable as a test); a real
        provider's measurement (MODEL_RESULT, its served checkpoint, badge-eligible); and an
        abstention, which remains a refusal and not a prediction — carried in its own coverage row
        with its denominator policy, never removed silently
the five areas, carried forward unchanged from d6e5f7c and NOT re-measured here: one has a quality
        figure and it is a RETAINED RECORD, two are NOT_MEASURED, two REFUSE BY NAME. A named
        refusal is not a completed family.
what is NOT done / refused / not measured:
  - NO_NEW_MODEL_MEASUREMENT. No classifier ran; the real Laya weights were not contacted; nothing
    here measures a checkpoint. No closure table (NO_NEW_MEASUREMENT).
  - the installed generic store was NOT changed and still accepts a well-shaped terminal from
    anyone; the boundary check is a repository-side gate a loader must call, not a store feature.
  - the product's own rule (M5PHET quality.py) still reads the withdrawn word as one of two
    disjuncts; that repair is prepared in §7 and NOT delivered.
  - a falsely declared provenance is not detected; the contract makes the declaration exist and be
    digest-bound, not true.
  - MODEL_IN_PROCESS_NOT_CHECKPOINTED cannot be checked against weight bytes, by design.
  - the sealed business corpus was not opened (use ledger 0 lines); the frozen paraphrase set was
    not touched; the development router corpus was not used to tune, select or fix anything.
  - no service was started, stopped or restarted; no GPU work; no production store touched — every
    store test ran on a temporary DuckDB file.
```

## 7. What the next hand should take

1. **The product's disjunct.** `M5PHET src/m5phet/quality.py:178` reads
   `weights is not False and backend != "fixture"`. The first half is provenance; the second is the
   word. The replacement is available now: consult whether the declared backend's capabilities name
   a **served checkpoint** at all, and withhold when they do not — which catches the same case
   without the literal. It is a small change with its own tests, in a different repository, and it
   should be delivered under its own branch and verified on its own port and state directory.
2. **A checkpoint in the quality record.** `news_signal.quality.v1` carries no checkpoint of its own,
   so no rule anywhere can check a record *against* the checkpoint that served it. Until that field
   exists, the strongest available statement remains "this record is not of the answering path", and
   never "this record is of it".
3. **The boundary in the loader.** `tools/admit_classification_terminal.py` exists and exits 2 on a
   refusal. Putting it in front of the warehouse loader is a deployment step and was not taken.

---

*Satoshi, successor technical lead — 2026-09-29. Nothing in this document carries execution
authority: `authorises_broker_deployment` is `false` and `execution_authority` is `NONE` in the
contract, in every receipt, in every badge and in the corpus manifest.*
