# CB-C: the classification recount, and the receipts that became warehouse rows

Satoshi, successor technical lead, 2026-09-29.

Lane **C** of the concurrent dispatch in
[SATOSHI_Q2_RESOURCE_SUCCESSOR_2026_09_29](../../handoffs/SATOSHI_Q2_RESOURCE_SUCCESSOR_2026_09_29.md)
(`04f02555`), reconciling the native reproduction at `19a37baf` and the metric
contract at `40224a3a` under the CB orders in
[SATOSHI_CLASSIFICATION_REFERENCE_ORDERS_2026_09_28](../../handoffs/SATOSHI_CLASSIFICATION_REFERENCE_ORDERS_2026_09_28.md).
This branch is based on `19a37baf`, which already carries `40224a3a` through
`d857d9fa`, with `04f02555` merged so the dispatch it answers is in the tree.
Nothing here is signed by, attributed to, or written on behalf of Musashi. No
service was started, stopped or restarted.

**Leading with the count, because the whole point of this lane is that the two
things are not the same claim.**

| | before this lane | after this lane |
| --- | ---: | ---: |
| retained warehouse projections (a file on disk) | **1** | 3 |
| **accepted warehouse rows** (a terminal the live warehouse hands back) | **0** | **3** |

Before this lane the live warehouse held **zero** rows under
`classification_metrics.v1`. CB03's return was right to say the projection was
*retained and not written*, and a reader skimming it could have concluded
otherwise, because the projection is complete, contract-shaped and indexed in a
manifest. It was still only a file. It now holds three terminals, 72 metric rows
and 15 artifact rows, and each terminal's digest was read back out of it.

**And one correction the recount forced, in the direction of claiming less.**
CB03 named World, 103/400 = 0.2575, as the strongest of the four legitimate
tie-breaks. Recounting the population's own support from the retained gold
labels gives world 103, **sports 123**, business 72, sci_tech 102, so the
strongest break is **Sports at 123/400 = 0.307500**. The honest statement is
**0.9525 against a paired naive between 0.180000 and 0.307500** on the same
rows — the skill margin against the strongest naive is 0.6450, not the 0.6950
the earlier upper end implied.

---

## 1. Resources, and what was not used

Coordinator only. Every child ran through `$HOME/.local/bin/crispdm-run` at the
already-deployed launcher, with a **fresh aggregate admission per child** and a
declared cap. No cap was shrunk to get past anything, nothing was displaced, no
other job was probed or signalled, and no admission was refused in this lane.

| Child | Cap | Whole-tree cgroup peak | Cap headroom |
| --- | ---: | ---: | ---: |
| `cbc-m-recount` (the recount) | 2.00 GiB | **11,980,800 B** (0.0112 GiB) | 179× |
| `cbc-m-receipts` (the three receipts) | 3.00 GiB | **258,584,576 B** (0.2409 GiB) | 12× |
| `cbc-m-warehouse-dry` (the write path, dry) | 3.00 GiB | **39,206,912 B** (0.0365 GiB) | 82× |
| `cbc-m-query` (the read-only cube query) | 2.00 GiB | **35,266,560 B** (0.0328 GiB) | 61× |

Each peak is the whole process tree's `memory.peak`, read **from inside the
launcher's own scope before the scope was removed**, not one process's RSS. The
120,000-row train parquet the paired naive's population digest is recomputed
from is what makes the receipts child the largest of the four.

**A defect observed in the launcher, reported because lane A is building exactly
this instrument.** The launcher's *retained lease* records under
`crispdm-admission retained` report `observed_peak_bytes` **27,262,976 B** for
`cbc-receipts-20260929`, against the 258,584,576 B the same workload's cgroup
actually peaked at — a **9.5× undercount**. The sampler does not capture a
representative peak for a child that finishes in a few seconds. The retained
record is therefore a **lower bound** for short children and must not be quoted
as a measured footprint; the numbers in the table above are the measured ones.
Nothing in this lane depended on the difference, because every peak is two
orders of magnitude under its cap either way.

**No GPU, and that is not a shortfall.** The published AG News cell is
`device: cpu` and the author's harness hard-codes CPU, so an idle accelerator is
not a missing resource here. The preferred external 5090 host was **not used and
is not eligible**: about 3 GiB of 14 available against a 4.93 GiB unreclaimable
kernel slab, read minutes before this lane started. Its cost and blocker are
retained and no substitute was presented under its name. The reusable state on
the secondary worker — the digest-verified checkpoint and the two pinned
environments — was **not needed at all**, because this lane is a recount and a
warehouse write: it loads no model and downloads nothing.

---

## 2. The recount: every retained number, recomputed from the per-row arrays

`tools/df_cbc_recount_20260929.py` reads the retained artefacts, **ignores every
stored aggregate while recomputing**, and only then compares. It loads no model,
opens no dataset and contacts no service.

The ten retained files' digests all match `MANIFEST.json`, and
`parity_400.json` binds the `native_published400.json` bytes **actually on
disk** (`6078a736…`), so the parity run cannot be pointing at a different
reference than the one retained.

| Recounted from per-row | Recount | CB03's stored value | Verdict |
| --- | ---: | ---: | --- |
| accuracy | 0.9525 | 0.9525 | exact |
| correct / denominator | 381 / **400** | 381 / 400 | exact |
| 4×5 confusion incl. ABSTAINED | `[[99,3,1,0,0],[2,121,0,0,0],[1,0,62,9,0],[0,0,3,99,0]]` | identical | exact |
| macro-F1 | 0.9467546527629132 | 0.9468 | agrees at the stored precision |
| ECE, 15 bins | 0.1550295715337212 | 0.155 | agrees at 3 dp |
| Brier | 0.11366765416255706 | 0.1137 | agrees at 4 dp |
| NLL | 0.2800873024645501 | 0.2801 | agrees at 4 dp |
| mean confidence | 0.7974704284662788 | 0.7975 | agrees at 4 dp |
| accuracy at 50 % coverage | 1.0 | 1.0 | exact |
| dropped | 0 | 0 | exact |

One check CB03 did not have to pass and does: the retained **probabilities
reconstruct from the retained logits at the retained temperature** to a maximum
absolute error of **2.22e-16** across all 400 rows — machine epsilon. The
per-row evidence is internally consistent, so the recount is a recount of the
model's own output and not of a summary of it.

The attribution ladder recounts too, from the three variant artefacts:

| Step | the one change | label agreement | flips | max abs Δp |
| --- | --- | ---: | ---: | ---: |
| N1 → N2 | laya 0.2.1 → 0.3.11 | **400/400** | 0 | **0.0** |
| N2 → N3 | budget 1024/256 → 512/192 | **400/400** | 0 | **0.0** |
| N3 → N4 | state envelope | 387/400 | **13** | 0.346653 |
| N4 → F1 | the framework itself | **400/400** | 0 | 5.02e-05 |

Recounted accuracies: N1 0.9525, N2 0.9525, N3 0.9525, N4 0.9350, F1 0.9350.
**The framework gap is carried forward, not re-derived.** Given the same input
string our classification path is exact to the half-ulp of the SDK's own
four-decimal answer (5e-05 is exactly half of the last retained decimal, and the
provider declares `probability_decimals: 4`). What it cannot do is **present**
the same input string: the shipped provider serialises
`canonical({asset, headline, body})` and requires all three non-empty, while the
author's state is `{"article": <text>}`. That is a shipped-provider limitation.
Also live and unfixed: a sequence budget hard-coded at 512/192, **below this
checkpoint's own declared 1024/256** — harmless at 232 tokens and waiting for a
longer input; and a checkpoint temperature `choice:11+ = 0.10058` that gets
clamped to 0.5 — irrelevant at four options, **relevant at seventy-seven**.

### 2.1 The correction

The train split is exactly balanced at 30,000 per class, so
`MAJORITY_CLASS_FROM_TRAIN` is a **four-way tie**, and CB01 broke it
alphabetically over the class names — selecting Business — **before any score
existed**. On the 400-row population that gives 72/400 = 0.180000, and that is
the value still carried, because a tie-break chosen after seeing the model's
score is a baseline chosen afterwards.

What the recount corrects is the other end of the range. Every legitimate break,
scored on the same 400 rows:

| tie-break | support | naive accuracy |
| --- | ---: | ---: |
| Business (alphabetical, **pinned**) | 72 | **0.180000** |
| Sci/Tech | 102 | 0.255000 |
| World | 103 | 0.257500 |
| **Sports (the strongest)** | **123** | **0.307500** |

CB03's `STRONGEST_TIE_BREAK_ACCURACY = 103 / 400` in
`tools/df_cb03_classification_receipt_20260929.py` hard-codes World, and its
return repeats it. It is wrong: Sports is the largest class in this population.
The receipt that CB03 wrote quotes the four supports correctly, in alphabetical
order — `(72/102/123/103)` — which is exactly how the largest count came to be
read off the wrong position. Every receipt written in this lane states the
corrected range in its own `limitations` field, and the corrected bounds travel
into the warehouse as the tags `paired_naive_tiebreak_range_low` 0.180000 and
`paired_naive_tiebreak_range_high` 0.307500.

### 2.2 The three facts that travel with every AG News number

Checked mechanically, not asserted:

1. **Not a held-out generalisation claim.** The benchmark's own results file
   marks that suite `in_training: true`. Stored as the tags `held_out: false` and
   `in_training: true` on every row written.
2. **The denominator is 400.** Recounted as 400 from the confusion's own totals.
   **7,600 appears in no metric value and as no denominator** anywhere in the
   receipts; it occurs only inside the prose of the `limitations` field, saying
   that the full split is a separate experiment.
3. **The naive is a four-way tie broken alphabetically before any score
   existed**, so the honest statement quotes the range 0.180000–0.307500 and
   never one end alone.

---

## 3. The receipts that were outstanding

`tools/df_cbc_receipts_20260929.py`, built from retained per-row arrays through
CB04's `app/classification_receipt.py` **unmodified** — no second contract, no
loosened check, no new schema. No model ran and no dataset was scored.

| Receipt | evidence_class | primary metric | value | receipt_sha256 | metric identity |
| --- | --- | --- | ---: | --- | --- |
| `cbc_native_accuracy` | MEASUREMENT | `classification.accuracy` | 0.9525 | `ca748b03…` | `1756f877…` |
| `cbc_native_macro_f1` | MEASUREMENT | `classification.macro_f1` | 0.9467546527629132 | `cd97d5b4…` | `a62f184c…` |
| `cbc_framework_accuracy` | MEASUREMENT | `classification.accuracy` | **0.9350** | `1d027020…` | `1756f877…` |

Two of these were genuinely outstanding:

- **a macro-F1 projection.** CB03 built the macro-F1 receipt and projected only
  the accuracy one, so the warehouse would have held one family and a reader
  could have been tempted to read the other off it. It is now its own terminal,
  under its own identity digest.
- **our framework's own score as a receipt of its own.** 0.9350 on the identical
  400 rows was measured, attributed and reported in CB03's prose, and had no
  receipt at all. That is the number a reader most needs not to lose behind the
  reproduction's 0.9525, so it now stands in the warehouse beside it, under a
  provider name that says what it is:
  `M5PHET_RUNTIME_news_signal_laya_news_sdk_0.3.11`.

**The metric contract was obeyed, not worked around.** Four refusals were run
rather than described, and all four refused by name: reading either accuracy
receipt as MAP, and comparing ACCURACY to MACRO_F1 in both directions
(`TOP_ONE_AGREEMENT` against `UNWEIGHTED_MEAN_OF_PER_CLASS_F1`). The one
comparison the contract admits — same family, same author name, same task, same
population, same denominator policy — returns our framework against the native
reproduction: left 0.9350, right 0.9525, **difference −0.0175**, both naives
0.18, population 400, independent units 400. Accuracy, macro-F1 and MAP remain
three different metrics here even where the numbers coincide.

**No quality badge was issued and none could be.** AG News is a
`PUBLIC_BENCHMARK` corpus positively known to be in the training mix, not a
`BUSINESS_HELD_OUT` measurement. **The 19-prompt router corpus appears nowhere in
this package** — not as a population, not as a denominator, not as evidence — and
the live warehouse holds **zero** terminals with `corpus_class:
ROUTER_PROMPT_CORPUS`, checked by query. Every row carries
`execution_authority: NONE` and `authorises_broker_deployment: false`.

---

## 4. From retained projection to accepted warehouse row

`tools/df_cbc_warehouse_20260929.py` goes through the **existing** route and
nothing else: register a campaign → take a governed delivery per unit → write the
terminal to the `O_EXCL` outbox → send → reconcile the campaign → **read the
terminal digest back out of the live warehouse** and compare its stored rows,
tags, costs and artifacts against what was sent. No table, column or schema was
added; no historical row was rewritten or reinterpreted; no migration was
required. The metric rows are `terminal_metrics` unmodified and the tags are
`terminal_tags` plus provenance.

Campaign `eee796e2afb0faf9b40b3b17e6cede18c442d3f8b22a9e024987042f82e99386`,
registered HTTP 201, three units, `terminal_lake: olap_cube`. Reconciliation
HTTP 200 with **no** missing units, no accounting-only and no lake-only
divergence; campaign closed; three terminals sent, **zero pending**, no failures.

Each unit took its own governed delivery of `agnews_zhang2015_test/test.parquet`
from the `sota_benchmarks` lake and the bytes were re-hashed on disk against the
pinned `71de87ec…` before any terminal was built: one `VERIFIED_TRANSFER` and two
`VERIFIED_CACHE`, all three verified.

**Acceptance, proved by reading it back:**

| Unit | terminal_sha256 (from the live warehouse) | rows sent → stored | tags | artifacts | primary value stored |
| --- | --- | ---: | --- | ---: | ---: |
| `native-accuracy` | `ca8cc67ee21e74a550c13e8c80e606bd05508c3924a0a4c3bd8e3b6a1c57bf13` | 24 → **24** | roundtrip | 5 | `classification.accuracy` **0.9525** |
| `native-macro-f1` | `9b051e353f4d408501ee8a3b2c2949647461bb1052ee0fd4952a9098261e900d` | 24 → **24** | roundtrip | 5 | `classification.macro_f1` **0.9467546527629132** |
| `framework-accuracy` | `5a0419f8ceee287d5fd91693024fdccaa55926a55d86c35b69c7b20e9b467ff9` | 24 → **24** | roundtrip | 5 | `classification.accuracy` **0.9350** |

Read through the canonical reader
(`tools/df_mod_e0_close.warehouse_terminals`): generation 1 each, status
`COMPLETED`, every metric key/split/horizon/unit/value matching to 9 decimals,
costs matching, **every** tag round-tripping, and each row's `receipt_sha256` tag
equal to the receipt it was built from. Then, independently of that report, the
contract's own published query template was run read-only against the live cube
and returns the rows:

```
framework-accuracy | M5PHET_RUNTIME_news_signal_laya_news_sdk_0.3.11 | MEASUREMENT | ACCURACY  | classification.accuracy = 0.935
native-accuracy    | NATIVE_AUTHOR_HARNESS_laya_0.2.1                | MEASUREMENT | ACCURACY  | classification.accuracy = 0.9525
native-macro-f1    | NATIVE_AUTHOR_HARNESS_laya_0.2.1                | MEASUREMENT | MACRO_F1  | classification.macro_f1 = 0.9467546527629132
```

Inventory under the contract in the live warehouse: **3 terminals, 72 metric
rows, 15 artifact rows, 1 campaign.** Every row carries `held_out: false`,
`in_training: true`, `independent_units: 400` and the corrected naive range.

Each terminal binds, as content hashes in `gov_terminal_artifact`, the five
retained files a re-analysis needs: `native_published400`, `parity_400`,
`CB03_PARITY_ATTRIBUTION`, `MANIFEST` and this lane's `cbc_receipts` bundle.

### 4.1 The one receipt that should NOT become a warehouse row, and why

CB03's `author_published_accuracy` — the author's own 0.9525, evidence class
`PUBLISHED_REFERENCE` — is retained and is **deliberately not written as metric
rows**. It shares its `metric_identity_sha256` with our ACCURACY measurement,
which is precisely what makes the two comparable, and precisely what would let
any aggregation keyed on metric identity — which is what the contract's own
query template prescribes — average a published number together with a measured
one. It is carried instead as read-only tags on the `native-accuracy` terminal
(`published_reference_value`, `published_reference_metric`,
`published_reference_source`, `published_reference_in_training`,
`published_reference_is_not_our_measurement`), where a reader can find it and an
aggregate cannot reach it.

### 4.2 A finding the write produced, for CB04's owner

The metric identity digest is a **terminal-level** tag, so it binds the
receipt's **primary** metric. A receipt's *secondary* rows therefore inherit an
identity that is not their own. Queried from the live warehouse:

```
identity 1756f877 (ACCURACY)  native-accuracy     classification.macro_f1 = 0.9467546527629132
identity a62f184c (MACRO_F1)  native-macro-f1     classification.macro_f1 = 0.9467546527629132
identity 1756f877 (ACCURACY)  framework-accuracy  classification.macro_f1 = 0.9252733932274245
```

The same macro-F1 value stands under two different identities, and one identity
covers two different macro-F1 values. Nothing here lets one metric be read as
another — the metric key and unit are correct on every row — but
`docs/CLASSIFICATION_METRIC_CONTRACT.md`'s guidance that "two rows agree only
when their identity digests agree" is **not sound for secondary rows**. Until
that is addressed, an aggregate must select on `metric` **and** on
`author_primary_metric_family`, or read primary rows only. The proper fix is a
per-row metric identity; it belongs in CB04's contract, not in a reader's
convention, and it is reported rather than patched here.

### 4.3 One property of the writer, stated so nobody assumes the other

The writer is **not idempotent**: its campaign key carries a timestamp, so a
second run would open a second campaign and add a second set of rows rather than
replacing the first. It was run **exactly once**. A correction to these rows goes
through `flush_governed_terminals.py --supersede` as the next generation of the
same campaign and unit, not through a second run of this tool.

---

## 5. What is still unmeasured

- **No new model measurement in this lane.** No classifier ran, no dataset was
  scored, nothing was downloaded. Every number here is a recount of a retained
  run or a property of the live warehouse. The reproduction was **not**
  duplicated.
- **No full 7,600-row evaluation.** Separate experiment, its own reference
  unresolved, and 7,600 never became the denominator of the 400-row score.
- **No BANKING77.** The 77-label truncation stress test — the one place the
  hard-coded 512/192 budget and the clamped `choice:11+` temperature can actually
  bite — is still not started. It is the single most informative unrun cell in
  this family.
- **No FOMC, no MASSIVE, no Financial PhraseBank measurement.** CB01 selected the
  FOMC reference and gave Financial PhraseBank no selection, with four reasons.
- **Business usefulness is not a completed family.** The 45-row corpus is sealed,
  held apart, and its items are fabricated and its labels are the author's;
  `INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT` is a named
  refusal, and **a named refusal is not a completed family**. Its use ledger is
  still at zero uses.
- **MAP has a key, a unit, an identity and a refusal, and no scorer.**
- **The provider quality badge has never been issued** — only its refusals have
  run. No badge may come from a declaration or from the 19-prompt router corpus,
  which measures the router over 19 prompts repeated five times and is not a
  classification benchmark.
- **The two framework defects are reported, not fixed.** The news-shaped
  classification entry point is the entire measured gap, and the sequence budget
  hard-coded below the checkpoint's declaration costs nothing on these 400 rows.
  Both live in a deployed component outside this lane.
- **Latency is `NOT_COMPARABLE`**: a single BLAS thread by admission policy.
- **The launcher's short-child peak undercount** (§1) is reported, not repaired.

---

## 6. Return, in the required shape

```
CB-C — classification recount, and retained projections turned into accepted warehouse rows
repo/branch/tip: predictor / satoshi/cb-classification-reconcile-20260929 / this commit
  based on 19a37baf (CB03 native reproduction, which carries 40224a3a CB04 contract),
  with 04f02555 (the dispatch) merged
files: docs/audits/work_plan/SATOSHI_CB_CLASSIFICATION_RECONCILE_2026_09_29.md
       tools/df_cbc_recount_20260929.py
       tools/df_cbc_receipts_20260929.py
       tools/df_cbc_warehouse_20260929.py
       docs/audits/evidence/cbc_reconcile_20260929/CBC_RECOUNT.json
       docs/audits/evidence/cbc_reconcile_20260929/CBC_RECEIPTS.json
       docs/audits/evidence/cbc_reconcile_20260929/CBC_WAREHOUSE.json

RETAINED PROJECTIONS vs ACCEPTED WAREHOUSE ROWS — the count first:
  before this lane: 1 retained projection, 0 accepted warehouse rows
  after  this lane: 3 retained projections, 3 ACCEPTED warehouse rows
  the live warehouse now holds 3 terminals / 72 metric rows / 15 artifact rows
  under classification_metrics.v1, in 1 campaign; it held 0 before.
  acceptance is proved by the terminal digest read back OUT of the live warehouse:
    native-accuracy     ca8cc67ee21e74a550c13e8c80e606bd05508c3924a0a4c3bd8e3b6a1c57bf13
    native-macro-f1     9b051e353f4d408501ee8a3b2c2949647461bb1052ee0fd4952a9098261e900d
    framework-accuracy  5a0419f8ceee287d5fd91693024fdccaa55926a55d86c35b69c7b20e9b467ff9
  NOT written, and why: the author's PUBLISHED_REFERENCE receipt shares its metric
    identity with our measurement, so a stored row would be reachable by an
    aggregate keyed on identity. Carried as read-only tags instead.

NO NEW MODEL MEASUREMENT. No classifier ran, no dataset was scored, nothing was
  downloaded, and the completed reproduction was NOT duplicated. This is a recount
  of retained artifacts plus a governed warehouse write.

THE RECOUNT: every retained CB03 number reproduces from the retained per-row arrays.
  accuracy 0.9525 (381 of 400), confusion identical, macro-F1 0.9467546527629132,
  ECE 0.155, Brier 0.1137, NLL 0.2801, mean confidence 0.7975, acc@50% 1.0, dropped 0;
  probabilities reconstruct from the retained logits at the retained temperature to
  2.22e-16; all ten manifest digests match; parity binds the native bytes on disk.
  Ladder: N1->N2 and N2->N3 bit-identical, N3->N4 13 flips / max |dp| 0.346653,
  N4->F1 400/400 / max |dp| 5.02e-05.

CORRECTION, claiming less: CB03 named World 0.2575 as the strongest of the four
  legitimate tie-breaks. The population's support is world 103 / SPORTS 123 /
  business 72 / sci_tech 102, so the strongest is SPORTS at 0.307500. The honest
  statement is 0.9525 against a paired naive BETWEEN 0.180000 AND 0.307500 on the
  same rows; the carried value stays the pinned 0.180000 because the tie was broken
  alphabetically before any score existed. Skill margin 0.6450, not 0.6950.

FACTS THAT TRAVEL WITH EVERY AG NEWS NUMBER, stored as tags on every row:
  held_out false / in_training true — the benchmark's own results file marks the
    suite in_training: true, so this is NOT a held-out generalisation claim;
  denominator 400 — 7,600 is in no metric value and is no denominator anywhere;
  paired naive 0.180000..0.307500, a four-way tie broken before any score existed.

THE FRAMEWORK GAP, carried forward and not re-derived: given the same input string
  our classification path is exact to the half-ulp of the SDK's own four-decimal
  answer (5e-05, probability_decimals 4). It CANNOT present the same input string:
  the shipped provider serialises {asset, headline, body} and requires all three
  non-empty. Also live and unfixed: a sequence budget hard-coded at 512/192 below
  this checkpoint's declared 1024/256, harmless at 232 tokens; and a checkpoint
  temperature choice:11+ 0.10058 clamped to 0.5, irrelevant at 4 options and
  relevant at 77.

THE METRIC CONTRACT was obeyed: four refusals RUN (read as MAP twice, compare
  ACCURACY to MACRO_F1 twice), all refusing by name; the one admitted comparison
  gives framework 0.9350 against native 0.9525, difference -0.0175, same population,
  same naive. No quality badge; the router corpus appears nowhere and the live cube
  holds zero ROUTER_PROMPT_CORPUS terminals; execution_authority NONE on every row.

resources: coordinator only, fresh aggregate admission per child, nothing displaced,
  no admission refused, no cap shrunk. Measured whole-tree cgroup peaks read from
  inside the scope before removal: recount 11,980,800 B / 2 GiB cap; receipts
  258,584,576 B / 3 GiB; warehouse-dry 39,206,912 B / 3 GiB; cube query
  35,266,560 B / 2 GiB. No GPU (the published cell is CPU and the harness hard-codes
  CPU, so an idle GPU is not a shortfall). The preferred 5090 host was NOT used and
  is NOT eligible: ~3 GiB of 14 against a 4.93 GiB unreclaimable slab; cost and
  blocker retained. No service started, stopped or restarted.

acceptance: every retained number recomputed from per-row arrays and equal to the
  retained aggregate; ten manifest digests verified; the corrected tie-break range
  in every receipt and in the stored tags; three receipts built through CB04's
  unmodified contract with four refusals exercised; delivered AG News bytes
  re-hashed against the pinned 71de87ec before any terminal was built; campaign
  registered 201, reconciled 200 with no divergence, three terminals sent and zero
  pending; and each terminal's digest, rows, tags, costs and artifacts read BACK OUT
  of the live warehouse and compared, then confirmed a second time by the contract's
  own published query run read-only against the live cube.

what is NOT done / refused / not measured:
  - no full 7,600-row evaluation; separate experiment, reference unresolved.
  - NO BANKING77. The 77-label truncation stress test is the one cell where the
    hard-coded budget and the clamped choice:11+ temperature can bite, and it is
    still not started. Most informative unrun cell in this family.
  - no FOMC, MASSIVE or Financial PhraseBank measurement.
  - business usefulness is NOT a completed family: 45 sealed rows, fabricated items,
    the author's own labels, INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_
    ENTITLEMENT named, use ledger at zero. A named refusal is not a completed family.
  - MAP has a key, a unit, an identity and a refusal, and no scorer.
  - the provider quality badge has never been issued; only its refusals have run.
  - the two framework defects are REPORTED, not fixed; they live in a deployed
    component outside this lane.
  - NEW, reported not repaired: the metric identity digest is a terminal-level tag,
    so a receipt's SECONDARY metric rows inherit an identity that is not their own.
    Proved from the live warehouse: macro-F1 0.9467546527629132 stands under two
    different identities. An aggregate must select on metric AND on
    author_primary_metric_family until CB04 gives each row its own identity.
  - NEW, reported not repaired: the launcher's retained lease records undercount a
    short child's peak by 9.5x (27,262,976 B recorded against 258,584,576 B
    measured). A retained peak is a LOWER BOUND for short children, never a
    footprint. Relevant to lane A, which is building that instrument.
  - the warehouse writer is NOT idempotent (timestamped campaign key) and was run
    exactly once; a correction goes through --supersede, not a second run.
  - latency NOT_COMPARABLE: single BLAS thread by admission policy.
  - no state-of-the-art claim, no broker deployment authority, no host name, IP,
    token or account identifier written anywhere in this package.
```

Satoshi, successor technical lead, 2026-09-29.
