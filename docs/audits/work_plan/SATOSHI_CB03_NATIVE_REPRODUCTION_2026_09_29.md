# CB03: the native AG News reproduction, then M5PHET parity

Satoshi, successor technical lead, 2026-09-29.

Additive lane. Classification ran independently, as the auditor stated it could:
neither the household lake restoration nor his triage was treated as a
prerequisite, and neither was touched. No service was started, stopped or
restarted. Nothing in this return is signed by, attributed to, or written on
behalf of Musashi.

**The one-line result.** Under the author's own recipe, on the exact governed
rows CB01 pinned, the native reproduction is **0.9525 accuracy against the
published 0.9525**, difference **0.000000**, with the paired train-derived
majority naive on those identical 400 rows at **0.180000**. Our own framework,
given the same rows, returns **0.9350** and disagrees with the reproduction on
**13 of 400 labels**. That gap is the finding of this package, and section 4
attributes it to three named causes rather than smoothing it.

---

## 1. Budget and model admission, before anything was loaded

Measured today, not assumed.

| Host | Role here | Memory available | GPU | Used for |
| --- | --- | --- | --- | --- |
| coordinator | lightweight orchestration only | 21 GiB | — | HTTP metadata reads, receipt build (3 GiB cap) |
| preferred external accelerator host | **not usable** | ~3 GiB of 14 GiB | RTX 5090 idle, 12 GiB laptop GPU idle | **nothing** |
| secondary worker | all model work | 19.71 GiB free for new work, zero pressure | RTX 4090, 16 GiB, idle, 32 °C | every child below |

The preferred host was re-probed and remains unusable for this work: about 3 GiB
of 14 GiB available, consistent with the standing ~4.38 GiB unreclaimable kernel
slab and the driver host-allocation failures since 2026-09-24. Its cold 5090 is
not availability. **Its cost and blocker are retained, and no substitute was
presented under its name.**

Every child ran through `$HOME/.local/bin/crispdm-run` at the already-deployed
revision (md5 `3779972aef0bea873fc90bff61a74fc1`, identical on the coordinator
and the secondary worker — it was **not** reinstalled). Fresh aggregate admission
per child. One request was **refused** on the slice aggregate budget
(`SLICE_AGGREGATE_BUDGET`: 15.45 G against a 14.00 G ceiling, 4.04 G in use by
another lane, 7.40 G in unrealised reservations). **The declared cap was not
shrunk to get past it.** The same 4 G request was re-queued with `-q` and
admitted when the other lane released. Nobody was displaced, and no other job
was probed, signalled or waited on.

| Child | Cap | Observed tree peak | Wall | Outcome |
| --- | --- | --- | --- | --- |
| `cb03-setup-20260929` | 4.00 GiB | 2.26 GiB | 41 s | weights + SDK installed |
| `cb03-train-smoke-20260929` | 6.00 GiB | **2.53 GiB** | 60 s | the bounded cost smoke |
| `cb03-published400-20260929` | 6.00 GiB | 2.50 GiB | 564 s | the reproduction |
| `cb03-framework-setup-20260929` | 4.00 GiB | 1.83 GiB | — | refused once, then queued and admitted |
| `cb03-framework-smoke-20260929` | 6.00 GiB | 1.94 GiB | 15 s | 8-row framework smoke |
| `cb03-m5phet-parity-400-20260929` | 6.00 GiB | 1.93 GiB | 313 s | the parity run |
| `cb03-parity-attribution-20260929` | 6.00 GiB | 2.55 GiB | ~1,900 s | the three attribution variants |
| `cb03-receipts-20260929` (coordinator) | 3.00 GiB | 0.03 GiB | 21 s | the receipts |
| `cb03attrcmp` (coordinator) | 2.00 GiB | 0.01 GiB | 2 s | the attribution comparison |

Every peak above is a whole-tree cgroup peak read by the launcher's sampler over
a child that outlived the sampling interval, so each is a measurement. **No peak
is null, and none of these is a floor**; the 8-row framework smoke is the one
child short enough that its 1.94 GiB should be read as a lower bound, and it is
superseded by the 400-row run's 1.93 GiB over 313 s.

**No GPU was used, and that is a finding rather than a shortfall.** The published
cell is `device: cpu` and the author's harness hard-codes `laya.load(..., device="cpu")`.
Reproducing a CPU cell on a GPU would have been a different experiment. The idle
RTX 4090 was therefore left idle; no device was asserted because no child opened
one, and `CUDA_VISIBLE_DEVICES=""` was set on every child so none could.

### 1.1 Installed pinned weights: identity verified, not assumed

The order said to reuse installed pinned weights **where they are identical**.
They are not installed. Both workers carry
`~/.cache/huggingface/hub/models--convaiinnovations--laya` containing exactly one
file — `refs/main`, holding `55cf4c4e…` — and **no blobs and no snapshots**
(12 KiB and 244 KiB of xet metadata). There was nothing to reuse, so the
checkpoint was downloaded and its identity established by digest:

- `convaiinnovations/laya-typed-decisions` at revision
  `1a793eb568e6718f15941d08f85432581df534e3`, `model.safetensors`
  842,609,220 bytes, sha256
  `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`, re-hashed
  on disk after download and **re-hashed again inside every child** before it was
  read as weights.
- The same weights are also carried inside the family repo
  `convaiinnovations/laya` as `typed-decisions/model.safetensors`, with the
  **identical** LFS sha256. So the two distribution paths are byte-identical, and
  that was checked rather than assumed.
- Apache-2.0, ungated, 842 MB of fp16. **No unofficial quantization, and no
  smaller model under this model's name.** The gated public FOMC checkpoint stays
  un-requested; nothing in this package touched it.

### 1.2 The exact checkpoint behind the published row

The author's script pins **no** checkpoint revision, so "the exact checkpoint"
had to be established from the distributor's history rather than from the paper:

| Revision | Date | `model.safetensors` sha256 | `encoder/config.json` | `rl_agent_config.json` | `tokenizer/*` |
| --- | --- | --- | --- | --- | --- |
| `843893f9` | 2026-09-18 17:55 | `4fa56de7…` | `d4be4829` | `3f8a5cfc` | `2f4d8583` / `ed1ffabc` |
| `f9ab0b22` | 2026-09-19 09:55 | `4fa56de7…` | `d4be4829` | `5f0e1d5f` | `2f4d8583` / `ed1ffabc` |
| `dd079950` | 2026-09-23 08:09 | `4fa56de7…` | `d4be4829` | `5f0e1d5f` | `2f4d8583` / `ed1ffabc` |
| **`1a793eb5`** (used here) | 2026-09-24 04:08 | **`4fa56de7…`** | `d4be4829` | `5f0e1d5f` | `2f4d8583` / `ed1ffabc` |

The published run is stamped **2026-09-19 11:57:16**. The revision live at that
moment was `f9ab0b22`, and every file that affects inference — weights, encoder
config, RL/temperature config, both tokenizer files — is **byte-identical**
between `f9ab0b22` and the `1a793eb5` we used. Only `README.md` differs across
those four revisions. **The checkpoint is identified, not approximated.**

### 1.3 The exact evaluator and the exact rows

- **Evaluator**: the author's own `research/scripts/bench_local.py`, imported
  unmodified, sha256 `08862fb4dba102db873e3e3ad428dedf92c12832b4e8ef71b35e433cce4eea22`,
  and the `jev.ag_news` case construction copied verbatim from their
  `research/scripts/bench_apps.py`, sha256
  `b25e7fb1228fb7d4d51d121ea08b3c3471b9c9af67d5a40fe415a2f33983e960` — both at
  `NandhaKishorM/laya` commit `ee760389dc69e28c66893b717fa87c84c0b6063a`
  (2026-09-20), the **earliest committed revision that carries those scripts**.
  Both are re-hashed inside the child and refused if they are not those bytes.
  `score_cases`, `softmax_t`, `temp_for`, `metrics`, `macro_f1`, `to_internal`
  and `load` are byte-identical between that revision and today's `main`; the
  **only** function that differs in the whole harness is `ece_score` (the first
  bin's left edge), so both conventions are computed and both are reported.
- **SDK**: `laya==0.2.1` from PyPI, the exact version string the published
  results file records (`"laya": "0.2.1"`), resolved from site-packages and
  recorded in the artefact as `laya_module`.
- **Rows**: the governed CB02 delivery, not a fresh fetch. `agnews_zhang2015_test/test.parquet`,
  sha256 `71de87ec…`, re-hashed in the child; the first 400 rows in file order;
  recomputed population digest
  `b4c5f991060bcefcc69fac9339b32086dcd0674ac15e44f41eeeeb7c9e782324` — **equal to
  the CB01 pin**, and the run refuses outright if it is not.

That last check is the whole point of the one deliberate substitution in the
recipe. The author calls `load_dataset("fancyzhx/ag_news", split="test")`, an
unpinned network fetch; we read the registered distributor parquet instead and
then **prove** the substitution changed no row and no order by reproducing the
digest CB01 pinned before any score existed.

### 1.4 Every label option fits untruncated

Checked per row, not inferred from an aggregate:

| | value |
| --- | --- |
| rows | 400 |
| options expected per row | 4 |
| rows where all 4 option markers were built | **400** |
| rows truncated in their options | **0** (empty list) |
| longest built sequence | 232 tokens |
| checkpoint's own budget | `max_len` 1024, `head_max_len` 256 |
| harness `dropped` | 0 |

Four options against a ~20-option budget, and the longest sequence used under a
quarter of the window. **No truncation check can fail on AG News**, which is
exactly why CB01 records BANKING77 as the real stress test: the same benchmark
reports 0.425 for two different checkpoints on 77 labels and attributes it to the
shared option-token budget. That test is not in this package.

---

## 2. The bounded train/dev-only cost smoke, before any evaluation

Run **before** the published population was opened, on the official **train**
split only (`agnews_zhang2015_train/train.parquet`, sha256 `fc508d6d…`, re-hashed
in the child), first 32 rows in file order.

| | value |
| --- | --- |
| population | `train_smoke`, 32 rows of the official train split |
| observed tree peak | **2.53 GiB** against a 6.00 GiB cap |
| wall | 60 s, 1,862 ms/case |
| option fit | 32/32 rows with all 4 options, longest 150 tokens |
| dropped | 0 |

The smoke is a **cost** measurement and nothing else. Its 0.9062 accuracy is on
32 train rows that happen to be a single class in file order; it is not a quality
claim and is not carried into any receipt.

What it bought: the 2.53 GiB peak sized the 6 GiB cap for the real run, and the
1,862 ms/case rate predicted the 400-row cost at about 12 minutes, which is what
it took. It also established that latency here is **not comparable** to the
author's: `crispdm-run` pins BLAS to a single thread, so our 1,410 ms/case
against the published 103.6 ms/case is an admission-policy artefact. Accuracy is
unaffected — the argmax of four well-separated logits does not move with thread
count — and latency is nowhere claimed as reproduced.

---

## 3. The native result, beside the published value and its same-row naive

Population: `agnews_test_first400_laya_published`, 400 rows, digest
`b4c5f991…`. Supervision regime: zero-shot prompted; no labelled row fitted any
head.

| Metric (author's own name) | Published | **Ours, native** | Difference | Paired naive, same 400 rows |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.9525 | **0.9525** | **0.000000** | **0.180000** (majority from train) |
| macro-F1 | 0.9468 | **0.9468** | 0.000000 | 0.076271 |
| ECE (15 bins) | 0.155 | **0.155** | 0.000000 | — |
| Brier | 0.1137 | **0.1137** | 0.000000 | — |
| NLL | 0.2801 | **0.2801** | 0.000000 | — |
| mean confidence | 0.7975 | **0.7975** | 0.000000 | — |
| accuracy at 50 % coverage | 1.0 | **1.0** | 0.000000 | — |
| dropped | 0 | **0** | — | — |
| ms/case | 103.6 | 1410.5 | **NOT_COMPARABLE** | — |

Every published metric of that cell reproduces at the published precision. The
confusion, reference rows by predicted columns in the author's option order
`world / sports / business / sci_tech`, plus an ABSTAINED column that is
structurally zero:

| reference \ predicted | world | sports | business | sci_tech | ABSTAINED | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| world | **99** | 3 | 1 | 0 | 0 | 103 |
| sports | 2 | **121** | 0 | 0 | 0 | 123 |
| business | 1 | 0 | **62** | 9 | 0 | 72 |
| sci_tech | 0 | 0 | 3 | **99** | 0 | 102 |

381 correct of 400 = 0.9525 exactly. The ECE figure is identical under both
committed bin conventions, so the one function that differs between the harness
revisions does not move this cell.

### 3.1 Three things that travel with that number and are not optional

1. **It is not a held-out score.** The author's own results file marks this suite
   `in_training: true` for all three checkpoints. No generalisation claim follows
   from 0.9525. (`banking77` in the same file is marked `in_training: false`.)
2. **The population is 400 rows and is not balanced** — 103 world / 123 sports /
   72 business / 102 sci_tech. The paired naive is therefore **0.180000**, not the
   full split's 0.250000. The full 7,600-row official test split is a **separate
   experiment**; its 0.250000 baseline and its denominator are attached to
   nothing here, and no 7,600-row score exists in this package.
3. **The pinned naive is the weakest of four legitimate tie-breaks, and that is
   recorded rather than quietly banked.** The train split is exactly balanced at
   30,000 per class, so "the majority class" is a four-way tie. CB01 broke it
   alphabetically before any score existed, which selects Business and gives
   72/400 = 0.180000. Breaking it toward World instead gives 103/400 = 0.2575.
   The honest skill statement is therefore **0.9525 against a naive between
   0.180000 and 0.2575 on the same rows**, and the receipt says so in its own
   `limitations` field. Choosing the tie-break after seeing the model score would
   be choosing a baseline afterwards, so the pinned 0.180000 is the one carried.

Also recorded: the author's `seed = 13` is **not** used for AG News. The
population is `list(d)[:400]`, deterministic file order, which is why the rows
were recoverable at all. There is no sampling temperature to match — the model is
non-autoregressive and emits its distribution in one forward pass, and every
"temperature" in its documentation is a per-(question type, option count)
calibration temperature read from the checkpoint's own config.

---

## 4. Then parity: the same rows through our framework

The native reference was **independently invoked** — its own process, its own
virtual environment, its own SDK version, under the author's harness — and its
artefact is read by the parity run as data. Nothing in the parity path recomputes
it, and nothing there can move it. The parity run refuses to start unless the
artefact it was handed carries the pinned 400-row population digest.

"Through M5PHET" means through `m5phet.runtime.run`, with the distribution's own
`laya_news` provider registered, so the runtime's capability check, fitted-state
binding, output-schema contract and classification payload validation all
executed. The question was built by the framework's own `ad_hoc_task`, and the
run refuses if the framework rebuilt the author's instruction string, option keys
or option order into anything else — it did not.

### 4.1 The headline parity result

| | value |
| --- | --- |
| rows | 400, digest `b4c5f991…` (identical to the reproduction) |
| framework answered | 400 |
| framework refused | 0 |
| **population coverage parity** | **PRESERVED** — 400/400 both sides, coverage 1.0 |
| **label parity** | **NOT PRESERVED** — 387/400 agree, 0.967500 |
| label disagreements | **13** |
| **probability parity** | **NOT PRESERVED** — max absolute difference **0.346685** |
| framework accuracy on answered | **0.9350** |
| native accuracy, same rows | 0.9525 |

**Our framework did not reproduce the published row.** It is 0.0175 below the
native reproduction on identical rows — 7 additional net errors out of 400 — and
its probabilities differ from the reference by up to 0.347 on a single option.
Abstention is the one thing that is identical: neither path refused any row, and
neither dropped any, so coverage parity holds at 400/400.

§4.2 shows the gap has a single cause, and it is not the wrapper: given the same
input string, our classification path agrees with the native reference on 400 of
400 labels to the SDK's own reported precision. What it cannot do is present the
same input string, because its classification entry point requires a news event.

### 4.2 Why, one cause at a time

Three things the framework imposes that the author's recipe does not. Each was
isolated by re-running the native recipe with exactly that one change, so no step
below carries more than one cause:

| Rung | SDK | budget | state envelope | accuracy | macro-F1 |
| --- | --- | --- | --- | ---: | ---: |
| N1 — the reproduction | laya 0.2.1 | 1024 / 256 | `{"article": …}` | **0.9525** | 0.9468 |
| N2 | laya 0.3.11 | 1024 / 256 | `{"article": …}` | 0.9525 | 0.9468 |
| N3 | laya 0.3.11 | **512 / 192** | `{"article": …}` | 0.9525 | 0.9468 |
| N4 | laya 0.3.11 | 512 / 192 | **`{asset, body, headline}`** | **0.9350** | 0.9253 |
| F1 — our framework | laya 0.3.11 | 512 / 192 | `{asset, body, headline}` | **0.9350** | — |

Each step, with its one change:

| Step | the one change | label agreement | label flips | max abs Δp | identical |
| --- | --- | ---: | ---: | ---: | --- |
| N1 → N2 | the SDK version the framework pins, 0.2.1 → 0.3.11 | **400/400** | 0 | **0.0** | **YES, bit-identical** |
| N2 → N3 | the budget the framework hard-codes, 1024/256 → 512/192 | **400/400** | 0 | **0.0** | **YES, bit-identical** |
| N3 → N4 | the state envelope the framework serialises | 387/400 | **13** | **0.346653** | no |
| N4 → F1 | the framework itself: provider, runtime, contract, per-row calls | **400/400** | 0 | 5.02e-05 | to the declared precision |

**The whole gap is the state envelope, and nothing else.** Two of the three
suspects are exonerated by measurement rather than by argument: the pinned SDK
version changes nothing on these rows, and neither does the hard-coded sequence
budget — both are bit-identical, 400 labels and every probability. The 13 label
flips and the 0.0175 accuracy loss appear at exactly one rung, the one where the
article stops being `{"article": <text>}` and becomes
`{"asset":"AGNEWS","body":<text>,"headline":"AG News item"}`.

A caveat kept rather than dropped: N2 → N3 being bit-identical is a statement
about **these** 400 rows, whose longest sequence is 232 tokens. A 512-token
budget is not equivalent to a 1024-token one in general — it is equivalent here
because nothing reached either limit. The hard-coded budget remains a defect
waiting for a longer input, not a non-issue.

One thing the newer SDK says out loud that the published run did not:
laya 0.3.11 warns at load that *this checkpoint ships a temperature outside
[0.5, 5]* — `choice:11+ = 0.10058…`, clamped to 0.5. That bucket is for
eleven-or-more-option questions, so it cannot touch a four-option AG News row,
which is consistent with N1 → N2 being bit-identical. It will touch BANKING77's
77 options, and it is recorded here for that reason.

The three imposed differences, named:

1. **The SDK version is pinned to a different one.**
   `news_signal.backends.SDK_COMMIT` is `1e28ac20…` (laya 0.3.11) and the backend
   **refuses** any install that is not that VCS commit
   (`PINNED_SDK_INSTALL_REQUIRED`, read from `direct_url.json`). The published
   cell was produced under laya 0.2.1. The framework cannot be asked for 0.2.1
   without editing it.
2. **The sequence budget is hard-coded below the checkpoint's own.**
   `LayaBackend.cfg` is `{"max_len": 512, "head_max_len": 192}`, while this
   checkpoint's `rl_agent_config.json` declares 1024 and 256. The framework gives
   the model a smaller window than the model declares. On AG News nothing is
   truncated either way — the longest sequence is 232 tokens — but the head and
   option budgets that `build_sequence` derives from `head_max_len` are not the
   same, so the encoded sequence is not the same sequence.
3. **The classification entry point is news-shaped, so the input cannot be
   identical.** `provider.infer` serialises `canonical({asset, headline, body})`
   as the state and requires all three to be non-empty strings. The author's state
   is `{"article": <text>}`. There is no configuration of the shipped provider
   that presents the author's string. For this run `asset` was `AGNEWS` and
   `headline` was `AG News item`, both declared in the artefact; the article text
   was the `body`. **This is a framework limitation, recorded as one** — it is
   not a knob that was set wrongly.

N4 against F1 is the only comparison that holds all three constant, so it is the
only one that measures the wrapper's own fidelity.

**The wrapper is faithful.** N4 → F1: 400 of 400 labels identical, and the
largest probability difference is 5.02e-05. That number is not noise and it is
not a defect — it is the SDK's own answer-object precision. `Agent.system_one`,
which is the call the provider makes, returns
`{"probabilities": {key: round(float(v), 4)}}`, while the benchmark's
`score_cases` returns raw logits that the harness softmaxes at full precision.
Half of the last retained decimal is 5e-05, which is exactly the bound observed.
The provider declares this as `probability_decimals: 4` in every payload and does
not re-round, rescale or renormalise on top of it. So **our framework's
classification path reproduces the native reference exactly, to the precision the
SDK itself reports** — once it is given the same input string.

### 4.3 What the disagreements look like

The 13 flips are not all near-ties, and averaging them would have hidden that.
Measured as the reference run's own top-two margin on each disagreeing row:

| | value |
| --- | --- |
| disagreements | 13 |
| smallest reference top-2 margin | 0.042208 |
| median | 0.190768 |
| largest | 0.403827 |

So some flips are coin-flips the envelope tipped, and at least one is a row where
the reference was ahead by 0.40 and the framework still answered differently.
Both matter: the first says the population contains rows this checkpoint is
genuinely unsure about, and the second says the envelope change is not a small
perturbation.

Direction of the loss: 0.9525 → 0.9350 is 381 correct → 374 correct, so the
envelope costs 7 net correct answers out of 400 while moving 13 labels. It is a
loss and not a wash, but it is not a uniform degradation either.

---

## 5. The metric contract, used rather than worked around

The results became classification receipts on the **existing** path:
`app/classification_receipt.py`, CB04's contract, unmodified. No second contract,
no loosened check, no new schema.

Three receipts, deliberately three:

| Receipt | evidence_class | primary metric | value | receipt_sha256 |
| --- | --- | --- | ---: | --- |
| ours, accuracy | `MEASUREMENT` | `classification.accuracy` | 0.9525 | `3cc60452…` |
| ours, macro-F1 | `MEASUREMENT` | `classification.macro_f1` | 0.946755 | `252d1726…` |
| the author's published value | `PUBLISHED_REFERENCE` | `classification.accuracy` | 0.9525 | `ae0ba9a5…` |

Each carries, as its own field and none recomputed at read time: the author's
primary metric under the author's own name; the paired train-derived majority
naive of the **same family**, scored on exactly the same rows, with its own train
population digest (re-hashed here from the delivered 120,000-row train parquet
and refused if it equalled the evaluation digest); the ordered class vocabulary
with its `label_order_sha256`; the 4×4 per-class confusion plus its ABSTAINED
column, closing over the population; the probability semantics with a separate
`calibrated` boolean; abstention coverage with its denominator policy; and the
calibration split.

Four contract decisions worth stating, because each was a refusal the contract
issued and not a preference we expressed:

- **Probability semantics are declared `SOFTMAX_POSTERIOR_UNCALIBRATED` with
  `calibrated: false`**, although the recipe *does* apply the checkpoint's own
  per-bucket temperatures. The author does not publish the split those
  temperatures were fitted on, and the contract refuses `calibrated: true`
  against a `NONE` calibration split. ECE is carried as measured, with its bin
  count, and is not evidence of calibration.
- **Macro-F1 is carried at full precision, 0.9467546527629132.** The author's
  harness rounds every metric to four decimals; the contract refuses a carried
  metric that disagrees with its own confusion by more than 1e-6, and 0.9468
  does. The rounding is recorded in the receipt rather than absorbed by widening
  the check.
- **The published receipt declares what is a declaration.** Its `declared_fields`
  names `author_primary_metric`. The author publishes no per-class confusion, no
  per-row predictions and no population digest for this cell, so the confusion,
  abstention, population and probability metrics in that receipt are **ours**,
  present because the contract requires every field, and its `limitations` says
  so.
- **The comparison went through the contract, not around it.**
  `compare(ours, published)` returned family `ACCURACY`, unit
  `accuracy_fraction`, left 0.9525, right 0.9525, **difference 0.0**, both naives
  0.18, population 400, independent units 400 — accepted because the metric
  identity, task, population and denominator policy all agree.

And two refusals were **run**, not described:

```
read_metric(ours_accuracy, "MAP")
  -> this receipt carries ACCURACY (the author calls it 'accuracy') and does not
     carry MAP; ACCURACY and MAP are different metrics and neither substitutes
     for the other

compare(ours_accuracy, ours_macro_f1)
  -> ACCURACY and MACRO_F1 are different metrics: TOP_ONE_AGREEMENT against
     UNWEIGHTED_MEAN_OF_PER_CLASS_F1. Both lie in a shared numeric interval and
     neither is a version of the other, so no difference, ratio or ranking
     between ACCURACY and MACRO_F1 is defined here
```

The three metric identity digests are distinct where the metrics are distinct and
equal where they are the same thing measured by two parties:
`ours_accuracy` and `author_published_accuracy` share
`1756f877…`, and `ours_macro_f1` is `a62f184c…`. Accuracy, macro-F1 and MAP are
not interchangeable here even when the numbers coincide.

**No quality badge was issued, and none could be.** Nothing in this package is a
`BUSINESS_HELD_OUT` measurement; AG News is a `PUBLIC_BENCHMARK` corpus and is
positively known to be in the training mix. **The 19-prompt router corpus appears
nowhere in this package** — not as a population, not as a denominator, not as
evidence. No number here comes from it, and no number here carries execution
authority: every receipt reads `authorises_broker_deployment: false`,
`execution_authority: NONE`.

---

## 6. Return, in the required shape

```
CB03 — native reproduction, then M5PHET parity
repo/branch/tip: predictor / satoshi/cb03-native-reproduction-20260929 / CB03_TIP
files: docs/audits/work_plan/SATOSHI_CB03_NATIVE_REPRODUCTION_2026_09_29.md
       tools/df_cb03_native_agnews_20260929.py
       tools/df_cb03_m5phet_parity_20260929.py
       tools/df_cb03_parity_attribution_20260929.py
       tools/df_cb03_classification_receipt_20260929.py
       docs/audits/evidence/cb03_20260929/MANIFEST.json
       docs/audits/evidence/cb03_20260929/native_published400.json
       docs/audits/evidence/cb03_20260929/smoke_train32.json
       docs/audits/evidence/cb03_20260929/parity_400.json
       docs/audits/evidence/cb03_20260929/parity_smoke8.json
       docs/audits/evidence/cb03_20260929/v_sdk0311_ckptbudget_author.json
       docs/audits/evidence/cb03_20260929/v_sdk0311_fwbudget_author.json
       docs/audits/evidence/cb03_20260929/v_sdk0311_fwbudget_fwenvelope.json
       docs/audits/evidence/cb03_20260929/CB03_PARITY_ATTRIBUTION.json
       docs/audits/evidence/cb03_20260929/CB03_CLASSIFICATION_RECEIPTS.json
       docs/audits/evidence/cb03_20260929/cb03_checkpoint_manifest.json
  Every run's artefact carries the per-row gold, prediction, all four
  probabilities, the raw logits and the temperature actually applied, uncompressed
  and row by row, so an independent analysis does not depend on our aggregates.

NATIVE RESULT, beside the published value and its same-row naive:
  AG News topic, laya-typed-decisions @1a793eb5 (weights 4fa56de7…),
  population agnews_test_first400_laya_published (400 rows, b4c5f991…)
    published  accuracy 0.9525  macro-F1 0.9468  (author's own results file,
               jev.ag_news/typed-decisions, marked in_training: true)
    ours       accuracy 0.9525  macro-F1 0.9468   difference 0.000000
    same-row paired naive, majority from train: 0.180000 accuracy
               (0.2575 under the strongest of four legitimate tie-breaks)
    every other published metric of that cell also reproduces: ECE 0.155,
    Brier 0.1137, NLL 0.2801, mean confidence 0.7975, acc@50% 1.0, dropped 0
  NOT a held-out generalisation claim: the benchmark's own results file marks
  this suite in_training: true.

DID OUR FRAMEWORK REPRODUCE IT EXACTLY? NO.
  M5PHET + news-signal laya_news, same 400 rows, same digest:
    accuracy 0.9350 against the native 0.9525
    label parity     387/400 = 0.967500      NOT PRESERVED (13 disagreements)
    probability parity  max |Δp| 0.346685    NOT PRESERVED
    abstention parity   0 refusals both sides    PRESERVED
    population coverage 400/400 both sides       PRESERVED
  WHERE it did not, attributed one cause at a time — two suspects exonerated by
  measurement and one cause isolated:
    laya 0.2.1 -> 0.3.11 (the SDK the framework pins)   400/400, max |dp| 0.0
        BIT-IDENTICAL, not the cause
    budget 1024/256 -> 512/192 (hard-coded in LayaBackend.cfg, below the
        checkpoint's own declared budget)              400/400, max |dp| 0.0
        BIT-IDENTICAL on these rows (longest sequence 232 tokens), not the cause
        here — still a defect for any longer input
    state envelope {"article": text} -> {asset, body, headline}
                                                   387/400, max |dp| 0.346653
        THE ENTIRE GAP. news_signal.provider.infer serialises
        canonical({asset, headline, body}) and requires all three non-empty, so
        the shipped provider CANNOT present the author's input string.
    the wrapper itself (N4 -> F1)               400/400, max |dp| 5.02e-05
        FAITHFUL to the SDK's own 4-decimal answer-object precision, which the
        provider declares as probability_decimals: 4 and does not re-round.
  disagreement profile: reference top-2 margin min 0.042208, median 0.190768,
    max 0.403827 — not all near-ties.
  also recorded: laya 0.3.11 warns this checkpoint ships choice:11+ temperature
    0.10058, clamped to 0.5. Irrelevant to 4 options, relevant to BANKING77's 77.

acceptance: weights identity verified by sha256 against the distributor and
  re-hashed inside every child; the checkpoint behind the published row
  identified by revision-by-revision digest comparison (inference-relevant files
  byte-identical to the revision live at the published run's timestamp); author
  harness re-hashed and refused unless byte-identical; governed AG News bytes
  re-hashed in every child; recomputed population digest equal to the CB01 pin or
  the run refuses; 400/400 rows carry all 4 options untruncated, longest sequence
  232 of 1024 tokens; train/dev-only cost smoke before any evaluation
  (peak 2.53 GiB, 60 s); receipts built by CB04's own contract with two refusals
  exercised and compare() returning difference 0.0 against the published value.

what is NOT done / refused / not measured:
  - no full 7,600-row test evaluation. It is a separate experiment with its own
    still-unresolved reference, and 7,600 never became the denominator of the
    400-row published score.
  - no BANKING77, no FOMC, no MASSIVE, no Financial PhraseBank measurement. The
    77-label truncation stress test is NOT in this package.
  - no GPU work. The published cell is device: cpu and the author's harness
    hard-codes CPU, so the idle RTX 4090 was left idle and no device was asserted.
  - the preferred external accelerator host was NOT used: ~3 GiB available
    against its unreclaimable slab. Cost and blocker retained; no substitute
    under its name.
  - one admission REFUSED on the slice aggregate budget; the declared cap was NOT
    shrunk, the request was queued and admitted later. Nothing displaced.
  - no unofficial quantization, no smaller model under the original name, no
    shortened option list, no gated checkpoint requested.
  - latency NOT_COMPARABLE: single BLAS thread by admission policy.
  - no quality badge, no BUSINESS_HELD_OUT measurement, no router-corpus number,
    no state-of-the-art claim, no broker deployment authority.
  - the three framework defects in §4.2 are REPORTED, not fixed: fixing the
    shipped provider is a change to a deployed component outside this lane.
```

Satoshi, successor technical lead, 2026-09-29.
