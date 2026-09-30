# BANKING77: the corrected scope, and the native reference path pinned to its one blocker

Satoshi, successor technical lead. 2026-09-29.
Order: the classification lane of the 2026-09-29 dispatch — correct the blanket
impossibility sentence beside the retained fit evidence, then continue the already
selected **native** BANKING77 reference path under its existing budget or state its exact
unfulfilled dependency. No unapproved download or training allocation is implied by that
order and none was taken.

Worktree `.worktrees/predictor-b77-native-20260929`, branch
`satoshi/banking77-native-path-20260929`, from `6a032341` (this lane's own tip; the
reference selections at `07846a7d` are its ancestor).

---

## 0. The correction first, because it is mine as much as the lane's

**As published**, `BANKING77_SEVENTY_SEVEN_OPTIONS_DO_NOT_FIT_THE_PROVIDER_NO_SCORE_PRODUCED`
reads as though the 77 labels cannot fit at any budget. **What was established is
narrower, and this is the sentence that stands:**

> **BANKING77 is unsupported by the SHIPPED one-shot wrapper and its serializations: the
> question builder caps options at 12, and the option payloads cost 706 and 894 tokens
> against tested head budgets of 192 and 256. That is not impossibility for any budget,
> provider or model.**

The correction sits **beside** the measurement, never in place of it. The retained fit
table is cited unchanged and was not edited:

| rendering | budget | option tokens | option budget | fits |
| --- | --- | ---: | ---: | --- |
| empty description (floor) | framework 512 / **192** | 706 | −514 | false |
| empty description (floor) | checkpoint 1024 / **256** | 706 | −450 | false |
| label as its own description | framework 512 / **192** | 894 | −702 | false |
| label as its own description | checkpoint 1024 / **256** | 894 | −638 | false |

`MAX_OPTIONS = 12` in the shipped question builder, so 13 and 77 options are refused
before any encoder is reached. Evidence, byte-identical to the one CB04 published:
`docs/audits/evidence/cb04_row_identity_20260929/BANKING77_LABEL_FIT.json`. The scope
correction is machine-readable next to it as
`BANKING77_LABEL_FIT_SCOPE_CORRECTION.json`, and §9 of the CB04 return now carries the
same paragraph without one earlier sentence being altered.

Two consequences, honoured throughout this return:

- **A larger budget is a different configuration.** Raising the context and printing the
  result against the published row as though nothing had changed is not allowed, and is
  not done anywhere here.
- **A hierarchy or a retrieval shortlist is a distinct method.** If one is ever used, its
  score must include **its routing errors over the full population**; otherwise it reports
  the accuracy of the easy subset. No shortlist is used in this return.

And the reason the correction matters operationally: **the shipped wrapper is one adapter,
not the task.** Nothing in classification is blocked on its 12-option interface. The
native route is the primary one and it is what the rest of this return advances.

---

## 1. The answer to the order's actual question

**The native path is now pinned in every part except the encoder, and its one unfulfilled
dependency is an acquisition, not a measurement.**

> **`BANKING77_NATIVE_PATH_BLOCKED_ON_MODEL_AND_ENVIRONMENT_ACQUISITION`** — an approved
> acquisition of **1,369,721,378 bytes** of model artifacts at the pinned revision, plus a
> **third** pinned Python 3.13 environment carrying `mteb`, `sentence-transformers`,
> `peft`, `datasets` and `scikit-learn` (**16,228,123 bytes** of direct wheels at tier 1,
> transitive closure unresolved), together with an explicit decision to execute the
> repository's remote code.

**No score exists and none is fabricated.** What did advance is everything that does not
need the encoder: the published row is now read from **its own artifact** instead of
transcribed, the population is re-hashed from governed bytes, the evaluator's ten
training draws are reproduced **offline** with a digest each, and three metric identities
in the published file are explained rather than noticed.

Evidence: `docs/audits/evidence/banking77_native_path_20260929/NATIVE_PATH.json`
(sha256 `e8044a41…`), produced by `tools/df_banking77_native_path_20260929.py`.

### The published value beside its same-row naive

| what | value | rows | evidence class |
| --- | ---: | ---: | --- |
| `jinaai/jina-embeddings-v5-text-small` @ `46ed7da5…`, MTEB `Banking77Classification` accuracy | **0.914578** | 3,080 | `PUBLISHED_REFERENCE` |
| its own ten per-experiment cells, mean recomputed here | **0.914578** | 3,080 | recomputed from the artifact |
| spread across the ten experiments | 0.910390 – 0.916883 (sd 0.001714) | 3,080 | recomputed |
| macro-F1 reported in the same file | 0.913809 | 3,080 | `PUBLISHED_REFERENCE` |
| **same-row naive**: train-fitted majority on the identical governed test rows | **0.012987** | 3,080 | pinned before any score existed |
| **our native measurement** | **NO_NEW_MEASUREMENT** | — | blocked on the dependency above |

The naive is exactly 1/77 = 40/3080, and that is not a coincidence — see §3.

---

## 2. The published row, read from its own artifact

CB01 carried `0.914578` as a number in a table. It is now an artifact.

`results/jinaai__jina-embeddings-v5-text-small/46ed7da5…/Banking77Classification.json`
from `embeddings-benchmark/results`, 4,004 bytes, sha256
`ba162c90aa25977f57d0cf25dc10199d9b0d6ecf429238415faa5f065b800fa9`, retained in this
lane's evidence directory. From its own contents:

- `dataset_revision` **`0fd18e25…`** — equal to the pin the population contract recorded
  before any score existed.
- `mteb_version` **`2.3.11`**. CB01 said "mteb 2.x"; it is now exact.
- ten `scores_per_experiment` cells whose mean is **0.914578**, reproducing the published
  aggregate from its own cells — the same standard the FOMC target was held to.
- `evaluation_time` 112.94 s, and **no device recorded anywhere in the file.**

That last point is a real gap and is carried as one: a CPU reproduction in bfloat16 is a
different device from whatever produced a 113-second run, and its numerics need not agree.
The device will be **declared**, never blended.

---

## 3. Why three identities in the published file are forced, not lucky

The published cell reports `f1 == f1_weighted`, `precision == precision_weighted` and
`recall == accuracy`, identically, in **all ten** experiments and in the aggregate. That
is not a rounding artefact and not a bug.

Re-hashed from the governed CB02 delivery in this process
(`d12d6e3b…`, 3,080 rows, 77 labels): **every class has exactly 40 test rows.** The test
split is exactly balanced. With equal support a weighted mean *is* the unweighted mean, and
accuracy is the support-weighted mean of per-class recall, so all three identities are
forced. The suite proves the implication both ways — it holds on a synthetic balanced
population and **fails** on an unbalanced one, so the check measures something.

Two conclusions, and the second is the one the contract needs:

1. The published `f1 = 0.913809` is **macro-F1 by the pinned source's own definition**
   (`f1_score(..., average="macro")`), not by this coincidence. It is therefore an
   admissible `MACRO_F1` comparison target.
2. `MACRO_F1` and `WEIGHTED_F1` remain **different declared families**, and a comparison
   between them still refuses by name *even here, where the two numbers are equal*.
   Accuracy against macro-F1 refuses likewise. Both refusals are asserted as tests.

And it corroborates population identity: the published artifact shows exactly the three
identities our registered rows predict.

---

## 4. The evaluator, pinned at the version that produced the row

Read from source at tag `2.3.11` and recorded by digest — `mteb/abstasks/classification.py`
`93fc2b33…`, `mteb/_evaluators/sklearn_evaluator.py` `7fd4a005…`,
`mteb/tasks/classification/eng/banking77_classification.py` `6b7f4d1f…`:

`AbsTaskClassification` with `LogisticRegression(n_jobs=-1, max_iter=100)` and
`random_state` set to the task seed, `samples_per_label = 8`, `n_experiments = 10`,
`train_split = "train"`, seed **42**, `main_score = "accuracy"`. Subsampled few-shot
**linear probing** — not zero-shot prompting, not full-train fine-tuning. Labels train the
head, so calling it zero-shot would be wrong.

Two things the task file says that were not known before:

- the task carries `superseded_by = "Banking77Classification.v2"`, a **different** dataset
  revision that "corrects errors found in the original data". Reproducing the published row
  therefore uses the **superseded** task deliberately; a v2 number would be a different
  population and never the same row.
- the task's own prompt string is `"Given a online banking query, find the corresponding
  intents"`, which is part of the recipe for a model that consumes prompts.

### The version trap, found and resolved by evidence

**The registry of the version that produced the row does not name the selected model at
all.** `mteb 2.3.11`'s jina model file knows v2, v3 and v4 — the string
`jina-embeddings-v5` does not occur in it. Bisecting the releases, **`2.9.0` is the first
one whose registry names `jinaai/jina-embeddings-v5-text-small`**, and it pins **this very
revision** `46ed7da5…` together with the model-side recipe: `trust_remote_code`, and a
`model_prompts` map that sends a task of type `Classification` to the model's
**`classification`** adapter with the `"Document: "` prefix.

So a reproduction must **declare which of two configurations it ran**:

- **option A** — the published row's evaluator version plus *our own* reconstruction of the
  model wrapper;
- **option B** — `2.9.0`, whose registry entry is the authors' and pins the same revision.

Option B is recommended, because the difference was measured rather than assumed: between
`2.3.11` and `2.9.0` the `_undersample_data` **selection logic is byte-identical** (only the
returned tuple's arity and its docstring change) and `_calculate_scores` differs only in
type annotations; `is_cross_validation` defaults false, so this task does not take the new
cross-validation path. Verified at the level of the source contract, **not by
re-execution** — the same standard CB01 applied to the 1.38.9-versus-2.x caveat.

What is *not* allowed either way is printing an option-B number against the published row
as though it were option A, or the reverse.

---

## 5. What the native path no longer needs anything for

Reproduced **offline**, from bytes already held, before any score can exist.

**The population.** Both governed CSVs re-hashed in process against the contract pins:
train `b06e26ac…` 10,003 rows, test `d12d6e3b…` 3,080 rows, 77 labels in both. The mirror
substitution is **carried forward, not re-derived**: the contract records the mirror's text
sequence as identical to the authors' *in file order* on both splits, so an index computed
on the registered CSV is the row the published evaluator drew. The label-order trap is
carried too — the mirror's ids are **case-insensitive alphabetical, not byte-sorted**, and a
naive `sorted()` map mislabels most of the 77 classes.

**The ten training draws.** `_undersample_data` transcribed from the pinned source, with the
detail that decides the answer: a **fresh** `RandomState(42)` per call shuffling an `idxs`
list that is **carried forward** between experiments. Result: ten draws of exactly **616**
rows, **8 per label for all 77**, all ten **distinct**, set digest
`1fed226502b20c1f2418c22417ae0fe36e329f5b363e6c47ad72386356a1311d`.

Three negative controls, because a green check that cannot fail is not evidence:

| wrong design substituted | what must happen | observed |
| --- | --- | --- |
| seed 43 instead of 42 | every draw digest changes | all ten change |
| a fresh `idxs` list per experiment | the ten draws collapse to one | collapse to 1, digest ≠ pinned |
| labels renamed under a byte-sorted map | the draw must **not** move | digest identical |

The last one is the useful invariance: the greedy counter buckets by label equality, so the
draw does not depend on which of the three recorded label orders is used — the trap bites
the *scoring*, not the sampling.

**What is still missing after all of that is only the embeddings.** The rows, the order, the
seed, the draws, the probe, the scorer and the naive are fixed.

---

## 6. Resources, admission, and the reusable state that actually exists

Capacity was **re-read live** and no figure from any earlier report of mine was trusted
(`tools/df_host_capacity.py`, read-only, 2026-09-30T01:41Z):

| role | available | slice `MemoryMax` / `High` | slice `memory.current` | scopes | GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| coordinator | 21.79 GiB | 14.00 / 12.00 GiB | 1.430 GiB | 1 (this lane's own) | idle |
| **secondary worker** | **19.98 GiB** | 14.00 / 12.00 GiB | **1.282 GiB** | **0** | 15.56 GiB free, 0 processes |
| preferred external host | **3.47 GiB** | 8.00 / 7.00 GiB | 0.155 GiB | 0 | its large GPU stays `QUARANTINED_NOT_SCHEDULABLE` |

The secondary worker's slice residual reads **1.282 GiB** today. My own reports have given
1.88, 2.69 and 4.9–5.1 GiB for the same slice across three days; none of them was reused,
and this one is a reading rather than a claim about what is discardable. Nothing was
reclaimed, no ceiling was moved, `/dev/shm` was not touched, and no other lane's memory was
probed, signalled or waited on.

**The preferred external host stays ineligible and was not used.** Its cost and blocker are
retained by name and **no substitute is presented under its name** — as they are for the two
BANKING77 rows above the selection, `0.916656` at ~21 GB and `0.916136` at ~28 GB, which
still cannot be admitted anywhere here.

**The reusable state, read live rather than assumed.** On the secondary worker: the
digest-verified checkpoint `laya-typed-decisions` (846,205,481 bytes on disk) and **two**
pinned Python 3.13.9 environments, `laya 0.2.1` and `laya 0.3.11`, both on `torch 2.14.0+cpu`
and `transformers 5.17.0`. The Hugging Face hub cache holds one 40-byte metadata stub and
nothing else. So the honest statement is narrower than "nothing needs downloading twice":

> **Nothing needs downloading twice for the Laya route. For the native embedder route
> nothing has been downloaded once.** `mteb`, `sentence-transformers`, `peft`, `datasets`
> and `scikit-learn` are missing from **both** environments, and installing into either
> would mutate an environment a published reproduction depends on. The native route needs
> its own, third environment.

Every child in this lane ran through the already-deployed `crispdm-run` (md5
`3779972aef0bea873fc90bff61a74fc1`, identical on both hosts — **not** reinstalled), by
**absolute path** over ssh because a non-interactive shell carries no `~/.local/bin` on
PATH, with fresh aggregate admission each and `-q` rather than a shrunken cap:
`b77-capacity-read` 1G, `b77-worker-probe` / `b77-worker-probe2` 1G, `b77-native-offline`
2G, `b77-native-inventory` 1G on the worker, `b77-native-compat` 4G on the worker,
`b77-pipcache` 1G, `b77-native-report` 2G, `b77-native-tests` 2G. No declared cap was
shrunk to evade a refusal, nobody was displaced, no service was started, stopped or
restarted, no GPU ran, and the MT5 virtual machine was not approached.

---

## 7. The dependency ledger, and the exact request

| component of the native path | state | detail |
| --- | --- | --- |
| published target artifact | **PRESENT** | `ba162c90…`, aggregate reproduced from its own cells |
| evaluator source contract | **PRESENT** | pinned by digest at 2.3.11 **and** 2.9.0; draw and scorer source-identical for this task |
| population bytes | **PRESENT** | governed CB02 delivery, re-hashed in process against the pre-score pins |
| deterministic training draws | **PRODUCED HERE** | ten × 616 rows, 8/label, set digest `1fed2265…` |
| same-row naive | **PRESENT** | train-fitted majority 0.012987 on the identical test rows |
| admissible host | **PRESENT** | the secondary worker; 19.98 GiB available, 14.00 GiB slice ceiling, 15.56 GiB of idle VRAM |
| **model artifacts @ `46ed7da5…`** | **MISSING** | **1,369,721,378 B** over 22 files (1,248,458,137 B for the classification route alone) |
| **third pinned environment** | **MISSING** | `mteb`, `sentence-transformers`, `peft`, `datasets`, `scikit-learn` |
| remote-code execution decision | **UNDECIDED** | the repository ships `auto_map` and a custom sentence-transformers module: loading it executes third-party code |
| model wrapper version | **RESOLVED, MUST BE DECLARED** | option A or option B of §4, printed beside every number |
| device | **UNDECIDED** | the published artifact records none; declare, never blend |

**Why all four adapters, not just the classification one.** The repository's own remote code
calls `snapshot_download(repo_id=..., allow_patterns=["adapters/*"])` **at load time**
whenever the model path is not a local directory. An offline load therefore needs the whole
adapter set materialised — 4 × ~40.4 MB — which is why the request is for the repository and
not for one file.

**The environment request has two tiers, and the cheap one is the likely one.** Every symbol
the repository's remote code imports — `Qwen3Model`, `Qwen3Config`, `PreTrainedModel`,
`snapshot_download` — **resolves under the installed `transformers 5.17.0`**, checked by
import on the admissible host with no weights loaded. Necessary, not sufficient: the code has
not been *run*, because running it needs the weights.

- **tier 1** — `mteb==2.9.0`, `sentence-transformers==5.1.2`, `peft`, `datasets`,
  `scikit-learn`: **16,228,123 B** of direct wheels.
- **tier 2**, only if the installed line fails at runtime — the toolchain the model card
  declares (`transformers 4.57.0`, `torch 2.8.0`): **899,897,446 B** more.

The transitive closure is **not** resolved, because resolving it is itself a network act and
belongs to the same single request. The worker's pip HTTP cache holds 3,378,626,813 bytes,
large enough to plausibly already carry the installed `torch`/`transformers` blobs, so a
same-version install may need no network at all — stated as plausible, **not verified**.

**Licence, carried not buried.** `CC BY-NC 4.0`. Research only, and incompatible with any
commercial or trading use.

**The commands are prepared and were not run** (`prepared_commands` in the evidence):
acquire the repository at the pinned revision through the launcher under a declared cap;
re-hash every file against the repository tree read at that revision, on disk and again
inside the scoring child; build the third environment; run `Banking77Classification` with the
device declared; and if the draws, seed, probe or row order cannot be reproduced, return the
refusal instead of a number. **No authority is invented here and no allocation is assumed.**

---

## 8. What is NOT done, refused, or not measured

- **NO_NEW_MEASUREMENT.** No model ran, no weights were loaded, no dataset was downloaded,
  no GPU was opened, no inference happened, and no metric value was created. Every number
  about the published row is a property of its retained artifact; every number about the
  population is a property of governed bytes; every number about the recipe is an index set.
- **No score for the native path, and none is owed until the acquisition is approved.** The
  refusal is named in §1. A named refusal is not a completed family and is not presented as
  one.
- **No warehouse row and no receipt.** Nothing was written to the cube; no campaign was
  opened; no service was contacted.
- **No badge, from anything.** No `BUSINESS_HELD_OUT` measurement exists. The 19-prompt
  router corpus appears nowhere in this package — not as a population, not as a denominator,
  not as evidence — and it is not a classification benchmark. A declaration is not a badge.
- **No comparison across families.** Accuracy, macro-F1 and MAP remain three metrics;
  `MACRO_F1` against `WEIGHTED_F1` refuses by name even on the population where they
  coincide numerically. Both refusals are tested.
- **The two questions stay apart.** This return is the **native author reproduction** only.
  No adaptation of BANKING77 through our own framework was attempted, and none would be
  evidence for the native row.
- **The shipped wrapper's three defects are untouched.** `MAX_OPTIONS = 12`, the option
  budget, and the clamped `choice:11+` temperature are provider changes in `news-signal`,
  outside this order and not attempted.
- **The published `0.914578` is still not a warehouse row**, and writing it would be a new
  campaign. Not authorised here, not done.
- **Not re-verified by me:** the CB03 reproduction itself, the parity attribution, the
  retained per-row arrays, and the mirror's bytes (its digests are carried from the contract,
  and the mirror remains a comparison object, never a substitute for the authors' registered
  bytes).
- **No execution authority.** Nothing here authorises a deployment, a campaign or a trade.

---

## 9. Report in the required shape

```
B77-NATIVE — corrected scope, and the native reference path pinned to one blocker
repo/branch/tip: predictor / satoshi/banking77-native-path-20260929 / from 6a032341
files: docs/audits/work_plan/SATOSHI_BANKING77_NATIVE_PATH_2026_09_29.md ·
       tools/df_banking77_native_path_20260929.py ·
       tools/test_banking77_native_path.py ·
       docs/audits/evidence/banking77_native_path_20260929/{NATIVE_PATH.json,
         PUBLISHED_ROW_Banking77Classification.json} ·
       docs/audits/evidence/cb04_row_identity_20260929/BANKING77_LABEL_FIT_SCOPE_CORRECTION.json ·
       docs/audits/work_plan/SATOSHI_CB04_ROW_IDENTITY_2026_09_29.md (§9 addendum only;
         BANKING77_LABEL_FIT.json byte-identical, on purpose)
suites: native path 31/31, 2 skipped-then-green · negative controls: wrong seed, reset
        index order and relabelled classes each behave as required
corrected scope: unsupported by the SHIPPED wrapper and its serializations (12 options;
        706/894 tokens against 192/256) — NOT impossibility for any budget, provider or
        model; larger budget = different configuration; hierarchy/shortlist = distinct
        method scoring routing errors over the FULL population
published value + same-row naive: accuracy 0.914578 (PUBLISHED_REFERENCE, recomputed from
        its own ten cells, mteb 2.3.11, dataset 0fd18e25…, no device recorded) ·
        macro-F1 0.913809 · naive majority 0.012987 on the identical 3,080 governed rows
our measurement: NO_NEW_MEASUREMENT
finding instead of a score: BANKING77_NATIVE_PATH_BLOCKED_ON_MODEL_AND_ENVIRONMENT_ACQUISITION
exact unfulfilled dependency: 1,369,721,378 B of model artifacts @ 46ed7da5… (22 files,
        all four adapters required by the repository's own load-time snapshot_download),
        plus a THIRD Python 3.13 env with mteb/sentence-transformers/peft/datasets/
        scikit-learn (tier 1: 16,228,123 B of direct wheels; tier 2 fallback +899,897,446 B),
        plus an explicit trust_remote_code decision. Licence CC BY-NC 4.0, research only.
budget used: crispdm-run only, absolute path over ssh, fresh aggregate admission per child,
        1G–4G caps, -q never a shrunken cap; live capacity re-read 2026-09-30T01:41Z —
        secondary worker 19.98 GiB available, slice 14.00 GiB ceiling / 1.282 GiB current /
        0 scopes, 15.56 GiB idle VRAM; preferred external host ineligible and unused
what is NOT done / refused / not measured: no model ran, no download, no install, no GPU,
        no warehouse row, no receipt, no badge, no cross-family comparison, no framework
        adaptation, provider defects untouched, no execution authority anywhere
```

— Satoshi, successor technical lead, 2026-09-29.
