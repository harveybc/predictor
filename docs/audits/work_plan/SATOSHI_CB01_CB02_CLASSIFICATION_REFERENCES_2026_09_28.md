# CB01-CB02: classification reference evidence and governed datasets

Satoshi, successor technical lead, 2026-09-28.

Additive lane. No other lane was interrupted, no running job was duplicated, and
the completed Weather preparation was not repeated. Nothing in this return is
signed by, attributed to, or written on behalf of Musashi.

**No model score of ours exists on any classification task.** Everything below is
a dated selection from primary sources, a governed dataset registration, and a
protocol pinned before a score can exist. Where a published number is quoted it is
quoted as published, with its artifact; no identical number is promised, and no
state-of-the-art claim is made beyond what task-specific candidate evidence in
section 2 supports.

**The internal 19-prompt router corpus is not used anywhere in this lane.** It
measures a router over 19 prompts repeated five times. It is not a classification
benchmark, it is not a population, and it is not a denominator. No number in this
document comes from it.

## 1. The dated selection, per task

Selection rule, from the plan: rank only the same dataset revision, label space,
split, supervision regime and metric; the selection is the strongest reference
that is **reproducible under exactly matched conditions on hardware we can admit
today**. A stronger published number that we cannot reproduce is recorded as a
candidate and named as not selected, with its blocker.

### 1.1 FOMC monetary-policy stance (Trillion Dollar Words)

**SELECTED 2026-09-28: RoBERTa-large under the authors' own released recipe, on
the Combined-S seed-944601 split, weighted F1.**

The matched published target is not the paper's headline. It is the per-seed cell
of the authors' own released grid log that corresponds to the exact bytes we
registered:

| | value |
| --- | --- |
| Published artifact | `grid_search_results/final_lab-manual-split-combine_roberta-large.xlsx`, repo `gtfintechlab/fomc-hawkish-dovish`, git blob `66d1da3994dc60a95d11461a03c83d36a1b2573e`, sha256 `b7636ea2201046f1723079bf5fc8c818376b1323b792ff51214b9b6252f99caa` |
| Selection rule reproduced | highest **mean validation** weighted F1 over the three seeds selects lr 1e-5, batch 16 |
| Aggregate at that cell | mean test weighted F1 **0.711364**, sd **0.010655** — which is the paper's Table 5 Combined-S entry `0.7113 (0.0106)`, reproduced from the artifact rather than transcribed |
| **Matched single-split target (seed 944601)** | test weighted F1 **0.714420**, test accuracy **0.711694** |
| Paired train-derived naive on the identical 496 test rows | majority accuracy **0.497984**, weighted F1 **0.331096**, macro F1 **0.221624**; stratified expected accuracy **0.372874** |

Why the runners-up were not selected:

| Candidate | Published | Not selected because |
| --- | --- | --- |
| Fedspeak uncertainty-aware LoRA on Qwen3-14B, AAAI 2026 Oral, arXiv:2508.08001v3 | weighted F1 **0.7426** on Combined **seed 5768** (Table 1, "All Categories") | **Strongest same-split, same-metric, peer-reviewed number we could verify — and not reproducible here.** The LoRA adapter weights are not published (the repo ships data and scripts only), and the base Qwen3-14B needs about 35 GB in fp16, which no host we can admit today provides. Also single-seed against the benchmark's three-seed protocol, so the gap is not variance-controlled: the same cell's own seed spread on Combined is 0.700018-0.732673. Its matched same-seed anchor, from the same released grid log, is **0.718609** (seed 5768, lr 1e-5, batch 16), giving a published gap of +2.40 points that we are not in a position to reproduce. |
| Published `gtfintechlab/FOMC-RoBERTa` checkpoint | — | **Gated: manual access approval required**, which we do not hold and did not seek. Separately, the released checkpoint is not the model behind the headline: it was saved from Combined-S, seed 944601, batch 4, lr 1e-6, a different grid cell from the selected one. A missing checkpoint is a missing artifact; the selection therefore reproduces the recipe from the ungated MIT `FacebookAI/roberta-large` base, and says so. |
| "Mind the Shift" DCS, arXiv:2603.14313v1 | accuracy **0.7108**, F1 **0.7387** | **NOT_COMPARABLE, and a preprint with no venue and no code release.** Metric differs (accuracy and macro F1 against the benchmark's weighted F1); the label space is never stated and the task is described throughout as hawkish-dovish, so a 3-class evaluation is not asserted; no subset, seed or evaluation-set size is given; and the paper describes the benchmark as post-meeting *statements*, which it does not contain. Corroborating: they measure `FOMC-RoBERTa` at 0.4337 accuracy where the released grid log puts that family at 0.711694 accuracy on Combined-S seed 944601 — a ~28-point gap on their own baseline, which is itself evidence the evaluation set or label space is not this benchmark's. |
| LabelFusion-TS, arXiv:2608.11753v1 | weighted F1 **70.2** (probability ensemble) | **NOT_COMPARABLE by split**, and a preprint with code "to be released". Same 3-class label space and the same weighted-F1 metric, but a new chronological split: a 2,312-sentence deduplicated pool, 1,274 training sentences up to September 2015, and a 418-sentence 2015-2022 test set. Different test set, size and temporal regime. The nearest legitimate anchor is the benchmark's own Appendix-E chronological check at 0.7114, and even that is a different cutoff and test size. The authors do not claim otherwise; their in-protocol RoBERTa-large baseline is 66.1. |
| FinMTEB / Fin-E5 | classification aggregate **0.7565** (Table 1) | **No per-task number exists in any primary FinMTEB source** — the strings `FOMC` and `FinancialPhrasebank` each occur exactly once in the paper, in the dataset summary table, and in no results table; the code repository contains no result files. Its FOMC artifact is also a **re-split** (1,281 train / 1,000 test) and not this benchmark's split. **Fin-E5's weights are not downloadable**: the model repository contains only `.gitattributes` and a README that points to a commercial API. An overall rank does not establish an individual classification row, and a weightless repository is not an executable artifact. |
| FinMTEB's declared classification metric | paper says **MAP** | **Paper-versus-code discrepancy, resolved before any score.** The released evaluator sets `main_score="accuracy"` for both `FOMCClassification` and `FinancialPhraseBankClassification` and never computes a MAP; its precision-style outputs are `average_precision_score`, which is not MAP. MAP is genuinely this paper's *reranking* metric. If FinMTEB is ever used here, the evaluator's accuracy is the metric and the prose is wrong; MAP, accuracy and macro-F1 are not interchangeable merely because all three lie in [0,1]. |

### 1.2 AG News topic classification

Two populations, kept apart on purpose.

**SELECTED 2026-09-28 for the published-row reproduction: `convaiinnovations/laya-typed-decisions`
at revision `1a793eb568e6718f15941d08f85432581df534e3`, accuracy 0.953 on the
400-row sampled population.** Reproducible: apache-2.0, ungated, 421,293,830
parameters, about 0.84 GB of fp16 weights, and the population is recoverable
exactly (see 3.2). Runner-up `convaiinnovations/laya` at `55cf4c4e`, accuracy
0.950, same population — not selected only because it is the lower of the two
published cells on the same rows.

Two things must travel with that number and are not optional:

- **It is not a held-out score.** The benchmark's own results file marks the AG
  News suite `in_training: true` for all three checkpoints. The 0.950-0.953 cells
  are in-training-mix numbers. `banking77` is marked `in_training: false`.
- **The paired naive on those exact 400 rows is not the balanced one.** The
  official test split is perfectly balanced at 1,900 per class, but the *first
  400 rows in file order* are not: majority accuracy on them is **0.180000**,
  weighted F1 **0.054915**. Using the full split's 0.25 as the baseline for the
  sampled row would understate the skill and misstate the population.

**No selection is made today for the full official 7,600-row test.** The
strongest value we could verify from a primary source is XLNet at 4.45 % test
error, i.e. **95.55 % accuracy**, from arXiv:1906.08237 Table 4, column `AG`. It
is not selected because no fine-tuned AG News checkpoint from that work is
published, so the reference is a number without an artifact; and because a
7,600-row score is a different population from the 400-row published row and must
never become its denominator. The full-test evaluation is a separate experiment
with its own reference, still unresolved.

The Laya table's own `Jev (published)` column of 0.910 is a third-party figure the
benchmark states it never measured, over 100 rows. It is not a candidate.

### 1.3 BANKING77 fine-grained intent — scoped, not started

The plan's starting candidate is displaced by dated evidence, which is exactly
what the plan said to check for. All rows below are the **same** pinned dataset
revision `0fd18e25b25c072e09e0d92ab615fda904d66300` and the same evaluator
contract (see 3.3).

| Model | MTEB `Banking77Classification` accuracy | Weights | fp16 weights | Licence | Result committed |
| --- | ---: | --- | ---: | --- | --- |
| `ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826` | **0.916656** | open | ~21 GB | MIT | 2026-08-24 |
| `codefuse-ai/F2LLM-v2-14B` | 0.916136 | open | ~28 GB | Apache-2.0 | 2026-03-18 |
| **`jinaai/jina-embeddings-v5-text-small`** | **0.914578** | open | **~1.2 GB** | CC BY-NC 4.0 | 2026-02-19 |
| `Qwen/Qwen3-Embedding-8B` (plan's starting candidate) | 0.872727 | open | ~14.1 GB | Apache-2.0 | 2025 |

**SELECTED 2026-09-28, subject to CB03 admission and not started: `jinaai/jina-embeddings-v5-text-small`
at revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1`, accuracy 0.914578.** It is
the strongest row that is reproducible under conditions we can actually admit: at
about 1.2 GB of weights it fits every host in the fleet, including a host with
constrained RAM, and it is the only top row that does not require the preferred
accelerator.

Why the higher rows were not selected: `Giga-Embeddings` (+0.002078) needs about
21 GB and `F2LLM-v2-14B` (+0.001558) about 28 GB, so both require the preferred
external accelerator; that host has roughly 4 GiB of available host RAM against
4.38 GiB of unreclaimable slab, so its cold GPU is not availability and neither
model can be admitted today. Their cost and exact blocker are retained here
rather than silently replaced. `Qwen3-Embedding-8B` is not selected because three
dated 2026 results beat it by 4.2-4.4 points on the identical dataset revision
under the identical evaluator contract; its own paper publishes **no** BANKING77
per-task number at all (the string "banking" does not occur in it), so the number
attributed to it exists only in the MTEB results repository.

Caveat carried, not hidden: the three 2026 results were produced by `mteb` 2.x
and the Qwen3 result by 1.38.9. The governing defaults were verified equal by
reading both code paths, and `_undersample_data` explicitly retains the v1
random-state for backward compatibility, but no same-version re-evaluation was
run. "Same evaluator" is therefore verified at the level of the source contract,
not by re-execution.

Also above the Qwen3 row but not candidates: `google/gemini-embedding-001`
0.942695, `mongodb/voyage-3-m-exp` 0.938019 and `Bytedance/Seed1.6-embedding`
0.920455 are API-only with no weights, so no exactly matched local reproduction
is possible.

`jina-embeddings-v5-text-small` is CC BY-NC 4.0. Research-only. That is
compatible with this lane and incompatible with any commercial or trading use.

### 1.4 Financial PhraseBank — NO SELECTION TODAY, with reasons

Not a blocked intention; a genuine negative result, recorded so nobody invents a
substitute later.

- **There is no official train/test split.** Confirmed from the distributor: every
  agreement subset exposes a single `train` split, and the original work evaluates
  by cross-validation. A fixed split therefore has to be *created*, and a created
  split cannot reproduce anyone's published number.
- **The only candidate protocol leaks.** FinMTEB's `FinancialPhraseBankClassification`
  sets `eval_splits=["train"]` while the evaluator hardcodes `train_split="train"`,
  so the probe is fitted on rows drawn from the same split it scores, and the
  dataset's own 1,000-row test split is never used. Adopting that as a reference
  would import the leak.
- **Fin-E5 is not an executable artifact** (1.1).
- **Row counts disagree between the paper and the shipped files**: 2,259 / 3,448 /
  4,211 / 4,840 in the paper against 2,264 / 3,453 / 4,217 / 4,846 in the
  distributed subsets, unexplained by any primary source. The exact-agreement
  subset the plan names is therefore ambiguous at the row level, and the plan's
  own rule is to resolve that before a score exists. It is not resolved.
- Licence CC BY-NC-SA 3.0: research-only, and the distributor requires contacting
  the authors for any commercial licence.

### 1.5 MASSIVE, English and Spanish — second stage, scoped only

60 intents, 18 domains, 51 languages; `en-US` and `es-ES` both 11,514 train /
2,033 dev / 2,974 test; data CC BY 4.0 (the repository *code* is Apache-2.0 — the
two are different and the data licence governs). MTEB task
`MassiveIntentClassification`, `eval_splits=["validation", "test"]`, subset keys
`en` and `es`, not the MASSIVE locale codes.

One trap recorded now: the Laya benchmark's 51-language MASSIVE sweep is **a
20-option question**, not the 60-label benchmark, at 100 cases per language. Its
English intent cell of 0.783 and Spanish 0.510 are therefore not
`MassiveIntentClassification` numbers and must never be placed in the same column.
Nothing was started for MASSIVE.

## 2. Governed datasets: what is delivered and verified, and what is missing

Existing lake and governance tooling only. **No second registry was built.** The
new tool `tools/df_extend_classification_lake_20260928.py` rebinds the same
operator adopter the Weather/Traffic extension used, so the additive-change,
rehearsal-bound, rollback-on-failed-post-check rules apply unchanged.

### 2.1 Governed-delivered and verified

Lake `sota_benchmarks`. Live route verified 2026-09-29 04:37 UTC (2026-09-28
local). All six resources returned `VERIFIED_TRANSFER`; the delivered file's
sha256 was re-hashed on disk and matched; availability stayed `UNDECLARED` with
label `UNKNOWN`; the campaign was closed with terminals through the outbox;
refusal checks held; and the warehouse content matched the emitted terminals.
Delivery was verified, not assumed.

| Governed resource | Rows | Bytes | sha256 |
| --- | ---: | ---: | --- |
| `agnews_zhang2015_train/train.parquet` | 120000 | 18585438 | `fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee` |
| `agnews_zhang2015_test/test.parquet` | 7600 | 1234829 | `71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb` |
| `fomc_tdw_shah2023_train/train.csv` | 1984 | 422592 | `3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7` |
| `fomc_tdw_shah2023_test/test.csv` | 496 | 103896 | `c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac` |
| `banking77_casanueva2020_train/train.csv` | 10003 | 839073 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `banking77_casanueva2020_test/test.csv` | 3080 | 239961 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

Every file was verified against **both** its expected sha256 and the
distributor's own object identity before anything was copied: the two AG News
parquets against the Hugging Face LFS oid, which is the sha256 itself; the four
CSVs against their upstream git blob ids
(`00253eaaaa017929bd0c60e61d79923b4eaaec8c`,
`230d4a28d6781b58a32d6a31d038e7599d623e16`,
`98e2543cf482d0dca7bfb175ebe35d98efad95be`,
`799687a8367359432985b8b13d85a2baf73f92dd`). The bytes are unchanged distributor
bytes; nothing was converted, re-encoded, re-split or imputed.

Original dataset citations, licences and pinned revisions:

| Corpus | Original citation | Distributor and revision | Licence | Use class |
| --- | --- | --- | --- | --- |
| AG News | X. Zhang, J. Zhao and Y. LeCun, "Character-level Convolutional Networks for Text Classification," NIPS, 2015, arXiv:1509.01626. Official sizes 120,000 / 7,600, 4 classes, 30,000 and 1,900 per class. | `fancyzhx/ag_news` @ `eb185aade064a813bc0b7f42de02595523103ca4` | **Undeclared by the distributor** (card states "unknown"). The underlying AG corpus's own terms page permits **non-commercial** use only and forbids redistribution under a different name. | `BENCHMARK/RESEARCH_ONLY_UNRESOLVED_LICENCE` |
| FOMC / Trillion Dollar Words | A. Shah, S. Paturi and S. Chava, "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis," ACL, pp. 6664-6679, 2023, doi:10.18653/v1/2023.acl-long.368. | `gtfintechlab/fomc_communication` @ `6b0283f55f0005a6d38d49f271d795c21fccc1a3` | **CC BY-NC 4.0** | `BENCHMARK/RESEARCH_ONLY_NONCOMMERCIAL` |
| BANKING77 | I. Casanueva, T. Temcinas, D. Gerz, M. Henderson and I. Vulic, "Efficient Intent Detection with Dual Sentence Encoders," NLP4ConvAI, pp. 38-45, 2020, arXiv:2003.04807. 13,083 examples over 77 intents: 10,003 train / 3,080 test. | `github.com/PolyAI-LDN/task-specific-datasets` @ `9d081458ff52e53cf7e848f414e6e9344e4e6696` (the authors' own CSVs) | **CC BY 4.0** | `BENCHMARK/PUBLIC` |

Licence contradiction recorded, not resolved by us: the MTEB task metadata
declares `license="mit"` for `Banking77Classification` and `apache-2.0` for
`MassiveIntentClassification`. Both contradict the upstream data licences. Cite
the distributors, not the benchmark harness.

### 2.2 How research-only material is stopped from becoming live-trading data

Not a label. A refusal, in code that was exercised.

These six resources are registered `untimed`. The deployed provider therefore
delivers them **whole-resource `AS_IS` with an `UNDECLARED` availability scope and
an `UNKNOWN` availability label, and refuses every date-ranged request outright**.
A point-in-time slice of them cannot be produced at all, which is precisely what a
live-trading consumer would need. The refusal checks passed live on each resource.
On top of that structural bar, each research-only resource carries an explicit
`use_class` plus `commercial_use: REFUSED` and `trading_use: REFUSED` in the
declared sheet and in the additive store receipt
`CLASSIFICATION_EXTENSION_20260928.json`. No commercial or trading entitlement is
created by this registration, and none is inferred from a permissively licensed
wrapper.

### 2.3 Still missing: pinned, deliberately not registered, and why

| Artifact | Pinned identity | Why it is not registered |
| --- | --- | --- |
| BANKING77 `categories.json`, the ordered official 77-label vocabulary | git blob `cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4`, sha256 `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63` | The deployed provider delivers only `.csv` and `.parquet`. The vocabulary and its order are pinned here and in the population contract instead, and must be verified out of band. |
| Financial PhraseBank v1.0, all four agreement subsets | `takala/financial_phrasebank` @ `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`, `data/FinancialPhraseBank-v1.0.zip`, CC BY-NC-SA 3.0 | Distributed as a zip of latin-1 `.txt`. Only a derived file would be registrable, and a derived file is no longer the distributor's bytes. Section 1.4 means there is nothing to score against it yet regardless. |
| MASSIVE intent, en-US and es-ES | `mteb/amazon_massive_intent` @ `940fd47a81eaa7f2cc7b129674d945d618ac38c2` (per-locale `.json.gz`); original `AmazonScience/massive` @ `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`, CC BY 4.0 | Same file-type rule. Second-stage corpus, deliberately not started. |
| `mteb/banking77` jsonl mirror @ `0fd18e25b25c072e09e0d92ab615fda904d66300` | test `fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595`, train `d411780d8c0e18e166f5664c6cfe90dc9de399d722aa7cde282e31a771323ea7` | Not a governed resource by design. It is the artifact the published evaluator reads, and it is held as a **comparison object** (3.3), never as a substitute for the authors' registered bytes. |

### 2.4 Defect found in the deployed provider, reported rather than papered over

The resource-contract validator was written for time series. It requires
non-empty `event_time_column`, `available_time_column`, `timezone` and `frequency`
strings even for a corpus that has no time axis at all. On the `untimed` delivery
path none of them is ever parsed. Rather than name a column that does not exist in
the file, this registration carries explicit sentinels
(`NOT_A_TIME_SERIES_NO_EVENT_TIME`, `NOT_A_TIME_SERIES_NO_SAMPLING_INTERVAL`, and
so on), and the correction belongs in the provider: a contract should be able to
say "this resource has no time axis" without a placeholder. No fabricated column
name was written.

### 2.5 Governance state: a correction to the premise I was given

I was told a new governed unit needs a data-gov service key we do not hold, and
that if so I should read already-delivered bytes from a verified cache and label
the result a declared transport rather than a new governed unit.

**That premise is wrong, and the honest report is the correction.** The predictor
actor's service key is present at the path the deployed adopter already reads, and
the Weather/Traffic extension used it live earlier the same day. I did not hunt
for a secret, copy one, invent an authorization, or write a runner that avoids
the dependency: I ran the existing adopter, which reads the existing key from its
existing location. It opened **twelve new governed units** — one completed unit
and one deliberately failed probe unit per resource, so no campaign is left open —
and all twelve reconciled. These are new governed units. They are **not** a
declared transport of cached bytes, and describing them as one would understate
what happened.

### 2.6 No collateral damage

The three Time-Series-Library resources were re-hashed after the change and all
three match their receipted digests. Their existing contracts and the original
`BUILD_RECEIPT.json` were not rewritten; the new provenance is in an additive
receipt beside it. The data-gov runtime config was byte-identical before and
after (`data_gov_config_unchanged: true`), and only the benchmark lake service
restarted. The warehouse and data-gov services were not restarted. The lake was
idle at the moment of the restart — the journal for the preceding twenty minutes
shows health probes only.

## 3. The exact executable protocol, pinned before any score exists

`docs/contracts/classification_populations.v1.json` and
`docs/contracts/classification_naive_baselines.v1.json` are produced by
`tools/df_classification_protocol_20260928.py` **from the governed delivery
cache**, and every file is re-hashed inside that process and refused unless its
sha256 equals the digest the governed delivery published. The identities are bound
to the delivered bytes, not to a filename.

### 3.1 Populations, with identities

Each population carries a `population_sha256` over its ordered
`(position, text, label)` triples in canonical separator-explicit JSON, so no
formatting choice can move the digest.

| Population | Rows | Role |
| --- | ---: | --- |
| `agnews_test_full` | 7600 | separate experiment; never the denominator of the published sampled row |
| `agnews_test_first400_laya_published` | 400 | the published Laya row's population |
| `banking77_train_authors_csv` / `banking77_test_authors_csv` | 10003 / 3080 | 77 distinct labels in both |
| `fomc_train_distributor_split` / `fomc_test_distributor_split` | 1984 / 496 | the Combined-S seed-944601 split |

### 3.2 AG News: the sampled row identities, resolved

The Laya benchmark script takes `list(d)[:400]` from the `fancyzhx/ag_news` test
split. That is **deterministic file order, not a random sample** — the script's
`seed = 13` is used only for the shuffled suites, not for AG News. The row
identities are therefore recoverable exactly, and are now pinned against the
registered revision. This matters because the script pins **no** dataset
revision, so a recovery is only meaningful against a named one; the benchmark
publishes aggregate metrics with no index, id, hash or per-row prediction field
for the classification suites, which is the single largest reproducibility gap in
those numbers.

Also pinned verbatim, because changing any of them makes a new variant rather
than a replication: the instruction string `What is the topic of \`article\`?`, the
four option keys `world` / `sports` / `business` / `sci_tech` in that order, and
their glosses "world news and international politics", "sports", "business and
economy", "science and technology".

**Every label option fits.** Four options are far inside the author-stated
~20-option budget, so no truncation check can fail on AG News. That is exactly
why BANKING77 is the stress test: the same benchmark reports 0.425 for two
different checkpoints on 77 labels and attributes it to the shared option-token
budget, not to a capability gap — about four tokens per label. A 77-option
prompt must be proven untruncated before any BANKING77 number of ours is
recorded, and the option list must never be shortened to make it fit.

There is no sampling temperature to match: the model is non-autoregressive and
emits a distribution in one forward pass. Every "temperature" in its benchmark
documentation is a *calibration* temperature, per (question type, option count)
bucket, clamped to [0.5, 5]. Since a temperature-scaled softmax has the same
argmax at every positive temperature, accuracy is clamp-invariant while ECE is
not. Calibration and accuracy are recorded separately for that reason.

### 3.3 BANKING77 version compatibility, resolved before any score

We registered the authors' own CSVs. The published evaluator reads a *different*
artifact: the `mteb/banking77` jsonl mirror at revision
`0fd18e25b25c072e09e0d92ab615fda904d66300`. Comparing a number measured on one to
a number published on the other is only legitimate if the rows are the same rows,
so this was checked, not assumed.

**Decision, recorded before any score: comparable, on content and on order.**
The mirror's `(text, label_text)` multiset equals the authors' `(text, category)`
multiset on both splits, and the text sequence is identical **in file order** as
well, on 10,003 train and 3,080 test rows.

One trap was found in the process and is the reason this check was worth running.
The mirror's integer label ids are **case-insensitive** alphabetical order, not a
bare `sorted()`. The first attempt at this check asserted a plain ASCII sort and
correctly failed. The two sorts genuinely differ: `Refund_not_showing_up` is
position 0 under an ASCII sort and position 61 under a case-insensitive one, so an
id map built with a bare `sorted()` would mislabel most of the 77 classes. Neither
sort is the authors' `categories.json` order, which begins `card_arrival`. All
three orderings are recorded in the contract, together with which one the mirror
actually uses, because a prompted classifier shown its options in a different
order is a different experiment. The residual caveat: the MTEB evaluator draws its
own 8-per-label subsample under seed 42, so an order-dependent recipe must still
re-derive that draw rather than rely on file order.

The evaluator contract, read from source and pinned: `AbsTaskClassification` with
scikit-learn `LogisticRegression(max_iter=100)`, `samples_per_label = 8` (616
training embeddings for 77 labels, out of 10,003), `n_experiments = 10`, seed 42,
`main_score = "accuracy"`. This is **subsampled few-shot linear probing, not
zero-shot prompting and not full-train fine-tuning.** A prompted zero-shot number
on the same test rows is a different supervision regime and is reported in its own
row, never merged. Calling the probe zero-shot would be wrong: labels train the
head. The published per-task artifact path convention is
`results/{model with "/" as "__"}/{model_revision}/{TaskName}.json` in
`embeddings-benchmark/results`, and the Qwen3-8B file records `mteb_version`
1.38.9, `dataset_revision 0fd18e25...`, `accuracy 0.872727` with ten
`scores_per_experiment` spanning 0.867532-0.879545.

### 3.4 FOMC split, resolved before any score

The distributor publishes exactly two files and no validation split. Their 1,984 /
496 sizes and the grid log's own test accuracies (0.493952 x 496 = 245 exactly)
identify them as the **Combined-S, seed 944601** split of the eight-variant,
three-seed family the authors released, and no sentence appears in both splits.

**Decision: a reproduction may use only that one fixed split.** The paper's
headline is a mean over three seeded splits; a single-split number is a different
estimator and must not be printed in the same column as it. That is why the target
in 1.1 is the per-seed cell 0.714420 and not 0.7113.

### 3.5 Paired naive baselines, fitted on train labels only

Computed on the identical governed test rows, before any model score exists, so
they cannot be chosen afterwards.

| Task | Test rows | Majority accuracy | Majority weighted F1 | Majority macro F1 | Stratified expected accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `agnews_test_full` | 7600 | 0.250000 | 0.100000 | 0.100000 | 0.250000 |
| `agnews_test_first400_laya_published` | 400 | 0.180000 | 0.054915 | 0.076271 | 0.250000 |
| `fomc_combined_s_seed944601` | 496 | 0.497984 | 0.331096 | 0.221624 | 0.372874 |
| `banking77_test` | 3080 | 0.012987 | 0.000333 | 0.000333 | 0.012987 |

Per-class precision, recall, F1 and support are in the contract file for every
class of every task.

### 3.6 What the metric contract must carry, and what it must not reuse

Recorded here as a requirement on CB04, not implemented by this return. The
existing TSL MSE/MAE receipt contract is **not** a classification contract.
Classification receipts need: task and supervision regime; dataset revision,
split and population identity; label vocabulary **and order**; scorer and
averaging (weighted F1, macro F1 and accuracy are three different metrics);
probability semantics with the calibration split declared, since an
entropy-derived confidence is not P(correct); abstention coverage and selective
risk, with refusals never removed from the accuracy denominator silently; and the
paired train-derived naive on the identical rows. Contamination is **UNKNOWN**
wherever it is unknown — and for AG News it is worse than unknown, it is
positively known to be in the training mix.

## 4. Resources: what was used, what was refused, what was not touched

The coordinator was used for lightweight orchestration and for the CPU-only
inventory, registration, population-pinning and naive-baseline steps, each through
`$HOME/.local/bin/crispdm-run` at the already-deployed revision, which was not
reinstalled. Caps were 3-6 GB; no cap was shrunk to evade anything, and no
request was refused.

**Nothing was run on any GPU.** The preferred external accelerator host was not
used: its cold GPU is not availability, and it has not been given a fresh host-RAM,
UUID, temperature, competition and budget admission by this lane. The secondary
worker holds an admitted forecasting execution and **was not displaced, probed for
capacity, or touched at all.** No agent fan-out beyond two bounded,
read-only literature agents that wrote nothing and ran no training.

Missing compute did not block this return, because the protocol and the dataset
registration do not need it. What it does block is stated plainly: sections 1.2,
1.3 and 3.3 are specifications awaiting CB03 admission, and the two BANKING77 rows
above our selection remain unreproducible until a host can admit ~21 GB and ~28 GB
of weights.

## 5. What is NOT done, refused, or not measured

- **No classification score of ours exists.** No native reproduction was run, no
  wrapper parity was executed, no business corpus was scored.
- **No claim that any model is state of the art.** Section 1 gives dated,
  task-specific candidate evidence and names what it could not verify.
- **No identical number is promised.** 0.714420, 0.953 and 0.914578 are other
  people's published values on named artifacts, quoted as such.
- Financial PhraseBank has no selection (1.4) and MASSIVE was not started (1.5).
- BANKING77 was scoped, not started, in line with the order.
- The gated FOMC checkpoint was **not** requested, and no substitute was presented
  under its name.
- No unofficial quantization, no smaller model under a larger model's name, no
  shortened option list.
- The provider defect in 2.4 was reported, not fixed: fixing it is a change to a
  deployed service outside this lane's scope.
- Financial PhraseBank's 5-to-6-row-per-subset discrepancy between the paper and
  the shipped files is **unresolved by any primary source**, and is recorded as
  unresolved rather than picked.
- Two classification-population rows in the Laya tables that I did **not** verify
  independently: the DAIR Emotion cells and the application-workflow suites. They
  are outside this lane's five tasks.

## 6. Return, in the required shape

```
CB01 — candidate matrix and dated selection
repo/branch/tip: predictor / satoshi/cb01-cb02-classification-references-20260928
files: docs/audits/work_plan/SATOSHI_CB01_CB02_CLASSIFICATION_REFERENCES_2026_09_28.md
       docs/contracts/classification_populations.v1.json
       docs/contracts/classification_naive_baselines.v1.json
       tools/df_classification_protocol_20260928.py
selections (dated 2026-09-28):
  FOMC/TDW   RoBERTa-large, authors' recipe, Combined-S seed 944601, weighted F1
             matched published target 0.714420 (their own grid artifact); naive 0.331096
  AG News    laya-typed-decisions @1a793eb5, 0.953 on the 400-row published population
             (in-training, not held out); naive on those exact rows 0.054915 weighted F1
  BANKING77  jina-embeddings-v5-text-small @46ed7da5, 0.914578 — scoped, NOT started
  PhraseBank NO SELECTION: no official split, the only candidate protocol leaks,
             Fin-E5 has no weights, row counts disagree with the paper
  MASSIVE    second stage, not started
runners-up not selected: Fedspeak 0.7426 (adapter weights unpublished, ~35 GB);
  gated FOMC-RoBERTa (manual access, and a different grid cell from the headline);
  Mind the Shift and LabelFusion-TS (NOT_COMPARABLE: metric/label space, and split);
  FinMTEB/Fin-E5 (no per-task row exists, weightless repo, re-split, MAP-vs-accuracy);
  XLNet 0.9555 (no released fine-tuned checkpoint, and a different population);
  Giga-Embeddings 0.916656 and F2LLM-v2-14B 0.916136 (~21 GB and ~28 GB, no admissible host);
  Qwen3-Embedding-8B 0.872727 (beaten by 4.2-4.4 points on the identical revision)
acceptance: no score of ours exists; every quoted value carries its artifact
what is NOT done / refused / not measured: see section 5

CB02 — governed datasets and exact executable protocols
repo/branch/tip: predictor / satoshi/cb01-cb02-classification-references-20260928
files: tools/df_extend_classification_lake_20260928.py
       docs/CLASSIFICATION_BENCHMARK_LAKE.md
governed-delivered and VERIFIED (live route, 2026-09-29 04:37 UTC): 6 resources,
  3 corpora — AG News train+test, FOMC train+test, BANKING77 train+test.
  All VERIFIED_TRANSFER, delivered sha256 re-hashed on disk and matched,
  availability UNDECLARED/UNKNOWN, terminals reconciled, warehouse content matched,
  date-range refusals held, 12 new governed units opened and closed.
still missing (pinned, not registered, with reasons): BANKING77 categories.json,
  Financial PhraseBank, MASSIVE en/es — none is .csv or .parquet, which is the only
  file type the deployed provider delivers.
acceptance: rehearsal on the disposable stack first (it caught a real campaign-key
  collision before any production change); TSL resources re-hashed and unchanged;
  data-gov config byte-identical; only the benchmark lake service restarted.
what is NOT done / refused / not measured: no second registry, no conversion of
  distributor bytes, no commercial or trading entitlement, provider contract defect
  reported not fixed, no GPU work, secondary worker's admitted job untouched.
```

Satoshi, successor technical lead, 2026-09-28.
