# CB04 row identity, and the BANKING77 label-fit gate

Satoshi, successor technical lead. 2026-09-29.
Order: Musashi's partial dictamen of request 23b2efa3, order 3 — take ownership of the
metric-row identity defect this lane's own write found, and take BANKING77 as the next
classification trial under its own authority and admission.

Worktree `.worktrees/predictor-cb04-rowid-20260929`, branch
`satoshi/cb04-row-identity-and-banking77-20260929`, from `96d41960`.

---

## 0. The two questions the order asked me to lead with

**Can a secondary metric row still inherit an identity that is not its own? — No, not in
the live warehouse's current generation.** It could when I was given this order, and I
proved it again read-only before touching anything: macro-F1 `0.9467546527629132` stood
under identity `1756f877` (an ACCURACY identity) on unit `native-accuracy` and under
`a62f184c` (a MACRO_F1 identity) on `native-macro-f1`, while `1756f877` also covered
macro-F1 `0.9252733932274245` on `framework-accuracy`. Each of the three accepted
terminals is now superseded by a generation 2 that carries
`classification_row_identity.v1`: a per-row identity digest, a per-row occurrence key,
and evidence class as a dimension separate from the identity. All three successors were
read back out of the live cube with every tag round-tripping, and the generation-1 rows
are untouched, digest for digest.

**Do all 77 BANKING77 label options fit untruncated? — No, and that is the finding.** No
score exists and none was produced. Measured with the checkpoint's own digest-verified
tokenizer: the 77 options need **706 tokens** at the absolute floor (empty descriptions)
and **894 tokens** rendered with the label as its own description, against a
`head_max_len` of **192** hard-coded in the framework and **256** declared by the
checkpoint. The option budget is negative in all four combinations — **−514 / −702** at
192 and **−450 / −638** at 256 — so **repairing the hard-coded 512/192 to the
checkpoint's own 1024/256 would not make the labels fit**. Under 512 the room left for
the customer message is **zero**. Before any of that, the shipped question builder
refuses outright: `MAX_OPTIONS = 12`, so a 77-option choice cannot be built at all.

---

## 1. The defect, and why the risk was the arithmetic rather than the label

`app/classification_receipt.metric_identity_sha256` digests `author_primary_metric`, and
`terminal_tags` puts the result on the terminal. A terminal-level digest binds the
receipt's **primary** metric, so every other row a receipt projects — its paired naive,
its secondary families, its probability metrics, its coverage, its 16 confusion cells —
reached the warehouse carrying an identity that is not its own. The contract's own
guidance, *two rows agree only when their identity digests agree*, is unsound for
secondary rows, and is now withdrawn in `docs/CLASSIFICATION_METRIC_CONTRACT.md`.

Nothing let one metric be read as another: every stored row's key and unit were correct.
The damage was arithmetic. Counted from the live rows, not asserted:

| from the live warehouse | value |
| --- | ---: |
| metric rows under `classification_metrics.v1` | 72 |
| distinct terminal-level identity tags across them | 2 |
| rows that are **one measurement reported twice** | **23** |
| distinct occurrences once deduplicated | **49** |
| distinct row identities | 25 |
| value conflicts inside one occurrence | 0 |

`native-accuracy` and `native-macro-f1` are the same native reproduction: they agree on
provider, checkpoint digest, supervision regime, evidence class, evaluation split,
population digest, protocol digest, scorer digest, seed and denominator policy, and
differ only in which family the receipt called primary. So each carries accuracy
`0.9525`, macro-F1 `0.9467546527629132`, the same Brier, ECE and NLL and the same
confusion. **An aggregate keyed on the identity tag would have averaged that measurement
twice**, and 23 of the 72 stored rows would have been double counted. `framework-accuracy`
is genuinely distinct — its own provider, protocol and scorer — and is correctly kept as
its own occurrence.

## 2. The repair: three separations, in a successor contract

`docs/contracts/classification_row_identity.v1.json`, enforced by
`app/classification_row_identity.py`.

**Its own identity per row.** `row_identity_sha256` digests only what decides what the
number in that row *means*: family, metric key, unit, kind, definition, the governing
denominator policy and the label order. The author's own name for a metric is
**attribution, not meaning**, so it leaves the digest and becomes `reported_under_name`
plus the refusal `compare()` already raises by name. That is not a loosening — it is what
makes the double count detectable at all: a macro-F1 carried as a primary in one terminal
and as a secondary in another can only be recognised as one measurement if neither the
reporter's label nor the row's role is inside the digest.

**Evidence class separated from the identity.** A `PUBLISHED_REFERENCE` accuracy and a
`MEASUREMENT` accuracy of one definition share an identity — that is exactly what makes
them comparable, and exactly why they must never be averaged. So provenance sits *beside*
meaning: `evidence_class` is out of the identity, into the occurrence key, and an
aggregate that would put both in one group is refused by name
(`EVIDENCE_CLASSES_MIXED_IN_ONE_AGGREGATE`). This removes the reason CB-C had to keep the
author's published `0.9525` out of the warehouse; storing it would be a new campaign and
is **not** done here.

**No double counting between terminals.** `metric_occurrence_sha256` names which
measurement a row reports. It deliberately excludes the row's role and the terminal,
campaign, unit, generation and receipt that carried it — a double count *between*
terminals is the failure it exists to catch, and a key carrying the terminal could not
catch it. It includes protocol digest, scorer digest and seed, so a genuine replicate
stays its own occurrence and is never deduplicated away. Two rows sharing an occurrence
key and disagreeing in value are `CONFLICTING_VALUES_FOR_ONE_OCCURRENCE`, a refusal, not
an average.

The corrected aggregation rule, now in the contract document and in the tags themselves:

> GROUP BY `metric_row_identity_sha256` **and** `evidence_class`; DEDUPLICATE ON
> `metric_occurrence_sha256` before any mean, count or ranking; a value conflict is a
> refusal; a group mixing evidence classes is split, never weighted.

Two secondary properties, both deliberate. A 77-class matrix contributes 308 rows whose
per-row digests no tag could hold, so the JSON tag maps carry the rows that fit and the
two map digests always cover **every** row; any row — including every confusion cell — is
recomputable from the stored tags with `recompute_from_tags(tags, metric_key)`, through
the same `describe_row` the producer used, so reader and producer cannot drift. And a
receipt whose primary denominator policy disagrees with its abstention denominator policy
is now refused by name, because `classification_metrics.v1` already checks every carried
value against the confusion under the abstention policy.

### What was deliberately not edited

`app/classification_receipt.py` and `docs/contracts/classification_metrics.v1.json` are
**byte-identical** to `96d41960`. The sealed CB04 business corpus manifest pins the digest
of both, so the corpus still pins exactly what it was sealed against — and two tests now
fail if either changes, turning a dead record into an active guard. The repair is a
successor module and a successor contract; producers under it call
`terminal_tags_with_row_identity`, which calls the old `terminal_tags` unmodified and adds
tags beside it. No existing tag changes value, and the nine stored metric fields are
returned untouched.

## 3. Tests red first, and proved red against the wrong design

`tools/test_classification_row_identity.py`, 29 tests. Four of them (`TheDefectAsShipped`)
assert the shipped behaviour rather than the repair, so they fail if a later edit quietly
changes the old function and leaves the successor tags describing a problem that no longer
exists.

A suite that passes is not evidence until it can fail, so
`tools/df_cb04_row_identity_mutation_probe_20260929.py` substitutes three wrong designs in
turn and reruns the 21 repair tests: the shipped defect (every row takes the terminal-level
primary identity) kills **13**, `row_role` inside the identity kills **5**, `evidence_class`
inside the identity kills **4**. Verdict `EVERY_MUTATION_IS_CAUGHT`, baseline and
after-restore both green. Evidence:
`docs/audits/evidence/cb04_row_identity_20260929/MUTATION_PROBE.json`.

`tools/test_classification_row_identity_warehouse.py`, 7 tests on a **disposable** store:
the installed `PredictorDuckdbStore`, the distribution the warehouse host runs, on a
temporary DuckDB file, with a substitution guard and nothing mocked. It writes two
terminals that reproduce the live shape and then reads **only** the store: under the legacy
tags the double count is visible from the stored rows; under the successor tags 40 stored
rows collapse to 21 distinct occurrences with 19 duplicates dropped and no conflict, and
macro-F1 and accuracy each land in exactly one group. A BANKING77-sized receipt — 313 rows
— round-trips, no tag exceeds its budget, and every row's digest is recomputed from the
stored tags and checked against the stored map digest.

## 4. The correction as a successor, never as an edit

`tools/df_cb04_row_identity_successors_20260929.py`. The three accepted rows stay exactly
as they are. Before building anything it checks, per unit, that the stored row **is** the
projection of the retained receipt: the stored `receipt_sha256` tag equals the receipt's
digest, the stored metric rows equal `terminal_metrics(receipt)` key for key and value for
value, and no contract tag has drifted. All three passed; a failure would have sent
nothing.

Each successor is generation 2 of the **same campaign and unit**, keeping the accepted
row's status, reason, clocks, costs, artifacts and delivery, with **byte-identical metric
rows** — this is a tag correction and the numbers were never in doubt — plus 19 added tags
and **zero changed tags**, its own reason, and the digest of the generation it supersedes.

| unit | gen 1 (unchanged) | gen 2 (current) | rows | tags |
| --- | --- | --- | ---: | --- |
| `native-accuracy` | `ca8cc67e…` | `46870f3c…` | 24 → 24 | +19, 0 changed |
| `native-macro-f1` | `9b051e35…` | `b61cffe9…` | 24 → 24 | +19, 0 changed |
| `framework-accuracy` | `5a0419f8…` | `808c27a6…` | 24 → 24 | +19, 0 changed |

3 sent, 0 pending, no failures; reconciliation HTTP 200 with no missing units, no
accounting-only and no lake-only divergence. Read back through the canonical reader: all
three generation-2 rows are current, every tag round-trips, the metric rows are identical
to the accepted ones, and the reader's `assert_row_identity_tags` accepts all three. The
campaign's terminal rows went from 3 to 6 — **history kept**, and an independent read-only
query confirms the three generation-1 digests are byte-for-byte what CB-C recorded.

Then the point of the exercise, computed from the live rows through the reader: 72 rows,
**49** distinct occurrences, **23** double-counted rows now collapsed, 25 distinct row
identities, **0** value conflicts.

I remembered my own note that the CB-C writer is **not idempotent** — its campaign key
carries a timestamp, so a second run would open a second campaign and add a second set of
rows. That is why this is a supersede and not a rerun. `TerminalOutbox.supersede` only
operates on a *pending* envelope and these were long sent, so the successors went through
the same durable `O_EXCL` outbox and `_send_pending` path with `generation = 1 + stored`,
carrying `supersedes_generation` and `supersedes_terminal_sha256` explicitly. This tool
cannot double-write: a second run offers the same digest for the same generation and the
service returns the existing receipt.

Evidence: `docs/audits/evidence/cb04_row_identity_20260929/SUCCESSORS.json` and
`SUCCESSORS_DRY_RUN.json`.

## 5. BANKING77: the gate ran, the answer is no, and the score is not the deliverable

`tools/df_banking77_label_fit_20260929.py`. Nothing was downloaded, no weights were
loaded, no GPU was used, and no inference ran. The fit does not need them:
`sequence_budget` makes the option budget `head_max_len - option_tokens`, independent of
the news text, so the question is decided by the 77 label strings and the checkpoint's own
tokenizer.

**The label order, carried forward and checked rather than re-derived.** Read from
`docs/contracts/classification_populations.v1.json`, which pins mirror revision
`0fd18e25…`: the mirror's ids **are** case-insensitive alphabetical and are **not**
byte-sorted, and a naive `sorted()` map would mislabel **52 of the 77 classes (67.5%)** —
id 0 would become `Refund_not_showing_up` instead of `activate_my_card`.

**G1 — the shipped question builder refuses before any encoder is reached.**
`news_signal/question.py` declares `MAX_OPTIONS = 12`. Imported and called, not
reimplemented: 77 options with descriptions → `QUESTION_OPTION_COUNT`; 77 bare labels →
`QUESTION_OPTION_COUNT`; 13 → `QUESTION_OPTION_COUNT`; 12 → built, 550 characters. A
77-way BANKING77 question cannot be asked through the shipped provider today. Separately,
every option requires a non-empty description, and BANKING77 ships none, so any
description is a rendering choice that must be declared.

**G2 / G3 — the token arithmetic, with the checkpoint's own tokenizer.** Run on the
secondary worker where the digest-verified checkpoint sits: all **24** files matched the
retained `cb03_checkpoint_manifest.json`, including both tokenizer files; tokenizer vocab
50280. Per option: min 6, max 27, mean 10.61 tokens asked; **none** exceeds
`OPTION_TOKEN_LIMIT = 48`, so `options_truncated` is empty — the failure is collective,
not per-option. The instruction head is 12 tokens and is cut to the 8-token floor in every
case.

| rendering | budget | option tokens | option budget | fits | room for the message |
| --- | --- | ---: | ---: | --- | ---: |
| empty description (floor) | framework 512 / **192** | 706 | **−514** | **false** | **0** |
| empty description (floor) | checkpoint 1024 / **256** | 706 | **−450** | **false** | 306 |
| label as its own description | framework 512 / **192** | 894 | **−702** | **false** | **0** |
| label as its own description | checkpoint 1024 / **256** | 894 | **−638** | **false** | 118 |

The conclusion is stronger than the order anticipated: the hard-coded 512/192 is real and
is **not** the binding constraint. Even at the checkpoint's own declared 1024/256 the
option budget is short by 450–638 tokens. A 77-way choice does not fit this encoder's head
budget at all, and under 512 the customer message would be dropped entirely — a score
computed there would be a measurement of the label list, not of the text.

**The clamped temperature also fires, and is material here.** For k = 77 the pinned SDK
0.3.11 selects bucket `choice:11+`; the checkpoint ships `0.10058280825614929` for it;
`clamp_temperature` bounds are `[0.5, 5.0]` and the clamp **fires**, replacing it with
`0.5` — a ~5× change in logit scaling, with the SDK's own warning that the affected
confidences are uncalibrated. Harmless at four options, where the bucket is never
selected; material at seventy-seven. So even with the option budget repaired, a BANKING77
receipt could not carry NLL, Brier or ECE as calibrated.

**Named refusal, recorded as such:**
`BANKING77_SEVENTY_SEVEN_OPTIONS_DO_NOT_FIT_THE_PROVIDER_NO_SCORE_PRODUCED`. No receipt,
no warehouse row, no metric and no badge. A named refusal is not a completed family, and I
do not present it as one.

**What stays retained about the candidate, without substitution.** The selected candidate
remains `jinaai/jina-embeddings-v5-text-small` @ `46ed7da5…` at `0.914578` on the pinned
dataset revision. That number is a `PUBLISHED_REFERENCE` under a different protocol — MTEB
few-shot linear probing over embeddings, not a zero-shot typed choice — so under the
contract it is not comparable to anything a Laya choice head would produce, and it is not
our measurement. The two rows above it (`0.916656`, ~21 GB; `0.916136`, ~28 GB) **cannot be
admitted** on any host here; their cost and that blocker stay retained by name and **no
substitute is presented under their name**.

Evidence: `docs/audits/evidence/cb04_row_identity_20260929/BANKING77_LABEL_FIT.json`.

## 6. Resources, admission and what I did not touch

Capacity was **re-read live**, not taken from any earlier report
(`tools/df_host_capacity.py`, read-only, 2026-09-30T00:02Z):

- the preferred external-accelerator host's slice ceiling is 8.00 GiB with 7.00 GiB high
  and 0.15 GiB charged; its second GPU stays quarantined. **Ineligible, and not used.**
- the secondary worker: 30.58 GiB total, **20.32 GiB available**, slice `MemoryMax`
  **14.00 GiB**, `MemoryHigh` 12.00 GiB, slice `memory.current`
  **2 021 437 440 B (1.88 GiB)** with **zero** scopes running, GPU 0 idle with 15.56 GiB
  free and no compute processes, launcher present, memguard active. The residual is that
  1.88 GiB reading, **not** the 4.9–5.1 GiB of earlier reports and not the 2.69 GiB of the
  dictamen's snapshot; I did not treat any part of it as discardable, did not count shmem
  twice, did not reclaim, did not touch `/dev/shm` and did not move any ceiling.

Every child ran through the already-deployed `crispdm-run`, freshly admitted, nothing
reinstalled: `cb04-rowid-mutation` and `cb04-rowid-unit` at 1G, `cb04-rowid-store` and
`cb04-suites` at 2G, `cb04-successors` at 1G, `b77-*` probes at 1G, and the worker child
`b77-label-fit-20260929` at **3G / 15m with `-q`**, invoked by the launcher's absolute path
because a non-interactive ssh shell does not carry `~/.local/bin` on PATH. No declared cap
was shrunk to evade a refusal, no reservation was reduced, no ceiling was changed and
nobody was displaced. No service was started, stopped or restarted; no GPU work ran.

## 7. What is NOT done, refused, or not measured

- **NO_NEW_MODEL_MEASUREMENT.** No classifier ran in this lane, no dataset was downloaded
  or scored, and no new metric value exists. Every number about the warehouse is a property
  of stored rows; every number about BANKING77 is a token count and a budget.
- **No BANKING77 score, and none is owed until the provider can ask the question.** The
  refusal is named above. Three separate repairs would be needed first — the 12-option
  ceiling, an option budget that 77 options can fit (which 1024/256 does **not** provide),
  and an honest treatment of the clamped `choice:11+` temperature — and I have not made
  any of them: that is a provider change in `news-signal`, outside this order.
- **No badge, from anything.** No `BUSINESS_HELD_OUT` measurement exists; the AG News rows
  are a `PUBLIC_BENCHMARK` positively known to be in the training mix. The 19-prompt router
  corpus appears nowhere in this package — not as a population, not as a denominator, not
  as evidence — and it is not a classification benchmark.
- **No comparison across families.** Accuracy, macro-F1 and MAP remain three metrics and
  comparisons across them still refuse by name. MAP still has a key, a unit, an identity and
  a refusal, and no scorer.
- **The author's published `0.9525` is still not a warehouse row.** The successor contract
  removes the reason it could not be one safely, but writing it would be a new unit in a
  closed campaign, i.e. a new campaign. Not authorised here, not done.
- **Not re-verified by me:** the AG News reproduction itself, the parity attribution, and
  the retained per-row arrays. I checked that the stored rows are the retained receipts'
  projection; I did not re-measure the model.
- **The dictamen's F1–F3 are not mine** and are untouched: the supervisor's foreign-evidence
  gate, the producer/consumer schema mismatch and the framework/telemetry mismatch belong to
  agents A and B. The launcher's 9.5× undercount of a short child's retained peak is still
  unrepaired, so a retained peak remains a lower bound and never a footprint — which is why
  the capacity numbers above come from a live read.
- **No execution authority.** `authorises_broker_deployment` is false and
  `execution_authority` is `NONE` in the new contract, in every receipt and in every row
  written. Nothing here authorises a deployment, a campaign or a trade.

---

## 8. Report in the required shape

```
CB04-ROWID — per-row metric identity, and the BANKING77 label-fit gate
repo/branch/tip: predictor / satoshi/cb04-row-identity-and-banking77-20260929 / from 96d41960
files: app/classification_row_identity.py · docs/contracts/classification_row_identity.v1.json ·
       docs/CLASSIFICATION_METRIC_CONTRACT.md (aggregation rule corrected) ·
       tools/test_classification_row_identity.py · tools/test_classification_row_identity_warehouse.py ·
       tools/df_cb04_row_identity_mutation_probe_20260929.py ·
       tools/df_cb04_row_identity_successors_20260929.py ·
       tools/df_banking77_label_fit_20260929.py ·
       docs/audits/evidence/cb04_row_identity_20260929/{MUTATION_PROBE,SUCCESSORS_DRY_RUN,SUCCESSORS,BANKING77_LABEL_FIT}.json
       (app/classification_receipt.py and classification_metrics.v1.json: byte-identical, on purpose)
suites: row identity 29/29 · disposable store 7/7 (installed provider) · schema 56/56 ·
        corpus 22/22 · pipeline 12/12 · pre-existing warehouse end-to-end 13/13 ·
        mutation probe EVERY_MUTATION_IS_CAUGHT (13/5/4 of 21 killed)
acceptance: successors {"send":{"sent":3,"pending":0,"failures":{}},"reconciliation":{"http":200,
            "missing_units":[],"accounting_only":[],"lake_only":[]},"accepted_successors":3,
            "history_kept":true,"aggregate":{"rows_in":72,"rows_counted":49,
            "double_counted_rows_now_collapsed":23,"value_conflicts":[]}}
            banking77 {"all_77_labels_fit_untruncated":false,"answered":true,
            "the_shipped_question_builder_can_ask_them":false,"score_is_the_deliverable":false,
            "finding_instead_of_a_score":"THE_SEVENTY_SEVEN_LABELS_DO_NOT_FIT_NO_SCORE_IS_PRODUCED"}
what is NOT done / refused / not measured: no model ran, no dataset scored, no BANKING77 score,
            no badge, no cross-family comparison, the published reference still not a row,
            provider repairs (MAX_OPTIONS 12, option budget, choice:11+ clamp) not attempted,
            dictamen F1-F3 untouched, no execution authority anywhere.
```

— Satoshi, successor technical lead, 2026-09-29.
