# DR05 / DR06 — three claims narrowed: the lag clock, the prefix output, E3's dependency

**Authority.** The owner's grant of 2026-09-26, carrying DR05 and DR06 of
[`SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26`](../../handoffs/SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md)
and findings F5–F7 of the day review. **Satoshi, successor technical lead, 2026-09-26.**
**Nothing in this document is written, signed or attributed in the reviewer's name**, and no reviewer's
name appears on any artifact it produces.

**What this document does.** It narrows three claims I made too broadly today. Every change here makes a
statement *smaller*, and no change widens one:

| | claim as it stood | claim after this document |
|---|---|---|
| **§2** | the lag table restated under a "bias-corrected" estimator, on a series with a position deleted | recomputed with the offsets applied on the **retained clock** under a **finite-pair mask**; the two estimators named apart; the label withdrawn; the inversion kept and fenced |
| **§3** | `MOD-FROZEN-PREFIX` effectively discharged because the delivered CSV and the panel agree | parity of **inputs** is not the module. The **output** of the fixed prefix is materialized with five stages' state, clock, rows/split, shape and version, and proved four ways — and that is still **not delivery** |
| **§4** | `H-CORE` a prerequisite of **all** of `MOD-E3` | `H-CORE` conditions its **own** RL comparison. The financial reference and the weekly forecasting/RL protocol proceed on their own data, contracts and risks |

**Tooling.** [`tools/df_dr06_lag_clock.py`](../../../tools/df_dr06_lag_clock.py) ·
[`tests/test_dr06_lag_clock.py`](../../../tests/test_dr06_lag_clock.py) ·
[`tools/df_dr05_prefix_output.py`](../../../tools/df_dr05_prefix_output.py) ·
[`tests/test_dr05_prefix_output.py`](../../../tests/test_dr05_prefix_output.py) ·
[`DR05_DR06_20260926/LAG_TABLE_ON_THE_CLOCK.json`](../evidence/DR05_DR06_20260926/LAG_TABLE_ON_THE_CLOCK.json) ·
[`DR05_DR06_20260926/PREFIX_OUTPUT_MATERIALIZATION.json`](../evidence/DR05_DR06_20260926/PREFIX_OUTPUT_MATERIALIZATION.json).
**59 rules pass** (28 + 31). `check_plan` returns `PASS` with zero issues and its 34 tests pass; that is
documentary coverage, not scientific approval, and it is reported here as neither.

**How it ran, and what it did not do.** Worktree `predictor-dr056-20260926`, branch
`satoshi/dr05-dr06-corrections-20260926`, base `6820fcae`. CPU only, `CUDA_VISIBLE_DEVICES=''`, every job
through the shared launcher `crispdm-run -m 2G -n dr056`, anaconda env `trading-stack`.
**No model was fitted. No allocation was taken. No reserved split was opened. No host or service was
started, stopped or restarted. No run root was written to. No governance or warehouse contact was made.
No committed sample was overwritten. No heavy job was dispatched to the coordinator, and nothing was
launched on my own reading of available memory.** The one time the shared launcher **refused** an
admission — `QUEUED HOST_HEADROOM: 2.00G requested; 1.42G free (MemAvailable 12.18G − 3.00G desktop
reserve − 7.76G held by 2 live reservations)` — I waited for the reservations to clear and re-submitted
**at the same 2G cap**. I did not raise the cap, disable anything, clear a cache or touch another
process's memory. §5 records that episode as the only admission event of this work.

**The execution checkout.** Both tools record their own `code_identity` — revision, and whether the
worktree was byte-clean — so the pinned-checkout requirement of DR06 is a measured fact in the artifact
rather than a sentence here. The primary checkout's untracked files (the owner's presentation sources,
a PowerPoint lock file, evidence directories) were **not cleaned, not staged and not committed**; this
work never ran there. §5.3 gives the revision each artifact was produced at.

---

## 2. The lag table, recomputed on the retained clock

### 2.1 The defect both earlier versions share

Each earlier version drops the slice's single non-finite position and only then applies the lag offsets:

```python
v = Y[rows]; v = v[np.isfinite(v)]        # tools/df_e1_data_audit.py::autocorrelation:262
                                          # tools/df_rp49_rp64_audit.py::data_audit:690
... v[:-k] against v[k:]
```

After a position is deleted, an index separation of *k* is no longer a separation of *k* minutes. Every
pair that straddles the deletion is really *k+1* minutes apart and is counted as if it were *k*. **The
repair is not to delete more carefully. It is to leave the clock alone**: apply the offsets on the
retained one-minute grid, and admit a pair only when **both** of its legs are finite.

The support, re-derived: the retained train clock is **40 200 positions**, slice rows `[0, 40199]`, with
**one** non-finite position at slice row **30 590**. The deleting population's 40 199 rows *are* the
published support — the published table was measured on the clock with one position removed, and the
correction keeps it.

**The clock is verified before the offsets are trusted.** From the delivered panel's own
`timestamp_label` column over exactly those 40 200 rows (`panel_sha256 b3192c0b…`): first stamp
`2009-08-23T12:45:00`, last `2009-09-20T10:44:00`, **distinct steps observed = {60 s}**, strictly
increasing, **0** duplicated stamps. An index offset of *k* **is** *k* minutes here. Had it not been, the
repair would have had to carry the stamps rather than the indices, and the check says so.

### 2.2 The two estimators, named apart

DR06 requires that the two estimators not share a word. They do not, and the tool fails the run if they
ever come to:

| | name | what it computes |
|---|---|---|
| **A** | **autocorrelation function with one whole-series denominator** | numerator over the admitted offsets; denominator the centred sum of squares over **every** admitted position, one mean. The denominator does not shrink with *k* while the numerator does |
| **B** | **Pearson correlation over lag-truncated pairs** | each leg of the admitted pairs centred and scaled on its **own** pair count; every admitted pair weighted once |

Word sets: `{autocorrelation, function, with, one, whole, series, denominator}` and
`{pearson, correlation, over, lag, truncated, pairs}` — disjoint, asserted by
`_assert_names_share_no_word` and by `test_a_name_that_shares_a_word_is_rejected`.

**The label "bias-corrected" is withdrawn.** It named B as if it were A rescaled, and it is not. The
arithmetic is published beside both so the claim can be checked instead of believed, and the rule set
rejects any record that reuses the label on a column of its own (it may appear only as a quoted or
backticked citation of the artifact that carried it).

### 2.3 The recomputed table, beside the two superseded versions

| offset | ① **as published** (A, positions deleted) | ② **today's restatement** (B, positions deleted) | ③ **this recomputation** (B, clock + finite-pair mask) | ④ *diagnostic*: A × n/(n−k) | ⑤ *diagnostic*: A on the clock |
|---|---:|---:|---:|---:|---:|
| 1 min | 0.96334498 | 0.96336110 | **0.96336093** | 0.96336894 | 0.96334019 |
| 60 min (one hour) | 0.40300242 | 0.40328253 | **0.40326867** | 0.40360483 | 0.40298753 |
| 1 440 min (one day) | 0.31718580 | 0.33515716 | **0.33508484** | 0.32897010 | 0.31711215 |
| 10 080 min (one week) | 0.26559305 | 0.36686409 | **0.36660132** | 0.35447973 | 0.26540006 |

Admitted pairs, per offset — ③ against ②: **40 197 / 40 138 / 38 758 / 30 119** against
40 198 / 40 139 / 38 759 / 30 119. Full precision for every cell is in the evidence JSON.

**① re-derives exactly.** All four published values reproduce to 1·10⁻¹⁰ from the retained bytes, and the
estimator that produced them is called directly and asserted against to 1·10⁻¹² — so ① is not a number
this document reprints, it is one it recomputes.

### 2.4 The three differences, each explained separately

They are three different things and this table refuses to let one sentence cover them.

| | what changed | where it shows | size |
|---|---|---|---|
| **① → ②** | the **estimator**: a fixed whole-series denominator became two legs each scaled on their own *n−k* pairs | this is the **whole** of the day/week reordering | **+0.10127104** at a week (0.26559305 → 0.36686409); +0.01797136 at a day; +0.00028011 at an hour |
| **② → ③** | the **row population**: deleting a position became applying the offsets on the clock under a finite-pair mask | nowhere in the ordering; everywhere in what was counted | −1.67·10⁻⁷ · −1.39·10⁻⁵ · −7.23·10⁻⁵ · **−2.63·10⁻⁴**. Mislabelled pairs repaired: **1 / 60 / 1 440 / 9 609** |
| **① vs ④** | nothing — ④ is ① rescaled by n/(n−k), the thing "bias-corrected" claimed to be | the weekly cell | ④ = 0.35447973, ③ = 0.36660132, **difference 0.01212159**. The two are not the same estimator and the label was wrong |

**On ② → ③ being small.** It is small in magnitude and that is not the point. At a weekly offset,
**9 609 of the 30 119 admitted pairs were counted at the wrong offset** — 32% of the population. A
population that counts pairs at an offset they do not have is wrong whatever the size of the
consequence, and the size could not have been known before it was measured. The rule set proves the
mechanism on a case where it is not small: on a period-4 sawtooth with one position deleted, the masked
route returns **+1.0** at an offset of 200 and the deleting route returns **0.246**, because 150 of its
199 pairs read the wrong phase.

### 2.5 The inversion persists — and it belongs to the estimator, not to the clock

| | week vs day | inversion? |
|---|---|---|
| ① as published (A, deleted) | 0.26559305 < 0.31718580 | no |
| ⑤ A on the clock | 0.26540006 < 0.31711215 | **no** |
| ② B, deleted | 0.36686409 > 0.33515716 | yes |
| ③ **B on the clock** | **0.36660132 > 0.33508484** | **yes** |

**Kept, as ordered — and narrowed by what column ⑤ shows.** The inversion survives the clock correction,
so the finding stands. But it does **not** appear under the published autocorrelation function *even on
the corrected clock*. So the inversion is a statement **under one named estimator**, not an
estimator-free fact about the series, and the earlier restatement's framing — a bias corrected, a truth
revealed — was too strong in that respect too. Ordering under ③, strongest first: **one minute → one
hour → one week → one day.**

**The reading that survives every column unchanged:** the daily offset carries less marginal linear
dependence than the hourly one (0.33508484 < 0.40326867 under ③, and the same inequality under ①, ②
and ⑤). A daily context remains a hypothesis.

### 2.6 This table is not a window selector

Written into the artifact as `not_a_window_selector`, and enforced:

- **What it is.** An ordering of the **marginal linear** dependence between the target and its own past
  at four offsets, on the retained training support of one household panel.
- **What it is not.** A context-window selector. It ranks no candidate window. It measures no
  *conditional* contribution of a lag to a predictor. It says nothing about whether a weekly context
  improves a forecast. **No model was fitted to produce any number in it.**
- **The inversion specifically.** The week standing above the day is a fact about marginal linear
  dependence at two offsets. A prefix that includes a weekly offset must be justified by **measured
  skill** under the programme's own protocol, never by this ordering.

`validate` rejects a record that loses any of those three statements, and rejects a record whose rows or
inversion note read as a choice (`"therefore use a weekly context window"`, `"the optimal window
follows from this row"` — both are built as mutants and both are rejected).

**And the earlier version's own selector sentence is withdrawn.**
`SATOSHI_RP59_LAG_TABLE_AND_MATERIALIZATION_2026_09_26.md` §3 told the consuming module that "a temporal
prefix must be chosen on the bias-corrected column… If the module chooses a prefix on published
autocorrelations, it must use this column". That is one step from an automatic selector and it is
retracted here: the column is not a licence to choose, and a module that needs a prefix needs measured
skill.

### 2.7 What is superseded, and what is withdrawn

**Superseded, twice over — and both versions are carried in every artifact this writes:**

1. **As published.** The four-row autocorrelation list of `SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21`
   (RP59) and the `autocorrelation.autocorrelation_by_lag_minutes` block of
   `docs/audits/evidence/d3_k5_20260917/RP59/DATA_TARGET_PREPROCESSING_AUDIT.json`. Two defects: the
   estimator was not named, and the population deleted a position before the offsets.
2. **Today's restatement.** The `bias_corrected` column of
   `docs/audits/evidence/RP59_LAG_AND_SPLITS_20260926/LAG_TABLE_AND_SPLIT_MATERIALIZATION.json` and the
   table of `SATOSHI_RP59_LAG_TABLE_AND_MATERIALIZATION_2026_09_26.md` §1.2, at
   `satoshi/rp59-lag-table-restatement-20260926` tip `6820fcae`. Two defects: it corrected an estimator
   while keeping the compressed clock, and it named a Pearson correlation as a rescaled autocorrelation
   function.
3. Consequently also **finding 10** of `rp49_rp64_disposition_20260926`, whose 0.367 / 0.335 pair is the
   clock-compressed population's.

**What is withdrawn: one label, and no value.** Not a single number is removed or overwritten. Both
superseded columns are published side by side with the recomputed one, each with its estimator *and* its
row population named. `test_a_record_that_drops_either_superseded_version_is_rejected` builds the mutant
that drops each, and requires both to be rejected. Superseded evidence is never carried invisibly in
this corpus, and a silent correction would be the same defect pointed the other way.

---

## 3. `MOD-FROZEN-PREFIX`: the prefix **output**, materialized and proved

### 3.1 Parity of inputs was never the module

Agreement between the delivered CSV and the prepared panel is parity of the prefix's **inputs**. It says
nothing about the prefix's **output**, which is what a consumer downstream of the fusion reads. The
module's deliverable is a versioned derived dataset *at that output*, and until that dataset exists there
is nothing for `H-CORE` to consume. That sentence is written into the artifact as
`what_parity_of_inputs_is_not` and `validate` rejects a record that omits it.

`MOD-FROZEN-PREFIX` stays **UNBLOCKED and NOT DELIVERED**. What follows is development work, materialized
and proved; §3.6 says why it is still not delivery.

### 3.2 The prefix, and where it stops

Read off the graph of the retained fitted cell, not assumed. The prefix **ends at `fusion_seq`**; the
core (`core_tcn1…5`), the head and the persistence skip are downstream of the cut and are **not**
materialized — they are what `H-CORE` varies. `prefix_contains_nothing_downstream` found **zero**
downstream layers inside the prefix, and the rule set builds a prefix cut one layer too late and
requires it to be flagged.

Donor: run root `e1_household_successor_v3`, cell **`R1_s1`** — detector imported from the auto-encoder
`ae_s1` and frozen (weight change 0.0 on every detector layer in the retained record), adapters fitted in
the cell. Store version **`dr05.prefix.57970b5e4d0aa302`**.

### 3.3 The five stages, each with its learned state, clock, rows/split, shape and version

| stage | kind | learned state | shape |
|---|---|---|---|
| **preprocessing** | `LEARNED` | the stored scaler (mean, sd), fitted on **train windows only**, 2 404 860 rows; digest over the 7+7 values | `[·, 7] → [·, 7]` |
| **groups** | `CONSTANT_BY_DESIGN` | the held-constant physical assignment `[0,1,0,2,2,2,0]` → `g0/g1/g2_select`; digest over the assignment. **Not learned, and declared not learned** | `[·, 60, 7] → [·,60,3] / [·,60,1] / [·,60,3]` |
| **detector** | `LEARNED` | `g{0,1,2}_det1_conv`, `_det1_proj`, `_det2_conv` — two residual causal Conv1D blocks per branch, 16 filters, kernel 3, dilation 1, elu; weights digest | per branch `[·,60,c] → [·,60,16]` |
| **adapter** | `LEARNED` | `g{0,1,2}_adapt`, TimeDistributed Dense(8), linear; weights digest | `[·,60,16] → [·,60,8]` |
| **fusion** | `CONFIGURATION` | `fusion_seq`, channel concatenation on axis 2 in sorted group order; digest over `{axis, order}`. No weights: the prefix's last stage | `[·,60,7] → [·,60,24]` |

Every stage carries, in the same record: the **clock** (panel rows `[1412361, 1462761)`,
`2009-08-23T12:45:00` → `2009-09-27T12:44:00`, observed step {60 s}, **timezone UNKNOWN** — the producer
never declared one, and it is carried, not resolved); its **rows and split**; its **shape**; and the
store **version**. `validate` rejects a stage missing any of those five, and a stage that declares
`LEARNED` for the fusion or `CONFIGURATION` for the detector.

**What was materialized**, with the reserve refused:

| split | origins | input rows (slice) | input rows (panel) | array | bytes | sha256 |
|---|---:|---|---|---|---:|---|
| train | 40 080 | `[0, 40199]` | `[1412361, 1452560]` | `prefix_output_train.npy` `(40080, 60, 24)` f32 | 230 860 928 | `8a240716…` |
| validation | 10 020 | `[40261, 50339]` | `[1452622, 1462700]` | `prefix_output_validation.npy` `(10020, 60, 24)` f32 | 57 715 328 | `0e6dd5bb…` |
| **test** | **0** | — | — | **none** | — | **`REFUSED_UNMATERIALIZED_BY_DESIGN`** |

Store: `$HOME/.local/state/crispdm-data-foundation/dr05_prefix_store_20260926`, 288.6 MB, **outside the
repository**, with `MANIFEST.json` binding version, code identity, clock, splits, arrays and stages.
Streamed to disk through a memmap, so the tensor is never resident: the store is larger than the cap the
job ran under, and an artifact that exists only when it fits in RAM is not an artifact.

### 3.4 The four proofs

| proof | what it did | result |
|---|---|---|
| **direct against cache** | 64 origins per split resampled and pushed through the prefix again; **92 160 elements** per split compared **element by element and on the bytes**, no tolerance | **VERIFIED** — identical, max absolute difference **0.0**, `bytes_identical` true, both splits |
| **reload in a fresh process** | a **separate interpreter**, given only the store directory and no value from the writing process, recomputed every file digest, both origin digests, the shapes, and the first and last window digests | **VERIFIED** — `all_digests_match` true; `6f95247b…` / `9ff12182…` (train), `38a0bc97…` / `40ecc325…` (validation) |
| **mutation of weights and of state** | four perturbations, one at a time: a detector scalar (+1e-2), an adapter scalar (+1e-2), the scaler state (sd[0] × 1.01, applied by inverting the stored scaling and re-applying the perturbed one), and the group assignment (two channels exchanged) | **VERIFIED** — each **changed the output** (max |Δ| 0.01385 / 0.02970 / 0.01713) **and** the bound identity; every weight case restored **bit for bit**; nothing was written to any run root |
| **causality** | the reach **measured** by perturbation, row by row, on the delivered bytes; then a **future** row perturbed in the panel | **VERIFIED** — measured reach **5** = declared branch reach 5; rows 0–4 back move the last position and **no** earlier row does; panel row 40 199 leaves the window that ends at 40 198 untouched **while moving the window that contains it** |

On the group-assignment case: a different grouping is a different **graph**, not a different weight, so
the output is deliberately **not** recomputed for it and the record says so (`output_changed: null`). What
must hold is that the version **binds** the assignment, so a store built under another grouping cannot be
mistaken for this one. It does.

### 3.5 One property measured rather than assumed

The store is **row-addressable**: the same panel row receives the same representation in every window
that places it at or beyond the reach. **1 393 positions compared across four window shifts, largest
absolute discrepancy exactly 0.0.** This is measured, not assumed, because a store that assumed it and
was wrong would be a silent defect.

Two consistency facts fall out of it and out of §2's clock, and they agree: the slice's **one** non-finite
row is 30 590; a detector reach of 5 contaminates rows 30 590–30 594; and every origin whose window
places one of those rows at or beyond the reach is inside the withdrawn set `[30590, 30649]`. The
withdrawal arithmetic of the preparation and the causal reach of the prefix describe the same rows.

**The reserve was not opened.** Every materialized row lies inside the consumed slice `[1412361,
1462761)`; the donor cell's `exposure` is `NO_TEST_ACCESS`; the `test` split's materialization check is
recorded as **null**, which the verdict function refuses — so it can never read as a pass. Its
**absence** is verified; its materialization is not, because there is none. `MOD-CONF` owns the reserve.

### 3.6 What this does **not** deliver

1. **Any claim that this is the RIGHT prefix.** The donor cell was **named, not chosen on evidence**.
   Which learned state `MOD-FROZEN-PREFIX` should freeze is a design decision the module still owes.
2. **A second family or a second panel.** One panel, one donor, two splits. A versioned derived dataset
   for the programme is more than that.
3. **Anything about the reserve** beyond its absence.
4. **Any `H-CORE` comparison.** The core is downstream of the cut and nothing here fitted one.

So the module's status moves to `IN_PROGRESS` with those four items named, and **not** to delivered. The
reserve was not opened to build any of this, and no allocation was requested for it.

---

## 4. `MOD-E3`: `H-CORE` removed as a blocker for **all** of E3

### 4.1 The block was mine and it was too wide

The `MOD-E3` ruling I wrote added `H-CORE` as a **global** prerequisite. The programme contract names
`MOD-E1` and `BUSINESS-CONTRACT`, and the task's own `depends_on` has only those two. `H-CORE` may
condition the RL comparison it is the counterpart of; it may not hold the financial reference, the
weekly protocol or the integration behind it. **Withdrawn as a prerequisite of the whole module.**

### 4.2 What keeps its real dependencies, and what proceeds

**Keeps them, unloosened.** `H-CORE`'s own reinforcement-learning comparisons. Nothing in this document
touches `MOD-CORE-PRETRAIN`'s ruling, its prerequisites, or the separate question of that protocol's
resolution (another lane's work today).

**Proceeds now, on its own data, contracts and risks:**

| lane | state on the artifacts | eligible next work, needing no fit and no `H-CORE` |
|---|---|---|
| the **weekly forecasting/RL protocol** | `tools/e3_weekly_controller.py`, `e3_weekly_runtime.py`, `e3_weekly_strategy.py` are implemented and exercised by 24 rules across three files; they run **fit-free** (`ConstantModel`, `ThresholdModel`), refuse a short execution latency before any bar, refuse `SHORT`, and read **no** household data | extend the protocol rules to a late release and to weekly operation — the existing files' own scope note says they validate neither; keep them on synthetic bars |
| the **financial reference** | `tools/df_fin_task.py` / `df_fin_runner.py` / `df_fin_loss_opt_design.py` are executable and the design is sealed; the delivery is a **governed ranged download** and **zero bytes have ever been delivered** | inspect the resource's schema and row count without a fit; establish the producing revision of the retained bytes; draft the resource contract **for the operator**; select candidate reference methods |
| **`BUSINESS-CONTRACT`** | `DESIGNED`, not `EXECUTED`: its content is prose in `13C` + `13D` and **no sealed artifact exists** | seal it as an artifact — observations, targets, actions, availability, weekly cycle, cost and the data gaps — which needs no fit, no delivery and no `H-CORE` |

### 4.3 The hard rule, and what the financial lane now needs

> **Household electricity is never a substitute for financial validation.** No electricity result may be
> carried across the domain gap, and **no runner may be written that avoids a missing financial
> dependency.**

Named, because DR05 requires naming rather than working around:

1. **An operator-declared resource availability contract** for
   `financial_files:market_data/forex/g10/eurusd/1h.parquet`. The files lake refuses a ranged delivery
   without it (`http 422 resource availability contract required`), `bytes_read: 0`,
   `reserve_read: false`. **This is the single critical-path item, and it is orthogonal to `H-CORE`.**
2. **Available time** for that resource — absent: no producer record states a publication time.
3. **Maximum completion lag** — absent, and it must not be assumed to be zero.
4. **The timezone of the labels** — disputed, and the producing revision of the retained bytes
   unestablished. Three of the resource contract's six fields have no evidence, and two of those three
   are exactly the ones a causal claim depends on.
5. **Gap semantics** — unestablished.
6. **A matched financial reference method** for `fx.eurusd.1h.FIN-LOSS-OPT` — `NOT_COMPARABLE` until one
   is re-executed under the contract. Nothing in the electricity work supplies one.
7. **`BUSINESS-CONTRACT` as an executed artifact**, with its own gap list enumerated: funding/swap
   (declared zero **with that statement**, or the owner supplies the schedule), macro-series availability
   delays, latency/queue, venue, monetary capital, lot minimums, SLA, the asset universe, real exposure
   limits and the operating retraining-and-release schedule.

### 4.4 What stays blocked, and why

`MOD-E3`'s **own comparison** remains blocked, on its own reasons and not on `H-CORE`:

- no governed forecasting result exists on the domain it trades — every retained E1 measurement is a
  household-electricity panel;
- the legacy price-series route is closed **permanently**: 114 tables `CAUSALITY_UNVERIFIED` with both
  lineages `UNBOUND`, because the producer of the decomposed inputs is not in this repository —
  `NOT_DISCHARGEABLE_BY_ARTIFACTS`, and no work here reaches it;
- `BUSINESS-CONTRACT` is `DESIGNED`, not `EXECUTED`.

One correction to my own earlier framing while I am here: the flagged-table rate is **85 of 113
measurable** tables (75%), not 85 of 167 — 54 carry no reconstructable predictions and were never tested
for the flag. That is a worse finding than the sentence it replaces, not a better one, and it does not
weaken the disposition: none of the 167 is usable as a benchmark.

### 4.5 One cross-domain carry, named rather than removed

`FIN_LOSS_OPT_DESIGN_2026_09_21.md` marks three **software** acceptance rows green on household evidence
with their financial legs held open as `xfail(strict)`, and uses the household factorial's cost as a
**budget projection** for the compact financial receiver. That is a cost projection and a software
acceptance, not a scientific claim, and both are fenced in both directions in the source. I name it here
so it is on the record and cannot later be read as a result. One structural hazard worth closing in the
lane that owns it: `tools/df_e1_governed.py:254-255` defaults `--lake public_panels` and `--resource
uci_235_individual_household_power/panel.parquet`. The financial runner always passes the sealed design's
own lake and resource, so that path is never reached from it — but a hand-driven governed acquisition for
a financial unit would silently fetch the electricity panel.

---

## 5. Costs, the one admission event, refusals, and limits

### 5.1 Cost

CPU only, no GPU, no allocation. The whole of §2 is four correlations on a 40 200-point series plus one
timestamp column read; §3 is two forward passes of an 8 127-parameter model's prefix over 50 100 windows,
streamed to disk, plus the four proofs. Every job ran under `crispdm-run -m 2G -n dr056` and finished
inside its wall. The store is 288.6 MB on `/home` (134 GB free at the time), written outside the
repository.

### 5.2 The one admission event

| when | what | outcome |
|---|---|---|
| during verification | `crispdm-run -m 2G` for the two rule sets and `check_plan` | **REFUSED** — `QUEUED HOST_HEADROOM: 2.00G requested; 1.42G free (MemAvailable 12.18G − 3.00G desktop reserve − 7.76G held by 2 live reservations)`; nothing was started, and no limit, slice or kernel setting was changed |
| after the reservations cleared | the same three jobs, **the same 2G cap** | admitted; 59 rules pass, `check_plan` `PASS` with zero issues, its 34 tests pass |

I did not raise the cap, disable `oomd`, enlarge a ceiling, clear a cache, empty swap or touch another
process. The shared launcher's new reservation accounting is what refused me, and waiting was the correct
response to it.

### 5.3 Identity of the execution checkout

Both artifacts carry their own `code_identity` — the revision, and whether the worktree was byte-clean —
so the pinned-checkout requirement is a field in the evidence rather than a sentence here, and §6 names
the revision each was finally produced at. The primary checkout was not used, not cleaned, not staged and
not committed: the owner's untracked presentation sources, lock file and evidence directories are exactly
as they were.

### 5.4 Six refusals

1. **I refused to delete positions and call the result a clock.** The offsets are applied on the retained
   grid under a finite-pair mask, and the clock is verified from the panel's own stamps before the
   offsets are trusted.
2. **I refused the label "bias-corrected".** It asserted an equivalence that does not hold, and the
   arithmetic that disproves it is published at 0.01212159.
3. **I refused to let the table select a window.** The non-selector note is in the artifact, the rule set
   rejects a record that loses it or that reads as a choice, and the earlier version's own
   "must choose a prefix on this column" sentence is retracted.
4. **I refused to report `MOD-FROZEN-PREFIX` as delivered.** Four items are named as still owed, and the
   materialization's own record lists what it does not deliver.
5. **I refused to open the reserve** to improve any of it, and refused to let the `test` split's absent
   materialization read as a passed check.
6. **I refused to widen the E3 correction into a licence.** `H-CORE` keeps its own dependencies, the
   financial lane's missing dependencies are named rather than routed around, and no electricity result
   crosses the domain gap.

### 5.5 Limits

- **No new measurement of any model.** §2 is reanalysis of retained bytes; §3 materializes a
  representation from weights that were already fitted; §4 changes a dependency, not a result.
- §2 is one panel's retained **training** support. Nothing in it speaks about the reserve, a second
  family, or measured skill at any offset.
- §3 is one panel, one donor cell, two splits, and float32 storage. The proofs are exact where they claim
  to be exact and the one property that is measured rather than proved (§3.5) is labelled as measured.
- §4's inventory of the financial lane's missing dependencies is read from retained designs, sealed
  records and service responses. It does not establish that closing them is sufficient, only that they
  are open.
- `check_plan`'s `PASS` is documentary coverage. It validates no scientific evidence and replaces no
  seal.

---

*Satoshi, successor technical lead. 2026-09-26, under the owner's grant of 2026-09-26.*
