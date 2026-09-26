# RP59 — the lag table restated, and materialization verified per split

**Authority.** The owner's grant of 2026-09-26. Satoshi, successor technical lead, 2026-09-26. This
document discharges the two repairs [`SATOSHI_RP57_RP64_DISPOSITION_2026_09_26`](SATOSHI_RP57_RP64_DISPOSITION_2026_09_26.md)
§5.2/§6 and [`SATOSHI_RP49_RP56_DISPOSITION_2026_09_26`](SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md)
§9 name as `MOD-FROZEN-PREFIX`'s **first two items**: restate the autocorrelation table under both
estimators, and verify governed sequence materialization **per split**. **Nothing here is signed,
quoted or attributed to Musashi**, and no reviewer's name appears on it.

**Why these two first, and in this order.** The lag table is not merely imprecise, it is *actively
misleading*, and the module about to consume it is the one whose whole job is choosing which temporal
prefix to freeze. A module that reads the published four-row list would rank the weekly grain last.
Under the corrected estimator it ranks **third of four, above the daily grain**. That is the repair
that had to come before the module starts, not after.

Worktree `predictor-lagtable-20260926`, branch `satoshi/rp59-lag-table-restatement-20260926`, base
`a84a913c`. CPU only, `CUDA_VISIBLE_DEVICES=''`, every job under
`crispdm-run -m 3G -t 600 -n lagtbl`, anaconda env `trading-stack` (Python 3.12.13). **No model was
fitted, no training was run, no allocation was taken, no host or service was touched, no governance or
warehouse contact was made, no run root was written to, no reserved split was opened, and no committed
sample was overwritten.** Two other agents were running on this machine throughout; `MemAvailable` was
read before each launch and never fell below 10.6 GiB.

**The governing rule of this document.** A check that could not run is a **refusal by name**, never a
pass. That rule is not prose here: `tools/df_rp59_lag_and_splits.py::_verdict` refuses any split whose
check set carries a `None` or is empty, and `tests/test_rp59_lag_and_splits.py` builds the mutants — a
dropped split, an emptied check set, a `None` check inside a `VERIFIED` split — and requires each to be
rejected.

**Tooling.** [`tools/df_rp59_lag_and_splits.py`](../../../tools/df_rp59_lag_and_splits.py) ·
[`tests/test_rp59_lag_and_splits.py`](../../../tests/test_rp59_lag_and_splits.py) ·
[`RP59_LAG_AND_SPLITS_20260926/LAG_TABLE_AND_SPLIT_MATERIALIZATION.json`](../evidence/RP59_LAG_AND_SPLITS_20260926/LAG_TABLE_AND_SPLIT_MATERIALIZATION.json).
**No estimator is implemented a third time.** The as-published column is produced by the tool that
produced the published table, `tools/df_e1_data_audit.py::autocorrelation`, called directly; the
bias-corrected column is the identical expression `tools/df_rp49_rp64_audit.py::data_audit` already
recomputes. A rule asserts the first of those to zero tolerance against the estimator's own output, so
a private reimplementation cannot creep in later.

---

## 1. Repair one — the lag table, restated under both estimators

### 1.1 What was published, and under what

[`SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21`](SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md) (RP59)
reads: *"Train autocorrelation is 0.963 at one minute, 0.403 at sixty and 0.317 at a day: the daily lag
carries **less** linear structure than the hour, so a daily context is a hypothesis, not a certainty."*
The four values live in
[`RP59/DATA_TARGET_PREPROCESSING_AUDIT.json`](../evidence/d3_k5_20260917/RP59/DATA_TARGET_PREPROCESSING_AUDIT.json)
under `autocorrelation.autocorrelation_by_lag_minutes`, measured on **40 199 consumed train rows** of
the household panel — the one-minute grid measured in the companion disposition over **50 400 rows with
zero missing minutes and zero duplicated stamps**.

All four re-derive here **exactly, to 1·10⁻¹⁰**, and the support re-derives too (40 199 rows). They are
not wrong. They are computed under one specific estimator and printed without naming it.

### 1.2 The restated table

| lag | as published (biased ACF) | shrinkage (n−k)/n | lag-truncated Pearson | published ÷ corrected |
|---:|---:|---:|---:|---:|
| 1 min | 0.963345 | 0.999975 | 0.963361 | 0.999983 |
| 60 min (one hour) | 0.403002 | 0.998507 | 0.403283 | 0.999305 |
| 1 440 min (one day) | **0.317186** | 0.964178 | **0.335157** | 0.946379 |
| 10 080 min (one week) | **0.265593** | **0.749247** | **0.366864** | **0.723955** |

Full precision, both columns, in the evidence JSON. The two estimators, named:

- **as published — biased ACF.** `Σ(x_t−x̄)(x_{t+k}−x̄) / Σ(x_t−x̄)²`: one **fixed whole-series
  denominator** over all *n* centred rows, while the numerator has only *n−k* products. Textbook, and
  the estimator `tools/df_e1_data_audit.py::autocorrelation` implements.
- **bias-corrected — lag-truncated Pearson.** `corrcoef(x[:-k], x[k:])`: each of the *n−k* overlapping
  pairs weighted once, each leg centred and scaled on its own *n−k* rows.

**Why the order flips, visible in one column.** The biased estimator shrinks by `(n−k)/n`. At
n = 40 199 that factor is 0.9985 at an hour — invisible — and **0.7492 at a week**, because a weekly lag
discards a quarter of the series. The weekly value is therefore multiplied by roughly three-quarters
before it is printed. The **shrinkage column is the whole explanation**, which is why it is published
beside the numbers rather than described.

Ranking by strength:

| | order, strongest first |
|---|---|
| as published | 1 min → 1 hour → **1 day → 1 week** |
| bias-corrected | 1 min → 1 hour → **1 week → 1 day** |

### 1.3 Which reading survives, and which reverses

| reading | as published | bias-corrected | verdict |
|---|---|---|---|
| the daily lag carries **less** linear structure than the hour, so a daily context is a hypothesis and not a certainty | 0.317186 < 0.403002 | **0.335157 < 0.403283** | **SURVIVES** |
| the four values **decay monotonically** with the lag, so the weekly grain carries less linear structure than the daily one | 0.265593 < 0.317186 | **0.366864 > 0.335157** | **REVERSES** |

The headline sentence of RP59 stands under either estimator, and stands for the reason it gave. The
**table's apparent decay does not**: the weekly value as published sits **27.6% below** its corrected
value (equivalently, the corrected value is **38.1% above** the published one — the same gap read the
other way, and both framings are printed in the evidence so neither can be quoted as a correction of
the other; the disposition's "understated by 27%" is the first of them).

### 1.4 What this supersedes, and what it withdraws

**Supersedes.** The four-row autocorrelation list of `SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21`
(RP59), and the `autocorrelation.autocorrelation_by_lag_minutes` block of
`RP59/DATA_TARGET_PREPROCESSING_AUDIT.json`, **as a ranking of temporal grains**.

**Why.** The published list is correct arithmetic on its stated support under an estimator whose
shrinkage grows with the lag, and it was printed as a decay without naming that estimator. Corrected
for its own bias, the weekly lag rises above the daily one and the printed order of the last two rows
is reversed.

**What is withdrawn: nothing.** No published value is wrong on its own support, and **not one is
removed or overwritten**. The as-published column is retained in full, labelled as published, in this
document, in the evidence JSON, and in the rule set — **this corpus never carries superseded evidence
invisibly**, and a restatement that quietly replaced four numbers would be the same defect in the other
direction. `tests/test_rp59_lag_and_splits.py::test_a_table_that_drops_either_column_or_the_note_is_rejected`
builds the mutant that drops each column, each estimator name, the shrinkage, the two readings and the
supersession note, and requires every one to be rejected.

**What neither column licenses.** These are **linear autocorrelations of the target**, not measured
skill. Neither column says a weekly context improves a forecast, and no run here tested one. The
restatement changes a ranking a module would read; it does not create a result.

---

## 2. Repair two — materialization verified per split, from the bytes

### 2.1 What was verified, and how the declaration was obtained

The audit required that *what was materialized for each split is checked rather than assumed*. Nothing
below is read from a field that asserts it:

| | |
|---|---|
| **declared** | the split's row block and origin range **re-derived** — the panel's own row count (2 075 259, from the delivered parquet), the retained family split rule (`tools/df_e1_tasks.py::SPLIT_FRACTIONS` = 0.7 / 0.15 / 0.15 and its edge rule), the design's window 60, horizon 60 and purge 120, and the within-slice fractions 0.8 / 0.2 |
| **materialized** | what `DATA.npz` and the auto-encoder cells' `arrays.npz` actually hold |
| **equality** | the split's own rows read out of the **delivered panel parquet** and compared element by element against the materialized target, and the scaled inputs inverted through the stored scaler back to the panel's columns |
| **exclusivity** | origin sets **and touched-row footprints** (first window row → last label row), pairwise over every materialized split |
| **counts** | against the declaration's enumerator, with the admissibility arithmetic re-derived from the non-finite rows |
| **purge** | the boundary the design **declares** against the boundary **observed** in the files |

Identity first: `DATA.npz` digests to `70485ac9d1a41ac8…`, which **is** the declared `data_sha256`; the
delivered panel on disk digests to `b3192c0bcb117b2e…`, which is the delivery's own digest and the
`panel_sha256` the preparation recorded. The verified bytes are the audited bytes.

**The declared blocks, re-derived:**

| | rows |
|---|---|
| family train / validation / test edges | `[0, 1452681)` / `[1452681, 1763969)` / `[1763969, 2075259)` |
| consumed slice | `[1412361, 1462761)` — 50 400 rows: the family train tail (28 days) plus the family validation head (7 days) |
| dev train block | slice `[0, 40320)` = panel `[1412361, 1452681)` |
| dev validation block | slice `[40320, 50400)` = panel `[1452681, 1462761)` |

**And a fact worth stating separately: the within-slice 0.8 boundary IS the family's own train /
validation edge.** Panel row 1412361 + 40320 = 1452681, exactly `int(2075259 × 0.7)`. The development
split is not a second, private cut placed inside one side of the family's cut — it reproduces it. That
is verified from the panel's row count, not read from the design's note.

### 2.2 Per split

| split | verdict | declared | materialized | checks |
|---|---|---|---|---|
| **train** | **VERIFIED** | origins 59 → 40 199 inclusive = **40 141** declared (re-derived as `o ∈ [w−1, T−1−purge]`, T = 40 320); enumerator 40 141 → admissible 40 081 → targets valid **40 080** | **40 080** origins, 59 → 40 199, rows touched `[0, 40259]`; withdrawn **60** for a non-finite input row and **1** for a non-finite label (the slice's single non-finite row is 30 590) | 8 / 8 |
| **validation** | **VERIFIED** | origins 40 320 → 50 339 inclusive = **10 020** (re-derived as `o ∈ [T, n−1−h]`); enumerator 10 020, admissible 10 020 | **10 020** origins, rows touched `[40261, 50399]`; nothing withdrawn | 8 / 8 |
| **pretrain_train** | **VERIFIED** | the declared head of the dev train origins: 40 080 − 6 012 − 120 = **33 948** | **33 948** in every retained AE cell (`ae_s1`, `ae_s2`, `ae_s3`, `pilot_ae`), **array-equal** to that head | 4 / 4 |
| **pretrain_internal_validation** | **VERIFIED** | the declared purged tail, `int(40080 × 0.15)` = **6 012**, purge `w+h` = 120 | **6 012** in every AE cell, array-equal to that tail, observed origin gap **121**, and its last label row 40 259 precedes the validation split's first window row 40 261 | 4 / 4 |
| **test** | **REFUSED_UNMATERIALIZED_BY_DESIGN** | panel `[1763969, 2075259)` — 311 290 rows, 298 224 usable windows declared | **nothing.** 0 origins, 0 rows | materialization: **not checkable** |

The withdrawal arithmetic re-derives exactly and is worth spelling out, because it is the kind of count
that usually has to be believed: the slice holds **one** non-finite row (30 590). A window ending at
origin *o* covers rows `[o−59, o]`, so that row sits inside **60** windows; it is the label row
(`o+60`) of exactly **one** further origin. 40 141 − 61 = **40 080**. The inputs also invert: the
scaled slice, multiplied back by the stored scaler, reproduces the panel's own target column to
**2.17·10⁻⁷ kW** on the train split and **2.16·10⁻⁷ kW** on the validation split — float32 storage, not
a second transformation.

**The refusal, by name.** `test`: **REFUSED — nothing was materialized for the test split, so there is
no materialization to verify.** This verdict is never reported as a passed check, and the tool cannot
report it as one: the split's materialization check is recorded as `None`, which `_verdict` refuses. Its
**absence** is verified instead, and that much is real: the consumed slice ends at panel row 1462761,
**301 208 rows before** the declared test block begins; **no retained array in any of the 15 scanned
cells carries a `test` population**; all 15 cells declare `exposure: NO_TEST_ACCESS`; the design's own
note declares the test rows never read. What cannot be verified is a materialization that does not
exist, and that is the honest limit — the reserve stays reserved, and this document does not open it to
say so.

### 2.3 No split's rows appear in another

Pairwise, on origins **and** on touched rows:

| pair | shared origins | shared touched rows | |
|---|---:|---:|---|
| train ↔ validation | **0** | **0** | disjoint |
| pretrain_train ↔ pretrain_internal_validation | **0** | **0** | disjoint |
| pretrain_train ↔ validation | **0** | **0** | disjoint |
| pretrain_internal_validation ↔ validation | **0** | **0** | disjoint |
| pretrain_train ↔ train | 33 948 | 34 128 | **declared nesting** — the pre-training splits are subsets of the dev train origins by design |
| pretrain_internal_validation ↔ train | 6 012 | 6 131 | **declared nesting**, same reason |

The two nested pairs are named as nesting and not counted as disjointness: what must be disjoint is
each pre-training split from the **supervised validation** split, and both are, on rows as well as on
origins. Touched-row footprints: pretrain_train `[0, 34127]`, pretrain_internal_validation
`[34129, 40259]`, train `[0, 40259]`, validation `[40261, 50399]`.

### 2.4 The declared purge boundary is the observed one — the check the artifact left null

`RP59/DATA_TARGET_PREPROCESSING_AUDIT.json` reports **`purge_declared: null`**. That is not an absent
declaration, it is a **lookup in the wrong file**: `df_e1_data_audit.split_step` searches `DATA.json`
for a `purge` key, and `DATA.json` does not carry one. The declaration lives in `DESIGN.json`, twice
(`task.purge` = **120** and `dev_subpartition.purge_between_splits` = **120**, in agreement). So the
round measured the boundary and never bound it to the declaration. Bound here:

| | |
|---|---|
| declared purge | **120** (both places agree) |
| observed origin gap | **121** = 40 320 − 40 199, i.e. the declared purge on inclusive endpoints |
| train rows touched | `[0, 40259]` |
| validation rows touched | `[40261, 50399]` |
| rows touched by neither | **1** — row 40 260 |

The three numbers are one fact, not three: the purge is a boundary on **origins**, so a declared purge
of 120 appears as an inclusive origin gap of 121 and leaves exactly one row (40 260) read by neither
split. The last train **label** row is 40 259; the first validation **window** row is 40 261. No row is
read by both splits, and the boundary declared is the boundary observed.

### 2.5 The verdicts carry to the phase-1 run root

`e1_phase1_v1b/DATA.npz` digests to `70485ac9d1a41ac8…` — **byte-identical** to the audited root's
prepared data. Identical prepared bytes are identical splits, so every per-split verdict above holds
for the phase-1 root without recomputation. This is the one extension the document makes, and it rests
on a digest rather than on a claim that the two runs "used the same rows".

---

## 3. What `MOD-FROZEN-PREFIX` may now rely on

Its deliverable is a *versioned derived dataset with direct/cache parity, temporal support and complete
prefix lineage*. Explicitly, as of this document:

**It may rely on:**

1. **The bias-corrected lag column as its ranking of temporal grains** — §1.2. Hour 0.403283, week
   0.366864, day 0.335157, minute 0.963361, on 40 199 consumed train rows. The **week ranks above the
   day**; the published list ranked it last. If the module chooses a prefix on published
   autocorrelations, it must use this column and cite the shrinkage that separates the two.
2. **The surviving reading**: the daily lag carries less linear structure than the hour (0.335157 <
   0.403283), so a daily context remains a hypothesis. Unchanged by the correction.
3. **Per-split materialization for `train` and `validation`, verified from the delivered bytes** —
   §2.2: the materialized origins are exactly the admissible declared origins, the counts match the
   declaration with the withdrawal arithmetic re-derived, the target rows **are** the panel rows the
   split declares, and the scaled inputs invert to the panel's own columns to 2·10⁻⁷ kW.
4. **Exclusivity** — §2.3: train and validation share no origin and no row; both pre-training splits
   are disjoint from the supervised validation split on rows as well as origins; the only overlaps are
   the two declared nestings inside `train`.
5. **The purge boundary as declared** — §2.4: 120, observed as an inclusive origin gap of 121 with one
   row read by neither split. The prefix lineage may state the purge as a fact now, not as a plan.
6. **The split geometry as reproducible arithmetic** — §2.1: every block and origin range above
   re-derives from the panel's row count, the retained family split rule and the design's
   window/horizon/purge. The module can regenerate the same splits without reading a stored origin
   array, and the within-slice boundary coincides with the family's own train/validation edge.
7. **That the reserve was not touched** — §2.2: 15 cells, all `exposure: NO_TEST_ACCESS`, no retained
   array carrying a test population, the consumed slice ending 301 208 rows before the test block.

**It may not rely on:**

1. **Any claim that a weekly context improves a forecast.** §1 corrects a ranking of linear
   autocorrelations. No weekly-context model was fitted, here or in either range.
2. **A verified materialization of the `test` split.** There is none, by design. `MOD-CONF` owns the
   reserve, and the refusal in §2.2 is a refusal, not a deferred pass.
3. **Direct/cache parity beyond what the RP49–RP56 disposition §5 established** — refusals that fire
   by name on cases built there. This document verified *materialization*, not a second cache route.
4. **Per-split materialization on any panel but this one.** One panel, one split pair, plus the two
   pre-training splits and the phase-1 root that shares the identical prepared bytes. A second family
   is a new measurement.
5. **Anything recomputed under today's code for the four blocks closed under drifted code.** Not this
   repair's work, not claimed.

Its remaining blockers are unchanged, and neither of them is one of these two repairs: both are now
discharged.

---

## 4. Costs, and what was refused

CPU only, no GPU. Every job under `crispdm-run -m 3G -t 600 -n lagtbl`, env `trading-stack`. The whole
measurement is two reads of a 2 075 259-row parquet, one 50 400-row array, four auto-encoder cells'
origin arrays and 15 `cell.json` records; it runs in seconds. **No training, no fitting, no allocation,
no replay, no new campaign, no governance or warehouse contact, no host or service started, stopped or
restarted, no run root written to, no committed sample overwritten, no reserved split opened.**
`MemAvailable` was read before each launch (13.5 GiB, then 10.7 GiB, with two other agents live) and
the 3G cap was never raised.

Four refusals:

1. **I refused to replace the published lag numbers.** A silent correction would be the same defect as
   the original, pointed the other way. Both columns are published, each with its estimator named, and
   the rule set fails if either is lost.
2. **I refused to write a third autocorrelation implementation.** The as-published column is produced
   by the tool that produced the published table, called directly and asserted against to zero
   tolerance; the corrected column is the expression the RP49–RP64 audit already recomputes.
3. **I refused to report the test split as verified.** Nothing was materialized for it, so there is no
   materialization to check; it is refused by name, its absence is verified instead, and the reserve
   was not opened to improve on that.
4. **I refused to let any check pass on absent bytes.** `_verdict` refuses a split whose check set
   carries a `None` or is empty; an absent run root refuses all five splits **by name** rather than
   dropping them from the record; the lag table refuses rather than reprinting published values it
   could not re-derive; and the mutants for all of that are in the rule set.

---

*Satoshi, successor technical lead. 2026-09-26, under the owner's grant of 2026-09-26.*
