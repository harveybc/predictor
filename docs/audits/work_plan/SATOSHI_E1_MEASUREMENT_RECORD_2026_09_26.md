# E1 measurement record — complete for what was measured, and nothing else

Satoshi, successor technical lead. 2026-09-26. Acting on the owner's grant of 2026-09-26.
Worktree `predictor-e1q2-20260926`, branch `satoshi/e1-seal-q2-context-20260926`, from the master tip `941eb5b3`
that carries the E1 partial seal.

> ## READ THIS HEADER OR READ NOTHING
>
> **What this document is.** The complete record of the E1 **measurements**: which bytes exist, which rule each number was
> produced under, and what each number is. It is generated from
> [`E1_SEAL.json`](../evidence/E1_SEAL_20260926/E1_SEAL.json) (`E1_PARTIAL_SEAL`, 30 artifacts, `tools/df_e1_seal.py`,
> identities recomputed from bytes) plus the bounded Q2_CONTEXT fits of today, recorded in
> [`SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md`](SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md).
>
> **What this document is NOT.** It is **not** a seal of the E1 experiment, and it **does not** assert that 13E v1 is a
> validated pretraining experiment. The 2026-09-19 review refuses exactly that claim, in these words:
>
> > **No sellar 13E v1 como experimento de preentrenamiento.**
> > — [`MUSASHI_RP17_RP24_REVIEW_2026_09_19.md`](MUSASHI_RP17_RP24_REVIEW_2026_09_19.md), `## Disposicion`
>
> That refusal is observed here and is gap **P1** below. Nothing in this record promotes, ranks, selects or recommends any
> architecture, representation, optimizer, loss, feature or policy.
>
> ### The five gaps, on the face of the record
>
> | # | gap | state | why it is still open |
> |---|---|---|---|
> | **G1** | `ML_BASELINES_CAUSALITY_UNVERIFIED` | `NOT_DISCHARGEABLE_BY_ARTIFACTS` | the legacy lineage is `CAUSALITY_UNVERIFIED` with **both lineages UNBOUND**: the producer of the decomposed inputs is not in this repository. No work here reaches it |
> | **G2** | `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` | **still `UNMET` after today's fits** | calendar measured, volume measured, **context only partly measured**: the two W1440 FULL-DEPTH arms of Q2_CONTEXT (`long_window_own_depth`, `long_window_local_support_67`) are still unfitted, so context is **not** separated from depth. Today's bounded block says so on its own face |
> | **G3** | `POST_HUBER_EXTERNAL_ACCEPTANCE` | `NOT_DISCHARGEABLE_BY_ARTIFACTS` | the phase-2 v2 correction is an artifact; its acceptance is a review. Only Musashi writes it |
> | **G4** | `MOD_E1_EXTERNAL_REVIEW` | `NOT_DISCHARGEABLE_BY_ARTIFACTS` | `MUSASHI_RP49_RP56_REVIEW_2026_09_20.md` and `MUSASHI_RP57_RP64_REVIEW_2026_09_21.md` **do not exist**, while their neighbours RP41–RP48, RP66–RP73, RP74–RP81 and RP82–RP89 do. Two specific reviews are missing from an otherwise reviewed chain |
> | **G5** | `RP136_RP139_MATCHED_ECL_ADAPTER` | `UNMET` | every retained block design still names a single-target, single-offset contract (`REGISTRY household_W60_h60`); no full-output adapter artifact is retained |
>
> ### The two prohibitions this record observes
>
> | # | prohibition | state |
> |---|---|---|
> | **P1** | `RP17_RP24_NO_SEAL_13E_V1` — *"No sellar 13E v1 como experimento de preentrenamiento"* | `PROHIBITION_OBSERVED`: **this record does not seal 13E v1 either**, in any version |
> | **P2** | `HUBER_NO_RETURN_TO_R0_R1_R2_YET` — *"No return to R0/R1/R2 yet"* | `PROHIBITION_OBSERVED`: no block measured here carries an R0/R1/R2 arm |
>
> ### The standing limitation: no causal claim here is verified against a retained-row error
>
> Every number in this record that is **verified** is verified the same way: a forecast predicts a retained row, the row is
> retained, and the prediction is compared with it. `tools/df_closure_table.py` verifies rows of role `forecast` or
> `control_forecast` and **only** those (`FORECAST_ROLES`, line 51; line 526 skips every other role).
>
> **A causal claim does not predict a retained row.** A factorial effect, a contrast, a γ, an ATE or a "representation X adds
> information" statement is a difference between procedures, not a prediction of a row that could be compared with that row.
> So no causal claim in this record is, or can be, verified against a retained-row error. This is a **standing limitation**,
> not a to-do: it is a property of what a causal claim is, and it does not go away when more rows are measured.
>
> Consequently:
>
> * **Nothing downstream may read this record as carrying a verified causal effect.** It carries verified *forecast errors*
>   and unverified *differences between them*.
> * **Anything downstream that would need a verified causal effect stays REFUSED** — named explicitly in §7: MOD-CORE-PRETRAIN,
>   MOD-FROZEN-PREFIX, MOD-CONF, MOD-E3, the 13E pretraining comparison, the matched ECL benchmark and any selection or
>   promotion of a representation.
> * The paired arm differences in §4 are reported with **both signs and no interval**, exactly as the block designs'
>   `reading_rules` require: *"three seeds on one task are development evidence"*.
>
> ### Two disclosures about the record's own bytes
>
> * **One design digest has no retained document.** `be2e776e5c64a842…`, the frozen Huber/AdamW v2 design, is pinned only by
>   `MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md`. The four `huber_adamw_musashi` rows of §4 rest on a digest, not on bytes.
> * **Four blocks were closed under code that has since changed.** `DEV_MATCHED`, `Q1_CALENDAR`, `Q3_VOLUME` and
>   `ARCH_X_CALENDAR` record `closure_code_drift` for `tools/df_e1_block.py`. Their numbers are the numbers that were
>   measured; **this record does not claim they reproduce under today's code.**

---

## 1. Why this record exists, and why it is not a seal

The standing index says *"next = E1 sealing after review"*. The "after review" is load-bearing and the review has not
happened: G4 above. What exists instead is
[`SATOSHI_E1_PARTIAL_SEAL_RETURN_2026_09_26.md`](SATOSHI_E1_PARTIAL_SEAL_RETURN_2026_09_26.md) and its artifact
`E1_SEAL.json`, verdict `E1_PARTIAL_SEAL`.

I asked first whether the 2026-09-19 review bars me from writing this record at all. It does not, and it says so twice:

> Disposicion: **mediciones conservables**; cierre compuesto y ficha E1 requieren correccion; RL semanal aun no demostrado
> como politica ejecutable.

> Conservar etapa original y controles nuevos. **Aceptar el recalculo descriptivo al alcance indicado**, corregir cierre
> antes de promover resultados. **No sellar 13E v1 como experimento de preentrenamiento.** Ninguna decision pendiente del
> owner impide reparar estos puntos.

Read literally, that disposition authorises exactly three things and forbids exactly two. It authorises preserving the
measurements, accepting the descriptive recalculation at the stated scope, and repairing the named points. It forbids
sealing 13E v1 as a pretraining experiment, and it forbids **promoting** results before the composite closure is corrected.
This record preserves and describes; it seals no experiment and promotes nothing. So it is written.

On the second prohibition — *"corregir cierre antes de promover resultados"* — the composite-closure defect it points at
(F1: `merge_successor` built a new population from the designs and discarded the parents' verified populations) was
repaired in RP25–RP32 and that repair **was** reviewed:
[`MUSASHI_RP25_RP32_REVIEW_2026_09_19.md`](MUSASHI_RP25_RP32_REVIEW_2026_09_19.md) exists. That repair is about MOD-E0's
composite closure, not about either E1 closure, and this record rests on neither a merge nor a composite: it rests on the
two E1 closures separately, each with its own population.

One more constraint of the chain, recorded rather than argued away:
[`MUSASHI_RP82_RP89_REVIEW_2026_09_21.md`](MUSASHI_RP82_RP89_REVIEW_2026_09_21.md) says *"This review does not authorize
another simplified-model or financial fit."* The bounded Q2_CONTEXT fits of today were run under the **owner's grant of
2026-09-26**, which is later and is the authority a reviewer's non-authorisation does not override. The tension is stated
here rather than omitted, and the fits are published NON_GOVERNING (§5).

## 2. Identity — twelve sealed designs, every digest re-derived from its own bytes

Recomputed under the repository's canonical rule (`sha256` of the body without `design_sha256`, `sort_keys`,
`separators=(",",":")`), the same rule `df_d2_design.seal_design` and `df_mod_e0.sha_obj` use. Every one re-derives.

| design | digest | state |
|---|---|---|
| `RP30/E1_PILOT_DESIGN.json` | `807a4a30577d2ea4…` | original household DEV pilot |
| `RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json` | `143abb57d97daa07…` | successor pilot, sealed before it ran |
| `RP63/PHASE1_DESIGN_SEALED.json` | `5cb8263d359f1500…` | diagnostic phase 1 |
| `RP65/PHASE2_DESIGN_SEALED.json` | `6b47509400ef5e7b…` | phase 2 v1, 27 cells, `SEALED_NOT_EXECUTED` |
| `RP66/PHASE2_DESIGN_SEALED_v2.json` | `0dcb92b9c05ad143…` | phase 2 v2, 36 cells, `SEALED_NOT_EXECUTED` |
| `RP66/blocks/e1_block_dev_matched_v1/DESIGN.json` | `6f35d165bb96cea8…` | abandoned root, preserved |
| `RP66/blocks/e1_block_dev_matched_v2/DESIGN.json` | `e43380914a276d62…` | DEV_MATCHED |
| `RP66/blocks/e1_block_q1_calendar_v1/DESIGN.json` | `1e306894ee94453a…` | Q1_CALENDAR |
| `RP66/blocks/e1_block_q2_context_v1/DESIGN.json` | `6d1aaecaf27c581c…` | Q2_CONTEXT — zero cells ever fitted |
| `RP66/blocks/e1_block_q3_volume_v1/DESIGN.json` | `fbbda9f3c75ea951…` | Q3_VOLUME |
| `RP74/blocks/e1_block_arch_x_calendar_v1/DESIGN.json` | `f4e616f68d707dd0…` | ARCH_X_CALENDAR, 15 cells, three hosts |
| `RP82/blocks/e1_block_context_daily_lag_v1/DESIGN.json` | `31e71033d93545d1…` | CONTEXT_DAILY_LAG |

Plus, in the same seal and each bound by file digest: two closures, five block reports, one report whose design is not
retained (header disclosure), the owner's closure table, and the nine review/return/task-sheet documents every quoted
condition comes from. **30 artifacts**, all under `docs/audits/evidence/` and `docs/audits/work_plan/` in this repository.

New today, and outside that seal because the seal was taken from bytes that existed before it:
`e1_block_q2_context_bounded_v1/DESIGN.json`, digest **`47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`**,
12 cells, 4 cost pilots — see §5.

## 3. The rules each number was produced under, copied as stated

* **Scaler.** `COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels mean
  0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd` — `sd_train`
  = 0.9125164391265214 kW.
* **Common evaluation.** `the intersection over the block's arms of admissible validation origins, with a finite label and a
  finite daily lookup, derived at prepare BEFORE any score; every arm scores on it` — 10,020 origins in every block.
* **Reading rules**, identical in every block design: *three seeds on one task are development evidence*; *a fit that
  reached the update ceiling is CENSORED wherever its best checkpoint fell*; *no cell is removed after its score is seen*;
  *a published number under another protocol never enters the comparison column*.
* **Recipe (TIER1).** MAE loss, Adam, lr 0.003, batch 64, ceiling 4,000 updates, validation every 200 observed updates,
  patience 3 events, restore best, `min_delta` 0.
* **Missing policy.** `rows with a non-finite input withdrawn from the enumeration; nothing imputed`.
* **Scope, on every block.** `DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no
  test rows read`.

## 4. The numbers as measured

### 4.1 The two closures, and the qualification that makes `ALL_VERIFIED` readable

| closure | verdict | declared | metrics | inference | regime | governance scope |
|---|---|---:|---:|---:|---:|---|
| `RP34/E1_PILOT_CLOSE_REPAIRED.json` | `ALL_VERIFIED` | 15 | 15 | 11 | 14 | 15 × `SCIENTIFICALLY_VERIFIED_HISTORICAL_UNGOVERNED` |
| `RP55/E1_SUCCESSOR_CLOSE.json` | `ALL_VERIFIED` | 15 | 15 | 11 | 14 | 15 × `VERIFIED_AND_GOVERNED` |

`ALL_VERIFIED` beside "inference 11 of 15" resolves, and the resolution is asserted as a test rather than repeated as a
claim: the four units whose inference is `NOT_APPLICABLE` are `pilot_ae`, `ae_s1`, `ae_s2`, `ae_s3` — auto-encoders, with no
forecast to replay — and the one unit whose regime is `NOT_APPLICABLE` is `controls`, which has no detector. No forecast
unit is unreplayed; no unit carries `METRICS_VERIFIED_INFERENCE_NOT_REPLAYED`; `register_problems` is empty; zero absent
ids and zero strangers on disk in either closure. `verified + NOT_APPLICABLE = declared` holds per closure and per fact.

**The 15 units of the original pilot stay `HISTORICAL_UNGOVERNED` here.** Nothing in this record promotes them.

### 4.2 The blocks that produced an outcome

| block | outcome | common rows | arms (mean MAE, kW) |
|---|---|---:|---|
| DEV_MATCHED | verified, 0 problems | 10,020 | `gru_adapted_w60` 0.484795 · `modular_w60` 0.518351 |
| Q1_CALENDAR | verified, 0 problems | 10,020 | `calendar` 0.444230 · `randomised_calendar_control` 0.504504 |
| Q3_VOLUME | verified, 0 problems | 10,020 | `volume_112d` 0.508690 · `volume_56d` 0.511739 |
| ARCH_X_CALENDAR | verified, 0 problems | 10,020 | `gru_calendar_w60` 0.437860 · `calendar` 0.449555 · `gru_adapted_w60` 0.482715 · `modular_w60` 0.491926 · `randomised_calendar_control` 0.496969 |
| CONTEXT_DAILY_LAG | verified, 0 problems | 10,020 | `daily_lag` 0.487187 · `modular_w60` 0.494256 |
| **Q2_CONTEXT** (v1, `6d1aaecaf2…`) | **`BUDGET_LIMITED_BEFORE_ANY_OUTCOME`** | — | **0 cells with an outcome**; 5 cost pilots only, 2,259.56 CPU s |
| **Q2_CONTEXT_BOUNDED** (v1, `47a270eec0…`, today) | see §5 and its own results document | 10,020 | 12 fits, 4 arms × 3 seeds, NON_GOVERNING |

Q2_CONTEXT v1's own reading, fixed in its artifact and not softened here: *"pilots only; no cell of this block was fitted to
a score, so it says nothing about its question — not 'no effect'."* That sentence is still true of v1 today: **no cell of
`e1_block_q2_context_v1` was fitted by this round or any other.** Today's fits are a different, separately sealed, bounded
block with a different digest, and they do not answer v1's question (G2).

### 4.3 The owner's closure table, bound rather than regenerated

[`RP82/CLOSURE_TABLE_RP89.json`](../evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json), `owner_closure_table.v2`,
2026-09-21T18:45:16Z, produced by `tools/df_closure_table.py`: **70 rows, 39 verified, 31 preserved with a qualified
scope, `problems: []`**; custody `ACCEPTED_ARTIFACT_CHAIN` 51 / `METRIC_ANCHORED` 19; preparation
`PREPARATION_ACCEPTED_ARTIFACT` 39 / `PREPARATION_LOCAL_ONLY` 31; eight runs, every design identity recomputing. Per row:
all thirteen required columns present, model and naive sharing population, horizon and scale, skill status `MEASURED`.

The naive is the same in all 70 rows: **persistence at the horizon on the identical evaluation origins, 0.617372 kW.**

**Every one of the 70 rows is `NOT_COMPARABLE` against the literature**, each with its reason and its planned matched
comparison. The Gasparin / Lukovic / Alippi electric-load table (arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1),
doi:10.1049/cit2.12060 — Table 5, MAE kW: FNN 0.53, DFNN 0.53, TCN 0.54, ERNN-MIMO 0.56, LSTM-MIMO 0.53, GRU-MIMO 0.52,
seq2seq-TF 0.57, seq2seq-SG 0.53) is **recorded and never placed in the comparison column**, with
`why_not: unknown identity fields cannot match as proof: ['target_transform']`.

**That, not any MAE number, is the honest headline of the E1 measurement programme.**

Best measured skill against persistence on the DEV validation week, as measured, with no interval claimed:
`gru_calendar_w60` 0.290768 · `calendar` 0.280449 · `gru_adapted_w60` 0.218114 · `daily_lag` 0.210870 ·
`modular_w60` 0.203193 · `randomised_calendar_control` 0.195025 · `volume_112d` 0.176039 · `volume_56d` 0.171102 ·
`R0` 0.114101 · `R2` 0.104818 · `R1` 0.098219.

These are eleven forecast skills on one task, three seeds, one previously inspected validation week. They are **not** a
causal effect and this record does not turn a difference between two of them into one (header, standing limitation).

## 5. What was measured today: the bounded Q2_CONTEXT block

Full numbers, closure table and cost in
[`SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md`](SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md). Enough here that a
reader of this record alone is not left guessing:

Q2_CONTEXT v1 could not be run on this host. Its two retained cost pilots measured `long_window_own_depth` at 4.165 CPU s
per update and **8.46 GB** peak RSS and `long_window_local_support_67` at 4.443 CPU s per update and **10.28 GB** peak RSS
— 17,507 s and 18,406 s per cell at the 4,000-update ceiling, and a resident set that would take this host's memory away
from its owner. So a **bounded** block was sealed **before any score of any cell existed**, on those pilots' own
measurements and on nothing else: the same rows, scaler, recipe, cadence, ceiling, seeds and contract, restricted to the
four arms whose pilots fit — `modular_w60`, `daily_lag`, `long_window_crop60` (the exact-information null) and
`short_window_deep_core`. The restriction is a resource declaration made before any score, never a removal after a score
was seen.

**What it measured.** Twelve fits, four arms x three paired seeds, on 38,700 identical train origins and the block's
10,020 common evaluation origins. All twelve stopped on validation, none was censored by the update ceiling, every one
replayed its checkpoint in a fresh process to `allclose(1e-6, 1e-6)`, and **every one beat persistence on the same rows**:

| arm | mean MAE kW | mean skill vs persistence | paired Delta MAE_z vs the W60 baseline | signs (+ / -) |
|---|---:|---:|---|---|
| `modular_w60` (baseline) | 0.499100 | 0.191573 | --- | --- |
| `daily_lag` | 0.494401 | 0.199184 | -0.012938 / +0.000587 / -0.003097 (mean -0.005149) | 1 / 2 |
| `long_window_crop60` (exact-information null) | 0.499100 | 0.191573 | +0.000000 / +0.000000 / +0.000000 | 0 / 0 |
| `short_window_deep_core` | 0.502205 | 0.186543 | +0.000402 / +0.011518 / -0.001711 (mean +0.003403) | 2 / 1 |

Persistence 0.617372 kW on those rows; the other two declared references land **worse** than persistence
(daily-seasonal 0.731659 kW, skill -0.185; train-constant 0.709950 kW, skill -0.150) and are published as they landed.

The daily-lag channel is smaller-error in two seeds of three and **larger in one**; the deeper dilated core is
larger-error in two of three. **Neither is a demonstrated effect on n = 3 and this record claims neither.** The one claim
that does not rest on n = 3 is the null: `long_window_crop60`, the W1440 input cropped to its last 60 rows before the
extractor, reproduced the W60 baseline's initial-weight digest in 3 of 3 seeds and its MAE **to the last bit** in 3 of 3
seeds. RP87 asserted that equivalence as a property of the code; today it is a live measurement on the block's own rows.

Three consequences this record prints rather than hides:

1. **G2 is still `UNMET`.** The bounded block does not contain the long-window full-depth treatment, so it does not
   separate context from depth and does not answer Q2_CONTEXT's question. Its own sealed `question` field says so.
2. **The bounded fits are NON_GOVERNING.** This host holds no data-gov service key, so the governed runner refuses before
   opening data. They ran through `tools/df_e1_block_ungoverned.py`, which reuses the sealed design and every sealed code
   path of `df_e1_block` unchanged and adds no science of its own, and which records what it lacks: no data-gov
   acquisition, no accepted terminal, no warehouse read — therefore every closure row's custody is `UNCHECKED`, the
   preparation's is `PREPARATION_LOCAL_ONLY`, and the closure's `verified` flag is `false` for that reason. Per
   `docs/GOVERNED_RUN.md`, such results "cannot promote a model, transformation, feature or experiment". They do not.
3. **The owner closure table for this run prints no error at all.** `tools/df_closure_table.py`'s policy is that a score
   with no accepted terminal receipt is reported as **no model error**, not as a qualified one. The
   `owner_closure_table.v2` for this run therefore lands with **12 rows, 0 verified, every error column `null`, custody
   `UNCHECKED` x 12**, and it is published exactly like that. The measured errors above are published beside it as an
   `unanchored_measurement_table.v1`, recomputed from each cell's retained arrays and cross-checked against the record's
   stored score to 1e-12, which carries the five columns the owner's rule names and is **never called verified**.

Three defects were found on the way, reported because they are defects and not because they help. All three are stated in
full, with their repair state, in the header of
[`SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md`](SATOSHI_Q2_CONTEXT_BOUNDED_RESULTS_2026_09_26.md); in brief:

1. **Q2_CONTEXT v1's prepared data does not honour its own design's declared train population.** The v1 design declares
   `train_population: COMMON_INTERSECTION` with the reason *"every arm of this block, the baseline included, trains on the
   SAME origins so context is never conflated with volume"*, but the retained `BLOCK_DATA.npz` of
   `e1_block_q2_context_v1` holds **per-arm** train origins — 40,020 / 38,700 / 38,700 / 38,700 / 40,080 — and its
   `BLOCK_DATA.json` carries neither `train_admissible_before_intersection` nor `common_train_origins`. The intersection
   code exists in today's `prepare()` and was exercised today: the bounded block's four arms all train on the **same
   38,700** origins, a subset of the source run's train origins, with the common evaluation identical to the source's
   10,020. **Had Q2_CONTEXT v1 ever been fitted on its retained preparation, it would have conflated the input contrast
   with the train volume it declares it controls.** It was never fitted, so no published number is affected.
   **REPORTED, not repaired**: what to do with v1's retained preparation is the owner's call. A second, smaller one in the
   same record: after the intersection is applied, `feasibility[arm]["train_admissible"]` still reports the
   **pre**-intersection count under a name that reads as the count used; the count actually used is
   `binding_to_source.common_train_origins`. Both are in the record, but a reader of the first field alone would be misled.
2. **`tools/df_closure_table.py`'s authoritative verifier crashed on a run with no accepted terminal at all** —
   `FileNotFoundError` from an unguarded `TERMINAL_RECEIPTS.json` read, while its two sibling readers guard it — instead of
   reaching its own typed branch one line below. **REPAIRED and tested.** The test pins both halves, including the half
   that matters more: with the guard in place, the tool still reports **no model error** for a unit with no accepted
   terminal. That policy is why this round's owner closure table has 12 rows and 0 numbers, and why the measured errors are
   published beside it as an `unanchored_measurement_table.v1` that is never called verified.
3. **The block closure's initial-weight pairing check reported a FALSE problem** on this block, because its key omitted
   `dilations` and `crop` — the fields that change the built graph — so `modular_w60` (8,127 parameters) and
   `short_window_deep_core` (12,047 parameters) collided and were required to share initial weights they cannot share.
   **REPAIRED and tested**, with the check's real intent (same graph, same seed, different input **data** must share
   initial weights) unchanged. No previously closed block combined such arms, so no closed block's verdict moves.

Both repairs touch the authoritative verifier and its closure. **Accepting a verifier change is a review, not an
artifact** (§8.4).

## 6. Tests

Re-run today in the anaconda env `trading-stack` against both repairs. `test_df_e1_seal.py`, `test_df_e1_block.py`,
`test_m4_c32_c38_reverify.py`, `test_df_e1_phase2_acceptance.py` and `test_df_e1_chronology.py` — **85 passed**;
`test_df_closure_table.py` including the new repair test — **37 passed**. The exact counts, the two new tests and a fourth
defect found while running the batteries (fourteen tests in two files go red whenever the checkout is not byte-clean, which
the suite can cause itself under `docs/audits/evidence/`, and which a concurrent edit also causes) are in §4 and §5 of
[`SATOSHI_E1_SEAL_AND_Q2_CONTEXT_2026_09_26.md`](SATOSHI_E1_SEAL_AND_Q2_CONTEXT_2026_09_26.md).

## 7. What stays REFUSED, by name

Because no causal claim here is verified against a retained-row error (header), and because G1–G5 stand:

| downstream item | state | blocked by |
|---|---|---|
| sealing 13E as a pretraining experiment, any version | **REFUSED** | P1 for v1; for v2/v3 the review that would seal them does not exist (G4) |
| `MOD-CORE-PRETRAIN` | `NOT_STARTED`, stays so | `depends_on: MOD-E1`; G4 |
| `MOD-FROZEN-PREFIX` | `NOT_STARTED`, stays so | `depends_on: MOD-E1`; G4 |
| `MOD-CONF` | `NOT_STARTED`, stays so | `depends_on: MOD-E1`; G4 |
| `MOD-E3` | `NOT_STARTED`, stays so | `depends_on: MOD-E1`; G4 |
| the matched ECL benchmark | **REFUSED** | G5: single-target single-offset contract, no retained full-output adapter |
| using any legacy phase-3 number as a benchmark | **REFUSED** | G1: `CAUSALITY_UNVERIFIED`, both lineages `UNBOUND` |
| accepting phase-2 v2 (`0dcb92b9…`) or the verifier guarantees | **REFUSED** | G3: acceptance is a review |
| answering Q2_CONTEXT's question | **REFUSED** | G2: the W1440 full-depth arms are unfitted |
| selecting, ranking or promoting any representation on these numbers | **REFUSED** | the standing limitation: a difference between forecast errors is not a verified causal effect |
| claiming the four drifted-code blocks reproduce under today's code | **REFUSED** | header disclosure; a reader who wants it must say so and pay for a replay |

## 8. What the owner or Musashi must still decide

1. **The two missing reviews** (`MUSASHI_RP49_RP56_REVIEW`, `MUSASHI_RP57_RP64_REVIEW`). This is the single decision that
   unblocks the most: four modules depend on it.
2. **Whether to accept phase-2 v2** and the verifier guarantees the post-Huber review declined to accept unchanged.
3. **Whether to authorise the two W1440 full-depth Q2_CONTEXT arms on a host that can hold 10.3 GB**, which is what G2
   needs and what this host cannot give.
4. **What to do about Q2_CONTEXT v1's retained preparation**, which does not honour its own declared train population (§5).
   It was never fitted, so nothing published rests on it; the options are re-prepare, supersede, or retire.
5. **What to do about the legacy lineage.** `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND` is not a defect of E1 and
   no work here fixes it; it may need retiring as a condition rather than standing as a permanent block. Retiring a
   condition is the owner's call.
6. **Whether to retain the Huber/AdamW v2 design document**, so the four `huber_adamw_musashi` rows rest on bytes.
7. **Whether the governed path should be available to the successor at all** — a data-gov service key. Without it every
   measurement taken here is NON_GOVERNING by construction, which is the honest but weakest scope available.

---

Satoshi, successor technical lead — 2026-09-26, on the owner's grant of 2026-09-26. Nothing in this document is written
under Musashi's name, and no review is signed on his behalf.
