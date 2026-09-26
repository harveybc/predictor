# E1: a partial seal that names its five gaps — identity, rules and numbers fixed; the experiment not sealed

Order: "E1 sealing", the thread [`MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md`](MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md) and
[`SATOSHI_PROGRAM_RP17_RP24_RETURN_2026_09_19.md`](SATOSHI_PROGRAM_RP17_RP24_RETURN_2026_09_19.md). Worktree `predictor-rp132`,
branch `satoshi/rp132-rp134-20260923`. **No model was fitted, loaded, scored or replayed; no GPU; no allocation; no new campaign,
no warehouse or governance contact.** The seal is built from bytes that already existed.

The instruction was to read the review's conditions for sealing, check each against the retained evidence, and seal what the
evidence supports while naming what it does not. The first thing to report is a premise correction, because everything else turns
on it.

---

## 1. The document named does not state sealing conditions — the companion one refuses the seal

`MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md` contains no occurrence of `seal`, `sellar`, `sellado`, `cierre` or `cerrar`.
It mentions "E1" once, in its own header. Its six sections are
`## Comparacion que faltaba`, `## Antecedente antiguo localizado`, `## Problemas e hipotesis concretas`,
`## Literatura consultada y uso preciso`, `## Entregable ejecutado y limites` — there is no `## Disposicion` section at all, and
its single `Disposicion:` line is scoped to a **legacy CSV**, not to E1:

> **Aclaracion posterior del owner (20-sep):** la unica mejora valida que recuerda fue en fase 3, despues de corregir una fuga
> causal en la descomposicion wavelet de una entrada; antes hubo un error anormalmente bajo (~0.001). Sospecha que el CSV citado
> aqui pertenece a una ejecucion afectada. NO he establecido esa identidad. Los numeros siguientes son
> transcripcion/comparacion aritmetica de un resumen, no evidencia aceptada de capacidad predictiva. Disposicion:
> `CAUSALITY_UNVERIFIED, no apto como benchmark hasta reconstruir ambos linajes.`

It is an ML-diagnostic review. Its imperatives gate *other* things — "antes de repetir una replica", "antes de usar espectros
para elegir contextos", "antes de un benchmark" — and its closing order is `Ejecutar RP57-RP64`, which was done.

The sealing language is in the **companion** review of the previous day,
[`MUSASHI_RP17_RP24_REVIEW_2026_09_19.md`](MUSASHI_RP17_RP24_REVIEW_2026_09_19.md), whose header disposition reads

> Disposicion: **mediciones conservables; cierre compuesto y ficha E1 requieren correccion; RL semanal aun no demostrado como
> politica ejecutable**.

and whose `## Disposicion` section closes with the sentence this whole thread is reaching for:

> Conservar etapa original y controles nuevos. Aceptar el recalculo descriptivo al alcance indicado, corregir cierre antes de
> promover resultados. **No sellar 13E v1 como experimento de preentrenamiento.** Ninguna decision pendiente del owner impide
> reparar estos puntos.

And the request side, from the RP17–RP24 return — **the only occurrence of the collocation "E1 sealing" in the repository**:

> ## Request
>
> One review of RP17–RP24: the contrast-bound estimator and its oracles, the completed 2 × 2 and its reading, the replay scope,
> the grains, **the E1 task sheet (13E) before any E1 sealing**, and the E3 environment tests under 13D.

So the state of the thread is: a **request** for a review before any E1 sealing (2026-09-19), an **explicit refusal** to seal 13E
v1 (2026-09-19), and **no seal anywhere since**. `PROJECT_METHOD_STATE.json` still carries `MOD-E1` as `"status": "EXECUTED"`
with `"adoption": "SUCCESSOR_EXECUTED_LOCAL_METRICS_RECOMPUTED_FULL_GOVERNANCE_REVIEW_NOT_REPEATED"` and a live `next_action`,
and `MOD-FROZEN-PREFIX`, `MOD-CORE-PRETRAIN`, `MOD-CONF` and `MOD-E3` are all `NOT_STARTED` with `depends_on: ["MOD-E1"]`.
**E1 sealing has never happened, and this return does not complete it either.**

## 2. What was built: `tools/df_e1_seal.py`

A seal is four things, and the tool does exactly those four:

| | |
|---|---|
| **identity** | every cited artifact is recomputed from its own bytes. A sealed design must re-derive its `design_sha256` under the repository's canonical rule (`sha256` of the body without that field, `sort_keys`, `separators=(",",":")`) — the same rule `df_d2_design.seal_design` and `df_mod_e0.sha_obj` use. A closure, a report or a closure-table run must bind a design that is itself in the inventory, or be declared as unretained |
| **rules** | the acceptance rule is copied **verbatim** from the artifact that stated it before its outcome existed — `reading_rules`, `common_evaluation_rule`, `scaler_rule`, `held_constant`, `tier`, `limits`, and each closure's own `population_rule` / `replay_rule` / `governance_meaning` |
| **numbers** | the measured numbers are copied as measured, each beside the qualification its producer attached — censored fits, `NOT_APPLICABLE` facts, blocks with no outcome |
| **conditions** | every condition of the review chain, with its source and quote, typed as `DISCHARGED`, `PROHIBITION_OBSERVED`, `UNMET` or `NOT_DISCHARGEABLE_BY_ARTIFACTS` |

The verdict is not an opinion: `E1_SEALED` only when every condition is `DISCHARGED` or `PROHIBITION_OBSERVED`,
`E1_PARTIAL_SEAL` otherwise. Identity, rules and numbers are sealed either way, because they are facts about bytes.

The seal **refuses rather than misleads**. A declared artifact that is absent, a design whose digest does not re-derive, a closure
binding a design outside the inventory, a closure-table run binding a digest that is neither retained nor declared unretained, a
"no outcome" block that turns out to have a `REPORT.json`, a table row whose model and naive do not share rows / horizon / scale,
or an empty table — each one raises and nothing is emitted.

Result: **`E1_PARTIAL_SEAL`, 30 artifacts sealed, five named gaps.**
[`E1_SEAL.json`](../evidence/E1_SEAL_20260926/E1_SEAL.json) ·
[`E1_SEAL.md`](../evidence/E1_SEAL_20260926/E1_SEAL.md) ·
[`seal_run.out`](../evidence/E1_SEAL_20260926/seal_run.out)

## 3. What the seal DOES seal

### 3.1 Identity — twelve sealed designs, recomputed, every one re-derives

| design | digest (recomputed) | state |
|---|---|---|
| `RP30/E1_PILOT_DESIGN.json` | `807a4a30577d2ea4…` | the original household DEV pilot |
| `RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json` | `143abb57d97daa07…` | successor pilot, sealed before it ran |
| `RP63/PHASE1_DESIGN_SEALED.json` | `5cb8263d359f1500…` | diagnostic phase 1 |
| `RP65/PHASE2_DESIGN_SEALED.json` | `6b47509400ef5e7b…` | phase 2 v1, 27 cells, `SEALED_NOT_EXECUTED` |
| `RP66/PHASE2_DESIGN_SEALED_v2.json` | `0dcb92b9c05ad143…` | phase 2 v2, 36 cells, `SEALED_NOT_EXECUTED` |
| `RP66/blocks/e1_block_dev_matched_v1/DESIGN.json` | `6f35d165bb96cea8…` | abandoned root, preserved |
| `RP66/blocks/e1_block_dev_matched_v2/DESIGN.json` | `e43380914a276d62…` | DEV_MATCHED |
| `RP66/blocks/e1_block_q1_calendar_v1/DESIGN.json` | `1e306894ee94453a…` | Q1_CALENDAR |
| `RP66/blocks/e1_block_q2_context_v1/DESIGN.json` | `6d1aaecaf27c581c…` | Q2_CONTEXT — the block with no outcome |
| `RP66/blocks/e1_block_q3_volume_v1/DESIGN.json` | `fbbda9f3c75ea951…` | Q3_VOLUME |
| `RP74/blocks/e1_block_arch_x_calendar_v1/DESIGN.json` | `f4e616f68d707dd0…` | ARCH_X_CALENDAR, 15 cells, three hosts |
| `RP82/blocks/e1_block_context_daily_lag_v1/DESIGN.json` | `31e71033d93545d1…` | CONTEXT_DAILY_LAG |

Plus two closures, five block reports, the owner's closure table, one report whose design is **not** retained (§5), and the nine
review/return/sheet documents the conditions are quoted from, each bound by file digest. **30 artifacts, all under
`docs/audits/evidence/` and `docs/audits/work_plan/` in this repository.** Every closure and report binds a design in this list;
every closure-table run binds one of these twelve or the one declared exception.

### 3.2 The numbers as measured — and the qualification that makes `ALL_VERIFIED` readable

| closure | verdict | declared | metrics | inference | regime | governance |
|---|---|---:|---:|---:|---:|---|
| `RP34/E1_PILOT_CLOSE_REPAIRED.json` | `ALL_VERIFIED` | 15 | 15 | 11 | 14 | 15 `HISTORICAL_UNGOVERNED` |
| `RP55/E1_SUCCESSOR_CLOSE.json` | `ALL_VERIFIED` | 15 | 15 | 11 | 14 | 15 `VERIFIED_AND_GOVERNED` |

`ALL_VERIFIED` beside "inference 11 of 15" is exactly the kind of pair a seal must resolve rather than repeat. It resolves: the
four units whose inference is `NOT_APPLICABLE` are `pilot_ae`, `ae_s1`, `ae_s2`, `ae_s3` — auto-encoders, with no forecast to
replay — and the one unit whose regime is `NOT_APPLICABLE` is `controls`, which has no detector. **No forecast unit is
unreplayed**, no unit carries the `METRICS_VERIFIED_INFERENCE_NOT_REPLAYED` scope, `register_problems` is empty, and there are
zero absent ids and zero strangers on disk in either closure. The seal asserts that arithmetic (`verified + NOT_APPLICABLE =
declared`, per fact) as a test, so the pair can never drift apart silently.

| block | outcome | common rows | arms (mean MAE, kW) |
|---|---|---:|---|
| DEV_MATCHED | verified, 0 problems | 10,020 | `gru_adapted_w60` 0.484795 · `modular_w60` 0.518351 |
| Q1_CALENDAR | verified, 0 problems | 10,020 | `calendar` 0.444230 · `randomised_calendar_control` 0.504504 |
| Q3_VOLUME | verified, 0 problems | 10,020 | `volume_112d` 0.508690 · `volume_56d` 0.511739 |
| ARCH_X_CALENDAR | verified, 0 problems | 10,020 | `gru_calendar_w60` 0.437860 · `calendar` 0.449555 · `gru_adapted_w60` 0.482715 · `modular_w60` 0.491926 · `randomised_calendar_control` 0.496969 |
| CONTEXT_DAILY_LAG | verified, 0 problems | 10,020 | `daily_lag` 0.487187 · `modular_w60` 0.494256 |
| **Q2_CONTEXT** | **`BUDGET_LIMITED_BEFORE_ANY_OUTCOME`** | — | **0 cells with an outcome**; pilots only, 2,259.56 CPU s |

Q2_CONTEXT is in the seal as a row, not as an omission. Its reading is fixed in the artifact: *"pilots only; no cell of this block
was fitted to a score, so it says nothing about its question — not 'no effect'."* The seal refuses if that claim ever becomes
false (a `REPORT.json` appearing in that root raises).

### 3.3 The owner's closure table, bound rather than regenerated

The owner's standing rule is that every closure carries the table: model error with its scale, the naive **on the same rows**, the
skill, the literature value with its source, comparability, `NOT_COMPARABLE` with its reason, generated from artifacts. That table
already exists for exactly these units —
[`RP82/CLOSURE_TABLE_RP89.json`](../evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json), `owner_closure_table.v2`,
2026-09-21, produced by `tools/df_closure_table.py`. The seal **binds** it and checks its invariants; it does not regenerate it
and it recomputed no error from arrays, which would need the arrays.

**70 rows, 39 verified, 31 preserved with a qualified scope, `problems: []`**, custody `ACCEPTED_ARTIFACT_CHAIN` 51 /
`METRIC_ANCHORED` 19, preparation `PREPARATION_ACCEPTED_ARTIFACT` 39 / `PREPARATION_LOCAL_ONLY` 31. Eight runs, every design
identity recomputing. The seal verified per row that all thirteen required columns are present, that model and naive share
population, horizon and scale, and that the skill status is `MEASURED`. The naive is the same across all 70 rows: *persistence at
the horizon on the identical evaluation origins*, 0.617372 kW.

Every one of the 70 rows is `NOT_COMPARABLE` against the literature, with its reason and its planned matched comparison — the
Gasparin/Lukovic/Alippi electric-load tables are recorded and never placed in the comparison column. **That, not any of the MAE
numbers, is the honest headline of the E1 measurement programme.** Best measured skill against persistence on the DEV validation
week: `gru_calendar_w60` 0.290768; the plain `modular_w60` baseline 0.203193; the successor's `R0` 0.114101, `R2` 0.104818,
`R1` 0.098219. Development evidence on one task, three seeds, one previously inspected validation week — as the designs' own
`reading_rules` say: *"three seeds on one task are development evidence"*, *"no cell is removed after its score is seen"*,
*"a published number under another protocol never enters the comparison column"*.

## 4. What the seal does NOT seal — the five gaps, each with its reason

| condition | source | state | why |
|---|---|---|---|
| `RP17_RP24_NO_SEAL_13E_V1` | RP17–RP24 review | `PROHIBITION_OBSERVED` | 13E v1 still carries *"E1 design (proposal references; to be sealed after review)"* and *"the E1 campaign is NOT launched on the v1 `eligible: true`"*; no retained artifact seals it. **This seal does not seal 13E v1 either** |
| `HUBER_NO_RETURN_TO_R0_R1_R2_YET` | Huber/AdamW results §6 | `PROHIBITION_OBSERVED` | no post-Huber block carries an R0/R1/R2 arm; the arms measured are the eight listed in §3.2 |
| `POST_HUBER_PHASE2_CORRECTED` | post-Huber review, finding 3 | `DISCHARGED` | v1 `6b475094…` preserved and superseded by v2 `0dcb92b9…`, both `SEALED_NOT_EXECUTED`, both digests re-derive |
| `OWNER_CLOSURE_TABLE_FROM_ARTIFACTS` | owner standing order | `DISCHARGED` | §3.3 |
| **`ML_BASELINES_CAUSALITY_UNVERIFIED`** | ML-baselines review | **`NOT_DISCHARGEABLE_BY_ARTIFACTS`** | the answering RP57–RP64 return records **both lineages as UNBOUND**: the producer of the decomposed inputs is not in this repository. What would discharge it: a reconstruction of a phase-3 run's consumed representation, from a producer that is not here. **No amount of work in this repository reaches it** |
| **`ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR`** | ML-baselines review, item 4 | **`UNMET`** | calendar measured, volume measured, **context not**: Q2_CONTEXT has zero cells with an outcome. What would discharge it: fitting those cells. **That is training, and this round is not authorised to run it** |
| **`POST_HUBER_EXTERNAL_ACCEPTANCE`** | post-Huber review | **`NOT_DISCHARGEABLE_BY_ARTIFACTS`** | the correction is an artifact; its acceptance is a review. What would discharge it: a Musashi review accepting phase-2 v2 and the verifier guarantees |
| **`MOD_E1_EXTERNAL_REVIEW`** | RP49–RP56 return | **`NOT_DISCHARGEABLE_BY_ARTIFACTS`** | the return says it itself — *"MOD-E1 is **VERIFIED, not externally reviewed**"* — and **neither `MUSASHI_RP49_RP56_REVIEW` nor `MUSASHI_RP57_RP64_REVIEW` exists in this repository**. What would discharge it: those reviews. Only Musashi writes them |
| **`RP136_RP139_MATCHED_ECL_ADAPTER`** | RP136–RP139 review, finding 4 | **`UNMET`** | every retained block design still names a single-target, single-offset contract — `REGISTRY household_W60_h60` — and no full-output adapter artifact is retained. What would discharge it: a full-output adapter with independent target/scaler/reduction parity, and a fit against it. Both are training |

Three of the five gaps are **not closable by any agent**: two need a review, one needs a producer that is not in this repository.
Two need training that nobody has authorised. That is why this is a partial seal and not a complete one, and the document says so
in its verdict field rather than in a footnote.

## 5. Two disclosures the seal makes about itself

1. **One design digest has no retained artifact.** The closure table and the Huber report both bind
   `be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965`, the frozen Huber/AdamW v2 design, pinned by
   `MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md` — but **the design document itself is not in this repository**, so the seal cannot
   recompute that identity from bytes. It is declared in `DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT` with its reason, so the
   binding check names it instead of quietly passing; and the seal refuses if that declaration ever becomes false. The four
   `huber_adamw_musashi` arms in §3.3 therefore rest on a digest, not on a document. **Retaining that design is a cheap, concrete
   repair and it is not mine to decide.**
2. **Four blocks were closed under code that has since changed.** Their reports record `closure_code_drift` for
   `tools/df_e1_block.py`: DEV_MATCHED sealed `51f7eb46…`, Q1_CALENDAR `798711c4…`, Q3_VOLUME `09a37e7e…`, ARCH_X_CALENDAR
   `8a9a7f91…`, all against today's `dbd27e0c…`. CONTEXT_DAILY_LAG closed under the sealed code. The recorded numbers are the
   numbers that were measured; **the seal does not claim they reproduce under today's code**, and a reader who wants that must
   say so and pay for a replay.

## 6. Tests added

`tests/test_df_e1_seal.py` — **25 passed**. The positive case, then the refusals, then the mutations.

- The real repository seals to `E1_PARTIAL_SEAL` with exactly the nine condition states of §4, and every gap carries a
  `what_would_discharge_it` or an `evidence` line.
- Every design identity in the output is the **recomputed** one, not the read-back one.
- The `verified + NOT_APPLICABLE = declared` arithmetic of §3.2 is asserted per closure and per fact, so `ALL_VERIFIED` can never
  come apart from its qualification.
- The closure table is checked row-wise: thirteen required columns, one evaluation population per arm, `NOT_COMPARABLE`
  throughout, and `rows_total == verified + preserved_qualified`.
- **Eight refusals**, each raising on a throwaway copy of exactly the declared artifacts: an absent artifact; an edited design; an
  edited design **with its declaration repaired to match** (the obvious attack — it dies on the closure table's binding instead of
  passing); a closure binding a design outside the inventory; the seal's own unretained-design disclosure contradicted; a
  closure-table run binding an unknown digest; a "no outcome" block that has one; a model and naive that do not share rows.
- **Four evidence flips**, proving no state is hard-coded: sealing 13E v1 turns `PROHIBITION_OBSERVED` into `UNMET`; giving
  Q2_CONTEXT an outcome turns the factor condition `UNMET` into `DISCHARGED` (which is cheaper to prove in a test than to run);
  an R0/R1/R2 arm after Huber breaks that prohibition; creating the two absent reviews changes the evidence line while the state
  correctly stays `NOT_DISCHARGEABLE_BY_ARTIFACTS`, because this seal never reads a review's presence as acceptance.
- The verdict rule is tested on synthetic condition lists in both directions.

Run with the adjacent E1 batteries: `tests/test_df_e1_seal.py tests/test_m4_c32_c38_reverify.py
tests/test_df_e1_phase2_acceptance.py tests/test_df_e1_chronology.py` — **65 passed in 116 s**.

## 7. Cost

CPU only, `CUDA_VISIBLE_DEVICES=''`, every job under `crispdm-run -m 4G/6G -t 300–900 -n e1seal`, anaconda env `trading-stack`
(Python 3.12.13). The seal itself runs in under a second; the test file in 0.33 s; the four-file run 116 s. No GPU, no allocation,
no training, no warehouse or governance contact, no host touched. Nothing under `docs/audits/evidence/d3_k5_20260917/` was
modified — the seal only reads it.

## 8. What I refused

1. **I refused to seal E1.** Five conditions are not met and three of them cannot be met from here. A complete seal would have
   required either asserting a review that does not exist or quietly dropping a condition from the list — and dropping a condition
   after seeing which ones fail is the same move as dropping a contrast after seeing its p-value.
2. **I refused to treat the ML-baselines review as the sealing authority.** It states no sealing condition (§1). Had I mined it for
   "conditions", I would have invented them; I quoted it instead and went to find the document that does gate the seal.
3. **I refused to fit Q2_CONTEXT**, which is the one gap a single authorised run would close. Its five pilots and 2,259.56 CPU s
   are retained; the cells were never fitted; the block says nothing about its question and the seal says that in those words.
4. **I refused to build a full-output ECL adapter**, and I refused to read the shared dataset name as a matched task — which is
   exactly what RP136–RP139 finding 4 warns against.
5. **I refused to regenerate the owner's closure table.** It exists, it was produced by the reviewed generator, and regenerating
   it would have meant either contacting the warehouse or recomputing errors from arrays. I bound it and checked its invariants.
6. **I refused to promote any `HISTORICAL_UNGOVERNED` unit.** The 15 units of the original pilot stay
   `SCIENTIFICALLY_VERIFIED_HISTORICAL_UNGOVERNED` in the seal, beside the 15 `VERIFIED_AND_GOVERNED` units of the successor.
7. **I refused to claim the drifted-code numbers reproduce.** §5.2.

## 9. What the owner or Musashi must still decide

1. **The two missing reviews.** `MUSASHI_RP49_RP56_REVIEW` and `MUSASHI_RP57_RP64_REVIEW` do not exist. The RP49–RP56 return
   declares MOD-E1 *"VERIFIED, not externally reviewed"* and the RP57–RP64 return asks for one review and never got a numbered
   one. Until they exist, `MOD_E1_EXTERNAL_REVIEW` cannot move, and with it `MOD-FROZEN-PREFIX`, `MOD-CORE-PRETRAIN`, `MOD-CONF`
   and `MOD-E3` stay `NOT_STARTED`. **This is the single decision that unblocks the most.**
2. **Whether to accept phase-2 v2** (`0dcb92b9…`, 36 cells, `SEALED_NOT_EXECUTED`) and the verifier guarantees the post-Huber
   review declined to accept unchanged.
3. **Whether to authorise the Q2_CONTEXT fits.** It is the one `UNMET` condition a bounded, budgeted CPU run would discharge, and
   the design is already sealed. I did not cost it, because costing an unauthorised run invites it.
4. **What to do about the legacy lineage.** `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND` is not a defect of E1 and no work
   here fixes it. It may need to be retired as a condition rather than kept as a permanent block — but retiring a condition is the
   owner's call, never mine.
5. **Whether to retain the Huber/AdamW v2 design document** (§5.1), so the four `huber_adamw_musashi` arms rest on bytes instead of
   a digest.
6. **Whether 13E is to be sealed at all**, and in which version. v1 is forbidden as a pretraining experiment; v2 corrected the
   R0/R1/R2 definitions; v3 (RP142, 2026-09-23) corrected the AE internal-validation paragraph. The seal observes the prohibition
   and takes no position on the successor.

One record-keeping note, for the same reason as the M4 return beside this one: the standing index says "next = E1 sealing after
review". The "after review" is load-bearing and the review has not happened. What exists now is this partial seal, which fixes
what retained bytes can fix so that the eventual review has something stable to rule on, and which names the five things it
cannot reach.
