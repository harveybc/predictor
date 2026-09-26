# Q2_CONTEXT, bounded: twelve fits, four arms, three seeds — and the question it still does not answer

Satoshi, successor technical lead. 2026-09-26. Acting on the owner's grant of 2026-09-26.
Worktree `predictor-e1q2-20260926`, branch `satoshi/e1-seal-q2-context-20260926`, from `941eb5b3`.
Evidence: [`docs/audits/evidence/E1_Q2_CONTEXT_BOUNDED_20260926/`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/).

> ## READ THIS HEADER OR READ NOTHING
>
> **What ran.** Twelve fits: four arms × three paired seeds, all CPU, all under `crispdm-run` with a measured memory cap,
> one cell resident at a time. All twelve stopped on validation, none was censored by the update ceiling, every one
> reproduced its checkpoint in a fresh process to `allclose(1e-6, 1e-6)`, and **every one beat persistence on the same
> 10 020 rows** (mean skill 0.186 to 0.199 by arm).
>
> **What it is NOT.**
>
> 1. **It does not answer Q2_CONTEXT's question.** The question is *"information beyond the hour, separated from depth and
>    from padding"*. Separating those needs the two W1440 **full-depth** arms — `long_window_own_depth` and
>    `long_window_local_support_67` — and neither is in this block. Their own retained cost pilots measured 4.165 and
>    4.443 CPU s per update at **8.46 GB and 10.28 GB peak RSS**: 17 507 s and 18 406 s per cell at the ceiling, and a
>    resident set that would take this host's memory away from its owner. The seal gap
>    `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` therefore **stays `UNMET`**, and the sealed design says so in its own
>    `question` field.
> 2. **No cell of `e1_block_q2_context_v1` was fitted by this round or any other.** This is a different block with a
>    different digest (`47a270eec01f203c…` against v1's `6d1aaecaf27c581c…`). Q2_CONTEXT v1's reading is unchanged:
>    *"pilots only; no cell of this block was fitted to a score, so it says nothing about its question — not 'no effect'."*
> 3. **Nothing here is verified end to end, and the owner closure table prints no number for it.** This host holds no
>    data-gov service key, so the governed runner refuses before opening data and no terminal was ever accepted.
>    `tools/df_closure_table.py`'s policy is that a score with no accepted terminal receipt is reported as **no model error
>    at all** — so the `owner_closure_table.v2` for this run has 12 rows, **0 verified**, every error column `null`, custody
>    `UNCHECKED` ×12. That table is published exactly as it landed (§N6), and the measured errors are published beside it in
>    an `unanchored_measurement_table.v1` that is never called verified (§N7).
> 4. **Nothing here is a verified causal effect.** The paired arm differences in §N4 are differences between forecast
>    errors. A causal claim does not predict a retained row, so no retained-row error verifies one. Three seeds, one task,
>    one previously inspected DEV validation week: development evidence. Both signs are printed and no interval is claimed
>    from n = 3. Nothing may promote, select or rank on any of it.
>
> **The headline numbers, and the one that matters most.**
>
> | arm | mean MAE kW | mean skill vs persistence | paired Δ MAE_z vs the W60 baseline | signs (+ / −) |
> |---|---:|---:|---|---|
> | `modular_w60` (baseline) | 0.499100 | 0.191573 | — | — |
> | `daily_lag` | 0.494401 | 0.199184 | −0.012938 · **+0.000587** · −0.003097 (mean −0.005149) | 1 / 2 |
> | `long_window_crop60` (exact-information null) | 0.499100 | 0.191573 | +0.000000 · +0.000000 · +0.000000 | 0 / 0 |
> | `short_window_deep_core` | 0.502205 | 0.186543 | +0.000402 · **+0.011518** · −0.001711 (mean +0.003403) | 2 / 1 |
>
> The daily-lag channel is smaller-error in two seeds of three and **larger in one**; the deeper dilated core is
> larger-error in two of three. Neither is a demonstrated effect on n = 3, and this document claims neither.
>
> **The strongest single result is the null.** `long_window_crop60` — the W1440 input cropped to its last 60 rows before
> the extractor — reproduced the W60 baseline's initial-weight digest in 3 of 3 seeds and its MAE **to the last bit** in
> 3 of 3 seeds. RP87 asserted that equivalence as a property of the code; here it is a live measurement on the block's own
> rows. A null that reproduces its treatment exactly is this block's positive control on its own plumbing — and it is the
> only claim in this document that does not depend on n = 3.
>
> ### Three defects found and what was done about each
>
> | # | defect | state |
> |---|---|---|
> | **D1** | **Q2_CONTEXT v1's prepared data does not honour its own design's declared train population.** The v1 design declares `train_population: COMMON_INTERSECTION` — *"every arm of this block, the baseline included, trains on the SAME origins so context is never conflated with volume"* — but the retained `BLOCK_DATA.npz` of `e1_block_q2_context_v1` holds **per-arm** train origins (40 020 / 38 700 / 38 700 / 38 700 / 40 080) and its record carries neither `train_admissible_before_intersection` nor `common_train_origins`. Had v1 ever been fitted on that preparation it would have conflated the input contrast with the very train volume it declares it controls. **It was never fitted, so no published number is affected.** Today's `prepare()` does implement the intersection and was exercised: this block's four arms all train on the **same 38 700** origins. **REPORTED, not repaired** — what to do with v1's retained preparation is the owner's call |
> | **D2** | **`tools/df_closure_table.py`'s authoritative verifier crashed with `FileNotFoundError` on a run with no accepted terminal at all.** `rows_from_run` read `TERMINAL_RECEIPTS.json` unguarded while its two sibling readers (`preparation_custody`, `_verify_fin_run`) guarded it, so it died before building a single row instead of reaching its own typed branch one line below (*"a registered forecast unit has NO accepted terminal receipt"*). An absent receipts file is the same state as an empty one. **REPAIRED** (one guard, matching its siblings) **and tested**: `tests/test_df_closure_table.py::test_a_run_with_no_receipts_file_at_all_is_typed_as_unanchored_and_never_raises` pins both halves — that it does not raise, and that it still reports **no number** |
> | **D3** | **The block closure's initial-weight pairing check reported a FALSE problem.** Its key was `(seed, family, channels, window)`, which omits `dilations` and `crop` — the two fields that change the built graph. `modular_w60` (8 127 parameters) and `short_window_deep_core` (12 047 parameters) therefore collided, and the closure reported *"unpaired initial weights within a seed for the same graph"* for two arms that are different models and **cannot** share initial weights. **REPAIRED** (the key now names every graph field) **and tested**: `tests/test_df_e1_block.py::test_2026_09_26_the_initial_weight_pairing_key_names_every_graph_field` shows different graphs no longer collide while the check's real intent — same graph, same seed, different input **data** must share initial weights (`calendar` vs `randomised_calendar_control`) — is unchanged. No previously closed block combined such arms, so no closed block's verdict moves |
>
> ### Cost, and what was not taken from the owner
>
> Prepare + 4 cost pilots + 12 fits + closure, all with `CUDA_VISIBLE_DEVICES=''`, every job through
> `$HOME/.local/bin/crispdm-run` with the cap chosen from `MemAvailable` read at launch (13.2–14.0 GiB free, 3 GiB host
> reserve): `-m 4G` for the seal, `-m 5G`/`-m 6G` for prepare, `-m 4G` for the pilots, **`-m 3G` for the twelve fits, one
> cell resident at a time** (measured peak RSS 0.87–1.06 GiB per cell), `-m 5G` for the closure. **No GPU, no service
> started, stopped or restarted, no reservation, no sweep, no financial fit.** Peak measured resident set of any single
> cell: 1.06 GiB. The two arms that would have needed 8.5 and 10.3 GiB were never started.

---

## 1. Why a bounded block at all, and why the restriction is not a removal after a score

`Q2_CONTEXT` v1 was sealed on 2026-09-21 and its five cost pilots ran; its own pilot report projected **111 000 CPU s** at
the ceiling against a 14 400 s campaign ceiling and closed as `BUDGET_LIMITED_BEFORE_ANY_OUTCOME` with **zero cells
fitted**. That decision is the tool's own gate: `df_e1_block.py execute` refuses unless `REPORT.pilot.json` says
`EXECUTE`, and I did not forge it.

Three further things made running v1 as sealed impossible on this host, all verified before anything was started:

1. **Its sealed code has drifted.** The v1 design pins `df_e1_block.py` at `798711c4…`; the file today is different, and
   `validate(design, strict_code=True)` — which `prepare`, `pilot` and `execute` all call — refuses on that drift.
2. **Its retained preparation contradicts its own declared train population** (D1). Fitting v1 on those arrays would have
   measured the wrong thing.
3. **Two of its five arms do not fit this host's memory** (header).

So a new block was **sealed before any cell of either block had a score**, on nothing but the two retained pilots' own
measurements, and committed as `8ca1869d` before the first fit: `Q2_CONTEXT_BOUNDED`, digest
`47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`. Same source run, same panel digest, same rows, same
pad, same scaler, same recipe (TIER1, patience 3), same cadence, same 4 000-update ceiling, same seeds, same benchmark
contract, same reading rules. Changed: the arm list, and nothing else.

The arms kept are the W60 baseline `modular_w60`, `daily_lag`, the exact-information null `long_window_crop60`, and
`short_window_deep_core`. The baseline is **fitted here, not reused** from `DEV_MATCHED`: under `COMMON_INTERSECTION` this
block's train origins are the four-arm intersection, which is not the population `DEV_MATCHED`'s baseline trained on, so
reusing it would have compared arms trained on different rows.

The sealed design's `informed_by` field carries the restriction and its reason verbatim, and §N1 reprints it. The design's
own `question` field states, before any number existed, that the block does **not** separate context from depth and does
**not** answer Q2_CONTEXT's question. Reading rule 3 — *"no cell is removed after its score is seen"* — is observed: the
restriction is a resource declaration made when no cell of either block had a score.

## 2. How the fits ran, given no service key

`df_e1_block.py`'s governed path raises before opening any data when `--api-key-file` is absent, which it is on this host.
Per `docs/GOVERNED_RUN.md`, *"Small mechanics tests may use other runners. Their results are non-governing and cannot
promote a model, transformation, feature or experiment."* `tools/df_e1_block_ungoverned.py` is such a runner. It reuses
`df_e1_block`'s own `validate`, `prepare`, `run_cell`, `baselines` and `close` unchanged and adds no science: no model, no
loss, no enumeration, no scoring, no scaler, no stopping rule of its own. It makes exactly one custody claim — that the
panel bytes it reads hash to the digest the design names, the digest a previous **governed** acquisition recorded as
`VERIFIED_TRANSFER` — and it labels that claim `BYTES_IDENTITY_ONLY`, not custody of a transfer. It writes no terminal, no
receipt and no delivery record, so nothing it produces can be mistaken for a governed artifact, and it writes
`UNGOVERNED_RUN.json` naming what it lacks.

The consequences are printed in §N6, §N8 and §N9 rather than argued away.

One further consequence, recorded because it is the same kind of fact the earlier blocks record: **D3's repair landed after
the design was sealed and after the twelve fits had run**, so the closure ran under a later revision of
`tools/df_e1_block.py` than the fits did. `REPORT.json` records that as `closure_code_drift` (sealed `24fdb67fca3c9878…`,
now `0708a56f71e9e997…`), §N8 prints it, and nothing here claims the closure reproduces under the sealed code. The twelve
fits do: every one of them ran under `24fdb67fca3c9878…`, the digest the design pins.

## 3. What a reader may and may not take from this

**May:** that four arms were fitted on 38 700 identical train origins and scored on 10 020 identical evaluation origins;
that each fit's error and the persistence error on those same rows are recomputed here from the retained arrays and agree
with the records to 1e-12; that all twelve beat persistence; that the exact-crop null reproduces its treatment bit for
bit; that the three declared references (persistence, daily-seasonal, train-constant) land where §N5 says, two of them
**worse** than persistence, published as they landed.

**May not:** that any arm is better than any other; that the daily-lag channel adds information; that depth at W60 helps
or hurts; that context beyond the hour matters or does not; that any of this is comparable to a published electric-load
number — every row is `NOT_COMPARABLE` with its reason and its planned matched comparison; or that any of it is verified,
governed, or a causal effect.

## 4. Numeric sections, generated from artifacts

Everything below is emitted by `tools/df_q2_context_tables.py` from `DESIGN.json`, `BLOCK_DATA.json`,
`REPORT.pilot.json`, `REPORT.json`, `BASELINES.json`, `REPLAYS.json`, `UNGOVERNED_RUN.json`, each cell's `arrays.npz` and
`cell.json`, and the `owner_closure_table.v2` produced by `tools/df_closure_table.py`. No number in it was typed by hand.
The generator refuses to emit anything if a recomputation disagrees with a record, if a model and its naive do not share
rows, horizon and scale, or if an arm is missing a seed.

Published copy: [`TABLES.md`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/TABLES.md) ·
[`CLOSURE_TABLE.json`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/CLOSURE_TABLE.json) ·
[`CLOSURE_TABLE.md`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/CLOSURE_TABLE.md) ·
[`UNANCHORED_MEASUREMENT_TABLE.json`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/UNANCHORED_MEASUREMENT_TABLE.json) ·
[`REDACTIONS.json`](../evidence/E1_Q2_CONTEXT_BOUNDED_20260926/REDACTIONS.json) (this repository is public; the machine's
host name is replaced by `<worker-host>` in the twelve places `REPORT.json` recorded it, and in nothing else).

---

### N1. The block as sealed

* design `47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`, schema `df_e1_block_design.v1`, block `Q2_CONTEXT_BOUNDED`, state at seal `SEALED_NOT_EXECUTED`, phase `DEVELOPMENT`
* tier: TIER1: RP66-RP73 blocks: patience 3 events (600 non-improving updates)
* prepared data `43c93db381de4e4281b0b00ae1b577e3f860ad29a477b13ac6dc86e2bbed5350`, panel rows 1410981..1462761 (pad 1380), common evaluation **10020 origins** (panel rows 41700..51719), sigma_evaluation 0.9125164391265214 kW
* train population `COMMON_INTERSECTION`: **38700 origins, identical for every arm**; subset of the source run's train origins: True; the 28 d baseline enumeration reproduces the source's train origins: True; the common evaluation equals the source's: True
* recipe: mae loss, adam, lr 0.003, batch 64, ceiling 4000 updates, validation every 200 observed updates, patience 3 events, restore_best True, min_delta 0.0
* scaler rule: COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd
* common evaluation rule: the intersection over the block's arms of admissible validation origins, with a finite label and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it

| arm | window | features | dilations | crop | role | parameters | per-arm train admissible before the intersection |
|---|---:|---|---|---:|---|---:|---:|
| `modular_w60` | 60 | base | — | — | ARM | 8127 | 40080 |
| `daily_lag` | 60 | daily_lag | — | — | ARM | 8208 | 40020 |
| `long_window_crop60` | 1440 | base | — | 60 | EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor | 8127 | 38700 |
| `short_window_deep_core` | 60 | base | [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] | — | ARM | 12047 | 40080 |

**Reading rules, verbatim from the sealed design:** *three seeds on one task are development evidence* · *a fit that reached the update ceiling is CENSORED wherever its best checkpoint fell* · *no cell is removed after its score is seen* · *a published number under another protocol never enters the comparison column*

**The block's own question, as sealed:** the part of the Q2_CONTEXT question a 30 GiB host can fit: does a causal daily-lag channel, or a deeper dilated core at W60, change error against the W60 baseline and against the W1440 exact-crop information null? The two W1440 FULL-DEPTH arms of Q2_CONTEXT (long_window_own_depth, long_window_local_support_67) are NOT in this block, so this block does NOT separate context from depth and does NOT answer the Q2_CONTEXT question: it measures the three arms whose retained cost pilots fit this host, and leaves the long-window treatment unmeasured

**Why the two W1440 full-depth arms of Q2_CONTEXT are not in this block, verbatim from the sealed design:** RESTRICTED BEFORE ANY SCORE from the arms of Q2_CONTEXT v1 (design 6d1aaecaf27c581c709a745a4f976c2e9dcc05594815b2ee1a9747595f4398b1, state BUDGET_LIMITED_BEFORE_ANY_OUTCOME, zero cells fitted), on the two RETAINED cost pilots' own measurements and on nothing else: long_window_own_depth 4.165 CPU s per update and 8 458 399 744 B peak RSS, long_window_local_support_67 4.443 CPU s per update and 10 279 276 544 B peak RSS, i.e. 17 507 s and 18 406 s per cell at the 4 000-update ceiling and a resident set that would take this host's memory away from its owner; the three arms kept cost 287 s, 312 s and 488 s per cell at the ceiling with peak RSS under 1 GiB. The restriction is a resource declaration made before any score of any cell existed, never a removal after a score was seen (reading rule 3); no arm, seed, recipe, scaler, row, cadence or ceiling of Q2_CONTEXT v1 is otherwise changed

### N2. The cost projection the block executed on

* four cost pilots spent 82.4 CPU s; projection at the 4 000-update ceiling 2654.3 CPU s, with 25 % headroom 3317.9 CPU s; campaign ceiling 14400 CPU s, closure reserve 2000 CPU s
* decision **EXECUTE** (`fits_the_ceiling`: True) — contrast with Q2_CONTEXT v1's own decision, `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`

| arm | CPU s per update (pilot) | peak RSS GiB (pilot) | projected CPU s per cell at the ceiling |
|---|---:|---:|---:|
| `modular_w60` | 0.0390 | 0.87 | 179.8 |
| `daily_lag` | 0.0392 | 0.87 | 180.9 |
| `long_window_crop60` | 0.0410 | 0.87 | 197.8 |
| `short_window_deep_core` | 0.0710 | 0.90 | 326.3 |

### N3. Every fit, as it landed

Errors recomputed here from each cell's retained `arrays.npz`, each one cross-checked against the value the cell record stored (a disagreement above 1e-12 refuses the whole table).

| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | stop | censoring | updates | best update | CPU s | peak RSS GiB | reload max err | fresh-process replay |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| `modular_w60_s1` | `modular_w60` | 1 | 0.552467 | 0.504135 | 0.617372 | 0.183418 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 54.1 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s1` | `daily_lag` | 1 | 0.539528 | 0.492328 | 0.617372 | 0.202542 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1600 | 1000 | 50.6 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s1` | `long_window_crop60` | 1 | 0.552467 | 0.504135 | 0.617372 | 0.183418 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 62.3 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s1` | `short_window_deep_core` | 1 | 0.552868 | 0.504501 | 0.617372 | 0.182824 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1400 | 800 | 93.7 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |
| `modular_w60_s2` | `modular_w60` | 2 | 0.537390 | 0.490377 | 0.617372 | 0.205702 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 3000 | 2400 | 82.5 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s2` | `daily_lag` | 2 | 0.537977 | 0.490913 | 0.617372 | 0.204834 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 2000 | 1400 | 62.6 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s2` | `long_window_crop60` | 2 | 0.537390 | 0.490377 | 0.617372 | 0.205702 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 3000 | 2400 | 95.9 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s2` | `short_window_deep_core` | 2 | 0.548908 | 0.500888 | 0.617372 | 0.188677 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 120.1 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |
| `modular_w60_s3` | `modular_w60` | 3 | 0.550991 | 0.502789 | 0.617372 | 0.185599 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 59.0 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s3` | `daily_lag` | 3 | 0.547895 | 0.499963 | 0.617372 | 0.190176 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 60.7 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s3` | `long_window_crop60` | 3 | 0.550991 | 0.502789 | 0.617372 | 0.185599 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 66.3 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s3` | `short_window_deep_core` | 3 | 0.549280 | 0.501227 | 0.617372 | 0.188128 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 122.4 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |

Censoring across the twelve fits: `STOPPED_ON_VALIDATION`. Initial-weight digests, per seed: seed 1: `modular_w60` b9caefebdc3c…, `daily_lag` 0f6b46da2c89…, `long_window_crop60` b9caefebdc3c…, `short_window_deep_core` 9d7edff6f248… · seed 2: `modular_w60` 84f67bc65cb5…, `daily_lag` f9c474649f56…, `long_window_crop60` 84f67bc65cb5…, `short_window_deep_core` 1a6308265c88… · seed 3: `modular_w60` e0e2765cded8…, `daily_lag` 7fc47852dea4…, `long_window_crop60` e0e2765cded8…, `short_window_deep_core` 59e492f0e56f…

### N4. Per arm, and the paired difference against the baseline arm

| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | paired Δ MAE_z vs `modular_w60`, per seed | mean Δ | signs (+ / −) |
|---|---:|---:|---:|---:|---|---:|---|
| `modular_w60` | 0.546949 | 0.008311 | 0.499100 | 0.191573 | — (this is the baseline arm) | — | — |
| `daily_lag` | 0.541800 | 0.005335 | 0.494401 | 0.199184 | -0.012938 · +0.000587 · -0.003097 | -0.005149 | 1 / 2 |
| `long_window_crop60` | 0.546949 | 0.008311 | 0.499100 | 0.191573 | +0.000000 · +0.000000 · +0.000000 | 0.000000 | 0 / 0 |
| `short_window_deep_core` | 0.550352 | 0.002187 | 0.502205 | 0.186543 | +0.000402 · +0.011518 · -0.001711 | 0.003403 | 2 / 1 |

Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and **no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal effect: no causal claim here is verified against a retained-row error, because a causal claim does not predict a retained row.

**The exact-information null, measured.** `long_window_crop60` is the W1440 input cropped to its last 60 rows before the extractor: by RP87 it is the SAME computation as `modular_w60` on the same origins. Measured here rather than assumed: identical initial-weight digest in 3 of 3 seeds, and MAE equal to the baseline's to the last bit in 3 of 3 seeds. A null that reproduces its treatment exactly is the block's own positive control on its plumbing.

### N5. The three declared references, on the same rows

Computed by `df_e1_block.baselines` on the block's 10020 common evaluation origins. The block's closure suppresses these when it fails, so they are published separately.

| reference | definition | MAE kW | MAE_z | skill vs persistence |
|---|---|---:|---:|---:|
| `persistence` | y(t) | 0.617372 | 0.676560 | 0.000000 |
| `daily_seasonal` | y(t+h-1440) | 0.731659 | 0.801804 | -0.185119 |
| `train_constant` | mean of the train labels of the 28 d tier; computed on the common evaluation set at closure, no fit, no terminal | 0.709950 | 0.778013 | -0.149955 |

Published exactly as they landed: `daily_seasonal`, `train_constant` are **worse** than persistence on these rows.

### N6. The owner closure table as it landed

`owner_closure_table.v2` from `tools/df_closure_table.py`, generated 2026-09-26T12:24:14Z: **12 rows, 0 verified, 0 preserved with a qualified scope**; custody classes {"UNCHECKED": 12}; preparation classes {"PREPARATION_LOCAL_ONLY": 12}.

**This is the load-bearing fact about this run, and it is printed before any number of mine.** The verifier's policy is that a score with **no accepted terminal receipt is not reported as a model error at all** — not as a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, so every error column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed.

| unit | task / horizon / split | metric and scale | model error | naive error | skill | literature value + source | placed in the comparison column | comparability | custody | binding |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| `modular_w60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |

**Why NOT_COMPARABLE, verbatim:** unknown identity fields cannot match as proof: ['target_transform']

**Planned matched comparison, verbatim:** read the primary source (or its code) and fill the field; a placeholder is not a protocol

**Every problem the table recorded, in full:**

* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt

### N7. The unanchored measurement table

`unanchored_measurement_table.v1`. It supplies the five columns the owner's rule names, for a run the owner closure table can only print as `null`. Each error is recomputed from the cell's own retained arrays and cross-checked against the record's stored score; the contract columns are taken from the landed owner table above. **It is not an `owner_closure_table.v2` row, it is never verified, and its custody is `UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.** Nothing may promote, select or rank on it.

| unit | metric and scale | model error | naive error, SAME rows | skill | rows (model / naive) | horizon (model / naive) | scale (model / naive) | literature value + source | comparability | custody |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `modular_w60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504135 | 0.617372 | 0.183418 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.492328 | 0.617372 | 0.202542 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504135 | 0.617372 | 0.183418 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504501 | 0.617372 | 0.182824 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490377 | 0.617372 | 0.205702 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490913 | 0.617372 | 0.204834 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490377 | 0.617372 | 0.205702 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.500888 | 0.617372 | 0.188677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502789 | 0.617372 | 0.185599 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.499963 | 0.617372 | 0.190176 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502789 | 0.617372 | 0.185599 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.501227 | 0.617372 | 0.188128 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |

**Fits that landed worse than their naive reference: 0 of 12.** Every fit's skill is printed above whatever its sign.

### N8. The closure as it landed

* `df_e1_block_report.v3`, design `47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`, block `Q2_CONTEXT_BOUNDED`
* **`verified`: False**
* common evaluation rows 10020; sigma_evaluation 0.9125164391265214; spent CPU 1012.7 s
* closure code drift: `{"df_e1_block.py": {"sealed": "24fdb67fca3c98781de1c1b56f69ff3dce02d8469f48e3de5c39991d1440d97f", "now": "0708a56f71e9e997d52490cd3fa14a487fae44a5172017b5788809455c575a99"}}`
* scope: DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no test rows read
* disposition: `{"disposition": "HISTORICAL_DEV_ONLY", "task_id": "uci_235.W60_h60.DEV_28d_7d", "policy": "docs/tres_temas_entrevista/program_v3/SOTA_FIRST_2026_09_21.md", "why": "previous exploratory pilot (household task / adapted models): preserved with its receipts and failures, excluded from active selection, ranking and recommendations"}`; active_selection: `null`
* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no paired contrast and no selected arm. The per-arm means and the paired differences in N4, and the references in N5, are therefore published OUTSIDE the closure, recomputed from the arrays.

**Every problem the closure recorded, in full:**

* daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt
* closure without a warehouse read: no accepted custody, nothing is verified

### N9. Governance, stated as it is

* classification **NON_GOVERNING**
* data_gov_acquisition: **ABSENT**
* accepted_terminal: **ABSENT**
* receipt: **ABSENT**
* warehouse_read: **ABSENT**
* why: **no data-gov service key is held on this host; the governed runner refuses before opening data, so this driver ran instead and its results promote nothing**
* every closure row's custody is UNCHECKED: no accepted payload anchors the score
* the preparation's custody is PREPARATION_LOCAL_ONLY
* the closure's `verified` flag is False for that reason and for no other unless it names one
* these numbers are DEVELOPMENT measurements and cannot select, rank or promote anything
* data custody `BYTES_IDENTITY_ONLY`, panel sha256 `b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db`, 10890295 bytes — the file's sha256 equals the design's source_run.panel_sha256, which a previous GOVERNED acquisition recorded as VERIFIED_TRANSFER; this driver re-verified the BYTES, not the transfer, and holds no delivery id, no availability contract and no acceptance
* interpreter: Python 3.12.13 (anaconda env `trading-stack`)


---

## 5. What stays refused after these fits

| item | state | reason |
|---|---|---|
| answering Q2_CONTEXT's question | **REFUSED** | the two W1440 full-depth arms are unfitted; `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` stays `UNMET` |
| claiming the daily-lag channel adds information | **REFUSED** | 1 of 3 seeds went the other way; n = 3, no interval |
| claiming depth at W60 helps or hurts | **REFUSED** | 2 of 3 seeds went the other way; n = 3, no interval |
| calling any of this verified | **REFUSED** | 0 of 12 owner-closure-table rows verified; no accepted terminal, no warehouse read |
| calling any difference a causal effect | **REFUSED** | a causal claim predicts no retained row, so no retained-row error verifies it |
| placing any number beside a published electric-load value | **REFUSED** | every row `NOT_COMPARABLE`, with its reason and planned matched comparison |
| promoting, selecting or ranking anything | **REFUSED** | `disposition: HISTORICAL_DEV_ONLY`, `active_selection: null`, NON_GOVERNING |
| fitting Q2_CONTEXT v1 on its retained preparation | **REFUSED** | D1: that preparation contradicts the design's declared train population |

## 6. What the owner or Musashi must decide

1. **A host that can hold 10.3 GiB, or a decision not to measure the long-window treatment.** This is what
   `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` needs and what this host cannot give. The two arms' per-cell cost at the
   ceiling is 17 507 s and 18 406 s; six cells is roughly 30 CPU hours.
2. **A data-gov service key for the successor, or an explicit decision that successor measurements stay NON_GOVERNING.**
   Without one, every fit taken here lands with `0 verified` rows by construction, and the owner's closure table can print
   no error for it. That is the honest scope, and it is the weakest one available.
3. **What to do with Q2_CONTEXT v1's retained preparation** (D1): re-prepare, supersede, or retire. It was never fitted, so
   nothing published rests on it.
4. **Whether to accept the two verifier repairs** (D2, D3). Both are tested and neither moves a closed block's verdict, but
   both touch the authoritative verifier and its closure, and accepting a verifier change is a review, not an artifact.
5. **Whether this bounded block should be re-run governed** once a key exists, so its numbers can carry custody. The
   design is sealed and the preparation is deterministic, so a governed re-run is a re-execution of a fixed design, not a
   new experiment.

---

Satoshi, successor technical lead — 2026-09-26, on the owner's grant of 2026-09-26. Nothing in this document is written
under Musashi's name, and no review is signed on his behalf.
