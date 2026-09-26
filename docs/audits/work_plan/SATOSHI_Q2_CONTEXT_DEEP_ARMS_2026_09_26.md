# Q2_CONTEXT_DEEP: the two full-depth arms were started, one was killed, and the question is still not answered

Satoshi, successor technical lead. 2026-09-26. Acting on the owner's grant of 2026-09-26 and on the coordinator's
stand-down of the same day. Nothing in this document is issued under Musashi's name.
Worktree `predictor-q2deep-20260926`, branch `satoshi/q2-context-deep-arms-20260926`, from `7d2b1c83`.
Evidence: [`docs/audits/evidence/E1_Q2_CONTEXT_DEEP_20260926/`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/).

> ## THE ANSWER FIRST
>
> **`ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` remains `UNMET`.** The named reason is not budget and not
> architecture: **the two W1440 full-depth arms await shared admission control on this host.** Six of the eighteen
> registered cells were never fitted — `long_window_own_depth_s1/s2/s3` and `long_window_local_support_67_s1/s2/s3` — so
> **the context contrast does not exist**, and no partial answer to the context question is manufactured from arms that
> did not run.
>
> **What the crossing needed and what it got.** The design's own crossing is raw input window {60, 1440} × core depth
> {5, 10} with volume held fixed by construction. Three of its four cells are filled; **the two that carry long context
> are empty**:
>
> | | depth 5 blocks | depth 10 blocks |
> |---|---|---|
> | **raw window 60** | `modular_w60` — mean MAE_z **0.563796** | `short_window_deep_core` — mean MAE_z **0.565225** |
> | **raw window 1440** | `long_window_local_support_67` — **NOT FITTED** | `long_window_own_depth` — **NOT FITTED** |
>
> | declared contrast | result |
> |---|---|
> | context at matched depth 10 — `long_window_own_depth − short_window_deep_core` | **NOT_MEASURED**; missing `long_window_own_depth_s1/s2/s3` |
> | context at matched depth 5 — `long_window_local_support_67 − modular_w60` | **NOT_MEASURED**; missing `long_window_local_support_67_s1/s2/s3` |
> | depth at matched context 1440 | **NOT_MEASURED**; all six deep cells missing |
> | interaction | **NOT_MEASURED**; all six deep cells missing |
> | depth at matched context 60 — `short_window_deep_core − modular_w60` | +0.011424 · +0.001678 · −0.008814, mean **+0.001429**, signs **2 / 1** — a direction on n = 3, **not an effect**, and both arms are `UNDERTRAINED_AT_CEILING` |
> | the causal daily-lag channel — `daily_lag − modular_w60` | −0.008649 · −0.005785 · −0.002357, mean **−0.005597**, signs **0 / 3** — smaller error in 3 of 3 seeds, still **a direction on n = 3 and not an effect**, and its baseline is `UNDERTRAINED_AT_CEILING` |
> | the exact-information null — `long_window_crop60 − modular_w60` | **+0.000000 in 3 of 3 seeds**, identical initial-weight digest in 3 of 3: the one claim here that does not depend on n = 3 |
>
> **What was delivered instead of an answer.**
>
> 1. **A sealed six-arm design extension that stands on its own** — `Q2_CONTEXT_DEEP`, digest
>    `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318`, sealed and committed at `b1a13928` **before any
>    cell of it had a score**, with both W1440 full-depth arms in it and a derived crossing record.
> 2. **The order's own question about the train origins, answered as arrays.** The two new arms **can** share the
>    identical train origins with the existing four: all six arms' `train_origins__*` arrays are `np.array_equal` to the
>    bounded block's **38 700**, the common evaluation is `np.array_equal` to its **10 020** and equal to the source run's,
>    and `Xs`, `Y`, `lag`, `calendar` and `ts_ns` are identical. Volume is not a factor of this block at all. No lesser
>    comparison had to be sealed.
> 3. **Twelve fits that did land, closed with their tables.** Four arms × three seeds, all on those identical rows, all
>    `CENSORED_BY_BUDGET` at the 600-update ceiling, **all twelve beating persistence** (mean skill 0.1646–0.1749), every
>    one reproduced in a fresh process to `allclose(1e-6, 1e-6)` with a **maximum absolute difference of 0.000e+00 kW over
>    all twelve**.
> 4. **The placement failure, reproduced rather than papered over**, with the day's four terminations recorded by
>    mechanism and the cost basis corrected.
>
> **What is NOT claimed.** No context effect, in either direction. No depth effect. No effect of the lag channel. Nothing
> verified: this host holds no data-gov service key, so no terminal was ever accepted and
> `tools/df_closure_table.py`'s policy reports **no model error at all** — the `owner_closure_table.v2` for this run has
> **18 rows, 0 verified, every error column `null`, custody `UNCHECKED` ×18**. That table is published exactly as it
> landed (§N8) and the measured errors are published beside it in an `unanchored_measurement_table.v1` that is never
> called verified (§N9). The verifier's custody policy was not touched, and no credentials were looked for.

---

## 1. Why the block had to be re-sealed rather than extended in place, and what changed

The seal gap has stood `UNMET` for one reason: nothing separates **input context length** from **model depth** from
**training volume**, so every apparent context effect is confounded. The two arms that would separate them —
`long_window_own_depth` and `long_window_local_support_67` — were never started, because their retained cost pilots
measured **4.165 and 4.4427 CPU s per train update** at **8 458 399 744 B and 10 279 276 544 B peak RSS**
(`MAIN_PROCESS_RSS_ONLY`; see §5), i.e. 17 507 s and 18 406 s of CPU **per cell** at the 4 000-update ceiling.

A design is sealed by the sha256 of its own body, so "extending" the bounded block's `DESIGN.json` produces a different
design either way. What this round did instead is stricter: it sealed the **six-arm** block `Q2_CONTEXT_DEEP` — the four
arms of `Q2_CONTEXT_BOUNDED` plus both W1440 full-depth arms — and set out to refit **all six**, so that no arm would ever
be compared across budgets. Two fields differ from the bounded block and no others: the arm list, and the recipe's update
ceiling (4 000 → **600** observed updates), with the resource limits a multi-GiB resident cell needs. Source run, panel
digest, rows, pad, scaler, `COMMON_INTERSECTION` train population, common-evaluation rule, loss, optimizer, learning rate,
batch, validation cadence (every 200 observed updates), patience (3 events), `min_delta`, `restore_best`, monitor, metric,
scale and the three seeds are identical.

The ceiling came from the **retained** pilots' own measured rates and from nothing else, it was fixed **before any cell of
this block had a score**, and its cost is stated in the sealed design's own `question` field rather than discovered
afterwards: with 3 checkpoint opportunities and patience 3, patience can never expire, so **every cell of every arm
reaches the ceiling and is `CENSORED_BY_BUDGET`**. The design also carries, sealed before the first fit, an
`UNDERTRAINED_AT_CEILING` rule that refuses to read a contrast as an effect for an arm still improving faster than 0.005
MAE_z over its last 200 updates in 2 or more of its 3 seeds. §N6 shows that rule firing on three of the four arms that
landed — including the baseline — which is why §N5's measured rows are directions and not effects.

## 2. What the crossing can identify, and the one thing the architecture family forbids

[`EXTENSION_SEAL.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/EXTENSION_SEAL.json), committed with the design before
any fit, **derives** the crossing from the sealed arms instead of asserting it. A causal kernel-3 dilated stack reaches
`1 + 2·Σd` samples, so an arm's usable context is `min(window_after_crop, reach)`:

| arm | raw window | core depth | core reach | usable context | parameters |
|---|---:|---:|---:|---:|---:|
| `modular_w60` | 60 | 5 | 63 | 60 | 8 127 |
| `long_window_crop60` | 1440 → crop 60 | 5 | 63 | 60 | 8 127 |
| `daily_lag` | 60 | 5 | 63 | 60 (+ the lag channel) | 8 208 |
| `long_window_local_support_67` | 1440 | 5 | 63 (67 with the branch extractor, per the arm's sealed role) | 63 | 8 127 |
| `short_window_deep_core` | 60 | 10 | 2 047 | 60 | 12 047 |
| `long_window_own_depth` | 1440 | 10 | 2 047 | 1 440 | 12 047 |

Two things follow, and the second is the honest limit of the design:

1. **The context contrast would have been clean.** `long_window_own_depth` and `short_window_deep_core` have the same core
   depth, **the same 12 047 parameters**, the same train origins, the same evaluation rows, the same scaler, the same
   recipe and the same seeds, and differ in exactly one thing: how many raw rows the window contains and therefore how
   many the core can use — 1 440 against 60. That is precisely the contrast the seal gap asks for, and it is the one that
   did not run.
2. **Context beyond 63 samples is confounded with depth by construction of the family.** There is no way to give a
   kernel-3 dilated core 1 440 samples of reach with 5 blocks; reach *is* a function of depth. So the `(1440, depth 5)`
   cell carries the long window's padding and 63–67 samples of reach, **not** 1 440 samples of information, and the
   depth-5 row of the crossing could never have been read as a context row. This was stated in the sealed design's
   `question` field before any number existed.

## 3. The placement failure, reproduced

The order I was given was to read `MemAvailable` immediately before each launch and refuse below the arm's measured pilot
peak plus 1 GiB. That is implemented as code in
[`tools/df_memory_gated_run.py`](../../../tools/df_memory_gated_run.py), which logs every reading with its verdict to
[`MEMORY_GATE.jsonl`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/MEMORY_GATE.jsonl) — 81 readings, 13 launches, 55 holds.
**The coordinator's finding is that this pattern is defective, and this round reproduced the defect rather than
discovering it by argument:**

* `pilot_long_window_own_depth` was launched at the **highest `MemAvailable` reading of the whole session, 13.06 GiB**,
  under `crispdm-run -m 9G`. It ran 23 minutes and was then killed. The reading held no commitment for the duration of
  the load, so it admitted a job it could not protect. Other agents' jobs (`crispdm-resol`, `crispdm-defrep`,
  `crispdm-df-e1-*`, a `crispdm-huntdgst` digest hunt) arrived on the same host after the reading was taken.
* `long_window_own_depth_s1` was then **held for 1 395 s of polling and never launched**, because the gate would not admit
  it. The stand-down stopped the poller and its retry loop. **No out-of-memory termination was converted into a retry
  with a bigger cap, and no arm that had not already started was started.**
* One reading of §N13 needs a caveat the generator cannot give itself: its counter says *"0 units NOT started
  because memory never allowed it"*, and that zero is an artefact of how the run ended, not a claim that nothing
  was left unstarted. The gate writes its `NOT_STARTED_MEMORY_NEVER_ALLOWED_IT` verdict only when its own wait
  budget expires; the stand-down stopped the poller first, so the verdict was never written. **Six cells were not
  started**, they are named in §N3 and in `TERMINATIONS.json`, and their last recorded gate verdict is
  `HELD_WAITING_FOR_MEMORY`.

[`TERMINATIONS.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/TERMINATIONS.json) records the day's **four** terminations
from systemd's own journal, with the two mechanisms kept apart:

| # | scope | mechanism | mine | cost lost |
|---|---|---|---|---|
| 1 | `crispdm-q2ctx-…-4121087` (12:59:55Z) | **CGROUP_MEMORY_MAX** (`Failed with result 'oom-kill'`, no oomd line) | no — earlier round | not recorded by this session |
| 2 | `crispdm-q2ctx-…-4121086` (13:14:07Z) | **CGROUP_MEMORY_MAX** | no — earlier round | not recorded by this session |
| 3 | `crispdm-huntdgst-…-429546` (18:10:54Z) | **USER_SESSION_MEMORY_PRESSURE** — oomd, 55.11 % > 50 % for > 20 s | no — another agent | not recorded by this session |
| 4 | `crispdm-q2deep-…-443210` (18:34:46Z) | **USER_SESSION_MEMORY_PRESSURE** — oomd, 55.45 % > 50 % for > 20 s | **yes** — `pilot_long_window_own_depth`, attempt 1, `-m 9G` | **1 292.106 CPU s / 1 389.019 s wall, no artifact, no number** |

Termination 4 was **not** a cap kill: the scope's cgroup peak at the kill was **7.4 G, 1.6 GiB below its own MemoryMax**.
Its `Pgscan` of 61 467 701 shows the job was being reclaimed against `MemoryHigh` (8.1 GiB, which `crispdm-run` sets to
90 % of the request) while the session-wide PSI pressure that systemd-oomd actually acts on crossed its limit. **No
per-job cap prevents that, because the trigger is the session's aggregate state, which no single job controls.** That is
the structural reason the deep arms await *shared* admission control: what is missing is a reservation held for the
duration of a load across all agents on the host, not a bigger number in one job's cap.

## 4. What may and may not be taken from the twelve fits that landed

**May:** that four arms were fitted on 38 700 identical train origins and scored on 10 020 identical evaluation origins,
checked as arrays; that each fit's error and the persistence error on those same rows are recomputed from the retained
arrays and agree with the records to 1e-12; that all twelve beat persistence; that the exact-crop null reproduces its
treatment bit for bit in 3 of 3 seeds; that the three declared references land where §N7 says, **two of them worse than
persistence**, published as they landed; that a 600-update ceiling costs a W60 arm about **+0.015 to +0.017 MAE_z**
against the bounded block's 4 000-update ceiling (§N10).

**May not:** that context beyond the hour matters or does not — **that contrast does not exist in this round**; that
depth at W60 helps or hurts; that the daily-lag channel adds information; that any arm is better than any other; that any
of this is comparable to a published electric-load number — every row is `NOT_COMPARABLE` with its reason and its planned
matched comparison; or that any of it is verified, governed, or a causal effect.

[`ASYMMETRIC_READING.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/ASYMMETRIC_READING.json) was committed at `91ad13a5`
while the twelve cheap cells were in hand and **no** long-window cell existed, fixing in advance how a long-window result
would have to be read given that the matched-update budget biases the contrast against it. **It never fired**, because no
long-window cell was ever produced. It stands as the rule for the round that runs them.

## 5. Two defects, one of them mine

| # | defect | state |
|---|---|---|
| **D1** | **The cost basis this whole lane placed jobs on is main-process RSS, not the process tree or its cgroup.** `df_e1_block.run_cell` records `resource.getrusage(RUSAGE_SELF).ru_maxrss` as `cost.peak_rss_bytes`. That figure excludes the parent driver in the same cgroup, any child, and the cgroup's page and kernel memory — and it is what the bounded block's restriction argument, this block's sealed `informed_by`, and this round's gate thresholds all used. The correct basis for the same arm, the cgroup peak of its whole scope, is **7.4 G** where the main-process figure is 7.88 GiB. [`FOOTPRINT_BASIS.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/FOOTPRINT_BASIS.json) lists every figure with what measured it, labels each one, and reports without explaining away that for several scopes the main-process high-water mark **exceeds** the cgroup peak systemd recorded. **REPORTED, not repaired**: repairing `run_cell` changes a file the sealed design pins, and the block must stay runnable for the arms that remain |
| **D2** | **I wrote the coordinator's host name into the sealed design.** `AGENTS.md` forbids a machine host name in this public repository; the block definition I added to `tools/df_e1_block.py` names the host, and the text is copied verbatim into `DESIGN.json` and `EXTENSION_SEAL.json`. **REPORTED, not repaired**, for three structural reasons recorded in [`REDACTIONS.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/REDACTIONS.json): the text is inside the body the design's sha256 covers, so redacting the published copy breaks the digest that makes the seal checkable; editing the source changes a digest the design pins in `source_code`, and `validate(strict_code=True)` would then refuse the design whose remaining arms must still be runnable; and re-sealing is unavailable once cells have scores. Every other published artifact is redacted (`REPORT.json` 12 occurrences, `TABLES.md` 1), and the name's seven prior occurrences in already-committed files of this repository are listed so the owner can set one policy rather than one per round |

## 6. Cost, and what was not taken from the owner

Seal, crossing derivation, prepare, 4 cost pilots, 12 fits, 1 terminated pilot, baselines, closure with 12 fresh-process
replays, closure table and tables: all with `CUDA_VISIBLE_DEVICES=''`, every job through
`$HOME/.local/bin/crispdm-run` with a cap chosen from a live `MemAvailable` reading, **one job resident at a time**
(`parallel_children: 1`). Caps used: `-m 6G` seal and prepare, `-m 4G` pilots and the twelve fits, `-m 5G` closure and
closure table, `-m 9G`/`-m 10G`/`-m 11G`/`-m 12G` requested for the deep units. Largest measured resident set of any cell
that landed: **1.06 GiB** (`MAIN_PROCESS_RSS_ONLY`). **No GPU, no service started, stopped or restarted, no reservation,
no sweep, no financial fit, no running process of anyone else touched, and no guard bypassed.** Cost lost to the one
termination: 1 292.106 CPU s.

## 7. What the next round needs, precisely

1. **Shared admission control** on this host: a reservation held for the duration of a load, visible to every agent, so
   two 8 GiB requests cannot both be admitted against 12 GiB of headroom. Until it exists, the six deep cells are not
   startable here, and this document does not propose starting them anyway.
2. **A cost pilot that records the cgroup peak of its own scope** (D1), so a placement decision rests on the process tree.
3. Then the six cells of `Q2_CONTEXT_DEEP` as sealed — the design, its crossing, its contrasts, its undertraining rule and
   its asymmetric reading rule are all committed and need no revision — followed by the crossing in §N5 with its
   `NOT_MEASURED` rows filled in. `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` is met on that day and not before.

## 8. Numeric sections, generated from artifacts

Everything below is emitted by [`tools/df_q2_deep_tables.py`](../../../tools/df_q2_deep_tables.py) from `DESIGN.json`,
`EXTENSION_SEAL.json`, `BLOCK_DATA.json`, `REPORT.json`, `BASELINES.json`, `REPLAYS.json`, `UNGOVERNED_RUN.json`,
`MEMORY_GATE.jsonl`, each cell's `arrays.npz` and `cell.json`, and the `owner_closure_table.v2` produced by
`tools/df_closure_table.py`. No number in it was typed by hand. The generator refuses to emit anything if a recomputation
disagrees with a record, if a model and its naive do not share rows, horizon and scale, if two arms did not score the
identical origins, or if a contrast names an arm the block does not have; a registered cell that was never fitted is
carried as `NOT FITTED` and named, and every contrast it blocks is printed as `NOT_MEASURED` with the missing cells listed.

Published copy: [`TABLES.md`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/TABLES.md) ·
[`CLOSURE_TABLE.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/CLOSURE_TABLE.json) ·
[`CLOSURE_TABLE.md`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/CLOSURE_TABLE.md) ·
[`UNANCHORED_MEASUREMENT_TABLE.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/UNANCHORED_MEASUREMENT_TABLE.json) ·
[`REDACTIONS.json`](../evidence/E1_Q2_CONTEXT_DEEP_20260926/REDACTIONS.json).

---

### N1. The block as sealed

* design `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318`, schema `df_e1_block_design.v1`, block `Q2_CONTEXT_DEEP`, state at seal `SEALED_NOT_EXECUTED`, phase `DEVELOPMENT`
* tier: TIER1 CADENCE AND PATIENCE, BUDGET-MATCHED CEILING: RP66-RP73 blocks: patience 3 events (600 non-improving updates); ceiling 600 observed updates for every arm of the block, see recipe.budget_declaration
* prepared data `2b50b777dab1cf5ba9443b9b08533c6c819150a22b1d515c99914b44aeb71ff9`, panel rows 1410981..1462761 (pad 1380), common evaluation **10020 origins** (panel rows 41700..51719), sigma_evaluation 0.9125164391265214 kW
* train population `COMMON_INTERSECTION`: **38700 origins, identical for every arm**; subset of the source run's train origins: True; the 28 d baseline enumeration reproduces the source's train origins: True; the common evaluation equals the source's: True
* recipe: mae loss, adam, lr 0.003, batch 64, ceiling **600 updates**, validation every 200 observed updates, patience 3 events, restore_best True, min_delta 0.0
* scaler rule: COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd
* common evaluation rule: the intersection over the block's arms of admissible validation origins, with a finite label and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it

| arm | raw window | crop | usable context samples | core depth (dilated blocks) | core receptive field | features | role | parameters | per-arm train admissible before the intersection |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| `modular_w60` | 60 | — | 60 | 5 | 63 | base | ARM | 8127 | 40080 |
| `daily_lag` | 60 | — | 60 | 5 | 63 | daily_lag | ARM | 8208 | 40020 |
| `long_window_crop60` | 1440 | 60 | 60 | 5 | 63 | base | EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor | 8127 | 38700 |
| `short_window_deep_core` | 60 | — | 60 | 10 | 2047 | base | ARM | 12047 | 40080 |
| `long_window_local_support_67` | 1440 | — | 63 | 5 | 63 | base | EXTRA_CONTEXT_67_SAMPLES (measured): NOT a null; the clamped core still reaches branch 5 + core 63 - 1 = 67 raw samples | 8127 | 38700 |
| `long_window_own_depth` | 1440 | — | 1440 | 10 | 2047 | base | ARM | 12047 | 38700 |

**Reading rules, verbatim from the sealed design:** *three seeds on one task are development evidence* · *a fit that reached the update ceiling is CENSORED wherever its best checkpoint fell* · *no cell is removed after its score is seen* · *a published number under another protocol never enters the comparison column*

**The block's own question, as sealed:** context beyond the hour, SEPARATED from receiver depth and from train volume, at a MATCHED update budget. Volume is held fixed BY CONSTRUCTION: train_population COMMON_INTERSECTION, so every arm trains on the SAME origins. The crossing is raw input window {60, 1440} x causal core depth {5 dilated blocks, 10 dilated blocks}: (60,5) modular_w60, (60,10) short_window_deep_core, (1440,5) long_window_local_support_67, (1440,10) long_window_own_depth, with long_window_crop60 as the exact-information null and daily_lag as the causal daily-lag channel. WHAT IT CANNOT SEPARATE, declared before any score: in this architecture family the receptive field is 1 + 2*sum(dilations), so a raw window longer than 67 samples is only USED when depth grows; the (1440,5) cell carries the long window's padding and 67 samples of reach, NOT 1440 samples of information, and context beyond 67 samples is confounded with depth BY CONSTRUCTION of the family. Only the depth-10 row can carry long context, so the context contrast at matched depth and matched volume is long_window_own_depth - short_window_deep_core, and the depth contrast at matched context is short_window_deep_core - modular_w60 and long_window_own_depth - long_window_local_support_67. SECOND DECLARED LIMIT: every cell is fitted to a ceiling of 600 observed updates, so every cell is CENSORED_BY_BUDGET wherever its best checkpoint fell; an arm whose validation MAE_z improved by more than 0.005 over its last 200 updates in 2 or more of its 3 seeds is UNDERTRAINED_AT_CEILING and its contrast is NOT read as a context or depth effect

**Why this block exists and where its ceiling comes from, verbatim from the sealed design:** SEALED BEFORE ANY SCORE OF ANY CELL OF THIS BLOCK. It extends Q2_CONTEXT_BOUNDED (design 47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea, twelve fits, four arms) with the two W1440 FULL-DEPTH arms that block declared it could not hold, and it refits ALL six arms under one budget so no arm is compared across budgets. The ceiling is 600 observed updates instead of 4 000, and that number comes from the RETAINED cost pilots' own measured rates on this same host (<worker-host>) and from nothing else: long_window_own_depth 4.165 CPU s per train update and 8 458 399 744 B peak RSS, long_window_local_support_67 4.4427 CPU s per update and 10 279 276 544 B peak RSS, modular_w60 0.0390, daily_lag 0.0392, long_window_crop60 0.0410 and short_window_deep_core 0.0710 CPU s per update with peak RSS under 1 GiB. At the 4 000-update ceiling the two deep arms would cost 17 507 s and 18 406 s of CPU PER CELL (about 30 CPU hours and roughly 19 h of wall for their six cells) and hold 7.9 and 9.6 GiB resident; at 600 updates they cost about 2 630 s and 2 760 s per cell. The reduction is a RESOURCE declaration made when no cell of this block had a score, never a removal or a re-budgeting after a score was seen (reading rule 3). Its cost is stated in the question field: every cell of every arm is CENSORED_BY_BUDGET and the block carries the UNDERTRAINED_AT_CEILING rule to say so per arm. No arm, seed, recipe field other than max_updates, scaler, row, cadence, monitor, patience or metric of Q2_CONTEXT_BOUNDED is otherwise changed

**The budget declaration, verbatim from the sealed recipe:** 600 observed updates, not the 4 000 of Q2_CONTEXT_BOUNDED, chosen BEFORE any cell of this block had a score from the retained cost pilots' own measured rates on this host: the two W1440 full-depth arms cost 4.165 and 4.4427 CPU s per train update, i.e. 17 507 s and 18 406 s of CPU per cell at a 4 000-update ceiling and roughly 19 h of wall for their six cells, against about 2 630 s and 2 760 s per cell at 600. Consequence, declared here and not discovered later: with validation every 200 updates there are 3 checkpoint opportunities and patience 3 can never expire, so EVERY cell of EVERY arm reaches the ceiling and is CENSORED_BY_BUDGET wherever its best checkpoint fell. Nothing in this block claims convergence, and the UNDERTRAINED_AT_CEILING rule in the block's question field refuses to read a contrast as an effect for an arm still improving at the ceiling.

### N2. The cost basis, and the projection that could not be produced

**There is no `REPORT.pilot.json` for this block, and the reason is part of the result.** `tools/df_e1_block_ungoverned.py pilot-report` builds its projection from EVERY pilot the design registers, and two of the six were never recorded: `pilot_long_window_own_depth` was terminated by systemd-oomd for user-session memory pressure after 1 292.106 CPU s, and `pilot_long_window_local_support_67` never launched. Under the stand-down neither was retried, so no projection over six arms exists and none is invented here. What follows is each pilot that DID land, read straight from its own cell record, beside the retained Q2_CONTEXT v1 measurements for the two arms that did not.

**Every peak figure in this section and in N3 is `MAIN_PROCESS_RSS_ONLY`** — `resource.getrusage(RUSAGE_SELF).ru_maxrss` as `df_e1_block.run_cell` records it. It is NOT a process tree or cgroup peak, and a placement decision needs the latter. See [`FOOTPRINT_BASIS.json`](FOOTPRINT_BASIS.json), which lists every figure with what measured it and the cgroup peaks systemd recorded for the same scopes.

| arm | pilot | CPU s per train update | peak RSS GiB (MAIN_PROCESS_RSS_ONLY) | projected CPU s per cell at this block's ceiling | source |
|---|---|---:|---:|---:|---|
| `modular_w60` | landed | 0.0414 | 0.87 | 24.8 | this block's own pilot |
| `daily_lag` | landed | 0.0407 | 0.87 | 24.4 | this block's own pilot |
| `long_window_crop60` | landed | 0.0424 | 0.87 | 25.4 | this block's own pilot |
| `short_window_deep_core` | landed | 0.0727 | 0.90 | 43.6 | this block's own pilot |
| `long_window_local_support_67` | **NOT RECORDED** | 4.4427 | 9.57 | 2665.6 | RETAINED Q2_CONTEXT v1 pilot, same host, 2026-09-21 |
| `long_window_own_depth` | **NOT RECORDED** | 4.1650 | 7.88 | 2499.0 | RETAINED Q2_CONTEXT v1 pilot, same host, 2026-09-21 |

* campaign ceiling 32400 CPU s, closure reserve 2000 CPU s, per-child CPU ceiling 5400 s, per-child wall ceiling 7200 s, parallel_children 1

### N3. Every fit, as it landed

Errors recomputed here from each cell's retained `arrays.npz`, each cross-checked against the value the cell record stored (a disagreement above 1e-12 refuses the whole table), and every arm verified to have scored the IDENTICAL origin array.

| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | stop | censoring | updates | best update | val MAE_z improvement over the last 200 updates | CPU s | peak RSS GiB (MAIN_PROCESS_RSS_ONLY) | reload max err | fresh-process replay |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `modular_w60_s1` | `modular_w60` | 1 | 0.559310 | 0.510380 | 0.617372 | 0.173303 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019321 | 27.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s1` | `daily_lag` | 1 | 0.550661 | 0.502487 | 0.617372 | 0.186087 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.016519 | 28.2 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s1` | `long_window_crop60` | 1 | 0.559310 | 0.510380 | 0.617372 | 0.173303 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019321 | 31.8 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s1` | `short_window_deep_core` | 1 | 0.570734 | 0.520804 | 0.617372 | 0.156417 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.005287 | 51.9 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s1` | `long_window_local_support_67` | 1 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s1` | `long_window_own_depth` | 1 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `modular_w60_s2` | `modular_w60` | 2 | 0.566499 | 0.516940 | 0.617372 | 0.162677 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.040595 | 27.9 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s2` | `daily_lag` | 2 | 0.560715 | 0.511662 | 0.617372 | 0.171227 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.023692 | 28.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s2` | `long_window_crop60` | 2 | 0.566499 | 0.516940 | 0.617372 | 0.162677 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.040595 | 32.4 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s2` | `short_window_deep_core` | 2 | 0.568177 | 0.518471 | 0.617372 | 0.160196 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 200 | +0.008548 | 57.5 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s2` | `long_window_local_support_67` | 2 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s2` | `long_window_own_depth` | 2 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `modular_w60_s3` | `modular_w60` | 3 | 0.565577 | 0.516098 | 0.617372 | 0.164040 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.021223 | 28.9 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s3` | `daily_lag` | 3 | 0.563220 | 0.513947 | 0.617372 | 0.167524 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.011196 | 30.2 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s3` | `long_window_crop60` | 3 | 0.565577 | 0.516098 | 0.617372 | 0.164040 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.021223 | 34.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s3` | `short_window_deep_core` | 3 | 0.556763 | 0.508056 | 0.617372 | 0.177067 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019160 | 58.2 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s3` | `long_window_local_support_67` | 3 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s3` | `long_window_own_depth` | 3 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |

Censoring across the 12 fits: `CENSORED_BY_BUDGET`.

**6 of the 18 registered cells were NEVER FITTED and are named here, not dropped: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3`.** The design still registers all 18; N13 carries every memory reading that refused to start them. Arms with all 3 seeds: `modular_w60`, `daily_lag`, `long_window_crop60`, `short_window_deep_core`. No contrast is taken across an incomplete arm, and every contrast the missing cells block is printed below as `NOT_MEASURED` with the cells that are missing.

Initial-weight digests, per seed: seed 1: `modular_w60` b9caefebdc3c…, `daily_lag` 0f6b46da2c89…, `long_window_crop60` b9caefebdc3c…, `short_window_deep_core` 9d7edff6f248… · seed 2: `modular_w60` 84f67bc65cb5…, `daily_lag` f9c474649f56…, `long_window_crop60` 84f67bc65cb5…, `short_window_deep_core` 1a6308265c88… · seed 3: `modular_w60` e0e2765cded8…, `daily_lag` 7fc47852dea4…, `long_window_crop60` e0e2765cded8…, `short_window_deep_core` 59e492f0e56f…

### N4. Per arm, and the paired difference against the baseline arm

| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | paired Δ MAE_z vs `modular_w60`, per seed | mean Δ | signs (+ / −) |
|---|---:|---:|---:|---:|---|---:|---|
| `modular_w60` | 0.563796 | 0.003912 | 0.514473 | 0.166673 | — (this is the baseline arm) | — | — |
| `daily_lag` | 0.558198 | 0.006647 | 0.509365 | 0.174946 | -0.008649 · -0.005785 · -0.002357 | -0.005597 | 0 / 3 |
| `long_window_crop60` | 0.563796 | 0.003912 | 0.514473 | 0.166673 | +0.000000 · +0.000000 · +0.000000 | 0.000000 | 0 / 0 |
| `short_window_deep_core` | 0.565225 | 0.007439 | 0.515777 | 0.164560 | +0.011424 · +0.001678 · -0.008814 | 0.001429 | 2 / 1 |

Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and **no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal effect: no causal claim here is verified against a retained-row error, because a causal claim does not predict a retained row.

### N5. The crossing: input context x core depth, with volume fixed by construction

Volume is not a factor here: every arm trained on the SAME 38700 origins and scored the SAME 10020, both checked as arrays above. The two factors that remain are the raw input window and the depth of the causal core.

| | depth 5 blocks | depth 10 blocks |
|---|---|---|
| **raw window 60** | `modular_w60`<br>mean MAE_z **0.563796**<br>8127 parameters, usable context 60 samples | `short_window_deep_core`<br>mean MAE_z **0.565225**<br>12047 parameters, usable context 60 samples |
| **raw window 1440** | `long_window_local_support_67`<br>**NOT FITTED** (no seed)<br>8127 parameters, usable context 63 samples | `long_window_own_depth`<br>**NOT FITTED** (no seed)<br>12047 parameters, usable context 1440 samples |

**The contrasts, exactly as the seal declared them before any fit.** Each is taken within a seed.

| declared contrast | what it isolates | per-seed Δ MAE_z | mean Δ | signs (+ / −) |
|---|---|---|---:|---|
| `causal_channel`<br>`daily_lag - modular_w60` | a causal daily-lag channel y(t+h-1440) added to the W60 receiver | -0.008649 · -0.005785 · -0.002357 | -0.005597 | 0 / 3 |
| `context_at_matched_depth_10`<br>`long_window_own_depth - short_window_deep_core` | input context 1440 vs 60 at depth 10, with capacity identical (12 047 parameters both), depth identical and volume identical: **the context contrast** | **NOT_MEASURED** — these cells were never fitted: `long_window_own_depth_s1`, `long_window_own_depth_s2`, `long_window_own_depth_s3` | — | — |
| `context_at_matched_depth_5`<br>`long_window_local_support_67 - modular_w60` | input context at depth 5, where the core reaches only 63 samples: a few extra samples plus the long window's padding, not 1440 samples of information | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_local_support_67_s2`, `long_window_local_support_67_s3` | — | — |
| `depth_at_matched_context_1440`<br>`long_window_own_depth - long_window_local_support_67` | core depth 10 vs 5 with a 1440-row raw window: depth AND the context depth unlocks, jointly, so it identifies neither alone | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3` | — | — |
| `depth_at_matched_context_60`<br>`short_window_deep_core - modular_w60` | core depth 10 vs 5 with the usable context pinned at 60 samples: **the depth contrast** | +0.011424 · +0.001678 · -0.008814 | +0.001429 | 2 / 1 |
| `exact_information_null`<br>`long_window_crop60 - modular_w60` | the W1440 input cropped to its last 60 rows: the same computation as the baseline (RP87), measured here rather than assumed | +0.000000 · +0.000000 · +0.000000 | +0.000000 | 0 / 0 |
| `interaction`<br>`(long_window_own_depth - short_window_deep_core) - (long_window_local_support_67 - modular_w60)` | does the context difference itself depend on depth | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3` | — | — |

Negative Δ = the first-named arm has the smaller error. **Three seeds is three seeds:** a sign count of 3 / 0 on n = 3 is a direction, not an effect, and no interval is claimed for any row above.

### N6. The UNDERTRAINED_AT_CEILING verdict, by the rule sealed before any fit

* statistic: improvement in validation MAE_z over the last 200 observed updates, i.e. val_mae(second-to-last event) - val_mae(last event), from each cell's own retained events
* threshold: **0.005** MAE_z
* verdict: an arm whose improvement exceeds the threshold in 2 or more of its 3 seeds is UNDERTRAINED_AT_CEILING, and no contrast involving it is read as a context or depth effect
* declared: before any cell of this block was fitted

| arm | improvement over the last 200 updates, per seed | seeds above the threshold | verdict |
|---|---|---:|---|
| `modular_w60` | +0.019321 · +0.040595 · +0.021223 | 3 | **UNDERTRAINED_AT_CEILING** |
| `daily_lag` | -0.016519 · +0.023692 · -0.011196 | 1 | **NOT_UNDERTRAINED_BY_THIS_RULE** |
| `long_window_crop60` | +0.019321 · +0.040595 · +0.021223 | 3 | **UNDERTRAINED_AT_CEILING** |
| `short_window_deep_core` | -0.005287 · +0.008548 · +0.019160 | 2 | **UNDERTRAINED_AT_CEILING** |
| `long_window_local_support_67` | — · — · — | — | **NOT_FITTED_NO_VERDICT** |
| `long_window_own_depth` | — · — · — | — | **NOT_FITTED_NO_VERDICT** |

**3 arm(s) are UNDERTRAINED_AT_CEILING: `modular_w60`, `long_window_crop60`, `short_window_deep_core`.** By the rule sealed before any fit, no contrast involving them is read as a context or depth effect. The numbers stay published exactly as they landed.

### N7. The three declared references, on the same rows

Computed by `df_e1_block.baselines` on the block's 10020 common evaluation origins. The block's closure suppresses these when it fails, so they are published separately.

| reference | definition | MAE kW | MAE_z | skill vs persistence |
|---|---|---:|---:|---:|
| `persistence` | y(t) | 0.617372 | 0.676560 | 0.000000 |
| `daily_seasonal` | y(t+h-1440) | 0.731659 | 0.801804 | -0.185119 |
| `train_constant` | mean of the train labels of the 28 d tier; computed on the common evaluation set at closure, no fit, no terminal | 0.709950 | 0.778013 | -0.149955 |

Published exactly as they landed: `daily_seasonal`, `train_constant` are **worse** than persistence on these rows.

### N8. The owner closure table as it landed

`owner_closure_table.v2` from `tools/df_closure_table.py`, generated 2026-09-26T19:32:27Z: **18 rows, 0 verified, 0 preserved with a qualified scope**; custody classes {"UNCHECKED": 18}; preparation classes {"PREPARATION_LOCAL_ONLY": 18}.

**This is the load-bearing fact about this run, and it is printed before any number of mine.** The verifier's policy is that a score with **no accepted terminal receipt is not reported as a model error at all** — not as a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, so every error column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed; the verifier's custody policy was not touched to make a number appear.

| unit | task / horizon / split | metric and scale | model error | naive error | skill | literature value + source | placed in the comparison column | comparability | custody | binding |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| `modular_w60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |

**Why NOT_COMPARABLE, verbatim:** unknown identity fields cannot match as proof: ['target_transform']

**Planned matched comparison, verbatim:** read the primary source (or its code) and fill the field; a placeholder is not a protocol

**Every problem the table recorded, in full:**

* Q2_CONTEXT_DEEP_20260926/daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt

### N9. The unanchored measurement table

`unanchored_measurement_table.v1`. It supplies the five columns the owner's rule names, for a run the owner closure table can only print as `null`. Each error is recomputed from the cell's own retained arrays and cross-checked against the record's stored score; the contract columns are taken from the landed owner table above. **It is not an `owner_closure_table.v2` row, it is never verified, and its custody is `UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.** Nothing may promote, select or rank on it.

| unit | metric and scale | model error | naive error, SAME rows | skill | rows (model / naive) | horizon (model / naive) | scale (model / naive) | literature value + source | comparability | custody |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `modular_w60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.510380 | 0.617372 | 0.173303 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502487 | 0.617372 | 0.186087 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.510380 | 0.617372 | 0.173303 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.520804 | 0.617372 | 0.156417 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516940 | 0.617372 | 0.162677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.511662 | 0.617372 | 0.171227 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516940 | 0.617372 | 0.162677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.518471 | 0.617372 | 0.160196 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516098 | 0.617372 | 0.164040 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.513947 | 0.617372 | 0.167524 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516098 | 0.617372 | 0.164040 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.508056 | 0.617372 | 0.177067 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |

**Fits that landed worse than their naive reference: 0 of 12.** Every fit's skill is printed above whatever its sign.

### N10. What the matched 600-update ceiling costs, measured on the arms both blocks share

The block `Q2_CONTEXT_BOUNDED` (design `47a270eec01f203c…`) fitted 4 of these arms on the SAME rows, the SAME scaler, the SAME seeds and the SAME cadence, but to a **4000-update** ceiling with patience 3 allowed to expire. Its numbers are a direct measurement of what this block's budget costs a cheap arm — and the only honest way to say how far an arm might still have had to travel.

| arm | mean MAE_z at 600 updates (this block) | mean MAE_z at 4000 updates (that block) | Δ (this − that) | best update there, per seed |
|---|---:|---:|---:|---|
| `modular_w60` | 0.563796 | 0.546949 | 0.016846 | 1200, 2400, 1200 |
| `daily_lag` | 0.558198 | 0.541800 | 0.016398 | 1000, 1400, 1200 |
| `long_window_crop60` | 0.563796 | 0.546949 | 0.016846 | 1200, 2400, 1200 |
| `short_window_deep_core` | 0.565225 | 0.550352 | 0.014873 | 800, 1200, 1200 |

This section compares two BLOCKS, not two arms: it measures the budget, and nothing in N5 is adjusted by it.

### N11. The closure as it landed

* `df_e1_block_report.v3`, design `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318`, block `Q2_CONTEXT_DEEP`
* **`verified`: False**
* common evaluation rows 10020; sigma_evaluation 0.9125164391265214; spent CPU 522.8 s
* closure code drift: `"none: closed under the sealed code"`
* scope: DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no test rows read
* disposition: `{"disposition": "HISTORICAL_DEV_ONLY", "task_id": "uci_235.W60_h60.DEV_28d_7d", "policy": "docs/tres_temas_entrevista/program_v3/SOTA_FIRST_2026_09_21.md", "why": "previous exploratory pilot (household task / adapted models): preserved with its receipts and failures, excluded from active selection, ranking and recommendations"}`; active_selection: `null`
* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no paired contrast and no selected arm. The per-arm means in N4, the crossing in N5 and the references in N7 are therefore published OUTSIDE the closure, recomputed from the arrays.

**Every problem the closure recorded, in full:**

* daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s3: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt
* closure without a warehouse read: no accepted custody, nothing is verified

**Fresh-process replays: 12 of 12 pass `allclose(1e-6, 1e-6)`; the maximum absolute difference over every replayed cell is 0.000e+00 kW.**

| cell | fresh-process replay | max abs difference (kW) |
|---|---|---:|
| `modular_w60_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `modular_w60_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `modular_w60_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s3` | allclose(1e-6) PASS | 0.000e+00 |

### N12. Governance, stated as it is

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

### N13. Placement: every memory reading taken before a launch

* `MEMORY_GATE.jsonl`: 81 readings — 13 launches, 55 holds, 13 finished jobs, 0 units NOT started because memory never allowed it
* MemAvailable at launch: min 11.24 GiB, max 13.06 GiB
* MemAvailable while held: min 5.89 GiB, max 13.14 GiB; longest single wait 1440 s

| label | verdict | MemAvailable GiB | measured pilot peak GiB | required (peak + margin) GiB | cap |
|---|---|---:|---:|---:|---|
| `pilot:pilot_long_window_own_depth` | HELD_WAITING_FOR_MEMORY | 10.20 | 7.88 | 8.88 | 9G |
| `pilot:pilot_long_window_own_depth` | LAUNCH | 13.06 | 7.88 | 8.88 | 9G |
| `pilot:pilot_long_window_own_depth` | FINISHED exit -9 | 5.50 (after) | — | — | — |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 5.89 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 13.13 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 13.14 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.38 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.59 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.58 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.58 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.56 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.53 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.52 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.45 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.38 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.35 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.40 | 9.57 | 10.57 | 11G |
| `modular_w60_s1#attempt1` | LAUNCH | 12.16 | 0.94 | 1.94 | 4G |
| `modular_w60_s1#attempt1` | FINISHED exit 0 | 12.11 (after) | — | — | — |
| `daily_lag_s1#attempt1` | LAUNCH | 12.12 | 0.94 | 1.94 | 4G |
| `daily_lag_s1#attempt1` | FINISHED exit 0 | 11.93 (after) | — | — | — |
| `long_window_crop60_s1#attempt1` | LAUNCH | 11.93 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s1#attempt1` | FINISHED exit 0 | 11.99 (after) | — | — | — |
| `short_window_deep_core_s1#attempt1` | LAUNCH | 11.99 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s1#attempt1` | FINISHED exit 0 | 11.99 (after) | — | — | — |
| `modular_w60_s2#attempt1` | LAUNCH | 11.99 | 0.94 | 1.94 | 4G |
| `modular_w60_s2#attempt1` | FINISHED exit 0 | 12.01 (after) | — | — | — |
| `daily_lag_s2#attempt1` | LAUNCH | 12.02 | 0.94 | 1.94 | 4G |
| `daily_lag_s2#attempt1` | FINISHED exit 0 | 11.82 (after) | — | — | — |
| `long_window_crop60_s2#attempt1` | LAUNCH | 11.83 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s2#attempt1` | FINISHED exit 0 | 11.82 (after) | — | — | — |
| `short_window_deep_core_s2#attempt1` | LAUNCH | 11.82 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s2#attempt1` | FINISHED exit 0 | 11.71 (after) | — | — | — |
| `modular_w60_s3#attempt1` | LAUNCH | 11.72 | 0.94 | 1.94 | 4G |
| `modular_w60_s3#attempt1` | FINISHED exit 0 | 11.24 (after) | — | — | — |
| `daily_lag_s3#attempt1` | LAUNCH | 11.24 | 0.94 | 1.94 | 4G |
| `daily_lag_s3#attempt1` | FINISHED exit 0 | 11.37 (after) | — | — | — |
| `long_window_crop60_s3#attempt1` | LAUNCH | 11.37 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s3#attempt1` | FINISHED exit 0 | 11.52 (after) | — | — | — |
| `short_window_deep_core_s3#attempt1` | LAUNCH | 11.52 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s3#attempt1` | FINISHED exit 0 | 12.13 (after) | — | — | — |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.13 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.28 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.32 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt2` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.14 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.61 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.62 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.64 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.59 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.60 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.31 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.39 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.71 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.37 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.39 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.37 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.45 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.36 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.20 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.92 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.23 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.80 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.82 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.83 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.76 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.86 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.81 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.80 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.88 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.18 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.16 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.06 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.17 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.17 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.27 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.90 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.16 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt2` | HELD_WAITING_FOR_MEMORY | 12.21 | 7.88 | 8.88 | 10G |

