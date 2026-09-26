# Disposition of RP57–RP64 — in place of the review that was never written

**Authority.** The owner's grant of 2026-09-26. The reviewer is not returning; the successor technical
lead was given full authority to decide these two ranges. This document stands in place of the absent
[`MUSASHI_RP57_RP64_REVIEW`], which does not exist in this repository and is not written here.
**Nothing below is signed, quoted or attributed to Musashi**, and no reviewer's name appears on it. It
is Satoshi's audit, published under Satoshi's name, dated today. The ruling on
`MOD-FROZEN-PREFIX`, `MOD-CORE-PRETRAIN`, `MOD-CONF` and `MOD-E3` is in
[`SATOSHI_RP49_RP56_DISPOSITION_2026_09_26`](SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md) §9, which
this document supplies the RP57–RP64 evidence for; §6 below repeats the four rulings with the
measurements from this range that carry them.

**What is audited.** [`SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21`](SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md)
**and its own corrections published beside it**,
[`SATOSHI_RP57_RP64_CORRECTIONS_2026_09_21`](SATOSHI_RP57_RP64_CORRECTIONS_2026_09_21.md), against the
order [`MUSASHI_PROGRAM_RP57_RP64`](../../handoffs/MUSASHI_PROGRAM_RP57_RP64_2026_09_20.md). The
corrections are audited as part of the round, not as a later excuse: two of the nine refutations below
were already anticipated there and are recorded as such.

Worktree `predictor-rpaudit-20260926`, branch `satoshi/rp49-rp64-disposition-20260926`, base
`941eb5b3`. CPU only, `CUDA_VISIBLE_DEVICES=''`, `crispdm-run -m 6G -t 900 -n rpaudit`, env
`trading-stack` (Python 3.12.13). **No training, no allocation, no GPU, no governance or warehouse
contact, no reserved split reopened, no committed sample overwritten.** Saved weights were loaded and
evaluated — inference — where that was the only way to check a claim.

**The governing rule.** A module is unblocked only if its own retained artifacts sustain it. A claim
whose artifact is gone is `UNVERIFIABLE_ARTIFACT_ABSENT`, never "accepted".

---

## 1. Method, and the total

Identity recomputed from bytes; numbers recomputed from arrays and then from the saved weights;
refusals exercised on malformed cases built here; counterexamples hunted actively.
[`tools/df_rp49_rp64_audit.py`](../../../tools/df_rp49_rp64_audit.py) ·
[`tests/test_rp49_rp64_audit.py`](../../../tests/test_rp49_rp64_audit.py) ·
[`RP49_RP64_AUDIT_20260926/AUDIT.json`](../evidence/RP49_RP64_AUDIT_20260926/AUDIT.json).

**60 VERIFIED, 9 REFUTED, 0 UNVERIFIABLE_ARTIFACT_ABSENT** over both ranges; **52 rules pass**. Six
of the nine refutations belong to this range and they are §5.1–§5.6. Every artifact both returns cite
is still on disk, which is why nothing had to be declared absent — and that is itself a finding in the
range's favour.

## 2. What re-derives exactly

### 2.1 Phase 1 (RP62/RP63) — the round's headline

| | recomputed | published |
|---|---|---|
| sealed design | `5cb8263d359f1500…`, re-derives; the published sealed document **is** the run's `DESIGN.json` | `5cb8263d…` |
| the phase ran on the successor run's own rows | `data_sha256` `70485ac9…`, the same prepared bytes | "on the successor run's own rows" |
| `core_mae` mean / sd (n=3) | 0.497770 / 0.000171 | 0.4978 / 0.0002 |
| `tcn_mse` mean / sd | 0.535948 / 0.005012 | 0.5359 / 0.0050 |
| `core_mse` mean / sd | 0.546930 / 0.007804 | 0.5469 / 0.0078 |
| parameter match | 8 061 (reference block) against 8 127 (ours), every seed | 8 061 against 8 127 |
| CPU | 2 024.488 s, exactly the sum of its own nine cells' process time | 2 024.5 s |

### 2.2 The comparison table (RP57), recomputed row by row from the arrays

MAE, RMSE and bias for all twelve methods of the `raw_kW` table reproduce to the printed precision;
the persistence-scaled column and **MASE m=1** reproduce; all twelve paired differences reproduce in
mean, sd **and** the count of rows where the first method is better (`R1_s1−R0_s1` +0.007482, sd
0.2404, 4 674 of 10 020 — exactly); the contiguous-block table reproduces. The per-seed honesty holds:
`R0_s3` beats the ridge by 1.37%, `R2_s1` by 0.59%, `R1_s2` by 0.14%, and the linear control wins on
the regime means, not on every cell.

**The three denominators, each on the support its own formula states** — and two of them had to be
*found*, because the return says "the same slice" without naming it:

| | support recomputed by the audit | value | pairs | published |
|---|---|---|---|---|
| persistence-scaled (horizon, train) | the 40 080 train origins, lag = 60 | 0.6162615768 | 40 080 | 0.61626 |
| conventional MASE m=1 | rows 0 → the last train **label** row (40 259), lag 1 | **0.0851237797** | **40 257** | 0.0851237797 over 40 257 |
| conventional MASE m=1440 | the same slice, the whole pair inside it | 0.6611885208 | 38 818 | reproduces R0_s1's 0.8356 |

The m=1 number matches to ten digits on a support the document never names; a claim that pins its own
support only through the digits it prints is one edit away from unverifiable, and that is worth saying
even when it verifies.

### 2.3 The publication (RP57)

**192 metric rows**, recounted under the publisher's own rule — 3 scales × (12 MAE + 36 skill) + 36
scaled errors + 12 paired differences = 192 — against `metrics_published: 192`. The receipt names a
report path in a session scratchpad that is long gone, and its `report_sha256`
`848c6ccb4c2b6ea7…` **is the digest of the retained `RP57/COMPARISON_REPORT.json`**: the published
rows still bind to bytes this repository holds. Against the run's own `RESULTS.json`: identical for
all nine cells, difference exactly 0.0.

### 2.4 The legacy inventory (RP58)

| | recomputed | published |
|---|---|---|
| tables inventoried | 167 (113 `RECONSTRUCTED_FROM_PREDICTIONS_AND_LABELS`, 54 `PUBLISHED_SUMMARY_ONLY`) | 167 (113, 54) |
| dispositions | `CAUSALITY_UNVERIFIED` 114 + `NO_DECOMPOSITION…` 53 = 167 | 114 / 53 |
| tables with a published naive | exactly **1**, and no other entry carries a non-null `naive_mae` | "exactly one" |
| flagged ≥50% over the naive | **85**, and the audit's own ≥50% rule selects the identical 85 files | 85 |

Four flagged tables were recomputed **from the committed prediction CSVs**, not from the inventory:
`phase_3_2_cnn`, `phase_3_2_lstm`, `phase_3_1_cnn` and `phase_1_ann_12600`. Every MAE, every naive and
every skill reproduces to twelve digits, on **5 984 identical rows** per horizon, with the naive taken
from the same rows and the same base column (`test_CLOSE`) as the model. The "naive on other rows"
counterexample was hunted here hardest and **was not found**; the refusal is built as a rule (drop one
row and the recomputation turns `REFUTED` instead of quietly agreeing on a different population).

### 2.5 The data path (RP59) and the optimisation probe (RP60)

A perfect one-minute grid over the consumed slice: 50 400 rows present, 50 400 expected, **zero**
missing minutes, **zero** duplicated stamps, no other step seen. The label is `Y[origin+60]` and
equals the panel column element by element — reproduced independently in §3.2, where all nine
phase-1 cells' stored labels are bitwise the panel targets. The scaled slice is centred on the
**window** grain as declared, and the two grains differ by 0.001670 in scaled units. The error
concentration reproduces: the 25% smallest movements carry 13.43% / 14.19% of the absolute error, the
10% largest carry 27.02% / 27.37%, and in that top decile the neural model and the linear control have
the **same** mean absolute error, 1.4930 kW, to four decimals.

RP60: the persistence route reproduces through the same inverse and metric to **2.16·10⁻⁷**; capacity
falls 6.9938 → 0.005537 on the fixed 256-window subset; `counted_updates == optimizer.iterations ==
400`; the full-scale scrambled-label fit is 0.60154 against 0.55250 true, gap 0.04904.

## 3. What the audit had to add, because the round did not do it

### 3.1 Phase 1 was never closed

**Finding (verification absent, not wrong).** The return says of phase 1: *"nine cells, every unit
registered, delivered, reported and reconciled."* All true. It is not a closure. The phase-1 run root
holds `DESIGN.json`, `DATA.npz`, `DELIVERIES.json`, `TERMINAL_RECEIPTS.json`, `REPORT.json` and
`attempts/` — and **no `CLOSE.json` and no `closure_replays/`**. The successor run's own verification
standard (*"a fresh process reloads the saved weights and regathers the windows from DATA; a gradient
summary is never accepted in place of this"*) was never applied to the phase that carries this round's
headline result.

### 3.2 So the audit performed it

The weights are retained, so the replay was performed here: **nine phase-1 cells, all 10 020
evaluation origins, a fresh process per cell**.

| | |
|---|---|
| cells replayed | 9 / 9 |
| max &#124;replay − stored&#124; | **8.34·10⁻⁷ kW** (the three `tcn_mse` arms reproduce **bitwise**), against the tolerance of 1·10⁻⁵ |
| published MAE | reproduces every cell |
| labels | equal the panel targets `Y[ev+60]` bitwise, every cell |
| parameters | 8 127 / 8 061, the recorded ones |

The same was done for the successor run's nine cells (RP49–RP56 disposition §4.1). **Eighteen cells,
every evaluation row, from the weights: the headline numbers of both ranges are the product of the
retained graphs and not of any later arithmetic.**

## 4. What this range establishes, plainly

1. **Training on the measure the question is judged by is worth 0.049 kW on this task** — as a
   *recipe*, loss and monitor together, which the round's own corrections document already says and
   which §5.1 sharpens further.
2. **The reference TCN block is worth 0.011 kW at matched parameters and identical training**
   (0.5359 against 0.5469) — our core is an adaptation and it costs something. Verified, and the
   parameter match was fixed by a rule stated before any fit.
3. **The published legacy corpus is unusable as a benchmark and the round proves it rather than
   asserting it.** 85 tables with extraordinary skill, 114 `CAUSALITY_UNVERIFIED`, both lineages
   `UNBOUND` because the producer of the decomposed inputs is not in this repository, and the
   mechanism family *identified but not identified exactly*: a centred smoother spanning the horizon
   reaches +46.76% where a trailing (causal) wavelet reaches −5.42%, and 96% is beyond even the
   centred filter. The round declares none of them `INVALID_CAUSAL_LEAK` on a score, which is the
   correct refusal.
4. **The causal instrument is real and still fires.** All **ten** negative-control classes of
   `tools/df_causal_battery.py` were re-run today and every one is `DETECTED` — including
   `full_series_dwt_as_time_row` and `centered_rolling_window`, the two classes RP58 names, plus
   `filtfilt`, `phase_compensation_shift_back`, `same_convolution_or_right_padding`,
   `shift_minus_k`, `full_series_fft_reconstruction_feature`, and the three state/ordering classes
   ([`CAUSAL_BATTERY_RERUN_NEGATIVE_CONTROLS.json`](../evidence/RP49_RP64_AUDIT_20260926/CAUSAL_BATTERY_RERUN_NEGATIVE_CONTROLS.json)).
   `failures: []`, 16 `SNAPSHOT_REFUSAL` and 28 `FIT_MODE` rules pass.
5. **The normalization space is now a declaration on an active path.** Built here and refused by
   name: `strict_normalization_contract: true` with no `prediction_space` raises *"this configuration
   does not declare prediction_space, and strict_normalization_contract forbids deciding it from the
   values"*; an invalid space raises; an undeclared space without strict mode still runs the old
   heuristic and records the guess as a guess. Eleven rules pass.
6. **The reserve was not touched.** `exposure: NO_TEST_ACCESS` is a closure refusal and the mutation
   that lets a pilot score the test split dies (`M10`, re-killed today in a 15/15 battery).

## 5. Counterexamples: what was hunted, and the six found in this range

| hunted | found? |
|---|---|
| a count off by one | **Yes** — §5.1 |
| a contrast confounded with its own budget | **Yes** — §5.1 |
| a number that does not re-derive under its stated support | **No**, once the estimator is read — §5.2, and the reading it supports does invert |
| a metric on rows other than the ones it claims | **Yes, inside one table** — §5.3 |
| a family quoted at the top of its own range | **Yes** — §5.4 |
| a control compared against an unmatched arm | **Yes** — §5.5 |
| an "exactly" that is not exact | **Yes** — §5.6 |
| a naive on rows the model never saw | **No** — §2.4, four tables recomputed from the CSVs on identical rows |
| a digest matching only because it was re-derived from the same corrupted source | **No** — the sealed design is byte-identical to the executed one and the metrics were re-derived from the weights |

### 5.1 The censored count is five, not four — and the winning arm got the bigger budget

The return says **"Four of nine cells are `CENSORED_BY_BUDGET`."** The retained
`RP63/PHASE1_REPORT.json` says **five**: `core_mae_s1`, `core_mae_s2`, `core_mse_s2`, `tcn_mse_s2`,
`tcn_mse_s3`. *Four* is the successor run's count (which is correct there — `R0_s2`, `R1_s2`, `R1_s3`,
`R2_s2`) and it was carried into the phase's paragraph. **This is the off-by-one.**

It matters more than a count, because of what the censoring is not symmetric about:

| arm | updates per seed | total | censored |
|---|---|---:|---:|
| `core_mae` | 4 000, 4 000, 3 762 | **11 762** | 2 / 3 |
| `tcn_mse` | 3 762, 4 000, 4 000 | 11 762 | 2 / 3 |
| `core_mse` | 3 762, 4 000, 2 508 | **10 270** | 1 / 3 |

The monitor that changed together with the loss also changed *when early stopping fired*, so the arm
that wins by 0.049 kW ran **14.5% more optimiser updates** than the arm it beats. The round's own
corrections document already established that RP63 moved two things at once (loss and monitor) and
that the monitor-fixed factorial reproduces `core_mae`'s 0.4978 — so the *conclusion* is answered
elsewhere. The **budget** leg of the confound is named here and nowhere else, and it is the leg that
matters for any module that plans to resolve a 0.01 kW effect.

### 5.2 The autocorrelation table re-derives exactly — under an estimator whose bias grows with the lag

The return reads: *"Train autocorrelation is 0.963 at one minute, 0.403 at sixty and 0.317 at a day:
the daily lag carries **less** linear structure than the hour, so a daily context is a hypothesis, not
a certainty."*

All four published values re-derive **exactly** (to 1·10⁻¹⁰) on the stated support — 40 199 consumed
train rows — under `tools/df_e1_data_audit.py::autocorrelation`, which is the textbook **biased** ACF:
`Σ(x_t−x̄)(x_{t+k}−x̄) / Σ(x_t−x̄)²`, a fixed whole-series denominator. That estimator shrinks by
`(n−k)/n`:

| lag (minutes) | published (biased ACF) | shrinkage (n−k)/n | lag-truncated Pearson |
|---:|---:|---:|---:|
| 1 | 0.963345 | 1.0000 | 0.963361 |
| 60 | 0.403002 | 0.9985 | 0.403283 |
| 1 440 (a day) | **0.317186** | 0.9642 | **0.335157** |
| 10 080 (a week) | **0.265593** | 0.7492 | **0.366864** |

**The sentence survives**: 0.335 < 0.403, so the daily lag does carry less linear structure than the
hour, and a daily context remains a hypothesis. **The table's apparent decay does not.** Corrected for
its own estimator's lag-dependent bias, the weekly lag is **0.367 — above the daily 0.335**, the
reverse of the order the published list shows, and understated by 27% as printed. Nothing in the
return draws a weekly conclusion, so no claim falls. But this is precisely the table a module that
must "fix all preprocessing/grouping/D/I/A/fusion states" would read to choose contexts, and as
published it would send that module the wrong way about the weekly grain. Restating it with its
estimator named, or recomputing it lag-truncated, costs nothing and no run.

### 5.3 The leak signature subtracts two skills measured on different rows

`RP58/LEAK_SIGNATURE_CALIBRATION.json` reports five representations in one column, and the
`signature` field computes `leaking_minus_causal_skill_points = 5.1409 − (−5.4182) = 10.559`. But
`TRAILING_WAVELET` is evaluated on **1 512** rows against a naive of 0.0018573, while the other four
are evaluated on **1 525** rows against 0.0018591: the trailing transform consumes a lead-in, so the
contrast that number computes **crosses two populations**. Every individual skill re-derives from its
own MAE and its own naive; it is the subtraction that does not share rows. The gap is 0.85% of the
rows and the conclusion (a causal trailing wavelet does not produce the published magnitude) survives
comfortably — but RP57, in this same round, makes "on the **same rows**" the standard the whole report
is built on, and one table of the round does not meet it.

### 5.4 "The phase-3 family by 96–97%" is the top of a range that starts at 27.7%

Recomputed from the artifact's own by-horizon skills: **24** phase-3-named tables are measurable, and
their best-horizon skill spans **27.72% to 97.23%**. Only **four** reach ≥96% — three in `phase_3_3`
at one hour and one in `phase_3_1` — and of those only one is at 96–97% across H1–H6 (`phase_3_2_lstm`,
94.7–96.6); `phase_3_2_cnn` is at 85.4% at H1. The family is not at 96–97%; its top four are.

Second, the base. "**85 of 167**" is true only if the other 82 were tested and passed. **54 of the
167 carry no reconstructable predictions and were never tested for the flag at all** (their indicator
reads `measured: false`, `why: "no reconstructable predictions beside this table"`). The rate among
the tables that could be measured is **85 of 113 — 75%**, which is a worse finding than the one the
sentence makes, not a better one. Neither correction weakens RP58's disposition: none of the 167 is
usable as a benchmark, and all 167 keep the disposition they were given.

### 5.5 The full-scale negative control has no matched true-label arm

`0.6015 scrambled against 0.5525 true` is measured and re-derives. The two arms differ in more than
their labels: the scrambled arm ran to the ceiling (**4 000** updates, **epoch 6** restored) while its
comparison partner is the finished run's `R0_s1`, which early-stopped at **3 762** updates and restored
**epoch 3**. No true-label arm was run *by the probe, at the probe's budget*. The round's corrections
document already withdraws "labels buy little" as a general claim and reduces the control to "what
that protocol recovers"; the budget and selection asymmetry between the two arms is named here.

### 5.6 "Exactly" is 5·10⁻⁷, and per row it is above this repository's own tolerance

The return says `core_mse_s1` = 0.5525, *"reproducing `R0_s1` of the finished run **exactly** through
a different runner."* Recomputed: `core_mse_s1` 0.5524988659, `R0_s1` 0.5524983663 — a mean agreement
of **5.0·10⁻⁷ kW**, and the two runners' per-row predictions differ by up to **6.5·10⁻⁵ kW**, which is
**above** the 1·10⁻⁵ replay tolerance this repository refuses on. The replication is real, it is a
genuine cross-runner check, and it is not an identity. At the four decimals published, both are
0.5525.

## 6. Ruling on RP57–RP64, and on the four modules

**ACCEPTED, with six findings of record; one claim corrected (the censored count), two claims
narrowed (the phase-3 family, the "85 of 167" base), one word withdrawn ("exactly"), and one
instrument flagged for restatement (the lag table). No result of the round is withdrawn.** The
round's three headline results all stand on recomputation: the recipe effect (0.049 kW, now known to
carry a budget leg), the reference-block cost (0.011 kW at matched parameters), and the unusability of
the legacy corpus. The round's own corrections document did two-thirds of the work this audit would
otherwise have had to do, and that is worth recording: a round that publishes corrections beside its
return rather than over it is a round an auditor can actually check.

The four modules, ruled in full in the companion disposition's §9, with this range's evidence:

| module | ruling | what this range contributes |
|---|---|---|
| **MOD-FROZEN-PREFIX** | **UNBLOCKED** | the prefix it must generalise is verified here: the grid, the label equality (reproduced bitwise on 18 cells), the scaler grain, the same-rows discipline of the comparison table. It carries §5.2 as its first repair — the lag table it would read to choose contexts must be restated with its estimator — and per-split materialization, which neither range does |
| **MOD-CORE-PRETRAIN** | **BLOCKED** | the instrument's resolution is the block. A scrambled-label fit loses **0.049 kW**, five times the ~0.01 effect this module must resolve (§2.5, §5.5), and the one recipe contrast run was not budget-matched (§5.1). **Must exist first:** a measured resolution for this protocol on this task (no new training needed), budget-matched arms with the monitor fixed, and MOD-FROZEN-PREFIX |
| **MOD-CONF** | **BLOCKED** | the reserve is intact and provably so (§4.6), and the method was still moving right through this range — objective, monitor, loss/optimiser, a renamed field with an erratum, a normalization space turned into a declaration. **Must exist first:** the Huber/AdamW v2 design retained as bytes (four standing arms currently rest on the digest `be2e776e…` with no document), and one sealed design freezing the confirmatory method. Its co-owner is Musashi and no grant to me replaces him |
| **MOD-E3** | **BLOCKED** | this range's price-series evidence is the 85 extraordinary legacy tables, **all** `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND` (§2.4, §5.4), and the mechanism identified only to a family (§4.3). **Must exist first:** `BUSINESS-CONTRACT` executed rather than designed; one governed forecasting result on the domain this module trades, since every retained E1 measurement is a household-electricity panel; and the H-CORE design MOD-CORE-PRETRAIN owes it |

## 7. What this disposition could not reach, and will not pretend to

- **Both legacy lineages stay `UNBOUND`.** The producer of the decomposed inputs is not in this
  repository. No work here reaches it, and the audit did not attempt a substitute: reconstructing a
  phase-3 run's consumed representation from a producer that is absent is not a measurement, it is a
  guess with a number attached. The disposition `CAUSALITY_UNVERIFIED` stands on all 114.
- **The mechanism behind 96–97% is not identified.** The family is (centred, symmetric filtering
  spanning the horizon, +46.76%); the exact mechanism is not, and 96% is beyond even that.
- **`OWNER_REPORTED` stays `OWNER_REPORTED`.** The owner's recalled phase-3 improvement is bound to
  nothing in this repository, and this audit binds it to nothing either.
- **The Huber/AdamW factorial was not re-derived**, because its design document is not in this
  repository (partial seal §5.1). Where this disposition leans on that factorial — §5.1, to say the
  monitor leg is answered — it leans on a result whose design is a digest without bytes, and it says
  so here rather than in a footnote. **Retaining that document is the cheapest repair available to
  this programme.**
- **Nothing was recomputed under today's code for the four blocks closed under drifted code.** Not
  this range's work, not claimed.

## 8. Costs, and what was refused

CPU only, no GPU — `cuInit` fails in the log and that is the proof. Every job under
`crispdm-run -m 6G -t 900 -n rpaudit`, env `trading-stack`. The heavy work is two full-evaluation-set
replay sets (18 cells × 10 020 origins, inference), a 15-mutant battery and one causal-battery
mechanics sample (12.5 s). The 52 rules run in 0.7 s. No training, no allocation, no new campaign, no
warehouse or governance contact, no host touched, no sample overwritten, no reserved split reopened.
The audit's own rules run beside the batteries they check: `tests/test_rp49_rp64_audit.py` with
`test_df_e1_seal`, `test_df_e1_close`, `test_df_e1_chronology`, `test_df_e1_first_child`,
`test_df_e1_receipt_concurrency`, `test_stl_norm_contract`, `test_df_e1_governed_route`,
`test_df_e1_governing_report`, `test_df_e1_pilot`, `test_df_e1_regimes` and `test_df_e1_loader` —
**202 passed** on a clean checkout. One defect of the audit's own making was found and repaired
on the way: loading a tool into `sys.modules` let two governed-route rules pick up this audit's
copy of a module and fail while passing in isolation. Every tool is now loaded under a private
name and registered nowhere. A battery that only passes when it runs alone is not a battery.


Five refusals:

1. **I refused to accept a headline result that had never been replayed.** Phase 1 had no closure; the
   replay was performed here, on every row of every cell, and only then was the result accepted.
2. **I refused to treat the round's corrections document as an excuse rather than as evidence.** It is
   audited as part of the round, and it is credited where it anticipated a finding.
3. **I refused to let "85 of 167" and "the phase-3 family by 96–97%" stand**, even though both make
   the round's own case look *stronger* than the artifacts do in one direction and weaker in the
   other. An auditor who only corrects in the flattering direction is not auditing.
4. **I refused to reconstruct a phase-3 lineage from a producer that is not here**, which would have
   turned an honest `UNBOUND` into a number nobody could check.
5. **I refused to write anything in the reviewer's name.** The two absent reviews stay absent. These
   two dispositions stand in their place by the owner's grant of 2026-09-26, under Satoshi's name, and
   each says so in its first line.

