# Disposition of RP49–RP56 — in place of the review that was never written

**Authority.** The owner's grant of 2026-09-26: the reviewer is not returning, and the successor
technical lead was given full authority to decide these two ranges. This document stands in place of
the absent [`MUSASHI_RP49_RP56_REVIEW`], which does not exist in this repository and is not written
here. **Nothing below is signed, quoted or attributed to Musashi.** It is Satoshi's audit and it is
published under Satoshi's name, dated today. The four modules whose only remaining blocker was
`MOD_E1_EXTERNAL_REVIEW` — [`SATOSHI_E1_PARTIAL_SEAL_RETURN_2026_09_26`](SATOSHI_E1_PARTIAL_SEAL_RETURN_2026_09_26.md) §4,
§9.1 — are ruled on in §6, one at a time, with their evidence.

**What is audited.** [`SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20`](SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md),
against the order [`MUSASHI_PROGRAM_RP49_RP56`](../../handoffs/MUSASHI_PROGRAM_RP49_RP56_2026_09_20.md).
Worktree `predictor-rpaudit-20260926`, branch `satoshi/rp49-rp64-disposition-20260926`, base
`941eb5b3`. CPU only, `CUDA_VISIBLE_DEVICES=''`, every job under
`crispdm-run -m 6G -t 900 -n rpaudit`, anaconda env `trading-stack` (Python 3.12.13). **No model was
fitted, no training was run, no allocation was taken, no host was touched, no warehouse or
governance service was contacted, and nothing under `docs/audits/evidence/d3_k5_20260917/` was
modified.** Model weights were *loaded and evaluated* — inference, not training — where a claim could
only be checked that way, and §4 says exactly where.

**The governing rule.** A module is unblocked only if its own retained artifacts sustain it. A claim
whose artifact is gone is `UNVERIFIABLE_ARTIFACT_ABSENT`, never "accepted". Approving for
convenience would be worse than the absence being repaired.

---

## 1. Method

Four things, in this order, and nothing was read as true because the return says it:

| | |
|---|---|
| **identity recomputed** | every design digest, data digest, array digest and population re-derived from the bytes by the repository's own canonical rule (`sha256` of the body without `design_sha256`, `sort_keys`, `separators=(",",":")` — `df_mod_e0.sha_obj`), never copied from the document that asserts it |
| **numbers recomputed** | every metric of the table re-derived from the stored arrays, and then from the **saved weights in a fresh process**, on the full evaluation set |
| **refusals exercised** | for each contract the range touches, a malformed case was built here and watched being refused **by name**; the round's own mutation battery was re-run from scratch |
| **counterexamples hunted** | a population off by one, a metric on rows other than the ones it claims, a digest that matches only because it was re-derived from the same corrupted source, a naive computed on rows the model never saw, a figure quoted against a ceiling it is not charged to |

The audit is a tool and a battery of rules, not a reading:
[`tools/df_rp49_rp64_audit.py`](../../../tools/df_rp49_rp64_audit.py) ·
[`tests/test_rp49_rp64_audit.py`](../../../tests/test_rp49_rp64_audit.py) ·
[`RP49_RP64_AUDIT_20260926/AUDIT.json`](../evidence/RP49_RP64_AUDIT_20260926/AUDIT.json).
Result over both ranges: **60 checks VERIFIED, 9 REFUTED, 0 UNVERIFIABLE** — every artifact the two
returns cite is still on disk, so nothing had to be declared absent. **52 rules pass.**

## 2. Identity, recomputed

| | recomputed | declared | |
|---|---|---|---|
| sealed successor design `RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json` | `143abb57d97daa07…` | `143abb57d97daa07…` | re-derives |
| the design the run executed (`DESIGN.json` in the run root) | file `a612eda185c0b5a1…` | file `a612eda185c0b5a1…` | **byte-identical to the sealed one** |
| prepared data `DATA.npz` | `70485ac9d1a41ac8…` | `70485ac9d1a41ac8…` | the closure's `data_sha256` IS the digest of the bytes |
| the nine cells' `arrays.npz` | each | each | all nine re-derive |
| retained `RESULTS`/`CLOSE`/`REPORT` copies under `RP55/` | — | — | byte-identical to the run root's |

"the successor re-seals to the identical digest" is not a re-seal that happened to land on the same
number: the sealed-before-execution document and the executed run's design are **the same bytes**.
The attack that matters — edit the design *and* repair its declared digest so it re-derives — is a
rule here: it re-derives and it is still not `143abb57…`, so the closure and the warehouse check both
refuse it.

## 3. The population: the sixteenth unit is `prepare`, and the return never names it

The return's table sets **`units declared / closed 16 / 16`** beside **`cells VERIFIED_AND_GOVERNED
15 / 15`** and **`warehouse by content 16 / 16`**. Recomputed:

- the closure's sealed population is **15** units, all 15 present, `absent_ids` empty,
  `strangers_on_disk` empty, `refused` 0;
- `metrics_verified` 15, `inference_verified` **11**, `regime_verified` **14** — and the
  qualification closes exactly: the four units with `inference: NOT_APPLICABLE` are `pilot_ae`,
  `ae_s1`, `ae_s2`, `ae_s3` (auto-encoders, no forecast to replay) and the one with
  `regime: NOT_APPLICABLE` is `controls` (no detector). `verified + NOT_APPLICABLE = declared`, per
  fact;
- the warehouse population is **16**, and the sixteenth member is exactly **`prepare`**, the
  governance unit — not a sixteenth cell, and no cell is missing from the 15.

**Finding 1 (presentation, no number wrong).** Two different populations are printed adjacently and
the extra member is never named anywhere in the return. A reader is invited to conclude that one of
sixteen units failed to verify. RP51 of this very round states the rule the presentation breaks: *"a
registered unit that did not close is **named**, never absorbed into a total."* Here a registered
unit that *did* close is absorbed into a total. The repair is one word. It is pinned as a rule so it
cannot be lost again.

## 4. The numbers, recomputed — including from the weights

Every number of the return's result table re-derives from the stored arrays:

| | recomputed | published |
|---|---:|---:|
| denominator (mean abs change over the **horizon**, train origins only) | 0.6162615768463073 | 0.6162615768463073 |
| R0 mean / sd (n=3) | 0.887495 / 0.012662 | 0.8875 / 0.0127 |
| R1 mean / sd | 0.903406 / 0.017462 | 0.9034 / 0.0175 |
| R2 mean / sd | 0.896795 / 0.023554 | 0.8968 / 0.0236 |
| linear ridge | 0.8851672 | 0.8852 |
| persistence / seasonal daily | 1.0018020 / 1.1872544 | 1.0018 / 1.1873 |
| paired R1−R0 / R2−R0 / R2−R1 | +0.015910 / +0.009300 / −0.006611 | +0.0159 / +0.0093 / −0.0066 |
| AE cost, amortised per consuming fit | 137.156 s, 22.859 s | 137.156 s, 22.86 s |
| fits `CENSORED_BY_BUDGET` | 4 (`R0_s2`, `R1_s2`, `R1_s3`, `R2_s2`), each at the 4 000 ceiling | four of nine |

Every cell's scaled error is its own MAE over the run's own denominator; every regime mean and sd is
the mean and sd of its own three cells; every paired difference is the mean of the per-seed
differences. Every fit's restored checkpoint is the argmin of its own validation curve — checked on
all nine, not sampled.

**The naive was hunted and is where it should be.** The denominator is computed on the **train**
origins (40 080 of them, lag = the horizon), i.e. on rows the model did see — which is what an
in-sample MASE denominator is — while the persistence *control* is computed on the evaluation rows
(0.6173721). The two differ by 0.18% and are never interchanged. The non-standard part is the lag
(the horizon, not one step), and RP57 renames the field for exactly that reason; this range published
it as "MASE" and that is the erratum RP57 carries.

**The evaluation rows are the same rows.** All nine cells' `arrays.npz` carry `eval_origins`
identical to the prepared set; train and evaluation origins are disjoint with a gap of **121** rows
against a declared purge of **120**; the 61 origins whose label row is the slice's single non-finite
value are excluded, which is why `40 141 − 61 = 40 080`.

**The regime identity, from the reloaded weights and not from the record's summary.** Per seed the
three regimes share one initial checkpoint; R1 and R2 import the same detector, whose file digest is
the bytes of that seed's own auto-encoder; R1's detector after the fit **is** the detector imported
before it (2 896 of 8 127 parameters frozen, confirmed from the reloaded graph); R0's and R2's moved.

### 4.1 The audit extended the replay from 512 rows to 10 020 — and it holds

**Finding 2 (verification coverage).** The closure's replay is a real fresh-process reload, and it
covers `ev[:512]` — the **first 512 contiguous** of 10 020 evaluation origins, **5.11%** of the rows
the published metric is computed on, and at 59/60 overlap barely more than one situation by this
programme's own reading rule. The published MAE was therefore never covered by a replay.

So the audit performed the replay the closure did not: all nine cells, **all 10 020 origins**, fresh
process per cell, weights reloaded from `weights.weights.h5`.

| | |
|---|---|
| cells replayed | 9 / 9, 10 020 rows each (19.6× the closure's coverage) |
| max &#124;replay − stored&#124; | **1.31·10⁻⁶ kW**, against this repository's own tolerance of 1·10⁻⁵ |
| MAE and scaled error | reproduce the published values to ten decimals, every cell |
| parameter counts | 8 127 total, R1 frozen 2 896 — the recorded ones |

The counterexample was hunted and **not** found: the coverage gap is real, the numbers behind it are
not wrong. The finding is that the retained closure does not establish what the audit had to
establish for it.

## 5. Refusals exercised, and the guards re-killed

Malformed cases **built here**, each refused by name (`tests/test_rp49_rp64_audit.py`):

| contract | malformed case | refusal |
|---|---|---|
| governed delivery (RP49/RP51) | a delivery belonging to another design | `REFUSED: the delivery in this root belongs to another design` |
| | a unit with no delivery of its own | `REFUSED: unit 'R0_s1' has no governed delivery of its own; it does not run` |
| | the delivered bytes changed after confirmation | `REFUSED: the delivered bytes changed after the delivery was confirmed` |
| | the cached bytes gone | `REFUSED: the delivered bytes are gone from the cache` |
| | no delivery at all | `REFUSED: this run has no governed delivery` |
| GOVERNED as facts (RP52) | delivery stamped after the start of the work | `the delivery did not precede the work` |
| | terminal accepted before the work started | `the terminal was accepted before the work started` |
| | an impossible terminal state | `carries an impossible state 'SORT_OF_DONE'` |
| | a digest the service could not have returned | `carries no digest the service could have returned` |
| | a reconciliation that is incomplete | `is absent, incomplete or unsuccessful` |
| | reconciliation lists missing | `does not list missing_units` |
| | another design / another unit / absent receipts | refused, and `HISTORICAL_UNGOVERNED`, never promoted |
| chronology (RP55 defect 4) | the same instant as `…19:01:29Z` and `…19:01:29+00:00` | **accepted** — and a real inversion in either direction still refused |

Re-run from scratch today, not read from the round's own report:

- **the whole mutation battery: 15 killed of 15, 0 survivors** — including `M02` (the scaler reaching
  later rows), `M03` (a window crossing a gap), `M04` (activity judged with a future row), `M05` (R1
  not frozen), `M10` (the pilot scoring the test split) and this round's five
  ([`MUTANT_BATTERY_RERUN.json`](../evidence/RP49_RP64_AUDIT_20260926/MUTANT_BATTERY_RERUN.json));
- the five batteries of the range's own repairs: `test_df_e1_first_child`,
  `test_df_e1_receipt_concurrency`, `test_df_e1_chronology`, `test_df_e1_close`,
  `test_stl_norm_contract` — **69 passed**; and `test_df_e1_governed_route`,
  `test_df_e1_governing_report`, `test_df_e1_pilot`, `test_df_e1_regimes`, `test_df_e1_seal` —
  **50 passed**;
- the four **financial-data** rules of the untimed-archive repair, in that repository —
  **4 passed**: the whole archive `AS_IS` with `UNDECLARED` scope and `UNKNOWN` availability, and any
  range refused, before a holdout date as well as after.

The suite totals the return reports are the totals in its own summary file: `3 failed, 2 289 passed,
39 skipped, 8 errors`, the 3 and the 8 being the repository's documented stale legacy suite
(`AGENTS.md`).

## 6. Counterexamples looked for, and what was found

| hunted | found? |
|---|---|
| a population off by one | **Yes, as presentation** (§3): 16 and 15 are two populations and the sixteenth is never named. No unit is missing |
| a metric computed on rows other than the ones it claims | **No.** All nine cells share the prepared evaluation origins; the labels are `Y[origin+60]`; train and evaluation are disjoint past the declared purge |
| a naive computed on rows the model never saw | **No.** The denominator is train-only by design and declared; the persistence control is on the evaluation rows; the two are never swapped. The lag is the horizon, which RP57 renames |
| a digest that matches only because it was re-derived from the same corrupted source | **No.** The sealed pre-execution design and the executed design are the same bytes; the data digest is the digest of the `.npz`; the metrics were re-derived a second way — from the saved weights |
| a figure quoted against a ceiling it is not charged to | **Yes** (§6.1) |
| a verification narrower than the claim it verifies | **Yes** (§4.1), and the claim survived the widening |

### 6.1 Finding 3 — the CPU figure set against the cap is not the one charged to it

The return says **"2 535.5 CPU seconds of an 11 000-second cap"**. `2 535.504` is
`spent_cpu_seconds_root` and it is exactly the sum of its own components
(137.156 + 992.425 + 506.0 + 829.399 + 65.673 + 4.851). But the report that governs the run carries
`already_spent_seconds: 250.0` and `spent_cpu_seconds: 2 785.504` against `cap_seconds: 11 000.0`:
**the runner charges 2 785.504 to that ceiling, not 2 535.504.** Both are far inside it and nothing
scientific turns on it; the accounting sentence is wrong by 250 seconds, and the same round's phase-1
sibling quotes the cap the other way (its `spent_cpu_seconds` has no `already_spent` term at all).
Two runs of one programme quote one ceiling by two conventions.

## 7. What this disposition does not reach

- **The two workers' cost remains UNMEASURED**, as the return declares it. Their jobs did not run
  inside an accounted scope, so no figure exists; the audit did not estimate one and will not. The
  repair the return names (routing worker jobs through the same wrapper) is still not done.
- **The `HISTORICAL_UNGOVERNED` units of the earlier pilot stay where they are.** Nothing here
  promotes one.
- **`ALL_VERIFIED` is not comparability.** Every row of the owner's closure table for these units is
  `NOT_COMPARABLE` against the literature, with its reason. This disposition changes none of that,
  and §6 of the partial seal remains the honest headline of the measurement programme.
- **No number of this range was recomputed under today's code for the four blocks closed under code
  that has since drifted** (partial seal §5.2). That was not this range's work and is not claimed.

## 8. Ruling on RP49–RP56

**ACCEPTED, with three findings of record and no number withdrawn.** The range's identity,
population, metrics, regime construction and governance all re-derive from its retained artifacts;
its refusals all still fire, by name; its five self-reported defects each die under their own
mutation, re-killed today. The three findings are two presentation defects (the unnamed sixteenth
unit, the cap convention) and one scope gap in the verification (the 512-row replay), and the audit
closed the third itself by replaying every row of every cell.

**The five defects the round found in its own work are the strongest evidence in it.** A round that
executes a route for the first time and reports five of its own defects with a mutant each is worth
more than a round that reports none.

## 9. Ruling on the four modules

The rule again: **a module is unblocked only if its own retained artifacts sustain it.** The blocker
being repaired is `MOD_E1_EXTERNAL_REVIEW`. Discharging it does not discharge anything else, and
three of the four modules fail on their own prerequisites.

### MOD-FROZEN-PREFIX — **UNBLOCKED**

Its deliverable is *"versioned derived dataset with direct/cache parity, temporal support and
complete prefix lineage"*, and its upstream (`MOD-ARCH-COMPARE`) is `EXECUTED`. What it must
generalise, this range built and this audit verified from bytes: a prefix whose design and data are
identity-pinned and whose sealed form is byte-identical to the executed one (§2); direct/cache parity
whose refusals fire by name on cases built here (§5); temporal support measured, not asserted — a
perfect one-minute grid over 50 400 rows with zero missing minutes and no duplicated stamps, the
label equal to the panel column element by element (reproduced independently), disjoint spans past
the declared purge, a scaler fitted on train windows only whose guards die under mutation (§4, §5).
No claim this module depends on is refuted.

It carries two repairs as its first two items, neither of which blocks starting: the lag table it
would read to choose contexts must be restated with its estimator and its bias (RP57–RP64
disposition §5.2), and governed sequence materialization must be verified **per split**, which no
artifact of either range does — there is one split pair on one panel and the test split was
deliberately never read.

### MOD-CORE-PRETRAIN — **BLOCKED**

Not for want of a review, and not because the pretraining evidence is bad: it is good and it is
negative, verified here at full evaluation-set scale from the saved weights — R1 is 0.0159 *worse*
than R0 and R2 is 0.0093 worse, with the frozen and fine-tuned detectors proved from the reloaded
graphs (§4). The block is the instrument's resolution. This module must measure *independent H-CORE
effects*, and the effects it would measure are ~0.01 in scaled error, while the same protocol with
**all 40 080 train labels scrambled** loses only 0.049 kW (RP60, verified) — five times the
difference to be resolved — and the arms of the one recipe contrast that was run were not
budget-matched (11 762 optimiser updates against 10 270; RP57–RP64 disposition §5.1). Running it now
would produce differences indistinguishable from the protocol's own noise.

**What must exist first:** (1) a measured resolution for this instrument — the smallest H-CORE
difference it can separate from its own seed and budget variation on this task, derived from the nine
retained cells and the scrambled-label control, which needs no new training; (2) budget-matched arms,
with the monitor held fixed, as the post-RP63 factorial did; (3) MOD-FROZEN-PREFIX, which it also
depends on and which is not done.

### MOD-CONF — **BLOCKED**

The good news is verified: **the reserve is intact.** `exposure: NO_TEST_ACCESS` is a closure
refusal, not a note, and the mutation that lets a pilot score the test split dies (M10, re-killed
today); no reserved split was reopened anywhere in either range. But this module's own next action is
*"review fixed method before independent E0 synthetic and E2 public reserves"*, and across these two
ranges the method was still moving: five runner defects found only on execution, then the objective
and the monitor changed, then the loss/optimiser factorial, then the scaled-error field renamed with
an erratum, then the normalization space turned from a shape into a declaration. Worse, the artifact
that pins the standing recipe — the Huber/AdamW v2 design `be2e776e5c64a842…` — **is not retained in
this repository** (partial seal §5.1): four arms of the current best recipe rest on a digest with no
document. A confirmatory protocol cannot be fixed against a design that does not exist as bytes.

**What must exist first:** (1) that design document retained, so the standing recipe rests on bytes;
(2) one sealed design that freezes the confirmatory method — arms, reserve, stopping rule — with no
recipe change between the freeze and the run. And a procedural limit the owner's grant does not
cross: this module's owner is *"Musashi + Satoshi"*. The grant authorises me to decide two absent
**range reviews**; it does not make me both owners of a module.

### MOD-E3 — **BLOCKED**

RP53's repair is real and verified as far as it goes — a terminal verdict with no fill releases or
updates the pending state from the broker's own events by `order_ref`, the decided quantity reaches
the broker through the environment's execution plugin, and the minimum executable latency is 2 bars,
measured. The return itself states the limit: *"None of this is profitability or a trained RL
policy."* Three things this module needs are absent from both ranges.

**What must exist first:** (1) `BUSINESS-CONTRACT` **executed**, not `DESIGNED` — observations,
targets, actions, availability, the weekly cycle, cost and the data gaps, as an artifact; (2) at
least one governed forecasting result **on the domain MOD-E3 trades**: every retained E1 measurement
is a household-electricity panel, and the only price-series tables in this corpus are the 85
extraordinary legacy ones, all `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND` (RP57–RP64
disposition §4); (3) the H-CORE design whose RL counterpart this module *is*, which MOD-CORE-PRETRAIN
owes it and which is blocked above.

---

## 10. Costs, and what was refused

CPU only, no GPU (`CUDA_VISIBLE_DEVICES=''`, `cuInit` fails in the log and that is the proof), every
job under `crispdm-run -m 6G -t 900 -n rpaudit`, env `trading-stack`. The audit tool runs in seconds;
the 52 rules in 0.7 s; the two full-evaluation-set replays (18 cells × 10 020 origins) and the
mutation battery are the only heavy work and all of it is inference or subprocess testing. No
training, no allocation, no new campaign, no governance or warehouse contact, no host touched, no
committed sample overwritten.

Four things were refused:

1. **I refused to accept the range on the strength of its own reporting.** Every number in §4 was
   re-derived, and then re-derived a second way from the weights.
2. **I refused to let the 512-row replay stand as the verification of a 10 020-row metric.** It is
   named as a finding and the gap was closed by measurement, not by argument.
3. **I refused to write, imply or sign anything in the reviewer's name.** The absent review stays
   absent; this is an audit that stands in its place and says so in its first line.
4. **I refused to unblock three modules whose own prerequisites are missing**, which is the whole
   point of being asked to audit rather than to unblock. One module is unblocked because its
   artifacts sustain it; three are not, each with its named missing thing.

