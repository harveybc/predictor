# Return RP17–RP24 — ML inference corrected, the missing control completed, E1 dimensioned from real data

Date: 2026-09-19. Executor: Satoshi. Order: [MUSASHI_PROGRAM_RP17_RP24_2026_09_19.md](../../handoffs/MUSASHI_PROGRAM_RP17_RP24_2026_09_19.md)
over the dictum [MUSASHI_RP9_RP16_REVIEW_2026_09_19.md](MUSASHI_RP9_RP16_REVIEW_2026_09_19.md) (base `6f6c1a0`) and
[13D](../../tres_temas_entrevista/program_v3/13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md). Branch `satoshi/r1-r6-20260914`;
commits of this round in `git log` after `612a593` (merge of the order); the closing commit is this file's last.
Evidence under `docs/audits/evidence/d3_k5_20260917/` (`RP17_PRE/`, `RP18_*`, `RP21_*`, `RP22_*`, `RP24_*`). One audit request, at the end.
Output state: `RP17_RP24_ML_CORRECTIONS_AND_E1_TASKS_READY_FOR_REVIEW`.

## What we keep, what conclusion changes, what the next experiment resolves

| kept | changed | next ML question |
|---|---|---|
| the 112 stage cells, the DX successor, weights, arrays, terminals, replays (120 attempts recomputed by the dictum to 7.77e-16) | γ of the common pair is NEGATIVE in every architecture (A −0.0749, B −0.0678, C −0.0675, 0 −0.0549), not positive: the mixed reading is withdrawn ([errata](SATOSHI_RP16_ERRATA_2026_09_19.md)) | with the 16 new r = 0 readout cells, the balanced factorial γ is now ESTIMABLE: −0.0224 / −0.0223 / −0.0244 / −0.0234 (A/B/C/0): the sequence-fusion advantage is larger with the planted lagged dependence by ≈ 0.02 MASE in every architecture (two replicates, descriptive) |
| readout effects as descriptive, per fusion and r | "readout explains most of the gap" withdrawn; the fusion effect at r = 0 is ≈ 0 (−0.008 / +0.004 / −0.006 / −0.011) and the readout effect is −0.03…−0.06 at r = 0 and −0.08…−0.11 at r = 1 | E1: the readout is a declared factor of the design, not a conclusion |
| costs, curves, stop reasons | no efficiency ordering of architectures; adequacy reported by reference (persistence, denominator, linear point and +0.03, oracle) | the prepared learning-curve ladder (300/600/1 100/2 200 updates) decides allowance vs capacity before any claim |
| E1 catalogue eligibility (licence) | v1 task eligibility withdrawn; task contracts built from governed bytes (usable windows per split and column, physical contexts, measured periodicities, DST, structural zeros, roles) | the E1 design sheet (13E) is submitted for review; the campaign is not launched |

## Per block: IMPLEMENTED · EXECUTED · VERIFIED · REVIEWED

| block | impl. | exec. | verified | reviewed | limits and defects |
|---|---|---|---|---|---|
| RP17 | yes | yes | PRE byte-equal to the dictum on the reviewed checkout with the preserved root (`RP17_PRE/RP16_REVIEW_REPRODUCED_PRE.json`); F4/F5 inventory frozen | no | — |
| RP18 | yes | yes | 9 numeric oracles through the real `effects()`, the survivor-average mutant killed for the expected reason, the real closure reproduces the dictum's γ to 1e-10 | no | the balanced factorial γ was NOT_ESTIMABLE until RP22 |
| RP19 | yes | yes | 18 tests: intact real replay; the dictum's NaN mutation and every field apart; partial document; non-zero exit; cache bound to implementation + environment; 24 cells re-replayed under current code (max |diff| 5.0e-7) | no | 96 stage replays keep the explicit HISTORIC_v1 scope (labelled, not promoted) |
| RP20 | yes | yes | grain tests; composition residual < 1e-9 with the deterministic term; backfill v2 published and reconciled (stage 120, DX 4, historic 69), content equal in the warehouse (265 rows per unit in the successor) | no | two stage units failed once in an outbox race with a concurrent run and were re-published |
| RP21 | yes | yes | adequacy and cost tables from the records; intersection of 24 task × seed pairs | no | learning-curve design PREPARED, not launched |
| RP22 | yes | 16/16 | successor closure TOTAL 24/24 (16 cells + 8 inherited donors, all CURRENT_CODE) and live equal; stage closure v3; merged 2 × 2 at both r | no | the first run crashed at its last terminal flush (outbox race); resumed under a later commit with byte-identical training code (recorded per unit) |
| RP23 | yes | yes | E1 loader tests (5) and RL environment tests (6) pass; contracts from governed bytes | no | the producer's "March DST hour of zeros" is NOT reproduced in the canonical panel (hours 0–5 scanned): recorded, not assumed |
| RP24 | yes | yes | suite, ledger, mutants POST, host states below | no | — |

## RP17 — PRE and withdrawals

`RP17_PRE/`: Musashi's `reproduce.py` executed on a worktree at `6f6c1a0` with `--run-root` (byte-equal output);
`E1_FAMILIES_v1_PRE.json`, the v1 effects/tables and the v1 estimator. Withdrawn: γ positive; readout as the
dominant attribution; `interpretable` by beating the persistence naive; the efficiency ordering; "no effect" for
H2 (errata). Kept: the measurements (bounded numeric backing). 13C's owner dependency: resolved for DEV by 13D;
real operating conditions remain unknown.

## RP18 — contrasts, not averages of survivors

`tools/df_mod_e0_arch_verify.py` v2: contracts derived from the design (members, weights, denominators per
replicate); binding of closure ↔ design identity and population; refusals for foreign design, duplicate,
unexpected arm, identity mismatch of a record, non-finite value; INCOMPLETE with n expected/observed and the
missing members; one estimator for point and replicate bootstrap; H2 requires every control; donor delta on the
SAME pair. Tests `tests/test_df_mod_e0_arch_effects.py`. Published as the successor table through governance
(campaign `mod-e0-arch-stage-v1-effects-v2`, 4 units, reconciled: `RP18_EFFECTS_V2_PUBLISH_RECEIPT.json`).

Stage alone (`RP18_ARCH_STAGE_EFFECTS_v2.json`): d_0 / d_1 / γ common: A −0.0654 / −0.1403 / −0.0749; B −0.0277 /
−0.0955 / −0.0678; C −0.0643 / −0.1318 / −0.0675; 0 −0.0749 / −0.1298 / −0.0549 (equal to the dictum); γ factorial
NOT_ESTIMABLE; readout 2 × 2 at r = 1: −0.110 / −0.077 / −0.102 / −0.095; donor Δ (A, B): +0.083 / +0.065.

## RP19 — reproduction with code scope

`df_mod_e0.py --replay` now records its identity (implementation digest, python, numpy, tensorflow, keras,
threads, platform); the closure requires finite, typed and complete restore/loss/activation evidence and a zero
process exit; the cache reuses a document only for the same bytes AND identity; older documents are reused only
under the explicit `HISTORIC_v1_UNBOUND_CODE` scope for attempts outside `--replay-units`. Tolerances unchanged.

## RP20 — grains and composition

`df_mod_e0_metrics.py` v2: base series, unique inputs, window tensor (with its repetition factor), targets shifted
by h — each with row identity, shape, dtype, bytes, mask, train-only scale, denominator; SNR v1 kept under its name
and SNR total (with the deterministic term) added with its convention; composition `x = s + periodic + cross +
deterministic + noise` checked (residual < 1e-9). Backfill v2 successors (tag `v2_grains_snr_total`) for the stage,
the DX successor and the historic run; live closures compare content, not counts.

## RP21 — adequacy and costs

`RP21_ARCH_STAGE_ADEQUACY.{json,md}`: per cell the references apart, best/last update, allowance, stop, best-at-
ceiling (no convergence declared there), reach, W/P, cost; causes per task (limited optimisation, context reach
below the slowest period, seed variation); costs on the exact intersection (24 task × seed pairs; per host stratum;
donor cost amortised over 12 uses: A 5.5 s, B 10.0 s, C 10.1 s, 0 2.3 s per use). ARCH-B at H2 h3 is within
+0.03 of the linear reference (gaps +0.0192 / +0.0178); in DX it is not (+0.039).

## RP22 — the 16 missing controls

Design `RP22_ARCH_READOUT_COMPLETION_DESIGN.json` (successor of the stage; 16 cells = 4 architectures × 2
replicates × {sequence_gap, summary_last} at r = 0; 8 inherited donors enumerated per cell with run/design/cell
identity; pilots' costs inherited by identity; projected 647 s with headroom). Code equivalence
(`RP22_CODE_EQUIVALENCE.json`): of the 36 functions of the training path only `run_cell` differs, in the DX
routing branch that no H3 cell executes. Executed on omega (416 s of children), donors symlinked read-only, no
retraining (`RP22_ARCH_RC_REPORT.json`: 16 COMPLETED, 8 INHERITED, missing 0). Closures: successor TOTAL 24/24
with CURRENT_CODE replays and live equal (`RP22_ARCH_RC_CLOSE.json`, `…_LIVE.json`); stage v3 with 24 fresh
replays (`RP22_ARCH_STAGE_CLOSE_V3.json`). Merged effects (`RP22_ARCH_MERGED_EFFECTS_v2.json`, tables `…_TABLES_v2.md`):

| arch | fusion 2×2 r=0 | readout 2×2 r=0 | interaction r=0 | fusion 2×2 r=1 | readout 2×2 r=1 | γ factorial (sd, n=2) | γ common pair |
|---|---|---|---|---|---|---|---|
| A | −0.0076 | −0.0578 | +0.0011 | −0.0300 | −0.1102 | −0.0224 (0.0062) | −0.0749 |
| B | +0.0041 | −0.0318 | +0.0240 | −0.0182 | −0.0773 | −0.0223 (0.0014) | −0.0678 |
| C | −0.0059 | −0.0584 | +0.0116 | −0.0303 | −0.1015 | −0.0244 (0.0034) | −0.0675 |
| 0 | −0.0113 | −0.0636 | +0.0074 | −0.0347 | −0.0951 | −0.0234 (0.0019) | −0.0549 |

Reading (descriptive, two replicates): the balanced fusion effect is ≈ 0 without the planted lag and ≈ −0.02…
−0.035 with it, so γ factorial ≈ −0.022…−0.024 in every architecture; the readout effect is present at both r and
larger at r = 1; the common pair mixes both (its γ ≈ −0.055…−0.075). The Conv core vs Dense core remains part of
the procedure, not an isolated test of information preservation; no universal winner; no reserve.

## RP23 — E1 and E3

[`E1_TASKS.json`](../../tres_temas_entrevista/program_v3/E1_TASKS.json) and [13E](../../tres_temas_entrevista/program_v3/13E_E1_TASK_SHEET_2026_09_19.md):
electricity (140 256 rows × 370 clients, 15 min, no gaps; structural zeros for half the clients during their first
year, masked; measured daily 24 h share 18 %, 12 h 12 %, weekly 1.5 %, 53 % slower than five weeks) and
household (2 075 259 rows × 7, 1 min, 1.25 % NaN; daily 7 %, 12 h 9 %); usable windows per split after masks,
purge and horizon, per target column (medians in 13E); contexts in physical units; eligibility separated
(catalogue / task / reserve); prior exposure of Beijing and appliances declared; UCI is a repository, not a
source. Tests `tests/test_df_e1_tasks.py` (masks, purge, structural zeros, future/prefix, gaps, train-only
periodicities). 13D adopted in 13B/13C. RL: `tests/test_e3_weekly_env.py` against the real GymFxEnv — weekly
policy change keeps equity/position/orders/commissions continuous; proportional costs at fills; an action
decided at bar t fills at bar t+1's open; the declared fallback before the release; min_equity termination;
13D normalisation (equity 1, long/flat, notional ≤ equity). Software tests; no financial return. Funding is
not replaced by zero outside the cash-spot scenario.

> **Errata (RP32, dictum F4).** Three phrases of this RP23 paragraph claimed more than their tests supported and
> are corrected here; the tests themselves are unchanged.
> 1. *Data availability.* `tests/test_e3_weekly_env.py` exercises the environment's bar clock and the ordering of
>    the declared instants. It does not measure any real feed's availability or publication delay, so it validates
>    no availability claim; the availability of each feature series remains DECLARED and UNVERIFIED.
> 2. *Long/flat and the fallback.* In those tests the policy, the release check and the fallback were written
>    inside the test. They therefore proved the environment answers such actions, not that a production component
>    decides them. The real component is `tools/e3_weekly_controller.py` with `tests/test_e3_weekly_controller.py`
>    (RP31), including a late release, an absent model, a refused short proposal and two mutants that must fail.
> 3. *Scaling / sizing.* "notional ≤ equity" was checked on a fixed equity. Sizing against a VARYING equity, cash
>    sufficiency per decision and the fill price gap are covered by the RP31 controller tests, not by RP23's.
> Nothing above turns these into a complete E3 experiment: E3 and H-CORE keep their dependencies and obligations.

## RP24 — closure

Suite: `pytest tests --continue-on-collection-errors`: **2 097 passed, 37 skipped, 3 failed, 8 errors** in 30 min 29 s — the 3 failures (`tests/integration_tests/test_configuration_handling.py`) and 8 collection errors are the stale legacy tests AGENTS.md documents, unchanged by this round (`RP24_FULL_SUITE_SUMMARY.txt`); every MOD-E0 / closure / replay / effects / grains / E1 / E3 test passes. Mutants POST: 12/12 guard mutants killed by the behaviour test (RP24_MUTANTS_POST.json; the estimator mutant of RP18 and the 14 replay-field mutations of RP19 are in their test files). CPU of this round (systemd ledger on omega since the order,
`RP24_CPU_LEDGER_omega.json`): **10 180 s** (29 transient units: PRE, tests, closures with fresh-process replays, backfills, the 16-cell successor, effects, the full suite, two mutant batteries) of 14 400 s; workers not used this round (no independent tasks
warranted them). Host states at close: omega load ≈ 3, 20 GiB available, HEAD = closing commit, tree clean; dragon load 0.1, 18 GiB available, worktree 6f6c1a0 clean, no active units; gamma load 0.1, 7 GiB available, worktree 6f6c1a0 clean, no active units (workers synced to the closing commit after this table). Every new unit has a terminal (16 RC cells, 4 effects units,
backfill successors); outboxes reconciled (missing 0 in every campaign of this round); warehouse content checked
through its API for the successor (265 rows per unit equal) and the stage/DX/historic backfills.

## Request

One review of RP17–RP24: the contrast-bound estimator and its oracles, the completed 2 × 2 and its reading, the
replay scope, the grains, the E1 task sheet (13E) before any E1 sealing, and the E3 environment tests under 13D.
