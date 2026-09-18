# 12B — Development utility design, successor of 12A (P1–P2), and its governed execution (P3)

Order: `MUSASHI_UTILITY_O1_O4_REVIEW_AND_P1_P4_2026_09_17.md`. Supersedes
`12A_NEXT_DEVELOPMENT_UTILITY_DESIGN_2026_09_17.md` (instance `386fe033…`, preserved as history,
never executed). Sealed instance: `docs/audits/evidence/d3_k5_20260917/P2_NEXT_DEV_DESIGN_12B.json`
(`55ef8321…`), schema `df_utility_dev_design.v2`, built by `tools/df_utility_next_design.py`.

## What changed against 12A, and why

| finding | 12A | 12B |
|---|---|---|
| H_A had no calibration path of its own | `calibrate()` always simulated raw/transformed | `calibrate(..., branch_a, branch_b)`; the record (`df_utility_calibration.v2`) names its pair, widths and rows policy; `calibration_supports` refuses another pair; the simulation and the experiment go through the same `contrast` callable with the same pair |
| inner contract not validated | only the outer digest and target/horizon | every family protocol is rebuilt and checked: digest, base digest, every inherited field (target, horizon, model, window, blocks, margin, alpha, ridge λ, min rows), family = member list, members = operators × hypotheses (no duplicate, no missing, pairs per hypothesis), plan derived from α/m and the bound resource's length, one calibration contract per member, eligibility from the bound cells record |
| purge ignored the wider history | `horizon + reach + window` | the boundary is derived per block from the support the features actually consumed (widths, gaps, delays, emission) and the training side's consumption (label to t+h, representation to t+reach); recorded by row identity; a violation refuses the contrast |
| H_T asked "at least as good" | non-inferiority wording | "does R alone IMPROVE on raw lags" — superiority at the sealed margin; margin and inference unchanged |
| equal width ≠ equal information | — | stated: the capacity control equalises the number of inputs only |
| replication by name | seeds 12/13 as lists | explicit map selection → replica of the SAME generator family, perturbation, SNR, missingness, length and variable, different seed and data digest, both read from the bank's `UNIT.json` |

## Calibration contracts and cost

One contract per (family, operator, hypothesis): 6 families × 3 operators × 2 hypotheses =
**36 contracts**, 358 simulations each (α/6 at 0.95), plus 36 contrasts. A record never
transfers across pairs, operators, protocols, lengths or harness code; no unfavourable bound
licenses more simulations. The runner (`tools/df_utility_dev_run.py`) measures a **governed cost
pilot** (6 simulations per contract type) and projects the campaign contract by contract before
launching; beyond the aggregate ceiling (4 h CPU including calibrations, contrasts, failures and
retries) it writes a feasible plan instead. Exhausted mid-way, it stops and keeps incompletes.

## Governance per family

freeze-pre → calibration campaign (one unit per contract, `operator__hypothesis`) → before_run
per child → isolated calibration child → seal one protocol per contract → contrasts campaign →
before_run per contrast → isolated child with its pair → terminal with the child's instants and
cost, tagged with hypothesis and pair → reconcile → DEVELOPMENT envelope. Resume re-verifies
completed attempts and binds them to their recorded job (harness O2).

## Reading the results

`DOES_NOT_ADVANCE` is not equivalence and does not show an operator useless elsewhere.
`INCONCLUSIVE_UNCALIBRATED` is descriptive (its contract's derived bound exceeds α/6): deltas,
losses, intervals, coverage and cost are still recorded. A favourable bound covers only the
measured null and scope. A pair that advances in the selection family **and** in its mapped
replica is `PROPOSED_FOR_REVIEW` — never confirmed, publicly eligible or licensed. α/6 per family
is not a global control over the campaign's discoveries.
