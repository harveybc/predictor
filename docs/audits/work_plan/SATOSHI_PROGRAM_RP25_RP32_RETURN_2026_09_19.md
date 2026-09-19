# Satoshi — return of RP25–RP32: composed E0 closure and the corrected E1 regimes

Order: [MUSASHI_PROGRAM_RP25_RP32](../../handoffs/MUSASHI_PROGRAM_RP25_RP32_2026_09_19.md) (dictum F1–F6 of
[MUSASHI_RP17_RP24_REVIEW](MUSASHI_RP17_RP24_REVIEW_2026_09_19.md)). Base `de8ae34`. Ending state:
`RP25_RP32_E0_COMPOSITION_AND_E1_REGIMES_READY_FOR_REVIEW`. Everything below was executed; nothing is
projected. No GPU, no live, no reserve, no financial operation, no RL training.

## RP25 — PRE and the corrected meaning of the regimes

PRE preserved before any correction in [`RP25_PRE/`](../evidence/d3_k5_20260917/RP25_PRE/): 13E v1, the v1
`df_e1_tasks.py`, the v1 `df_mod_e0_arch_verify.py` and `RP24_REVIEW_REPRODUCED_PRE.json`, which is **byte-equal**
to the reviewer's own `results.json`: the five union cases, the E1 cases (target, metadata, gap) and the RL action
that opens a short reproduce before the repair, not after it.

[13E v2](../../tres_temas_entrevista/program_v3/13E_v2_E1_TASK_SHEET_AND_REGIMES_2026_09_19.md) supersedes 13E v1
in exactly one point (v1 stays in the repository, with a pointer at its head): **R0** = random detector, trainable;
**R1** = detector pre-trained by the masked auto-encoder on this task's train windows, frozen; **R2** = the SAME
imported weights, adjustable. Grouping, fusion, readout, architecture and preprocessing are other factors and are
held constant in the regime comparison. ARCH-0 is an architecture control, never the definition of R0, and E0's
H3 arm (whole extractor frozen on the synthetic generator) keeps its own scope and is not re-presented as R1.

## RP26 — valid union of closures, E0 closed without retraining

`validate_closure` now validates each closure on its own before any union (design digest, self-digest, population,
members vs cells on disk, pilots and inherited donors by their contracts), and `merge_successor` validates BOTH,
the parent–successor relation, equal design keys, and for every inherited donor its identity, required state,
weights/code, replay scope (VERIFIED + CURRENT_CODE in both closures) and the equivalence of data, split, window,
horizon, training and donor. A disagreement between two verifications becomes a recorded disposition
(`CONTRADICTION_EXCLUDED` / `DONOR_CONTRADICTION_EXCLUDED`), never a silent choice of the parent's.

Composed population: 128 cells (112 parent + 16 successor), 124 verified, 4 DX cells FAILED and named;
contradictions: none. Effects and tables published through the governance route as
`mod-e0-arch-stage-v1-effects-v3-composed`, reconciled (missing 0), with the previous ones preserved
([`RP26_ARCH_COMPOSED_EFFECTS_v3.json`](../evidence/d3_k5_20260917/RP26_ARCH_COMPOSED_EFFECTS_v3.json),
[tables](../evidence/d3_k5_20260917/RP26_ARCH_COMPOSED_TABLES_v3.md),
[receipt](../evidence/d3_k5_20260917/RP26_EFFECTS_V3_PUBLISH_RECEIPT.json)). No model was retrained.
Tests: `tests/test_df_mod_e0_arch_effects.py`, 24 cases, including the five dictum cases, an altered design with
the old digest and with a recomputed one, a changed regime/window/donor, an omitted population, historic vs
current replay scope, the manual γ per replicate as a positive control and the refusal of the full CLI.

## RP27 — one executable E1 task, by roles, support and availability

`tools/df_e1_loader.py` is the contract the real loader consumes: roles resolved by name (features, targets,
metadata, controls, ignore), an unforeseen numeric column **refused** (admitted only when declared ignored or
metadata), timestamps never coerced to numbers, and ONE enumerator that emits origin ids, times, support,
labels and masks — every count in this round derives from it. Household keeps `Global_active_power` as target and,
in the primary pilot, its own past as a feature (so the model has the information persistence and seasonal naive
have); the ablation without its history exists as a separate, named contract. Gaps, duplicates, disorder, missing
targets and extra metadata are tested against the full loader and the tensor the model receives: a window whose
support crosses a gap or an ambiguous row is withdrawn, time is never compressed. The scaler is fitted on train
windows only, and altering validation/test changes neither the scaler nor any previous window. Electricity
activation is causal (judged at the origin), never-active clients are named and a measured zero after activation
stays a valid label. Every arm scores one common evaluation set with the same origins, targets and denominator.
Tests: `tests/test_df_e1_loader.py`, 9 cases (a missing target withdraws 60 origins with history and 1 label
without; an ambiguous DST label withdraws 61 windows).

## RP28 — temporal data and spectral resolution

DST traced raw → parser → panel without downloading anything
([`RP28_DST_TRACE.json`](../evidence/d3_k5_20260917/RP28_DST_TRACE.json)): the panel equals the raw bytes, the
March change days carry 96 rows, and the zero hour is 01:00–01:45 for most active clients but not for all of them;
the producer's statement is now recorded as **hours of 23/25-hour days**, not "23/25 records", and the October
aggregated interval and the March ambiguous labels are dispositions with their support, not filters. Household is
not blocked by an electricity-specific problem.

Spectra (`tools/df_e1_spectrum.py`): segment window, resolution, observable frequencies, normalisation, the column
rule and missing handling are declared; a band without bins is `NO_RESUELTO`, never a measured zero (household
weekly and 35-day bands are `NO_RESUELTO`; the daily band is measured). Sensitivity to the last third of train is
reported, per-variable and aggregate grains are told apart, and the first 32 electricity columns are declared a
sample by column order, not a representative set. Contexts are fixed in physical time **with the model's real
reach**: the pilot's ARCH-A with the sequence readout reaches 7 samples (420 s) inside a 60-sample (3 540 s)
window, and that is stated in the design rather than implied by W.

## RP29 — R0/R1/R2 implemented and proven

`tools/df_e1_regimes.py`: shared initial checkpoint per seed, masked auto-encoder pre-training of the detector with
a **separate decoder** (saved apart, never connected at inference; the modular model has no decoder layer), the loss
counting masked positions only with the mask carried in the target tensor, trained on train windows with an internal
validation of inputs only. R1 leaves the detector's weights and non-trainable states invariant (no gradient
variable, identical digest after the fit); R0 and R2 receive gradients at the detector and move it; the integrator,
adapter and core are never frozen by module name. Proven by gradients, weight changes, resume and reload — not by
`trainable=True` in a config. Controls: shifted/leaked labels (the leak shows, the real tensors do not carry it),
a model with zero updates keeps every weight, and selection on the test split is impossible by construction.
Tests: `tests/test_df_e1_regimes.py`, 4 cases.

## RP30 — household DEV pilot (executed)

Design sealed before any outcome ([`E1_PILOT_DESIGN.json`](../evidence/d3_k5_20260917/RP30/E1_PILOT_DESIGN.json),
digest `807a4a30…`): task `W60_h60` (context 3 540 s, horizon 3 600 s, model reach 420 s), DEV sub-partition = the
last 28 days of the family's train split plus the first 7 days of its validation split (rows 1 412 361–1 462 761;
test rows never read), graph with units/layers/activations/reach, physical grouping held constant, seeds 1–3 paired
(one initial checkpoint per seed shared by R0/R1/R2; one auto-encoder per seed shared by R1/R2), update ceiling
4 000 (auto-encoder 1 500), batch 64, validation stopping with restored checkpoint. Cost and memory pilot first
(300 updates): 0.038 s/update for the auto-encoder, 0.100 s/update for the fit, peak RSS ≈ 0.8 GB; projection
5 325 s with 25 % headroom against the remaining ceiling — it fit, so the pilot was launched, not mutilated.
Windows are gathered per batch; only the train tensor for the scaler is materialised once (135 MB, under the
1 GB preflight cap).

Common evaluation set: **10 020 origins**, identical for every arm (validation admissible 10 020, all with a finite
label and a finite daily lookup). MASE denominator (train persistence at the horizon) 0.6163 kW.

| unit | MASE | MAE (kW) | updates | stop | fit s | child CPU s |
|---|---|---|---|---|---|---|
| R0 s1 / s2 / s3 | 0.8831 / 0.8981 / 0.9149 | 0.5442 / 0.5535 / 0.5638 | 4000 / 4000 / 3762 | budget / budget / early | 246 / 245 / 236 | 258 / 253 / 246 |
| R1 s1 / s2 / s3 | 0.8865 / 0.8955 / 0.9046 | 0.5463 / 0.5519 / 0.5575 | 3135 / 4000 / 3762 | early / budget / early | 55 / 75 / 69 | 64 / 85 / 78 |
| R2 s1 / s2 / s3 | 0.8887 / 0.8912 / 0.9206 | 0.5477 / 0.5492 / 0.5673 | 3135 / 4000 / 3762 | early / budget / early | 192 / 244 / 247 | 201 / 253 / 257 |

| regime | MASE mean | MASE sd (3 seeds) | | control | MASE | MAE (kW) |
|---|---|---|---|---|---|---|
| R0 | 0.8987 | 0.0159 | | persistence | 1.0018 | 0.6174 |
| R1 | 0.8956 | 0.0091 | | seasonal naive (daily) | 1.1873 | 0.7317 |
| R2 | 0.9002 | 0.0177 | | linear ridge (60×7 window) | **0.8852** | **0.5455** |

Paired differences (same seed): R1−R0 mean −0.0032 (per seed +0.0033 / −0.0026 / −0.0103); R2−R0 mean +0.0014
(+0.0055 / −0.0069 / +0.0056); R2−R1 mean +0.0046 (+0.0022 / −0.0043 / +0.0159). **Every paired difference is
smaller than the spread across seeds**: at this budget and on this task the pilot distinguishes no regime, and the
honest reading is that the linear control matches or beats all three. Nothing here confirms or refutes H1.

Costs by phase (CPU seconds): auto-encoders 132 (3 seeds), fits R0 758 / R1 227 / R2 711, controls 5, cost pilot 65;
root total 1 897 s. Same task budget per regime (the same update ceiling); the auto-encoder that R1/R2 consume is
reported apart and added only in the explicitly separate total-cost reading. R1 is the cheapest fit by a factor of
three because its frozen detector has no gradients, which is a cost fact, not a quality fact. Curves and truncation
are recorded per unit: no unit hit the ceiling with its best epoch last, so no result here is a truncation artefact.
Identity proven per seed: shared initial checkpoint, R1 and R2 importing the same detector bytes, R0's detector the
random initial one, R1 frozen (no gradient, identical digest), R0 and R2 learning.

Governance disposition, stated rather than worked around: the public panels are **not** a resource of any lake the
deployed data-gov serves (`financial_files`, `olap_cube`, `predictor_examples`), and a DATASETS campaign completes a
unit only through governed deliveries of every declared dataset, while a SYNTHETIC campaign would misdeclare real
bytes. Every one of the 15 units therefore carries a `governed_terminal.v1` document in
[`RP30/TERMINALS/`](../evidence/d3_k5_20260917/RP30/TERMINALS) plus
[`CAMPAIGN_PROPOSAL.json`](../evidence/d3_k5_20260917/RP30/CAMPAIGN_PROPOSAL.json), the campaign that reports them
unchanged once the panel is registered. The missing object is one lake registration of `public_panels_c126_v2`
in data-gov (config plus a service restart: an owner action, not something to be forged here).
Full table: [`RESULTS.md`](../evidence/d3_k5_20260917/RP30/RESULTS.md) /
[`RESULTS.json`](../evidence/d3_k5_20260917/RP30/RESULTS.json). Tests: `tests/test_df_e1_pilot.py`, 5 cases.

## RP31 — the real weekly long/flat controller

`tools/e3_weekly_controller.py` is a production component, not a policy inside a test: it decides the available
model, applies the declared fallback (`last_valid_or_flat`) and emits the scenario's action. Clock order
cutoff ≤ fit_start < fit_end ≤ release is enforced at construction; release ≤ first_decision is the PLAN, so a late
release is recorded as a miss and covered by the previous model or flat, never shifted backwards. Weeks come from
timestamps ([Monday 00:00, next Monday 00:00) UTC). Positions, equity and orders stay continuous across the week
boundary. A SHORT proposal is refused and recorded; closing uses the explicit close-to-flat action, never a
reversal; sizing is re-derived from the CURRENT equity with the notional bounded by it. Each decision records
(decision time, order time, expected fill time) with a one-bar positive latency, and fills happen at open[t+1],
which differs from close[t] in the fixture. `tests/test_e3_weekly_controller.py`: 8 cases, including an absent
model, a late model, refused proposals and four mutants (no latency, flat→short, week by counter, release ignored)
that must and do fail. Offline software: no financial operation, no RL training, and this is **not** a completed
E3; E3 and H-CORE keep their dependencies.

## RP32 — closure

**RP23 phrasing corrected** where it exceeded its tests, in [the RP17–RP24 return](SATOSHI_PROGRAM_RP17_RP24_RETURN_2026_09_19.md)
(errata block), [13C](../../tres_temas_entrevista/program_v3/13C_E3_WEEKLY_CYCLE_CONTRACT_2026_09_18.md) and
[13B](../../tres_temas_entrevista/program_v3/13B_NEXT_STEPS_E1_E3_AND_LANES_2026_09_18.md): the environment tests
validate neither data availability, nor a late release, nor weekly operation, and "notional ≤ equity" had been
checked only on a fixed equity.

**Outbox race** reproduced on a fixture (`tests/test_governed_outbox_race.py`): two processes given the same
terminal outbox directory race and the loser dies on a missing envelope (`FileNotFoundError`), exactly as it
happened; after the race no terminal is lost and none is duplicated in the accounting (every envelope is one file,
in `sent/` or `pending/`, one unit per sent envelope; the server may be told twice, which its write-once accounting
per unit and generation makes harmless), and a second flush drains what remains. A private outbox per process has
no race, which is the operational rule this round follows.

Tests, budget and hosts: see the closing table below.

## Request

One review of RP25–RP32: the validated composition of the two closures, the E1 task contract and its enumerator,
the DST and spectral dispositions, the proven regimes, the household DEV pilot with its null reading and its
governance disposition, and the weekly controller as software.
