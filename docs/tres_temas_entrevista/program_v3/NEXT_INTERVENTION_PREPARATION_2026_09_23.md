# Preparation of the next doctoral intervention against the strong reference

Status: **PREPARED, NOT AUTHORIZED TO FIT.** Written under RP139 while the RP135 L512 campaign trains on WORKER_B.
Nothing here starts a run, changes a sealed design, or touches the pinned training checkout. It states what the already
planned intervention is, what must be held fixed, what is still missing before it may be measured, and what it would cost.

## 1. Which intervention, and why this one

The plan's own order is fixed and this document does not reorder it. The queue in
[EXPERIMENT_EXECUTION_QUEUE.json](EXPERIMENT_EXECUTION_QUEUE.json) lists three future designs, all
`PREPARE_NOT_AUTHORIZED_TO_FIT`: the doctoral extractor/pretraining intervention against the accepted reference, then
FIN-LOSS-OPT, then weekly forecasting and RL. The first is the subject here, as RP135 and RP139 direct.

The intervention is **the learning regimes of the modular detector**, defined in
[13E v2](13E_v2_E1_TASK_SHEET_AND_REGIMES_2026_09_19.md):

| regime | detector weights at the start | during the fit | what it isolates |
|---|---|---|---|
| R0 | random, from the replicate's shared initial checkpoint | trainable | no pre-training: the control |
| R1 | imported from the masked auto-encoder pre-trained on this task's TRAIN windows | frozen, byte-identical after the fit | pre-trained features as a fixed representation |
| R2 | the same imported bytes as R1 | adjustable | pre-training as an initialisation |

R1 and R2 differ in exactly one factor, whether the detector is optimised. R0 and R2 differ in exactly one factor, where the
detector starts. A regime is proven per unit by `tools/df_e1_regimes.py`, never declared. The chain after it is
MOD-FROZEN-PREFIX and then MOD-CORE-PRETRAIN, which has its own separate gate and is not folded into this contrast.

Two conflations the plan forbids and this preparation keeps apart: **ARCH-0**, the no-learned-extractor architecture control,
is never the definition of R0; and the E0 H3 arm that freezes the whole extractor trained on the synthetic generator is not
R1, now or retrospectively.

## 2. What the pre-training is, exactly

**Dated correction, 2026-09-23 (RP142).** The first version of this section said the auto-encoder's internal validation uses
the inputs of the DEV validation windows. That repeated an obsolete rule and is wrong. The rule the code actually implements,
and the one this successor fixes, is a **purged tail of the TRAIN origins**. The earlier sentence is superseded here and in
the dated successor to the 13E task sheet; neither is erased.

Masked reconstruction of the preprocessed input: a fraction of time x channel positions is zeroed and the loss counts the
masked positions only, with the mask travelling in the target tensor. It consumes **the outer TRAIN windows only**. Its
internal validation is a chronological tail of those same TRAIN origins, separated from the AE's own training origins by a
purge of at least `window + horizon` origins so that no validation target draws on a row any training window saw. The outer
validation split is **never read during pre-training**: it may later select the downstream forecasting checkpoint, as
declared, but it is neither AE training data nor AE early-stopping data. No outer-test access for any selection, at any stage.

Membership is **proved, not asserted**, before any fit: the AE validation origins are a subset of the outer TRAIN origins,
disjoint from the outer validation origins, disjoint from the AE training origins, and separated from them by the declared
purge; and a future perturbation of rows beyond each window's support must leave that window's inputs and targets unchanged.
Labels being absent from a window is not evidence that it belongs to TRAIN.

The decoder is a separate per-branch 1x1 convolutional stack, saved apart and never connected at inference. Reconstruction
error is a diagnostic and never a result of the task. One auto-encoder per seed: R1 and R2 of a seed consume that seed's
detector file with its digest recorded per unit, and R0 of the seed shares the same initial checkpoint. Three seeds show
optimisation variability and are not a power calculation.

## 3. Held fixed across the regimes

Grouping, fusion, readout, architecture, preprocessing, window, horizon, split, scaler, update ceiling, stopping rule and the
evaluation set. None of these varies in the regime comparison. The matched naive and seasonal controls stay in the same rows
as the model, on the identical target rows.

## 4. What "matched against the strong reference" requires

The governing sentence is [SOTA_FIRST](SOTA_FIRST_2026_09_21.md): after independent reproduction acceptance, the doctoral
interventions are measured against the strong reference **under matched data, tuning and training budgets**. Operationally:

- **Data identity.** The 321-client hourly ECL benchmark, the same processed dataset digest, channel ordering, temporal
  boundaries and scaler-fit population as the reference. UCI235 is a different task and our custom client subset is not the
  benchmark; neither may stand in.
- **Budget.** The regimes are compared at the same task budget, the same update ceiling for the fit, and — as a second and
  explicitly separate reading — at total cost including the auto-encoder that R1 and R2 consume. A truncated curve or a
  `BUDGET_LIMITED` verdict is a statement about the budget, never evidence that pre-training does not help.
- **Metric.** The reference's own normalized reduction, reported as the reference reports it, with our float64 diagnostics
  kept separate. No comparison of figures produced under different protocols.
- **Comparability label.** Against protocol A (Table 8, L = 96) the comparison is to a reproduced reference. Against
  protocol B it inherits `PUBLISHED_WITH_UNRESOLVED_LOOKBACK`: executing the released L512 script does not resolve Table 9's
  undisclosed per-horizon lookback, so that column is not an exact matched comparator.

## 5. Prerequisites, and which are unmet today

| prerequisite | source | state on 2026-09-23 |
|---|---|---|
| independent acceptance of the reference reproduction | SOTA_FIRST; master plan RP90-RP97 | **unmet.** Protocol A: T = 192, 336, 720 in operational agreement; the three T = 96 cells are MEASURED_REPLAY_UNVERIFIED and their original-device replay is the open item. Protocol B: 3 of 12 cells trained, none replayed or closed |
| all four horizons x three seeds under the accepted recipe | PROJECT_METHOD_STATE, `sota_first_override.completion_requires` | **unmet** for protocol B; protocol A has no verified four-horizon mean |
| sealed matched modular contrast | EXPERIMENT_EXECUTION_QUEUE | **not written yet.** This document is its input, not the seal |
| historical custody and content-specific deletion repaired | same | RP132-RP134 delivered the consumers; eight legacy regeneration records stay qualified |
| governed data delivery for the intervention's inputs | CONCURRENT_EXECUTION | available for ECL through the restored transport; measured this round |
| owner authorization to fit | queue status | **PREPARE_NOT_AUTHORIZED_TO_FIT** |

## 6. Resource plan, measured rather than assumed

The regimes fit the modular detector, not TimeFilter, so the per-cell cost is not the reference's. What is measured today and
what the seal will need to add:

- Measured now: the L512 reference costs about 1,258 CPU seconds per H96 cell on the external 5090, and the completed
  protocol-A campaign shows cost is driven by how many epochs early stopping actually runs, not by the horizon.
- To be measured before the seal: one bounded pilot of the auto-encoder and of a single R0 unit, to fix the update ceiling and
  the per-unit cost. That pilot is itself an experiment and needs its own authorization; it is named here, not run.
- Placement: the external 5090 stays first choice and is fully occupied by RP135 until the twelve cells finish. WORKER_A's
  RTX 4090 and the coordinator's RTX 4070 are eligible after fresh admission, which the coordinator did not pass today at
  22 % utilization. The internal 5070 Ti shares host RAM with the external card and is used only if combined admission fits.
- Isolation: a separate worktree and a separate run root, never the campaign's.

## 7. The financial lane, prepared in parallel

FIN-LOSS-OPT is designed and NOT started: the full factorial of MAE and Huber against Adam and AdamW, paired seeds, two
receivers never compared across, weekly walk-forward folds with weekly retraining, horizons declared in advance at 6 and 72
hourly steps, skill against persistence on the same rows, and the reserve from 2025-01-01 never read. Two things block it and
neither is repaired by anything in this round:

1. **The governed financial resource is undeliverable.** The sealed cost pilot was refused by the service with 422, a
   resource without an availability contract, and the HistData producer lineage stays unchanged with no metadata invented.
2. **There is no financial reference.** The benchmark contract `fx.eurusd.1h.FIN-LOSS-OPT` is NOT_COMPARABLE until a
   reference method is re-executed under it. The electricity result does not supply one: the electricity ranking does not
   choose the financial or RL winner, and there is no financial winner today.

The preparation work that can proceed without opening the reserve is the availability contract and the choice of a financial
reference method with its own domain literature. Both are document work, and neither is a fit.

## 8. What this preparation explicitly does not claim

It does not select an architecture, optimizer, loss, pre-training regime or trading policy from electricity results. It does
not treat a running executor or an accepted terminal as scientific acceptance. It proposes no filler experiment to occupy
idle hardware, and it does not revive the suspended household pilots. If the regime contrast comes out against pre-training,
that is a result and the modular extension survives it; the plan says so and this preparation does not hedge it.
