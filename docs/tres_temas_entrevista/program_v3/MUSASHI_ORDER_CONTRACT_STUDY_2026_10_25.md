# SAC Order Contract Study

Owner priority: verify the business learned by the policy before optimizing
hyperparameters. This is the SAC lane only; no work or approval is needed from
the expert-guided lane. Shared data, profiles, encodings and selected branches
remain recipe R3 (120 inputs); no feature-selection rerun.

## Closed Parent

Recipe-native campaign spec
7568df4f3ba7b6e80fb86cce70c42adeb9b4d0b6a227fd947183baa0ba8518cc
finished GROUPED and FLAT at 100000 decisions each. 24/24 reports delivered
with exact warehouse readback and second-machine copies.

| Final policy | TRAIN profit USD | INNER_A profit USD | 2024 profit USD | 2024 trades | 2024 Sharpe | Activity valid in 2024 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GROUPED | 3447.79346 | 143.93964 | -54.358155 | 22 | -0.086136 | no |
| FLAT | -2455.50995 | -2342.44674 | -3694.08158 | 257 | -3.033467 | yes |

Capital 10000; net after configured costs. Costs are not broker verified. FIT
2020 through 2023H1, INNER_A 2023H2, OBS_2024 full development year; TEST closed.
Do not select the positive intermediate grouped checkpoint as the final model.

## New Executable Plan

agent-multi code b117dce0, branch codex/rl-static-temporal-20261018,
docs/work_plan/62_ORDER_POLICY_CONTRACT.md. Spec SHA256:
79c4b87eb64d7285583614afe624a196dbf7c632f0351d2d0f4aa2c4130bfd5e.

1. Real-engine synthetic alpha (16 signed scenarios): next-open fill, long and
   short, SL/TP/ambiguous bars, gaps, day cap, delayed reopening, forced flat,
   sizing, costs and reward/equity reconciliation. 56 focused CPU tests passed.
2. Saved grouped checkpoints 25k/50k/75k/100k on TRAIN and INNER_A: authenticate
   files, cache raw actions and compare deterministic/stochastic distributions.
   Replay target deadbands 0.15/0.25/1/3/0.45 on the same support. The original
   threshold must reproduce native numerical metrics exactly. No selection.
3. Fresh TARGET control followed by COMMANDS5, both grouped, seed7, 100000
   decisions, identical recipe and optimizer. TARGET final policy hash must
   match the parent before the dependent fit starts. Checkpoints/reports at
   25k/50k/75k/100k. No early stopping, refit, expert, feedback or TEST.
4. Automatic final comparison, warehouse report readback and second-machine
   artifact copy. Reboot timer, resource admission, exclusive queue lock and
   bounded fresh-attempt recovery. Integrity failures do not silently modify
   settings or get interpreted as scientific outcomes.

TARGET's middle scalar means flat, not HOLD. COMMANDS5 explicitly distinguishes
HOLD/open-long/open-short/close-long/close-short; an inapplicable command is a
recorded HOLD. Existing protections always execute. Reversal is two decisions
in COMMANDS5, not close-and-open in one decision. This is an action-package
contrast, not an isolated threshold test. The exact bin edges are in the agent
plan and tested in float32 and float64.

No agent-state channels are introduced. Both policies remain partially observed;
explicit commands alone do not make inventory or remaining capacity observable.
State-routed specialists would explicitly convey position and are a later
separate design. Learned order fraction is also deferred: this contrast keeps
fraction1.0 of rel_volume0.1 times balance as margin budget, leverage50, with
mandatory SL/TP. The existing engine's declared protection-distance adjustments
are reported; market observation values are never clipped.

## Deployment Observation

At 2026-10-25T02:36:31Z the primary GPU queue was in DIAGNOSING, checkpoint25000,
TRAIN; alpha passed all16 scenarios in the deployed runtime. Service RAM was
2117378048 bytes. Collector timer enabled and first copy/pump succeeded with
zero scientific reports expected yet. This confirms operation, not a new fit
or completed diagnostic. Generated status and receipts supersede this snapshot.

No changes to DOIN, the expert lane, the shared engine, parent results or data.
Next learning-parameter search remains after order-contract evidence, followed
by capacity/context and finally periodic refitting. Do not auto-launch a large
optimizer from this study or select on 2024.
