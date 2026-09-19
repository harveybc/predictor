# 13A — MOD-E0-DEV: numerical design sheet (RP2), sealed `96f447e6 (RP2) / 6173c5d1 (RP4 stage v2, max_updates 3000)…`

Source: `tools/df_mod_e0_design.py` → `docs/audits/evidence/d3_k5_20260917/RP2_MOD_E0_DESIGN.json`.
Question (master v3 §4): when temporal heterogeneity and lagged relations between variables vary,
what do profile grouping (H2) and sequence preservation until fusion (H3) contribute under
comparable information, training and resources? DEVELOPMENT mechanism pilot; no confirmation.

## Question, estimands, unit

* H2: `e(h) = MASE(profiles, h) − MASE(random, h)`; slope of e over h (least squares, equal weights).
  Negative favours profiles; a negative slope means the advantage grows with heterogeneity. What the
  pilot reports is a **descriptive difference between two grouping procedures** on three trajectories:
  a small e(h) is not evidence of equivalence nor of absence of an effect (RP12 reframing); the
  observed slope was positive (not favourable to H2) with an interval that excludes zero only barely.
* H3: `d_r = MASE(sequence, r) − MASE(summary, r)`; `γ = d_1 − d_0`. Negative d favours sequence fusion.
  It is a contrast between two **complete procedures** (sequence → Conv1D core → last position vs global
  average → Dense core), sharing one frozen extractor that was pre-trained with the sequence receiver:
  readout, optimisation and pre-training compatibility differ with the fusion, so the advantage belongs to
  the executed procedure and is not attributable exclusively to "preserving the sequence until fusion".
  r = 0 removes the planted lagged link, not every cross-predictability (common periodic components).
  The ARCH-A/B/C/0 successor adds the readout control that separates pooling from history preservation.
* Unit: the replicate (independent generator trajectory). Random assignments are averaged within
  the replicate; optimiser seeds, windows and origins never count as units. Precision (replicate SD,
  bootstrap over replicates) is DESCRIPTIVE with three trajectories: 2 000 resamples of three units are
  not 2 000 replicates, and the SD itself has a 95 % interval spanning a factor of about 4
  (`RP12_PRECISION_DESIGN.json`). Confirmatory sizes are not proposed; the precision design gives, per
  number of replicates and for SD × 0.5 / 1 / 2, the interval half-widths and the effect detectable at
  80 % power, to be read against a margin fixed later in E0-CONF.

## Generator (ML01) — every number with its derivation

| item | value | why | alternative / sensitivity |
|---|---|---|---|
| variables p | 8 in two latent groups (4 + 4) | smallest population where grouping is non-trivial and both branches have equal size | 12 (3 groups) in E1 |
| sampling | unit step, shared | H2 forbids changing sampling or volume across levels | — |
| period P_A | 24 samples | P_A = 24 samples gives 86 cycles in training and 2 cycles in the context; small enough for many cycles, large enough for the branch receptive field (65) to cover the context | [12, 48] |
| heterogeneity h | [0, 1, 2, 3]; φ_A = 0.5 + 0.12h, φ_B = 0.5 − 0.12h, P_B = P_A(1 + 0.5h) | h scales BOTH persistence gap (0.24 h) and period gap (50 % h) so that level 0 is a homogeneous population (grouping arbitrary by construction) and level 3 separates groups by profile alone; amplitude, sampling, p, N and noise are constant across h (H2 rule) | levels are the H2 x-axis; level 0 is the internal null |
| amplitude / noise | a = 1.0; σ_AR = 0.5; σ_n = 0.3 | constant across levels (H2 rule); noise SD below the periodic amplitude so profiles are identifiable | σ_n = 0.6 (sensitivity, E1) |
| cross-dependence | β = 0.8, τ = 3 | tau = 3 < window and > horizon: the dependence is causal and inside the context; beta = 0.8 gives the oracle a visible gain over the marginal oracle (smoke: cross-correlation 0.68 at lag tau under r = 1 vs 0.00 under r = 0, marginal variance equal within 4 %) | τ = 6 |
| r = 0 control | phantom partner: an independent trajectory with group A's parameters feeds the same β-lagged term | the lagged partner term is fed by an independent PHANTOM partner generated with group A's parameters (own AR, periodic and noise draws): marginals of B preserved in distribution (ML04 checks variance and ACF across replicates, cross-correlation 0.68 vs 0.00); no temporal shuffle | — |
| oracle | generator conditional expectation with all latent states (AR state, phantom); reported as an unattainable floor; the observer-attainable bar is a ridge on the same 48×8 window | ML07 needs an attainable reference; the linear bar is fitted train-only per cell | — |

## Support, volume, geometry (ML02)

| item | value | why |
|---|---|---|
| context W | 48 (= 2 cycles of P_A; 2.0 cycles A; at h=3 P_B = 24·(1 + 0.5·3) = **60**, so W/P_B = **0.80** cycles, or (W−1)/P_B = **0.783** between the first and last observation of the window) | covers τ and the branch receptive field; alternatives 24 / 96 declared, not run (RP12 erratum: an earlier version wrote P_B = 72 and 0.67) |
| branch receptive field | 65 samples (kernel 3; dilations 1,1,2,4,8,16) | must reach the whole context (ML07 diagnostic) |
| horizon | 1, direct, all variables | multi-step is E1 work |
| N per trajectory | 3000: train rows [47, 2102) = 2055 windows (33 mini-batches of 64 per epoch), validation [2151, 2551) = 400, test [2600, 2999) = 399; purge W + h = 49 | purge derived from the consumed support; windows overlap (not replicates); the update allowance is enforced inside an epoch (RP11) |
| replicates | [1, 2, 3] independent trajectories per condition; 3 predefined random assignments per H2 condition | independence from generator seeds only |
| MASE denominator | train MAE of the seasonal naive with the variable's declared period; shared by every arm; zero → NO_APLICA before scoring | contract of metrics v1 |

## Learners (ML05/ML06), the same information in every arm

Per branch: detector = 2 residual causal TCN blocks (16 filters, kernel 3) → integrator = residual
dilated blocks (2, 4, 8, 16) → linear adapter TimeDistributed Dense(8); the temporal axis is kept
(no Flatten; `build_branch` of the legacy plugins is NOT reused). Fusion: sequence = channel
concatenation of aligned sequences → Conv1D(16,3) core → last position → Dense(p); summary = global
average per branch → concatenation → Dense(32) → Dense(p). Both heads predict the increment over the
last observation (persistence skip, identical in every arm). Trainable parameters: H2 full model
9608; H3 sequence 920 vs summary
808 (15 % (declared in this pilot)). H3: the extractor is trained once per replicate
(profile assignment, sequence fusion, R0), saved, and **loaded frozen** into both arms; only fusion,
core and head train (weights verified unchanged). Grouping: ACF at lags 1–24, Welch bands, trend and
seasonal strength on training rows, scaled by development SD, constants excluded, average linkage.

Training rule (sealed after the ML07 diagnostic, `RP3_ML07_RECEIVER_DIAGNOSTIC.json`): ELU, Adam 3e-3,
batch 64, ≤ 6 000 updates (stage v2: ≤ 3 000, i.e. 90 epochs × 33 = 2 970 executed at most), early
stopping on validation **mse** (patience 30, best restored and verified by re-evaluation), mse as the
optimisation loss, MAE/MSE/RMSE/MASE as the metrics; the validation curves plotted are the mse loss on
the scaled targets (RP12 erratum: an earlier figure label said MAE); no tuning after outcomes.

Profiles (RP12): ACF at lags 1–24, Welch bands, trend and seasonal strength. The strengths are
Hyndman's F_T = 1 − Var(R)/Var(T+R) and F_S = 1 − Var(R)/Var(S+R) on a **centred moving-average
decomposition** of the declared period (descriptor v2; the executed pilot used v1, whose F_T divided by
Var(X) — corrected after the review; the reanalysis without training shows the 15 ordered partitions,
kept masks, sizes, redistributions, inputs and donors unchanged: `RP12_PROFILE_REANALYSIS.json`).
The decomposition is a train-batch characterisation computed once the training block is closed, not
a causal online operator; nothing in it is used at prediction time.

## Controls and what the model never sees

Positive learning (ML07): at h = 3, r = 1 the receiver must beat the naive forecaster and stay
within 0.03 MASE of the linear window bar on validation. Grouping positive: ARI = 1 vs the latent
groups at h ≥ 2 and not at h = 0. Nulls: level 0 (H2), r = 0 (H3). Adverse: a temporal shuffle is
rejected as a control. Never inputs: latent labels, generator parameters, phases, oracle, future.

## Budget

14400 s CPU aggregate (pilots + cells + failures + verification), 25 % headroom on the
projection before dispatch; prior adequacy pilots' 76 s recorded separately.
Cells: 66 (H2: 4 levels × 3 replicates × (profiles + 3 random) = 48; H3: 2 r × 3 replicates × (extractor + 2 arms) = 18).
