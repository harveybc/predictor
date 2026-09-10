# STEP 11 — Controlled Redundancy / Corruption-Hardening of the Existing Feature Extractor

**Status:** Research protocol — final draft for independent agent audit  
**Date:** 2026-09-05  
**Normative dependency:** `WORKPLAN_PATCH_002_AGENT_AUDIT_CONSTRAINTS.md`, especially §6. Where this document and PATCH 002 conflict, PATCH 002 governs.  
**Code owner if later implemented:** `harveybc/feature-extractor` only.  
**Explicit non-scope:** STEP 10 synchronization is not modified here.

---

# 0. One-line definition

STEP 11 asks:

> **Does deliberate redundancy through train-only masking/corruption make the latent representation produced by the existing feature extractor more robust and more useful out of sample, without changing its latent interface or building another autoencoder?**

The experiment is deliberately small.

The valid conclusion set is:

\[
\boxed{
\{\text{helps},\text{ties},\text{hurts}\}
}
\]

A null result closes STEP 11 successfully.

The protocol does **not** assume that masked/denoising pretraining should become a permanent default.

---

# 1. Communications analogy

Channel coding deliberately introduces structured redundancy so information can survive corruption.

The ML analogue considered here is much narrower:

\[
\boxed{
\text{train the existing reconstruction model under controlled partial corruption}
}
\]

so that its encoder must learn a latent representation that is less dependent on any single input value, timestamp or channel.

Let:

\[
X
\]

be a clean training window.

A corruption operator:

\[
C_{\eta}
\]

produces:

\[
\widetilde X
=
C_{\eta}(X),
\]

where:

\[
\eta
\]

controls corruption severity.

The existing autoencoder is trained as:

\[
\boxed{
\widetilde X
\rightarrow
Encoder
\rightarrow
Z
\rightarrow
Decoder
\rightarrow
\widehat X
}
\]

with the **clean original** as the reconstruction target:

\[
\boxed{
\mathcal L_{AE}
=
d(
X,
\widehat X
).
}
\]

The corruption is therefore extra information pressure during training, not a replacement target.

---

# 2. This is not a new autoencoder

The existing `feature-extractor` repository already owns the reconstruction stage.

It currently:

- trains Keras/TensorFlow encoder/decoder pairs;
- consumes sliding windows of preprocessed CSV data;
- evaluates reconstruction quality;
- saves trained encoder and decoder artifacts;
- exports the encoder for downstream `predictor` phases.

Its working registered encoder/decoder families include:

- ANN;
- CNN;
- LSTM;
- Transformer;
- VAE;
- VAE-small.

Therefore STEP 11 does **not** authorize:

- another AE repository;
- another generic DAE architecture;
- a new Transformer “for channel coding”;
- a new downstream predictor architecture.

The experimental variable is the **training corruption policy applied to the current extractor**.

---

# 3. Existing repository contract

Current logical pipeline:

```text
preprocessed CSV
    ↓
existing feature-extractor
    ↓
existing encoder plugin
    ↓
existing latent interface
    ↓
existing decoder plugin
    ↓
reconstruction
```

STEP 11 modifies only the training input:

```text
clean training window X
    ↓
train-only corruption Cη
    ↓
X_tilde
    ↓
SAME existing encoder
    ↓
SAME latent dimensionality/interface Z
    ↓
SAME existing decoder
    ↓
reconstruct CLEAN X
```

At ordinary validation/test inference:

```text
clean X
    ↓
encoder
    ↓
Z
```

No ordinary OOS corruption is applied.

---

# 4. Architecture/interface invariants

The following are frozen between baseline and corruption arms:

- encoder plugin family;
- decoder plugin family;
- latent dimension;
- window size;
- input feature set;
- preprocessing;
- normalization;
- reconstruction target;
- loss definition, unless an existing plugin requires its native loss;
- optimizer family;
- early-stopping policy;
- downstream latent interface;
- downstream simple core architecture;
- downstream target/horizon;
- random-seed protocol.

The corruption arm is not allowed to win by increasing:

\[
\dim Z.
\]

Gate 11C explicitly enforces this.

---

# 5. Distinction from STEP 03

STEP 03 and STEP 11 both use perturbations, but their scientific questions are different.

| Property | STEP 03 | STEP 11 |
|---|---|---|
| Perturbation purpose | Measure degradation / evaluate denoising | Training regularizer |
| Location | Preprocessing/input evaluation | Existing extractor training |
| Primary question | Can noise be estimated/removed? | Does corruption-hardened representation generalize better? |
| OOS primary evaluation | Raw vs denoised | Clean latent vs robust-trained latent |
| Null result | Denoising does not help | Corruption training does not help |

Thus:

\[
\boxed{
\text{training corruption in STEP 11}
\neq
\text{noise-estimation experiment in STEP 03}.
}
\]

---

# 6. Distinction from compression lane C6

STEP 11 does not optimize:

\[
\mathcal L_{task}
+
\lambda R_Z.
\]

It does not estimate latent bit rate.

It does not establish a MacKay neuron-capacity limit.

It does not estimate Kolmogorov complexity.

Latent rate–distortion remains C6 and is a later question.

---

# 7. Distinction from ordinary channel/error-correcting codes

STEP 11 does not insert:

- Hamming codes;
- BCH;
- Reed–Solomon;
- LDPC;
- Turbo codes

into CSV features.

The communications analogy is functional:

> deliberately train with missing/corrupted information so the representation becomes less brittle.

No symbolic parity bits are introduced into the financial dataset.

---

# 8. State of the art — denoising autoencoders

Vincent et al. introduced the denoising autoencoder training principle:

1. corrupt the input;
2. reconstruct the uncorrupted input;
3. learn representations robust to partial corruption.

This is the direct classical prior art for STEP 11.

The important lesson is not a particular network architecture.

It is:

\[
\boxed{
\widetilde X
\rightarrow
Z
\rightarrow
X.
}
\]

The current project already has the reconstruction architecture, so only this training principle needs to be tested.

---

# 9. State of the art — masked autoencoding as a general reconstruction principle

Masked Autoencoders (MAE) popularized high-ratio masked reconstruction in vision.

For STEP 11, MAE is used only as evidence for the general principle:

\[
\boxed{
\text{partial observation}
\rightarrow
\text{reconstruct missing information}
}
\]

and **not** as a recommendation to replace the existing extractor with a Vision Transformer or MAE architecture.

The project does not need a ViT.

---

# 10. Time-series-specific warning: masking is not neutral

Time-series data are not images.

Temporal differences, local changes and ordering carry meaning.

SimMTM explicitly argues that naive random point masking can destroy important temporal variations and make the reconstruction pretext task poorly matched to time-series semantics.

Therefore STEP 11 cannot assume:

\[
\text{random mask}
=
\text{best mask}.
\]

The initial experiment must compare at least:

- dispersed point/cell masking;
- contiguous temporal-span masking.

---

# 11. SimMTM prior art

SimMTM (NeurIPS 2023) constructs multiple masked versions of each time series and uses representation similarity/local-manifold structure to reconstruct masked points.

Its relevance to STEP 11 is twofold:

1. masked time-series reconstruction has strong published prior art;
2. naive masking can be too destructive, motivating complementary corrupted views.

However:

> SimMTM-style multi-view consistency is **deferred**.

It is considered only if the small existing-AE masking experiment reaches Gate 11B with a promising but unstable/ambiguous result.

STEP 11 does not begin by reimplementing SimMTM.

---

# 12. Ti-MAE prior art

Ti-MAE is another time-series masked-autoencoding approach that reconstructs masked time-series observations to learn downstream-useful representations.

It supports the scientific plausibility of:

\[
\boxed{
\text{masked reconstruction}
\rightarrow
\text{useful time-series latent}
}
\]

but it does not justify changing the current project architecture.

It is a methodological reference, not an implementation requirement.

---

# 13. Recent time-series MAE work

Recent work continues to explore masked autoencoding for time-series representation learning, including TS-MAE (Information Sciences, 2025).

This confirms that masked reconstruction remains an active representation-learning direction.

Nevertheless the project's question is intentionally more conservative:

> Does adding corruption to **our existing feature extractor** help our downstream \(P\)?

That question can be answered without adopting a new state-of-the-art encoder.

---

# 14. Channel masking prior art

Time-series/sensor literature has shown that masks can operate not only in time but across channels/features.

This motivates a limited channel-mask diagnostic for multivariate project inputs.

It does **not** imply that feature channels are interchangeable.

Whole-channel masking is meaningful only when enough other information exists to reconstruct the missing channel.

---

# 15. Multi-view / consistency prior art is secondary

Methods such as SimMTM and later time-series self-supervised frameworks use multiple corrupted views, contrastive/siamese consistency or neighborhood reconstruction.

These are **second-line** STEP-11 options.

They are opened only if simple corruption demonstrates value.

There is no justification for adding consistency machinery if:

\[
P_{corrupt}
\leq
P_{baseline}.
\]

---

# 16. Project code audit — existing extractor is sufficient

The current repository describes itself as the active representation-learning stage between preprocessing and predictor training.

It already exposes pluginized encoder/decoder families and saves encoder artifacts for downstream use.

Its example configurations include `phase_3_2_daily` with ANN, CNN, LSTM and Transformer encoder/decoder pairs.

A current daily CNN configuration uses:

- `normalized_d1.csv` for training;
- `normalized_d2.csv` for validation;
- `normalized_d3.csv` for test;
- CNN encoder;
- CNN decoder;
- explicit saved encoder/decoder artifacts.

This provides a ready small project-owned bank for Gate 11A.

---

# 17. Repository-quality warning

The repository README currently notes that its full test suite has collection errors, although plugin imports and CLI smoke tests have been verified.

Therefore Gate 11A is essential:

> no corruption result is interpreted until the baseline retrain is reproducible enough for the experiment.

STEP 11 does not require a full repository refactor before research, but all experiment-specific tests must pass.

---

# 18. Primary experimental chain

The normative experiment is:

```text
frozen preprocessed CSV
    ↓
existing feature extractor
    ↓
latent Z, fixed dimension
    ↓
simple fixed downstream core
    ↓
OOS predictive metric P
```

The core is intentionally weak/simple.

Eligible examples:

- deterministic small ANN already available in the project;
- DLinear-like fixed linear baseline if already available in the work environment.

Not eligible in STEP 11:

- PatchTST;
- new Transformers;
- foundation models;
- `agent-multi` RL;
- large architecture campaigns.

---

# 19. Meaning of “frozen simple core”

“Frozen” means the **architecture and training protocol are fixed across all latent arms**.

The downstream core still has to be fitted on training data for each latent representation.

It must not receive additional depth/width because one encoder arm is corrupted.

The comparison tests representation quality, not model-search skill.

---

# 20. Arm 0 — current AE baseline

Train the selected existing AE exactly under its current clean reconstruction regime:

\[
X
\rightarrow
Encoder_0
\rightarrow
Z_0
\rightarrow
Decoder_0
\rightarrow
\widehat X.
\]

Reconstruction target:

\[
X.
\]

No extra corruption.

This arm must first reproduce known behavior within preregistered tolerance.

---

# 21. Arm 1 — same AE with train-only corruption

Training:

\[
\widetilde X
=
C_\eta(X).
\]

Then:

\[
\widetilde X
\rightarrow
Encoder_\eta
\rightarrow
Z_\eta
\rightarrow
Decoder_\eta
\rightarrow
\widehat X.
\]

Target remains:

\[
X.
\]

Validation/test primary inference uses:

\[
X
\]

clean.

---

# 22. Arm 2 — optional clean + robust latent without widening interface

If Gate 11B passes, compare a parallel representation.

Both encoders return:

\[
Z_0,Z_\eta\in\mathbb R^d.
\]

Do **not** concatenate to:

\[
2d.
\]

Initial fixed-dimension fusion:

\[
\boxed{
Z_{blend}
=
\frac12
(
Z_0+Z_\eta
)
}
\]

or another predeclared parameter-free dimension-preserving combination.

A trainable fusion layer is outside the first experiment because it introduces extra capacity.

---

# 23. Why the clean+robust arm is optional

If:

\[
P(Z_\eta)
\leq
P(Z_0)
\]

with no robustness benefit, there is no reason to test a two-encoder deployment.

STEP 11 may close after Arm 1.

---

# 24. Primary corruption family A — dispersed masking

For input tensor:

\[
X\in\mathbb R^{T\times F},
\]

draw mask:

\[
M_{t,f}
\sim
Bernoulli(1-p).
\]

Masked input:

\[
\widetilde X
=
M\odot X
+
(1-M)\odot v_{mask}.
\]

Preferred first baseline:

\[
v_{mask}=0
\]

**only because the project's normalized value space generally makes zero a meaningful neutral reference.**

The implementation must verify this assumption for the selected data.

---

# 25. Mask indicator question

If missingness could be confused with a legitimate zero, a mask indicator can be supplied.

But adding:

\[
M
\]

as extra input channels changes the interface.

Therefore the first STEP-11 experiment should **not** add mask-indicator channels unless the selected normalization makes zero masking invalid.

Any mask-indicator experiment is a separate capacity/interface ablation.

---

# 26. Primary corruption family B — contiguous temporal-span masking

Choose contiguous intervals:

\[
[t_0,t_0+\ell-1]
\]

and mask them.

Total masked fraction approximately:

\[
p.
\]

This tests whether the encoder can reconstruct temporal structure when local segments disappear.

It is particularly important because time-series semantics often reside in local temporal variation.

---

# 27. Secondary diagnostic — feature/channel masking

For feature index:

\[
f,
\]

mask complete feature trajectories inside the window:

\[
X_{:,f}.
\]

This asks:

> can the representation reconstruct one feature from temporal/cross-feature context?

Use only for genuinely multivariate input.

Do not mask:

- labels;
- target columns;
- metadata required for sample validity.

---

# 28. Secondary diagnostic — additive train corruption

Optional single diagnostic arm:

\[
\widetilde X
=
X+\epsilon.
\]

Where:

\[
\epsilon
\]

is generated only during training.

This is not STEP 03 because the objective here is:

\[
\text{representation hardening}
\]

rather than measuring/removing a real noise component.

To prevent duplication, use a single preregistered training SNR derived from the STEP-03 scale, not a new exhaustive noise campaign.

---

# 29. Do not corrupt reconstruction target

Always:

\[
\text{input}
=
\widetilde X
\]

and:

\[
\boxed{
\text{target}
=
X.
}
\]

Training:

\[
(\widetilde X,X).
\]

Never:

\[
(\widetilde X,\widetilde X).
\]

Otherwise the model is trained to reproduce the corruption itself.

---

# 30. No target masking

Downstream predictive target:

\[
Y
\]

is never corrupted or masked.

STEP 11 acts only on autoencoder input windows.

---

# 31. Causality rule for temporal masks

At forecast endpoint:

\[
t,
\]

the original training window contains:

\[
X_{\le t}.
\]

Corruption may remove information from that historical window.

It may never fill a masked value using:

\[
X_{>t}.
\]

Reconstruction target can be the clean historical value because it is known inside the **training example**.

No future beyond the window endpoint is used.

---

# 32. Primary OOS evaluation is clean

For validation/test primary curve:

\[
X_{clean}
\rightarrow
Encoder
\rightarrow
Z
\rightarrow
Core.
\]

No random corruption.

This measures whether robustness training improved ordinary generalization.

---

# 33. Stress OOS evaluation is separate

After corruption configuration is frozen, a separate stress suite can evaluate:

\[
C_\rho(X_{val/test})
\]

for multiple stress levels.

Stress results are labeled explicitly:

\[
\boxed{
\text{OOS stress}
}
\]

and are never mixed with the clean OOS performance curve.

Stress performance does not replace clean performance.

---

# 34. Preregistered minimal corruption grid

To avoid mask-rate fishing, preregister a small grid.

Recommended initial rates:

\[
\boxed{
p\in\{0.05,\;0.15,\;0.30\}
}
\]

for:

- dispersed masking;
- temporal-span masking.

This produces:

\[
6
\]

corruption configurations plus the clean baseline.

Optional diagnostics, not part of the primary grid:

- feature/channel mask at:
  \[
  p=0.15;
  \]
- one additive-noise training arm at a fixed preregistered SNR.

No additional rate is introduced after looking at validation unless STEP 11 is explicitly reopened as a new experiment.

---

# 35. Mask-rate interpretation

The grid intentionally covers:

- low corruption:
  \[
  5\%;
  \]
- moderate:
  \[
  15\%;
  \]
- relatively strong:
  \[
  30\%.
  \]

It deliberately does **not** import the 75% image-MAE masking ratio.

Image masking ratios are not a justified prior for financial time-series windows.

---

# 36. Span-length definition

For temporal masking, choose span length from:

\[
\ell
=
\max(
1,
\lfloor pT\rfloor
)
\]

for one contiguous span in the minimal experiment.

Do not additionally optimize:

- number of spans;
- span-length distribution;
- patch size

in the first pass.

That would turn one hypothesis into a masking-search campaign.

---

# 37. Corruption RNG

Every run stores:

- corruption seed;
- training seed;
- exact mask type;
- exact mask rate.

For each training sample/epoch, masks may be regenerated deterministically from the run seed.

The clean baseline must use the same model seed schedule.

---

# 38. Primary benchmark choice

STEP 11 does **not** open a broad benchmark campaign.

Choose one small bank.

Preferred project-owned option:

\[
\boxed{
\texttt{feature-extractor/examples/config/phase_3_2_daily}
}
\]

because it already uses:

- training data `normalized_d1.csv`;
- validation `normalized_d2.csv`;
- test `normalized_d3.csv`;
- registered AE plugins;
- saved encoder artifacts.

Optional external sanity dataset:

- one ETT dataset only.

No Traffic.

No PEMS.

No PatchTST.

No `agent-multi`.

---

# 39. Architecture choice for Gate 11A

Do not sweep all existing encoders.

Choose one existing stable architecture before results are viewed.

Recommended first candidate:

\[
\boxed{
\text{current CNN encoder/decoder pair}
}
\]

because:

- it exists and is registered;
- a `phase_3_2_daily` config exists;
- it is computationally simpler than a new large model.

If agents identify another existing extractor champion with better reproducibility, they may choose it **before** starting STEP 11.

The architecture is then frozen.

---

# 40. Existing daily configuration as reproducibility anchor

The current `phase_3_2_daily` CNN config references:

- `normalized_d1.csv` as train;
- `normalized_d2.csv` as validation;
- `normalized_d3.csv` as test;
- CNN encoder;
- CNN decoder;
- saved encoder and decoder outputs.

Its existing parameters should be used as the starting baseline rather than constructing a new “STEP-11 model.”

---

# 41. Downstream core

Use one low-capacity deterministic downstream model.

Preferred order:

1. existing small deterministic ANN daily predictor if convenient;
2. otherwise a fixed linear/DLinear-like head.

The architecture/hyperparameters are preregistered.

There is no downstream model search in STEP 11.

---

# 42. Reconstruction metric is diagnostic, not the final objective

Measure:

\[
L_{recon}
=
d(
X,\widehat X
).
\]

But success is not:

\[
L_{recon}\downarrow
\]

alone.

Primary success is downstream:

\[
\boxed{
P_{OOS}
}
\]

using the frozen simple core.

A robust representation can tolerate slightly worse reconstruction while improving downstream generalization.

---

# 43. Latent robustness metric

For clean window \(X\) and stressed copy:

\[
\widetilde X=C_\rho(X),
\]

compute:

\[
Z=E(X)
\]

and:

\[
\widetilde Z=E(\widetilde X).
\]

Candidate normalized drift:

\[
\boxed{
D_Z
=
\frac{
E\|Z-\widetilde Z\|_2^2
}{
E\|Z\|_2^2+\epsilon
}.
}
\]

Smaller:

\[
D_Z
\]

means greater local latent stability.

This is diagnostic, not sufficient for success.

---

# 44. Latent cosine stability

Also report:

\[
S_{cos}
=
E
\left[
\frac{
Z^T\widetilde Z
}{
\|Z\|\|\widetilde Z\|
}
\right].
\]

This complements Euclidean drift.

Do not optimize corruption solely for latent invariance; a completely collapsed latent is perfectly invariant and useless.

---

# 45. Collapse diagnostics

Report:

- latent variance per dimension;
- effective rank;
- pairwise latent distance statistics;
- downstream \(P\).

Reject representation collapse even if:

\[
D_Z
\]

is excellent.

---

# 46. Clean reconstruction diagnostics

On clean validation/test:

\[
X
\rightarrow
E_\eta
\rightarrow
D_\eta
\rightarrow
\hat X.
\]

Report whether corruption training causes unacceptable clean-reconstruction degradation.

This is a secondary guardrail.

---

# 47. OOS corruption-stress curve

After final corruption arm selection:

\[
\rho
\in
\{0,\rho_1,\rho_2,\ldots\}.
\]

Measure:

\[
P_{stress}(\rho)
\]

for both:

- baseline encoder;
- robust-trained encoder.

The stress operator must match a clearly labeled corruption family.

Primary purpose:

> verify that “robust” actually means less brittle under missing/corrupted inputs.

---

# 48. STEP-03 relationship for stress noise

If stress uses additive noise, reuse STEP-03 noise definitions/SNR conventions.

Do not create a second noise ontology.

STEP 11 asks whether the robust-trained encoder's degradation curve is flatter:

\[
\left|
\frac{\partial P}{\partial noise}
\right|_{\text{robust}}
<
\left|
\frac{\partial P}{\partial noise}
\right|_{\text{baseline}}.
\]

This remains a stress diagnostic.

---

# 49. Tail/event preservation — Gate 11D

Robustness cannot be purchased by smoothing away rare important events.

For project financial data, define tail slices using thresholds estimated on training only.

Examples:

\[
|\Delta X|
>
Q_{0.95}^{train}
\]

or high-volatility windows:

\[
V
>
Q_{0.95}^{train}(V).
\]

Report downstream \(P\) separately on:

- ordinary windows;
- tail/event windows.

If the robust encoder materially harms tails/events, that corruption configuration is rejected.

---

# 50. Tail definition must be frozen

Validation/test quantiles are not used to redefine “tail.”

Thresholds come from training.

This avoids moving the goalposts based on OOS data.

---

# 51. Primary metric

Let:

\[
P
\]

denote the preregistered downstream metric for the selected small task.

Examples:

- MAE;
- RMSE;
- \(R^2\);
- one binary metric if the selected task is classification.

Only one primary metric is used for gate decisions.

Secondary metrics remain diagnostic.

---

# 52. Seed protocol

At minimum use multiple deterministic seeds sufficient to distinguish a stable effect from optimizer noise.

Recommended small protocol:

\[
\boxed{
3\text{ seeds}
}
\]

for:

- baseline;
- each primary corruption arm.

No large hyperparameter campaign.

---

# 53. Clean OOS comparison

For corruption configuration \(\eta\):

\[
\Delta P_{\eta}
=
P_{\eta}^{clean\ OOS}
-
P_{0}^{clean\ OOS}
\]

with sign interpreted according to the metric.

Report:

- mean;
- median;
- seed dispersion;
- block bootstrap/temporal uncertainty where practical.

---

# 54. Equivalence / tie region

Before inspecting results, define a practical equivalence margin:

\[
\epsilon_P.
\]

Then classify:

## Helps

\[
\Delta P>\epsilon_P.
\]

## Ties

\[
|\Delta P|\leq\epsilon_P.
\]

## Hurts

\[
\Delta P<-\epsilon_P.
\]

The exact \(\epsilon_P\) must be determined from baseline variance and task scale before final validation interpretation.

---

# 55. Gate 11A — baseline reproducibility

**Requirement:** the current AE can be retrained on the chosen small bank and reproduce its known qualitative/numerical behavior within preregistered tolerance.

Must verify:

- training completes;
- early stopping behaves;
- latent artifact shape is correct;
- encoder save/load works;
- downstream simple core receives the expected latent;
- baseline seed variance is characterized.

**If Gate 11A fails:**

\[
\boxed{
\text{STOP STEP 11 interpretation.}
}
\]

No masking result is trusted.

---

# 56. Gate 11B — stable corruption effect

At least one train-only corruption configuration must move downstream clean OOS \(P\) stably relative to baseline.

Possible results:

- stable improvement;
- stable degradation;
- stable practical tie.

A stable null/tie is still a scientifically valid STEP-11 conclusion.

**If all effects are unstable/noisy:**

close STEP 11 as inconclusive/unsupported and keep current AE.

Do not open a larger GPU search automatically.

---

# 57. Gate 11C — fixed latent interface

The latent dimension remains:

\[
d
\]

for every arm.

No widening.

No hidden extra channels.

No doubled clean+robust concatenation.

If an arm requires larger representation capacity, its gain is not evidence for deliberate redundancy at equal interface.

---

# 58. Gate 11D — tail/event safety

A corruption arm is rejected if tail/event performance materially collapses beyond the preregistered tolerance.

This gate overrides a small aggregate improvement.

---

# 59. The three hypotheses — maximum allowed

## H11.1 — Controlled train-only corruption can improve the existing latent

For at least one preregistered corruption configuration:

\[
\boxed{
P(E_{\eta}(X))
>
P(E_0(X))
}
\]

on clean OOS data using the same latent dimension and the same simple downstream core.

**Falsified if:** no corruption arm produces stable improvement.

A tie or degradation is a valid conclusion.

---

## H11.2 — The benefit is not universal

There exists a corruption regime:

\[
\eta
\]

or mask rate:

\[
p
\]

where downstream performance degrades.

Expected shape may be:

\[
\text{low/moderate corruption}
\rightarrow
\text{possible regularization benefit}
\]

and:

\[
\text{excessive corruption}
\rightarrow
\text{information destruction}.
\]

**Falsified if:** every tested stronger corruption is uniformly equal/better — in which case the tested range was probably insufficient and no universal claim is still allowed.

---

## H11.3 — STEP-11 robust latent is conceptually distinct from prior residuals

The representation:

\[
Z_{\eta}
\]

learned from corruption-hardened reconstruction is not identified with:

- STEP-03 denoising residual;
- STEP-05 innovation;
- STEP-09 private component \(U\).

This is a boundary hypothesis.

It is considered violated if implementation/documentation treats these as numerically interchangeable objects without evidence.

---

# 60. Explicitly forbidden thesis

The following statement is **not** an allowed STEP-11 conclusion:

> “Masked pretraining always improves time-series representation.”

The strongest allowed positive conclusion is local:

> For the selected existing extractor, dataset/task and preregistered corruption regime, controlled train-only corruption produced a reproducible OOS benefit under the frozen interface and safety gates.

---

# 61. Development-selection rule

The primary corruption grid is preregistered once.

Validation may select among those configurations.

Do not repeatedly add:

- new mask rates;
- new mask shapes;
- new noise levels

after observing validation results.

A second grid requires a formally new experiment version.

---

# 62. Validation corruption rule

Primary validation:

\[
\boxed{
\text{clean input}.
}
\]

No random validation corruption is used to select the main OOS winner.

Optional corruption stress on validation is reported separately and must not be averaged into primary validation \(P\).

---

# 63. Test rule

The selected corruption policy is frozen before test.

Primary test is clean.

Then, optionally, a predeclared stress suite may run on corrupted copies.

The test is not used to select:

- mask type;
- mask rate;
- architecture;
- fusion.

---

# 64. Early stopping rule

The current AE uses validation-driven early stopping in its training pipeline.

For STEP 11, early stopping remains based on **clean validation reconstruction** unless the existing architecture contract requires a documented alternative.

This ensures the corruption arm is still judged on its ability to model clean data.

---

# 65. No validation leakage through corruption statistics

Corruption operator parameters come from:

- preregistered constants; or
- training-only statistics.

Examples:

- channel scale for Gaussian noise:
  training statistics only;
- mask probabilities:
  fixed before validation.

---

# 66. CPU-first execution

Default environment:

```text
CUDA_VISIBLE_DEVICES=""
```

unless an explicitly authorized idle GPU is used for one bounded retraining run.

No campaigns.

No distributed optimization.

No RL.

No background agent sweep.

---

# 67. Code-location contract

If implementation is later approved, changes live only in:

```text
harveybc/feature-extractor
```

Preferred insertion point:

- the current autoencoder training-data path;
- or a small reusable corruption function called immediately before the existing model receives training input.

Do not modify:

- `preprocessor` for STEP 11;
- `predictor` architecture for STEP 11;
- `agent-multi`.

---

# 68. Configuration contract

Recommended new configuration keys:

```text
train_corruption_enabled
train_corruption_type
train_corruption_rate
train_corruption_seed
train_corruption_noise_snr_db
```

These are proposed names, not mandated API.

When disabled:

\[
\boxed{
\text{bit-for-bit/behavior-compatible clean training path}
}
\]

should be preserved as far as the existing pipeline permits.

---

# 69. Corruption implementation contract

Function concept:

\[
C(
X,
type,
rate,
seed
)
\rightarrow
\widetilde X.
\]

It must:

- never mutate the clean target;
- never use future values;
- be deterministic under seed;
- preserve tensor shape;
- preserve dtype;
- emit a corruption manifest.

---

# 70. Required unit tests if code is implemented

At minimum:

1. `enabled=False` returns exact clean input.
2. fixed seed returns identical corruption.
3. different seed changes mask.
4. masked fraction approximately matches requested rate.
5. target tensor remains unchanged.
6. output shape equals input shape.
7. temporal mask never reads outside input window.
8. validation/test clean path is unchanged.
9. saved encoder output dimension is unchanged.

---

# 71. Required integration smoke test

One tiny train run verifies:

```text
clean CSV
→ corrupted train input
→ same AE
→ save encoder
→ reload encoder
→ latent shape == baseline latent shape
→ simple downstream core accepts latent
```

This is part of Gate 11A/11C.

---

# 72. Existing architecture choice is not reopened

STEP 11 does not decide whether:

- CNN beats LSTM;
- Transformer beats CNN;
- VAE beats AE.

That is another research question.

Pick one reproducible existing extractor and hold it fixed.

---

# 73. Reconstruction-target scope

For an ordinary AE reconstructing the full window:

\[
Y_{AE}=X.
\]

For conditional VAE configurations with specialized reconstruction targets, STEP 11 must preserve that existing target contract exactly.

Corruption applies to encoder input, not target semantics.

---

# 74. Why VAE is not the first STEP-11 arm

VAE introduces stochastic latent regularization already.

This can confound the corruption effect.

Therefore a deterministic/non-VAE existing AE such as CNN is preferable for the first experiment.

VAE can be a later replication only if STEP 11 shows a stable effect.

---

# 75. Why no second DAE

Calling the corruption-trained current model a “denoising AE” in a theoretical sense is fine.

Creating a distinct DAE implementation is not.

The same plugin architecture remains authoritative.

---

# 76. Why no PatchTST

STEP 11 is about extractor robustness.

A stronger forecasting backbone can absorb or hide representation differences.

Therefore no PatchTST benchmark is required.

The intentionally weak downstream core makes the latent accountable.

---

# 77. Why no Traffic/PEMS

STEP 11 is not a general masked-pretraining benchmark paper.

The scientific question can be falsified on:

- a small external sanity dataset; or
- the existing project daily bank.

Large multichannel benchmark campaigns add cost without being necessary to decide whether the existing extractor benefits.

---

# 78. Why no `agent-multi`

RL/policy results combine:

- state representation;
- optimization;
- reward;
- exploration;
- replay;
- environment dynamics.

That makes them a poor primary assay for the narrow latent-robustness question.

STEP 11 closes before RL integration.

---

# 79. Stress families after winner selection

If a winner exists, optional robustness evaluation may use:

- same corruption type at larger rates;
- feature dropout;
- additive noise using STEP-03 SNR conventions;
- bounded missing-span stress.

These are evaluation-only.

Do not retrain additional models during stress analysis.

---

# 80. Robustness slope

For stress severity:

\[
\rho,
\]

define:

\[
P(\rho).
\]

A robustness diagnostic is:

\[
\boxed{
S_R
=
\frac{
P(\rho_2)-P(\rho_1)
}{
\rho_2-\rho_1
}.
}
\]

Compare baseline and robust-trained encoder.

The preferred robust representation has smaller adverse degradation magnitude.

---

# 81. Robustness does not override clean OOS utility

A model that is excellent under artificial corruption but worse under ordinary clean production data is not automatically preferred.

Primary decision order:

1. clean OOS utility;
2. tail/event safety;
3. stress robustness;
4. reconstruction/latent diagnostics.

---

# 82. Minimal experimental matrix

| ID | AE architecture | Train input | Reconstruction target | OOS primary | Latent dim |
|---|---|---|---|---|---:|
| R00 | existing fixed AE | clean | clean | clean | \(d\) |
| R01 | same | point mask 5% | clean | clean | \(d\) |
| R02 | same | point mask 15% | clean | clean | \(d\) |
| R03 | same | point mask 30% | clean | clean | \(d\) |
| R04 | same | span mask 5% | clean | clean | \(d\) |
| R05 | same | span mask 15% | clean | clean | \(d\) |
| R06 | same | span mask 30% | clean | clean | \(d\) |
| R07 | same | channel mask 15% | clean | clean | \(d\) |
| R08 | same | one noise diagnostic | clean | clean | \(d\) |
| R09 | clean + selected robust | fixed-dim blend | — | clean | \(d\) |

R07–R09 are optional second-stage diagnostics.

R00–R06 form the primary small experiment.

---

# 83. Stop conditions

Immediately stop expansion if:

- Gate 11A fails;
- latent interface changes;
- corruption path leaks future information;
- primary corruption arms all materially hurt \(P\);
- tail/event gate fails for every candidate.

A failed STEP is not repaired by adding a larger architecture.

---

# 84. Decision table

| Observation | Verdict |
|---|---|
| Stable clean OOS gain + tails safe | Helps |
| Clean OOS practical tie + stress robustness improved | Ties / optional robustness value |
| Clean OOS tie + no stress advantage | No value; retain baseline |
| Clean OOS degradation | Hurts |
| Aggregate gain but tail/event collapse | Reject corruption |
| Unstable across seeds | Unsupported |
| Gain requires wider latent | Capacity confound; fail 11C |

---

# 85. Multiple comparisons

The small preregistered grid still creates multiple comparisons.

Declare:

- one primary corruption family comparison;
- one primary downstream metric;
- one practical equivalence margin.

Report all grid results.

Do not publish only the best mask rate.

---

# 86. Statistical uncertainty

Time-series OOS errors are dependent.

Use:

- temporal/block bootstrap where applicable;
- seed variation;
- per-period breakdown.

No iid timestamp bootstrap.

The experiment is small enough that transparent full reporting is preferable to elaborate significance mining.

---

# 87. Latent dimension audit

For every artifact store:

\[
shape(Z).
\]

Gate 11C expects:

\[
shape(Z_\eta)
=
shape(Z_0).
\]

If dimensionality differs, the comparison is not a STEP-11 primary comparison.

---

# 88. Tail/event audit

Store:

- train-derived tail threshold;
- number of tail examples;
- baseline tail \(P\);
- corrupted-trained tail \(P\);
- difference;
- confidence interval if feasible.

Do not hide a tail failure inside aggregate MAE.

---

# 89. Reproducibility artifacts

Recommended outputs:

1. `step11_protocol.json`
2. `step11_baseline_reproduction.json`
3. `step11_corruption_grid.json`
4. `step11_corruption_manifest.parquet`
5. `step11_reconstruction_metrics.parquet`
6. `step11_latent_stability.parquet`
7. `step11_downstream_clean_oos.parquet`
8. `step11_tail_event_metrics.parquet`
9. `step11_stress_oos.parquet`
10. `step11_seed_summary.parquet`
11. `step11_verdict.json`
12. `step11_audit_report.md`

---

# 90. Reproducibility manifest

Record:

- repository commit;
- encoder plugin;
- decoder plugin;
- latent dimension;
- dataset hashes;
- split files;
- normalization config;
- window size;
- model seed;
- corruption seed;
- corruption type/rate;
- downstream core configuration;
- CPU/GPU declaration;
- package versions.

---

# 91. Gate implementation summary

## 11A — baseline reproducibility

Current AE retrains and behaves as expected.

If fail:

\[
STOP.
\]

## 11B — stable corruption effect

At least one primary corruption arm has a stable interpretable effect.

If none:

\[
\boxed{
\text{STEP 11 closes; keep current AE.}
}
\]

## 11C — fixed interface

Same latent dimension.

If fail:

\[
\boxed{
\text{capacity confound.}
}
\]

## 11D — tail/event safety

Tail/event behavior does not materially collapse.

If fail:

\[
\boxed{
\text{reject corruption policy.}
}
\]

---

# 92. Prior-art reuse matrix

| Question | Prior art | STEP-11 action |
|---|---|---|
| Corrupted-input clean-target AE | Vincent et al. DAE | Use training principle only |
| Masked reconstruction concept | MAE | Conceptual analogy only; no ViT |
| Time-series masked reconstruction | Ti-MAE | Reference only |
| Masking can damage temporal variation | SimMTM | Use to justify mask-type ablation |
| Multiple masked views | SimMTM | Deferred if simple mask is promising |
| Siamese temporal consistency | TimeSiam | Deferred; not primary |
| Channel masking | time-series sensor masking literature | Optional diagnostic |
| New time-series MAE architecture | TS-MAE / related | Do not replace existing extractor |

---

# 93. State-of-the-art conclusion

The literature already establishes that:

\[
\boxed{
\text{partial corruption + reconstruction}
}
\]

can improve representation learning.

It also establishes that time-series masking is **not trivial** because:

- temporal variations can be destroyed;
- channels may contain coupled information;
- mask geometry matters;
- downstream utility is not guaranteed.

Therefore the project does not need to prove the generic existence of masked autoencoding.

The project-specific question is smaller and more useful:

\[
\boxed{
\text{Does the existing financial feature extractor benefit from a small, controlled train-only corruption policy?}
}
\]

---

# 94. Project-specific opportunity

If H11.1 survives, the result is useful because the project can retain:

- the same repository;
- the same plugins;
- the same saved encoder interface;
- the same downstream predictor contract;

while potentially producing a more robust latent.

That would make STEP 11 a true **hardening step**, not an architectural replacement.

---

# 95. Domain-general interpretation

This protocol generalizes naturally to any reconstruction-based extractor:

\[
X
\rightarrow
Encoder
\rightarrow
Z
\rightarrow
Decoder.
\]

The domain-specific choices are only:

- what corruption is meaningful;
- what tails/events must be preserved;
- what simple downstream assay represents useful knowledge.

The architecture contract stays stable.

---

# 96. References — IEEE style

[1] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and Composing Robust Features with Denoising Autoencoders,” in *Proceedings of the 25th International Conference on Machine Learning (ICML)*, pp. 1096–1103, 2008, doi: 10.1145/1390156.1390294. Available: https://doi.org/10.1145/1390156.1390294

[2] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked Autoencoders Are Scalable Vision Learners,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000–16009, 2022, doi: 10.1109/CVPR52688.2022.01553. Available: https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html

[3] J. Dong, H. Wu, H. Zhang, L. Zhang, J. Wang, and M. Long, “SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023. Available: https://papers.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html ; code: https://github.com/thuml/SimMTM

[4] Z. Li, Z. Rao, L. Pan, P. Wang, and Z. Xu, “Ti-MAE: Self-Supervised Masked Time Series Autoencoders,” arXiv:2301.08871, 2023. Available: https://arxiv.org/abs/2301.08871

[5] J. Dong, H. Wu, Y. Wang, Y.-Z. Qiu, L. Zhang, J. Wang, and M. Long, “TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling,” in *Proceedings of the 41st International Conference on Machine Learning*, PMLR vol. 235, pp. 11412–11436, 2024. Available: https://proceedings.mlr.press/v235/dong24e.html

[6] “TS-MAE: A Masked Autoencoder for Time Series Representation Learning,” *Information Sciences*, vol. 690, art. 121576, 2025, doi: 10.1016/j.ins.2024.121576. Available: https://doi.org/10.1016/j.ins.2024.121576

[7] J. Wang, W. Cui, T. Zhu, H. Ning, et al., “An Improved Masking Strategy for Self-Supervised Masked Reconstruction in Human Activity Recognition,” *IEEE Sensors Journal*, 2024, doi: 10.1109/JSEN.2024.3390755. Available: https://doi.org/10.1109/JSEN.2024.3390755

---

# 97. Verified project resources

## Existing extractor

https://github.com/harveybc/feature-extractor

The current repository README states that it trains encoder/decoder pairs on preprocessed financial time-series windows, evaluates reconstruction and saves encoder artifacts consumed by downstream predictor phases.

## Existing registered working families

- ANN
- CNN
- LSTM
- Transformer
- VAE
- VAE-small

## Small project-owned experiment bank

https://github.com/harveybc/feature-extractor/tree/master/examples/config/phase_3_2_daily

Primary suggested starting config:

`examples/config/phase_3_2_daily/phase_3_2_cnn_1d_config.json`

---

# 98. Final status

**STEP 11 is now deliberately narrow, falsifiable, and compliant with PATCH 002.**

It does not create a new model family.

It asks only whether:

\[
\boxed{
\text{controlled redundancy during training}
}
\]

makes the **existing** latent representation more robust/useful.

The final decision rule is:

\[
\boxed{
\text{helps}
\;\;|\;\;
\text{ties}
\;\;|\;\;
\text{hurts}.
}
\]

If Gate 11B is null, STEP 11 closes and the existing feature extractor remains unchanged.

If a corruption policy helps, the project's next question is not “add more masking methods.” The protocol should move forward to the next main-chain step using the hardened extractor as an eligible representation candidate.
