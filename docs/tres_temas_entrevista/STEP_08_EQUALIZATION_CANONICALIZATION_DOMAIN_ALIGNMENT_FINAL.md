# STEP 08 — Equalization, Canonicalization, Domain Alignment and Adaptive Calibration for Multivariate Time-Series ML

**Status:** Research protocol — final draft for agent audit and repository integration  
**Date:** 2026-09-05  
**Scope:** Communications-inspired equalization translated into systematic compensation of source-, domain-, instrument-, scale-, spectrum- and distribution-specific distortions before downstream feature extraction, core fusion and predictive heads.  
**Primary application:** multivariate financial time series at 1 h and 4 h periodicities, using the project’s standard chronological splits (4 years train + 1 year validation + 1 year test).  
**Prerequisites:** STEPS 01–07 and `WORKPLAN_PATCH_001_HISTORICAL_CHAIN_AND_COMPRESSION_EXTENSIONS.md`.  
**Next main-chain step:** STEP 09 — interference / echo / crosstalk cancellation.

---

# 0. Executive summary

STEP 08 asks:

> **Can systematic distortions introduced by the observation/source/domain process be compensated so that different windows, assets, sensors, regimes or venues are presented to the model in a more canonical form without erasing target-relevant information?**

The classical communications model is:

\[
y = h*x+n
\]

or, in finite-dimensional form:

\[
\mathbf y = \mathbf H\mathbf x+\mathbf n.
\]

An equalizer seeks:

\[
\hat{\mathbf x}
=
\mathbf W\mathbf y
\]

such that the distortion induced by \(\mathbf H\) is reduced.

The corresponding general ML model proposed here is:

\[
\boxed{
\mathbf y_t^{(d)}
=
\mathcal G_d
\left(
\mathbf s_t
\right)
+
\mathbf n_t
}
\]

where:

- \(\mathbf s_t\): latent/canonical information-bearing process;
- \(d\): source/domain/instrument/regime;
- \(\mathcal G_d\): systematic observation/channel transformation;
- \(\mathbf n_t\): noise handled primarily in STEP 03.

STEP 08 seeks an equalizer/canonicalizer:

\[
\boxed{
\mathbf z_t
=
\mathcal E_d
\left(
\mathbf y_{\le t}^{(d)}
\right)
}
\]

such that \(\mathbf z_t\):

1. removes or reduces nuisance distortion;
2. preserves target-relevant information;
3. is causal;
4. transfers better across domains/regimes;
5. makes a shared downstream model easier to train.

The state of the art shows that this is **not equivalent to ordinary z-score normalization**. Relevant mature families include:

- Wiener/MMSE and zero-forcing equalization;
- LMS/RLS adaptive equalization;
- blind/self-recovering equalization;
- adaptive normalization for time series (DAIN);
- reversible instance normalization (RevIN);
- slice adaptive normalization (SAN);
- Dish-TS;
- Non-stationary Transformers;
- frequency-adaptive normalization (FAN);
- frequency-domain normalization (FredNormer);
- domain alignment (CORAL, MMD/DAN, DANN, optimal transport);
- time-series-specific sensor/domain alignment;
- causal conditional-shift adaptation;
- test-time adaptation (TAFAS);
- 2026 frequency-aware test-time calibration (FAC).

The central scientific warning from modern time-series forecasting is equally important:

\[
\boxed{
\text{more stationarization/alignment}
\not\Rightarrow
\text{more predictive information}
}
\]

Over-equalization can erase:

- shocks;
- level information;
- volatility state;
- non-stationary events;
- domain-specific predictive mechanisms.

Therefore STEP 08 always compares:

\[
\text{raw}
\]

versus:

\[
\text{equalized}
\]

versus:

\[
\boxed{
[\text{raw},\text{equalized}]
}
\]

before permitting irreversible replacement of the raw branch.

---

# 1. Historical communications basis

Equalization compensates distortion introduced by a communications channel.

For discrete convolution:

\[
y[n]
=
\sum_k h[k]x[n-k]
+
n[n].
\]

In frequency:

\[
Y(\omega)
=
H(\omega)X(\omega)
+
N(\omega).
\]

An ideal noiseless inverse would be:

\[
\hat X(\omega)
=
\frac{Y(\omega)}{H(\omega)}.
\]

However, if:

\[
|H(\omega)|\approx0,
\]

then direct inversion strongly amplifies noise.

This is the classical reason why:

> **perfect inversion is often inferior to regularized/noise-aware equalization.**

That principle maps directly to ML canonicalization.

---

# 2. STEP 08 is not STEP 03

STEP 03 models:

\[
X=S+N
\]

and asks:

> what variability behaves like noise and can be attenuated?

STEP 08 models:

\[
Y=\mathcal G(S)+N
\]

and asks:

> what **systematic transformation** has distorted the useful process?

Examples:

- source gain;
- source offset;
- changing volatility scale;
- frequency response;
- sensor calibration;
- broker/venue conventions;
- unit/scaling differences;
- nonlinear response;
- covariance rotation;
- domain-specific distribution shift.

Thus:

\[
\boxed{
\text{noise removal}
\neq
\text{equalization}
}
\]

even though the noise model affects the optimal equalizer.

---

# 3. STEP 08 is not STEP 09

STEP 08 primarily compensates:

\[
\boxed{
\text{source/channel distortion of one information stream or domain}
}
\]

whereas STEP 09 will ask:

\[
\boxed{
\text{how much of one stream is shared contamination/interference from others?}
}
\]

There can be mathematical overlap in multivariate linear systems, but the research questions are intentionally separated.

---

# 4. Linear equalization model

Finite-dimensional model:

\[
\mathbf y
=
\mathbf H\mathbf x
+
\mathbf n.
\]

Desired equalizer:

\[
\hat{\mathbf x}
=
\mathbf W\mathbf y.
\]

If:

\[
\mathbf W\mathbf H\approx\mathbf I,
\]

the equalizer removes channel distortion.

---

# 5. Zero-forcing equalization

When \(\mathbf H\) is known and invertible/pseudoinvertible:

\[
\boxed{
\mathbf W_{ZF}
=
\mathbf H^\dagger
}
\]

so that, ideally:

\[
\mathbf W_{ZF}\mathbf H
=
\mathbf I.
\]

For full-column-rank \(\mathbf H\):

\[
\mathbf H^\dagger
=
(\mathbf H^H\mathbf H)^{-1}\mathbf H^H.
\]

## Strength

In the noiseless case, it directly removes linear channel distortion.

## Failure mode

\[
\boxed{
\text{ZF can amplify noise severely near channel nulls.}
}
\]

This becomes one of STEP 08's first synthetic falsification tests.

---

# 6. MMSE equalization

Rather than enforce perfect inversion, minimize:

\[
E[
\|\mathbf x-\mathbf W\mathbf y\|^2
].
\]

A common form under simplified assumptions is:

\[
\boxed{
\mathbf W_{MMSE}
=
(\mathbf H^H\mathbf H+\sigma_n^2\mathbf I)^{-1}
\mathbf H^H
}
\]

with appropriate scaling/generalization when source covariance is not identity.

More generally:

\[
\boxed{
\mathbf W_{MMSE}
=
C_{xy}C_{yy}^{-1}.
}
\]

The key principle is:

\[
\boxed{
\text{accept some residual distortion to avoid excessive noise amplification.}
}
\]

This is likely one of the most transferable principles from equalization to ML preprocessing.

---

# 7. Wiener filtering / regularized inverse

In scalar frequency-domain form, a Wiener estimate can be expressed conceptually as:

\[
\hat X(\omega)
=
W(\omega)Y(\omega)
\]

with:

\[
W(\omega)
=
\frac{
H^*(\omega)S_{XX}(\omega)
}{
|H(\omega)|^2S_{XX}(\omega)
+
S_{NN}(\omega)
}.
\]

Thus frequency components are inverted only to the degree justified by signal and noise power.

This naturally links:

- STEP 03 spectral SNR;
- STEP 06 spectral representation;
- STEP 08 frequency-selective equalization.

---

# 8. Adaptive equalization

When the channel changes:

\[
H=H_t,
\]

a fixed equalizer becomes stale.

Adaptive methods update:

\[
\mathbf w_t.
\]

A canonical LMS-style update is:

\[
\boxed{
\mathbf w_{t+1}
=
\mathbf w_t
+
\mu e_t\mathbf x_t
}
\]

up to convention-dependent complex conjugation/factors.

RLS instead minimizes exponentially weighted historical squared error and adapts faster at higher computational cost.

These algorithms establish a critical STEP-08 principle:

\[
\boxed{
\text{equalization can be causal and adaptive rather than one fixed train-time transform.}
}
\]

---

# 9. Blind equalization

Communications systems sometimes cannot use a known training sequence indefinitely.

Godard's self-recovering equalization showed that structural assumptions on the transmitted signal can support adaptive equalization without a known training sequence.

ML analogue:

> infer/correct domain distortion using distributional or structural constraints without target labels.

This connects directly to:

- unsupervised domain adaptation;
- test-time normalization;
- distribution alignment;
- self-supervised calibration.

---

# 10. Generalized time-series observation model

For domain \(d\):

\[
\mathbf y_t^{(d)}
=
\mathcal G_d
(
\mathbf s_{t-L:t}
)
+
\mathbf n_t.
\]

The transformation can contain:

\[
\mathcal G_d
=
\mathcal N_d
\circ
\mathcal M_d
\circ
\mathcal H_d
\circ
\mathcal A_d
\]

where, conceptually:

- \(\mathcal A_d\): affine gain/offset;
- \(\mathcal H_d\): linear temporal filtering/frequency response;
- \(\mathcal M_d\): multivariate covariance/mixing transformation;
- \(\mathcal N_d\): nonlinear observation/calibration function.

Equalization seeks:

\[
\mathcal E_d
\approx
\mathcal G_d^{-1}
\]

only to the degree that inversion is stable and useful.

---

# 11. Canonical representation objective

For multiple domains:

\[
d\in\{1,\ldots,D\},
\]

seek:

\[
z
=
E_d(y)
\]

such that:

## Domain discrepancy is reduced

\[
D(
p(z|d=i),
p(z|d=j)
)
\downarrow.
\]

## Task information is preserved

\[
P(Y|z)
\]

remains accurate.

The desired objective is **not** pure invariance.

Conceptually:

\[
\boxed{
\min
L_{\text{task}}
+
\lambda D_{\text{domain}}
+
\eta L_{\text{reconstruction/invertibility}}
}
\]

with exact terms determined by the experiment.

---

# 12. Existing project baseline

The current `predictor` repository already has explicit normalization plumbing through `use_normalization_json`, including normalization metadata loading and final prediction denormalization paths.

Therefore STEP 08 must treat the project's current training-derived normalization as:

\[
\boxed{
\text{baseline E0}
}
\]

rather than rebuilding basic normalization infrastructure.

The research question is what additional equalization/canonicalization mechanisms add value beyond this existing baseline.

Relevant project locations include:

- `app/data_processor.py`
- `preprocessor_plugins/helpers.py`
- `pipeline_plugins/default_pipeline.py`
- `pipeline_plugins/stl_norm.py`
- current experiment JSON configurations.

---

# 13. State of the art — adaptive normalization before modern deep forecasting

## 13.1. DAIN

Deep Adaptive Input Normalization (DAIN) was proposed specifically for time-series forecasting and evaluated on:

- a large-scale limit-order-book financial dataset;
- load forecasting.

DAIN learns normalization parameters end-to-end rather than applying a fixed preprocessing rule.

Its importance for this project is direct:

\[
\boxed{
\text{financial time-series adaptive normalization is established prior art.}
}
\]

Therefore a custom trainable normalizer should be benchmarked against DAIN rather than invented without comparison.

---

# 14. RevIN — reversible instance normalization

RevIN normalizes each input instance using instance-specific statistics and reverses that transformation at the output.

Conceptually:

\[
x'
=
\frac{x-\mu_x}{\sigma_x}.
\]

Model predicts:

\[
\hat y'
\]

and then:

\[
\hat y
=
\sigma_x\hat y'
+
\mu_x
\]

with learnable affine terms in the actual method.

Key principle:

\[
\boxed{
\text{remove nuisance distribution shift temporarily, restore scale at output.}
}
\]

This is exceptionally close to the equalization analogy because it explicitly uses a reversible compensation layer.

---

# 15. Over-stationarization: a central warning

Non-stationary Transformers showed that straightforward stationarization can make series easier to predict but can also remove non-stationary information necessary for distinguishing bursty or unusual events.

Their framework combines:

1. Series Stationarization;
2. De-stationary Attention.

This creates a mandatory STEP-08 control:

\[
\boxed{
\text{equalize nuisance}
\quad\text{without deleting useful non-stationarity.}
}
\]

Any method that improves average MSE while destroying shock/event information must not be accepted automatically.

---

# 16. SAN — local temporal-slice equalization

Slice Adaptive Normalization (SAN) challenges the assumption that one entire input window shares common statistics.

Instead it works at local temporal slices and models evolving statistics.

This is important for financial series because:

\[
\mu_t,\sigma_t
\]

can change substantially inside a long lookback window.

STEP-08 experiment:

\[
\text{global train normalization}
\]

vs.

\[
\text{instance normalization}
\]

vs.

\[
\text{slice/local normalization}.
\]

---

# 17. Dish-TS — input/output distribution shift

Dish-TS explicitly separates:

- intra-input distribution shift;
- shift between lookback/input distribution and future/horizon distribution.

This is a deeper version of equalization because the correct transformation may depend not only on observed input statistics but also on the expected output-domain statistics.

Project implication:

> Do not assume that making the historical lookback stationary automatically creates a representation aligned with the future forecasting horizon.

---

# 18. Frequency Adaptive Normalization (FAN)

FAN extends normalization into frequency structure.

It identifies instance-wise dominant frequency components and models discrepancies in those components.

This is particularly relevant after STEP 06:

\[
\boxed{
\text{equalization can be spectral, not only mean/variance normalization.}
}
\]

FAN therefore becomes a high-priority benchmark rather than an optional curiosity.

---

# 19. FredNormer

FredNormer analyzes normalization from the frequency domain and adaptively reweights components based on frequency stability/sample-specific behavior.

This strengthens the STEP-08 hypothesis:

> source/domain distortion may act differently at different frequencies.

A scalar z-score cannot correct a frequency-dependent transfer function.

---

# 20. Koopman approaches as optional dynamics canonicalization

Koopa and related Koopman-based forecasting methods seek measurement spaces in which nonlinear/non-stationary dynamics become easier to model.

This is not a classical equalizer, but it is relevant to a broader canonicalization question:

\[
x
\rightarrow
g(x)
\]

where dynamics become more stable/linear.

Koopa explicitly separates time-invariant and time-variant components and uses context-dependent operators.

This belongs to an **advanced dynamics-canonicalization extension**, not the first STEP-08 baseline.

---

# 21. Generic domain alignment: CORAL

CORAL aligns second-order statistics between domains.

For source covariance:

\[
C_S
\]

and target covariance:

\[
C_T,
\]

the idea is to transform features so their covariance structures align.

This is a strong low-cost baseline for:

\[
\boxed{
\text{multivariate covariance equalization}.
}
\]

But covariance alignment alone does not preserve target semantics automatically.

---

# 22. MMD / Deep Adaptation Networks

Deep Adaptation Networks use Maximum Mean Discrepancy-like distribution matching in RKHS to reduce source-target distribution discrepancy.

General principle:

\[
\mathrm{MMD}^2
(
P_S,
P_T
)
\downarrow.
\]

This provides a more general domain-equalization baseline than mean/covariance normalization.

---

# 23. Domain-adversarial equalization

DANN learns representations that:

1. predict the task;
2. make source/target domain difficult to discriminate.

Conceptually:

\[
\min_E
L_{\text{task}}
-
\lambda L_{\text{domain classifier}}.
\]

This can be interpreted as learned latent equalization:

\[
\boxed{
\text{remove domain identity while preserving task identity.}
}
\]

But negative transfer is possible if the domain-specific signal is genuinely predictive.

---

# 24. Optimal transport alignment

Optimal transport seeks a coupling between source and target distributions minimizing transport cost while preserving useful geometry.

This is a powerful canonicalization tool when shift is more complex than affine/covariance change.

It should be included only after simpler normalization/alignment baselines.

---

# 25. Time-series-specific domain adaptation

Generic domain adaptation methods can fail because time-series shifts affect:

- temporal dependencies;
- lags;
- frequency content;
- label distributions;
- sensor/channel semantics.

Relevant prior art includes:

- time-series adaptation under feature and label shifts;
- Sensor Alignment (SEA);
- sparse associative structure alignment;
- weakly guided/domain-adversarial forecasting;
- causal conditional-shift forecasting.

Thus STEP 08 must avoid importing image-domain adaptation methods without temporal controls.

---

# 26. SEA — sensor-level alignment

SEnsor Alignment explicitly recognizes that multivariate channels can have different domain shifts.

It aligns:

- local sensor features;
- sensor correlations;
- global features;
- spatio-temporal dependencies.

This is especially relevant to the project's feature-family/branch architecture.

Project translation:

\[
\boxed{
\text{equalization may need to be feature/branch-specific rather than one global transform.}
}
\]

---

# 27. Domain adaptation under feature and label shifts

ICML 2023 work on time-series domain adaptation explicitly addresses both:

- feature shift;
- label shift.

This is critical because naive equalization assumes:

\[
P(Y|X)
\]

is stable while only:

\[
P(X)
\]

moves.

Financial regimes can violate that assumption.

Therefore STEP 08 must diagnose:

\[
\boxed{
\text{covariate/domain shift}
\quad\text{vs.}\quad
\text{conditional/label mechanism shift}.
}
\]

---

# 28. Causal conditional shift

Transferable Time-Series Forecasting Under Causal Conditional Shift emphasizes that time-series transfer can be affected by:

- offsets;
- time lags;
- changing distributions;
- domain-specific conditional dependencies.

This is especially relevant to the financial data lake because:

> two assets may not merely be differently scaled versions of one process; their mechanisms can differ.

Canonicalization is valid only when the transformed domains share something meaningfully invariant.

---

# 29. Weakly guided robust adaptation

DARF/Weakly Guided Adaptation for Robust Time Series Forecasting uses domain-adversarial ideas while preserving multivariate correlations and handling abrupt but legitimate changes.

This reinforces an important distinction:

\[
\boxed{
\text{rare abrupt change}
\neq
\text{outlier}
\neq
\text{nuisance domain shift}.
}
\]

This should be explicitly tested in financial event windows.

---

# 30. Test-time adaptation: TAFAS

TAFAS adapts a pretrained forecaster at test time under temporal distribution shift using gated calibration and partially observed/revealed target information.

This establishes that:

\[
\boxed{
\text{equalization/adaptation need not stop at training time.}
}
\]

However, project governance must be stricter than common benchmark settings.

No future target may be used before it has matured and become historically available.

---

# 31. 2026 protocol-clean TTA and Frequency-Aware Calibration

Recent 2026 work on principled time-series TTA explicitly criticizes heterogeneous use of revealed targets and proposes adaptation using only matured ground truth.

It also introduces Frequency-Aware Calibration (FAC), which directly parameterizes forecast corrections in the frequency domain.

This is highly relevant to this project because it connects:

\[
\boxed{
\text{causal test-time adaptation}
+
\text{STEP-06 frequency representation}
+
\text{STEP-08 equalization}.
}
\]

FAC should be treated as current state-of-the-art prior art for frequency-aware calibration rather than reinvented.

---

# 32. Negative transfer and confidence-gated alignment

Open-set/time-varying domains can contain states not represented in training.

Aggressive alignment can map an unknown regime incorrectly into a known source domain.

Recent 2026 time-series adaptation work uses confidence-aware selective alignment to reduce this risk.

Project implication:

> domain alignment should eventually be gated by OOD/uncertainty evidence rather than applied blindly.

This links STEP 08 to the project's existing OOD work.

---

# 33. Taxonomy of equalizers for this project

## E0 — Existing fixed training normalization

Current project baseline.

## E1 — Global affine equalizer

Training-only:

\[
z
=
\frac{x-\mu_{train}}{\sigma_{train}}.
\]

## E2 — Robust affine equalizer

Median/IQR/MAD.

## E3 — Instance equalizer

RevIN-style.

## E4 — Slice/local equalizer

SAN-style.

## E5 — Learned adaptive equalizer

DAIN-style.

## E6 — Input/output distribution equalizer

Dish-TS-style.

## E7 — Frequency-adaptive equalizer

FAN/FredNormer-style.

## E8 — Linear inverse / MMSE equalizer

Known/estimated transfer function.

## E9 — Covariance equalizer

Whitening/CORAL.

## E10 — Latent distribution equalizer

MMD/DAN.

## E11 — Adversarial domain equalizer

DANN/DARF.

## E12 — Optimal-transport equalizer

OT-based.

## E13 — Causal test-time equalizer

TAFAS/FAC-like.

## E14 — Dynamics canonicalizer

Koopman/learned measurement-space variant.

---

# 34. Synthetic equalization benchmark is mandatory

Before financial interpretation, create canonical process:

\[
S_t.
\]

Generate observed domains:

\[
Y_t^{(d)}
=
G_d(S_t)+N_t.
\]

Because \(S_t\) is known, equalization quality is directly measurable.

---

# 35. Synthetic channel C0 — identity

\[
Y=S.
\]

Equalizers should not materially damage the clean signal.

This is a mandatory negative control.

---

# 36. Synthetic channel C1 — gain and offset

\[
Y_t
=
aS_t+b.
\]

Tests:

- training z-score;
- RevIN;
- DAIN;
- oracle affine inverse.

Expected:

simple methods should solve this case.

If a complex equalizer cannot, implementation is invalid.

---

# 37. Synthetic channel C2 — slowly varying affine drift

\[
Y_t
=
a_tS_t+b_t
\]

with:

\[
a_t,b_t
\]

varying smoothly.

Tests:

- fixed training normalization;
- RevIN;
- SAN;
- adaptive LMS-like normalization;
- test-time calibration.

---

# 38. Synthetic channel C3 — FIR / frequency response

\[
Y_t
=
h*S_t+N_t.
\]

Generate:

- low-pass;
- high-pass tilt;
- notch;
- resonant filter;
- mild multipath/FIR.

Compare:

- ZF;
- MMSE/Wiener;
- learned Conv inverse;
- FAN/FredNormer-like frequency compensation.

---

# 39. Synthetic channel C4 — spectral null/noise amplification

Construct:

\[
|H(f_0)|
\approx0.
\]

This directly tests:

\[
\boxed{
ZF
\quad\text{vs.}\quad
MMSE.
}
\]

Expected:

ZF reconstruction may improve channel inversion but amplify noise around \(f_0\).

MMSE should dominate for suitable noisy conditions.

---

# 40. Synthetic channel C5 — nonlinear monotonic response

Examples:

\[
Y
=
\operatorname{sgn}(S)|S|^\gamma
\]

or:

\[
Y
=
\log(1+\alpha S)
\]

for positive-domain variants.

Compare:

- fixed affine normalization;
- monotonic spline/calibration;
- invertible neural transform;
- raw + canonical representation.

---

# 41. Synthetic channel C6 — covariance rotation/scaling

For multivariate canonical process:

\[
\mathbf Y
=
A\mathbf S+\mathbf n.
\]

Use non-diagonal:

\[
A.
\]

Test:

- per-channel z-score;
- whitening;
- CORAL;
- multivariate linear inverse.

This separates marginal normalization from joint covariance equalization.

---

# 42. Synthetic channel C7 — distribution shift without mechanism shift

Generate domains:

\[
P_d(X)
\]

different while:

\[
P(Y|S)
\]

remains fixed.

This is the favorable case for domain-invariant canonicalization.

Test:

- CORAL;
- MMD;
- DANN;
- OT;
- source-specific equalizers + shared predictor.

---

# 43. Synthetic channel C8 — conditional mechanism shift

Change:

\[
P_d(Y|S).
\]

This is a critical adversarial control.

Expected:

forcing domain invariance can produce negative transfer.

A valid STEP-08 protocol must detect this rather than declaring every lower domain discrepancy a success.

---

# 44. Synthetic channel C9 — unknown/open-set domain

Create validation regime outside training support.

Test:

- unconditional alignment;
- OOD-gated alignment;
- fallback to raw/source-specific branch.

This connects with confidence-aware/open-set adaptation.

---

# 45. Equalization metrics — canonical reconstruction

For synthetic known canonical signal:

\[
D_{canon}
=
E[
\|
S-\hat S
\|^2
].
\]

Also report:

- MAE;
- correlation;
- spectral error;
- phase error where appropriate.

---

# 46. Distribution-alignment metrics

Possible diagnostics:

## Mean shift

\[
\|\mu_S-\mu_T\|.
\]

## Covariance difference

\[
\|C_S-C_T\|_F.
\]

## MMD

\[
\mathrm{MMD}(P_S,P_T).
\]

## Wasserstein / Sinkhorn distance

\[
W(P_S,P_T).
\]

## Domain classifier accuracy

Train classifier:

\[
D(z)\rightarrow d.
\]

Perfectly equalized domain-invariant representation tends toward chance-level domain prediction.

But:

> low domain-classifier accuracy is not sufficient for success.

---

# 47. Task-preservation metric

A canonicalizer is accepted only if:

\[
P(Y|Z)
\]

is preserved or improved.

Therefore measure:

\[
\Delta P
=
P(Z)-P(X).
\]

A transform with excellent domain alignment but worse forecasting is a failure for this project.

---

# 48. Equalization Pareto frontier

For configuration \(e\), define:

- domain discrepancy:
  \[
  D_e;
  \]
- forecasting performance:
  \[
  P_e;
  \]
- computational cost:
  \[
  C_e.
  \]

Build Pareto frontier:

\[
\boxed{
D_e
\leftrightarrow
P_e
\leftrightarrow
C_e.
}
\]

The goal is not:

\[
D_e\rightarrow0
\]

at any cost.

---

# 49. Information-preservation control

Always compare:

## Raw

\[
X.
\]

## Equalized

\[
E(X).
\]

## Parallel

\[
[X,E(X)].
\]

If:

\[
P([X,E(X)])
>
P(E(X)),
\]

then equalization has removed target-relevant information that the raw branch still contains.

---

# 50. Reversibility control

Compare:

- irreversible standardization/canonicalization;
- reversible transforms.

Hypothesis:

> reversible equalization should be safer when level/scale/non-stationary information matters to the target.

RevIN is the primary established baseline.

---

# 51. Financial canonicalization candidates

Potential nuisance transformations in financial inputs include:

- absolute price scale;
- volatility scale;
- pip/tick magnitude conventions;
- asset-specific amplitude distributions;
- session-dependent variance;
- long-run inflation/level drift;
- broker/source scaling;
- feature-family units;
- changing spectral emphasis.

Do **not** assume all of these are nuisances.

For example:

\[
\sigma_t
\]

may itself be target-relevant.

Therefore equalized representation should often include explicit removed-statistic side channels:

\[
[\tilde X,\mu,\sigma].
\]

---

# 52. Removed-statistics side channel

If:

\[
\tilde X_t
=
\frac{
X_t-\mu_t
}{
\sigma_t
},
\]

preserve:

\[
\mu_t,
\sigma_t
\]

as possible separate inputs.

This is analogous to:

\[
\boxed{
\text{canonical signal}
+
\text{channel state information}.
}
\]

This may be one of the most important STEP-08 architectural patterns.

---

# 53. Channel-state-information branch

Define:

\[
CSI_t
=
[
\mu_t,
\sigma_t,
\text{spectral tilt},
\text{domain ID/proxy},
\text{equalizer parameters},
\text{OOD score}
].
\]

Then:

\[
B_{canonical}
=
E(X)
\]

and:

\[
B_{channel}
=
CSI.
\]

Core receives both.

This prevents the equalizer from having to choose between:

- invariance;
- preserving useful domain state.

---

# 54. Equalization and STEP-06 spectral branches

For spectrum:

\[
Y(f)=H_d(f)S(f)+N(f).
\]

Candidate spectral equalization:

\[
\hat S(f)
=
W_d(f)Y(f).
\]

Possible \(W_d\):

- inverse response;
- Wiener/MMSE;
- learned frequency weights;
- FAN;
- FredNormer.

This is a particularly clean communications analogue.

---

# 55. Equalization and STEP-03 SNR

Use:

\[
SNR(f)
=
\frac{
S_{SS}(f)
}{
S_{NN}(f)
}.
\]

An equalizer should be less aggressive where inversion would amplify noise.

Hypothesis:

\[
W(f)
=
g(
H(f),SNR(f)
)
\]

should dominate noise-unaware inverse filtering.

---

# 56. Equalization and STEP-04 quantization

After equalization, dynamic range may change.

Therefore:

\[
Q(E(X))
\]

and:

\[
E(Q(X))
\]

are not generally equivalent.

Test ordering:

\[
\boxed{
E\rightarrow Q
}
\]

vs.

\[
\boxed{
Q\rightarrow E.
}
\]

Communications intuition usually favors compensation before final decision where feasible, but the project must test the practical order.

---

# 57. Equalization and STEP-05 source coding

A successful equalizer may reduce domain-specific redundancy and make source statistics more stable.

Measure:

\[
h(E(X))
\]

versus:

\[
h(X).
\]

But lower entropy is not necessarily better.

The useful question remains:

\[
\boxed{
\text{Does canonicalization create a representation whose structure transfers better?}
}
\]

---

# 58. Equalization and STEP-07 detectors

Template detector learned in canonical domain:

\[
D(s)
\]

should ideally transfer across observed domains after:

\[
E_d.
\]

This enables a powerful experiment:

## Without equalization

\[
D(Y^{(d_2)}).
\]

## With equalization

\[
D(E_{d_2}(Y^{(d_2)})).
\]

Hypothesis:

> canonicalization should increase cross-domain detector transfer.

---

# 59. Existing modular architecture integration

Recommended flow:

\[
X^{(d)}
\rightarrow
\begin{cases}
B_{raw}=X\\
B_{eq}=E_d(X)\\
B_{state}=CSI_d
\end{cases}
\]

Then:

\[
Z_i
=
Encoder_i(B_i).
\]

Core:

\[
Z
=
C(
Z_{raw},
Z_{eq},
Z_{state},
\ldots
).
\]

This is preferable to replacing all current raw inputs initially.

---

# 60. Feature-family-specific equalization

Do not apply one transform indiscriminately to:

- price levels;
- returns;
- volatility;
- oscillator indicators;
- calendar sin/cos;
- macro rates;
- binary/event features;
- token streams.

Examples:

## Bounded indicator

May need no scale equalization.

## Cyclic calendar feature

Already canonical:

\[
(\sin\theta,\cos\theta).
\]

## Price

May benefit from relative/volatility-aware canonicalization.

## Yield/rate

Level can be economically meaningful and should not automatically be removed.

The equalizer registry must be typed by feature semantics.

---

# 61. Domain definitions for financial experiments

Potential domains:

- asset;
- asset class;
- timeframe;
- historical regime;
- volatility regime;
- broker/venue;
- session;
- data vendor;
- pre/post structural market change.

A domain definition must be frozen before validation.

Do not create domains by inspecting test performance.

---

# 62. Public benchmark layer

Use a subset shared with STEPS 04–07:

- ETTh1;
- ETTm1;
- Weather;
- Electricity;
- Traffic;
- PEMS03;
- Solar.

Why these are useful:

- non-stationarity;
- seasonal frequency shift;
- different dimensionalities;
- sensor/channel heterogeneity;
- distribution drift.

---

# 63. Synthetic shift benchmark layer

Required before public forecasting claims:

1. affine shift;
2. variance drift;
3. spectral transfer distortion;
4. covariance rotation;
5. nonlinear calibration;
6. distribution shift;
7. conditional mechanism shift;
8. open-set regime.

This allows the equalizer's failure envelope to be measured explicitly.

---

# 64. Financial benchmark layer

Primary:

- EURUSD 1 h;
- EURUSD 4 h;
- ETHUSDT 4 h;
- other governed asset/timeframe cells.

Possible cross-domain tests:

- train shared representation across related FX pairs;
- train across related crypto assets;
- transfer across historical regimes.

Do not pool incompatible domains merely to create more rows.

---

# 65. Test-time adaptation governance

At inference timestamp:

\[
t,
\]

only information available by \(t\) may be used.

If a prediction target for earlier timestamp:

\[
t-h
\]

has now matured, it may be used only if the production system would genuinely know it at time \(t\).

This should be encoded as:

\[
\boxed{
\text{matured-ground-truth adaptation only}.
}
\]

Recent 2026 TTA work independently emphasizes this protocol-clean requirement.

---

# 66. TTA update modes

Compare:

## Static

\[
\theta_t
=
\theta_{train}.
\]

## Statistics-only

Update mean/variance or normalization statistics.

## Affine-only

Update small normalization affine parameters.

## Calibration-only

Small correction head.

## Frequency calibration

FAC-style.

## Full model adaptation

Late-stage control only.

The project should prefer the smallest safe adaptation mechanism that yields reproducible benefit.

---

# 67. Adaptation rollback

Every adaptive equalizer must support fallback.

If:

\[
OOD_t>\tau
\]

or adaptation worsens calibration:

\[
\theta_t
\rightarrow
\theta_{\text{safe}}.
\]

This is particularly important in financial markets where a novel regime may make adaptation actively harmful.

---

# 68. Equalizer parameter telemetry

Persist:

- current mean/scale;
- local slice statistics;
- spectral weights;
- covariance transform;
- adaptation step size;
- domain discrepancy;
- OOD score;
- rollback status.

This turns equalization into an auditable system rather than an opaque preprocessing function.

---

# 69. Falsifiable hypotheses — classical equalization

## H8.1 — Known-channel inversion works in the noiseless case

For synthetic known invertible linear channel:

\[
D(
S,
W_{ZF}Y
)
<
D(S,Y).
\]

**Falsified if:** implementation fails a known recoverable channel.

---

## H8.2 — Zero forcing exhibits noise amplification near spectral nulls

When:

\[
|H(f_0)|\approx0
\]

and noise is nonzero:

\[
D_{ZF}
>
D_{MMSE}
\]

for suitable SNR.

**Falsified if:** controlled experiment fails to show the expected tradeoff.

---

## H8.3 — STEP-03 noise-aware MMSE approaches oracle equalization

Using estimated noise statistics:

\[
\widehat W_{MMSE}
\]

approaches the oracle detector/equalizer using true noise statistics.

**Falsified if:** STEP-03 estimation is insufficient for useful compensation.

---

# 70. Falsifiable hypotheses — normalization/canonicalization

## H8.4 — Fixed global z-score is insufficient under dynamic affine drift

For:

\[
Y_t=a_tS_t+b_t,
\]

RevIN/SAN/DAIN-style methods outperform fixed training normalization.

**Falsified if:** global normalization matches adaptive methods.

---

## H8.5 — Reversible equalization preserves target information better than irreversible stationarization

\[
P_{Rev}
>
P_{Irrev}
\]

under shifts where level/scale contains useful target information.

**Falsified if:** reversibility offers no benefit.

---

## H8.6 — Local/slice normalization is superior under rapid within-window drift

\[
P_{SAN-like}
>
P_{instance/global}
\]

when local statistics vary.

**Falsified if:** slice localization does not help.

---

# 71. Falsifiable hypotheses — frequency-domain equalization

## H8.7 — Frequency-aware normalization improves frequency-dependent shift

Under spectral transfer distortion:

\[
P_{FAN/Fred}
>
P_{time-only norm}.
\]

**Falsified if:** scalar/time-domain normalization is sufficient.

---

## H8.8 — Noise-aware spectral equalization dominates blind inversion

\[
P_{Wiener/MMSE}
>
P_{inverse-only}
\]

under nontrivial spectral noise.

**Falsified if:** noise awareness adds no value.

---

# 72. Falsifiable hypotheses — preserving non-stationarity

## H8.9 — Over-equalization hurts shock/event prediction

Aggressive stationarization may improve average MSE but reduce:

- event AP;
- shock lead-time;
- tail-conditioned forecasting quality.

**Falsified if:** aggressive equalization improves both ordinary and event-sensitive metrics.

---

## H8.10 — Raw + equalized representation can outperform equalized-only

\[
P([X,E(X)])
>
P(E(X))
\]

when equalization removes some useful non-stationary state.

**Falsified if:** raw branch is always redundant.

---

# 73. Falsifiable hypotheses — multivariate/domain equalization

## H8.11 — Feature/sensor-specific alignment beats one global alignment

For heterogeneous multivariate channels:

\[
P(E_1(X_1),\ldots,E_m(X_m))
>
P(E_{global}(X)).
\]

**Falsified if:** one global transform is sufficient.

---

## H8.12 — Covariance equalization improves transfer when second-order shift dominates

\[
P_{CORAL/whiten}
>
P_{marginal-only}.
\]

**Falsified if:** covariance correction gives no gain.

---

## H8.13 — Latent domain alignment improves transfer when task mechanism is stable

MMD/DANN/OT should improve target-domain performance under covariate/domain shift with stable:

\[
P(Y|S).
\]

**Falsified if:** domain alignment does not help under controlled favorable assumptions.

---

## H8.14 — Naive domain invariance causes negative transfer under conditional/label shift

When:

\[
P_d(Y|X)
\]

changes, aggressive alignment can hurt.

**Falsified if:** alignment remains universally beneficial.

This is intentionally a protective hypothesis.

---

# 74. Falsifiable hypotheses — source-specific canonicalization

## H8.15 — Source-specific equalizers + shared core outperform fully separate models under fixed total capacity

\[
P(
\{E_d\}
+
C_{shared}
)
>
P(
\{C_d\}_{separate}
)
\]

under related domains.

**Falsified if:** domain-specific separate models remain superior.

---

## H8.16 — Canonicalization reduces the model capacity required for transfer

For target performance \(P_0\):

\[
|\theta_{eq+shared}|
<
|\theta_{raw}|.
\]

**Falsified if:** canonicalization does not reduce complexity/sample requirements.

---

# 75. Falsifiable hypotheses — test-time adaptation

## H8.17 — Small causal TTA improves gradual drift

Statistics-only/affine/calibration updates improve OOS performance under gradual shift.

**Falsified if:** static equalizer consistently dominates.

---

## H8.18 — Aggressive TTA can hurt abrupt/financial shifts

Large/full-model adaptation may underperform conservative updates.

**Falsified if:** aggressive adaptation is universally safer.

---

## H8.19 — Frequency-aware calibration helps when residual shift is spectral

FAC-like correction:

\[
P_{FAC}
>
P_{time-calibration}
\]

for controlled frequency-domain drift.

**Falsified if:** no spectral-specific advantage appears.

---

## H8.20 — Matured-ground-truth-only adaptation is sufficient to obtain meaningful gains

A protocol-clean TTA system using only historically matured labels retains useful adaptation benefit.

**Falsified if:** gains depend on unavailable future information.

---

# 76. Falsifiable hypotheses — OOD and negative transfer

## H8.21 — Confidence/OOD-gated alignment is safer for open-set regimes

\[
P_{gated}
>
P_{always-align}
\]

when target domain includes unknown states.

**Falsified if:** gating provides no benefit.

---

## H8.22 — No universal equalizer exists

Best equalizer depends on the distortion class.

Expected pattern:

- affine shift:
  RevIN/SAN;
- spectral distortion:
  FAN/MMSE;
- covariance/domain shift:
  CORAL/MMD/OT;
- gradual deployment drift:
  TTA.

**Falsified if:** one method dominates all controlled distortion families.

---

# 77. Additional hypothesis — channel-state side information

## H8.23 — Preserving equalizer state improves prediction

If normalization removes:

\[
\mu_t,\sigma_t,\text{spectral state},
\]

feeding these as side information improves:

\[
P([E(X),CSI])
>
P(E(X)).
\]

**Falsified if:** equalizer state is always irrelevant.

---

# 78. Additional hypothesis — detector transfer

## H8.24 — Canonicalization improves STEP-07 detector portability

Detector learned on domain \(d_1\) transfers better to domain \(d_2\) after equalization:

\[
P(
D(E_{d_2}(X^{d_2}))
)
>
P(
D(X^{d_2})
).
\]

**Falsified if:** equalization does not improve cross-domain pattern stability.

---

# 79. Primary ablation matrix

| ID | Raw | Equalizer | CSI side branch | Domain adaptation | TTA |
|---|---:|---|---:|---:|---:|
| E00 | Yes | Existing training norm | No | No | No |
| E01 | No | RevIN | No | No | No |
| E02 | Yes | RevIN | No | No | No |
| E03 | No | SAN | No | No | No |
| E04 | No | DAIN | No | No | No |
| E05 | No | FAN/FredNormer | No | No | No |
| E06 | Yes | Best equalizer | Yes | No | No |
| E07 | Yes | Best equalizer | Yes | CORAL | No |
| E08 | Yes | Best equalizer | Yes | MMD/DANN | No |
| E09 | Yes | Best equalizer | Yes | Best | Conservative TTA |
| E10 | Synthetic | ZF | No | No | No |
| E11 | Synthetic | MMSE/Wiener | No | No | No |

---

# 80. Synthetic experiment matrix

For each distortion \(G_d\), evaluate:

\[
S
\rightarrow
G_d
\rightarrow
Y
\rightarrow
E
\rightarrow
\hat S.
\]

Report:

1. canonical reconstruction;
2. distribution discrepancy;
3. target prediction;
4. noise amplification;
5. equalizer stability;
6. runtime.

---

# 81. Public forecasting experiment

Recommended baseline backbones:

- DLinear;
- PatchTST;
- one current project-like Conv/LSTM model.

Equalizers:

- fixed train normalization;
- RevIN;
- SAN;
- FAN;
- selected domain alignment.

Do not modify the backbone while comparing equalizers.

---

# 82. Financial experiment

Freeze:

- feature set;
- target;
- windows;
- heads;
- optimizer;
- seeds.

Only equalizer varies.

Start with:

\[
\text{existing normalization}
\]

versus:

\[
\text{RevIN}
\]

versus:

\[
\text{SAN}
\]

versus:

\[
\text{FAN/FredNormer}
\]

versus:

\[
[\text{raw},\text{best equalized},CSI].
\]

Only then add latent domain adaptation.

---

# 83. Model-family controls

The equalizer should be tested with at least:

1. low-capacity linear/DLinear control;
2. sequence model;
3. project modular predictor.

Reason:

> equalization may help a small model but be redundant for a very large model.

That outcome itself is scientifically informative.

---

# 84. Parameter-matched comparison

If an equalizer adds trainable parameters:

\[
N_E,
\]

compare against a raw model with approximately:

\[
N_{raw}
=
N_{base}+N_E.
\]

Otherwise improved performance may simply reflect added capacity.

---

# 85. Compute-matched comparison

Record:

- equalizer fitting cost;
- per-window preprocessing latency;
- online adaptation latency;
- GPU/CPU memory;
- downstream model training time.

Adaptive normalization should not be accepted solely on statistical improvement if deployment cost is disproportionate.

---

# 86. Distribution-shift diagnostics

For each split/domain report:

- mean;
- variance;
- skew;
- kurtosis;
- PSD change;
- covariance change;
- MMD;
- Wasserstein;
- domain classifier accuracy;
- OOD score;
- clipping/saturation where relevant.

This establishes what type of shift the equalizer is actually solving.

---

# 87. Event/tail-preservation diagnostics

Condition metrics on:

- top volatility decile;
- large absolute returns;
- macro release windows where available;
- regime transitions;
- jump windows.

A canonicalizer that improves ordinary windows but destroys event behavior is not universally better.

---

# 88. Calibration

For probabilistic heads report:

- NLL;
- Brier score;
- reliability;
- uncertainty coverage.

Distribution shift frequently affects calibration before mean error visibly deteriorates.

---

# 89. Statistical testing

Use:

- Diebold–Mariano where appropriate;
- block/stationary bootstrap;
- multiple seeds;
- FDR correction for large equalizer grids.

Do not use iid resampling of timestamps.

---

# 90. Equalizer selection protocol

Training:

- fit source statistics;
- fit equalizer parameters;
- fit domain alignment.

Validation:

- choose equalizer family;
- choose hyperparameters;
- choose adaptation rules.

Test:

- one frozen final evaluation.

Test must not be used to decide whether the domain was “successfully equalized.”

---

# 91. Adaptive equalizer experiment

For validation stream:

\[
t=1,\ldots,T.
\]

At each time:

1. observe current causal input;
2. apply current equalizer;
3. predict;
4. when a historical target matures, optionally update permitted small parameters;
5. log update;
6. never revisit future observations.

This should be implemented as a replayable deterministic protocol.

---

# 92. Failure modes

## 92.1. Over-stationarization

Removes useful non-stationary state.

## 92.2. Noise amplification

Inverse filtering amplifies poorly observed frequencies.

## 92.3. Negative transfer

Domain alignment forces unlike mechanisms together.

## 92.4. Leakage

Local/instance statistics accidentally use horizon/future values.

## 92.5. TTA future-target leakage

Partially observed target used before maturity.

## 92.6. Domain definition leakage

Regimes constructed using future outcomes.

## 92.7. Covariance instability

High-dimensional inversion without shrinkage.

## 92.8. Equalizer collapse

Learned transform erases meaningful variation.

## 92.9. Raw-channel dominance

One branch dominates due to width rather than information.

## 92.10. Test-time catastrophic adaptation

Adaptive parameters drift into a novel regime.

---

# 93. Equalizer audit checklist

- [ ] STEP 03 noise and STEP 08 distortion are distinguished.
- [ ] STEP 09 interference is not prematurely folded into equalization.
- [ ] Existing project normalization is used as E0 baseline.
- [ ] Synthetic identity control is present.
- [ ] ZF noise amplification is tested.
- [ ] MMSE/Wiener noise-aware baseline is present.
- [ ] RevIN is benchmarked before custom reversible normalization.
- [ ] SAN is benchmarked for local drift.
- [ ] DAIN is benchmarked for trainable adaptive normalization.
- [ ] FAN/FredNormer are reviewed before custom frequency normalization.
- [ ] Non-stationary Transformer over-stationarization warning is explicitly tested.
- [ ] Dish-TS input/output shift idea is considered.
- [ ] CORAL is used as simple covariance-alignment baseline.
- [ ] MMD/DANN/OT are only promoted after simpler baselines.
- [ ] Time-series-specific domain adaptation prior art is included.
- [ ] Conditional/label shift is tested as a negative-transfer control.
- [ ] TTA uses matured historical targets only.
- [ ] OOD-gated fallback exists for adaptive alignment.
- [ ] Equalizer state can be exposed as side information.
- [ ] Raw + equalized branch is tested.
- [ ] Event/tail preservation is measured.
- [ ] Parameter-matched controls are included.
- [ ] Domain discrepancy is never used as the sole success metric.
- [ ] Final test remains untouched until selection.

---

# 94. Reuse matrix

| Problem | Existing prior art | Recommended action |
|---|---|---|
| Fixed project normalization | current `predictor` JSON pipeline | Reuse as baseline |
| Adaptive financial normalization | DAIN | Benchmark/reuse concepts |
| Reversible normalization | RevIN | High-priority baseline |
| Local/slice normalization | SAN | High-priority drift baseline |
| Input/output distribution shift | Dish-TS | Benchmark/reference |
| Over-stationarization | Non-stationary Transformers | Mandatory control |
| Frequency adaptive normalization | FAN | High-priority STEP-06/08 bridge |
| Frequency-domain normalization | FredNormer | Comparator |
| Covariance alignment | CORAL | Reuse |
| Kernel distribution alignment | DAN/MMD | Reuse |
| Adversarial alignment | DANN | Reuse |
| Optimal transport | Courty et al./POT ecosystem | Reuse |
| Sensor-level time-series UDA | SEA | Benchmark/reference |
| Time-series feature+label shift | ICML 2023 method | Reference/benchmark |
| Causal conditional shift | TPAMI 2024 | Reference |
| Robust forecasting adaptation | DARF | Reference |
| Test-time adaptation | TAFAS | High-priority adaptive baseline |
| Frequency-aware TTA | FAC 2026 | Current research reference |
| Dynamic canonical dynamics | Koopa/Koopman | Advanced extension |

---

# 95. Recommended implementation order

1. Synthetic identity/affine channels.
2. ZF/MMSE on known FIR channel.
3. Noise-aware spectral equalization.
4. Existing project normalization baseline.
5. RevIN.
6. SAN.
7. DAIN.
8. FAN/FredNormer.
9. Raw + equalized + CSI side branch.
10. Covariance whitening/CORAL.
11. MMD/DANN.
12. Time-series-specific domain alignment.
13. Public benchmark shift experiments.
14. Financial regime/domain transfer.
15. Conservative TTA.
16. FAC-style frequency calibration.
17. OOD-gated adaptation.
18. Koopman/dynamics canonicalization only if justified.

This ordering deliberately moves from identifiable distortion to increasingly unconstrained learned alignment.

---

# 96. Decision gates

## Gate 8A — Classical correctness

Known synthetic channels are equalized as expected.

## Gate 8B — Noise-aware correctness

MMSE/Wiener behaves better than ZF in noisy ill-conditioned channels.

## Gate 8C — Dynamic normalization value

At least one adaptive normalization outperforms existing fixed normalization under controlled drift.

## Gate 8D — Information preservation

Event/tail performance is not destroyed.

## Gate 8E — Frequency equalization value

Spectral methods add value when the shift is frequency-dependent.

## Gate 8F — Multivariate alignment value

Covariance/sensor-specific equalization improves transfer.

## Gate 8G — Negative-transfer safety

Method correctly fails/gates when conditional mechanism changes.

## Gate 8H — Public transfer

Benefit survives public benchmarks.

## Gate 8I — Financial validation

Benefit survives project validation.

## Gate 8J — TTA governance

Adaptive method passes causal matured-target replay.

## Gate 8K — Held-out confirmation

Only selected configuration reaches test.

---

# 97. Deliverables

Recommended artifacts:

1. `step08_equalizer_registry.json`
2. `step08_synthetic_channel_spec.json`
3. `step08_channel_oracle_results.parquet`
4. `step08_zf_mmse_surface.parquet`
5. `step08_normalization_ablation.parquet`
6. `step08_frequency_equalization.parquet`
7. `step08_domain_discrepancy.parquet`
8. `step08_event_tail_metrics.parquet`
9. `step08_domain_classifier_metrics.parquet`
10. `step08_tta_replay.parquet`
11. `step08_equalizer_state.parquet`
12. `step08_statistical_tests.json`
13. `step08_audit_report.md`
14. reproducibility manifest:
    - data hashes;
    - split dates;
    - equalizer parameters;
    - adaptation timestamps;
    - target maturity rules;
    - seeds;
    - git commits;
    - software versions.

---

# 98. State-of-the-art conclusions

The review establishes five major conclusions.

## 98.1. Normalization already evolved into learned equalization

DAIN, RevIN, SAN and Dish-TS demonstrate that normalization in forecasting is now a learnable/adaptive distribution-correction problem, not merely preprocessing.

## 98.2. Time-domain statistics are insufficient for all shifts

FAN and FredNormer show that non-stationarity/equalization can be frequency-specific.

## 98.3. Equalization can remove useful information

Non-stationary Transformers explicitly expose the risk of over-stationarization.

## 98.4. Multivariate domain alignment must respect channel structure

SEA and other time-series adaptation work show that per-sensor and temporal dependencies matter.

## 98.5. Deployment adaptation is now part of the state of the art

TAFAS and 2026 FAC-style work demonstrate causal-ish/test-time calibration as an active direction, with protocol cleanliness becoming a central issue.

---

# 99. Project-specific research opportunity

The likely contribution is not:

> “normalize financial time series.”

That is established.

The stronger research program is:

\[
\boxed{
\text{estimated channel/noise state}
\rightarrow
\text{typed equalizer}
\rightarrow
\text{canonical representation}
+
\text{channel-state side information}
\rightarrow
\text{STEP-06 representation}
\rightarrow
\text{STEP-07 detector}
\rightarrow
\text{shared modular core}
}
\]

with:

\[
\boxed{
\text{equalizer chosen according to measured distortion class}
}
\]

rather than one universal normalization layer.

---

# 100. Domain-general architecture implication

For any sequential/sensor domain:

\[
Y^{(d)}
\rightarrow
E_d
\rightarrow
Z^{canonical}
\]

while retaining:

\[
CSI_d.
\]

Then a common downstream architecture can potentially operate on:

\[
[Z^{canonical},CSI_d].
\]

This could generalize to:

- finance;
- industrial sensors;
- biomedical signals;
- energy;
- traffic;
- weather;
- telecommunications;
- robotics.

The equalizer plugin changes by domain; the architectural contract remains stable.

---

# 101. References — IEEE style

[1] N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series: With Engineering Applications*. Cambridge, MA, USA: MIT Press, 1949. Available: https://mitpress.mit.edu/9780262230025/extrapolation-interpolation-and-smoothing-of-stationary-time-series/

[2] B. Widrow and M. E. Hoff, “Adaptive Switching Circuits,” IRE WESCON Convention Record, Part 4, pp. 96–104, 1960. Available: https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf

[3] D. N. Godard, “Self-Recovering Equalization and Carrier Tracking in Two-Dimensional Data Communication Systems,” *IEEE Transactions on Communications*, vol. 28, no. 11, pp. 1867–1875, Nov. 1980, doi: 10.1109/TCOM.1980.1094608. Available: https://doi.org/10.1109/TCOM.1980.1094608

[4] N. Passalis, A. Tefas, J. Kanniainen, M. Gabbouj, and A. Iosifidis, “Deep Adaptive Input Normalization for Time Series Forecasting,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 31, no. 9, pp. 3760–3765, Sep. 2020, doi: 10.1109/TNNLS.2019.2944933. Available: https://doi.org/10.1109/TNNLS.2019.2944933

[5] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, “Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift,” in *International Conference on Learning Representations (ICLR)*, 2022. Available: https://openreview.net/forum?id=cGDAkQo1C0p

[6] Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting,” in *Advances in Neural Information Processing Systems*, vol. 35, 2022. Available: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html

[7] Z. Liu, M. Cheng, Z. Li, Z. Huang, Q. Liu, Y. Xie, and E. Chen, “Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023. Available: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html

[8] W. Fan, P. Wang, D. Wang, D. Wang, Y. Zhou, and Y. Fu, “Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting,” arXiv:2302.14829, 2023. Available: https://arxiv.org/abs/2302.14829

[9] W. Ye, S. Deng, Q. Zou, and N. Gui, “Frequency Adaptive Normalization for Non-Stationary Time Series Forecasting,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024, doi: 10.52202/079017-0985. Available: https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html

[10] X. Piao, Z. Chen, Y. Dong, Y. Matsubara, and Y. Sakurai, “FredNormer: Frequency Domain Normalization for Non-stationary Time Series Forecasting,” arXiv:2410.01860, 2024. Available: https://arxiv.org/abs/2410.01860

[11] Y. Liu, C. Li, J. Wang, and M. Long, “Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023. Available: https://proceedings.neurips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html

[12] B. Sun, J. Feng, and K. Saenko, “Correlation Alignment for Unsupervised Domain Adaptation,” arXiv:1612.01939, 2016. Available: https://arxiv.org/abs/1612.01939

[13] M. Long, Y. Cao, J. Wang, and M. I. Jordan, “Learning Transferable Features with Deep Adaptation Networks,” in *Proceedings of the 32nd International Conference on Machine Learning*, PMLR vol. 37, pp. 97–105, 2015. Available: https://proceedings.mlr.press/v37/long15.html

[14] Y. Ganin *et al.*, “Domain-Adversarial Training of Neural Networks,” *Journal of Machine Learning Research*, vol. 17, no. 59, pp. 1–35, 2016. Available: https://www.jmlr.org/papers/v17/15-239.html

[15] N. Courty, R. Flamary, D. Tuia, and A. Rakotomamonjy, “Optimal Transport for Domain Adaptation,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 9, pp. 1853–1865, Sep. 2017, doi: 10.1109/TPAMI.2016.2615921. Available: https://doi.org/10.1109/TPAMI.2016.2615921

[16] Y. Wang, Y. Xu, J. Yang, Z. Chen, M. Wu, X. Li, and L. Xie, “SEnsor Alignment for Multivariate Time-Series Unsupervised Domain Adaptation,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 8, pp. 10253–10261, 2023, doi: 10.1609/aaai.v37i8.26221. Available: https://doi.org/10.1609/aaai.v37i8.26221

[17] H. He, O. Queen, T. Koker, C. Cuevas, T. Tsiligkaridis, and M. Zitnik, “Domain Adaptation for Time Series Under Feature and Label Shifts,” in *Proceedings of the 40th International Conference on Machine Learning*, PMLR vol. 202, pp. 12746–12774, 2023. Available: https://proceedings.mlr.press/v202/he23b.html

[18] Z. Li, R. Cai, T. Z. J. Fu, Z. Hao, and K. Zhang, “Transferable Time-Series Forecasting Under Causal Conditional Shift,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 4, pp. 1932–1949, Apr. 2024, doi: 10.1109/TPAMI.2023.3304354. Available: https://doi.org/10.1109/TPAMI.2023.3304354

[19] Y. Cheng, P. Chen, C. Guo, K. Zhao, Q. Wen, B. Yang, and C. S. Jensen, “Weakly Guided Adaptation for Robust Time Series Forecasting,” *Proceedings of the VLDB Endowment*, vol. 17, no. 4, pp. 766–779, 2023, doi: 10.14778/3636218.3636231. Available: https://doi.org/10.14778/3636218.3636231

[20] H. Kim, S. Kim, J. Mok, and S. Yoon, “Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 17, pp. 17868–17876, 2025, doi: 10.1609/aaai.v39i17.33965. Available: https://doi.org/10.1609/aaai.v39i17.33965

[21] H. Wang, R. Xu, G. Kementzidis, K. Cho, S. Ramirez Villarreal, and Y. Deng, “Towards Principled Test-Time Adaptation for Time Series Forecasting,” arXiv:2605.17250, 2026. Available: https://arxiv.org/abs/2605.17250

[22] Z. Cai, G. Bai, R. Jiang, X. Song, and L. Zhao, “Continuous Temporal Domain Generalization,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024. Available: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html

[23] “Confidence-Aware Optimal Transport for Open Set Time Series Adaptation Under Non-Stationary Shifts,” *IEEE Signal Processing Letters*, vol. 33, pp. 1155–1159, 2026, doi: 10.1109/LSP.2026.3661475. Available: https://doi.org/10.1109/LSP.2026.3661475

---

# 102. Final status

**STEP 08 is theoretically specified after a dedicated state-of-the-art review and is ready for independent agent audit.**

The strongest project-specific principle is:

\[
\boxed{
\text{Do not normalize everything into sameness.}
}
\]

Instead:

\[
\boxed{
\text{identify the distortion class}
\rightarrow
\text{apply the least destructive equalizer}
\rightarrow
\text{retain channel-state information}
\rightarrow
\text{verify OOS task preservation}
}
\]

The desired endpoint is not one universal normalization plugin.

It is a typed equalization layer capable of selecting or learning the appropriate compensation for:

- affine drift;
- volatility/scale shift;
- frequency-response shift;
- covariance/domain shift;
- deployment-time drift;

while preserving raw information when canonicalization is uncertain.

If agent audit approves this protocol, proceed to:

**STEP 09 — interference / echo / crosstalk cancellation: separating shared/common components from unique innovation across multiple input series.**
