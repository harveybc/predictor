# STEP 04 — Quantization, Companding and Informational Resolution for Time-Series ML

**Status:** Research protocol — final draft for agent audit and repository integration  
**Date:** 2026-09-05  
**Scope:** Input representation for multivariate time-series machine learning, with special attention to financial series, modular multi-branch predictors, causal preprocessing, noise-aware resolution, scalar/non-uniform quantization, companding, tokenization, and later latent vector quantization.  
**Prerequisites:** STEP 01 — Sampling/Nyquist; STEP 02–03 — noise/SNR estimation and denoising protocol.  
**Next planned communications analogue:** STEP 05 — source/symbol coding, entropy coding, dictionaries/tokens and representation efficiency.

---

# 0. Executive summary

This step asks a precise question:

> **Given that an observed feature has finite signal-to-noise ratio and finite predictive usefulness, what amplitude resolution should be delivered to a machine-learning model, and can a quantized or companded representation improve generalization, robustness or information efficiency relative to raw floating-point values?**

The telecommunications analogue is not merely “rounding numbers.” It is the chain:

\[
\text{source amplitude statistics}
\rightarrow
\text{dynamic-range conditioning}
\rightarrow
\text{companding if required}
\rightarrow
\text{quantization}
\rightarrow
\text{discrete representation}
\]

In the proposed ML analogue:

\[
X_j
\rightarrow
\widehat{\mathrm{SNR}}_j
\rightarrow
\text{training-only scale/range}
\rightarrow
C_j(\cdot)
\rightarrow
Q_j(\cdot)
\rightarrow
\text{representation delivered to branch }j
\]

where:

- \(X_j\) is one feature or one feature family;
- \(C_j\) is an optional companding transform;
- \(Q_j\) is a scalar or learned quantizer;
- all fitted parameters are estimated from training data only.

The central hypotheses are falsable. The protocol does **not** assume that quantization helps. It explicitly permits the final conclusion that raw continuous data are superior.

---

# 1. Corrected placement of companding in the research roadmap

The original roadmap placed “coding/companding” after quantization. This should be corrected.

In classical PCM systems, companding is conceptually part of **non-uniform quantization**. One can view the system as:

\[
x
\overset{C}{\longrightarrow}
u
\overset{Q_\mathrm{uniform}}{\longrightarrow}
q
\overset{C^{-1}}{\longrightarrow}
\hat{x}
\]

where:

- \(C\) compresses dynamic range;
- uniform quantization operates in the compressed domain;
- \(C^{-1}\) expands the reconstruction.

Therefore the roadmap should become:

1. **Sampling / Nyquist**
2. **Noise and SNR estimation**
3. **Denoising / signal-preservation validation**
4. **Quantization, non-uniform quantization and companding**
5. **Source/symbol coding and entropy-efficient representation**
6. **Representation/modulation analogues: amplitude, phase, frequency, time-frequency**
7. **Matched filtering / specialized detection / branch-specific processing**
8. **Controlled redundancy / error-correction analogues**
9. **End-to-end evaluation and information bottlenecks**

This document closes the theoretical and experimental design of Step 4, subject to audit.

---

# 2. Research question

For each input feature \(X_j\), determine whether there exists a finite effective resolution:

\[
b_j^*
\]

or equivalently a finite set of reconstruction levels:

\[
\mathcal{Q}_j
=
\{q_{j,1},\ldots,q_{j,L_j}\}
\]

such that increasing numerical resolution above \(b_j^*\) produces no statistically meaningful improvement in out-of-sample forecasting performance.

The step also tests whether the optimal resolution depends on:

- signal-to-noise ratio;
- feature distribution;
- dynamic range;
- temporal regime;
- denoising;
- feature family;
- model architecture;
- whether the quantized representation replaces or supplements the raw input.

---

# 3. Classical scalar quantization

Let:

\[
x\in[x_{\min},x_{\max}]
\]

and let a scalar quantizer have:

\[
L=2^b
\]

reconstruction levels.

For a uniform quantizer, the nominal step is approximately:

\[
\Delta
=
\frac{x_{\max}-x_{\min}}{L}
\]

and the reconstruction is:

\[
x_q=Q_\Delta(x).
\]

The quantization error is:

\[
e_q=x-x_q.
\]

Under the classical high-resolution model, away from saturation and under suitable regularity assumptions, the quantization error is often approximated as uniform over one cell:

\[
e_q\sim \mathcal{U}
\left(
-\frac{\Delta}{2},
\frac{\Delta}{2}
\right)
\]

leading to:

\[
\mathrm{Var}(e_q)
\approx
\frac{\Delta^2}{12}.
\]

This approximation is useful but must **not** be treated as universally valid. It can fail for:

- very low bit depth;
- deterministic or highly structured signals;
- coarse quantization;
- saturation;
- non-stationary distributions;
- quantizers whose error is correlated with the source.

Gray and Neuhoff provide the canonical broad review of quantization theory and its history [2].

---

# 4. Saturation and dynamic range are first-class variables

A quantizer requires a representable interval:

\[
[a_j,b_j].
\]

Validation or test observations may fall outside the training range. A production-ready quantizer must therefore define an explicit policy.

The default experiment should use:

\[
a_j=Q_{p_\ell}(X_j^{train}),
\qquad
b_j=Q_{p_u}(X_j^{train})
\]

with robust training-only percentiles such as:

\[
p_\ell\in\{0,0.001,0.005,0.01\}
\]

and:

\[
p_u=1-p_\ell.
\]

Values beyond the range may be saturated:

\[
x<a_j\Rightarrow Q(x)=q_{\min}
\]

\[
x>b_j\Rightarrow Q(x)=q_{\max}.
\]

Every run must report:

\[
r_{\mathrm{clip}}
=
\frac{\#\{x<a_j\lor x>b_j\}}
{\#\{x\}}.
\]

A configuration that appears accurate only because aggressive clipping suppresses extreme but valid financial moves must be treated with caution.

---

# 5. Noise-aware quantization

From STEP 03, suppose feature \(j\) has an estimated training-only noise scale:

\[
\widehat{\sigma}_{N,j}.
\]

If one wants quantization noise variance to be no greater than a fraction \(\beta\) of estimated observation noise:

\[
\sigma_{q,j}^2
\leq
\beta
\widehat{\sigma}_{N,j}^2,
\]

using the high-resolution approximation:

\[
\sigma_{q,j}^2
\approx
\frac{\Delta_j^2}{12}
\]

gives:

\[
\boxed{
\Delta_j
\leq
\sqrt{12\beta}\,
\widehat{\sigma}_{N,j}
}
\]

with:

\[
\beta>0.
\]

This produces a **noise-informed candidate resolution**, not an automatically correct one.

The experimental grid should include:

\[
\beta
\in
\left\{
\frac{1}{16},
\frac{1}{8},
\frac{1}{4},
\frac{1}{2},
1,
2
\right\}
\]

subject to computational budget.

Important caveat:

> Differences smaller than one instantaneous noise standard deviation are not necessarily devoid of predictive information. Weak signals may become detectable through temporal integration, multivariate conditioning, repeated observations or nonlinear structure.

Therefore this formula is a **proposal generator**, not a deletion rule.

---

# 6. Shannon capacity, mutual information and the required correction in interpretation

For an AWGN channel:

\[
C
=
B\log_2(1+\mathrm{SNR})
\]

bits/s.

At Nyquist sampling:

\[
f_s=2B,
\]

there are:

\[
\frac{C}{f_s}
=
\frac{1}{2}
\log_2(1+\mathrm{SNR})
\]

bits per real-valued sample.

Equivalently, for a Gaussian signal \(S\) observed through independent Gaussian additive noise \(N\):

\[
X=S+N,
\]

the mutual information is:

\[
I(S;X)
=
\frac12
\log_2
(1+\mathrm{SNR})
\]

bits per scalar sample.

This quantity may be useful as a **Gaussian/AWGN equivalent information benchmark**:

\[
b_{AWGN,j}
=
\frac12
\log_2
(1+\widehat{\mathrm{SNR}}_j).
\]

However:

> \[
> b_{AWGN}
> \]
> is **not** the number of ADC/quantizer bits that the feature “should use.”

It is an information-theoretic rate under a specific probabilistic model.

The research question is instead whether the empirically observed useful resolution:

\[
b_j^*
\]

has any reproducible relationship with:

\[
b_{AWGN,j}.
\]

That relationship must be tested, not assumed.

---

# 7. Rate–distortion theory as the deeper formal analogue

Shannon's rate–distortion framework asks:

> What minimum information rate is required to reproduce a source subject to a tolerated distortion?

For a distortion function:

\[
d(x,\hat{x})
\]

and allowable expected distortion:

\[
D,
\]

the rate–distortion function is:

\[
R(D).
\]

For a memoryless Gaussian source:

\[
X\sim\mathcal{N}(0,\sigma_X^2)
\]

with squared-error distortion:

\[
D
=
E[(X-\hat X)^2],
\]

the classic result is:

\[
R(D)
=
\frac12
\log_2
\left(
\frac{\sigma_X^2}{D}
\right)
\]

for:

\[
0<D<\sigma_X^2.
\]

This result is central to Step 4 [1], [2].

A noise-related experimental proposal is to set:

\[
D_j
=
\gamma
\widehat{\sigma}_{N,j}^2
\]

and evaluate whether:

\[
R_j(D_j)
\]

provides a useful predictor of empirically sufficient resolution.

Again, this is not asserted to be exact for financial series because:

- features are not generally Gaussian;
- samples are temporally dependent;
- sources are multivariate;
- the predictive distortion relevant to ML is not necessarily MSE reconstruction distortion;
- observation noise is not automatically the correct distortion threshold.

This is therefore a falsifiable bridge between rate–distortion theory and predictive preprocessing.

---

# 8. Reconstruction distortion is not predictive distortion

A key methodological distinction:

## 8.1. Representation distortion

\[
D_{repr}
=
E[(X-Q(X))^2].
\]

## 8.2. Predictive distortion

\[
D_{pred}
=
L(Y,\hat{Y}_{Q(X)})
-
L(Y,\hat{Y}_{X}).
\]

Two quantizers can have equal reconstruction MSE and very different forecasting effects.

Therefore Step 4 must optimize and report both:

1. fidelity to input data;
2. downstream forecasting utility.

The central objective is not:

\[
\min D_{repr}
\]

alone.

It is closer to:

\[
\max
\left(
\text{predictive generalization}
\right)
\]

subject to controlled representation rate/distortion.

---

# 9. Uniform quantization

Uniform scalar quantization is the mandatory control.

Candidate bit depths:

\[
b
\in
\{
1,2,3,4,5,6,8,10,12,16
\}.
\]

Depending on feature range and training cost, 1-bit may be included as a deliberately destructive stress condition rather than a serious candidate.

The primary curve is:

\[
\boxed{
P_j(b)
}
\]

where \(P\) is downstream forecasting performance.

A plateau candidate is:

\[
b_j^*
=
\min
\left\{
b:
P_j(b')
\approx P_j(b)
\;\forall b'>b
\right\}
\]

under a predeclared equivalence criterion.

This should preferably be defined statistically rather than visually.

---

# 10. Non-uniform quantization

Uniform quantization is inefficient when source density is strongly non-uniform.

Financial features frequently exhibit:

- strong central concentration;
- heavy tails;
- skewness;
- regime-dependent scale;
- nonlinear transforms;
- bounded indicators.

Therefore Step 4 must include non-uniform quantization.

---

# 11. Lloyd–Max quantization

Lloyd and Max derive optimal scalar quantizers under a distortion criterion, classically squared error [3], [4].

For reconstruction points:

\[
q_k
\]

and decision boundaries:

\[
t_k,
\]

the squared-error optimum satisfies two alternating conditions.

## 11.1. Nearest-neighbor partition

For squared error:

\[
t_k
=
\frac{q_k+q_{k+1}}{2}.
\]

## 11.2. Centroid reconstruction

\[
q_k
=
E[X\mid t_{k-1}<X\leq t_k].
\]

The algorithm alternates these operations until convergence.

All estimation must use:

\[
X^{train}
\]

only.

For validation/test:

- decision boundaries are frozen;
- reconstruction values are frozen;
- no distributional refit occurs.

Lloyd–Max is a crucial comparator because it answers:

> Does a distribution-adaptive quantizer outperform equal-width quantization at equal cardinality?

---

# 12. Quantile/equiprobable quantization

Define thresholds using training-only quantiles:

\[
t_k
=
F_{train}^{-1}
\left(
\frac{k}{L}
\right).
\]

This attempts to allocate approximately equal training probability mass to each cell.

Advantages:

- simple;
- robust to strongly non-uniform marginal distributions;
- prevents excessive state waste in low-density regions.

Risks:

- narrow bins near dense centers can become highly sensitive to distribution shift;
- tail states can become very wide;
- train/validation regime shifts can cause symbol-frequency collapse.

The following must be reported for each split:

\[
H(Q(X))
=
-\sum_k p_k\log_2 p_k
\]

and occupancy:

\[
o_k
=
\frac{n_k}{N}.
\]

Unused or near-empty bins are diagnostically important.

---

# 13. Companding belongs inside Step 4

A compander applies a nonlinear monotonic transformation:

\[
u=C(x)
\]

before uniform quantization.

The receiver/reconstruction can apply:

\[
\hat{x}
=
C^{-1}(Q(u)).
\]

This emulates non-uniform quantization while using a uniform quantizer in the compressed domain.

The ITU-T G.711 telephony standard is the classic practical example, using A-law and \(\mu\)-law PCM companding with 8-bit samples and an 8 kHz sampling rate [5].

---

# 14. Continuous \(\mu\)-law reference transform

For normalized:

\[
x\in[-1,1],
\]

a common continuous \(\mu\)-law form is:

\[
C_\mu(x)
=
\operatorname{sgn}(x)
\frac{
\ln(1+\mu |x|)
}{
\ln(1+\mu)
}.
\]

Classically:

\[
\mu=255.
\]

The inverse is:

\[
C_\mu^{-1}(y)
=
\operatorname{sgn}(y)
\frac{
(1+\mu)^{|y|}-1
}{
\mu
}.
\]

For ML experiments, \(\mu=255\) should be included as a standards-inspired baseline, but \(\mu\) may also be tuned using training/validation under the project governance rules.

---

# 15. Continuous A-law reference transform

For normalized:

\[
x\in[-1,1],
\]

the canonical continuous A-law form can be expressed as:

\[
C_A(x)
=
\operatorname{sgn}(x)
\begin{cases}
\dfrac{A|x|}{1+\ln A},
&
0\leq |x|<1/A
\\[8pt]
\dfrac{1+\ln(A|x|)}{1+\ln A},
&
1/A\leq |x|\leq 1.
\end{cases}
\]

with conventional:

\[
A\approx87.6.
\]

The actual G.711 implementation is defined by its standardized encoding tables/segments; the continuous expression is a conceptual analytical form.

---

# 16. Why companding may matter in financial features

Suppose a return feature is concentrated near zero with rare large excursions.

Uniform quantization allocates many reconstruction states to regions rarely occupied.

A compressive transform can allocate:

- more effective resolution near common small variations;
- coarser resolution in extreme tails.

This resembles the original telephony motivation: signal amplitude is not uniformly distributed, so uniform linear resolution may be wasteful.

The ML hypothesis is:

\[
P(Q(C(X)))
\geq
P(Q(X))
\]

at equal:

\[
L.
\]

But the opposite result is entirely plausible because tail events may carry unusually high predictive value.

Therefore a **tail-preservation diagnostic** is mandatory.

---

# 17. Tail-preservation diagnostics

For each feature and quantizer, measure performance separately for target observations conditioned on extreme input regimes, e.g.:

\[
|X_j|>Q_{0.95}(|X_j|)
\]

and:

\[
|X_j|>Q_{0.99}(|X_j|).
\]

Report:

- overall MAE/MSE;
- tail-conditioned MAE/MSE;
- directional metrics if applicable;
- clipping rate;
- number of unique tail symbols;
- reconstruction distortion in tails.

A quantizer that improves aggregate loss by destroying rare extreme-event information should not automatically be accepted.

---

# 18. Distinguish four different concepts

The agents must not conflate:

## 18.1. Scalar value quantization

\[
x_t\rightarrow q_t.
\]

## 18.2. Symbolic discretization

\[
x_t\rightarrow z_t\in\{1,\ldots,L\}.
\]

## 18.3. Token embedding

\[
z_t\rightarrow e(z_t)\in\mathbb{R}^d.
\]

## 18.4. Vector quantization of a latent representation

\[
z
\rightarrow
e_{k^*},
\qquad
k^*
=
\arg\min_k\|z-e_k\|.
\]

The initial Step 4 experiment should isolate **scalar input representation effects** before changing the neural architecture.

---

# 19. Mandatory two-mode implementation

Each scalar quantizer should support two downstream modes.

## 19.1. Reconstruction-value mode

The quantized symbol is converted back to its representative numeric value:

\[
x_t
\rightarrow
Q(x_t)
\in\mathbb{R}.
\]

The existing model architecture remains unchanged.

Purpose:

> isolate the effect of reduced input resolution.

## 19.2. Token mode

The quantized cell index is treated as a categorical token:

\[
x_t
\rightarrow
z_t
\rightarrow
E(z_t).
\]

Purpose:

> test whether categorical/tokenized representation is advantageous.

Token mode changes the model input mechanism and must therefore be treated as a distinct architectural experiment.

---

# 20. State of the art: direct evidence that time-series quantization/tokenization is viable

The literature review conducted for this document shows that the idea is already a meaningful research direction rather than a purely speculative analogy.

---

# 21. Rabanser et al.: discretization can improve neural forecasting

Rabanser et al. systematically evaluated binning/discretization of time-series inputs/outputs with feed-forward, recurrent and convolutional neural models [6].

Their reported empirical conclusion was that binning often improved forecasting relative to normalized real-valued inputs, while the exact binning strategy was often less important than the existence of discretization itself.

Implication for this project:

> Quantization/discretization should be treated as a serious baseline, but their result does not establish a noise-aware optimum or prove transfer to financial data.

Our protocol extends the question by explicitly introducing:

- SNR;
- denoising;
- rate/distortion;
- per-feature resolution;
- multivariate bit allocation;
- multi-branch integration.

---

# 22. Chronos: scaling + quantization as a time-series language

Chronos explicitly converts numerical time series into a fixed token vocabulary using scaling and quantization, then trains language-model architectures over the resulting sequence [7].

The current official implementation exposes a tokenizer class named:

`MeanScaleUniformBins`

in the public `amazon-science/chronos-forecasting` repository.

This is directly reusable as:

- an implementation reference;
- a uniform-bin tokenizer baseline;
- a comparison point for preprocessing semantics.

The repository is Apache-2.0 licensed at the time of this review.

Chronos does **not** by itself answer the project's noise-floor or per-feature rate-allocation question.

---

# 23. TOTEM: learned discrete time-series codebooks

TOTEM learns discrete time-series representations using a VQ-VAE-style tokenizer and applies the learned tokenization across forecasting, imputation and anomaly-detection tasks [8].

The study reports nearly 500 experiments across multiple datasets/tasks and provides an official public implementation.

Relevance:

- proves that learned discrete time-series representations can be general-purpose;
- highly relevant to the project's modular encoder architecture;
- especially relevant to the later extension:
  \[
  E(X)\rightarrow Q_V(E(X)).
  \]

However, this should be treated as **Step 4B/latent extension**, after scalar input quantization is isolated.

Before code reuse, agents should verify the current repository license and dependency compatibility.

---

# 24. TimeVQVAE: vector quantization in time-frequency representations

TimeVQVAE uses vector quantization for time-series generation and explicitly models low- and high-frequency latent spaces [9].

This is especially relevant because the project already uses:

- wavelets;
- multitaper;
- frequency-related feature branches;
- generative feature-extractor experiments.

The official public implementation is available and reported as MIT-licensed at the time of review.

This provides reusable prior art for later:

\[
\text{time/frequency branch}
\rightarrow
\text{VQ latent}
\rightarrow
\text{core}.
\]

It is **not** a substitute for the scalar Step 4 experiment.

---

# 25. WaveToken: strongest direct analogue found

WaveToken, published at ICML 2025, is particularly important [10].

Its pipeline is:

\[
\text{scaling}
\rightarrow
\text{wavelet decomposition}
\rightarrow
\text{thresholding}
\rightarrow
\text{quantization}
\rightarrow
\text{autoregressive token forecasting}.
\]

This is extremely close to the direction independently reached in this project:

\[
\text{noise/threshold estimation}
\rightarrow
\text{multi-domain representation}
\rightarrow
\text{quantization}.
\]

WaveToken reports:

- evaluation over 42 datasets;
- a vocabulary of 1024 tokens;
- strong in-domain and zero-shot forecasting;
- good handling of trends, spikes and non-stationary frequency structure.

Consequences:

1. **Do not reinvent wavelet coefficient tokenization from zero.**
2. Use WaveToken's design as a formal comparator/reference.
3. The project's differentiator should be:
   - noise-aware thresholding/quantization;
   - SNR-to-resolution hypotheses;
   - per-feature allocation;
   - financial multi-branch integration;
   - explicit falsification analysis.

At the time of this search, an obvious official standalone WaveToken repository was not identified in the primary publication page; agents should verify current code availability before implementation.

---

# 26. Sparse-VQ Transformer: adjacent evidence

Sparse-VQ Transformer proposes vector quantization inside a forecasting Transformer and reports improvements on multiple benchmarks [11].

Its quantization operates primarily as an architectural latent mechanism rather than raw scalar preprocessing.

Therefore it is relevant to:

- later core/latent quantization;
- evidence that VQ can suppress noise/redundancy;
- architecture-level ablations.

It should not be used as evidence that scalar financial feature quantization is automatically beneficial.

---

# 27. SAX: older symbolic time-series precedent

Symbolic Aggregate approXimation (SAX) converts real-valued time series into a symbolic representation [12].

Although originally motivated heavily by time-series mining and efficient representation rather than modern deep forecasting, SAX demonstrates that:

\[
\text{continuous series}
\rightarrow
\text{discrete alphabet}
\]

is a mature idea.

SAX should be included as an inexpensive symbolic baseline when appropriate.

---

# 28. Baseline forecasting models to reuse

The experimental protocol should avoid proving the effect only with the project's proprietary architecture.

At least two standard forecasting baselines should be used.

## 28.1. DLinear

DLinear/LTSF-Linear is a simple and highly reproducible forecasting family [13].

Advantages:

- low compute;
- public official code;
- public benchmark scripts;
- Apache-2.0 repository;
- suitable for large quantization grids.

## 28.2. PatchTST

PatchTST is a well-known patch-based Transformer forecasting model [14].

Advantages:

- materially different inductive bias from DLinear;
- public reference implementation;
- widely used benchmark datasets.

The quantization effect should ideally survive both a simple model and a stronger sequence model before being attributed to a general preprocessing principle.

---

# 29. Reuse matrix

| Component | Existing prior art | Reuse recommendation |
|---|---|---|
| Uniform scalar/token quantization | Chronos `MeanScaleUniformBins` [7] | Inspect and adapt as benchmark/reference |
| Generic discretization experiments | Rabanser et al. [6] | Reproduce selected binning baselines |
| Learned discrete tokenizer | TOTEM [8] | Later latent/token branch experiment |
| Time-frequency VQ | TimeVQVAE [9] | Later wavelet/frequency latent branch |
| Wavelet threshold + quantization | WaveToken [10] | High-priority methodological comparator |
| Uniform/non-uniform quantization theory | Gray & Neuhoff [2], Lloyd [3], Max [4] | Use as theoretical foundation |
| Companding | ITU-T G.711 [5] | Use A-law/\(\mu\)-law controls |
| Simple benchmark predictor | DLinear [13] | Primary high-throughput baseline |
| Transformer benchmark predictor | PatchTST [14] | Secondary architecture baseline |
| Symbolic baseline | SAX [12] | Optional low-cost baseline |

---

# 30. Experimental data governance

For project datasets:

\[
4\text{ years train}
+
1\text{ year validation}
+
1\text{ year test}.
\]

Periodicities:

- 1 h;
- 4 h.

All learned preprocessing parameters must originate in training only.

This includes:

- means;
- standard deviations;
- robust scales;
- clipping bounds;
- min/max;
- quantiles;
- companding scale;
- \(\mu\);
- \(A\) if tuned;
- Lloyd–Max centroids;
- Lloyd–Max boundaries;
- number of levels if treated as preprocessing hyperparameter;
- SNR estimates;
- noise thresholds;
- codebook initialization if learned;
- normalization statistics.

Validation is used for:

- model selection;
- quantizer configuration comparison;
- early stopping;
- branch selection.

Test is reserved for the **final selected configuration**.

Performing hundreds of quantizer comparisons on test would convert test into a validation set and invalidate the claimed generalization estimate.

---

# 31. Causality

For online deployment, the quantizer at time \(t\) must be representable as:

\[
Q_t
=
Q(
x_t;
\theta_{train}
)
\]

or, if adaptation is explicitly allowed:

\[
Q_t
=
Q(
x_t;
\theta_t
)
\]

where:

\[
\theta_t
=
g(x_1,\ldots,x_t).
\]

No future data may determine:

- cell boundaries;
- dynamic range;
- scaling;
- local volatility;
- regime-conditioned codebooks.

An adaptive quantizer is acceptable only if its update is strictly causal and independently audited.

---

# 32. Phase 0 — Reproduce published benchmark behavior

Before the project's novel hypotheses are tested:

1. select a public dataset;
2. reproduce DLinear baseline to an acceptable tolerance against published/reference results;
3. optionally reproduce PatchTST;
4. log:
   - software versions;
   - data split;
   - seed;
   - metrics;
   - preprocessing;
   - sequence length;
   - horizon.

This prevents a broken benchmark implementation from contaminating the quantization conclusions.

---

# 33. Phase 1 — Raw continuous baseline

For every dataset/model:

\[
X_{raw}
\rightarrow M
\rightarrow \hat{Y}.
\]

Use the same normalization conventions as the reference model.

Record:

- MAE;
- MSE/RMSE;
- \(R^2\), where meaningful;
- per-horizon metrics;
- training time;
- inference time;
- memory;
- parameter count;
- prediction loss sequence.

This run is the reference for all quantization comparisons.

---

# 34. Phase 2 — Uniform scalar quantization sweep

Run:

\[
b
\in
\{
1,2,3,4,5,6,8,10,12,16
\}.
\]

For every \(b\), freeze:

- model architecture;
- training schedule;
- sequence length;
- target;
- feature set;
- seeds.

Only input resolution changes.

Output:

\[
P(b).
\]

Primary question:

> Does forecasting performance plateau before floating-point resolution?

---

# 35. Phase 3 — Distribution-aware quantization

Compare at equal \(L\):

1. equal-width uniform;
2. quantile/equiprobable;
3. Lloyd–Max;
4. optional robust-k-means scalar quantizer;
5. optional SAX-compatible symbolic representation.

The primary comparison is:

\[
P(Q_{uniform})
\]

vs.

\[
P(Q_{quantile})
\]

vs.

\[
P(Q_{LloydMax}).
\]

---

# 36. Phase 4 — Companding

For the same \(b\), compare:

\[
Q(X)
\]

against:

\[
Q(C_{\mu}(X))
\]

and:

\[
Q(C_A(X)).
\]

Include at minimum:

- \(\mu=255\);
- \(A=87.6\);
- training-only scaling to normalized dynamic range.

Optional train/validation grid:

\[
\mu
\in
\{15,31,63,127,255,511\}.
\]

A-law should initially use the conventional value unless there is a strong reason to tune it.

---

# 37. Phase 5 — SNR × resolution surface

Reuse STEP 03's synthetic noise protocol.

For:

\[
s
\in
\{
\infty,40,30,20,15,10,5,0
\}\text{ dB}
\]

and:

\[
b
\in
\{
2,3,4,5,6,8,10,12,16
\}
\]

evaluate:

\[
\boxed{
P=P(\mathrm{SNR},b)
}.
\]

This is one of the central experiments.

Expected—but falsable—behavior:

\[
\frac{\partial b^*}{\partial \mathrm{SNR}}
>0.
\]

That is:

> higher SNR should support finer useful resolution.

If no relationship appears, the noise-resolution hypothesis is weakened.

---

# 38. Phase 6 — Denoising × quantization

For every representative noise level:

\[
X^{noisy}
\]

compute:

\[
\hat S=D(X^{noisy})
\]

and test:

\[
Q_b(X^{noisy})
\]

versus:

\[
Q_b(\hat S).
\]

Produce:

\[
P_{raw}(\mathrm{SNR},b)
\]

and:

\[
P_{denoised}(\mathrm{SNR},b).
\]

A specific falsable prediction is:

\[
b^*_{denoised}
\geq
b^*_{noisy}
\]

because removing noise may make finer distinctions informative.

This prediction must not be enforced in the analysis.

---

# 39. Phase 7 — Noise-derived quantizer

From STEP 03:

\[
\widehat{\sigma}_{N,j}.
\]

Construct:

\[
\Delta_j(\beta)
=
\sqrt{12\beta}\,
\widehat{\sigma}_{N,j}.
\]

Derive the resulting number of levels:

\[
L_j
\approx
\frac{b_j-a_j}{\Delta_j}
\]

and effective bit count:

\[
b_j^{noise}
=
\left\lceil
\log_2 L_j
\right\rceil.
\]

Compare this automatic rule against:

- best manually swept validation bit depth;
- uniform fixed bit depths;
- Lloyd–Max;
- quantile quantizer.

Primary criterion:

> Can SNR/noise statistics predict a near-optimal quantizer without exhaustive search?

---

# 40. Phase 8 — Gaussian-equivalent information predictor

Compute:

\[
b_{AWGN,j}
=
\frac12
\log_2
(1+\widehat{\mathrm{SNR}}_j).
\]

Do **not** directly use this as ADC bit depth.

Instead test whether:

\[
b_{AWGN,j}
\]

is correlated with:

\[
b_j^*.
\]

Suitable analyses:

- Spearman rank correlation;
- robust regression;
- feature-family stratification;
- confidence intervals via block-aware resampling where applicable.

Falsification:

> If no stable association exists across datasets/features/models, reject its usefulness as a practical resolution estimator.

---

# 41. Phase 9 — Rate–distortion-derived candidate resolution

For each feature, define candidate distortion targets:

\[
D_j(\gamma)
=
\gamma
\widehat{\sigma}_{N,j}^2.
\]

For a Gaussian approximation:

\[
R_j(D)
=
\frac12
\log_2
\frac{\widehat{\sigma}_{X,j}^2}{D_j}.
\]

Use:

\[
R_j(D)
\]

only as a theoretical candidate.

Compare:

\[
R_j(D)
\]

with:

\[
b_j^*.
\]

The experiment asks:

> Does a rate–distortion estimate derived from signal/noise scales have predictive value for the useful input resolution of the downstream model?

---

# 42. Phase 10 — Single-feature sensitivity

For each feature \(j\):

- keep all other features continuous;
- quantize only \(X_j\).

Measure:

\[
\Delta P_j(b)
=
P(X_1,\ldots,Q_b(X_j),\ldots,X_m)
-
P(X_1,\ldots,X_j,\ldots,X_m).
\]

This identifies:

- features highly sensitive to resolution;
- features tolerant to aggressive quantization;
- features potentially redundant;
- features for which tails matter.

---

# 43. Phase 11 — Feature-family quantization

Instead of one feature at a time, group by semantic family:

- raw price/OHLC;
- returns/differences;
- technical indicators;
- volatility;
- macro/fundamental;
- calendar;
- external assets;
- decomposition components;
- frequency-domain features.

Assign a shared or family-specific resolution:

\[
b_k.
\]

This maps directly to the project's multi-branch design.

---

# 44. Phase 12 — Multivariate bit allocation

Let:

\[
\mathbf{b}
=
(b_1,\ldots,b_m)
\]

and define total nominal representation budget:

\[
B_{total}
=
\sum_{j=1}^m b_j.
\]

Study:

\[
\max_{\mathbf b}
P(Q_{\mathbf b}(X))
\]

subject to:

\[
\sum_j b_j\leq B_{\max}.
\]

Compare:

## A. Equal allocation

\[
b_j=b.
\]

## B. SNR-proportional allocation

\[
b_j=f(\widehat{\mathrm{SNR}}_j).
\]

## C. Distortion-derived allocation

\[
b_j=f(R_j(D_j)).
\]

## D. Validation-optimized allocation

Constrained search/optimizer using validation only.

The goal is not merely compression; it is to discover whether representational capacity should be allocated unequally across features.

---

# 45. Information-efficiency metrics

In addition to forecasting loss, report:

## 45.1. Nominal bits per timestamp

\[
B_{nominal}
=
\sum_j\log_2 L_j.
\]

## 45.2. Empirical symbol entropy

\[
H_j
=
-\sum_k p_{j,k}\log_2p_{j,k}.
\]

## 45.3. Joint nominal rate

\[
H_{sum}
=
\sum_jH_j.
\]

This is not equal to joint entropy when features are dependent, but is useful diagnostically.

## 45.4. Occupancy ratio

\[
r_{occ,j}
=
\frac{
\#\{\text{used bins}\}
}{
L_j
}.
\]

## 45.5. Input reconstruction distortion

\[
D_{repr,j}
=
E[(X_j-\hat X_j)^2].
\]

## 45.6. Quantization SQNR proxy

\[
\mathrm{SQNR}_j
=
10\log_{10}
\frac{
E[X_j^2]
}{
E[(X_j-\hat X_j)^2]
}.
\]

All scale-dependent metrics must be interpreted after defining normalization precisely.

---

# 46. Predictive-efficiency frontier

For each configuration \(c\), form:

\[
(R_c,P_c)
\]

where:

- \(R_c\) is nominal or empirical representation rate;
- \(P_c\) is predictive performance.

Compute the empirical Pareto frontier:

\[
\boxed{
\text{representation rate}
\leftrightarrow
\text{forecasting quality}
}.
\]

A quantized representation is especially compelling if it achieves:

\[
R_c<R_{raw}
\]

with:

\[
P_c\geq P_{raw}.
\]

Even when actual disk/network compression is not the objective, this provides a direct information-efficiency interpretation.

---

# 47. Raw vs quantized vs parallel representation

The project's modular architecture should exploit parallel representations.

Minimum conditions:

## A. Raw only

\[
X.
\]

## B. Quantized only

\[
Q(X).
\]

## C. Raw + quantized

\[
[X,Q(X)].
\]

## D. Denoised only

\[
D(X).
\]

## E. Quantized denoised

\[
Q(D(X)).
\]

## F. Raw + denoised + quantized

\[
[X,D(X),Q(D(X))].
\]

The central question is not only whether quantization replaces raw inputs, but whether it provides a complementary inductive representation.

---

# 48. Multi-branch integration

For branch \(k\):

\[
X_k
\rightarrow
T_k(X_k)
\rightarrow
E_k
\rightarrow
Z_k.
\]

Possible \(T_k\):

- identity;
- denoising;
- uniform quantization;
- companding + quantization;
- Lloyd–Max;
- quantile quantization;
- wavelet + threshold + quantization;
- later VQ latent.

Core:

\[
Z_{core}
=
C(Z_1,\ldots,Z_K).
\]

Predictive heads:

\[
\hat Y_h
=
H_h(Z_{core}).
\]

This aligns with the existing design in which preprocessing branches feed a consolidation core and downstream predictive heads.

---

# 49. Core/head architecture must remain controlled

When testing preprocessing:

> Do not simultaneously change the downstream core or predictive heads unless the experimental question explicitly targets an interaction.

Otherwise:

\[
\Delta P
\]

cannot be attributed to quantization.

Initial benchmark experiments should freeze:

- Conv1D configuration;
- BiLSTM configuration;
- Bayesian layer configuration;
- dense bias/head structure;
- optimizer;
- learning-rate schedule;
- early stopping;
- target definitions;
- horizons.

---

# 50. Hypotheses — final falsable set

## H4.1 — Finite useful resolution

There exists a finite:

\[
b^*
\]

such that:

\[
P(b>b^*)
\approx P(b^*).
\]

**Falsified if:** predictive performance continues to improve materially throughout the tested resolution range.

---

## H4.2 — Over-resolution can be neutral or harmful

For sufficiently high \(b\), additional resolution may not improve generalization and may increase model sensitivity to noise.

**Falsified if:** higher resolution is monotonically and materially superior across datasets/models.

---

## H4.3 — Resolution depends on SNR

\[
b^*=f(\mathrm{SNR})
\]

with:

\[
\frac{\partial b^*}{\partial \mathrm{SNR}}>0
\]

on average.

**Falsified if:** no stable positive relation exists.

---

## H4.4 — Noise-aware quantization is competitive

A quantizer derived from:

\[
\widehat{\sigma}_N
\]

achieves performance close to the validation-optimal quantizer without exhaustive search.

**Falsified if:** noise-derived settings systematically underperform ordinary fixed-bit baselines.

---

## H4.5 — Non-uniform quantization benefits non-uniform sources

At equal number of levels:

\[
P(Q_{nonuniform}(X))
\geq
P(Q_{uniform}(X))
\]

for sufficiently non-uniform feature distributions.

**Falsified if:** uniform consistently matches or dominates.

---

## H4.6 — Companding improves resolution allocation for concentrated distributions

At equal \(L\):

\[
P(Q(C(X)))
>
P(Q(X))
\]

for selected strongly concentrated/heavy-tailed feature families.

**Falsified if:** companding provides no reproducible advantage or damages tail forecasting.

---

## H4.7 — Denoising increases useful resolution

\[
b^*_{denoised}
\geq
b^*_{noisy}
\]

under controlled additive-noise experiments.

**Falsified if:** no relationship is observed.

---

## H4.8 — Per-feature allocation is superior to uniform allocation

At equal total nominal budget:

\[
P(\mathbf b_{adaptive})
>
P(\mathbf b_{uniform}).
\]

**Falsified if:** equal allocation is consistently as good or better.

---

## H4.9 — Quantized representation can be complementary

\[
P([X,Q(X)])
>
\max(P(X),P(Q(X))).
\]

**Falsified if:** the parallel representation never provides reproducible gain.

---

## H4.10 — AWGN-equivalent information predicts useful resolution

\[
b_{AWGN}
=
\frac12\log_2(1+\widehat{\mathrm{SNR}})
\]

has a stable positive association with empirical:

\[
b^*.
\]

**Falsified if:** association is absent or unstable across features/datasets/models.

---

## H4.11 — Rate–distortion proxy predicts useful resolution

A training-only:

\[
R(D)
\]

using noise-informed distortion has measurable predictive value for:

\[
b^*.
\]

**Falsified if:** the rate–distortion proxy has no useful relationship with empirical resolution.

---

## H4.12 — The effect generalizes across architectures

The sign of the best quantization effect should be reproducible in at least:

- DLinear;
- PatchTST or equivalent modern model;
- the project's modular predictor.

**Falsified if:** apparent gains occur only in one architecture.

---

# 51. Statistical evaluation

Forecast errors are temporally dependent; ordinary iid t-tests are insufficient.

Recommended methods:

## 51.1. Diebold–Mariano

Use the Diebold–Mariano framework for comparing predictive loss sequences where appropriate [15].

For two methods:

\[
d_t
=
L(e_{1,t})-L(e_{2,t}).
\]

Test:

\[
H_0:E[d_t]=0.
\]

---

## 51.2. Block-aware confidence intervals

Use stationary/bootstrap methods that preserve temporal dependence [16].

Do not bootstrap individual timestamps independently.

---

## 51.3. Multiple comparisons

The quantization grid can produce many hypotheses.

Use:

- predeclared primary comparisons;
- familywise/FDR correction where exploratory comparisons are large;
- Benjamini–Hochberg as one possible FDR procedure [17].

The selected correction policy must be declared before inspecting final test results.

---

# 52. Seed protocol

For stochastic neural models:

\[
s\in\{s_1,\ldots,s_K\}
\]

with sufficiently many seeds for confidence assessment.

Report:

- median;
- mean;
- standard deviation;
- confidence interval;
- worst seed;
- best seed.

A preprocessing improvement that exists only on one seed is not considered established.

---

# 53. Validation-selection protocol

For each candidate quantizer:

1. fit preprocessing statistics on training;
2. train model on training;
3. use validation for early stopping/configuration ranking;
4. freeze final selected preprocessing/model design;
5. perform final test evaluation once.

If retraining on train+validation is desired after selection, that must be defined in advance and applied consistently to all final comparator methods.

---

# 54. Public benchmark protocol

At least one benchmark family should use established splits and published metrics.

Recommended:

- ETT;
- Electricity;
- Traffic;
- Weather;
- Exchange Rate.

Use reference model repositories wherever practical rather than rebuilding benchmark pipelines without necessity.

---

# 55. Financial-data protocol

After public validation, repeat on project data:

- EURUSD;
- ETHUSDT;
- other available assets;
- 1 h;
- 4 h;
- technical features;
- fundamental features;
- cross-asset/exogenous inputs.

Quantizers should be fitted separately where feature semantics differ strongly.

A single universal binning rule for all features is not assumed.

---

# 56. Distribution-shift diagnostics

Quantization is particularly vulnerable to shift.

For train/validation/test report:

- bin occupancy distribution;
- Jensen–Shannon divergence of symbol distributions;
- clipping rate;
- missing-bin rate;
- new-extreme rate;
- median/scale changes;
- tail occupancy.

A quantizer whose validation/test symbol distribution collapses is not robust even if training distortion is excellent.

---

# 57. Regime dependence

A future extension may define:

\[
Q(x\mid r_t)
\]

where \(r_t\) is a causally inferred regime.

However, regime-dependent quantization must **not** be introduced into the primary experiment because it mixes two hypotheses.

Primary Step 4 uses a fixed training-derived quantizer.

Regime-adaptive quantization belongs in an extension after the base effect is established.

---

# 58. Do not confuse preprocessing quantization with model compression

This research is primarily about:

\[
\text{input information representation}.
\]

It is not initially about:

- INT8 neural weights;
- post-training model quantization;
- quantization-aware training of weights;
- GPU inference compression;
- mixed precision.

Those are separate engineering problems.

They may later interact with Step 4 but should not contaminate the initial causal question.

---

# 59. Learned vector quantization — deferred extension

After scalar input quantization has been characterized, investigate:

\[
Z=E(X)
\]

followed by:

\[
Q_V(Z).
\]

This is directly motivated by:

- TOTEM [8];
- TimeVQVAE [9];
- Sparse-VQ [11].

Questions:

- does a discrete latent codebook reduce nuisance variability?
- does it improve branch interoperability?
- does it improve transfer across assets?
- does it provide a compact state alphabet for the core?
- does it reduce overfitting?

This is **not** required to close initial Step 4.

---

# 60. Wavelet-domain quantization — high-priority extension

Because the project already has wavelet/multitaper infrastructure, WaveToken [10] makes the following experiment especially important:

\[
X
\rightarrow
W(X)
\rightarrow
\text{threshold}
\rightarrow
Q(W(X))
\rightarrow
\text{branch encoder}.
\]

Compare against:

\[
Q(X)
\]

and:

\[
W(Q(X)).
\]

Order matters.

The hypotheses:

\[
Q(W(X))
\neq
W(Q(X))
\]

and one may be materially superior.

WaveToken suggests that decomposition before quantization is particularly promising.

---

# 61. Failure modes and risks

## 61.1. Leakage

Quantile/Lloyd-Max thresholds fitted on the full dataset invalidate evaluation.

## 61.2. Tail destruction

Companding or clipping may remove rare event structure.

## 61.3. False noise assumption

Small-amplitude variations may be predictively useful in aggregate.

## 61.4. Distribution shift

A train-optimal quantizer may be badly calibrated in future regimes.

## 61.5. Architecture confounding

Token embeddings change model architecture and cannot be compared as pure preprocessing.

## 61.6. Too many experiments

The grid can explode combinatorially.

Use staged gates.

## 61.7. Reconstruction fallacy

Lower reconstruction error does not guarantee better forecasting.

## 61.8. Quantization artifacts

Coarse bins may introduce artificial step transitions or harmonics.

## 61.9. Target leakage via feature construction

All upstream denoising/decomposition rules must remain causal.

---

# 62. Recommended staged gates

## Gate 4A — Baseline validity

Public model reproduces known behavior.

## Gate 4B — Uniform quantization signal

At least one nontrivial bit depth produces statistically comparable or improved validation performance.

If all coarse quantization harms substantially, deeper quantizer searches may be deprioritized.

## Gate 4C — SNR relation

Evidence exists that useful resolution changes with controlled noise.

## Gate 4D — Non-uniform/companding value

At least one adaptive representation improves the rate/performance frontier.

## Gate 4E — Generalization

Effect survives public dataset and project data.

## Gate 4F — Architecture transfer

Effect survives at least two model families.

## Gate 4G — Test confirmation

One preselected final candidate is evaluated on test.

---

# 63. Minimum experiment matrix

| ID | Input | Quantizer | Companding | Noise | Denoising | Architecture |
|---|---|---|---|---|---|---|
| Q00 | Raw | None | None | Native | No | DLinear |
| Q01 | Raw | Uniform | None | Native | No | DLinear |
| Q02 | Raw | Quantile | None | Native | No | DLinear |
| Q03 | Raw | Lloyd–Max | None | Native | No | DLinear |
| Q04 | Raw | Uniform | \(\mu\)-law | Native | No | DLinear |
| Q05 | Raw | Uniform | A-law | Native | No | DLinear |
| Q06 | Raw | Uniform | None | Controlled | No | DLinear |
| Q07 | Denoised | Uniform | None | Controlled | Yes | DLinear |
| Q08 | Raw | Uniform | None | Native | No | PatchTST |
| Q09 | Raw | Best Step-4 | Best | Native | Best | Project model |
| Q10 | Raw + Quantized | Best Step-4 | Best | Native | Optional | Project multi-branch |

This is the minimum; not the full grid.

---

# 64. Deliverables

Agents implementing Step 4 should generate:

1. `quantizer_spec.json`
2. `training_only_fit_manifest.json`
3. `quantizer_boundaries.parquet`
4. `symbol_occupancy.parquet`
5. `distortion_metrics.parquet`
6. `forecast_metrics.parquet`
7. `noise_resolution_surface.parquet`
8. `bit_allocation_results.parquet`
9. `statistical_tests.json`
10. `audit_report.md`
11. reproducibility manifest with:
    - dataset hash;
    - split dates;
    - git commit;
    - environment;
    - seeds;
    - model configuration.

Names are recommendations, not mandatory repository contracts.

---

# 65. Acceptance criteria

Step 4 is empirically supported only if:

1. a finite-resolution plateau can be demonstrated or rigorously rejected;
2. quantization effects are reproducible across seeds;
3. no preprocessing parameter uses validation/test improperly;
4. quantizer clipping and occupancy are audited;
5. tail behavior is reported;
6. raw vs quantized comparisons preserve downstream architecture where required;
7. SNR/noise relationships are tested explicitly;
8. at least one public benchmark is included;
9. at least two model architectures are included before claiming generality;
10. final conclusions are based on validation-selected configurations and a held-out test confirmation.

---

# 66. Possible conclusions

The experiment may legitimately conclude any of the following.

## Result A — Quantization is beneficial

\[
P(Q(X))>P(X).
\]

## Result B — Quantization is neutral but cheaper

\[
P(Q(X))\approx P(X)
\]

with much lower representation entropy/rate.

## Result C — Quantization is useful only as parallel representation

\[
P([X,Q(X)])>P(X).
\]

## Result D — Denoising is necessary before quantization

\[
P(Q(D(X)))>P(Q(X)).
\]

## Result E — Non-uniform quantization is required

\[
P(Q_{nonuniform}(X))
>
P(Q_{uniform}(X)).
\]

## Result F — Feature-specific resolution matters

\[
P(Q_{\mathbf b}(X))
>
P(Q_b(X)).
\]

## Result G — Quantization is harmful

\[
P(Q(X))<P(X).
\]

Result G is scientifically useful and would falsify the proposed preprocessing benefit.

---

# 67. Proposed interpretation if hypotheses survive

If the main hypotheses survive, the communications-to-ML analogy becomes:

\[
\boxed{
\text{estimated noise floor}
\rightarrow
\text{effective distinguishable resolution}
\rightarrow
\text{rate/distortion choice}
\rightarrow
\text{non-uniform amplitude representation}
\rightarrow
\text{model input}
}
\]

Rather than requiring a model to infer all useful invariances from high-precision raw numbers, the preprocessing layer would present a representation whose resolution is matched to the statistically defensible information content of each feature.

This would be directly analogous in spirit—not identical mathematically—to the way communications systems allocate finite representation levels according to source/channel constraints.

---

# 68. State-of-the-art conclusion

The literature review changes the project plan in several useful ways.

## 68.1. Already established enough not to reinvent

- scalar quantization theory;
- Lloyd–Max optimization;
- A-law/\(\mu\)-law companding;
- symbolic time-series discretization;
- uniform tokenization of time series;
- learned VQ time-series tokenizers;
- wavelet-domain quantization/tokenization.

## 68.2. Project-specific research gap worth testing

The combination:

\[
\boxed{
\text{measured SNR/noise}
\rightarrow
\text{falsifiable effective-resolution prediction}
\rightarrow
\text{feature-specific quantization/companding}
\rightarrow
\text{multi-branch financial forecasting}
}
\]

was not identified in the reviewed work as an already standardized solution.

That is the part that should be experimentally investigated rather than assumed.

---

# 69. Recommendation for implementation order

1. DLinear + one public benchmark + raw baseline.
2. Uniform \(b\)-sweep.
3. Controlled SNR × \(b\) surface.
4. Quantile and Lloyd–Max.
5. A-law/\(\mu\)-law companding.
6. Noise-derived \(\Delta_j\).
7. Denoising × quantization.
8. Feature-specific sensitivity.
9. Family-level bit allocation.
10. PatchTST replication.
11. Project multi-branch model.
12. Wavelet-domain quantization.
13. Latent VQ/TOTEM/TimeVQVAE-inspired extension.

This ordering maximizes falsifiability while minimizing premature engineering complexity.

---

# 70. Audit checklist

Before approval, agents should verify:

- [ ] Equations distinguish channel capacity from quantizer bit depth.
- [ ] Gaussian rate–distortion assumptions are explicitly limited.
- [ ] \(\Delta^2/12\) is labelled as a high-resolution approximation.
- [ ] Companding is placed inside the quantization step.
- [ ] G.711 is used only as communications precedent, not as financial prescription.
- [ ] All fitted preprocessing statistics are training-only.
- [ ] Validation is not used to refit source statistics.
- [ ] Test is not repeatedly inspected.
- [ ] Quantizer saturation policy is explicit.
- [ ] Clipping rates are logged.
- [ ] Tail-event metrics are logged.
- [ ] Reconstruction and predictive distortion are separated.
- [ ] Raw and quantized representations are compared under identical architectures.
- [ ] Tokenization is treated as a distinct architectural condition.
- [ ] Public benchmark baseline is reproduced first.
- [ ] Multiple seeds are used.
- [ ] Forecast comparisons account for temporal dependence.
- [ ] Multiple-comparison policy is predeclared.
- [ ] SNR × bit-depth surface is produced.
- [ ] Noise-aware automatic resolution is compared against exhaustive validation search.
- [ ] Multi-branch tests include raw + transformed representations.
- [ ] WaveToken is reviewed before implementing wavelet tokenization from scratch.
- [ ] Chronos/TOTEM/TimeVQVAE code licenses are verified before reuse.
- [ ] Conclusions permit full falsification of the hypothesis.

---

# 71. References — IEEE style

[1] C. E. Shannon, “Coding Theorems for a Discrete Source With a Fidelity Criterion,” in *IRE National Convention Record*, Part 4, pp. 142–163, 1959. Available: https://mast.queensu.ca/~math474/shannon59.pdf

[2] R. M. Gray and D. L. Neuhoff, “Quantization,” *IEEE Transactions on Information Theory*, vol. 44, no. 6, pp. 2325–2383, Oct. 1998, doi: 10.1109/18.720541. Available: https://doi.org/10.1109/18.720541

[3] S. P. Lloyd, “Least Squares Quantization in PCM,” *IEEE Transactions on Information Theory*, vol. 28, no. 2, pp. 129–137, Mar. 1982, doi: 10.1109/TIT.1982.1056489. Available: https://doi.org/10.1109/TIT.1982.1056489

[4] J. Max, “Quantizing for Minimum Distortion,” *IRE Transactions on Information Theory*, vol. 6, no. 1, pp. 7–12, Mar. 1960, doi: 10.1109/TIT.1960.1057548. Available: https://doi.org/10.1109/TIT.1960.1057548

[5] ITU-T, “Pulse Code Modulation (PCM) of Voice Frequencies,” Recommendation G.711, 1988. Available: https://www.itu.int/rec/T-REC-G.711

[6] S. Rabanser, T. Januschowski, V. Flunkert, D. Salinas, and J. Gasthaus, “The Effectiveness of Discretization in Forecasting: An Empirical Study on Neural Time Series Models,” arXiv:2005.10111, 2020. Available: https://arxiv.org/abs/2005.10111

[7] A. F. Ansari *et al.*, “Chronos: Learning the Language of Time Series,” *Transactions on Machine Learning Research*, 2024. Available: https://openreview.net/forum?id=gerNCVqqtR ; official implementation: https://github.com/amazon-science/chronos-forecasting

[8] S. J. Talukder, Y. Yue, and G. Gkioxari, “TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis,” *Transactions on Machine Learning Research*, 2024. Available: https://openreview.net/forum?id=QlTLkH6xRC ; official implementation: https://github.com/SaberaTalukder/TOTEM

[9] D. Lee, S. Malacarne, and E. Aune, “Vector Quantized Time Series Generation with a Bidirectional Prior Model,” in *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 206, pp. 7665–7693, 2023. Available: https://proceedings.mlr.press/v206/lee23d.html ; implementation: https://github.com/ML4ITS/TimeVQVAE

[10] L. Masserano *et al.*, “Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization,” in *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR, vol. 267, pp. 43248–43275, 2025. Available: https://proceedings.mlr.press/v267/masserano25a.html

[11] Y. Zhao, T. Zhou, C. Chen, L. Sun, Y. Qian, and R. Jin, “Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting,” arXiv:2402.05830, 2024. Available: https://arxiv.org/abs/2402.05830

[12] J. Lin, E. Keogh, S. Lonardi, and B. Chiu, “A Symbolic Representation of Time Series, with Implications for Streaming Algorithms,” in *Proc. 8th ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery*, 2003, doi: 10.1145/882082.882086. Available: https://doi.org/10.1145/882082.882086

[13] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are Transformers Effective for Time Series Forecasting?,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 9, pp. 11121–11128, 2023, doi: 10.1609/aaai.v37i9.26317. Available: https://doi.org/10.1609/aaai.v37i9.26317 ; official implementation: https://github.com/cure-lab/LTSF-Linear

[14] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers,” in *International Conference on Learning Representations (ICLR)*, 2023. Available: https://openreview.net/forum?id=Jbdc0vTOcol ; implementation reference: https://github.com/yuqinie98/PatchTST

[15] F. X. Diebold and R. S. Mariano, “Comparing Predictive Accuracy,” *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995, doi: 10.1080/07350015.1995.10524599. Available: https://doi.org/10.1080/07350015.1995.10524599

[16] D. N. Politis and J. P. Romano, “The Stationary Bootstrap,” *Journal of the American Statistical Association*, vol. 89, no. 428, pp. 1303–1313, 1994, doi: 10.1080/01621459.1994.10476870. Available: https://doi.org/10.1080/01621459.1994.10476870

[17] Y. Benjamini and Y. Hochberg, “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, vol. 57, no. 1, pp. 289–300, 1995, doi: 10.1111/j.2517-6161.1995.tb02031.x. Available: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

---

# 72. Final status

**STEP 04 is theoretically specified and experimentally falsifiable.**

The document deliberately does **not** declare quantization beneficial before experimentation.

The strongest research contribution to investigate is not generic discretization—which is established prior art—but the following structured hypothesis:

\[
\boxed{
\text{training-only noise/SNR estimation}
\rightarrow
\text{predicted useful resolution}
\rightarrow
\text{quantization/companding}
\rightarrow
\text{feature-specific or branch-specific representation}
\rightarrow
\text{out-of-sample forecasting validation}
}
\]

If agents approve the protocol after audit, the project can proceed to **STEP 05 — source/symbol coding and entropy-efficient representation**.
