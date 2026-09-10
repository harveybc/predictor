# STEP 10 — Synchronization, Temporal Alignment and Timing Recovery
## Causal Lead–Lag Estimation, Dynamic Alignment and Asynchronous Multivariate Time-Series Modeling

**Status:** Research protocol — final draft for independent agent audit  
**Date:** 2026-09-05  
**Scope:** Temporal synchronization and alignment of multiple sequential information sources before or inside modular machine-learning models. Initial target domain: financial multivariate time series; intended abstraction: general multisensor/multisource systems.  
**Prerequisites:** STEPS 01–09, particularly STEP 06 phase/cross-spectral representations and STEP 09 shared/private dependency maps.  
**Project split discipline:** 4 years training + 1 year validation + 1 year test for project datasets unless a canonical public benchmark specifies another split.  
**Next main-chain step:** STEP 11 — controlled redundancy / corruption-hardening of the existing feature extractor.

---

# 0. Executive conclusion

STEP 10 asks a deceptively simple question:

> **If two series have observations labeled with the same or different timestamps, are the information-bearing events actually aligned in time for the task being learned?**

The communications analogue is timing recovery/synchronization:

- estimate where a received symbol begins;
- estimate relative propagation delay;
- compensate timing error before detection.

For multivariate time series the corresponding problem is broader:

\[
\boxed{
\text{clock time}
\neq
\text{observation time}
\neq
\text{publication time}
\neq
\text{effective information time}
}
\]

and even two perfectly timestamped signals can exhibit a genuine dynamic lead–lag relation:

\[
X_i(t)
\leadsto
X_j(t+\tau_{ij}).
\]

A useful STEP-10 system must therefore distinguish at least four phenomena:

1. **timestamp/clock misalignment** — data are attached to wrong or incompatible clocks;
2. **asynchronous sampling** — channels are observed at different times;
3. **physical/informational propagation delay** — one process genuinely reacts later;
4. **elastic pattern timing** — similar patterns unfold at different local speeds.

These cases require different methods.

The state-of-the-art review shows that the project should not invent one universal “lag layer.” Mature and recent relevant families include:

- cross-correlation and generalized cross-correlation (GCC);
- GCC-PHAT and SNR/coherence-weighted delay estimation;
- cross-spectral phase-slope delay estimation;
- Hayashi–Yoshida methods for nonsynchronous financial observations;
- Dynamic Time Warping (DTW);
- Canonical Time Warping (CTW);
- Deep Canonical Time Warping (DCTW);
- soft-DTW and soft-DTW divergence;
- spatio-temporal alignment with optimal transport;
- 2025 lag-aware forecasting methods such as MRLCD-A and TLGNN;
- 2025 financial lead–lag graph models such as CGLR;
- 2026 differentiable lag-shape alignment (DGLASA-Net);
- 2026 Temporal Graph Neural Networks for financial lead–lag detection;
- continuous-time/irregular-sampling models such as GRU-D, Latent ODE, mTAN and Neural CDE.

The last family is especially important because it provides a major null hypothesis:

\[
\boxed{
\text{explicit alignment may be unnecessary or harmful if the model can consume true observation times directly.}
}
\]

The safe project architecture is therefore initially:

\[
\boxed{
\text{raw timestamped inputs}
+
\text{lag/alignment metadata}
+
\text{optionally causally aligned view}
}
\]

rather than destructively rewriting the raw time axis.

---

# 1. Historical communications basis

Suppose two received observations are delayed versions of the same source:

\[
x_1(t)
=
s(t)
+
n_1(t)
\]

\[
x_2(t)
=
a\,s(t-\tau)
+
n_2(t).
\]

Timing recovery estimates:

\[
\hat\tau.
\]

A basic estimator maximizes cross-correlation:

\[
\boxed{
\hat\tau
=
\arg\max_\tau
R_{12}(\tau)
}
\]

where:

\[
R_{12}(\tau)
=
E[x_1(t)x_2(t+\tau)].
\]

In sampled form:

\[
\hat k
=
\arg\max_k
\sum_n
x_1[n]x_2[n+k].
\]

This is the simplest STEP-10 baseline.

---

# 2. Generalized Cross-Correlation

Knapp and Carter's generalized cross-correlation framework estimates time delay after frequency-dependent prefiltering.

Let cross-spectrum be:

\[
G_{12}(\omega)
=
X_1(\omega)X_2^*(\omega).
\]

Generalized correlation:

\[
R^{(g)}_{12}(\tau)
=
\int
\Psi(\omega)
G_{12}(\omega)
e^{j\omega\tau}
d\omega.
\]

Then:

\[
\boxed{
\hat\tau
=
\arg\max_\tau
R^{(g)}_{12}(\tau).
}
\]

The weighting:

\[
\Psi(\omega)
\]

can emphasize frequencies with higher reliable information and suppress noisy bands.

This gives a direct connection to:

- STEP 03 SNR;
- STEP 06 spectrum/coherence;
- STEP 10 delay estimation.

---

# 3. GCC-PHAT

A common phase-transform weighting is:

\[
\Psi_{PHAT}(\omega)
=
\frac{1}
{
|G_{12}(\omega)|+\epsilon
}.
\]

Thus:

\[
R_{PHAT}(\tau)
=
\mathcal F^{-1}
\left[
\frac{
G_{12}(\omega)
}{
|G_{12}(\omega)|+\epsilon
}
\right].
\]

It emphasizes relative phase rather than spectral amplitude.

This is an important baseline but **not automatically optimal for financial data**. It was developed for delay estimation under signal-processing assumptions that may not match heterogeneous economic series.

---

# 4. SNR/coherence-weighted lag estimation

Knapp–Carter's broader result is more important to this project than PHAT itself:

> delay estimation can improve when frequency bands are weighted according to signal/noise reliability.

Candidate project weighting:

\[
\Psi(\omega)
=
g(
\widehat{SNR}_1(\omega),
\widehat{SNR}_2(\omega),
\widehat C_{12}(\omega)
).
\]

This is a natural reuse of prior STEPS rather than a new independent feature-engineering trick.

---

# 5. Cross-spectral phase-slope delay

For two signals related approximately by a delay:

\[
y(t)
=
x(t-\tau),
\]

their cross-spectrum phase approximately satisfies:

\[
\phi_{xy}(f)
=
-2\pi f\tau.
\]

Thus:

\[
\boxed{
\tau
=
-\frac{1}{2\pi}
\frac{d\phi}{df}.
}
\]

In practice:

1. compute cross-spectrum;
2. retain frequency regions with sufficient coherence;
3. unwrap phase carefully;
4. fit a robust line to phase versus frequency.

This was already foreshadowed in STEP 06 and becomes an explicit STEP-10 estimator.

---

# 6. Coherence-gating is mandatory for phase-delay estimates

Phase is unstable where spectral magnitude/coherence is weak.

Therefore delay from phase slope is valid only on bands satisfying a predeclared reliability criterion such as:

\[
C_{xy}(f)
>
\tau_C.
\]

Do not estimate:

\[
\tau
\]

from arbitrary low-coherence phase.

---

# 7. Delay is not necessarily constant

Classical model:

\[
\tau_{ij}(t)=\tau_{ij}.
\]

Real systems may exhibit:

\[
\boxed{
\tau_{ij}(t)
}
\]

that changes with:

- regime;
- liquidity;
- market session;
- event type;
- volatility;
- source latency;
- physical operating state.

STEP 10 must therefore compare:

- fixed delay;
- rolling/dynamic delay;
- distribution over delays.

---

# 8. Three alignment geometries

## 8.1. Rigid shift

\[
y(t)
\approx
x(t-\tau).
\]

One delay parameter.

## 8.2. Piecewise/dynamic shift

\[
y(t)
\approx
x(t-\tau(t)).
\]

## 8.3. Elastic warping

\[
y(\gamma(t))
\approx
x(t)
\]

for monotone nonlinear:

\[
\gamma(t).
\]

Do not use DTW when the scientific hypothesis is merely one fixed lead/lag; it introduces unnecessary flexibility.

---

# 9. Dynamic Time Warping

For sequences:

\[
X=(x_1,\ldots,x_n)
\]

and:

\[
Y=(y_1,\ldots,y_m),
\]

DTW seeks monotone path:

\[
\pi
\]

minimizing:

\[
\boxed{
DTW(X,Y)
=
\min_\pi
\sum_{(i,j)\in\pi}
d(x_i,y_j).
}
\]

It handles local stretching/compression of time.

But DTW is not inherently causal for online forecasting.

---

# 10. Causal DTW warning

A standard DTW alignment between complete windows can map:

\[
x_t
\]

to:

\[
y_{t+k}
\]

inside the same offline pair.

That is acceptable for retrospective shape comparison, but not automatically a legal online feature.

For forecast time \(T\), a causal alignment must satisfy:

\[
\boxed{
\text{aligned feature at }T
\text{ depends only on observations available by }T.
}
\]

A future point cannot be pulled backward into the present.

---

# 11. Sakoe–Chiba and causal alignment constraints

Constrain warping path to a band:

\[
|i-j|
\leq
w.
\]

Additional project constraints should include:

- monotonicity;
- maximum lag;
- slope constraint;
- no future-to-past mapping for production features.

This transforms generic DTW into a task-valid alignment problem.

---

# 12. Canonical Time Warping

Canonical Time Warping combines:

- Canonical Correlation Analysis;
- Dynamic Time Warping.

It is designed for cases where two sequences live in different observation spaces.

This is directly relevant when aligning:

- price representation with macro/event features;
- different sensor modalities;
- raw vs transformed representations.

CTW is a strong classical baseline before deep alignment.

---

# 13. Deep Canonical Time Warping

DCTW learns nonlinear representations:

\[
f_\theta(X)
\]

and:

\[
g_\phi(Y)
\]

that are simultaneously:

- highly correlated;
- temporally alignable.

It was introduced for heterogeneous multimodal sequences.

Project use is gated behind CTW/soft-DTW evidence; it is not a first-line method.

---

# 14. Soft-DTW

Soft-DTW replaces the hard minimum over alignment paths with a differentiable soft minimum.

This permits alignment losses inside neural training.

However:

\[
softDTW_\gamma
\]

has an entropic bias and is not itself always a proper divergence.

---

# 15. Soft-DTW divergence

Blondel, Mensch and Vert introduced a corrected soft-DTW divergence that is non-negative and minimized when sequences are equal under stated conditions.

For any differentiable alignment loss experiment, prefer comparing:

- soft-DTW;
- soft-DTW divergence;
- ordinary pointwise loss.

Do not silently treat soft-DTW as a metric.

---

# 16. Spatio-temporal alignment with optimal transport

Spatio-Temporal Alignments (STA) combine:

- soft-DTW in time;
- optimal transport for spatial/feature mismatch.

This is relevant when channels are not directly comparable one-to-one.

Potential future application:

\[
\text{time alignment}
+
\text{feature-space alignment}.
\]

But it is computationally heavier than simple lag estimation and should be gated.

---

# 17. State of the art — 2025 explicit lag-alignment forecasting

MRLCD-A (2025) proposes a plug-and-play lag detection/alignment module based on rolling multivariate cross-correlation.

It evaluates on:

- ETTh1;
- ETTh2;
- ETTm1;
- ETTm2;
- Weather;
- Exchange;
- TCCON atmospheric datasets.

It reports improvements when attached to mainstream forecasting models.

This is high-priority prior art because it directly tackles the same problem:

\[
\boxed{
\text{detect rolling lag}
\rightarrow
\text{align variables}
\rightarrow
\text{forecast}.
}
\]

Do not build a custom rolling-lag alignment module before benchmarking MRLCD-A's formulation.

---

# 18. Robust temporal alignment for multivariate forecasting — 2025

A 2025 *Expert Systems with Applications* paper explicitly studies robust temporal alignment for MTS forecasting and introduces:

- delay-related covariance alignment deviation;
- information-gain regularization;
- outlier filtering.

This matters because naive cross-correlation maxima can be distorted by:

- anomalies;
- low-information coincidences;
- outliers.

STEP 10 must include robustness diagnostics, not just lag estimation.

---

# 19. TLGNN — time-lagged relation graph

A 2025 Time-Lagged Relation Graph Neural Network explicitly models lagged relationships as graph edges rather than assuming delay-free dependencies.

This is a key baseline when many variables have different lead/lag relationships.

Graph form:

\[
i
\xrightarrow{\tau_{ij}}
j.
\]

This becomes especially relevant after STEP 09's dependency grouping.

---

# 20. CGLR — financial lead-lag-aware graph forecasting

A 2025 financial-market model, CGLR, explicitly:

1. learns important relationships;
2. estimates lead-lag displacement with cross-correlation/FFT;
3. incorporates lag information into graph message passing.

It was evaluated on Chinese A-share market data.

This is highly relevant prior art for the financial domain.

Therefore:

> do not claim novelty for “FFT cross-correlation + graph message passing for financial lead-lag.”

The project-specific question is how such lag information fits into the full STEPS 01–09 preprocessing chain.

---

# 21. DGLASA-Net — 2026 differentiable global/local alignment

DGLASA-Net (2026) introduces:

- sliding-window cross-correlation;
- global lag bias;
- learnable alignment matrix;
- differentiable global/local lag correction;
- frequency-domain shape-sensitive loss.

It evaluates across ETT, Exchange, Weather, Electricity/Traffic and atmospheric datasets.

This is one of the closest current state-of-the-art systems to STEP 10.

Project action:

> benchmark its global/local alignment logic before inventing a differentiable lag layer.

---

# 22. 2026 Temporal Graph Learning for financial lead–lag

Krstev et al., published July 2026 in *Machine Learning*, formulate financial lead–lag detection as temporal link prediction on dynamic graphs.

Their benchmark includes:

- stocks;
- commodities;
- five years of daily prices;
- financial indicators;
- sentiment features.

They compare multiple temporal graph models, including:

- JODIE;
- DySAT;
- TGAT;
- TGN;
- APAN;
- GraphMixer;
- adapted financial graph baselines.

GraphMixer reportedly performs best among the evaluated models.

This is immediately relevant to any proposed dynamic lag graph in this project.

---

# 23. Finance-specific lead–lag is established prior art

Financial lead–lag analysis predates modern neural models.

Empirical high-frequency work has documented asymmetric lead–lag relationships associated with:

- liquidity;
- trading intensity;
- event periods.

Lead–lag network methods have also been used to create predictive signals in equity markets.

Therefore the novelty target is not:

> “financial variables can lead one another.”

The research target is:

\[
\boxed{
\text{causal, dynamic, representation-aware alignment integrated into our modular pipeline}
}
\]

with strict point-in-time data governance.

---

# 24. Nonsynchronous financial observations

At high frequency, two assets are rarely observed at exactly the same timestamps.

Naive previous-tick interpolation can bias covariance/lead–lag estimates.

Hayashi–Yoshida-style estimators are designed for nonsynchronous observations without forcing all data onto one artificial grid.

This becomes a crucial alternative baseline whenever project data move below bar-level sampling into tick/quote data.

---

# 25. Hayashi–Yoshida and microstructure caveat

Research on nonsynchronous trading and microstructure noise shows that estimator efficiency depends strongly on:

- noise level;
- correlation;
- sampling scheme.

Therefore:

\[
\boxed{
\text{“more sophisticated asynchronous correction” is not universally superior.}
}
\]

This is a falsifiable benchmark, not a dogma.

---

# 26. Irregularly sampled data: alignment versus continuous-time modeling

There is a fundamentally different approach:

> do not align/resample at all.

Instead represent each observation with its actual time.

This family must be included as a control.

---

# 27. GRU-D

GRU-D incorporates:

- missingness mask;
- time since last observation;
- learnable decay.

It demonstrates that observation timing/missingness can be predictive information.

Project lesson:

\[
\boxed{
\text{time since observation is a feature, not merely a preprocessing inconvenience.}
}
\]

---

# 28. Latent ODE / ODE-RNN

Latent ODE models handle arbitrary observation gaps through continuous-time hidden dynamics.

This is relevant when a common regular grid is artificial.

---

# 29. Neural Controlled Differential Equations

Neural CDEs directly model partially observed, irregularly sampled multivariate streams.

They are a high-quality baseline for the hypothesis:

\[
\boxed{
\text{a continuous-time model can consume asynchronous observations without explicit alignment.}
}
\]

---

# 30. mTAN

Multi-Time Attention Networks map irregular observations to learned continuous-time embeddings and reference points via time attention.

This is another direct alternative to explicit resampling/lag correction.

---

# 31. Alignment decision taxonomy

Before choosing a method, classify the problem.

## A — Clock error

Timestamp itself is wrong or source clocks disagree.

Use metadata/system synchronization.

## B — Irregular observation

True event times differ.

Use irregular-time models or valid resampling.

## C — Fixed propagation delay

\[
\tau=\text{constant}.
\]

Use correlation/GCC/phase slope.

## D — Dynamic lead–lag

\[
\tau=\tau(t).
\]

Use rolling lag, dynamic graph, adaptive estimator.

## E — Elastic shape timing

\[
t\rightarrow\gamma(t).
\]

Use DTW/soft-DTW/CTW.

## F — Event-to-market timing

Use event timestamp + causal decay/attention, not generic waveform warping.

---

# 32. Operational timestamp contract

Every input record must ideally distinguish:

- `event_time`
- `observation_time`
- `publication_time`
- `ingestion_time`
- `as_of_time`

where applicable.

For market OHLC bars:

- bar-open time;
- bar-close/finalization time.

For macro data:

- scheduled release time;
- actual publication time;
- revision/vintage time.

For news/events:

- first-publication time.

For broker execution:

- exchange/venue timestamp;
- receipt timestamp where available.

---

# 33. Point-in-time availability function

Define:

\[
A_i(t)
\]

as the latest observation from source \(i\) actually available to the system by decision time \(t\).

All STEP-10 transforms must operate on:

\[
\boxed{
A_i(t)
}
\]

rather than retrospectively cleaned timestamps.

This is the most important causality rule in the step.

---

# 34. Alignment cannot move future information backward

Suppose training finds:

\[
X_i
\text{ leads }
X_j
\text{ by }2\text{ bars}.
\]

At time \(t\), it is legal to use:

\[
X_i(t)
\]

to help predict:

\[
X_j(t+2).
\]

It is **not** legal to create an aligned feature for time \(t\) using:

\[
X_j(t+2).
\]

Thus:

\[
\boxed{
\text{alignment metadata can shift interpretation forward;
data cannot be shifted backward from the future.}
}
\]

---

# 35. Safe aligned-view construction

For leader \(i\) and lagger \(j\) with estimated:

\[
\tau_{ij}>0,
\]

safe options include:

## Lag metadata

\[
B_{\tau}
=
\tau_{ij}(t).
\]

## Leader feature

\[
B_{lead}
=
X_i(t).
\]

## Historical aligned comparison

Compare:

\[
X_i(t-\tau_{ij})
\]

with:

\[
X_j(t)
\]

when both are historical.

## Forecast routing

Use leader representation at \(t\) for horizon approximately \(\tau_{ij}\).

Do not use future lagger values.

---

# 36. STEP 09 boundary

STEP 09 may use fixed causal lag structures such as VAR.

STEP 10 owns:

- estimating lag;
- updating lag;
- aligning dynamic relationships;
- deciding whether lag is fixed or variable.

This prevents hidden synchronization optimization inside source separation.

---

# 37. STEP 06 boundary

STEP 06 computes:

- cross-spectrum;
- coherence;
- phase.

STEP 10 converts those into:

- estimated delay;
- lag confidence;
- alignment state.

So:

\[
\boxed{
\text{phase representation}
\rightarrow
\text{timing inference}.
}
\]

---

# 38. STEP 05 boundary

STEP 05 conditional code gain can identify promising pairs/groups before expensive lag search.

For pair \(i,j\):

\[
G_{i\to j}>0
\]

does not tell us:

\[
\tau_{ij}.
\]

STEP 10 adds temporal direction/offset.

---

# 39. Pair-selection gate

Do not search all:

\[
O(m^2L)
\]

lags blindly for very large \(m\).

Candidate pair preselection:

1. semantic/domain relations;
2. STEP-05 conditional code gain;
3. STEP-06 coherence;
4. STEP-09 common-factor groups;
5. training-only mutual information.

Then estimate lags.

---

# 40. Fixed cross-correlation baseline

For stationary windows:

\[
\rho_{ij}(k)
=
corr(
X_i(t-k),
X_j(t)
).
\]

Estimate:

\[
\hat k_{ij}
=
\arg\max_{k\in[-K,K]}
|\rho_{ij}(k)|.
\]

Report:

- sign;
- maximum correlation;
- second-best margin;
- stability by block.

---

# 41. Prewhitening before cross-correlation

Autocorrelation can create broad/spurious cross-correlation peaks.

Classical remedy:

1. fit univariate model to \(X_i\);
2. filter both \(X_i\) and \(X_j\) with same filter;
3. compute residual cross-correlation.

This should be included as a baseline.

Do not confuse this detection prewhitening with STEP-08 domain equalization.

---

# 42. Mutual-information lag estimator

For nonlinear relation:

\[
\hat\tau
=
\arg\max_\tau
I(
X_i(t-\tau);
X_j(t)
).
\]

Potential benefit:

- nonlinear dependencies.

Risks:

- high estimator variance;
- bias;
- computational cost;
- multiple comparisons.

Use only after correlation baseline.

---

# 43. Granger lead–lag control

For pair/group:

\[
X_i
\]

Granger-predicts \(X_j\) if past \(X_i\) improves prediction of \(X_j\) beyond past \(X_j\).

This tests directed predictive information.

Important:

\[
\boxed{
\text{Granger causality}
\neq
\text{structural causality}.
}
\]

It is a useful lead–lag validation layer.

---

# 44. Transfer entropy optional control

Transfer entropy can capture nonlinear directed dependence:

\[
TE_{i\to j}
=
I(
X_{i,past};
X_{j,future}
\mid
X_{j,past}
).
\]

High computational/statistical burden means it is not first-line.

---

# 45. Dynamic rolling lag estimator

For trailing window:

\[
W_t
=
[t-L+1,t],
\]

estimate:

\[
\hat\tau_{ij}(t)
=
\arg\max_\tau
\rho_{ij,t}(\tau).
\]

Outputs:

- current lag;
- peak score;
- confidence;
- lag-change rate.

This is the simplest dynamic synchronization branch.

---

# 46. Lag uncertainty

Instead of one argmax, convert lag scores to distribution:

\[
p_t(\tau)
=
\frac{
\exp(\beta s_t(\tau))
}{
\sum_k\exp(\beta s_t(k))
}.
\]

Then compute:

\[
E[\tau_t]
\]

and:

\[
H(\tau_t).
\]

High entropy means uncertain synchronization.

This uncertainty should be passed to the core.

---

# 47. Lag-state branch

Recommended outputs:

\[
B_{lag}
=
[
E[\tau],
Var(\tau),
H(\tau),
score_{\max},
margin,
coherence
].
\]

This is safer than forcing a hard shift when lag evidence is weak.

---

# 48. Alignment confidence gate

Only create an explicitly aligned branch if:

\[
Q_{align}
>
\tau_Q.
\]

Otherwise:

- preserve raw stream;
- pass lag uncertainty only;
- allow downstream model to handle relationship.

This is analogous to receiver synchronization lock.

---

# 49. Lock / unlock state

Communications receivers often distinguish:

- synchronization acquired;
- synchronization lost.

STEP-10 analogue:

\[
LOCK_t
\in
\{0,1\}.
\]

Criteria may include:

- lag peak prominence;
- phase-slope fit quality;
- stability;
- coherence.

Expose:

\[
LOCK_t
\]

to downstream logic.

---

# 50. Lag hysteresis

Without hysteresis, dynamic lag estimates can jump rapidly.

Possible update:

\[
\tau_t
=
\begin{cases}
\hat\tau_t, & \text{if confidence sufficiently improves}\\
\tau_{t-1}, & \text{otherwise}.
\end{cases}
\]

This is an engineering extension after basic validation.

---

# 51. Lag Kalman/state-space model

Treat true delay as latent state:

\[
\tau_t
=
\tau_{t-1}
+
w_t.
\]

Observation:

\[
z_t
=
\tau_t
+
v_t
\]

where \(z_t\) comes from cross-correlation/phase estimator.

Kalman filtering can stabilize lag tracking.

This is preferable to arbitrary smoothing when the dynamic-lag model is plausible.

---

# 52. Change-point model for lag

Some systems have piecewise-stable lead/lag:

\[
\tau_t
=
\tau_r
\quad
\text{within regime }r.
\]

A change-point model may be more appropriate than continuous lag drift.

Test both on synthetic data.

---

# 53. MRLCD-A benchmark role

For fixed-grid MTS forecasting, MRLCD-A should be a mandatory plugin/reference comparator because it already combines:

- rolling lag detection;
- variable alignment;
- channel dependency;
- plug-and-play forecasting backbone compatibility.

The project should reproduce a subset before custom dynamic alignment.

---

# 54. DGLASA-Net benchmark role

DGLASA-Net should be a high-priority advanced comparator because it jointly models:

- local lag;
- global lag;
- learnable alignment;
- shape preservation.

It is especially relevant after the project's STEP-06 spectral/shape work.

---

# 55. Financial CGLR role

CGLR should be benchmarked conceptually/experimentally for financial cross-asset alignment.

Its architecture demonstrates a mature route:

\[
\boxed{
\text{relation discovery}
\rightarrow
\text{FFT/cross-correlation lag}
\rightarrow
\text{lag-aware graph routing}.
}
\]

The project should not duplicate this mechanism without comparative evidence.

---

# 56. 2026 Temporal Graph lead–lag role

For many assets:

\[
G_t=(V,E_t)
\]

with directed lag edges:

\[
i\xrightarrow{\tau}j.
\]

The 2026 temporal-graph benchmark demonstrates that dynamic graph modeling is now an explicit financial lead–lag research direction.

Therefore any STEP-10 graph extension must compare against:

- simple lag graph;
- GraphMixer-like temporal graph model;
- at least one stronger TGNN where feasible.

---

# 57. Irregular-time null model

For truly asynchronous inputs, compare explicit synchronization to direct continuous-time modeling.

Minimum controls:

- forward-fill + age/mask;
- GRU-D;
- mTAN;
- Neural CDE.

Latent ODE is optional if computational budget permits.

---

# 58. No forced interpolation principle

If observation times carry information, interpolation may erase it.

For source \(i\), preserve:

\[
\Delta t_i(t)
=
t-t_{last,i}
\]

and:

\[
m_i(t)
\]

mask/availability.

Even after resampling, age/missingness side information should be retained initially.

---

# 59. Macro/event alignment

For scheduled macro event \(e\):

- scheduled time:
  \[
  t_{sched};
  \]
- actual publication:
  \[
  t_{pub};
  \]
- system availability:
  \[
  t_{avail}.
  \]

Pre-release inputs may use only:

\[
t<t_{pub}
\]

information.

Post-release reaction features use:

\[
t\geq t_{avail}.
\]

Do not align the event to the nearest bar if doing so moves release information backward.

---

# 60. Event-to-bar mapping

For bar:

\[
[t_b,t_b+\Delta),
\]

if an event arrives at:

\[
t_e\in[t_b,t_b+\Delta),
\]

it is not available at bar open.

If prediction happens at bar close, event may be available.

Therefore availability depends on the exact execution cycle.

The artifact must record the decision timestamp contract.

---

# 61. Multi-timeframe alignment

For:

- 15m;
- 30m;
- 1h;
- 4h

series, naive interpolation/tile-up can distort timing.

Candidate representations:

1. exact nested historical bins;
2. age since sub-bar close;
3. separate branch with native resolution;
4. learned cross-timeframe attention.

The project's existing high-frequency branches already favor keeping some information at native sub-sequence resolution; STEP 10 should preserve that principle.

---

# 62. Aggregation delay

A 4h OHLC value is fully known only at bar close.

Thus using 4h close at the bar-open timestamp creates a 4h lookahead.

Every aggregated feature must be timestamped by **availability/finalization time**, not semantic interval label.

---

# 63. Time-zone / daylight-saving synchronization

For global financial/macro sources:

- UTC normalization;
- exchange-local calendars;
- daylight-saving transitions;
- holiday schedules

must be handled deterministically before statistical lag inference.

Clock/calendar errors should not be “learned” by lag models.

---

# 64. Market-session alignment

A cross-asset lag may be caused by:

- one market being closed;
- session overlap;
- stale prices.

Therefore include:

- market-open masks;
- time since last trade/bar;
- session identifiers.

Do not interpret closed-market staleness as propagation delay.

---

# 65. Synthetic benchmark T0 — known fixed delay

Generate:

\[
y_t
=
x_{t-\tau}
+
n_t.
\]

Sweep:

\[
\tau
\]

and SNR.

Compare:

- cross-correlation;
- GCC-PHAT;
- SNR-weighted GCC;
- phase-slope.

Metrics:

- lag error;
- lock probability;
- false lock;
- runtime.

---

# 66. Synthetic benchmark T1 — colored/noisy bands

Generate frequency-dependent noise.

Hypothesis:

SNR/coherence-weighted GCC should outperform unweighted delay estimation.

---

# 67. Synthetic benchmark T2 — time-varying lag

\[
\tau_t
=
\tau_0
+
\delta_t.
\]

Variants:

- smooth drift;
- random walk;
- piecewise constant;
- abrupt change.

Compare:

- fixed lag;
- rolling xcorr;
- state-space lag;
- MRLCD-A-like logic;
- DGLASA-like advanced module.

---

# 68. Synthetic benchmark T3 — elastic time warp

Construct:

\[
y(t)
=
x(\gamma(t))
+
n(t)
\]

with monotone nonlinear \(\gamma\).

Compare:

- rigid lag;
- DTW;
- soft-DTW;
- CTW/DCTW for multiview variants.

---

# 69. Synthetic benchmark T4 — heterogeneous observation space

Create latent signal \(s_t\), then:

\[
x_t=f(s_t)
\]

\[
y_t=g(s_{\gamma(t)})
\]

with different feature maps.

Test:

- DTW;
- CTW;
- DCTW.

This is the correct scenario for canonical warping.

---

# 70. Synthetic benchmark T5 — asynchronous sampling

Sample each channel at irregular observation times.

Compare:

- resample/interpolate;
- forward-fill + age/mask;
- GRU-D;
- mTAN;
- Neural CDE.

This determines whether explicit synchronization is justified.

---

# 71. Synthetic benchmark T6 — lag without causality

Construct common driver:

\[
z_t
\]

that creates apparent lag between:

\[
x_t
\]

and:

\[
y_t
\]

without direct causal effect.

This tests the warning:

\[
\boxed{
\text{lead–lag}
\neq
\text{causality}.
}
\]

---

# 72. Synthetic benchmark T7 — common periodicity false lag

Two independent series share periodic frequency but unrelated phase/realizations.

Naive cross-correlation can show spurious peaks.

Test:

- prewhitening;
- surrogate controls;
- block stability.

---

# 73. Synthetic benchmark T8 — event propagation

Create event at:

\[
t_e
\]

with response kernels:

\[
h_i(t-t_e)
\]

for multiple channels.

Estimate:

- onset delay;
- peak delay;
- decay.

This maps closely to macro/news propagation.

---

# 74. Public benchmark layer

Recommended fixed-grid benchmarks:

- ETTh1;
- ETTh2;
- ETTm1;
- ETTm2;
- Weather;
- Exchange;
- Electricity;
- Traffic;
- Solar;
- PEMS.

These overlap prior steps and current lag-alignment literature.

---

# 75. Irregular-time public benchmark layer

Use datasets established in irregular-time literature such as:

- PhysioNet;
- MIMIC;
- Human Activity

only for validating asynchronous-time handling, not for finance-specific claims.

---

# 76. Financial benchmark layer

Initial:

- EURUSD 1 h;
- EURUSD 4 h;
- ETHUSDT 4 h;
- related cross-asset features;
- point-in-time macro/fundamental events where available.

For actual high-frequency quote data later:

- nonsynchronous estimators become much more important.

---

# 77. Pairwise finance benchmark

Before graph models, run selected economically plausible pairs.

For pair:

\[
(i,j)
\]

report rolling:

- lag;
- cross-correlation;
- coherence;
- phase delay;
- conditional code gain;
- Granger score;
- lag stability.

This produces interpretable evidence before deep graph learning.

---

# 78. Dynamic lead–lag graph

Nodes:

\[
V=\{1,\ldots,m\}.
\]

Edge:

\[
e_{ij,t}
=
(
\tau_{ij,t},
q_{ij,t}
).
\]

Where:

- \(\tau\): delay;
- \(q\): confidence/strength.

Potential graph branch input:

\[
G_t.
\]

This is gated behind pairwise lag validity.

---

# 79. Lag graph sparsification

Keep only edges satisfying:

\[
q_{ij,t}>\tau_q
\]

and stability criteria.

Avoid fully connected dynamic graphs unless evidence justifies them.

---

# 80. Direction convention

Define one unambiguous convention.

Recommended:

\[
\tau_{i\to j}>0
\]

means:

> information/movement in \(i\) tends to precede corresponding movement in \(j\) by \(\tau\).

Every artifact and plot must use the same convention.

---

# 81. Time-unit convention

Store lag in both:

- integer sample units;
- physical duration.

Example for 4h:

\[
k=2
\Rightarrow
\tau=8h.
\]

This prevents confusion across timeframes.

---

# 82. Sub-sample delay

Integer bars may be too coarse.

Options:

- parabolic interpolation around correlation peak;
- phase-slope estimate;
- fractional-delay filters.

For 1h/4h financial bars, sub-bar lag should be interpreted cautiously because the underlying aggregated data may not support that resolution.

---

# 83. Nyquist reminder

STEP 01 constrains lag resolution.

A 4h bar series cannot reliably recover arbitrary sub-hour propagation structure from bar values alone.

Do not report false precision beyond the sampling process.

---

# 84. Alignment as feature vs alignment as data transform

Compare:

## Mode A — metadata only

\[
[X,\tau,q].
\]

## Mode B — aligned historical view

\[
[X,X^{aligned}].
\]

## Mode C — hard replacement

\[
X^{aligned}.
\]

Hard replacement is never the first experiment.

---

# 85. Alignment as attention bias

Instead of physically shifting data, use lag to bias attention:

\[
A_{ij}(t,s)
\]

toward:

\[
s\approx t-\tau_{ij}.
\]

This can preserve raw chronology while giving the model alignment information.

Potentially safer than resampling.

---

# 86. Lag-aware positional encoding

Encode:

\[
PE_{ij}
=
f(\tau_{ij}).
\]

Use in graph/message passing or cross-attention.

CGLR-like models provide prior art.

---

# 87. Alignment branch contract

Recommended outputs:

## Lag mean

\[
\hat\tau.
\]

## Lag distribution

\[
p(\tau).
\]

## Confidence

\[
q.
\]

## Lock state

\[
LOCK.
\]

## Alignment transform metadata

method/window/bands.

## Optional aligned historical tensor

only when causally valid.

---

# 88. Alignment plugin registry

Potential logical plugins:

- `xcorr_fixed`
- `gcc_phat`
- `gcc_snr`
- `phase_slope`
- `mutual_information_lag`
- `rolling_xcorr`
- `lag_state_space`
- `dtw_causal`
- `soft_dtw_alignment`
- `ctw`
- `mrlcd_a_reference`
- `dynamic_lag_graph`
- `irregular_time_passthrough`

These are conceptual contracts, not an instruction to implement all at once.

---

# 89. Falsifiable hypotheses — classical delay

## H10.1 — Cross-correlation recovers fixed delay at adequate SNR

For synthetic known delay:

\[
|\hat\tau-\tau|
\]

falls within declared tolerance.

**Falsified if:** implementation fails simple delay recovery.

---

## H10.2 — SNR/coherence weighting improves lag recovery under colored noise

\[
E_\tau^{weighted}
<
E_\tau^{unweighted}.
\]

**Falsified if:** weighting provides no controlled benefit.

---

## H10.3 — Phase-slope and correlation estimates agree under pure delay

For coherent narrow/model-valid data:

\[
\hat\tau_{phase}
\approx
\hat\tau_{xcorr}.
\]

**Falsified if:** systematic disagreement appears under ideal conditions.

---

# 90. Falsifiable hypotheses — dynamic lag

## H10.4 — Fixed lag fails when lead–lag varies materially

\[
P_{dynamic}
>
P_{fixed}
\]

under synthetic/real stable dynamic-lag evidence.

**Falsified if:** fixed lag remains sufficient.

---

## H10.5 — Lag uncertainty is predictive of relationship reliability

High:

\[
H(p(\tau))
\]

should correspond to weaker transfer benefit.

**Falsified if:** uncertainty has no relation to alignment utility.

---

## H10.6 — State-space smoothing improves lag tracking under gradual drift

\[
E_\tau^{state}
<
E_\tau^{raw\ rolling}.
\]

**Falsified if:** smoothing only adds delay or error.

---

# 91. Falsifiable hypotheses — elastic alignment

## H10.7 — DTW/soft-DTW helps only when local time deformation exists

Under rigid-shift data:

\[
P_{rigid}
\geq
P_{DTW}
\]

at lower complexity.

Under elastic data:

\[
P_{DTW}
>
P_{rigid}.
\]

**Falsified if:** elastic methods dominate even without deformation or fail when deformation is present.

---

## H10.8 — CTW/DCTW helps when observation spaces differ

\[
Align_{CTW/DCTW}
>
Align_{DTW}
\]

for synthetic multimodal sequences.

**Falsified if:** joint feature-space alignment offers no gain.

---

# 92. Falsifiable hypotheses — forecasting

## H10.9 — Lag metadata can improve forecasting without hard shifting

\[
P([X,\tau,q])
>
P(X).
\]

**Falsified if:** lag state never adds incremental value.

---

## H10.10 — Causal aligned view can improve forecasting when stable lead–lag exists

\[
P([X,X^{aligned}])
>
P(X).
\]

**Falsified if:** aligned view is redundant or harmful.

---

## H10.11 — Hard replacement is riskier than parallel alignment

Expected:

\[
P([X,X^{aligned}])
\geq
P(X^{aligned}).
\]

**Falsified if:** hard alignment consistently dominates.

---

# 93. Falsifiable hypotheses — finance

## H10.12 — Lead–lag relations are regime-dependent

\[
\tau_{ij}(t)
\]

or confidence changes across:

- volatility;
- session;
- event regimes.

**Falsified if:** fixed lags are stable enough.

---

## H10.13 — Liquidity/session metadata explains some apparent lead–lag

After conditioning on market openness/staleness/liquidity, some lag edges weaken.

**Falsified if:** metadata does not explain lag changes.

---

## H10.14 — Financial dynamic-lag graphs improve only after stable pairwise evidence

Graph model should not outperform robust pairwise baselines when edges are unstable/noisy.

This is a protective hypothesis.

---

## H10.15 — Temporal graph learning can add value for large cross-asset systems

For sufficiently rich stable dynamic relations:

\[
P_{TGNN}
>
P_{pairwise/simple}.
\]

**Falsified if:** GraphMixer/TGNN adds no benefit.

---

# 94. Falsifiable hypotheses — asynchronous data

## H10.16 — Explicit resampling is not universally optimal

For irregular synthetic/public data:

\[
P_{CDE/mTAN/GRUD}
>
P_{naive\ resample}
\]

in at least some settings.

**Falsified if:** simple resampling consistently dominates.

---

## H10.17 — Observation-age/mask metadata contains useful information

\[
P([X,\Delta t,m])
>
P(X_{imputed}).
\]

**Falsified if:** timing/missingness metadata is always redundant.

---

# 95. Falsifiable hypotheses — causality and leakage

## H10.18 — Retrospective unconstrained alignment overstates forecasting value

Offline DTW/full-window alignment should often produce optimistic estimates relative to causal alignment.

**Falsified if:** both are identical across controlled scenarios.

---

## H10.19 — Publication-time alignment matters for macro/event data

Using true point-in-time availability should differ materially from date-only/nearest-bar mapping in at least some event windows.

**Falsified if:** timestamp precision never changes results.

---

# 96. Falsifiable hypotheses — interaction with previous steps

## H10.20 — STEP-08 canonicalization can improve lag estimation when source distortions obscure timing

\[
E_\tau(E(X))
<
E_\tau(X)
\]

under controlled source distortion.

**Falsified if:** equalization does not help.

---

## H10.21 — STEP-09 grouping reduces lag-search variance

Estimating lags inside stable dependency groups should yield more reproducible lag graphs than exhaustive all-pair search.

**Falsified if:** grouping provides no stability benefit.

---

## H10.22 — STEP-06 cross-phase provides complementary lag evidence

Combining:

- cross-correlation;
- cross-phase/coherence

improves confidence calibration versus either alone.

**Falsified if:** no complementary value exists.

---

# 97. Falsifiable hypotheses — model size

## H10.23 — Explicit lag information can reduce downstream model burden

At fixed target performance:

\[
|\theta_{lag-aware}|
<
|\theta_{raw-large}|.
\]

**Falsified if:** alignment never reduces required capacity.

---

# 98. Falsifiable hypotheses — no universal synchronizer

## H10.24 — Synchronization method depends on misalignment class

Expected:

- fixed propagation delay:
  GCC/correlation;
- dynamic fixed-grid lag:
  rolling/state-space/MRLCD;
- elastic shape:
  DTW/soft-DTW;
- heterogeneous modalities:
  CTW/DCTW;
- asynchronous sampling:
  CDE/mTAN/GRU-D.

**Falsified if:** one method dominates all controlled classes.

---

# 99. Minimum experiment matrix

| ID | Problem | Method | Output | Forecast integration |
|---|---|---|---|---|
| T00 | Fixed lag | xcorr | \(\tau,q\) | metadata |
| T01 | Fixed noisy lag | GCC-PHAT | \(\tau,q\) | metadata |
| T02 | Colored noise | SNR-GCC | \(\tau,q\) | metadata |
| T03 | Spectral delay | phase slope | \(\tau,q\) | metadata |
| T04 | Dynamic lag | rolling xcorr | \(\tau_t,q_t\) | metadata/aligned |
| T05 | Dynamic lag | state-space | \(\tau_t,q_t\) | metadata/aligned |
| T06 | Elastic | DTW | path | diagnostic |
| T07 | Elastic | soft-DTW | soft path/loss | model |
| T08 | Heterogeneous | CTW | aligned latent | model |
| T09 | Irregular | GRU-D | latent | direct |
| T10 | Irregular | mTAN | latent | direct |
| T11 | Irregular | Neural CDE | latent | direct |
| T12 | Fixed-grid MTS | MRLCD-A ref | aligned MTS | backbone |
| T13 | Dynamic lag MTS | DGLASA ref | aligned MTS | backbone |
| T14 | Finance graph | CGLR/TGNN ref | lag graph | graph/core |
| T15 | Project | best lag metadata | \([X,\tau,q]\) | project core |
| T16 | Project | best aligned view | \([X,X^a,\tau,q]\) | project core |

---

# 100. Synthetic evaluation metrics

## Lag absolute error

\[
E_\tau
=
|\hat\tau-\tau|.
\]

## Lock rate

\[
P(LOCK=1|\text{true relation}).
\]

## False-lock rate

\[
P(LOCK=1|\text{no relation}).
\]

## Lag jitter

\[
Var(\hat\tau_t-\hat\tau_{t-1}).
\]

## Dynamic tracking error

\[
\frac1T
\sum_t
|\hat\tau_t-\tau_t|.
\]

---

# 101. Alignment-path metrics

For known warp:

- path distance;
- timing deviation;
- monotonicity violations;
- percentage future-to-past illegal mappings.

Causal production transform must have:

\[
\boxed{
0
}
\]

future-information violations.

---

# 102. Forecasting metrics

- MAE;
- RMSE;
- \(R^2\);
- per-horizon metrics;
- direction/turning-point timing where relevant;
- calibration;
- event lead-time metrics;
- economic/policy metrics only after predictive validation.

---

# 103. Lag stability metrics

By temporal block:

- median lag;
- IQR;
- entropy;
- edge persistence;
- sign stability;
- confidence.

A lag with high average correlation but unstable sign/offset is not a strong synchronization candidate.

---

# 104. Multiple-comparison problem

Searching:

\[
m(m-1)K
\]

pair-lag combinations generates many false maxima.

Required controls:

- pair preselection;
- permutation/surrogate nulls;
- FDR correction;
- block stability;
- validation confirmation.

Do not rank lags from raw maxima alone.

---

# 105. Surrogate/null lag test

Destroy cross-series temporal relation while preserving selected marginal properties.

Compare observed lag score:

\[
s_{obs}
\]

against null distribution:

\[
s^{(b)}.
\]

Potential controls:

- block permutation;
- phase randomization;
- circular shift respecting autocorrelation.

Use the least destructive null appropriate to the hypothesis.

---

# 106. Causal rolling protocol

For each decision time \(t\):

1. build training/trailing history available by \(t\);
2. estimate lag using only past/current available samples;
3. emit \(\tau_t,q_t,LOCK_t\);
4. construct legal aligned historical representation;
5. forecast;
6. advance clock;
7. update after new observations arrive.

No global full-series lag estimate may be recomputed using future validation/test observations.

---

# 107. Training-only static lag mode

Alternative cheap baseline:

1. estimate \(\tau_{ij}\) using training only;
2. freeze it;
3. apply same mapping to validation/test.

This is the cleanest initial test of whether stable lag exists.

---

# 108. Validation-adaptive lag mode

If dynamic alignment is allowed:

- estimator design/hyperparameters chosen using train+validation protocol;
- at validation/test replay, estimator updates only from past observed covariates, never from future targets.

---

# 109. Target-maturity rule

If a lag estimator uses forecast residuals or target feedback, adaptation at time \(t\) may use only targets whose horizon has fully matured by \(t\).

This reuses STEP-08 test-time-adaptation governance.

---

# 110. Architecture integration

For group \(g\):

\[
X_g
\rightarrow
Synchronizer_g
\rightarrow
(
\tau_g,
q_g,
LOCK_g,
X_g^{aligned}
).
\]

Parallel branch:

\[
B_{raw,g}=X_g
\]

\[
B_{lag,g}=[\tau_g,q_g,LOCK_g]
\]

\[
B_{align,g}=X_g^{aligned}.
\]

Core:

\[
Z
=
C(
B_{raw},
B_{lag},
B_{align},
\ldots
).
\]

---

# 111. Do not make lag a one-hot hard choice initially

If:

\[
p(\tau)
\]

is uncertain, preserve distribution/soft alignment.

A hard argmax loses uncertainty and can create discontinuous features.

---

# 112. Soft lag mixture

For discrete lags:

\[
\tau_k,
\]

define aligned representation:

\[
\tilde X_j(t)
=
\sum_k
p_t(\tau_k)
X_j(t-\tau_k).
\]

This is fully causal if every sampled point satisfies:

\[
t-\tau_k
\leq
t.
\]

This provides a differentiable/uncertainty-aware alternative to hard shifting.

---

# 113. Attention equivalent

Cross-attention can be biased by:

\[
p_t(\tau)
\]

instead of explicitly summing shifted samples.

This may fit the project modular architecture better.

---

# 114. Event-token integration

Synchronization events can become tokens:

- `LAG_LOCK_ACQUIRED`
- `LAG_LOCK_LOST`
- `LEAD_LAG_SHIFT`
- `CROSS_ASSET_DELAY_SPIKE`
- `EVENT_RESPONSE_ONSET`

Use existing event-token infrastructure.

---

# 115. RL integration

Do not alter SAC/PPO state until lag evidence improves supervised/diagnostic prediction.

Later:

\[
o_t
=
[
X_t,
\tau_t,
q_t,
LOCK_t
].
\]

RL should consume synchronization evidence, not learn illegal future alignment.

---

# 116. Main failure modes

## 116.1. Future-to-past leakage

Most severe.

## 116.2. Retrospective DTW optimism

Offline perfect path unavailable live.

## 116.3. Spurious periodic lag

Shared cycles create false peaks.

## 116.4. Stale-price lag

Closed/inactive market appears to lag.

## 116.5. Aggregation timestamp leakage

Bar value timestamped at open although known at close.

## 116.6. Macro revision leakage

Revised values used at historical release time.

## 116.7. Multiple-comparison maxima

Large lag grid creates false relationships.

## 116.8. Lag sign convention errors

Leader/lagger reversed.

## 116.9. False sub-sample precision

4h data used to claim minute-level delays.

## 116.10. Lag instability

Dynamic relation too unstable to use.

## 116.11. Overwarping

DTW aligns unrelated shapes.

## 116.12. Causality overclaim

Lead–lag presented as causal mechanism.

## 116.13. Alignment destroys missingness information

Irregular observation pattern was itself useful.

## 116.14. Graph complexity inflation

TGNN gain caused by parameter count.

---

# 117. Audit checklist

- [ ] Lag sign convention is explicit.
- [ ] Time unit and sample-unit lag are both stored.
- [ ] Decision timestamp contract is explicit.
- [ ] Bar availability/finalization time is correct.
- [ ] Macro/event publication/vintage time is point-in-time valid.
- [ ] Fixed-lag and dynamic-lag hypotheses are separated.
- [ ] Rigid lag and elastic warping are not conflated.
- [ ] Standard DTW is never used as live causal preprocessing without constraints.
- [ ] GCC baseline is implemented before a learned lag model.
- [ ] SNR/coherence weighting is compared against unweighted correlation.
- [ ] Phase-slope delay is coherence-gated.
- [ ] Pair search is prefiltered or multiple-comparison controlled.
- [ ] Lag stability is measured by time block.
- [ ] Lag uncertainty/confidence is retained.
- [ ] Raw input remains available in first ablations.
- [ ] Hard alignment is compared against metadata-only and parallel modes.
- [ ] MRLCD-A is reviewed before custom rolling-lag alignment.
- [ ] DGLASA-Net is reviewed before custom differentiable global/local alignment.
- [ ] CGLR and 2026 temporal financial graph prior art are reviewed before custom lag graph.
- [ ] Irregular-time controls (GRU-D/mTAN/Neural CDE) are included where applicable.
- [ ] Hayashi–Yoshida-style methods are considered for nonsynchronous tick data.
- [ ] No future target enters lag estimation before maturity.
- [ ] No causal claim is made from lead–lag alone.
- [ ] Parameter/compute-matched controls are included for deep alignment models.
- [ ] STEP 09 does not hide dynamic lag optimization.
- [ ] STEP 06 representation and STEP 10 timing inference remain conceptually distinct.

---

# 118. Reuse matrix

| Problem | Prior art | Project action |
|---|---|---|
| Fixed delay | cross-correlation | Reuse |
| Noise-aware delay | GCC / GCC-PHAT | Reuse |
| Spectral delay | cross-phase slope | Reuse STEP-06 outputs |
| Fixed-grid rolling lag | MRLCD-A | High-priority comparator |
| Robust lag alignment | 2025 robust temporal alignment | Reference |
| Dynamic differentiable lag | DGLASA-Net | High-priority comparator |
| Lag graph | TLGNN | Benchmark/reference |
| Financial lag graph | CGLR | Benchmark/reference |
| Financial temporal graph | Krstev et al. 2026 | Current SOTA reference |
| Elastic warping | DTW | Reuse library |
| Differentiable warping | soft-DTW/divergence | Reuse |
| Multiview warping | CTW / DCTW | Benchmark before custom |
| Spatio-temporal feature mismatch | STA / OT | Late-stage |
| Irregular observations | GRU-D | Baseline |
| Irregular continuous time | mTAN | Baseline |
| Irregular continuous dynamics | Neural CDE | High-priority baseline |
| Nonsynchronous finance | Hayashi–Yoshida | Use for tick-level work |

---

# 119. Recommended implementation order

1. Timestamp/availability audit.
2. Synthetic fixed-delay xcorr.
3. GCC-PHAT.
4. SNR/coherence-weighted GCC.
5. Cross-phase-slope delay.
6. Static train-only financial/public lags.
7. Stability + surrogate/null tests.
8. Lag metadata branch.
9. Rolling lag.
10. Lag state-space smoothing.
11. MRLCD-A reference.
12. CGLR/TLGNN reference on selected MTS.
13. DGLASA-Net reference if dynamic lag evidence exists.
14. DTW/soft-DTW only for demonstrated elastic timing.
15. CTW/DCTW only for heterogeneous representations.
16. Irregular-time baselines for asynchronous sources.
17. Dynamic lag graph only after pairwise evidence.
18. Project multi-branch integration.
19. RL/event-token integration only after predictive gates.

---

# 120. Decision gates

## Gate 10A — Timestamp integrity

Point-in-time availability contract passes audit.

## Gate 10B — Fixed-delay correctness

Classical estimators recover synthetic delays.

## Gate 10C — Noise-aware benefit

GCC/SNR weighting behaves correctly under controlled noise.

## Gate 10D — Real lag evidence

At least one pair/group shows stable training/validation lead–lag evidence beyond surrogates.

## Gate 10E — Metadata value

Lag metadata improves or meaningfully calibrates forecasting.

## Gate 10F — Alignment value

Causal aligned view adds value beyond metadata-only.

## Gate 10G — Dynamic justification

Rolling/dynamic lag beats fixed lag before DGLASA/TGNN complexity.

## Gate 10H — Elastic justification

DTW family is used only if rigid alignment is insufficient and warping is causal/useful.

## Gate 10I — Irregular-time comparison

Explicit alignment beats or complements continuous-time alternatives where asynchronous sampling exists.

## Gate 10J — Financial/public transfer

Effect survives public and project validation.

## Gate 10K — Held-out confirmation

Only frozen selected configuration reaches test.

---

# 121. Recommended artifacts

1. `step10_timestamp_contract.json`
2. `step10_source_availability_manifest.parquet`
3. `step10_pair_candidates.parquet`
4. `step10_fixed_lag_estimates.parquet`
5. `step10_gcc_results.parquet`
6. `step10_phase_delay_results.parquet`
7. `step10_lag_stability.parquet`
8. `step10_surrogate_nulls.parquet`
9. `step10_dynamic_lag_tracks.parquet`
10. `step10_lag_uncertainty.parquet`
11. `step10_alignment_ablation.parquet`
12. `step10_irregular_time_baselines.parquet`
13. `step10_dynamic_graph_edges.parquet`
14. `step10_forecast_metrics.parquet`
15. `step10_statistical_tests.json`
16. `step10_audit_report.md`
17. reproducibility manifest:
    - source time zones;
    - availability definitions;
    - bar finalization semantics;
    - lag sign convention;
    - lag search bounds;
    - window sizes;
    - estimator parameters;
    - dataset hashes;
    - split dates;
    - seeds;
    - git commits;
    - library versions.

---

# 122. State-of-the-art synthesis

The review produces six strong conclusions.

## 122.1. Simple delay estimation remains fundamental

Cross-correlation and GCC are still the correct controls for genuine delay problems.

## 122.2. Modern forecasting now treats lag alignment explicitly

MRLCD-A, TLGNN and robust temporal-alignment work show that lag correction has moved into mainstream MTS forecasting.

## 122.3. 2026 methods are differentiable and graph-based

DGLASA-Net and temporal financial graph learning show that current research treats lag as:

- local/global;
- time-varying;
- graph-structured;
- learnable.

## 122.4. Financial lead–lag has specialized prior art

CGLR and the 2026 temporal-graph benchmark mean the project must compare against current financial lag-aware models rather than transplant generic signal-processing delay estimators alone.

## 122.5. Explicit alignment is not always required

GRU-D, mTAN, Latent ODE and Neural CDE demonstrate a mature alternative: preserve true observation times and let the model reason in continuous/irregular time.

## 122.6. Causality governance dominates everything

The mathematically best retrospective alignment is useless if it requires future observations.

---

# 123. Project-specific research opportunity

The project's opportunity is not:

> “discover that time series have lags.”

It is the integration:

\[
\boxed{
\text{point-in-time timestamp contract}
\rightarrow
\text{STEP-05 dependency evidence}
\rightarrow
\text{STEP-06 phase/coherence evidence}
\rightarrow
\text{STEP-09 dependency groups}
\rightarrow
\text{causal lag distribution}
\rightarrow
\text{alignment confidence/lock}
\rightarrow
\text{raw + lag-aware branch}
\rightarrow
\text{modular core}
}
\]

with dynamic alignment promoted only after simpler fixed-lag evidence survives.

---

# 124. Domain-general architecture implication

For a generic multisensor system:

\[
X_i(t_i)
\]

may differ in:

- measurement clock;
- sample time;
- propagation delay;
- sampling rate.

STEP 10 produces:

\[
\boxed{
(
X_i,
t_i,
\Delta t_i,
\tau_{ij},
q_{ij},
LOCK_{ij}
)
}
\]

rather than forcing every stream blindly onto an artificial identical grid.

This abstraction generalizes naturally to:

- finance;
- industrial sensors;
- biomedical monitoring;
- robotics;
- energy;
- weather;
- traffic;
- telecommunications.

---

# 125. References — IEEE style

[1] C. H. Knapp and G. C. Carter, “The Generalized Correlation Method for Estimation of Time Delay,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 24, no. 4, pp. 320–327, Aug. 1976, doi: 10.1109/TASSP.1976.1162830. Available: https://doi.org/10.1109/TASSP.1976.1162830

[2] H. Sakoe and S. Chiba, “Dynamic Programming Algorithm Optimization for Spoken Word Recognition,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 26, no. 1, pp. 43–49, Feb. 1978, doi: 10.1109/TASSP.1978.1163055. Available: https://doi.org/10.1109/TASSP.1978.1163055

[3] F. Zhou and F. De la Torre, “Canonical Time Warping for Alignment of Human Behavior,” in *Advances in Neural Information Processing Systems*, vol. 22, 2009. Available: https://papers.nips.cc/paper_files/paper/2009/hash/2ca65f58e35d9ad45bf7f3ae5cfd08f1-Abstract.html

[4] G. Trigeorgis, M. A. Nicolaou, S. Zafeiriou, and B. W. Schuller, “Deep Canonical Time Warping,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 5110–5118. Available: https://openaccess.thecvf.com/content_cvpr_2016/html/Trigeorgis_Deep_Canonical_Time_CVPR_2016_paper.html

[5] M. Cuturi and M. Blondel, “Soft-DTW: A Differentiable Loss Function for Time-Series,” in *Proceedings of the 34th International Conference on Machine Learning*, PMLR vol. 70, pp. 894–903, 2017. Available: https://proceedings.mlr.press/v70/cuturi17a.html

[6] M. Blondel, A. Mensch, and J.-P. Vert, “Differentiable Divergences Between Time Series,” in *Proceedings of AISTATS*, PMLR vol. 130, pp. 3853–3861, 2021. Available: https://proceedings.mlr.press/v130/blondel21a.html

[7] H. Janati, M. Cuturi, and A. Gramfort, “Spatio-Temporal Alignments: Optimal Transport Through Space and Time,” in *Proceedings of AISTATS*, PMLR vol. 108, pp. 1695–1704, 2020. Available: https://proceedings.mlr.press/v108/janati20a.html

[8] D. Sun, J. Qin, Z. Zhang, X. Qin, and H. Zhang, “MRLCD-A: Lag-aware Alignment for Multivariate Time Series Forecasting in Multiple Scenarios,” *Information Processing & Management*, vol. 62, no. 5, art. 104191, 2025, doi: 10.1016/j.ipm.2025.104191. Available: https://doi.org/10.1016/j.ipm.2025.104191

[9] X. Wang, M. Zhang, and J. Su, “Robust Temporal Alignment for Multivariate Time Series Forecasting,” *Expert Systems with Applications*, vol. 289, art. 128299, 2025, doi: 10.1016/j.eswa.2025.128299. Available: https://doi.org/10.1016/j.eswa.2025.128299

[10] “Time-lagged Relation Graph Neural Network for Multivariate Time Series Forecasting,” *Engineering Applications of Artificial Intelligence*, vol. 139, Part B, art. 109530, 2025, doi: 10.1016/j.engappai.2024.109530. Available: https://doi.org/10.1016/j.engappai.2024.109530

[11] S. Xiao, Q. Li, X. Gong, J. Zhao, L. Gu, and L. Peng, “Unveiling Risk Propagation: A Lead-Lag-Aware Framework for Financial Market Prediction,” *Expert Systems with Applications*, vol. 288, art. 128143, 2025, doi: 10.1016/j.eswa.2025.128143. Available: https://doi.org/10.1016/j.eswa.2025.128143

[12] D. Sun, J. Qin, H. Zhang, H. Ma, D. Wang, and Z. Liao, “DGLASA-Net: Breaking Local Stationarity via Lag-Shape Alignment for Multi-Scenario Forecasting and Decision Making,” *Advanced Engineering Informatics*, 2026, art. 104501, doi: 10.1016/j.aei.2026.104501. Available: https://doi.org/10.1016/j.aei.2026.104501

[13] I. Krstev, D. Rigoni, I. Mishkovski, and L. Pasa, “A Temporal Graph Learning Framework for Lead-Lag Detection in Financial Markets,” *Machine Learning*, vol. 115, art. 173, 2026, doi: 10.1007/s10994-026-07113-y. Available: https://doi.org/10.1007/s10994-026-07113-y

[14] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent Neural Networks for Multivariate Time Series with Missing Values,” *Scientific Reports*, vol. 8, art. 6085, 2018, doi: 10.1038/s41598-018-24271-9. Available: https://doi.org/10.1038/s41598-018-24271-9

[15] Y. Rubanova, R. T. Q. Chen, and D. Duvenaud, “Latent Ordinary Differential Equations for Irregularly-Sampled Time Series,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. Available: https://neurips.cc/virtual/2019/poster/13668

[16] P. Kidger, J. Morrill, J. Foster, and T. Lyons, “Neural Controlled Differential Equations for Irregular Time Series,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. Available: https://papers.nips.cc/paper_files/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html

[17] S. N. Shukla and B. M. Marlin, “Multi-Time Attention Networks for Irregularly Sampled Time Series,” in *International Conference on Learning Representations*, 2021. Official implementation: https://github.com/reml-lab/mTAN

[18] “Covariance Measurement in the Presence of Non-synchronous Trading and Market Microstructure Noise,” *Journal of Econometrics*, vol. 160, no. 1, pp. 58–68, 2011, doi: 10.1016/j.jeconom.2010.03.015. Available: https://doi.org/10.1016/j.jeconom.2010.03.015

[19] N. Huth and F. Abergel, “High Frequency Lead/Lag Relationships — Empirical Facts,” *Journal of Empirical Finance*, 2014. Available: https://www.sciencedirect.com/science/article/pii/S0927539814000048

[20] “Lead–lag Detection and Network Clustering for Multivariate Time Series with an Application to the US Equity Market,” *Machine Learning*, 2022/2023, doi: 10.1007/s10994-022-06250-4. Available: https://doi.org/10.1007/s10994-022-06250-4

[21] “Local Lead–Lag Relationships and Nonlinear Granger Causality: An Empirical Analysis,” *Entropy*, vol. 24, no. 3, art. 378, 2022, doi: 10.3390/e24030378. Available: https://doi.org/10.3390/e24030378

---

# 126. Final status

**STEP 10 is theoretically specified after a dedicated state-of-the-art review and is ready for independent agent audit.**

The central principle is:

\[
\boxed{
\text{Never align timestamps merely to make tensors look synchronous.}
}
\]

Instead:

\[
\boxed{
\text{identify the timing problem}
\rightarrow
\text{estimate lag with uncertainty}
\rightarrow
\text{preserve point-in-time legality}
\rightarrow
\text{emit lag state/lock}
\rightarrow
\text{add a causal aligned view only when justified}
}
\]

The most important null alternative is equally explicit:

\[
\boxed{
\text{sometimes the correct solution is not synchronization, but a model that understands irregular time directly.}
}
\]

If agent audit approves this protocol, proceed to **STEP 11 — controlled redundancy / corruption-hardening of the existing feature extractor**, under `WORKPLAN_PATCH_002`: no new unrelated autoencoder is introduced.
