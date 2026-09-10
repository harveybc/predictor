# STEP 09 — Interference, Echo and Crosstalk Cancellation
## Shared/Common Components, Unique Innovation and Multivariate Source Separation for Time-Series ML

**Status:** Research protocol — final draft for independent agent audit  
**Date:** 2026-09-05  
**Scope:** Multivariate sequential data; initial target domain is financial time series, but the abstraction is intended to generalize to multisensor and multi-source domains.  
**Prerequisites:** STEPS 01–08, with STEP 08 required to define an operational observation/channel domain before any source/domain-specific cancellation experiment is promoted.  
**Project split discipline:** 4 years training + 1 year validation + 1 year test for project datasets, unless a public benchmark defines its own canonical split.  
**Next main-chain step:** STEP 10 — synchronization / temporal alignment / timing recovery.

---

# 0. Executive conclusion

STEP 09 translates three classical communications problems:

- interference cancellation;
- echo cancellation;
- crosstalk/source separation;

into the multivariate ML question:

> **When several observed series contain overlapping/common structure, can we separate what is shared from what is unique without destroying information that matters for the downstream task?**

The canonical project decomposition is:

\[
\boxed{
X_{i,t}
=
C_{i,t}
+
U_{i,t}
}
\]

where:

- \(C_{i,t}\): component of series \(i\) explainable from shared factors, reference series or multivariate context;
- \(U_{i,t}\): residual/private/idiosyncratic innovation.

Crucially:

\[
\boxed{
C_{i,t}\neq \text{noise}
}
\]

and:

\[
\boxed{
U_{i,t}\neq \text{signal}
}
\]

in general.

A common market/macro factor may be highly predictive. A private residual may be unpredictable noise. Therefore the primary architecture is **not destructive subtraction**. The initial safe representation is:

\[
\boxed{
[X,\;C,\;U]
}
\]

or groupwise equivalents, allowing the downstream core to decide which parts are useful.

The communications analogy is strongest in the adaptive-noise-cancellation condition:

\[
d_t=s_t+n_t,
\]

with a reference \(r_t\) correlated with interference \(n_t\) but ideally uncorrelated with desired signal \(s_t\). An adaptive filter estimates:

\[
\hat n_t=f(r_{\le t})
\]

and outputs:

\[
e_t=d_t-\hat n_t.
\]

If the reference also contains desired signal, cancellation can destroy the very information we want. That failure mode becomes a mandatory negative control for this project.

The state-of-the-art review indicates that no new monolithic “interference network” should be invented first. Mature relevant families already exist:

1. adaptive noise cancellation / LMS-NLMS-RLS;
2. regression residualization / VAR innovations;
3. PCA and static factor models;
4. generalized dynamic factor models;
5. robust PCA / low-rank + sparse decomposition;
6. ICA / SOBI / IVA blind source separation;
7. CCA / Deep CCA for common subspaces;
8. shared/private representation learning;
9. channel-independent versus channel-mixed forecasting architectures;
10. modern disentangled multivariate forecasting.

The first implementation must begin with **existing low-cost evidence and classical baselines** before deep shared/private models.

---

# 1. Agent-audit constraints adopted before STEP 09

The following constraints are treated as normative for this protocol.

## 1.1. STEP 08 must define an operational channel/domain

Equalization is not synonymous with the existing z-score normalizer.

Before source/domain-specific cancellation is interpreted, STEP 08 must specify what the operational distortion/channel is, for example:

- asset/instrument;
- source/vendor;
- venue;
- volatility regime;
- known transfer function;
- controlled synthetic channel.

Whitening used purely to optimize a STEP-07 matched detector remains a STEP-07 detection operation unless it is explicitly part of a broader source/channel canonicalization contract.

---

## 1.2. C4 side information begins as evidence, not code

The first STEP-09 side-information artifact is the conditional code-gain table already implied by STEP-05 H5.8:

\[
G_{j\leftarrow i}
=
L_j^{self}
-
L_j^{cond(i)}.
\]

No new neural component is required to produce the first grouping evidence.

This table should be generated/consumed before implementing shared/private encoders.

---

## 1.3. Sparse/CSC does not open inside STEP 09

Sparse coding and convolutional sparse coding remain in the compression-transversal lane.

They are gated behind STEP-07 detector evidence and must beat or complement strong inexpensive detector baselines such as MiniRocket before they justify implementation effort.

No custom K-SVD implementation is proposed.

---

## 1.4. Latent rate–distortion is not a MacKay-capacity claim

Compression lane C6 belongs to the feature extractor/core latent after SNR/representation experiments.

It is not interpreted as:

- a neuron-capacity theorem;
- MacKay's “bits per neuron” result;
- the information-content calculation of STEP 02.

STEP 09 does not attempt to estimate network Kolmogorov capacity.

---

## 1.5. STEP 11 does not create another autoencoder

The project already has reconstruction/feature-extractor infrastructure.

STEP 11 will later ask whether controlled corruption/masking of the existing extractor training improves latent robustness; it does not begin by creating another unrelated autoencoder.

---

# 2. Historical basis — adaptive interference cancellation

Widrow et al.'s adaptive noise-cancellation formulation uses:

\[
d_t=s_t+n_{0,t}
\]

as the primary observation and a reference:

\[
r_t
\]

correlated with \(n_{0,t}\).

Adaptive filter:

\[
\hat n_t
=
w_t^Tr_t.
\]

Output:

\[
e_t
=
d_t-\hat n_t.
\]

The filter minimizes:

\[
E[e_t^2].
\]

Under the important condition that the reference is correlated with interference but not with desired signal, the filter can estimate/cancel the interference with limited desired-signal distortion.

This reference-signal condition is the core scientific analogy for STEP 09.

---

# 3. The reference-contamination problem

Suppose:

\[
r_t
=
n_{r,t}
+
\alpha s_t.
\]

Then a sufficiently powerful cancellation filter can learn to subtract part of:

\[
s_t
\]

itself.

Therefore:

\[
\boxed{
\text{cross-series predictability}
\neq
\text{permission to remove the predicted component}.
}
\]

For ML, another series \(X_j\) may explain \(X_i\) precisely because both respond to the same useful economic driver.

The initial representation must preserve both:

\[
\hat C_i
\]

and:

\[
U_i=X_i-\hat C_i.
\]

---

# 4. Operational multivariate model

Let:

\[
\mathbf X_t
=
[X_{1,t},\ldots,X_{m,t}]^T.
\]

A common/private representation can be written:

\[
\boxed{
\mathbf X_t
=
\mathbf C_t
+
\mathbf U_t
}
\]

where:

\[
\mathbf C_t
=
F(
\mathbf X_{\le t}
)
\]

is the shared/explainable component and:

\[
\mathbf U_t
=
\mathbf X_t-\mathbf C_t.
\]

Alternative latent-factor form:

\[
\boxed{
\mathbf X_t
=
\Lambda \mathbf f_t
+
\mathbf u_t
}
\]

with:

- \(\mathbf f_t\): lower-dimensional common factors;
- \(\Lambda\): loadings;
- \(\mathbf u_t\): idiosyncratic component.

Dynamic version:

\[
X_{i,t}
=
\chi_{i,t}
+
\xi_{i,t},
\]

where \(\chi\) is a dynamic common component and \(\xi\) the idiosyncratic component.

---

# 5. “Interference” must be target-relative

In communications, interference has an operational definition:

> unwanted energy that degrades detection/decoding of the desired signal.

For ML, define a component \(C\) as nuisance only relative to target \(Y\).

A component is a cancellation candidate only if evidence shows:

1. it is predictable/shared from reference inputs;
2. removing it does not reduce target information;
3. residual/private representation improves or preserves OOS task performance.

Therefore:

\[
\boxed{
\text{shared}
\not\Rightarrow
\text{interference}.
}
\]

The term “interference cancellation” is historical; the safe ML operation is initially **shared/private decomposition**.

---

# 6. Three different decompositions that must not be conflated

## 6.1. Signal/noise

STEP 03:

\[
X=S+N.
\]

## 6.2. Predictable/innovation

STEP 05:

\[
X=\hat X+E.
\]

## 6.3. Shared/private

STEP 09:

\[
X=C+U.
\]

These can overlap but are not equivalent.

For example:

- a common macro shock can be shared but unpredictable;
- a private series-specific seasonal effect can be predictable;
- measurement noise can be private but useless.

---

# 7. First no-new-code artifact: conditional code-gain matrix

From STEP 05 source models:

\[
L_j^{self}
=
-\sum_t
\log_2
p(
X_{j,t}
\mid
X_{j,<t}
)
\]

and:

\[
L_j^{cond(i)}
=
-\sum_t
\log_2
p(
X_{j,t}
\mid
X_{j,<t},
X_{i,\le t}
).
\]

Define:

\[
\boxed{
G_{i\rightarrow j}
=
L_j^{self}
-
L_j^{cond(i)}.
}
\]

Interpretation:

> how many source-model bits are saved when series \(i\) is available while describing series \(j\).

This is the first candidate map of side-information usefulness.

---

# 8. Conditional-code-gain table

Minimum table:

| source/reference \(i\) | target stream \(j\) | self bits/symbol | conditional bits/symbol | gain | stability by block |
|---|---|---:|---:|---:|---:|
| \(X_i\) | \(X_j\) | \(L_j^{self}\) | \(L_j^{cond(i)}\) | \(G_{i\to j}\) | statistic |
| ... | ... | ... | ... | ... | ... |

This table can guide:

- group formation;
- candidate reference sets;
- branch organization;
- pairwise ablation priority.

It does **not** prove causality or justify subtraction.

---

# 9. Reference-set selection

For target stream \(X_i\), candidate reference set:

\[
R_i
\subseteq
\{X_j:j\neq i\}.
\]

Possible selection rules, all training-only:

1. prior/domain grouping;
2. conditional code gain;
3. partial correlation;
4. mutual information;
5. coherence from STEP 06;
6. cross-validation performance of a reference predictor;
7. sparse regression selection.

Test data must never determine \(R_i\).

---

# 10. Baseline R0 — no cancellation

\[
B_{raw}=\mathbf X.
\]

This is mandatory.

---

# 11. Baseline R1 — pairwise linear residualization

For one stream:

\[
X_{i,t}
=
\beta_0
+
\beta^T R_{i,t}
+
U_{i,t}.
\]

Fit \(\beta\) on training only.

Then:

\[
\hat C_{i,t}
=
\hat\beta_0
+
\hat\beta^T R_{i,t}
\]

and:

\[
U_{i,t}
=
X_{i,t}
-
\hat C_{i,t}.
\]

Compare:

\[
X_i
\]

vs.

\[
U_i
\]

vs.

\[
[X_i,\hat C_i,U_i].
\]

Use Ridge/Elastic Net when references are collinear.

---

# 12. Contemporaneous versus lagged reference models

## Contemporaneous

\[
X_{i,t}
=
f(
X_{-i,t}
)
+
U_{i,t}.
\]

Valid for forecasting future targets if all \(X_{-i,t}\) are genuinely available at the decision timestamp.

## Lagged

\[
X_{i,t}
=
f(
X_{-i,\le t-1}
)
+
U_{i,t}.
\]

Stronger temporal-causality interpretation.

Both should be tested.

Do not interpret contemporaneous residualization as causal influence.

---

# 13. VAR innovation baseline

For multivariate process:

\[
\mathbf X_t
=
A_1\mathbf X_{t-1}
+
\cdots
+
A_p\mathbf X_{t-p}
+
\boldsymbol\epsilon_t.
\]

The innovation:

\[
\boxed{
\boldsymbol\epsilon_t
=
\mathbf X_t
-
\sum_{\ell=1}^{p}
A_\ell\mathbf X_{t-\ell}
}
\]

is the part not explained by multivariate past.

This is a strong low-cost baseline.

Important:

\[
\boldsymbol\epsilon_t
\]

is a **multivariate temporal innovation**, not specifically a shared/private residual.

It bridges STEP 05 and STEP 09.

---

# 14. Regularized VAR / reduced-rank regression

When dimensionality is large, full VAR estimation becomes unstable.

Candidate controls:

- Ridge VAR;
- Lasso VAR;
- Elastic Net VAR;
- reduced-rank VAR/regression.

Recent robust reduced-rank work remains relevant for high-dimensional MTS.

The project should prefer library implementations over custom solvers.

---

# 15. PCA as a static common-factor baseline

Standardized training matrix:

\[
X
\in
\mathbb R^{T\times m}.
\]

PCA:

\[
X
\approx
F\Lambda^T.
\]

Common reconstruction:

\[
C
=
F\Lambda^T.
\]

Residual:

\[
U=X-C.
\]

Vary:

\[
k
=
\#\text{components}.
\]

The project already contains PCA-based evidence/proxy infrastructure in `agent-multi`; reuse it rather than creating a separate PCA implementation.

---

# 16. PCA interpretation warning

PCA maximizes explained variance.

It does **not** guarantee:

- source independence;
- target relevance;
- causal factors;
- optimal forecasting components.

A high-variance common factor may be useless for the target; a low-variance residual may be highly predictive.

---

# 17. Dynamic factor models

The generalized dynamic factor model writes a large set of time series as:

\[
X_{i,t}
=
\chi_{i,t}
+
\xi_{i,t},
\]

where:

- \(\chi_{i,t}\): dynamic common component;
- \(\xi_{i,t}\): idiosyncratic component.

Forni, Hallin, Lippi and Reichlin explicitly developed this for high-dimensional macroeconomic panels with cross-sectional and temporal dynamics.

This is one of the strongest state-of-the-art **classical** analogues for STEP 09.

---

# 18. Dynamic factor relevance to finance

Large macroeconomic factor models have been used successfully for forecasting and macro-financial analysis.

Stock–Watson diffusion-index forecasting summarizes hundreds of predictors into a small set of PCA factors.

FRED-MD/FRED-QD provide standardized public large-dimensional macroeconomic datasets where extracted factors have documented forecasting value.

Large-dimensional dynamic factor models have also been applied to stock-return forecasting.

Therefore FRED-MD becomes a useful **external STEP-09 factor-decomposition benchmark** in addition to ETT/PEMS-style neural benchmarks.

---

# 19. Recommended factor benchmark stack

## Synthetic

Known:

\[
X=\Lambda F+U.
\]

## FRED-MD

Large monthly macro panel with public vintages.

## Project financial data

Technical/fundamental/cross-asset feature families.

Questions:

- does estimated common factor recover known synthetic \(F\)?
- does private residual retain target information?
- are factor loadings stable across time/regimes?

---

# 20. Robust PCA

Robust PCA / Principal Component Pursuit models:

\[
\boxed{
X=L+S
}
\]

with:

- \(L\): low-rank component;
- \(S\): sparse component.

Canonical convex formulation:

\[
\min_{L,S}
\|L\|_*
+
\lambda\|S\|_1
\]

subject to:

\[
X=L+S.
\]

This is valuable for separating:

- broad common structure;
- sparse shocks/anomalies.

---

# 21. Robust-PCA warning for financial data

A sparse component:

\[
S
\]

may contain:

- macro release shock;
- crash;
- jump;
- liquidity event;
- regime transition.

These are often **precisely** the observations we care about.

Therefore never treat:

\[
S=\text{noise}
\]

by default.

Minimum representation:

\[
\boxed{
[X,L,S].
}
\]

---

# 22. ICA — blind source separation

Assume:

\[
\mathbf X
=
A\mathbf S
\]

where latent components:

\[
S_1,\ldots,S_k
\]

are statistically independent and non-Gaussian.

ICA seeks:

\[
W
\]

such that:

\[
\hat{\mathbf S}
=
W\mathbf X
\]

has maximally independent components.

Hyvärinen and Oja provide the canonical review.

Potential STEP-09 use:

- blind separation of latent drivers;
- feature extraction;
- cross-channel source decomposition.

---

# 23. ICA assumptions and identifiability warnings

ICA requires strong assumptions.

Problems include:

- Gaussian sources are not identifiable by ordinary ICA;
- independent sources may not exist;
- component order/sign are arbitrary;
- nonstationarity can change the mixing matrix;
- financial factors are often dependent.

ICA therefore begins as a synthetic/diagnostic method, not a default financial preprocessor.

---

# 24. SOBI — time-series-aware blind separation

Second-Order Blind Identification (SOBI) exploits multiple lagged covariance matrices and temporal coherence rather than only higher-order instantaneous statistics.

This makes SOBI especially relevant to time series.

Synthetic test:

\[
X=A S
\]

where latent \(S_k\) have distinct autocorrelation structures.

Compare:

- PCA;
- ICA;
- SOBI.

---

# 25. Independent Vector Analysis (IVA)

IVA extends ICA to multidimensional source components and can preserve dependencies within each source vector while separating source vectors from one another.

It is established in multichannel blind source separation and data fusion.

Potential relevance:

- grouped frequency-domain components;
- multiple representations of the same latent driver;
- multi-view/multi-asset source separation.

IVA should be considered only after ICA/SOBI synthetic gates pass.

---

# 26. CCA — shared subspace rather than subtraction

For two views:

\[
X,
Y,
\]

CCA finds directions:

\[
a^TX
\]

and:

\[
b^TY
\]

with maximum correlation.

This explicitly models:

\[
\boxed{
\text{shared information between two views}.
}
\]

The orthogonal/residual subspaces can then represent private components.

CCA is therefore a natural bridge between side information and common/private decomposition.

---

# 27. Deep CCA

DCCA learns nonlinear transforms:

\[
f_\theta(X),
g_\phi(Y)
\]

whose outputs are maximally correlated.

Use only after linear CCA establishes that a shared subspace exists.

DCCA should not be the first implementation.

---

# 28. Shared/private neural representations

Domain Separation Networks formalized a common/private representation pattern:

\[
Z
=
[
Z_{shared},
Z_{private}
]
\]

with:

- shared representation encouraged to align;
- private representations preserving domain-specific information;
- orthogonality/difference loss;
- reconstruction regularization.

This is conceptually very close to the safe STEP-09 architecture.

However:

> STEP 09 should not create a second generic autoencoder.

If a deep shared/private experiment is eventually justified, extend the existing feature-extractor/plugin infrastructure or add a shared/private head to it rather than create an unrelated reconstruction stack.

---

# 29. Modern time-series disentanglement prior art

Recent multivariate time-series work already shows multiple relevant directions.

## CauDiTS — ICML 2024

Separates:

- domain-common causal rationales;
- domain-specific correlations.

Although focused on unsupervised domain adaptation/classification, it is highly relevant to the common/private decomposition idea.

## DisenTS — 2024/ongoing literature

Uses multiple specialist forecasters and routing to model diverse channel-evolving patterns instead of forcing all channels through one unified model.

## TimeDRL — ICDE 2024

Disentangles timestamp-level and instance-level representations.

These methods confirm that “disentangle first” is active prior art; the project contribution must come from the full communications/information-processing chain and falsifiable cancellation tests.

---

# 30. Channel independence is a mandatory negative/control baseline

PatchTST's channel-independent design processes each variable independently while sharing model weights.

Its success demonstrates:

\[
\boxed{
\text{cross-channel mixing can itself introduce harmful noise.}
}
\]

Therefore STEP 09 must compare:

\[
\text{channel independent}
\]

against:

\[
\text{channel mixed}
\]

and:

\[
\text{shared/private parallel}.
\]

---

# 31. Cross-channel modeling baseline

iTransformer treats each variate as a token and uses attention to capture multivariate correlations.

Crossformer explicitly models cross-time and cross-dimension dependence.

These are strong controls for the hypothesis:

> perhaps the downstream model can learn the useful shared structure directly without explicit cancellation.

---

# 32. 2025–2026 hybrid channel-independence/mixing prior art

Recent work increasingly combines both paradigms rather than choosing one.

## CSformer — AAAI 2025

Channel independence followed by mixing.

## FusionTimePatch — AAAI 2026

Runs channel-independent and channel-mixed views in parallel and fuses them.

## Dual-channel Transformer — 2026

Uses separate temporal/channel-independent and spatial/channel-dependent encoders with selective fusion.

These results strongly support the STEP-09 safe architecture:

\[
\boxed{
\text{private view}
+
\text{shared/cross-channel view}
}
\]

instead of destructive elimination of one side.

---

# 33. Project-specific implication

The project should **not** start by asking:

> “Which correlated features should we delete?”

Instead ask:

> “Can we expose the common component and the private innovation as separate representations?”

For feature family \(G_k\):

\[
X_{G_k}
\rightarrow
\begin{cases}
C_{G_k}\\
U_{G_k}\\
X_{G_k}
\end{cases}
\]

then branch encoders produce:

\[
Z_{raw,k},
Z_{common,k},
Z_{private,k}.
\]

Core decides fusion.

---

# 34. Common/private branch contract

For group \(G\), decomposition plugin should produce:

## Raw

\[
X_G.
\]

## Shared/common

\[
C_G.
\]

## Private/residual

\[
U_G.
\]

## Optional factor state

\[
F_G.
\]

## Diagnostics

- explained variance;
- conditional code gain;
- residual correlation;
- residual entropy;
- target-independent fit error;
- stability metrics.

---

# 35. Do not remove common factors before target ablation

Initial comparison:

## A

\[
P(X).
\]

## B

\[
P(C).
\]

## C

\[
P(U).
\]

## D

\[
P([C,U]).
\]

## E

\[
P([X,C,U]).
\]

Only if:

\[
P(U)
\]

or:

\[
P([C,U])
\]

reliably dominates raw can destructive substitution be considered.

---

# 36. Residual correlation test

If decomposition is meant to remove shared linear structure, test residual matrix:

\[
U.
\]

Compute:

\[
R_U
=
corr(U).
\]

Compare with:

\[
R_X.
\]

A successful shared-component extractor should often reduce relevant off-diagonal dependence.

But:

> low residual correlation is a diagnostic, not the forecasting objective.

---

# 37. Conditional-code-gain after decomposition

Compute:

\[
G^{before}_{i\to j}
\]

and:

\[
G^{after}_{i\to j}.
\]

If common structure has been separated:

\[
G^{after}
<
G^{before}
\]

for intended relationships.

This connects STEP 05 evidence with STEP 09 decomposition.

---

# 38. Target information after decomposition

For each branch representation:

\[
R\in\{X,C,U,[C,U]\},
\]

measure downstream forecasting value.

If feasible, use conditional target-information proxies:

\[
I(R;Y\mid other\ branches)
\]

or supervised ablations.

No component is classified as nuisance without target evidence.

---

# 39. Synthetic benchmark S0 — exact adaptive cancellation

Generate:

\[
d=s+n_0
\]

and reference:

\[
r=n_1
\]

with:

\[
corr(n_0,n_1)>0
\]

and:

\[
corr(s,r)=0.
\]

Compare:

- no cancellation;
- fixed regression;
- LMS;
- NLMS;
- RLS.

Metrics:

- output SNR;
- desired-signal distortion;
- residual interference power;
- convergence speed.

---

# 40. Synthetic benchmark S1 — contaminated reference

Generate:

\[
r=n_1+\alpha s.
\]

Sweep:

\[
\alpha.
\]

Measure:

\[
\boxed{
\text{signal cancellation vs reference contamination}.
}
\]

This is one of the most important STEP-09 experiments.

Expected:

as \(\alpha\) increases, destructive cancellation risk increases.

---

# 41. Synthetic benchmark S2 — static factor mixture

\[
X
=
\Lambda F+U.
\]

Known \(F\), \(\Lambda\), \(U\).

Compare:

- PCA;
- factor analysis;
- ICA;
- CCA where views exist.

Measure recovery:

- subspace angle;
- factor correlation;
- residual reconstruction;
- downstream target prediction.

---

# 42. Synthetic benchmark S3 — dynamic common factors

\[
X_{i,t}
=
\sum_{k,\ell}
b_{ik\ell}
F_{k,t-\ell}
+
U_{i,t}.
\]

Compare:

- PCA;
- VAR;
- dynamic factor model;
- temporal neural baseline.

This tests whether static PCA misses delayed common structure.

---

# 43. Synthetic benchmark S4 — independent-source mixture

\[
X=A S.
\]

Sources are non-Gaussian and independent.

ICA should recover sources up to:

- sign;
- scale;
- permutation.

This is the proper place to validate ICA.

---

# 44. Synthetic benchmark S5 — temporally distinct latent sources

Create sources with distinct autocorrelation.

SOBI should have an advantage.

---

# 45. Synthetic benchmark S6 — low-rank + sparse events

\[
X=L+S.
\]

Sparse \(S\) contains known event pulses.

Compare:

- PCA;
- Robust PCA.

Then test two targets:

## Target A

depends on \(L\).

## Target B

depends on sparse \(S\).

This demonstrates why “remove the sparse component” is not universally valid.

---

# 46. Synthetic benchmark S7 — channel-mixing relevance

Generate two cases.

## Case A — cross-channel structure is useful

\[
Y=f(C).
\]

## Case B — private structure is useful

\[
Y=f(U).
\]

Compare channel-independent, channel-mixed and shared/private models.

This directly tests the modern CI-versus-CM debate.

---

# 47. Financial benchmark FRED-MD

FRED-MD is a strong external benchmark because:

- it is public;
- it contains a large macro panel;
- factor extraction is established;
- real-time vintages exist;
- factors have documented forecasting utility.

Use it to test:

\[
\text{raw panel}
\]

vs.

\[
\text{common factors}
\]

vs.

\[
\text{idiosyncratic residuals}.
\]

Do not tune on the same historical vintages used for final comparison.

---

# 48. Generic MTS public benchmarks

For neural channel dependence/disentanglement:

- ETTh1/ETTh2;
- ETTm1/ETTm2;
- Electricity;
- Traffic;
- Weather;
- Solar;
- PEMS03/04/07/08.

These overlap STEPS 06–08 and enable cross-step comparability.

---

# 49. Project financial benchmark

Initial cells:

- EURUSD 1 h;
- EURUSD 4 h;
- ETHUSDT 4 h;
- selected technical/fundamental/cross-asset groups.

Do not pool all features into one factor model blindly.

Use semantically coherent groups first.

---

# 50. Candidate project grouping

Possible initial groups:

## Price/return family

- OHLC-derived;
- returns;
- range;
- realized volatility.

## Technical/statistical family

- rolling indicators;
- trend;
- momentum;
- volatility.

## Cross-asset family

related assets/indexes.

## Macro/fundamental family

rates, inflation, activity, volatility indexes where point-in-time available.

## Calendar/event family

keep mostly separate because their semantics are not continuous sensor mixtures.

Grouping must be frozen from training/domain knowledge, not test behavior.

---

# 51. FX/market-factor example

For multiple FX returns, common drivers can include broad:

- USD factor;
- risk-on/risk-off component;
- rate-differential component.

For equities:

- market factor;
- sector factor;
- idiosyncratic residual.

The project should treat these as **candidate common components**, not as interference a priori.

---

# 52. Adaptive factor cancellation

If loadings vary:

\[
\Lambda=\Lambda_t,
\]

fixed factor residualization becomes stale.

Candidate adaptive methods:

- rolling Ridge;
- recursive least squares;
- Kalman/state-space dynamic loadings;
- causal online factor updates.

Compare fixed versus adaptive on controlled regime-shift data.

---

# 53. State-space common/private model

Possible linear Gaussian model:

\[
F_t=AF_{t-1}+w_t
\]

and:

\[
X_t=\Lambda F_t+u_t.
\]

Kalman filtering yields causal factor estimates.

This is a strong model-based alternative to neural shared/private representations.

---

# 54. Robust dynamic decomposition

Financial time series contain heavy tails/outliers.

Candidate robust versions:

- Huber regression;
- Student-\(t\) state-space innovations;
- robust covariance;
- robust PCA;
- quantile/median regression.

Do not assume Gaussian residuals.

---

# 55. Reference-feature leakage rule

A reference series may be used at time \(t\) only if available at time \(t\).

For macro/fundamental data:

- release timestamp;
- vintage;
- revision policy

must be honored.

A revised macro series is not valid historical side information unless the vintage available at \(t\) is used.

---

# 56. Same-bar simultaneity rule

If all market features for bar \(t\) are finalized before prediction for:

\[
t+h,
\]

same-bar cross-feature decomposition is endpoint-causal.

However:

> same-bar relationships do not establish causal direction.

STEP 10 will later handle dynamic timing/alignment more explicitly.

---

# 57. STEP 10 boundary — no hidden synchronization inside STEP 09

Do not optimize arbitrary pairwise delays:

\[
\tau_{ij}
\]

inside STEP 09 merely to improve cancellation.

Allowed in STEP 09:

- fixed causal lags specified by model order;
- historical past context.

Dynamic temporal alignment/timing recovery belongs to STEP 10.

This preserves historical/conceptual separation.

---

# 58. STEP 08/09 order experiment

Because equalization can change cross-series dependence:

\[
E(C(X))
\neq
C(E(X)).
\]

Compare:

## Order A

\[
X
\rightarrow
\text{equalization}
\rightarrow
\text{common/private decomposition}.
\]

## Order B

\[
X
\rightarrow
\text{common/private decomposition}
\rightarrow
\text{equalization}.
\]

Default hypothesis:

> STEP 08 canonicalization first should often improve comparability across channels, but this must be tested.

---

# 59. STEP 03/09 interaction

Noise can inflate apparent cross-series relationships or mask them.

Compare decomposition on:

\[
X
\]

versus:

\[
D(X).
\]

But do not assume denoising first is universally correct.

A common weak component could be accidentally removed by the denoiser.

---

# 60. STEP 06/09 interaction

Common/private decomposition can operate in:

- time domain;
- frequency domain;
- wavelet bands;
- spectral factor space.

For cross-spectrum:

\[
S_X(f)
\]

one may seek frequency-dependent common factors.

This is advanced and should follow the time-domain baseline.

---

# 61. Frequency-domain factor decomposition

At each frequency:

\[
\mathbf S_X(f)
\]

can reveal common dynamic factors.

This connects directly to the generalized dynamic-factor model and STEP 06.

Potential representation:

\[
C(f),U(f).
\]

Do not implement until static/dynamic time-domain factor baselines pass.

---

# 62. STEP 07/09 interaction

Pattern detector outputs:

\[
A_t
\]

may themselves share common activation structure.

Potential later decomposition:

\[
A_t
=
A_t^{common}
+
A_t^{private}.
\]

This may identify:

- market-wide motif;
- asset-specific motif.

Not a primary STEP-09 experiment.

---

# 63. Existing `agent-multi` reuse

`agent-multi` already includes PCA-based evidence/proxy paths and deterministic evidence-screen support.

Therefore:

- reuse existing PCA implementation/configuration where compatible;
- do not create another project-specific PCA engine;
- extend evidence artifacts only as needed for common/private decomposition.

The repository does not currently surface a dedicated ICA/dynamic-factor/shared-private cancellation module in the inspected code-search results.

---

# 64. Strong low-cost baseline ladder

Before deep disentanglement:

1. no cancellation;
2. pairwise/multivariate Ridge residualization;
3. VAR innovations;
4. PCA common/private;
5. dynamic factor model;
6. robust PCA;
7. ICA/SOBI synthetic gates;
8. CCA for grouped views.

Only after these:

9. shared/private neural representation;
10. disentangled MoE;
11. deep multivariate models.

---

# 65. Modern channel-model baseline ladder

For public MTS forecasting, compare:

## Channel independent

PatchTST-like.

## Channel mixed

iTransformer / Crossformer-like.

## Hybrid CI + CM

CSformer / FusionTimePatch / dual-channel reference.

## Explicit common/private decomposition

proposed STEP-09 branch.

This is essential because a hybrid neural architecture may already capture the benefit without explicit residualization.

---

# 66. Common/private deep model — only if gated

If classical evidence justifies it, extend existing feature-extractor infrastructure.

Conceptual outputs:

\[
Z_s
=
E_{shared}(X)
\]

and:

\[
Z_{p,i}
=
E_{private,i}(X_i).
\]

Possible regularizers:

## Orthogonality

\[
L_{orth}
=
\sum_i
\|
Z_s^TZ_{p,i}
\|_F^2.
\]

## Reconstruction

\[
L_{rec}
=
\|
X-\hat X(Z_s,Z_p)
\|^2.
\]

## Redundancy reduction

penalize duplicated private codes.

## Task loss

\[
L_{task}.
\]

Total:

\[
L
=
L_{task}
+
\lambda_oL_{orth}
+
\lambda_rL_{rec}.
\]

This is not authorized for implementation until classical gates pass.

---

# 67. Target-aware shared/private selection

Even if decomposition itself is unsupervised, branch promotion uses validation target evidence.

For component \(R\), define incremental utility:

\[
\Delta P_R
=
P([Base,R])
-
P(Base).
\]

Promote only if:

\[
\Delta P_R
\]

is stable across:

- seeds;
- blocks;
- horizons.

---

# 68. Common-factor attribution

For interpretability:

\[
C_{i,t}
=
\sum_k
\lambda_{ik}F_{k,t}.
\]

Store:

- factor loading;
- explained share;
- factor time series;
- top contributing original features;
- factor-target association;
- regime stability.

Do not assign economic names automatically.

---

# 69. Private-residual diagnostics

For:

\[
U_i,
\]

report:

- variance ratio;
- autocorrelation;
- entropy rate;
- cross-correlation;
- conditional code gain;
- spectral density;
- predictability;
- target relevance.

A useful private component should ideally expose structure not duplicated in the common branch.

---

# 70. Cancellation efficiency

For known synthetic interference:

\[
\eta_C
=
1
-
\frac{
P_{residual\ interference}
}{
P_{original\ interference}
}.
\]

Also report desired-signal distortion:

\[
D_S
=
E[
(s-\hat s)^2
].
\]

A canceller with high \(\eta_C\) but high \(D_S\) is unacceptable.

---

# 71. Financial analogue of signal-distortion metric

Because true desired signal is unknown, use parallel controls:

1. downstream OOS forecasting;
2. event/tail retention;
3. raw+private/common ablation;
4. target-conditioned incremental utility;
5. known synthetic mixture experiments.

---

# 72. Falsifiable hypotheses — adaptive cancellation

## H9.1 — Valid reference cancellation improves SNR

Under synthetic:

\[
corr(r,n)>0
\]

and:

\[
corr(r,s)=0,
\]

adaptive cancellation improves output SNR.

**Falsified if:** LMS/NLMS/RLS implementation fails controlled behavior.

---

## H9.2 — Reference contamination causes desired-signal attenuation

As:

\[
corr(r,s)
\]

increases, desired-signal distortion increases.

**Falsified if:** controlled contamination has no effect.

This is a mandatory protective result.

---

## H9.3 — Adaptive cancellation beats fixed regression under time-varying mixing

When mixing coefficients drift:

\[
a=a_t,
\]

RLS/causal adaptive regression should outperform fixed coefficients.

**Falsified if:** fixed model remains equivalent.

---

# 73. Falsifiable hypotheses — side information

## H9.4 — Conditional code gain predicts explainable shared structure

Pairs/groups with high:

\[
G_{i\to j}
\]

should exhibit larger reduction in source residual redundancy after conditioning.

**Falsified if:** code gain does not correspond to decomposition behavior.

---

## H9.5 — Conditional code gain alone does not justify cancellation

High \(G_{i\to j}\) can coexist with high target value in the shared component.

This is a deliberate negative/control hypothesis.

---

# 74. Falsifiable hypotheses — common/private decomposition

## H9.6 — Parallel common/private representation can outperform raw-only

\[
P([X,C,U])
>
P(X).
\]

**Falsified if:** explicit decomposition is always redundant.

---

## H9.7 — Private-only representation is not universally superior

\[
P(U)
\]

should underperform whenever common factors contain target information.

This is a protective hypothesis.

---

## H9.8 — Common-only representation can improve parameter efficiency for highly redundant panels

For large \(m\):

\[
P(C)
\approx
P(X)
\]

with lower representation dimension.

**Falsified if:** factor compression consistently loses important information.

---

# 75. Falsifiable hypotheses — static versus dynamic factors

## H9.9 — Dynamic factors outperform static PCA when common structure has lagged dynamics

\[
P(C_{dynamic},U_{dynamic})
>
P(C_{PCA},U_{PCA}).
\]

**Falsified if:** static PCA is sufficient.

---

## H9.10 — PCA remains a strong low-cost control

A complex common/private method must provide incremental OOS value beyond PCA at justified compute cost.

**Falsified if:** PCA is too weak even in synthetic low-rank conditions.

---

# 76. Falsifiable hypotheses — robust PCA

## H9.11 — Robust PCA better separates low-rank common structure under sparse contamination

Synthetic recovery:

\[
D(L,\hat L_{RPCA})
<
D(L,\hat L_{PCA}).
\]

**Falsified if:** robust PCA does not improve in its intended regime.

---

## H9.12 — Sparse component may carry target information

\[
P([L,S])
>
P(L)
\]

for event-driven targets.

**Falsified if:** sparse events never contribute.

---

# 77. Falsifiable hypotheses — blind source separation

## H9.13 — ICA recovers independent non-Gaussian sources in controlled mixtures

Recovery correlation/subspace metric exceeds PCA.

**Falsified if:** implementation fails the ICA-identifiable case.

---

## H9.14 — SOBI improves separation when sources differ primarily in temporal covariance

**Falsified if:** temporal second-order structure gives no benefit.

---

## H9.15 — ICA/SOBI need not improve financial forecasting

If real financial latent drivers violate independence/stationarity assumptions, BSS may not help.

This is another explicit protective hypothesis.

---

# 78. Falsifiable hypotheses — channel-independence/mixing

## H9.16 — Best architecture depends on shared/private structure

Channel-independent models should do better when cross-channel relations are noisy/irrelevant.

Channel-mixed models should improve when true common drivers are target-relevant.

**Falsified if:** one paradigm dominates all controlled cases.

---

## H9.17 — Hybrid CI+CM is a strong benchmark for explicit decomposition

A STEP-09 common/private architecture must beat or materially complement a modern hybrid CI+CM control.

**Falsified if:** explicit decomposition adds no benefit beyond hybrid neural mixing.

---

# 79. Falsifiable hypotheses — capacity

## H9.18 — Explicit decomposition can reduce required core capacity

At target performance \(P_0\):

\[
|\theta_{decomposed+core}|
<
|\theta_{raw-core}|.
\]

**Falsified if:** explicit decomposition requires equal/larger capacity.

---

# 80. Falsifiable hypotheses — transfer and stability

## H9.19 — Common factors transfer better across related domains than private residuals

For related assets/domains:

\[
Transfer(C)
>
Transfer(U)
\]

on average.

**Falsified if:** no differential transfer exists.

---

## H9.20 — Private residuals can improve asset-specific heads

Asset-specific task:

\[
P([C,U_i])
>
P(C).
\]

**Falsified if:** idiosyncratic residual never helps.

---

## H9.21 — Fixed loadings degrade under regime-changing mixing

Adaptive/dynamic factor models should gain under loading drift.

**Falsified if:** static loadings remain stable enough.

---

# 81. Falsifiable hypotheses — STEP interactions

## H9.22 — STEP-08 equalization can improve common/private separation

For source-specific affine/spectral distortions:

\[
Sep(E(X))
>
Sep(X).
\]

**Falsified if:** equalization does not improve decomposition.

---

## H9.23 — Denoising before decomposition is not universally optimal

Some weak shared predictive structures can be lost by aggressive STEP-03 denoising.

**Falsified if:** denoising-first universally dominates.

---

## H9.24 — Decomposition improves detector portability

STEP-07 detector trained on common component may transfer across related assets/domains better than detector trained on raw inputs.

**Falsified if:** no transfer benefit exists.

---

# 82. Falsifiable hypotheses — target-relative cancellation

## H9.25 — A shared component should be cancelled only when its incremental target value is low

Define:

\[
\Delta P_C
=
P([Base,C])-P(Base).
\]

Destructive cancellation is permitted only if:

\[
\Delta P_C
\approx0
\]

or negative under predeclared equivalence margins.

**Falsified if:** cancellation is beneficial even when shared component has strong independent target information — which would indicate another mechanism needs investigation.

---

# 83. Minimum experiment matrix

| ID | Method | Output | Synthetic | Public MTS | Finance |
|---|---|---|---:|---:|---:|
| I00 | Raw | \(X\) | Yes | Yes | Yes |
| I01 | Ridge residual | \(C,U\) | Yes | Yes | Yes |
| I02 | VAR innovation | \(\hat X,\epsilon\) | Yes | Yes | Yes |
| I03 | PCA | \(C,U,F\) | Yes | Yes | Yes |
| I04 | Dynamic factor | \(C,U,F\) | Yes | FRED-MD | Yes |
| I05 | Robust PCA | \(L,S\) | Yes | Optional | Yes |
| I06 | ICA | \(S\) | Yes | Optional | Gate |
| I07 | SOBI | \(S\) | Yes | Optional | Gate |
| I08 | CCA | shared/private | Yes | Yes | Yes |
| I09 | Channel-independent | representation | Controlled | Yes | Yes |
| I10 | Channel-mixed | representation | Controlled | Yes | Yes |
| I11 | Hybrid CI+CM | representation | Controlled | Yes | Yes |
| I12 | Explicit shared/private | \(C,U\) | Controlled | Gate | Gate |

---

# 84. Public benchmark strategy

## Layer A — identifiability

Synthetic mixtures with known truth.

## Layer B — macro factor benchmark

FRED-MD.

## Layer C — multivariate forecasting

ETT, Electricity, Weather, Traffic, Solar, PEMS.

## Layer D — project data

EURUSD/ETHUSDT and governed feature families.

This separates source-separation validity from forecasting utility.

---

# 85. Metrics — source decomposition

## Common recovery

Subspace angle / correlation with oracle factors.

## Private recovery

\[
corr(U,\hat U).
\]

## Explained ratio

\[
R_C
=
\frac{
Var(C)
}{
Var(X)
}.
\]

## Residual cross-dependence

\[
\|corr(U)-I\|.
\]

## Conditional code gain reduction

\[
G^{before}-G^{after}.
\]

---

# 86. Metrics — task preservation

- MAE;
- RMSE;
- \(R^2\);
- per-horizon metrics;
- NLL/calibration for probabilistic outputs;
- direction metrics only as secondary where appropriate;
- event/tail metrics;
- seed stability.

---

# 87. Metrics — cancellation safety

- desired-signal distortion on synthetic data;
- tail-event retention;
- common-branch target utility;
- private-branch target utility;
- raw+decomposed uplift;
- distribution shift of residuals.

---

# 88. Statistical validation

Use:

- block/stationary bootstrap;
- Diebold–Mariano where appropriate;
- multiple seeds;
- FDR correction for broad decomposition grids;
- predeclared primary hypotheses.

Do not bootstrap timestamps iid.

---

# 89. Component-count selection

For PCA/factor methods:

\[
k
\]

must be selected with training/validation only.

Candidate criteria:

- explained variance curve;
- Bai–Ng information criteria where appropriate;
- validation forecasting performance;
- stability across temporal blocks.

Do not use test to pick \(k\).

---

# 90. Factor sign/permutation invariance

Factor components may be identifiable only up to:

- sign;
- rotation;
- permutation

depending on method.

Comparisons must use invariant metrics:

- subspace distance;
- canonical correlations;
- best permutation matching.

Do not compare raw factor column indices blindly across runs.

---

# 91. ICA reproducibility

Because ICA components can reorder/sign-flip:

- align components across seeds;
- report stability;
- reject unstable components from interpretation.

---

# 92. Regime stability

For each component:

\[
F_k
\]

compare across temporal blocks:

- loadings;
- variance;
- target association;
- code gain;
- detector activation.

A common component that changes meaning every block is not a stable reusable representation.

---

# 93. Online adaptation

After static baseline:

\[
\Lambda_t,
\beta_t
\]

may update causally.

Allowed methods:

- RLS;
- rolling Ridge;
- Kalman filter.

All updates use only data available by time \(t\).

---

# 94. No hidden target adaptation

Source decomposition is initially unsupervised with respect to future target.

Validation target is used only to decide:

- representation promotion;
- hyperparameters;
- branch inclusion.

Do not use future target directly to compute a contemporaneous residual representation.

---

# 95. Information graph

Construct directed graph:

\[
G=(V,E)
\]

with:

\[
w_{i\to j}
=
G_{i\to j}
\]

from STEP 05 conditional code gain.

Use it only to prioritize groups/references.

No causal label is assigned.

---

# 96. Factor/reference groups from graph communities

Optional training-only grouping:

- threshold graph;
- community detection;
- spectral clustering.

Then factor decomposition occurs within each group.

This may prevent one giant global factor model from mixing semantically unrelated variables.

---

# 97. Grouping control

Compare:

## Global factor model

all channels.

## Semantic groups

domain-defined.

## Code-gain groups

data-driven.

## Independent channels

no mixing.

The best grouping is empirical.

---

# 98. Architecture integration

For group \(g\):

\[
X_g
\rightarrow
D_g
\rightarrow
(C_g,U_g,F_g).
\]

Branch encoders:

\[
Z_{raw,g}=E_{raw,g}(X_g)
\]

\[
Z_{common,g}=E_{common,g}(C_g)
\]

\[
Z_{private,g}=E_{private,g}(U_g).
\]

Fusion:

\[
Z
=
Core(
\{Z_{raw,g},Z_{common,g},Z_{private,g}\}_g
).
\]

Predictive heads remain existing project heads initially.

---

# 99. Parameter-match requirement

When adding common/private branches, match total capacity against a raw-only model.

Otherwise improved performance can be caused merely by:

\[
\#parameters\uparrow.
\]

---

# 100. Core freeze rule

The first STEP-09 ablation freezes:

- target definition;
- horizons;
- core architecture;
- output heads;
- optimizer;
- early stopping;
- sequence length.

Only decomposition changes.

---

# 101. RL integration is late-stage

Do not send decomposed components directly into SAC/PPO/DQN until:

1. forecasting/state-prediction evidence exists;
2. component stability exists;
3. no leakage exists;
4. validation utility is reproducible.

RL reward can mask whether decomposition itself is useful.

---

# 102. Event-token integration

If factor/private changes are converted to events, possible tokens include:

- `COMMON_FACTOR_SHOCK`;
- `PRIVATE_RESIDUAL_SPIKE`;
- `FACTOR_LOADING_SHIFT`;
- `CROSS_CHANNEL_DECOUPLING`.

Use existing event-token infrastructure rather than create a new sequence model.

---

# 103. Major failure modes

## 103.1. Signal cancellation

Reference contains desired signal.

## 103.2. Common-factor deletion

Shared component carries target value.

## 103.3. Private-noise glorification

Residual is mostly noise but is treated as “unique information.”

## 103.4. PCA variance fallacy

High explained variance mistaken for predictive value.

## 103.5. ICA assumption failure

Latent sources not independent/non-Gaussian enough.

## 103.6. Robust-PCA event deletion

Sparse shocks removed despite predictive importance.

## 103.7. Leakage through contemporaneous unavailable references

Reference not actually known at prediction time.

## 103.8. Dynamic-lag leakage

STEP-10 alignment accidentally optimized on future.

## 103.9. Factor instability

Components rotate/change meaning across regimes.

## 103.10. Parameter-count confounding

More branches = more capacity.

## 103.11. Grouping overfit

Reference groups chosen using validation/test repeatedly.

## 103.12. Revised macro data leakage

Non-vintage data used historically.

---

# 104. Reuse matrix

| Problem | Prior art / existing resource | Action |
|---|---|---|
| Conditional side information | STEP 05 H5.8 | Generate table first; no new model |
| PCA proxy | current `agent-multi` evidence screen | Reuse |
| Adaptive cancellation | LMS/NLMS/RLS libraries | Reuse |
| Static residualization | scikit-learn Ridge/ElasticNet | Reuse |
| VAR | statsmodels or validated library | Reuse |
| Static factors | sklearn/statsmodels | Reuse |
| Dynamic factor | statsmodels / established econometric implementation | Reuse |
| Robust PCA | established PCP/RPCA implementation | Benchmark; no custom solver initially |
| ICA | sklearn FastICA / established library | Reuse |
| SOBI | validated BSS implementation | Reuse |
| IVA | established IVA library | Late-stage |
| CCA | sklearn CCA | Reuse |
| DCCA | public/reference implementation | Only after CCA gate |
| Channel-independent model | PatchTST | Benchmark |
| Channel-mixed model | iTransformer/Crossformer | Benchmark |
| Hybrid CI+CM | CSformer / FusionTimePatch | Benchmark/reference |
| Deep common/private | DSN/CauDiTS concepts | Extend existing extractor only if gated |

---

# 105. Recommended implementation order

1. Generate STEP-05 conditional code-gain table.
2. Synthetic adaptive-cancellation oracle.
3. Reference-contamination sweep.
4. Ridge residualization.
5. VAR innovations.
6. PCA common/private.
7. Dynamic factor benchmark.
8. FRED-MD factor experiment.
9. Robust PCA synthetic shock test.
10. ICA synthetic identifiability test.
11. SOBI temporal-source test.
12. CCA grouped-view test.
13. Channel-independent vs channel-mixed controls.
14. Hybrid CI+CM control.
15. Financial groupwise common/private ablation.
16. Adaptive loadings/RLS.
17. Only then evaluate a deep shared/private extension.

---

# 106. Decision gates

## Gate 9A — Side-information evidence

Conditional code-gain table shows stable candidate relationships.

## Gate 9B — Cancellation correctness

Synthetic adaptive cancellation recovers expected behavior.

## Gate 9C — Contamination safety

Reference-contamination failure mode is correctly detected.

## Gate 9D — Classical decomposition value

At least one Ridge/VAR/PCA/factor representation adds validation value.

## Gate 9E — Dynamic factor justification

Dynamic model beats static PCA where lagged common dynamics exist.

## Gate 9F — BSS justification

ICA/SOBI pass synthetic identifiability before finance.

## Gate 9G — Channel baseline

Explicit decomposition competes with strong CI/CM/hybrid neural controls.

## Gate 9H — Information preservation

Common and sparse/event branches are not blindly discarded.

## Gate 9I — Public transfer

Effect survives public benchmark.

## Gate 9J — Financial validation

Effect survives project validation.

## Gate 9K — Held-out confirmation

Only preselected frozen configuration reaches final test.

---

# 107. Recommended artifacts

1. `step09_conditional_code_gain.parquet`
2. `step09_reference_groups.json`
3. `step09_synthetic_mixture_spec.json`
4. `step09_adaptive_cancellation_results.parquet`
5. `step09_reference_contamination_surface.parquet`
6. `step09_var_innovations.parquet`
7. `step09_pca_factor_outputs.parquet`
8. `step09_dynamic_factor_outputs.parquet`
9. `step09_rpca_outputs.parquet`
10. `step09_ica_sobi_recovery.parquet`
11. `step09_common_private_metrics.parquet`
12. `step09_target_ablation.parquet`
13. `step09_factor_stability.parquet`
14. `step09_statistical_tests.json`
15. `step09_audit_report.md`
16. reproducibility manifest with:
    - data hashes;
    - vintage identifiers;
    - split boundaries;
    - feature groups;
    - reference sets;
    - factor count;
    - model orders;
    - seeds;
    - git commits;
    - package versions.

---

# 108. State-of-the-art synthesis

The literature produces a clear hierarchy.

## 108.1. Communications already solved the “reference cancellation” logic

Adaptive noise cancellation provides the strongest conceptual warning:

> a reference is useful only when it tracks interference without carrying the desired signal.

That becomes the central anti-signal-destruction control.

## 108.2. Econometrics already has a mature common/idiosyncratic decomposition

Dynamic factor models are established for large macro/financial panels.

Do not replace them with an unvalidated neural latent model by default.

## 108.3. Blind source separation is mature but assumption-heavy

ICA/SOBI/IVA are excellent controlled tools, not universal financial preprocessors.

## 108.4. Modern MTS models show the CI/CM dilemma is real

Channel independence can suppress irrelevant cross-channel noise; channel mixing captures useful shared structure.

Recent hybrid systems increasingly model both in parallel.

## 108.5. The safest architecture is therefore dual

\[
\boxed{
\text{private/channel-specific view}
+
\text{shared/cross-channel view}
}
\]

with raw data initially retained.

---

# 109. Project-specific opportunity

The project-specific contribution is not “use PCA” or “use ICA.”

It is the systematic pipeline:

\[
\boxed{
\text{conditional information map}
\rightarrow
\text{reference/group selection}
\rightarrow
\text{common/private decomposition}
\rightarrow
\text{parallel raw/common/private branches}
\rightarrow
\text{representation-aware detectors}
\rightarrow
\text{shared modular core}
}
\]

combined with the earlier:

- SNR;
- denoising;
- quantization;
- entropy;
- spectral representation;
- equalization stages.

---

# 110. Domain-general interpretation

For general multisensor data:

\[
X_i
=
\text{shared environment}
+
\text{sensor-specific phenomenon}
+
\text{measurement noise}.
\]

For biomedical signals:

\[
X_i
=
\text{common physiological source}
+
\text{local source}
+
\text{artifact}.
\]

For industrial systems:

\[
X_i
=
\text{plant-wide operating state}
+
\text{machine-local state}
+
\text{sensor error}.
\]

For finance:

\[
X_i
=
\text{market/macro common state}
+
\text{asset-specific state}
+
\text{noise}.
\]

The same plugin contract can therefore generalize even when the actual decomposition method differs by domain.

---

# 111. References — IEEE style

[1] B. Widrow, J. R. Glover, J. M. McCool, J. Kaunitz, C. S. Williams, R. H. Hearn, J. R. Zeidler, E. Dong, and R. C. Goodlin, “Adaptive Noise Cancelling: Principles and Applications,” *Proceedings of the IEEE*, vol. 63, no. 12, pp. 1692–1716, Dec. 1975, doi: 10.1109/PROC.1975.10036. Available: https://doi.org/10.1109/PROC.1975.10036

[2] A. Hyvärinen and E. Oja, “Independent Component Analysis: Algorithms and Applications,” *Neural Networks*, vol. 13, no. 4–5, pp. 411–430, 2000, doi: 10.1016/S0893-6080(00)00026-5. Available: https://doi.org/10.1016/S0893-6080(00)00026-5

[3] A. Belouchrani, K. Abed-Meraim, J.-F. Cardoso, and E. Moulines, “A Blind Source Separation Technique Using Second-Order Statistics,” *IEEE Transactions on Signal Processing*, vol. 45, no. 2, pp. 434–444, Feb. 1997, doi: 10.1109/78.554307. Available: https://doi.org/10.1109/78.554307

[4] Z. Luo, “Independent Vector Analysis: Model, Applications, Challenges,” *Pattern Recognition*, 2023. Available through: https://www.sciencedirect.com/science/article/abs/pii/S0031320323000778

[5] M. Forni, M. Hallin, M. Lippi, and L. Reichlin, “The Generalized Dynamic-Factor Model: Identification and Estimation,” *The Review of Economics and Statistics*, vol. 82, no. 4, pp. 540–554, 2000, doi: 10.1162/003465300559037. Available: https://doi.org/10.1162/003465300559037

[6] M. Forni and M. Lippi, “The Generalized Dynamic Factor Model: Representation Theory,” *Econometric Theory*, 2001. Available: https://www.cambridge.org/core/journals/econometric-theory/article/generalized-dynamic-factor-model-representation-theory/72A2EBF07C02C9D8790DF070070E8146

[7] J. H. Stock and M. W. Watson, “Macroeconomic Forecasting Using Diffusion Indexes,” *Journal of Business & Economic Statistics*, vol. 20, no. 2, pp. 147–162, 2002, doi: 10.1198/073500102317351921. Available: https://doi.org/10.1198/073500102317351921

[8] B. S. Bernanke, J. Boivin, and P. Eliasz, “Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach,” *The Quarterly Journal of Economics*, vol. 120, no. 1, pp. 387–422, 2005, doi: 10.1162/0033553053327452. Available: https://doi.org/10.1162/0033553053327452

[9] M. W. McCracken and S. Ng, “FRED-MD: A Monthly Database for Macroeconomic Research,” *Journal of Business & Economic Statistics*, vol. 34, no. 4, pp. 574–589, 2016. Dataset: https://www.stlouisfed.org/research/economists/mccracken/fred-databases

[10] M. W. McCracken and S. Ng, “FRED-QD: A Quarterly Database for Macroeconomic Research,” *Federal Reserve Bank of St. Louis Review*, vol. 103, no. 1, pp. 1–44, 2021, doi: 10.20955/r.103.1-44. Available: https://doi.org/10.20955/r.103.1-44

[11] E. J. Candès, X. Li, Y. Ma, and J. Wright, “Robust Principal Component Analysis?,” *Journal of the ACM*, vol. 58, no. 3, 2011, doi: 10.1145/1970392.1970395. Available: https://doi.org/10.1145/1970392.1970395

[12] G. Andrew, R. Arora, J. Bilmes, and K. Livescu, “Deep Canonical Correlation Analysis,” in *Proceedings of the 30th International Conference on Machine Learning*, PMLR vol. 28, pp. 1247–1255, 2013. Available: https://proceedings.mlr.press/v28/andrew13.html

[13] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan, “Domain Separation Networks,” in *Advances in Neural Information Processing Systems*, vol. 29, 2016. Available: https://proceedings.neurips.cc/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html

[14] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers,” in *International Conference on Learning Representations*, 2023. Official implementation: https://github.com/yuqinie98/PatchTST

[15] Y. Zhang and J. Yan, “Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting,” in *International Conference on Learning Representations*, 2023. Available: https://mlanthology.org/iclr/2023/zhang2023iclr-crossformer/

[16] Y. Liu *et al.*, “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting,” in *International Conference on Learning Representations*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html

[17] J. Lu and S. Sun, “CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series,” in *Proceedings of the 41st International Conference on Machine Learning*, PMLR vol. 235, pp. 33113–33146, 2024. Available: https://proceedings.mlr.press/v235/lu24i.html

[18] Z. Liu *et al.*, “DisenTS: Disentangled Channel Evolving Pattern Modeling for Multivariate Time Series Forecasting,” arXiv:2410.22981, 2024. Available: https://arxiv.org/abs/2410.22981

[19] H. Wang *et al.*, “CSformer: Combining Channel Independence and Mixing for Robust Multivariate Time Series Forecasting,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 20, pp. 21090–21098, 2025, doi: 10.1609/aaai.v39i20.35406. Available: https://doi.org/10.1609/aaai.v39i20.35406

[20] W. Zhang *et al.*, “Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global–Local Representation Synergy in Multivariate Time Series Forecasting,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 33, pp. 28427–28436, 2026, doi: 10.1609/aaai.v40i33.40072. Available: https://doi.org/10.1609/aaai.v40i33.40072

[21] Z. Huang, F. Zhang, and Y. Liu, “Dual-Channel Transformer: Integrating Independence and Dependence for Time Series Forecasting,” *Expert Systems with Applications*, vol. 300, art. 130258, 2026, doi: 10.1016/j.eswa.2025.130258. Available: https://doi.org/10.1016/j.eswa.2025.130258

[22] B.-L. Zhang, A. C. C. Ling, and S. Yan, “Robust Estimation of Multivariate Time Series Data Based on Reduced Rank Model,” *Journal of Forecasting*, vol. 44, no. 2, 2025. Reference landing page: https://doi.org/10.1002/for.3205

[23] Y.-C. Lee *et al.*, “TimeDRL: Disentangled Representation Learning for Multivariate Time-Series,” in *2024 IEEE 40th International Conference on Data Engineering (ICDE)*, 2024, doi: 10.1109/ICDE60146.2024.00054. Available: https://doi.org/10.1109/ICDE60146.2024.00054

[24] M. W. McCracken and S. Ng, “FRED-MD and FRED-QD: Monthly and Quarterly Databases for Macroeconomic Research,” Federal Reserve Bank of St. Louis, continuously updated. Available: https://www.stlouisfed.org/research/economists/mccracken/fred-databases

---

# 112. Final status

**STEP 09 is theoretically specified after state-of-the-art review, incorporates the agent-audit constraints, and is ready for independent audit.**

The central principle is:

\[
\boxed{
\text{Do not cancel correlation; decompose it.}
}
\]

The safe first representation is:

\[
\boxed{
\text{raw}
+
\text{common/shared}
+
\text{private/idiosyncratic}
}
\]

and only experimental target evidence can justify discarding one of those components.

The highest-priority first artifacts are intentionally low-cost:

1. STEP-05 conditional code-gain matrix;
2. synthetic adaptive-cancellation/reference-contamination experiment;
3. Ridge/VAR residuals;
4. PCA/dynamic-factor common/private branches.

Deep disentanglement is gated behind those controls.

After audit, proceed to:

**STEP 10 — synchronization / temporal alignment / timing recovery**, where dynamic inter-series delays and asynchronous information arrival are treated explicitly rather than hidden inside the cancellation model.
