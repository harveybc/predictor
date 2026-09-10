# STEP 05 — Source/Symbol Coding, Entropy Modeling, Predictive Residuals, Contexts, Dictionaries and MDL for Time-Series ML

**Status:** Research protocol — final draft for agent audit and repository integration  
**Date:** 2026-09-05  
**Scope:** Source coding and entropy-efficient representation after quantization, with emphasis on multivariate time series, financial forecasting, causal preprocessing, predictive residuals, context models, dictionaries/motifs, entropy-rate estimation, MDL, information bottleneck and modular multi-branch ML architectures.  
**Prerequisites:**  
- STEP 01 — Sampling / Nyquist  
- STEP 02–03 — Noise / SNR estimation and denoising  
- STEP 04 — Quantization, non-uniform quantization and companding  

**Next planned communications analogue:** STEP 06 — representation/modulation analogues: amplitude, frequency, phase, time-frequency and multi-domain signal representation.

---

# 0. Executive summary

After quantization, classical source coding asks:

> **How can the remaining statistical redundancy in a symbol sequence be represented with the smallest expected number of bits while preserving the information that must be recoverable?**

This step is highly relevant to the project's original analogy, but an important correction is required:

> **Huffman, arithmetic coding or ANS bitstreams are not automatically useful neural-network inputs.**

Entropy coders are designed to generate compact bitstreams. Their individual bit patterns are largely implementation artifacts. If the data are decoded before entering the model, forecasting information is unchanged; if the compressed bitstream itself is fed directly to a neural model, temporal and semantic geometry may be made substantially harder to learn.

Therefore the useful ML analogue is primarily found in the **source model that makes compression possible**, not in the final bit-packing mechanism.

The research focus becomes:

\[
\boxed{
\text{quantized observations}
\rightarrow
\text{causal probability/context model}
\rightarrow
\text{innovation / motifs / conditional structure}
\rightarrow
\text{surprisal / residual / token representation}
\rightarrow
\text{ML branches}
}
\]

while entropy coding is retained as:

1. a theoretical benchmark;
2. a diagnostic of remaining redundancy;
3. a way to estimate effective code length;
4. a possible deployment/storage layer;
5. a bridge to Minimum Description Length (MDL).

The strongest Step-5 hypotheses concern:

- entropy rate;
- memory/context length;
- predictive residual coding;
- variable-order contexts;
- dictionaries/motifs;
- multivariate conditional coding;
- task-relevant versus task-irrelevant compressibility;
- surprisal/novelty as a feature;
- MDL as a representation/model-selection diagnostic.

---

# 1. Correct position in the communications/source-coding chain

A clean conceptual chain is:

\[
\text{continuous source}
\rightarrow
\text{sampling}
\rightarrow
\text{quantization}
\rightarrow
\text{source symbols}
\rightarrow
\text{source model}
\rightarrow
\text{entropy/source coding}
\rightarrow
\text{channel coding}
\rightarrow
\text{modulation/transmission}.
\]

For this project:

1. sampling/Nyquist is STEP 01;
2. noise/SNR and denoising are STEPS 02–03;
3. quantization/companding is STEP 04;
4. **source modeling and entropy-efficient coding are STEP 05**;
5. channel coding/redundancy will be treated later and must not be confused with source coding.

Source coding attempts to **remove redundancy**.

Channel coding later deliberately **adds controlled redundancy** to survive transmission errors.

These are opposite operations and must remain separate.

---

# 2. Source coding theorem: first theoretical anchor

For a discrete memoryless source \(X\) with alphabet:

\[
\mathcal A=\{a_1,\ldots,a_K\}
\]

and probabilities:

\[
p_i=P(X=a_i),
\]

the Shannon entropy is:

\[
H(X)
=
-\sum_{i=1}^{K}
p_i\log_2p_i.
\]

For an optimal binary prefix code with expected length:

\[
\bar L
=
\sum_i p_i\ell_i,
\]

the classical bound is:

\[
H(X)
\leq
\bar L
<
H(X)+1.
\]

Block coding can make the overhead per source symbol arbitrarily small under the standard asymptotic assumptions.

This establishes the first key principle:

> **The number of possible quantization levels is not equal to the actual average information rate if their probabilities are unequal.**

A \(2^b\)-level quantizer has a nominal maximum:

\[
b
\]

bits per symbol, but its empirical entropy may be:

\[
H(X_Q)
\ll b.
\]

Therefore STEP 04 determines **resolution/cardinality**, whereas STEP 05 determines **how much statistical information the resulting symbol stream actually carries after redundancy is exploited**.

---

# 3. Memory changes the relevant lower bound

For a stationary stochastic process:

\[
X_1,X_2,\ldots,
\]

the relevant asymptotic quantity is not merely marginal entropy:

\[
H(X_t),
\]

but entropy rate:

\[
\boxed{
h_X
=
\lim_{n\rightarrow\infty}
\frac{1}{n}
H(X_1,\ldots,X_n)
}
\]

when the limit exists.

Equivalently for a stationary process:

\[
h_X
=
\lim_{k\rightarrow\infty}
H(X_t\mid X_{t-1},\ldots,X_{t-k}).
\]

Thus:

\[
h_X
\leq
H(X_t).
\]

The difference:

\[
H(X_t)-h_X
\]

represents information that can, in principle, be predicted from temporal dependence.

This is directly aligned with the project's objective.

---

# 4. The deepest Step-5 connection: prediction and compression are equivalent under probabilistic coding

Let a causal predictor assign:

\[
p_\theta(x_t\mid x_{<t}).
\]

The ideal code length associated with observation \(x_t\) is:

\[
\ell_t
=
-\log_2
p_\theta(x_t\mid x_{<t}).
\]

The complete sequence obtains:

\[
L_\theta(x_{1:n})
=
-\sum_{t=1}^{n}
\log_2
p_\theta(x_t\mid x_{<t}).
\]

Arithmetic coding or ANS can transform those probabilities into a lossless bitstream whose length approaches this negative log-likelihood plus small implementation overhead.

Therefore:

\[
\boxed{
\text{good causal probabilistic prediction}
\Longleftrightarrow
\text{short code length}
}
\]

under the chosen source model.

This equivalence has classical roots and has been demonstrated explicitly with modern neural probabilistic models; *Language Modeling Is Compression* (ICLR 2024) is a recent prominent example.

This is a major bridge between communications/compression and ML.

---

# 5. Critical distinction: compression of \(X\) is not automatically prediction of target \(Y\)

A model may compress:

\[
X
\]

extremely well by exploiting regularities that are irrelevant to:

\[
Y.
\]

For example, a calendar feature could contain perfectly compressible periodic structure that has little incremental information about a particular forecast target.

Therefore:

\[
\text{compressibility of }X
\not\Rightarrow
\text{predictive usefulness for }Y.
\]

This is one of the central controls of STEP 05.

The project must distinguish:

## 5.1. Source redundancy

\[
H(X)-h_X.
\]

## 5.2. Predictive information relevant to the task

Conceptually:

\[
I(X_{\leq t};Y_{>t}).
\]

## 5.3. Redundancy between input variables

For features \(X_i,X_j\):

\[
I(X_i;X_j\mid\text{past/context}).
\]

## 5.4. Unique incremental information

Conceptually:

\[
I(X_j;Y\mid X_{-j},\text{past}).
\]

STEP 05 must never equate “compressible” with “valuable to forecasting.”

---

# 6. Information Bottleneck: formal task-relevant compression

The Information Bottleneck (IB) framework directly formalizes the problem of finding a compressed representation \(T\) of \(X\) that preserves information about a relevant variable \(Y\).

A common form is:

\[
\min_{p(t|x)}
I(X;T)
-
\beta I(T;Y).
\]

This means:

- compress \(X\);
- preserve what matters for \(Y\).

This is substantially closer to the project's actual purpose than ordinary lossless source coding.

The original Tishby–Pereira–Bialek framework explicitly describes IB as finding a short code for \(X\) that preserves information about \(Y\).

A 2024 IEEE TPAMI survey reviews more than two decades of Information Bottleneck theory and deep-learning applications.

In 2025, a Conditional Information Bottleneck approach was published specifically for multivariate time-series forecasting, targeting inter-series and intra-series correlations.

Therefore:

> **Do not invent an ad-hoc “compress but preserve target information” objective without first benchmarking against IB/CIB formulations.**

---

# 7. Minimum Description Length: second major formal bridge

The Minimum Description Length principle asks us to consider not only the encoded data but also the cost of describing the model.

A simplified conceptual score is:

\[
L(M)+L(D\mid M),
\]

where:

- \(L(M)\): number of bits required to describe the model;
- \(L(D\mid M)\): number of bits required to describe the data given that model.

A model that memorizes the dataset may achieve small:

\[
L(D\mid M)
\]

while having enormous:

\[
L(M).
\]

MDL penalizes that tradeoff.

Rissanen's work explicitly formulated statistical modeling as shortest data description.

Subsequent MDL theory connects universal coding, model selection and statistical inference.

Recent representation-learning work has also derived generalization bounds in terms of representation description length.

This is directly relevant to the user's intuition that the model's stored “knowledge” should be related to the compressible structure of the data.

However:

> **Kolmogorov complexity, MDL, network parameter count, mutual information and source entropy are related concepts but are not interchangeable numerical quantities.**

The project should measure them separately.

---

# 8. Classical source-coding families to preserve in the benchmark

STEP 05 should not attempt to rediscover existing coding algorithms.

The relevant families are:

1. memoryless entropy codes;
2. context/statistical models;
3. dictionary methods;
4. predictive/residual methods;
5. transforms that expose redundancy;
6. modern neural probabilistic source models.

---

# 9. Huffman coding

Huffman coding assigns shorter prefix codewords to more frequent symbols.

For symbol probability \(p_i\), ideal information is:

\[
-\log_2p_i.
\]

A prefix code uses integer code lengths:

\[
\ell_i\in\mathbb N.
\]

Huffman minimizes expected code length among binary prefix codes for a known discrete distribution.

Its role in STEP 05 is:

- classical baseline;
- transparent relationship between empirical frequency and code length;
- useful implementation comparator.

It is **not** expected to create a better neural representation merely by converting symbols to Huffman bits.

---

# 10. Arithmetic coding

Arithmetic coding decouples:

1. the probability model;
2. the mechanism that turns probabilities into bits.

For a sequence probability:

\[
P_\theta(x_{1:n})
=
\prod_t
P_\theta(x_t\mid x_{<t}),
\]

ideal code length is:

\[
-\log_2P_\theta(x_{1:n}).
\]

Arithmetic coding can approach that length closely.

This separation is essential for our project:

> **The probability model is scientifically interesting for ML; the arithmetic coder is mainly a bitstream realization of that model.**

---

# 11. ANS / Finite State Entropy

Asymmetric Numeral Systems (ANS) provides entropy coding with compression efficiency comparable to arithmetic/range coding and computational properties attractive for production systems.

Finite State Entropy (FSE), based on ANS, is used by Zstandard.

The current Zstandard specification uses:

- Huffman coding for literals;
- FSE for other symbol classes.

ANS is particularly useful if the project eventually wants to materialize actual near-entropy compressed streams.

For predictive-feature research, however:

\[
-\log_2p(x_t\mid context)
\]

usually matters more than the particular final entropy coder.

---

# 12. Dictionary coding: Lempel–Ziv

Lempel–Ziv methods exploit repeated substrings without requiring the true source distribution to be known beforehand.

The 1977 Ziv–Lempel universal sequential compression algorithm established a powerful general principle:

> repeated structure can be discovered online and replaced with references/dictionary structure.

This suggests an ML analogue:

\[
\text{repeated temporal subsequence}
\rightarrow
\text{motif/token ID}.
\]

But this analogue must be experimentally evaluated.

A dictionary token may:

- improve sample efficiency by collapsing repeated structures;
- expose motif recurrence;
- make long contexts shorter;

or it may:

- discard amplitude differences;
- fragment rare events;
- make regime shifts difficult;
- create unstable dictionaries.

---

# 13. PPM: contexts instead of fixed-order frequencies

Prediction by Partial Matching (PPM) combines:

- adaptive context models;
- variable-order history;
- entropy coding.

It addresses a fundamental problem:

> longer contexts can be more predictive but suffer sparse counts.

This is directly analogous to selecting:

- lookback length;
- temporal receptive field;
- attention context;
- motif context length.

PPM therefore belongs in STEP 05 as a conceptual and empirical baseline for:

\[
\text{context depth}
\rightarrow
\text{conditional entropy}.
\]

---

# 14. Context Tree Weighting (CTW)

Context Tree Weighting (CTW) is a universal source-coding/prediction method for finite-alphabet sequences that efficiently combines multiple context depths.

Instead of selecting one single Markov order, CTW effectively averages/weights context-tree models.

For a symbolized time series:

\[
Z_t=Q(X_t),
\]

CTW can estimate:

\[
p(z_t\mid z_{<t})
\]

and hence:

\[
\hat h_{CTW}.
\]

A comparative entropy-estimation study found CTW repeatedly among the most accurate tested methods for finite-valued stationary time series.

This makes CTW a high-priority baseline for STEP 05.

---

# 15. Bayesian Context Trees: important current state of the art

Bayesian Context Trees (BCT) extend context-tree ideas into a Bayesian framework.

Especially relevant is recent work for real-valued time series:

- discrete contexts are formed from quantized recent observations;
- each context can select a different real-valued local model;
- Bayesian inference combines context structures.

A 2026 *International Journal of Forecasting* paper develops a Bayesian Context Trees State Space Model (BCT-X) and demonstrates it with AR and ARCH-type base models, including financial applications.

This is highly relevant because it independently combines:

\[
\boxed{
\text{quantization}
+
\text{contexts}
+
\text{conditional local models}
+
\text{online forecasting}
}
\]

which is very close to the path reached by STEPS 04–05.

Therefore:

> **Before implementing a novel variable-order context branch, agents must evaluate BCT/BCT-X as prior art and benchmark.**

---

# 16. Predictive/residual coding: likely the strongest direct analogue for time series

Instead of encoding:

\[
x_t,
\]

predict:

\[
\hat x_t
=
f(x_{<t})
\]

and encode only innovation:

\[
e_t
=
x_t-\hat x_t.
\]

If the predictor captures temporal structure:

\[
H(E)
<
H(X)
\]

or, more precisely, the residual can become easier to code.

This is the core logic of predictive coding.

For ML, the analogous representation is:

\[
\boxed{
X_t
\rightarrow
(\hat X_t,E_t)
}
\]

or simply:

\[
E_t.
\]

The residual represents what the causal source model could **not** explain.

This is potentially much more useful than feeding an entropy-coded bitstream.

---

# 17. Time-series compression confirms the predictive-residual principle

The modern time-series compression literature heavily exploits prediction/differences.

## 17.1. Gorilla

Facebook/Meta's Gorilla time-series database uses:

- delta-of-delta encoding for timestamps;
- XOR differences between consecutive floating-point values.

This exploits temporal similarity in lossless form.

## 17.2. Sprintz

Sprintz combines:

- online forecasting;
- encoding of prediction errors;
- bit packing;
- run-length coding;
- entropy coding.

The authors report that their online forecasting component can outperform simple delta coding for sensor time series.

This provides an important precedent:

\[
\text{prediction}
\rightarrow
\text{smaller residual}
\rightarrow
\text{better compression}.
\]

## 17.3. Modern surveys

A major ACM Computing Surveys review classifies time-series compression into families including:

- dictionary-based;
- functional approximation;
- autoencoder-based;
- sequential/predictive;
- combined approaches.

A 2025 survey/benchmark of neural lossless universal compression further shows that learned probabilistic models are now a mature compression research direction across heterogeneous data.

---

# 18. Key ML translation: innovations branch

For each feature or family:

\[
X_{j,t},
\]

fit a causal training-only source predictor:

\[
\hat X_{j,t}
=
f_j(X_{\leq t-1}).
\]

Construct innovation:

\[
E_{j,t}
=
X_{j,t}-\hat X_{j,t}.
\]

The project's multi-branch architecture can test:

## A. Raw

\[
B_1=X.
\]

## B. Innovation

\[
B_2=E.
\]

## C. Predicted component

\[
B_3=\hat X.
\]

## D. Raw + innovation

\[
B_4=[X,E].
\]

## E. Raw + predictable component + innovation

\[
B_5=[X,\hat X,E].
\]

The experiment determines whether separating:

\[
\text{expected structure}
+
\text{innovation}
\]

makes downstream learning easier.

---

# 19. This must not duplicate STEP 03 denoising

Denoising:

\[
X=S+N
\]

tries to separate signal from noise.

Predictive coding:

\[
X=\hat X+E
\]

separates predictable structure from innovation.

They are not the same.

The innovation:

\[
E
\]

may contain:

- noise;
- unpredictable but economically meaningful events;
- regime changes;
- new information.

Therefore:

\[
E\neq N
\]

in general.

This distinction is mandatory.

---

# 20. Surprisal as an explicit ML feature

Given a causal source model:

\[
p_\theta(z_t\mid z_{<t}),
\]

define surprisal:

\[
\boxed{
S_t
=
-\log_2
p_\theta(z_t\mid z_{<t})
}.
\]

Interpretation:

- common, expected observation:
  \[
  S_t\text{ small};
  \]
- rare/unexpected observation:
  \[
  S_t\text{ large}.
  \]

For financial time series, this can act as a causal measure of:

- novelty;
- regime mismatch;
- anomaly;
- model surprise;
- information arrival.

This provides a natural candidate branch:

\[
B_{surprise}
=
S_t.
\]

It is one of the cleanest source-coding-to-ML homologues.

---

# 21. Cross-feature conditional surprisal

For multivariate inputs:

\[
\mathbf X_t
=
(X_{1,t},\ldots,X_{m,t}),
\]

we can estimate:

\[
p(X_{j,t}
\mid
X_{j,<t},
X_{-j,\leq t}).
\]

Then:

\[
S_{j,t}^{cond}
=
-\log_2
p(
X_{j,t}
\mid
X_{j,<t},
X_{-j,\leq t}
).
\]

Compare with:

\[
S_{j,t}^{self}
=
-\log_2
p(
X_{j,t}
\mid
X_{j,<t}
).
\]

The reduction:

\[
\Delta S_{j,t}
=
S_{j,t}^{self}
-
S_{j,t}^{cond}
\]

quantifies how much other variables help explain feature \(j\) at that time under the chosen model.

This can help identify:

- redundant features;
- informative cross-series relationships;
- candidate branch groupings;
- changing dependence structure.

---

# 22. Conditional entropy for branch organization

For branch candidate \(G_k\), estimate:

\[
H(G_k\mid G_{<k},\text{past}).
\]

A branch that is highly predictable from other branches may contain little unique information.

But removal is justified only if:

\[
I(G_k;Y\mid G_{-k})
\]

is also low.

Therefore the branch-selection principle is:

\[
\boxed{
\text{low unique conditional source information}
+
\text{low incremental target information}
\Rightarrow
\text{candidate redundancy}
}
\]

not merely low entropy.

---

# 23. Dictionary/motif branch

After quantization or suitable normalization, discover repeating subsequences:

\[
M_k
=
(z_t,\ldots,z_{t+\ell-1}).
\]

Map them to tokens:

\[
M_k\rightarrow d_k.
\]

Potential representations:

- dictionary ID;
- motif duration;
- reconstruction error;
- time since last occurrence;
- motif frequency;
- next-symbol distribution conditioned on motif;
- target distribution after motif.

This creates a higher-level alphabet.

Potential gain:

\[
\text{long repeated subsequence}
\rightarrow
\text{single compact symbol}.
\]

Potential risk:

> dictionary compression optimizes recurrence, not necessarily predictive value.

Thus motifs must be evaluated against targets.

---

# 24. BWT and transform coding: useful diagnostic, dangerous direct preprocessing

The Burrows–Wheeler transform (BWT) reorganizes a block so similar contexts cluster, enabling later compression stages such as move-to-front and entropy coding.

For ML forecasting, however:

- BWT reorders symbols;
- it is naturally block-based;
- a naive implementation is not causal;
- temporal adjacency is destroyed in transformed order.

Therefore:

> BWT is useful as evidence that representation can expose hidden redundancy, but it is **not** a default online ML preprocessing transform.

It may be used:

- as a compression diagnostic;
- in offline representation studies;
- to inspire context grouping;

but not inserted blindly before a temporal neural model.

---

# 25. Modern learned entropy models

Modern learned compression typically separates:

1. learned transform/encoder;
2. quantized latent;
3. learned entropy model;
4. arithmetic/ANS bitstream.

For latent \(Z\):

\[
R
\approx
E[-\log_2p_\theta(Z)].
\]

This is now standard in learned image compression and increasingly general lossless compression.

Hierarchical priors and autoregressive priors reduce rate by modeling latent dependencies more accurately.

This matters to the project because its existing autoencoder/BiGAN-like feature extractor could eventually support:

\[
X
\rightarrow
E_\phi(X)=Z
\rightarrow
Q(Z)
\rightarrow
p_\theta(Q(Z))
\]

and hence an explicit rate term.

But that belongs to a later latent-compression extension, not the first STEP-05 experiment.

---

# 26. Bits-back coding: advanced later extension

Latent-variable models can support near-optimal lossless compression through bits-back schemes.

Bits Back with ANS (BB-ANS) demonstrated practical lossless compression using VAEs and ANS.

This is important prior art for any future proposal claiming:

> “the latent variables themselves can be entropy-coded efficiently.”

Do not implement a bespoke VAE source-coding scheme before studying BB-ANS.

This is an advanced extension, not required for the primary forecasting preprocessing test.

---

# 27. Compression can be a direct estimator of temporal complexity

For a quantized sequence:

\[
Z_{1:n},
\]

define normalized compressed length:

\[
R_C
=
\frac{L_C(Z_{1:n})}{n}
\]

bits/symbol.

Candidate compressors:

- Huffman;
- arithmetic coding with fixed distribution;
- PPM;
- CTW;
- LZ;
- Zstd;
- Brotli;
- neural probability model + arithmetic/ANS.

For a good universal coder on an appropriate stationary ergodic source:

\[
R_C
\rightarrow
h_Z
\]

asymptotically under relevant assumptions.

Thus compression provides a practical proxy for entropy rate.

---

# 28. Financial prior art: compression and predictability are already connected

The literature has directly used compression-based entropy estimators in finance.

Examples include:

- entropy and nonlinear predictability of stock-market returns;
- Lempel–Ziv-based predictability estimators for financial time series;
- universal compression applied to intraday foreign-exchange data;
- variable-order Markov models for market-efficiency testing.

This is important:

> The project should not claim novelty for “using compression to measure financial predictability.”

The potentially novel contribution is the **integration of source-coding diagnostics into a controlled preprocessing and multi-branch representation protocol**, especially following the noise and quantization stages.

---

# 29. Current 2026 context-tree forecasting result is particularly important

The 2026 BCT-X work is a strong warning against reinventing a simplistic variable-order Markov pipeline.

It already demonstrates:

\[
\text{quantized context}
\rightarrow
\text{context tree}
\rightarrow
\text{real-valued local model}
\rightarrow
\text{online forecasting}
\]

with AR/ARCH examples and financial relevance.

Project action:

1. reproduce or benchmark against BCT-X where feasible;
2. inspect whether its context construction maps naturally to project features;
3. use it as a classical/non-neural baseline before proposing a custom “dictionary context branch.”

---

# 30. Proposed STEP-05 experimental phases

The phases are intentionally staged to avoid combinatorial explosion.

---

# 31. Phase 0 — Freeze STEP-04 output alphabet

Select a small number of STEP-04 representations:

1. raw continuous;
2. best uniform quantized;
3. best non-uniform quantized;
4. best companded + quantized.

Do not carry every STEP-04 candidate forward.

The source-coding stage needs a finite alphabet for classical discrete methods.

---

# 32. Phase 1 — Zero-order entropy baseline

For each quantized feature \(Z_j\), estimate from training only:

\[
\hat p_j(k)
=
\frac{n_{j,k}}{N}.
\]

Compute:

\[
\hat H_0(Z_j)
=
-\sum_k
\hat p_j(k)
\log_2
\hat p_j(k).
\]

Compare with nominal bit depth:

\[
b_j.
\]

Define marginal redundancy:

\[
R_{marg,j}
=
b_j-\hat H_0(Z_j).
\]

This measures unequal symbol occupancy, not temporal predictability.

---

# 33. Phase 2 — Conditional entropy versus context depth

Estimate:

\[
H_k
=
H(Z_t\mid Z_{t-1},\ldots,Z_{t-k})
\]

for increasing:

\[
k.
\]

Candidate methods:

- finite-order Markov plug-in;
- PPM;
- CTW;
- Bayesian Context Trees.

Plot:

\[
\boxed{
k
\rightarrow
\hat H_k
}
\]

and determine a context plateau:

\[
k^*.
\]

This may reveal a characteristic memory scale in symbol space.

---

# 34. Hypothesis: compression memory depth predicts model lookback requirements

Compare:

\[
k^*_{entropy}
\]

with forecasting performance across model lookback:

\[
W.
\]

Hypothesis:

\[
W^*
\]

should be related to the depth at which:

\[
H_k
\]

stops decreasing materially.

This is not mathematically guaranteed because:

- predictive information about \(X_t\) is not identical to information about future target \(Y\);
- continuous amplitudes were quantized;
- neural models may exploit distributed multivariate context.

It remains a falsifiable empirical hypothesis.

---

# 35. Phase 3 — Classical compressor benchmark

For each quantized stream evaluate:

- Huffman;
- LZ-based compressor;
- PPM if implementation is stable;
- CTW;
- Zstd;
- Brotli.

Report:

\[
\text{bits/symbol}.
\]

Purpose:

1. estimate remaining source redundancy;
2. test sensitivity to different pattern models;
3. establish a practical compression baseline.

Do not feed these compressed bitstreams to the model yet.

---

# 36. Phase 4 — Predictive residual representation

Fit causal predictors using training only.

Candidate hierarchy:

## P0

Persistence:

\[
\hat x_t=x_{t-1}.
\]

## P1

Linear AR.

## P2

Robust/regularized AR.

## P3

Small causal neural predictor.

## P4

BCT/BCT-X or variable-context predictor where appropriate.

Compute:

\[
E_t=x_t-\hat x_t.
\]

Compare:

- raw;
- residual;
- raw + residual;
- predicted component + residual.

---

# 37. Phase 5 — Compression quality of residuals

For each predictor:

\[
f,
\]

measure:

\[
H(E_f)
\]

or quantized residual code rate:

\[
R(E_f).
\]

A source predictor is compression-effective when:

\[
R(E_f)
<
R(X).
\]

Then separately test downstream forecast performance.

This gives a two-dimensional evaluation:

\[
\boxed{
\text{source redundancy reduction}
\quad\text{vs}\quad
\text{target forecasting improvement}
}
\]

which is essential.

---

# 38. Phase 6 — Surprisal branch

Using training-only source models, compute causal:

\[
S_t
=
-\log_2
p(z_t\mid context_t).
\]

Compare:

- no surprisal;
- surprisal only;
- raw + surprisal;
- residual + surprisal;
- raw + residual + surprisal.

Possible source models:

- zero-order;
- fixed-order Markov;
- PPM;
- CTW/BCT;
- neural autoregressive model.

---

# 39. Phase 7 — Surprisal and market/event regimes

Test whether large:

\[
S_t
\]

corresponds to:

- volatility spikes;
- regime transitions;
- macroeconomic release windows;
- large return innovations;
- forecasting error spikes.

This is descriptive analysis on training/validation only.

No manual selection of event categories using test.

---

# 40. Phase 8 — Dictionary/motif representation

Candidate methods:

- exact LZ-like phrase dictionary;
- approximate motif dictionary;
- SAX words;
- learned sequence codebook.

For each motif token \(d_t\), derive:

- frequency;
- duration;
- reconstruction error;
- recurrence interval;
- conditional next-token entropy.

Compare:

\[
X
\]

vs:

\[
D(X)
\]

vs:

\[
[X,D(X)].
\]

---

# 41. Phase 9 — Multivariate conditional coding

For each feature \(j\), compare source-model code lengths:

## Self history

\[
L_j^{self}
=
-\sum_t
\log_2
p(
x_{j,t}
\mid
x_{j,<t}
).
\]

## Full causal multivariate context

\[
L_j^{multi}
=
-\sum_t
\log_2
p(
x_{j,t}
\mid
x_{j,<t},
x_{-j,\leq t}
).
\]

Define conditional coding gain:

\[
G_j
=
L_j^{self}
-
L_j^{multi}.
\]

Large:

\[
G_j
\]

means other features explain part of feature \(j\)'s uncertainty.

This can guide:

- feature grouping;
- branch design;
- redundancy analysis.

---

# 42. Phase 10 — Target-relevance control

For every source-compression feature \(R(X)\), also estimate incremental forecasting value.

A representation is not accepted merely because:

\[
L(R(X))
<
L(X).
\]

Require:

\[
P([X,R(X)])
>
P(X)
\]

or equivalent evidence of useful target preservation.

This phase directly tests the gap between source coding and task-relevant coding.

---

# 43. Phase 11 — Information Bottleneck baseline

For selected feature groups, evaluate an IB/CIB-inspired representation:

\[
T=f_\phi(X)
\]

with objective conceptually of the form:

\[
I(X;T)-\beta I(T;Y).
\]

Practical implementations may use variational bounds.

Compare with:

- unsupervised source coding;
- raw features;
- autoencoder latent;
- quantized latent.

Because the project predicts multiple horizons, \(Y\) may be:

\[
Y=
(Y_{h_1},\ldots,Y_{h_K}).
\]

This is more faithful to the actual forecasting objective than compression of \(X\) alone.

---

# 44. Phase 12 — MDL / prequential diagnostic

For a model \(M\), estimate a prequential or otherwise well-defined description length:

\[
L(D,M).
\]

Compare across:

- model size;
- preprocessing representation;
- validation performance;
- test performance.

Question:

> Does lower description length correlate with better out-of-sample generalization in this project?

This is an empirical question.

Do not assert MDL equivalence with Kolmogorov complexity or with network storage capacity.

---

# 45. Phase 13 — Learned entropy model of latent representations

Deferred until input-level experiments succeed.

For encoder:

\[
Z=E_\phi(X),
\]

quantize:

\[
\hat Z=Q(Z)
\]

and learn:

\[
p_\theta(\hat Z).
\]

Rate:

\[
R_Z
=
E[-\log_2p_\theta(\hat Z)].
\]

Possible objective:

\[
\mathcal L
=
\mathcal L_{forecast}
+
\lambda R_Z.
\]

This explicitly penalizes latent description length.

Compare against:

- autoencoder reconstruction-only;
- existing BiGAN/AE latent;
- VQ latent;
- IB latent.

---

# 46. Causal training/validation/test governance

Project split:

\[
4\text{ years train}
+
1\text{ year validation}
+
1\text{ year test}.
\]

The following must be fitted on training only:

- symbol distributions;
- Huffman trees;
- dictionaries;
- context trees;
- PPM counts;
- CTW/BCT hyperparameters where learned;
- motif dictionaries;
- source predictor parameters;
- quantizers;
- entropy models;
- normalization;
- residual predictors.

Validation may select among configurations.

Test is evaluated only after final selection.

---

# 47. Adaptive source models are allowed if causal

A static train-only source model may become stale under nonstationarity.

An online adaptive model is valid if:

\[
\theta_t
=
g(
x_1,\ldots,x_t
)
\]

and never uses future data.

Compare:

## Static

\[
p_{\theta_{train}}.
\]

## Causal adaptive

\[
p_{\theta_t}.
\]

Hypothesis:

> adaptive contexts should reduce code length under distribution shift.

But lower code length does not automatically imply better target forecasting.

---

# 48. Regime shifts can be measured through code length

For a causal source model, define rolling average surprisal:

\[
\bar S_t^{(W)}
=
\frac{1}{W}
\sum_{i=t-W+1}^{t}
S_i.
\]

A distribution shift should often produce:

\[
\bar S_t
\uparrow.
\]

This can become an OOD/regime feature.

Potential integration:

\[
B_{OOD}
=
[
\bar S_t,
\Delta\bar S_t
].
\]

This is especially relevant to the project's existing interest in OOD scores and regime modeling.

---

# 49. Multi-branch architecture mapping

The project architecture naturally supports STEP 05.

For feature family \(k\):

\[
X_k
\]

create:

\[
B_{k,raw}=X_k
\]

\[
B_{k,res}=X_k-\hat X_k
\]

\[
B_{k,surp}
=
-\log_2p(X_k\mid context)
\]

\[
B_{k,motif}=D_k(X_k)
\]

\[
B_{k,context}=C_k(X_k).
\]

Each specialized encoder:

\[
Z_{k,r}
=
E_{k,r}(B_{k,r}).
\]

Core:

\[
Z_{core}
=
C(
Z_{1,raw},
Z_{1,res},
Z_{1,surp},
\ldots
).
\]

Heads:

\[
\hat Y_h
=
H_h(Z_{core}).
\]

---

# 50. Do not create uncontrolled branch explosion

Every new representation increases:

- model capacity;
- multiple comparisons;
- overfitting risk;
- compute;
- attribution ambiguity.

Therefore use staged branch gates.

A branch is promoted only if it has evidence of incremental value.

---

# 51. Final falsifiable hypotheses

## H5.1 — Temporal context reduces source entropy

For at least some series:

\[
H(Z_t\mid Z_{t-1:t-k})
<
H(Z_t).
\]

**Falsified if:** conditional models provide no reproducible code-length reduction beyond finite-sample noise.

---

## H5.2 — Entropy-rate plateau identifies finite useful source memory

There exists:

\[
k^*
\]

such that increasing context beyond \(k^*\) yields negligible reduction in estimated entropy rate.

**Falsified if:** no stable plateau exists or estimates are too unstable to identify one.

---

## H5.3 — Source-memory depth relates to effective ML lookback

\[
k^*_{source}
\]

has a reproducible association with:

\[
W^*_{forecast}.
\]

**Falsified if:** no relationship survives datasets/models.

---

## H5.4 — Predictive residuals reduce source redundancy

For a causal predictor:

\[
E_t=X_t-\hat X_t,
\]

there exists an appropriate quantized representation such that:

\[
R(E)
<
R(X).
\]

**Falsified if:** residual coding consistently fails to reduce source rate.

---

## H5.5 — Innovation representation can improve forecasting

\[
P([X,E])
>
P(X)
\]

for at least some feature families.

**Falsified if:** innovation branches provide no reproducible incremental predictive value.

---

## H5.6 — Surprisal is a useful novelty representation

\[
P([X,S])
>
P(X)
\]

where:

\[
S_t=-\log_2p(x_t\mid context).
\]

**Falsified if:** surprisal never improves forecasting/OOD/regime discrimination beyond controls.

---

## H5.7 — Better compression does not necessarily imply better forecasting

Across candidate representations:

\[
\mathrm{corr}
(
\Delta R,
\Delta P
)
\]

is not assumed to be monotonic.

This is intentionally a **null-control hypothesis**.

If compression gain and forecasting gain are perfectly aligned across domains, that would be surprising and informative.

---

## H5.8 — Conditional multivariate coding identifies redundancy

For some feature \(j\):

\[
L_j^{multi}
<
L_j^{self}.
\]

The gain should correspond to measurable cross-feature dependence.

**Falsified if:** multivariate conditional models provide no stable gain.

---

## H5.9 — Conditional coding gain predicts branch grouping value

Features with strong mutual conditional coding gain should benefit from joint or interacting branches.

**Falsified if:** coding relationships do not correspond to any downstream architectural benefit.

---

## H5.10 — Dictionary/motif tokenization can improve sample efficiency

A motif representation may improve forecasting at fixed model capacity:

\[
P([X,D(X)])
>
P(X).
\]

**Falsified if:** motif tokens provide no incremental value or cause instability under regime shift.

---

## H5.11 — Adaptive source models are more robust under nonstationarity

On validation segments with distribution shift:

\[
L_{adaptive}
<
L_{static}.
\]

**Falsified if:** adaptive models do not reduce code length or overreact to noise.

---

## H5.12 — Lower code length alone is insufficient for target relevance

Representations can exist with:

\[
R_1<R_2
\]

but:

\[
P_1\leq P_2.
\]

This hypothesis is expected to hold and protects against the mistaken conclusion that compression ratio is the forecasting objective.

---

## H5.13 — IB/CIB representation gives a better rate–relevance frontier than source-only compression

At comparable representation complexity:

\[
I(T;Y)
\]

or forecasting performance should be higher for task-aware compression.

**Falsified if:** unsupervised source compression matches or exceeds task-aware representation consistently.

---

## H5.14 — MDL is associated with generalization

Lower training-set description length including model cost should show some reproducible association with validation/test performance.

**Falsified if:** MDL ranking is unrelated or inversely related under robust evaluation.

---

## H5.15 — Huffman/arithmetic/ANS bit identity is not the useful ML representation

At equal decoded symbol information, changing only the final entropy coder should not materially change forecasting if decoding occurs before ML.

**Falsified only if:** an implementation introduces a meaningful architectural interaction; such a result must be treated as architecture-specific rather than information-theoretic.

This is a critical negative control.

---

# 52. Primary experimental metrics

## 52.1. Marginal entropy

\[
H_0.
\]

## 52.2. Conditional entropy

\[
H_k.
\]

## 52.3. Entropy-rate estimate

\[
\hat h.
\]

## 52.4. Compression rate

\[
R_C
=
\frac{\text{compressed bits}}{\text{symbols}}.
\]

## 52.5. Compression redundancy

\[
\rho_C
=
R_C-\hat h.
\]

## 52.6. Surprisal

\[
S_t.
\]

## 52.7. Residual entropy/rate

\[
R(E).
\]

## 52.8. Forecast metrics

- MAE;
- RMSE/MSE;
- \(R^2\);
- per-horizon metrics;
- calibration if probabilistic;
- NLL where applicable.

## 52.9. Computational metrics

- CPU/GPU time;
- memory;
- latency;
- model parameters;
- branch complexity.

---

# 53. Entropy estimator benchmark

No single finite-sample entropy-rate estimator should be trusted blindly.

Compare where feasible:

1. plug-in finite-order;
2. Lempel–Ziv estimator;
3. CTW;
4. Bayesian Context Tree estimator;
5. cross-entropy of a held-out causal neural source model.

The 2008 comparative study found CTW particularly accurate in its tested binary settings, but project data differ substantially.

Agents should report estimator disagreement.

---

# 54. Synthetic validation before financial interpretation

Use known processes with analytically or numerically known entropy rates:

- iid Bernoulli;
- finite-order Markov;
- hidden Markov;
- AR process after controlled quantization;
- regime-switching synthetic process.

The source-coding pipeline must first recover expected behavior.

Only then interpret financial entropy rates.

---

# 55. Public time-series benchmark

Use at least one public dataset to validate:

- quantization;
- context models;
- residual coding;
- surprisal branch.

Recommended datasets may overlap STEP 04:

- ETT;
- Electricity;
- Traffic;
- Weather;
- Exchange Rate.

Do not claim generality from project financial data alone.

---

# 56. Financial benchmark

Then evaluate:

- EURUSD;
- ETHUSDT;
- other available assets;
- 1 h;
- 4 h;
- technical inputs;
- fundamental inputs;
- cross-asset variables.

Particularly important comparisons:

\[
\text{price level}
\]

vs:

\[
\text{return}
\]

vs:

\[
\text{predictive residual}
\]

vs:

\[
\text{source surprisal}.
\]

---

# 57. Feature-family redundancy graph

Construct graph nodes:

\[
V
=
\{
G_1,\ldots,G_K
\}
\]

for feature families.

Define directed edge weight:

\[
w_{i\rightarrow j}
=
L_j^{self}
-
L_j^{cond(i)}.
\]

Interpretation:

> how much conditioning on family \(i\) reduces code length of family \(j\).

This graph can suggest branch interactions.

It does not prove causality.

Do not label edges causal without a separate causal protocol.

---

# 58. Compression-based OOD monitoring

Fit source model on training.

Validation/test pointwise surprisal:

\[
S_t
=
-\log_2p_{\theta_{train}}(x_t\mid context).
\]

Define standardized:

\[
Z_{S,t}
=
\frac{S_t-\mu_{S,train}}
{\sigma_{S,train}}.
\]

Potential OOD score:

\[
OOD_t=Z_{S,t}.
\]

Compare against existing project OOD methods:

- Mahalanobis;
- KNN latent distance;
- regime entropy.

This is a natural extension of existing project governance.

---

# 59. Important warning: no causal claims from compressibility alone

If:

\[
X_i
\]

helps compress:

\[
X_j,
\]

it only establishes predictive/statistical dependence under the chosen model.

It does not establish:

\[
X_i\rightarrow X_j
\]

causally.

Any causal interpretation requires separate assumptions/tests.

---

# 60. Statistical validation

Forecasting comparisons should use temporally appropriate procedures.

Recommended:

- Diebold–Mariano where applicable;
- block bootstrap / stationary bootstrap;
- seed distributions;
- multiple-comparison control;
- predeclared primary comparisons.

Compression-rate comparisons can often be measured deterministically for a fixed stream, but uncertainty across temporal blocks/regimes should still be characterized.

---

# 61. Baseline hierarchy

The minimal benchmark should contain:

## Classical source models

- zero-order frequency;
- finite-order Markov;
- LZ;
- CTW/BCT.

## Practical compressors

- Zstd;
- Brotli.

## Predictive models

- persistence;
- AR;
- small neural autoregressive model.

## Forecast models

- DLinear;
- PatchTST or equivalent;
- project modular predictor.

## Task-aware compression

- IB/CIB-style baseline.

---

# 62. Reuse matrix — do not reinvent these components

| Problem | Existing prior art | Project action |
|---|---|---|
| Prefix entropy coding | Huffman | Reuse library/reference only |
| Near-entropy bit coding | Arithmetic coding | Reuse implementation |
| Fast near-entropy coding | ANS/FSE | Reuse proven implementation |
| Repeated substring compression | LZ family | Benchmark, do not rederive |
| Variable-order contexts | PPM | Benchmark/reference |
| Universal context mixture | CTW | High-priority baseline |
| Bayesian variable contexts | BCT/BCT-X | High-priority modern baseline |
| Time-series predictive compression | Sprintz | Reuse concepts/benchmark |
| Time-series float compression | Gorilla/Chimp-family literature | Storage diagnostic, not ML default |
| Source-compression taxonomy | ACM time-series compression survey | Use taxonomy |
| Neural universal compression | NNLCB survey/benchmark | Use as neural compression map |
| Prediction ↔ compression | ICLR 2024 Language Modeling Is Compression | Use theoretical/empirical bridge |
| Task-relevant compression | Information Bottleneck / CIB-MTSF | Benchmark |
| Model + data complexity | MDL | Use as selection diagnostic |
| Latent lossless compression | BB-ANS | Later latent extension |

---

# 63. What not to reinvent

Do **not** spend project effort implementing bespoke versions of:

- Huffman;
- arithmetic coding;
- ANS/FSE;
- Zstd;
- Brotli;
- generic LZ compressors.

Use standard libraries when actual bitstreams are required.

Research effort should focus on:

- source modeling;
- representation;
- causal residuals;
- conditional information;
- feature grouping;
- task relevance.

---

# 64. What may actually be novel/useful in this project

The state-of-the-art review suggests that the following combination is a meaningful research target:

\[
\boxed{
\text{noise-aware quantized time series}
\rightarrow
\text{causal source/context model}
\rightarrow
\text{innovation + surprisal + motif representations}
\rightarrow
\text{conditional redundancy graph}
\rightarrow
\text{multi-branch ML core}
\rightarrow
\text{predictive heads}
}
\]

with:

\[
\boxed{
\text{compression efficiency}
\neq
\text{target relevance}
}
\]

explicitly tested through IB/forecasting ablations.

This integration is more defensible than “feed compressed bits into a neural network.”

---

# 65. Minimum experimental matrix

| ID | Representation | Source model | Extra feature | Forecast model |
|---|---|---|---|---|
| S00 | Raw | None | None | DLinear |
| S01 | Quantized | Zero-order | None | DLinear |
| S02 | Quantized | CTW/BCT | Surprisal | DLinear |
| S03 | Raw | AR | Residual | DLinear |
| S04 | Raw + residual | AR | Residual | DLinear |
| S05 | Raw + surprisal | CTW/BCT | Surprisal | DLinear |
| S06 | Raw + residual + surprisal | Best causal model | Both | DLinear |
| S07 | Raw + motif | LZ/SAX-like | Motif token | DLinear |
| S08 | Best S0x | Best source model | Best feature | PatchTST |
| S09 | Best S0x | Best source model | Best feature | Project model |
| S10 | IB/CIB representation | Task-aware | Latent | Project model |

---

# 66. Recommended implementation order

1. Freeze one STEP-04 alphabet.
2. Compute zero-order entropy.
3. Compute conditional entropy vs context.
4. Run CTW/BCT/LZ compression diagnostics.
5. Validate on synthetic known processes.
6. Build persistence/AR residuals.
7. Test raw vs residual vs raw+residual.
8. Add surprisal.
9. Compare static vs causal adaptive source models.
10. Test multivariate conditional code gain.
11. Build feature-family redundancy graph.
12. Test motifs/dictionaries.
13. Compare on public forecasting benchmark.
14. Test DLinear and PatchTST.
15. Integrate best representations in project multi-branch model.
16. Add IB/CIB baseline.
17. Add MDL/prequential analysis.
18. Only then consider learned latent entropy models / BB-ANS.

---

# 67. Audit checklist

- [ ] Source coding and channel coding are explicitly separated.
- [ ] Entropy \(H(X)\) is not confused with entropy rate \(h_X\).
- [ ] Quantizer bit depth is not confused with source entropy.
- [ ] Actual entropy coder bits are not assumed to be semantically useful ML inputs.
- [ ] Probability/source model is separated from entropy coder implementation.
- [ ] Compression gain is not equated with target forecasting gain.
- [ ] Innovation is not equated with noise.
- [ ] All context/source models are training-only or strictly causal adaptive.
- [ ] Test data are not used for dictionary/context selection.
- [ ] Entropy-rate estimators are validated on synthetic known processes.
- [ ] CTW/BCT are benchmarked before inventing custom context trees.
- [ ] BCT-X 2026 is reviewed before custom financial variable-order Markov development.
- [ ] LZ/dictionary methods are treated as prior art.
- [ ] Surprisal is computed causally.
- [ ] Cross-feature coding gain is not labeled causal.
- [ ] Motif recurrence is distinguished from target relevance.
- [ ] IB/CIB is included as task-relevant compression control.
- [ ] MDL is not equated with Kolmogorov complexity.
- [ ] Compression rate, forecasting performance and model complexity are all reported.
- [ ] Multiple-comparison correction is predefined.
- [ ] Raw branches remain available in initial multi-branch ablations.
- [ ] Latent entropy coding is deferred until input-level hypotheses are tested.
- [ ] BB-ANS is reviewed before custom latent lossless coding.
- [ ] BWT/block transforms are not inserted into online temporal models without causal analysis.
- [ ] Source-model surprisal is compared with existing OOD metrics.
- [ ] Adaptive source models never consume future information.

---

# 68. Deliverables

Recommended artifacts:

1. `step05_source_model_spec.json`
2. `symbol_entropy_train.parquet`
3. `conditional_entropy_by_context.parquet`
4. `entropy_rate_estimators.parquet`
5. `compressor_benchmark.parquet`
6. `source_model_surprisal.parquet`
7. `predictive_residuals.parquet`
8. `residual_entropy_metrics.parquet`
9. `multivariate_conditional_code_gain.parquet`
10. `feature_redundancy_graph.graphml`
11. `motif_dictionary.parquet`
12. `forecast_ablation_metrics.parquet`
13. `mdl_prequential_metrics.parquet`
14. `statistical_tests.json`
15. `step05_audit_report.md`
16. reproducibility manifest with dataset hashes, split dates, seeds, commits and environment.

---

# 69. Decision gates

## Gate 5A — Entropy estimator validity

Synthetic sources produce expected entropy-rate behavior.

## Gate 5B — Temporal redundancy exists

Conditional/source models reduce code rate versus zero-order baseline.

## Gate 5C — Predictive residual compression exists

Residuals are measurably more compressible for at least some features.

## Gate 5D — Representation value

Residual/surprisal/motif representation provides incremental validation forecasting value.

## Gate 5E — Multivariate conditional value

Cross-feature conditioning provides stable code gain and useful branch guidance.

## Gate 5F — Task relevance

Compression-oriented representations survive comparison against task-aware IB/CIB baselines.

## Gate 5G — Architecture transfer

Effect survives more than one forecasting architecture.

## Gate 5H — Final held-out confirmation

Only preselected configurations reach test.

---

# 70. Possible scientifically valid outcomes

## Outcome A

Temporal source compression reveals useful memory/context structure.

## Outcome B

Residual and surprisal branches improve forecasting.

## Outcome C

Compression diagnostics help branch grouping but not direct prediction.

## Outcome D

Motif/dictionary representations help only selected feature families.

## Outcome E

IB/CIB strongly outperforms source-only compression.

## Outcome F

Compression ratio has little relationship to target forecasting.

## Outcome G

Raw inputs remain superior.

All outcomes are scientifically informative.

---

# 71. State-of-the-art conclusions

The literature review changes the proposed STEP-05 plan in several important ways.

### 71.1. We should not reinvent entropy coders

Huffman, arithmetic coding, ANS/FSE, LZ and modern general-purpose compressors are mature.

### 71.2. We should not claim novelty for compression-based financial predictability

There is prior work using:

- entropy rate;
- Lempel–Ziv;
- universal compression;
- variable-order Markov models

to study stock and FX predictability.

### 71.3. Context models are more advanced than a hand-built fixed Markov chain

CTW and BCT/BCT-X provide principled variable-context alternatives.

### 71.4. Prediction and compression are formally linked

Modern probabilistic ML makes this connection operational:

\[
-\log p(x_t\mid context)
\]

is simultaneously:

- predictive log-loss;
- ideal code length;
- surprisal.

### 71.5. Task-relevant compression already has a formal theory

Information Bottleneck/CIB should be a baseline, not reinvented ad hoc.

### 71.6. The project's strongest opportunity is integration

The potentially valuable investigation is:

\[
\boxed{
\text{source redundancy diagnostics}
+
\text{innovation}
+
\text{surprisal}
+
\text{conditional cross-feature information}
+
\text{multi-branch architecture}
}
\]

under strict causal validation.

---

# 72. References — IEEE style

[1] C. E. Shannon, “A Mathematical Theory of Communication,” *Bell System Technical Journal*, vol. 27, pp. 379–423 and 623–656, 1948. Available: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf

[2] D. A. Huffman, “A Method for the Construction of Minimum-Redundancy Codes,” *Proceedings of the IRE*, vol. 40, no. 9, pp. 1098–1101, Sep. 1952, doi: 10.1109/JRPROC.1952.273898. Available: https://doi.org/10.1109/JRPROC.1952.273898

[3] J. Ziv and A. Lempel, “A Universal Algorithm for Sequential Data Compression,” *IEEE Transactions on Information Theory*, vol. 23, no. 3, pp. 337–343, May 1977, doi: 10.1109/TIT.1977.1055714. Available: https://www.itsoc.org/publications/papers/a-universal-algorithm-for-sequential-data-compression

[4] J. G. Cleary and I. H. Witten, “Data Compression Using Adaptive Coding and Partial String Matching,” *IEEE Transactions on Communications*, vol. 32, no. 4, pp. 396–402, Apr. 1984, doi: 10.1109/TCOM.1984.1096090. Available: https://doi.org/10.1109/TCOM.1984.1096090

[5] I. H. Witten, R. M. Neal, and J. G. Cleary, “Arithmetic Coding for Data Compression,” *Communications of the ACM*, vol. 30, no. 6, pp. 520–540, Jun. 1987, doi: 10.1145/214762.214771. Available: https://doi.org/10.1145/214762.214771

[6] F. M. J. Willems, Y. M. Shtarkov, and T. J. Tjalkens, “The Context-Tree Weighting Method: Basic Properties,” *IEEE Transactions on Information Theory*, vol. 41, no. 3, pp. 653–664, May 1995, doi: 10.1109/18.382012. Available: https://doi.org/10.1109/18.382012

[7] P. M. Fenwick, “The Burrows–Wheeler Transform for Block Sorting Text Compression: Principles and Improvements,” *The Computer Journal*, vol. 39, no. 9, pp. 731–740, 1996, doi: 10.1093/comjnl/39.9.731. Available: https://doi.org/10.1093/comjnl/39.9.731

[8] J. Rissanen, “Modeling by Shortest Data Description,” *Automatica*, vol. 14, no. 5, pp. 465–471, 1978, doi: 10.1016/0005-1098(78)90005-5. Available: https://doi.org/10.1016/0005-1098(78)90005-5

[9] A. Barron, J. Rissanen, and B. Yu, “The Minimum Description Length Principle in Coding and Modeling,” *IEEE Transactions on Information Theory*, vol. 44, no. 6, pp. 2743–2760, Oct. 1998, doi: 10.1109/18.720554. Available: https://doi.org/10.1109/18.720554

[10] N. Tishby, F. C. Pereira, and W. Bialek, “The Information Bottleneck Method,” in *Proc. 37th Annual Allerton Conference on Communication, Control, and Computing*, 1999, pp. 368–377. Available: https://arxiv.org/abs/physics/0004057

[11] Y. Gao, I. Kontoyiannis, and E. Bienenstock, “Estimating the Entropy of Binary Time Series: Methodology, Some Theory and a Simulation Study,” *Entropy*, vol. 10, no. 2, pp. 71–99, 2008. Available: https://doi.org/10.3390/entropy-e10020071

[12] B. Ryabko, “Applications of Universal Source Coding to Statistical Analysis of Time Series,” arXiv:0809.1226, 2008. Available: https://arxiv.org/abs/0809.1226

[13] A. Shmilovici, Y. Kahiri, I. Ben-Gal, and S. Hauser, “Measuring the Efficiency of the Intraday Forex Market with a Universal Data Compression Algorithm,” *Computational Economics*, 2009. Research record available: https://cris.tau.ac.il/en/publications/measuring-the-efficiency-of-the-intraday-forex-market-with-a-univ/

[14] A. Maasoumi and J. Racine, “Entropy and Predictability of Stock Market Returns,” *Journal of Econometrics*, vol. 107, no. 1–2, pp. 291–312, 2002, doi: 10.1016/S0304-4076(01)00125-7. Available: https://doi.org/10.1016/S0304-4076(01)00125-7

[15] S. Li and A. Lin, “Exploring the Relationship among Predictability, Prediction Accuracy and Data Frequency of Financial Time Series,” *Entropy*, vol. 22, no. 12, art. 1381, 2020, doi: 10.3390/e22121381. Available: https://doi.org/10.3390/e22121381

[16] T. Pelkonen *et al.*, “Gorilla: A Fast, Scalable, In-Memory Time Series Database,” *Proceedings of the VLDB Endowment*, vol. 8, no. 12, pp. 1816–1827, 2015, doi: 10.14778/2824032.2824078. Supporting engineering description: https://engineering.fb.com/2017/02/03/core-infra/beringei-a-high-performance-time-series-storage-engine/

[17] D. Blalock, S. Madden, and J. Guttag, “Sprintz: Time Series Compression for the Internet of Things,” *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 2018. Preprint and implementation: https://arxiv.org/abs/1808.02515

[18] F. P. García *et al.*, “Time Series Compression Survey,” *ACM Computing Surveys*, 2023, doi: 10.1145/3560814. Available: https://doi.org/10.1145/3560814

[19] H. Sun *et al.*, “A Survey and Benchmark Evaluation for Neural-Network-Based Lossless Universal Compressors toward Multi-Source Data,” *Frontiers of Computer Science*, vol. 19, art. 197360, 2025, doi: 10.1007/s11704-024-40300-5. Available: https://doi.org/10.1007/s11704-024-40300-5

[20] J. Duda, “Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding,” arXiv:1311.2540, 2013/2014. Available: https://arxiv.org/abs/1311.2540

[21] IETF, “Zstandard Compression and the ‘application/zstd’ Media Type,” RFC 8878, 2021. Available: https://www.rfc-editor.org/rfc/rfc8878.html

[22] G. Delétang *et al.*, “Language Modeling Is Compression,” in *International Conference on Learning Representations (ICLR)*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html

[23] L. Blier and Y. Ollivier, “The Description Length of Deep Learning Models,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018. Available: https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html

[24] M. Sefidgaran, A. Zaidi, and P. Krasnowski, “Minimum Description Length and Generalization Guarantees for Representation Learning,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023. Available: https://papers.neurips.cc/paper_files/paper/2023/hash/054e9f9a286671ababa3213d6e59c1c2-Abstract-Conference.html

[25] S. Hu, Z. Lou, X. Yan, and Y. Ye, “A Survey on Information Bottleneck,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 8, pp. 5325–5344, Aug. 2024, doi: 10.1109/TPAMI.2024.3366349. Available: https://doi.org/10.1109/TPAMI.2024.3366349

[26] X. Li, L. Duan, L. Yu, K. Yue, and Y. Li, “Conditional Information Bottleneck-Based Multivariate Time Series Forecasting,” in *Proceedings of IJCAI 2025*, pp. 5634–5642, 2025, doi: 10.24963/ijcai.2025/627. Available: https://www.ijcai.org/proceedings/2025/627

[27] I. Papageorgiou and I. Kontoyiannis, “Context-Tree Weighting for Real-Valued Time Series: Bayesian Inference with Hierarchical Mixture Models,” arXiv:2106.03023. Available: https://arxiv.org/abs/2106.03023

[28] I. Papageorgiou and I. Kontoyiannis, “The Bayesian Context Trees State Space Model for Time Series Modelling and Forecasting,” *International Journal of Forecasting*, vol. 42, no. 2, pp. 474–491, 2026, doi: 10.1016/j.ijforecast.2025.07.009. Available: https://doi.org/10.1016/j.ijforecast.2025.07.009

[29] D. Minnen, J. Ballé, and G. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018. Available: https://research.google/pubs/joint-autoregressive-and-hierarchical-priors-for-learned-image-compression/

[30] J. Townsend, T. Bird, and D. Barber, “Practical Lossless Compression with Latent Variables using Bits Back Coding,” arXiv:1901.04866, 2019. Available: https://arxiv.org/abs/1901.04866

[31] J. Alakuijala and Z. Szabadka, “Brotli Compressed Data Format,” RFC 7932, IETF, 2016. Available: https://www.rfc-editor.org/rfc/rfc7932

---

# 73. Final status

**STEP 05 is theoretically specified, grounded in classical and current source-coding literature, and structured for falsifiable experimentation.**

The most important correction to the initial intuition is:

\[
\boxed{
\text{the useful ML analogue is usually the source model and the exposed structure, not the final compressed bitstream}
}
\]

The strongest candidate representations for the project are:

\[
\boxed{
\text{innovation}
+
\text{surprisal}
+
\text{context state}
+
\text{motif/dictionary token}
+
\text{conditional cross-feature information}
}
\]

with task relevance explicitly controlled through forecasting ablations and Information Bottleneck baselines.

If agent audit approves this protocol, proceed to:

**STEP 06 — amplitude, frequency, phase and time-frequency representations as the analogue of exploiting additional signal dimensions in modulation/communications.**
