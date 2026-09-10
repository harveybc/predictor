# STEP 07 — Matched Filtering, Template/Pattern Detection and Representation-Aware Model Banks

**Status:** Research protocol — final draft for agent audit and repository integration  
**Date:** 2026-09-05  
**Scope:** Receiver-side pattern detection after the representation stages defined in STEPS 01–06, with particular attention to matched filtering, generalized matched filters, template banks, shapelets, convolutional detector banks, motif discovery, time-frequency pattern detectors, graph/multivariate detectors, self-supervised/foundation encoders, and integration into modular multi-branch forecasting/RL architectures.  
**Primary application:** multivariate financial time series at 1 h and 4 h periodicities, with 4-year training + 1-year validation + 1-year test splits.  
**Project repositories reviewed:** `harveybc/predictor`, especially `examples/config/phase_1b_binary` and current predictor plugins; `harveybc/agent-multi`, especially model/training work-plan and current event-token/SAC infrastructure.  
**Prerequisites:** STEPS 01–06.  
**Next communications analogue after this step:** pulse shaping / receiver-side optimal detection and then controlled redundancy/error-correction analogues, depending on the final consolidated roadmap.

---

# 0. Executive conclusion

STEP 07 is the first stage in which the communications analogy becomes a **decision/detection problem** rather than only a representation problem.

The classical receiver question is:

> **Given a received signal corrupted by noise, is a known or partially known pattern present?**

For a known template \(s\) in additive Gaussian noise, matched filtering provides the optimal linear detector in the classical sense of maximizing output SNR. In colored Gaussian noise, the corresponding generalized matched-filter statistic weights the template by the inverse noise covariance.

For this project the direct translation is **not**:

> “Use another Conv1D.”

The rigorous translation is:

\[
\boxed{
\text{STEP-06 representation}
\rightarrow
\text{detector matched to its geometry/statistics}
\rightarrow
\text{pattern evidence}
\rightarrow
\text{existing core}
\rightarrow
\text{existing predictive heads}
}
\]

The central design principle is:

\[
\boxed{
\text{different representations require different detector families}
}
\]

A 1D raw/innovation sequence, a complex Fourier spectrum, a wavelet scalogram, a Hilbert phase trajectory, and a cross-spectral matrix are not geometrically equivalent objects. It would be scientifically weak to force all of them through the same detector merely because one architecture is convenient.

The project already possesses a broad neural architecture family — ANN, CNN/Conv1D+BiLSTM, LSTM, Transformer, N-BEATS, TFT, TCN, MIMO and multi-branch composites — so STEP 07 should **not** re-run those architectures indiscriminately. Instead it should add detector families with genuinely distinct inductive biases, use current project models as controlled downstream cores/baselines, and preserve the architecture modularity.

The most important new detector families to benchmark are:

1. exact matched filter / normalized cross-correlation;
2. generalized matched filter with noise whitening;
3. trainable matched-filter/template banks;
4. learned shapelets;
5. ROCKET / MiniRocket / MultiRocket;
6. HYDRA;
7. Matrix Profile and multivariate L-MAP;
8. InceptionTime as a learned multi-scale convolution baseline;
9. representation-specific 2D time-frequency detectors;
10. graph/spectral detectors for cross-series structures;
11. self-supervised/foundation encoders only after strong low-cost baselines are established.

---

# 1. Relationship to STEP 06

STEP 06 can produce heterogeneous representations such as:

\[
R_t
\in
\{
X_t,
D(X_t),
E_t,
FFT(X),
|FFT(X)|,
\phi(X),
STFT(X),
CWT(X),
DWT(X),
Hilbert(X),
S_{XY}(f),
C_{XY}(f),
\ldots
\}.
\]

STEP 07 must therefore define a **representation-to-detector contract**.

For each representation \(R\), the implementation must declare:

- tensor geometry;
- physical/statistical interpretation;
- units/scaling;
- temporal support;
- whether values are real, complex, angular, categorical or graph-structured;
- causal availability time;
- expected invariances;
- noise model, where one is assumed;
- detector family;
- detector output contract;
- whether the detector is fixed, training-only fitted, or online adaptive;
- whether the detector is supervised, self-supervised, or unsupervised.

---

# 2. Pattern detection is not forecasting

A detector should answer a bounded statistical question such as:

\[
\text{“How strongly does pattern }k\text{ occur near time }t\text{?”}
\]

rather than directly replacing the entire forecasting model.

Let:

\[
D_k(R_t)
=
a_{k,t}.
\]

The downstream model receives:

\[
[a_{1,t},\ldots,a_{K,t}]
\]

alongside selected original/transformed inputs.

Then the project core learns:

\[
Z_t
=
C(
R_t,
A_t
)
\]

and predictive head \(h\) computes:

\[
\hat Y_{t,h}
=
H_h(Z_t).
\]

This maintains separation between:

1. signal representation;
2. pattern evidence;
3. representation consolidation;
4. predictive decision.

---

# 3. Classical matched filtering

Consider:

\[
H_0:
x=n
\]

versus:

\[
H_1:
x=a s+n,
\]

where:

- \(s\) is a known deterministic template;
- \(a\) is an amplitude, possibly known or unknown;
- \(n\) is additive noise.

Under white Gaussian noise with covariance:

\[
C_n=\sigma_n^2 I,
\]

the matched-filter statistic is proportional to:

\[
T(x)
=
s^T x.
\]

A normalized statistic is:

\[
\boxed{
T_N(x)
=
\frac{s^Tx}
{\sigma_n\|s\|_2}
}
\]

which, under \(H_0\), has a standard normal distribution under the ideal assumptions.

For false-alarm probability:

\[
P_{FA}=\alpha,
\]

one may set:

\[
\gamma
=
\Phi^{-1}(1-\alpha)
\]

and declare detection when:

\[
T_N>\gamma.
\]

This gives STEP 07 an unusually clean synthetic falsification benchmark.

---

# 4. Why the matched filter matters to our overall chain

The prior stages developed:

\[
\text{noise/SNR}
\rightarrow
\text{denoising}
\rightarrow
\text{resolution}
\rightarrow
\text{representation}.
\]

Matched filtering now explicitly uses knowledge of:

- the expected pattern;
- the noise background;
- the representation geometry.

Thus:

\[
\boxed{
\text{STEP 03 noise model}
+
\text{STEP 06 representation}
\rightarrow
\text{STEP 07 optimal-pattern detector candidate}
}
\]

This is one of the strongest mathematical links in the entire communications-to-ML analogy.

---

# 5. Generalized matched filter under colored noise

If:

\[
n\sim\mathcal N(0,C_n)
\]

with non-white covariance:

\[
C_n\neq\sigma^2I,
\]

the optimum linear statistic is proportional to:

\[
T_C(x)
=
s^T C_n^{-1}x.
\]

Normalized:

\[
\boxed{
T_{C,N}(x)
=
\frac{
s^T C_n^{-1}x
}{
\sqrt{s^TC_n^{-1}s}
}
}
\]

assuming covariance is known or estimated.

Equivalent interpretation:

1. whiten the signal/noise;
2. correlate with the whitened template.

Define whitening matrix:

\[
W
\]

such that:

\[
WC_nW^T=I.
\]

Then:

\[
x_w=Wx,
\qquad
s_w=Ws
\]

and matched filtering becomes:

\[
T=s_w^Tx_w.
\]

This directly reuses STEP 03's noise-estimation work.

---

# 6. Critical covariance-governance rule

For project data:

\[
\widehat C_n
\]

must be estimated from training-only data or updated strictly causally.

For high-dimensional feature windows:

\[
p\gg N
\]

or poorly conditioned covariance matrices, direct inversion is dangerous.

Candidate regularized estimator:

\[
\widehat C_\lambda
=
(1-\lambda)\widehat C
+
\lambda I
\]

or shrinkage covariance.

All regularization parameters must be selected on training/validation only.

---

# 7. Unknown amplitude and sign

If:

\[
a
\]

is unknown but sign is known, matched-filter output can be optimized/thresholded directly.

If sign is unknown, useful statistics include:

\[
|s^Tx|
\]

or:

\[
(s^Tx)^2.
\]

For financial motifs where bullish/bearish patterns are sign-symmetric, explicitly compare:

- signed detector;
- absolute detector;
- paired \(s\) and \(-s\) templates.

---

# 8. Unknown temporal location

If template location is unknown, compute correlation over lag:

\[
r_{xs}[\tau]
=
\sum_n
x[n]s[n-\tau].
\]

Detection statistic:

\[
T_{\max}
=
\max_{\tau\in\mathcal T}
r_{xs}[\tau].
\]

Return not only:

\[
T_{\max}
\]

but:

\[
\hat\tau
=
\arg\max_\tau
r_{xs}[\tau].
\]

The lag becomes a pattern-position feature.

Because maximizing across many lags inflates false-alarm probability, threshold calibration must account for the maximum statistic rather than reuse a single-lag threshold.

---

# 9. Unknown pattern duration / scale

A fixed template is insufficient when the same motif appears at different durations.

Construct scale bank:

\[
s_{k,\lambda}
=
\mathrm{Resample}(s_k,\lambda).
\]

Detector:

\[
T_k
=
\max_{\lambda,\tau}
\mathrm{corr}
(
x,
s_{k,\lambda,\tau}
).
\]

Return:

- maximum activation;
- detected lag;
- detected scale;
- second-best margin;
- uncertainty.

This is a direct precursor to convolutional multi-scale detector banks.

---

# 10. Matched filtering and Conv1D: exact relationship and important correction

A Conv1D layer computes a learned cross-correlation-like operation:

\[
y_k[t]
=
\sum_i
w_k[i]x[t+i]
+b_k.
\]

This resembles a bank of template correlators.

However:

> A learned convolutional kernel is **not automatically a matched filter**.

It becomes a classical matched filter only under additional conditions tying:

- kernel shape to the target template;
- weighting to the noise covariance;
- decision statistic to the associated likelihood-ratio problem.

Therefore in this document:

- **matched filter** = detector with explicit signal/noise statistical interpretation;
- **convolutional pattern detector** = learned or random kernel bank;
- **matched-filter analogue** = convolutional detector whose function resembles template matching but lacks full detection-theory optimality.

This terminology should remain strict.

---

# 11. Representation-to-detector map

## 11.1. Raw / denoised / innovation 1D sequences

Recommended detector families:

- normalized cross-correlation;
- white-noise matched filter;
- generalized matched filter;
- shapelets;
- ROCKET-family kernel transforms;
- HYDRA;
- InceptionTime;
- TCN;
- Matrix Profile / streaming motif distance;
- learnable motif/template banks.

## 11.2. Complex Fourier representation

Recommended:

- Hermitian complex correlation;
- complex cosine similarity;
- spectral template bank;
- real/imaginary Conv1D;
- phase-normalized matching;
- frequency-selective learned masks.

## 11.3. Magnitude spectrum / PSD

Recommended:

- 1D spectral template correlation;
- bandwise filter banks;
- spectral shapelets;
- MLP/FreTS/FITS-style detector;
- peak/band detector with uncertainty.

## 11.4. STFT / CWT

Recommended:

- 2D matched filter;
- 2D normalized correlation;
- Conv2D/Inception-style detector;
- ridge/path detector;
- lightweight axial attention;
- time-frequency shapelets.

## 11.5. DWT / multiscale branches

Recommended:

- one detector bank per scale;
- causal TCN;
- MiniRocket/HYDRA per scale;
- learned shapelet bank;
- branch-specific Conv1D.

## 11.6. Hilbert amplitude / phase

Recommended:

- amplitude template correlation;
- circular phase similarity;
- phase-locking/synchrony descriptors;
- \(\cos(\Delta\phi)\)-based template scores;
- phase-aware Conv1D;
- PARCNet-inspired SFM reference module.

## 11.7. Cross-spectrum / coherence matrices

Recommended:

- graph neural networks;
- graph convolution;
- graph attention;
- spectral graph models;
- StemGNN-style graph Fourier + temporal Fourier processing;
- change detectors on coherence/phase-lag graphs.

## 11.8. STEP-05 token / motif / surprisal streams

Recommended:

- context trees;
- HMM/state models;
- event-token Transformer;
- hazard models;
- sequence classifiers.

A matched filter is not a natural default for categorical event streams.

---

# 12. Complex matched filtering

For complex template:

\[
s\in\mathbb C^N
\]

and observation:

\[
x\in\mathbb C^N,
\]

use Hermitian inner product:

\[
\boxed{
T(x)
=
s^H x
}
\]

where:

\[
s^H
=
(s^*)^T.
\]

Normalized:

\[
\rho_c
=
\frac{
s^Hx
}{
\|s\|_2\|x\|_2
}.
\]

Useful outputs:

\[
|\rho_c|
\]

and:

\[
\arg(\rho_c).
\]

This is appropriate for STEP-06 complex Fourier/STFT branches.

---

# 13. Phase-invariant versus phase-sensitive detection

Two valid but different questions exist.

## Phase-sensitive detector

Use:

\[
s^Hx.
\]

Pattern must align in phase.

## Phase-invariant detector

Use:

\[
|s^Hx|.
\]

Global phase rotation is ignored.

These detectors test different hypotheses and should both be benchmarked where phase is meaningful.

---

# 14. Circular phase matching

Given phase trajectories:

\[
\phi_x[t],
\qquad
\phi_s[t],
\]

use circular similarity:

\[
c_\phi
=
\frac1N
\sum_t
\cos(
\phi_x[t]-\phi_s[t]
).
\]

Weighted by amplitude:

\[
c_{\phi,A}
=
\frac{
\sum_t
w_t
\cos(
\phi_x[t]-\phi_s[t]
)
}{
\sum_t w_t
}
\]

with:

\[
w_t
=
g(A_x[t],A_s[t]).
\]

This avoids assigning strong meaning to phase where amplitude is near zero.

---

# 15. 2D matched filtering for time-frequency maps

For representation:

\[
X(\tau,f)
\]

and template:

\[
S(\tau,f),
\]

2D correlation:

\[
T(\Delta\tau,\Delta f)
=
\sum_{\tau,f}
X(\tau,f)
S^*(\tau-\Delta\tau,f-\Delta f).
\]

This permits pattern detection under:

- temporal shift;
- frequency shift;
- local chirp/ridge changes.

The synthetic chirp experiments from STEP 06 are ideal validation data.

---

# 16. Shapelets: discriminative learned subsequences

A shapelet:

\[
S_k
\]

is a discriminative subsequence.

Classical feature:

\[
d_k(X)
=
\min_\tau
\|
X_{\tau:\tau+L_k-1}
-
S_k
\|^2.
\]

Grabocka et al. showed that shapelets can be learned directly rather than enumerated exhaustively.

Relevance:

\[
\boxed{
\text{learnable template}
+
\text{shift invariance}
+
\text{interpretable local pattern}
}
\]

which makes shapelets a natural ML generalization of template detection.

---

# 17. Learnable multivariate shapelets

For:

\[
X\in\mathbb R^{T\times C},
\]

a multivariate shapelet may span:

- all channels;
- selected channels;
- learned channel projections.

This is preferable when the pattern is inherently joint, e.g.:

- price change + volatility response;
- asset move + cross-asset reaction;
- macro surprise + yield/FX response.

Use channel-selection regularization where possible to prevent opaque all-channel templates.

---

# 18. Matrix Profile: unsupervised motif/discord detector

For a subsequence length:

\[
m,
\]

Matrix Profile stores the distance from each subsequence to its nearest nontrivial neighbor.

Low distance:

\[
\rightarrow
\text{motif-like recurrence}.
\]

High distance:

\[
\rightarrow
\text{discord/anomaly}.
\]

It provides an unsupervised pattern discovery control without neural training.

This is valuable because STEP 07 should distinguish:

- recurring pattern;
- discriminative/predictive pattern.

---

# 19. Matrix Profile causality warning

Standard all-pairs Matrix Profile computed on the entire train+validation/test history is **not valid** for online prediction because a subsequence can match a future occurrence.

For forecast time \(t\), allowed reference set must be:

\[
\mathcal R_t
\subseteq
\{1,\ldots,t\}.
\]

Preferred project modes:

## Frozen training dictionary

Validation/test subsequences query only motifs discovered in training.

## Streaming/left profile

Query current subsequence only against past observations.

Future nearest neighbors are forbidden.

---

# 20. L-MAP 2026: multivariate learnable matrix profile

AAAI 2026 introduced Learnable Multivariate Matrix Profile (L-MAP).

Important ideas:

- subsequences partitioned using frequency-domain information;
- locality-sensitive hashing for grouping;
- subsequences modeled as graphs;
- triplet learning captures inter-dimensional relations;
- matrix profile constructed in learned latent space.

This is especially relevant after STEP 06 because it directly combines:

\[
\text{frequency representation}
+
\text{multivariate graph structure}
+
\text{motif discovery}.
\]

Project action:

> Benchmark L-MAP before designing a custom deep multivariate motif detector.

---

# 21. ROCKET family: detector bank without expensive representation learning

ROCKET transforms a time series using many convolutional kernels and summary operations, then fits a linear classifier.

The core idea is remarkably aligned with this project:

\[
\boxed{
\text{large bank of local detectors}
\rightarrow
\text{summary evidence}
\rightarrow
\text{simple decision model}
}
\]

This provides a powerful control against the assumption that every pattern detector needs a large trainable deep network.

---

# 22. MiniRocket

MiniRocket retains most of ROCKET's accuracy while making the transform almost deterministic and dramatically faster.

This makes it an excellent high-throughput control for STEP 07.

Recommended use:

1. validate representation;
2. run MiniRocket feature extraction;
3. use Ridge/logistic/linear regression baseline;
4. feed selected MiniRocket evidence into the project core as an optional branch.

If a complex learned detector cannot outperform this baseline reliably, it should not be promoted.

---

# 23. MultiRocket

MultiRocket expands MiniRocket by:

- raw series;
- first differences;
- multiple pooling operators.

This matters because it already tests a principle similar to the overall project:

> distinct representations of the same signal can create complementary pattern evidence.

Therefore MultiRocket is a **mandatory low-cost benchmark**.

---

# 24. HYDRA

HYDRA uses groups of competing random convolutional kernels.

Within a group:

\[
\text{kernel activation}
\rightarrow
\arg\max
\rightarrow
\text{winner count}.
\]

The resulting counts behave like a learned/dictionary-like pattern representation.

This is a particularly clean bridge between:

- matched-filter banks;
- convolution;
- dictionary coding;
- pattern recognition.

Recommended project experiment:

\[
\text{MultiRocket}
\]

vs:

\[
\text{HYDRA}
\]

vs:

\[
\text{MultiRocket + HYDRA}.
\]

---

# 25. HIVE-COTE 2.0 and the lesson of representation diversity

HIVE-COTE 2.0 combines heterogeneous classifier families, including:

- dictionary-based representations;
- interval-based features;
- shapelet-type information;
- ROCKET/Arsenal kernels.

It achieved very strong average performance across UCR/UEA benchmark archives.

The critical lesson is not that HIVE-COTE should be embedded directly in the financial predictor.

The lesson is:

\[
\boxed{
\text{heterogeneous representation/detector diversity can beat one universal representation}
}
\]

which strongly supports the project's multi-branch philosophy.

---

# 26. InceptionTime

InceptionTime applies multiple convolutional kernel sizes in parallel.

This gives explicit scale diversity:

\[
k_1,
k_2,
k_3,\ldots
\]

within one learned detector.

It is therefore a direct learned baseline for:

\[
\text{pattern duration unknown}.
\]

Project use:

- TSC benchmark;
- raw/innovation branch;
- parameter-matched competitor to custom multi-scale filter banks.

---

# 27. Existing TCN implementation in `predictor`

The repository already contains a TCN plugin with:

- causal dilated Conv1D;
- exponential dilation;
- residual connections;
- normalization;
- dropout;
- per-horizon heads.

Therefore:

> do not implement another ordinary dilated Conv1D detector as a new STEP-07 contribution.

Instead, use the existing TCN as:

1. a strong learnable receptive-field baseline;
2. a downstream detector for selected STEP-06 1D branches;
3. a comparator to explicit matched-filter/shapelet/kernel-bank methods.

---

# 28. Existing broad predictor model family

The current predictor repository already includes at least:

- ANN;
- CNN;
- LSTM;
- Transformer;
- N-BEATS;
- TFT;
- TCN;
- MIMO;

with binary/directional variants in the current tree.

The current `phase_1b_binary` champion configuration uses a TFT-based predictor with a window of 124 and existing train/validation/test separation.

Therefore STEP 07 should not be framed as:

> “find one new SOTA forecasting architecture.”

It should instead determine:

\[
\boxed{
\text{which detector family best matches each representation}
}
\]

while reusing existing project cores/heads.

---

# 29. Existing composite architecture is already a good STEP-07 host

The project already has a composite multi-branch predictor separating:

- full-window CLOSE;
- high-frequency 15m chunk;
- high-frequency 30m chunk;
- point/context features;

followed by:

- concatenation;
- fused Conv1D;
- per-head Conv1D;
- BiLSTM;
- Bayesian Flipout output;
- deterministic bias path.

This is exactly the kind of architecture needed to integrate detector evidence.

The recommended STEP-07 change is therefore **branch specialization**, not replacement of the complete predictor.

---

# 30. Endpoint-causal versus timestamp-causal modeling

This distinction is essential.

## Endpoint-causal

At prediction time \(t\), model may use all samples:

\[
x_{t-W+1:t}.
\]

A bidirectional model operating **inside that fully observed historical window** does not leak beyond \(t\).

## Timestamp-causal

A feature emitted for every internal timestamp \(u\) must depend only on:

\[
x_{\leq u}.
\]

A BiLSTM, centered convolution or full-window Hilbert transform may violate timestamp causality for internal positions even though the final endpoint forecast remains causal.

Every STEP-07 branch must declare which causality contract it satisfies.

---

# 31. Representation-specific detector recommendation table

| STEP-06 output | Geometry | Primary detector | Strong baseline | Advanced candidate |
|---|---|---|---|---|
| Raw / denoised | 1D real | matched/Conv bank | MiniRocket | InceptionTime / shapelets |
| Innovation | 1D real | matched/shapelet | MultiRocket | TCN |
| FFT magnitude | 1D nonnegative spectral | spectral templates | MLP | learned spectral kernels |
| FFT complex | 1D complex | Hermitian matching | real/imag MLP | complex/spectral detector |
| Phase | circular | phase correlation | sin/cos MLP | phase-aware Conv |
| STFT | 2D complex | 2D correlation | Conv2D | axial attention |
| CWT | 2D complex/multiscale | 2D template | Conv2D | wavelet shapelets |
| DWT bands | multi-1D | per-band banks | MiniRocket per band | branch TCN/Inception |
| Hilbert amp/phase | 1D + circular | amplitude/phase matching | PARCNet SFM ref | phase-gated Conv |
| Coherence graph | graph x frequency | graph detector | MLP summaries | StemGNN/MTGNN |
| Motif token | categorical | context/event model | HMM/Markov | Transformer |
| Surprisal | scalar/vector sequence | event detector | logistic/hazard | TCN/Transformer |

---

# 32. Graph-based pattern detectors

Cross-series data create a different problem.

Let feature/asset graph:

\[
G_t
=
(V,E_t).
\]

Node values:

\[
x_{i,t}.
\]

Edge weights may represent:

- coherence;
- phase lag;
- correlation;
- conditional code gain;
- economically specified adjacency.

Graph detector outputs pattern evidence:

\[
Z_t
=
GNN(G_t,X_t).
\]

---

# 33. Graph WaveNet / MTGNN / StemGNN role

These established architectures address learned multivariate dependencies.

## Graph WaveNet

Learns adaptive adjacency plus temporal convolutions.

## MTGNN

Learns directed relations and combines graph propagation with temporal convolution.

## StemGNN

Jointly uses:

- Graph Fourier Transform for inter-series relations;
- DFT for temporal spectral structure.

StemGNN is especially relevant because it sits exactly at:

\[
\boxed{
\text{STEP 06 spectral representation}
+
\text{STEP 07 multivariate pattern detector}
}
\]

and should be benchmarked before building a bespoke spectral graph detector.

---

# 34. TimeMixer++ as a recent pattern-machine baseline

TimeMixer++, ICLR 2025, explicitly presents itself as a general Time Series Pattern Machine.

It combines:

- multiple time scales;
- multiple frequency resolutions;
- time-image decomposition;
- multi-scale mixing;
- multi-resolution mixing.

It spans forecasting, classification, anomaly detection and other tasks.

Project action:

> Use TimeMixer++ as a high-level representation/pattern-extraction reference, especially when evaluating whether separate manually designed branches are superior to a more unified multi-scale pattern machine.

---

# 35. Self-supervised representation detectors

When labels are scarce, a pretrained representation may identify useful patterns before target-specific fine-tuning.

Candidate established methods:

- TS2Vec;
- TimesURL;
- MOMENT;
- Mantis.

These should be evaluated **after** low-cost supervised/kernel controls.

Reason:

> foundation-model complexity is not evidence of superiority.

A simple MultiRocket/HYDRA detector remains a mandatory comparator.

---

# 36. TS2Vec

TS2Vec learns hierarchical contrastive representations at timestamp/subsequence scales.

Potential use:

\[
X
\rightarrow
E_{\mathrm{TS2Vec}}(X)
\rightarrow
\text{small pattern head}.
\]

Useful test:

> Does a frozen TS2Vec encoder improve pattern detection under low-label regimes?

Do not automatically fine-tune large encoders on the full project dataset before establishing a frozen baseline.

---

# 37. TimesURL

TimesURL combines:

- time-frequency augmentation;
- contrastive learning;
- reconstruction.

This is especially aligned with STEP 06.

Potential experiment:

\[
T_{STEP6}(X)
\rightarrow
E_{\mathrm{TimesURL}}
\]

versus:

\[
X
\rightarrow
E_{\mathrm{TimesURL}}.
\]

But avoid double-transforming if the pretrained model expects raw inputs.

---

# 38. MOMENT

MOMENT is an open time-series foundation-model family trained on a broad Time Series Pile.

Use as:

- frozen embedding baseline;
- limited-supervision comparator;
- anomaly/pattern representation benchmark.

It should not replace purpose-built project detectors without evidence.

---

# 39. 2025–2026 foundation/in-context classifiers: watchlist, not first-line baseline

Recent models include:

- Mantis;
- TiCT;
- TimEE;
- RocketPFN.

TimEE (2026) is a small in-context time-series classifier pretrained on synthetic tasks.

RocketPFN (2026 preprint) combines ROCKET features with a pretrained tabular foundation model and reports performance competitive with HIVE-COTE 2.0 on large UCR evaluations.

These are scientifically interesting, but:

- some are recent preprints;
- protocols continue to evolve;
- their classification task differs from financial forecasting.

Therefore classify them as **research watchlist / low-label pattern-classification comparators**, not as mandatory production architecture.

---

# 40. PARCNet benchmark audit requested by project owner

The official PARCNet repository currently contains experiment scripts for exactly these 12 benchmark names:

1. `electricity`
2. `ETTh1`
3. `ETTh2`
4. `ETTm1`
5. `ETTm2`
6. `PEMS03`
7. `PEMS04`
8. `PEMS07`
9. `PEMS08`
10. `solar`
11. `traffic`
12. `weather`

This exact list should be used when reproducing the public code package.

---

# 41. Are PARCNet's benchmarks useful to this project?

Yes, but for different reasons.

## ETT family

Useful for:

- long-horizon temporal forecasting;
- periodic/spectral patterns;
- direct comparison with many modern forecasting papers.

## Electricity

Useful for:

- high-dimensional multivariate structure;
- periodicity;
- channel relationships.

## Weather

Useful for:

- multiple physical signals;
- mixed temporal scales;
- nonlinear relationships.

## Solar

Useful for:

- strong periodicity;
- amplitude variation;
- phase/timing behavior.

## Traffic

Useful for:

- large multivariate systems;
- strong periodicity;
- graph-like spatial dependencies.

## PEMS03/04/07/08

Especially valuable for STEP 07 because:

- hundreds of channels/sensors;
- repeated local patterns;
- cross-channel dependencies;
- graph/multivariate detector evaluation.

Thus PARCNet's suite is a very good **STEP-06/07 general forecasting benchmark layer**.

---

# 42. What PARCNet's 12 benchmarks do NOT provide

They do not replace:

## UCR/UEA

for explicit time-series **pattern classification**.

## Synthetic detection suite

for matched-filter optimality and controlled SNR.

## Project financial datasets

for actual domain transfer.

Therefore recommended benchmark stack is:

\[
\boxed{
\text{Synthetic detection}
\rightarrow
\text{UCR/UEA pattern recognition}
\rightarrow
\text{PARCNet 12 forecasting datasets}
\rightarrow
\text{project financial datasets}
}
\]

This triangulates detector correctness, generic pattern-recognition performance, forecasting transfer and domain utility.

---

# 43. PARCNet implementation audit note

The official code deserves an implementation-level audit before copying its Hilbert processing.

The released SFM implementation:

- expects a tensor described as \((B,D,L)\);
- calls SciPy `hilbert(..., axis=1)`.

The repository's later `DataEmbedding` implementation permutes the sequence/channel dimensions and linearly maps sequence length into a learned embedding before SFM.

Therefore the exact axis on which Hilbert is applied in the released implementation is not simply equivalent to:

> “apply causal Hilbert transform along the original raw time axis.”

This may be intentional, but it must be reproduced exactly for paper benchmarking and **kept separate** from the project's own causal temporal Hilbert branch.

Do not silently merge the two interpretations.

---

# 44. PARCNet reproducibility examples

The official ETTh1 script uses:

- sequence length \(96\);
- prediction horizons \(96,192,336,720\);
- 7 input channels;
- cycle \(24\);
- 30 training epochs;
- batch size 32.

The official PEMS03 script uses:

- sequence length \(96\);
- horizons \(12,24,48,96\);
- 358 channels;
- cycle \(288\);
- 30 epochs;
- batch size 32.

This diversity makes the suite useful for testing whether a detector scales from modest channel count to genuinely high-dimensional multivariate input.

---

# 45. Benchmark layer A — synthetic matched-filter validation

This layer is mandatory.

Construct synthetic patterns with exact ground truth.

## A1 — known pulse

\[
s[n].
\]

## A2 — sinusoidal burst

\[
s[n]
=
w[n]\cos(2\pi fn+\phi).
\]

## A3 — chirp

\[
f=f(t).
\]

## A4 — wavelet-shaped transient

\[
s[n]
=
\psi_{a,b}[n].
\]

## A5 — multivariate coordinated motif

\[
\mathbf s[n]
=
[s_1[n],\ldots,s_C[n]].
\]

## A6 — phase-shifted motif

constant magnitude, variable phase.

---

# 46. Synthetic noise regimes

For each template test:

## White Gaussian

\[
n_t\sim N(0,\sigma^2).
\]

## Colored AR(1)

\[
n_t
=
\rho n_{t-1}
+
\epsilon_t.
\]

## Heteroscedastic

\[
\sigma_t=f(t).
\]

## Heavy-tailed

Student-\(t\) or robust alternative.

## Cross-correlated multivariate noise

\[
n_t
\sim
N(0,\Sigma).
\]

Sweep:

\[
SNR
\in
\{
40,30,20,15,10,5,0,-5
\}\mathrm{dB}
\]

where meaningful.

---

# 47. Synthetic detector comparison

At minimum:

1. ordinary correlation;
2. normalized correlation;
3. white matched filter;
4. covariance-aware matched filter;
5. learnable Conv1D;
6. learnable shapelet;
7. MiniRocket;
8. HYDRA.

Primary metrics:

- \(P_D\) at fixed \(P_{FA}\);
- ROC-AUC;
- PR-AUC for rare occurrence;
- localization error:
  \[
  |\hat\tau-\tau|;
  \]
- scale error;
- calibration;
- runtime.

---

# 48. Exact theoretical falsification test

Under white Gaussian noise and known template:

> A learned detector should not be described as “better than the matched filter” solely because its training-set classification accuracy is higher.

Under the exact matched-filter assumptions, the classical detector has a known optimality result.

If a learned model appears superior, first test whether:

- amplitude is unknown;
- template varies;
- noise is non-Gaussian;
- localization uncertainty exists;
- training procedure introduces additional prior knowledge;
- metric differs from the classical detection objective.

This protects the experiment against misleading claims.

---

# 49. Benchmark layer B — UCR/UEA pattern-recognition sanity suite

Recommended primary controls:

- MiniRocket;
- MultiRocket;
- HYDRA;
- InceptionTime;
- HIVE-COTE 2.0 where computationally feasible;
- TS2Vec frozen features;
- Mantis/TimEE/RocketPFN only as current research comparators.

Purpose:

> verify that the detector implementation behaves competitively on established pattern-recognition problems before financial integration.

---

# 50. Benchmark layer C — PARCNet 12 forecasting suite

Use the official 12 script families.

Initial focus subset for computational efficiency:

1. ETTh1;
2. ETTm1;
3. Weather;
4. Electricity;
5. PEMS03;
6. Traffic.

This subset spans:

- small/medium multivariate;
- high-dimensional multivariate;
- regular cycles;
- local transients;
- graph-like dependencies.

After gates pass, expand to all 12.

---

# 51. Benchmark layer D — project financial data

Use:

- EURUSD 1 h;
- EURUSD 4 h;
- ETHUSDT 4 h;
- other currently governed datasets.

Split:

\[
4\text{ years train}
+
1\text{ year validation}
+
1\text{ year test}.
\]

All pattern/template learning:

\[
\theta_D
\]

must be fitted within training only.

---

# 52. How STEP 07 should enter `predictor`

The initial experiment should **freeze the existing downstream core/head architecture** as much as practical.

Add branch:

\[
B_D
=
D(R(X)).
\]

Then compare:

## Base

\[
P(X).
\]

## Detector only

\[
P(D(R(X))).
\]

## Parallel

\[
P([X,D(R(X))]).
\]

## Representation + detector

\[
P([X,R(X),D(R(X))]).
\]

The parallel condition is especially important because a detector necessarily compresses information.

---

# 53. Detector output contract

Every detector plugin should return a consistent logical structure.

Recommended fields:

## Sequence activation

\[
A
\in
\mathbb R^{T\times K}
\]

optional.

## Summary activation

\[
a
\in
\mathbb R^K.
\]

## Pattern IDs

top-\(K_p\) identifiers.

## Location

\[
\hat\tau_k.
\]

## Scale

\[
\hat\lambda_k.
\]

## Frequency/band

\[
\hat f_k.
\]

## Phase

\[
\hat\phi_k.
\]

## Confidence / calibrated probability

\[
p_k.
\]

## SNR/quality

\[
q_k.
\]

## Validity mask

\[
m_k.
\]

This lets the core consume detector evidence without knowing implementation details.

---

# 54. Do not pass every activation map by default

A detector bank can produce:

\[
T\times K
\]

features.

Blindly concatenating all maps can explode dimension and model capacity.

Test three modes:

## Summary

top activation statistics only.

## Sparse top-\(K\)

keep only strongest detectors.

## Full map

only when sequence-level localization is necessary.

This is another falsifiable capacity/representation tradeoff.

---

# 55. Pattern-bank construction strategies

## Fixed analytical templates

e.g. pulses/chirps/wavelets.

## Training-discovered motifs

Matrix Profile / clustering.

## Supervised learned shapelets

optimize discrimination.

## Random kernels

ROCKET/HYDRA.

## Learned Conv kernels

gradient-trained.

## Foundation/pretrained kernels

fixed encoder.

These represent fundamentally different priors and should not be conflated.

---

# 56. Template origin audit

Every template must declare provenance:

- analytical;
- training-only extracted;
- learned supervised;
- learned self-supervised;
- external pretrained;
- manually specified.

A template derived from test examples invalidates evaluation.

---

# 57. Template-bank diversity regularization

Learned templates often collapse to similar patterns.

Possible diversity penalty:

\[
L_{div}
=
\sum_{i\neq j}
|\rho(s_i,s_j)|^2.
\]

Or orthogonality penalty:

\[
L_{orth}
=
\|
SS^T-I
\|_F^2.
\]

Hypothesis:

> pattern-bank diversity improves coverage and reduces redundant detector capacity.

This must be ablated.

---

# 58. Supervised versus unsupervised pattern detectors

## Unsupervised

discover recurrence:

\[
p(X).
\]

Examples:

- Matrix Profile;
- clustering;
- autoencoder/VQ motif discovery.

## Supervised

optimize target relevance:

\[
p(Y|X).
\]

Examples:

- shapelets;
- learned Conv filters;
- hazard detectors.

A frequent motif may have zero predictive value.

Therefore recurrence and prediction must remain separate metrics.

---

# 59. Auxiliary pattern labels

If economically meaningful labels exist, detector pretraining may use:

- trend onset;
- volatility burst;
- jump;
- reversal;
- breakout;
- rush onset;
- rush continuation;
- rush termination;
- liquidity stress;
- macro-release reaction.

But labels must be formally defined from future outcomes and kept strictly in target construction, never leaked into inputs.

---

# 60. Integration with `agent-multi` rush detector

The existing `agent-multi` work plan already defines a probabilistic multi-horizon rush detector with:

- onset probability;
- continuation/termination;
- direction;
- intensity;
- duration;
- adverse regime probability;
- volatility/jump/liquidity stress;
- calibrated confidence.

This is an excellent downstream testbed for STEP-07 pattern evidence.

Recommended experiment:

\[
\text{base rush detector}
\]

vs:

\[
\text{base}
+
\text{pattern evidence}.
\]

Do not allow pattern detector to submit actions directly.

---

# 61. Event-token Transformer integration

`agent-multi` already has a train-only event-token Transformer encoder.

STEP-07 detector outputs can be tokenized as events such as:

- `PATTERN_K_ONSET`;
- `PATTERN_K_PEAK`;
- `PHASE_SYNC_HIGH`;
- `COHERENCE_BREAK`;
- `MOTIF_DISCORD`;
- `SURPRISAL_SPIKE`.

These can enter the existing event-token context encoder.

This is preferable to building an unrelated second event Transformer.

---

# 62. Pattern detector versus RL policy

For SAC/PPO/DQN:

\[
o_t
=
[
x_t,
d_t
]
\]

where:

\[
d_t
\]

is pattern evidence.

The RL policy should remain responsible for action selection.

STEP 07 must first establish that:

\[
d_t
\]

contains stable predictive/state information before measuring policy-level utility.

Otherwise RL can obscure whether detector itself works.

---

# 63. Detector selection ladder

For every representation \(R\):

1. analytical baseline;
2. cheap kernel bank;
3. interpretable learned template;
4. deep representation-specific detector;
5. pretrained/foundation model.

Example for raw sequence:

\[
\text{correlation}
\rightarrow
\text{MiniRocket}
\rightarrow
\text{shapelets}
\rightarrow
\text{InceptionTime}
\rightarrow
\text{foundation encoder}.
\]

Do not begin at stage 5.

---

# 64. Falsifiable hypotheses — detection theory

## H7.1 — Matched-filter optimality recovery

Under known-template AWGN synthetic conditions, matched filtering attains the expected detection advantage over generic correlation/noise-unaware controls.

**Falsified if:** implementation fails controlled theoretical behavior.

---

## H7.2 — Noise-aware generalized matched filter improves colored-noise detection

\[
P_D^{GMF}
>
P_D^{MF}
\]

at matched:

\[
P_{FA}
\]

when covariance is materially non-white.

**Falsified if:** whitening/covariance weighting gives no reproducible gain under correctly generated colored noise.

---

## H7.3 — STEP-03 noise estimation is good enough to support detection

A matched filter using estimated:

\[
\widehat C_n
\]

approaches performance of the oracle covariance detector.

**Falsified if:** estimation error removes most expected benefit.

---

# 65. Falsifiable hypotheses — pattern uncertainty

## H7.4 — Detector banks outperform one fixed template when scale/shape varies

\[
P(D_{bank})
>
P(D_{single}).
\]

**Falsified if:** additional bank diversity provides no benefit.

---

## H7.5 — Learnable shapelets outperform fixed correlation for uncertain motifs

When patterns vary around a family:

\[
P(D_{shapelet})
>
P(D_{fixed}).
\]

**Falsified if:** simple templates remain equivalent or superior.

---

## H7.6 — Random kernel banks are a strong low-cost control

MiniRocket/MultiRocket/HYDRA achieve a competitive detection/forecasting frontier relative to much larger learned detectors.

**Falsified if:** they consistently fail on the chosen representations/tasks.

---

# 66. Falsifiable hypotheses — representation matching

## H7.7 — Detector/representation matching matters

For representation \(R\), a geometry-aware detector:

\[
D_R
\]

outperforms a generic detector:

\[
D_G.
\]

Example:

\[
D_{complex}(FFT_{complex})
>
D_{real\ generic}(FFT_{complex}).
\]

**Falsified if:** generic detector matches all specialized variants.

---

## H7.8 — Phase-aware detector adds value when STEP-06 phase hypothesis survives

\[
P([X,D_\phi])
>
P(X).
\]

**Falsified if:** phase detector contributes no incremental value.

---

## H7.9 — 2D time-frequency detectors outperform flattened detectors

For STFT/CWT representations:

\[
P(D_{2D})
>
P(D_{flattened}).
\]

**Falsified if:** spatial/time-frequency geometry gives no advantage.

---

## H7.10 — Graph detector adds value for cross-series representations

For cross-spectral/coherence inputs:

\[
P(D_{graph})
>
P(D_{independent}).
\]

**Falsified if:** graph modeling gives no OOS gain.

---

# 67. Falsifiable hypotheses — architecture and capacity

## H7.11 — Existing multi-branch architecture benefits from detector evidence

\[
P([X,D(R(X))])
>
P(X).
\]

**Falsified if:** all detector evidence is redundant with the existing core.

---

## H7.12 — Pattern representation can reduce required core size

For equal target performance \(P_0\):

\[
|\theta_{\text{detector+small core}}|
<
|\theta_{\text{large raw core}}|.
\]

**Falsified if:** explicit pattern evidence requires equal/larger total capacity.

---

## H7.13 — Parameter-matched detector branches retain their gain

At equal total trainable parameter count, detector-enabled model remains superior.

**Falsified if:** gain disappears under capacity matching.

---

# 68. Falsifiable hypotheses — scale, memory and locality

## H7.14 — Multi-scale detection improves variable-duration motifs

\[
P(D_{multiscale})
>
P(D_{fixed-scale}).
\]

**Falsified if:** fixed scale is sufficient.

---

## H7.15 — STEP-05 context depth predicts useful detector receptive field

Optimal detector receptive field:

\[
W_D^*
\]

has stable association with source-memory estimate:

\[
k^*.
\]

**Falsified if:** no relationship generalizes.

---

# 69. Falsifiable hypotheses — target relevance

## H7.16 — Frequent motifs are not necessarily predictive

Matrix-Profile motif frequency alone is insufficient to rank downstream predictive utility.

This is a deliberate null-control hypothesis.

---

## H7.17 — Supervised detector relevance exceeds recurrence-only relevance

For forecasting:

\[
P(D_{supervised})
>
P(D_{motif-only})
\]

when recurrence is not target-aligned.

**Falsified if:** recurrence-only motifs consistently suffice.

---

# 70. Falsifiable hypotheses — rare event detection

## H7.18 — Pattern detector improves rare-event PR metrics

For rush/jump/event onset:

\[
AP_{\text{base+detector}}
>
AP_{\text{base}}.
\]

**Falsified if:** only accuracy improves while PR/lead-time does not.

---

## H7.19 — Pattern evidence improves calibrated hazard prediction

Brier/calibration and lead-time metrics improve.

**Falsified if:** detector produces overconfident alerts without utility.

---

# 71. Falsifiable hypotheses — foundation / pretrained models

## H7.20 — Pretrained encoders help most in low-label regimes

Benefit:

\[
\Delta P_{\text{pretrained}}
\]

should be larger when labeled training set is reduced.

**Falsified if:** no low-label advantage appears.

---

## H7.21 — Foundation models must beat simple kernel baselines to justify integration

At equal compute/latency constraints:

\[
P_{\text{foundation}}
>
P_{\text{MultiRocket/HYDRA}}.
\]

**Falsified if:** simpler methods dominate.

---

# 72. Falsifiable hypotheses — transfer and stability

## H7.22 — Detector patterns generalize across temporal regimes

Training-learned templates retain useful activation semantics in validation/test.

**Falsified if:** template activation distribution collapses or reverses.

---

## H7.23 — Some generic detector features transfer across assets

At least selected detector banks learned from one family transfer to related assets better than random initialization.

**Falsified if:** asset-specific retraining is always required.

---

## H7.24 — Detector performance depends on SNR

Pattern detector advantage:

\[
\Delta P_D
=
f(SNR).
\]

Expected—but not enforced—result:

- classical matched-filter advantage increases in low-SNR known-template regimes;
- large learned models may degrade more sharply when training examples are limited/noisy.

---

# 73. Detector output calibration

Pattern activation:

\[
a
\]

is not automatically a probability.

If downstream system needs:

\[
P(pattern|x),
\]

calibrate with training/validation-only:

- Platt/logistic scaling;
- isotonic regression;
- temperature scaling;
- Bayesian uncertainty model.

Report:

- Brier score;
- expected calibration error;
- reliability curves.

---

# 74. Rare-event metrics

Do not rely on accuracy.

Required:

- PR-AUC / average precision;
- ROC-AUC;
- precision at operating threshold;
- recall;
- false alerts per unit time;
- detection delay;
- mean lead time;
- event-level F1;
- Brier score;
- calibration;
- downstream economic/policy utility only after detector validity.

---

# 75. Forecasting metrics

When detector evidence feeds predictor:

- MAE;
- RMSE;
- \(R^2\);
- per-horizon error;
- direction accuracy if secondary;
- calibration/NLL for probabilistic heads;
- Diebold–Mariano where appropriate;
- block-bootstrap confidence intervals.

---

# 76. Synthetic detector statistics

For known events:

\[
P_D
=
P(T>\gamma|H_1).
\]

False alarm:

\[
P_{FA}
=
P(T>\gamma|H_0).
\]

Localization:

\[
E_\tau
=
|\hat\tau-\tau|.
\]

Scale:

\[
E_\lambda
=
|\hat\lambda-\lambda|.
\]

These must be reported before downstream forecasting.

---

# 77. Parameter-matched experiment

Let:

\[
N_{\theta,base}
\]

be base model parameters.

Detector model must have comparator:

\[
N_{\theta,det}
\approx
N_{\theta,base}.
\]

Also report total FLOPs/latency.

Otherwise:

> representation and detector effects are confounded with added capacity.

---

# 78. Compute-matched experiment

For SOTA comparisons, parameter count alone is insufficient.

Report:

- training GPU hours;
- inference latency;
- CPU feasibility;
- peak GPU memory;
- number of trainable parameters;
- preprocessing cost.

A detector that gains:

\[
0.1\%
\]

at 100x compute may not be rational.

---

# 79. Pattern-bank interpretability

For learned templates, archive:

- template values;
- channel masks;
- scale;
- average activation;
- activation examples;
- target-conditioned activation;
- validation stability.

For Conv kernels, optionally compute nearest training subsequence:

\[
\arg\max_x
\mathrm{activation}(w,x)
\]

to obtain a concrete exemplar.

---

# 80. Activation stability

For detector \(k\), compare distribution:

\[
p_{train}(a_k),
p_{val}(a_k),
p_{test}(a_k).
\]

Use:

- Wasserstein distance;
- KS statistic;
- PSI where useful;
- calibration drift.

Large shift flags detector fragility.

---

# 81. Detector redundancy

For detector outputs:

\[
A_1,\ldots,A_K,
\]

measure:

- Pearson/Spearman;
- mutual information;
- conditional coding gain from STEP 05;
- target-conditional incremental utility.

Prune redundant detectors only from training/validation evidence.

---

# 82. Diversity versus redundancy

A large bank is useful only if patterns are complementary.

A proposed criterion:

\[
J
=
\text{predictive utility}
-
\lambda
\text{redundancy}.
\]

Do not optimize solely for low detector correlation; two uncorrelated useless detectors remain useless.

---

# 83. Negative controls

Mandatory controls:

## Shuffled labels

Detector should lose supervised target utility.

## Time-shifted template

Activation should decline as alignment is destroyed.

## Phase-randomized input

Phase-sensitive detectors should degrade predictably.

## Noise-only data

False-alarm rate should match calibration.

## Random kernels

Compare against learned kernels.

## Random template dictionary

Compare against motif/shapelet bank.

---

# 84. Leakage controls

Forbidden:

- motif dictionary built from validation/test;
- shapelet learned from test;
- matrix-profile nearest neighbor from future;
- normalization using future;
- centered detector consuming future beyond forecast endpoint;
- post-event labels entering pre-event features;
- graph edges selected using test performance.

---

# 85. Minimal STEP-07 experiment matrix

| ID | Representation | Detector | Task | Model downstream |
|---|---|---|---|---|
| D00 | synthetic raw | correlation | known-pattern detection | threshold |
| D01 | synthetic raw | matched filter | known-pattern detection | threshold |
| D02 | synthetic raw | generalized MF | colored-noise detection | threshold |
| D03 | raw | MiniRocket | UCR/UEA classification | linear |
| D04 | raw + diff | MultiRocket | UCR/UEA classification | linear |
| D05 | raw + diff | HYDRA | UCR/UEA classification | linear |
| D06 | raw | learned shapelets | UCR/UEA | linear/MLP |
| D07 | raw | InceptionTime | UCR/UEA | native |
| D08 | STEP06 best 1D | MiniRocket/HYDRA | forecasting | DLinear/core |
| D09 | STEP06 time-frequency | Conv2D | forecasting | core |
| D10 | STEP06 phase | phase detector | forecasting | core |
| D11 | cross-spectrum | StemGNN/graph | forecasting | core/head |
| D12 | best detector | best PARCNet subset | forecasting | benchmark |
| D13 | best detector branch | project EURUSD/ETH | forecasting | project core |
| D14 | detector events | rush detection | hazard | agent-multi |
| D15 | detector embeddings | SAC state | RL | frozen SAC protocol |

---

# 86. Recommended implementation order

1. matched-filter synthetic oracle;
2. generalized matched filter under colored noise;
3. scale/lag template bank;
4. MiniRocket;
5. MultiRocket;
6. HYDRA;
7. learned shapelets;
8. Matrix Profile causal motif branch;
9. L-MAP feasibility study;
10. InceptionTime baseline;
11. STEP-06 complex/phase detector;
12. STEP-06 STFT/CWT detector;
13. graph/cross-spectrum detector;
14. public UCR/UEA sanity benchmark;
15. PARCNet benchmark subset;
16. all PARCNet 12 only after gates;
17. project predictor parallel detector branches;
18. rush detector integration;
19. event-token integration;
20. RL integration only after predictive evidence;
21. foundation/in-context models as late comparative layer.

---

# 87. Decision gates

## Gate 7A — Matched-filter correctness

Synthetic AWGN behavior matches theory.

## Gate 7B — Noise-aware detector correctness

Generalized MF improves controlled colored-noise detection.

## Gate 7C — Pattern-bank value

Variable-shape/scale tasks justify a detector bank.

## Gate 7D — Cheap baseline competence

MiniRocket/MultiRocket/HYDRA achieve expected public benchmark quality.

## Gate 7E — Representation-specific value

At least one STEP-06 representation gains from a specialized detector.

## Gate 7F — Parameter-matched gain

Benefit survives equal-capacity comparison.

## Gate 7G — Public forecasting transfer

Benefit survives PARCNet/public datasets.

## Gate 7H — Financial validation

Benefit survives project validation split.

## Gate 7I — Rare-event calibration

Rush/event detector improves AP/calibration/lead time.

## Gate 7J — Final test

Only frozen preselected configurations reach test.

---

# 88. Project-specific detector registry proposal

The detector layer should be plugin-based, conceptually analogous to current predictor plugins.

Possible logical names:

- `matched_filter`
- `generalized_matched_filter`
- `template_bank`
- `shapelet_bank`
- `minirocket`
- `multirocket`
- `hydra`
- `matrix_profile`
- `lmap`
- `inception_time`
- `phase_match`
- `tf_conv`
- `spectral_graph`
- `selfsupervised_encoder`

This is a **contract recommendation**, not a request to implement all plugins immediately.

---

# 89. Interface requirements

Each detector should declare:

- `representation_type`;
- `input_shape`;
- `output_shape`;
- `causality_mode`;
- `fit_scope`;
- `requires_labels`;
- `requires_noise_model`;
- `supports_multivariate`;
- `supports_variable_length`;
- `template_count`;
- `receptive_field`;
- `calibration_method`;
- `uncertainty_method`;
- `random_seed`;
- `artifact_hash`.

---

# 90. Multi-branch core integration strategy

Recommended logical flow:

\[
X
\rightarrow
\begin{cases}
R_0(X)\rightarrow D_0\\
R_1(X)\rightarrow D_1\\
\vdots\\
R_K(X)\rightarrow D_K
\end{cases}
\]

then:

\[
Z_i
=
E_i(
R_i,
D_i
).
\]

Core:

\[
Z
=
C(
Z_0,\ldots,Z_K
).
\]

Heads:

\[
\hat Y_h
=
H_h(Z).
\]

The detector can be inside branch encoder or immediately before it.

---

# 91. No forced one-model-for-all-branches rule

Examples:

## Raw branch

TCN / Conv1D / HYDRA.

## Wavelet detail branch

small causal Conv1D.

## STFT branch

Conv2D.

## Complex spectral branch

real/imag spectral MLP/Conv.

## Cross-series graph branch

GNN.

## Event token branch

Transformer.

This heterogeneity is a feature, not an architectural failure, provided the core receives a stable standardized representation.

---

# 92. Standardized branch latent dimension

To prevent one branch dominating merely by width, each branch may project to:

\[
Z_i
\in
\mathbb R^{d}
\]

with fixed:

\[
d.
\]

Alternative:

\[
d_i
\propto
\text{validated information contribution}.
\]

Begin with equal \(d\), then test adaptive capacity allocation later.

---

# 93. Core fusion candidates

Reuse current project fusion first.

Later candidates:

- concatenation + Conv1D;
- gated fusion;
- cross-attention;
- mixture-of-experts gate;
- Bayesian fusion;
- low-rank tensor fusion.

Do not change fusion during first detector ablation.

---

# 94. Mixture-of-experts extension

If different detector branches specialize by regime:

\[
Z
=
\sum_i
g_i(X)Z_i
\]

with:

\[
\sum_i g_i=1.
\]

Hypothesis:

> dynamic gating is useful only after stable branch specialization is demonstrated.

Do not use MoE to rescue weak branches prematurely.

---

# 95. Detector auxiliary losses

Possible multi-task training:

\[
L
=
L_{forecast}
+
\lambda_1 L_{pattern}
+
\lambda_2 L_{calibration}
+
\lambda_3 L_{diversity}.
\]

But initial STEP-07 experiments should isolate detector quality first.

---

# 96. Contrastive pattern learning

For learned motifs/embeddings:

- positive pairs = transformations that preserve desired motif semantics;
- negatives = distinct motifs/regimes.

Critical risk:

> augmentation may destroy financial temporal meaning.

Use STEP-06/TimesURL guidance and audit every augmentation.

---

# 97. Dynamic Time Warping (DTW) role

DTW is useful when motif speed/duration varies.

Template distance:

\[
d_{DTW}(x,s).
\]

Pros:

- warp invariance.

Cons:

- computational cost;
- may match economically different shapes too flexibly;
- standard DTW does not use noise covariance.

Use as interpretable control, not default detector.

---

# 98. Soft-DTW

Soft-DTW provides differentiable approximate alignment.

Potential learned template objective:

\[
L
=
softDTW(X,S).
\]

The predictor repository already includes a configurable soft-DTW-related loss in TCN code paths, so the project has conceptual infrastructure to explore differentiable alignment.

Still, target use and detector use are different and should remain separated.

---

# 99. Event onset versus whole-window classification

A classifier:

\[
f(X_{t-W+1:t})
\]

can detect whether a pattern exists in the window.

An onset detector must predict:

\[
P(
T_{\mathrm{event}}=t
\mid
X_{\leq t}
).
\]

These tasks need different labels and metrics.

For rush/event detection, prefer:

- hazard models;
- onset probability;
- time-to-event.

---

# 100. Bayesian uncertainty

Existing project Bayesian heads can remain downstream.

Detector uncertainty can be added using:

- MC dropout;
- bootstrap template variability;
- Bayesian shapelets;
- ensemble detector variance;
- posterior predictive probability.

Do not treat detector activation amplitude as uncertainty.

---

# 101. Calibration under regime shift

Evaluate calibration by:

- year;
- volatility regime;
- asset;
- session;
- event/non-event windows.

A detector whose average calibration is good but fails during high-volatility periods is not production-ready.

---

# 102. SNR-conditioned performance surface

Reuse STEP 03.

For detector \(D\):

\[
P
=
P(
D,
SNR,
\text{pattern variation}
).
\]

Construct surface:

\[
P_D(SNR,\delta_s)
\]

where:

\[
\delta_s
\]

quantifies template deformation.

This reveals where:

- classical matched filters dominate;
- flexible learned detectors become worthwhile.

---

# 103. Detector complexity frontier

For each detector:

\[
c_D
=
\{
\text{params},
\text{FLOPs},
\text{latency},
\text{memory}
\}.
\]

Performance:

\[
p_D.
\]

Compute Pareto frontier:

\[
\boxed{
c_D
\leftrightarrow
p_D
}.
\]

This is important because random-kernel methods can be extremely competitive at low training cost.

---

# 104. Final scientific acceptance criteria

STEP 07 is supported only if:

1. synthetic matched-filter tests reproduce theory;
2. leakage/causality audit passes;
3. public pattern benchmarks pass;
4. representation-specific detectors are compared fairly;
5. parameter-matched controls are included;
6. cheap baselines are not skipped;
7. detector outputs are calibrated where interpreted probabilistically;
8. recurring motifs are not conflated with target-relevant motifs;
9. public forecasting transfer exists;
10. financial validation exists;
11. final held-out test remains untouched until selection;
12. any RL utility claim is downstream of predictive detector evidence.

---

# 105. Possible scientifically valid outcomes

## Outcome A

Classical matched-filter-style detectors are highly competitive after proper denoising/whitening.

## Outcome B

Random convolution banks dominate learned deep detectors on limited data.

## Outcome C

Shapelets provide the best interpretability/performance compromise.

## Outcome D

2D time-frequency detectors add value only on selected signal families.

## Outcome E

Graph detectors are crucial for high-dimensional multivariate series but not low-dimensional FX features.

## Outcome F

Foundation encoders help only in low-label regimes.

## Outcome G

Existing TFT/TCN core already learns most pattern structure, making explicit detectors redundant.

## Outcome H

Detector branches allow a much smaller core with equal performance.

## Outcome I

No detector branch generalizes to financial OOS data.

All are valid results.

---

# 106. State-of-the-art synthesis

The state of the art makes one conclusion especially clear:

> There is no single universally superior “pattern recognition network” for all time-series geometries.

Strong families exploit different biases:

- matched filters: known pattern + noise model;
- shapelets: local discriminative subsequences;
- Matrix Profile/L-MAP: recurrence/motif structure;
- ROCKET-family/HYDRA: large efficient convolutional detector banks;
- HIVE-COTE: heterogeneous representation diversity;
- InceptionTime: learned multi-scale convolution;
- TS2Vec/TimesURL/MOMENT/Mantis: reusable representation learning;
- StemGNN/graph models: cross-series dependencies;
- TimeMixer++: integrated multi-scale/multi-resolution patterns.

This strongly supports the project's modular architecture.

---

# 107. Most important project-specific research hypothesis

The central STEP-07 hypothesis should be stated as:

\[
\boxed{
\text{A detector chosen to match the geometry and noise properties of each STEP-06 representation can expose pattern evidence that a finite downstream core learns more efficiently than from raw data alone.}
}
\]

The strongest evidence would be:

1. improved OOS performance;
2. under equal parameter/compute budgets;
3. with raw branch retained;
4. across public and financial datasets;
5. with stable detector semantics;
6. without test-driven selection.

---

# 108. Recommended immediate experiment

The highest-value first experiment after implementation infrastructure exists is:

## Synthetic

\[
\text{correlation}
\leftrightarrow
\text{MF}
\leftrightarrow
\text{GMF}
\leftrightarrow
\text{MiniRocket}
\leftrightarrow
\text{learned Conv}.
\]

Then:

## Public TSC

\[
\text{MiniRocket}
\leftrightarrow
\text{MultiRocket}
\leftrightarrow
\text{HYDRA}
\leftrightarrow
\text{shapelets}
\leftrightarrow
\text{InceptionTime}.
\]

Then:

## STEP-06 output

\[
\text{raw}
\leftrightarrow
\text{complex spectrum}
\leftrightarrow
\text{wavelet}
\]

with representation-appropriate detectors.

Only after these gates:

\[
\text{project multi-branch predictor}.
\]

---

# 109. References — IEEE style

[1] G. L. Turin, “An Introduction to Matched Filters,” *IRE Transactions on Information Theory*, vol. 6, no. 3, pp. 311–329, Jun. 1960, doi: 10.1109/TIT.1960.1057571. Available: https://doi.org/10.1109/TIT.1960.1057571

[2] J. Grabocka, N. Schilling, M. Wistuba, and L. Schmidt-Thieme, “Learning Time-Series Shapelets,” in *Proc. 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2014, pp. 392–401, doi: 10.1145/2623330.2623613. Available: https://doi.org/10.1145/2623330.2623613

[3] C.-C. M. Yeh *et al.*, “Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets,” in *2016 IEEE 16th International Conference on Data Mining (ICDM)*, 2016, doi: 10.1109/ICDM.2016.0179. Available: https://doi.org/10.1109/ICDM.2016.0179

[4] M. Lin, Y. Wang, X. Hong, and W. Li, “Learnable Matrix Profile for Motif Discovery on Multivariate Time Series,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 18, pp. 15252–15260, 2026, doi: 10.1609/aaai.v40i18.38550. Available: https://doi.org/10.1609/aaai.v40i18.38550

[5] A. Dempster, F. Petitjean, and G. I. Webb, “ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels,” *Data Mining and Knowledge Discovery*, 2020. Preprint: https://arxiv.org/abs/1910.13051

[6] A. Dempster, D. F. Schmidt, and G. I. Webb, “MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification,” in *Proc. 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2021, pp. 248–257, doi: 10.1145/3447548.3467231. Available: https://doi.org/10.1145/3447548.3467231

[7] C. W. Tan, A. Dempster, C. Bergmeir, and G. I. Webb, “MultiRocket: Multiple Pooling Operators and Transformations for Fast and Effective Time Series Classification,” *Data Mining and Knowledge Discovery*, vol. 36, pp. 1623–1646, 2022, doi: 10.1007/s10618-022-00844-1. Available: https://doi.org/10.1007/s10618-022-00844-1

[8] A. Dempster, D. F. Schmidt, and G. I. Webb, “HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification,” *Data Mining and Knowledge Discovery*, 2023, doi: 10.1007/s10618-023-00939-3. Available: https://doi.org/10.1007/s10618-023-00939-3

[9] M. Middlehurst, J. Large, M. Flynn, J. Lines, A. Bostrom, and A. Bagnall, “HIVE-COTE 2.0: A New Meta Ensemble for Time Series Classification,” *Machine Learning*, vol. 110, pp. 3211–3243, 2021, doi: 10.1007/s10994-021-06057-9. Available: https://doi.org/10.1007/s10994-021-06057-9

[10] H. I. Fawaz *et al.*, “InceptionTime: Finding AlexNet for Time Series Classification,” *Data Mining and Knowledge Discovery*, vol. 34, pp. 1936–1962, 2020, doi: 10.1007/s10618-020-00710-y. Available: https://doi.org/10.1007/s10618-020-00710-y

[11] S. Bai, J. Z. Kolter, and V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” arXiv:1803.01271, 2018. Available: https://arxiv.org/abs/1803.01271

[12] Z. Yue *et al.*, “TS2Vec: Towards Universal Representation of Time Series,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 8, pp. 8980–8987, 2022, doi: 10.1609/aaai.v36i8.20881. Available: https://doi.org/10.1609/aaai.v36i8.20881

[13] J. Liu and S. Chen, “TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 12, pp. 13918–13926, 2024, doi: 10.1609/aaai.v38i12.29299. Available: https://doi.org/10.1609/aaai.v38i12.29299

[14] M. Goswami *et al.*, “MOMENT: A Family of Open Time-series Foundation Models,” in *Proceedings of the 41st International Conference on Machine Learning*, PMLR vol. 235, pp. 16115–16152, 2024. Available: https://proceedings.mlr.press/v235/goswami24a.html

[15] V. Feofanov *et al.*, “Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification,” arXiv:2502.15637, 2025; subsequent project materials report ICML 2026 publication status. Available: https://arxiv.org/abs/2502.15637

[16] D. Cao *et al.*, “Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. Available: https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html

[17] Z. Wu *et al.*, “Graph WaveNet for Deep Spatial-Temporal Graph Modeling,” in *Proceedings of IJCAI*, 2019. Available: https://www.ijcai.org/proceedings/2019/264

[18] Z. Wu, S. Pan, G. Long, J. Jiang, X. Chang, and C. Zhang, “Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks,” in *Proc. ACM SIGKDD*, 2020, doi: 10.1145/3394486.3403118. Available: https://doi.org/10.1145/3394486.3403118

[19] S. Wang *et al.*, “TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis,” in *International Conference on Learning Representations*, 2025. Available: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html

[20] Z. Li, H. Chai, Z. Song, S. Liu, and X. Liu, “PARCNet: Phase-Aware Residual Correction Network for Efficient Multivariate Time Series Forecasting,” *Knowledge-Based Systems*, vol. 348, art. 116370, 2026, doi: 10.1016/j.knosys.2026.116370. Available: https://doi.org/10.1016/j.knosys.2026.116370

[21] J. Küken, S. B. Hoo, M. Mráz, F. Hutter, and L. Purucker, “TimEE: End-to-end Time Series Classification via In-Context Learning,” arXiv:2607.07500, 2026. **Preprint/current research comparator.** Available: https://arxiv.org/abs/2607.07500

[22] F. M. O'Rourke, A. Trisovic, and D. Bertsimas, “RocketPFN: Accurate Time Series Classification via In-Context Learning,” arXiv:2606.21786, 2026. **Preprint/current research comparator.** Available: https://arxiv.org/abs/2606.21786

[23] C.-C. M. Yeh *et al.*, “TiCT: A Synthetically Pre-Trained Foundation Model for Time Series Classification,” arXiv:2511.19694, 2025. **Preprint/current research comparator.** Available: https://arxiv.org/abs/2511.19694

[24] A. Bagnall *et al.*, “The Great Multivariate Time Series Classification Bake Off: A Review and Experimental Evaluation of Recent Algorithmic Advances,” *Data Mining and Knowledge Discovery*, 2021, doi: 10.1007/s10618-020-00727-3. Available: https://doi.org/10.1007/s10618-020-00727-3

---

# 110. Verified code/resources for reproducibility

## PARCNet

Official repository:

https://github.com/9527dandelion/PARCNet

Official scripts directly expose the 12 dataset configurations listed in this document.

## Project Predictor

Repository:

https://github.com/harveybc/predictor

Relevant paths:

- `examples/config/phase_1b_binary/`
- `predictor_plugins/predictor_plugin_tcn.py`
- `predictor_plugins/predictor_plugin_composite.py`
- binary/direction predictor plugin families.

## Project Agent Multi

Repository:

https://github.com/harveybc/agent-multi

Relevant areas:

- `docs/work_plan/04_MODELS_POLICIES_AND_TRAINING.md`
- `tools/project3_event_token_transformer.py`
- SAC actor-critic plugins and Project-3 evidence infrastructure.

---

# 111. Final status

**STEP 07 is now theoretically grounded, representation-aware, aligned with the project's current modular architecture, and organized as a falsifiable experimental protocol.**

The intended conceptual chain is now:

\[
\boxed{
\text{sampling}
\rightarrow
\text{noise/SNR}
\rightarrow
\text{denoising}
\rightarrow
\text{quantization}
\rightarrow
\text{source/context structure}
\rightarrow
\text{amplitude/frequency/phase/time-frequency representations}
\rightarrow
\text{matched pattern detection}
\rightarrow
\text{core fusion}
\rightarrow
\text{predictive heads/policy}
}
\]

The principal research discipline is:

> **Do not choose a model because it is fashionable; choose and falsify a detector because its inductive bias matches the mathematical object produced by the preceding representation stage.**

This is the STEP-07 equivalent of receiver design in communications.
