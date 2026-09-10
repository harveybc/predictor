# STEP 06 — Amplitude, Frequency, Phase and Time–Frequency Representations for Time-Series Machine Learning

**Status:** Research protocol — final draft for agent audit and repository integration  
**Date:** 2026-09-05  
**Scope:** Systematic exposure of complementary signal representations before and inside modular multi-branch forecasting models, with emphasis on amplitude, spectral magnitude, phase, complex spectra, instantaneous amplitude/frequency, multiscale and time–frequency structure, and cross-series spectral dependence.  
**Primary application:** multivariate financial time series, 1 h and 4 h periodicities, 4-year training + 1-year validation + 1-year test splits.  
**Prerequisites:**  
- STEP 01 — Sampling / Nyquist  
- STEP 02–03 — Noise / SNR estimation and denoising  
- STEP 04 — Quantization / non-uniform quantization / companding  
- STEP 05 — Source coding / entropy / context / innovation / surprisal  

**Next planned communications analogue:** STEP 07 — matched filtering, template detection, convolutional receptive fields and branch-specific detection.

---

# 0. Executive summary

The central question of STEP 06 is:

> **Does a forecasting model generalize better when complementary mathematical views of the same underlying time series are made explicitly available, rather than requiring the model to infer all such views from raw samples alone?**

The communications analogy is not that financial data should literally be “modulated” like a telephone waveform. The rigorous analogue is that a signal admits multiple complementary coordinates or degrees of freedom:

\[
x_t
\longleftrightarrow
\left\{
\text{amplitude},
\text{frequency content},
\text{phase},
\text{time-localized frequency},
\text{multiscale structure},
\text{cross-series spectral relationships}
\right\}.
\]

A raw time-domain sample sequence is already mathematically complete in the invertible-transform sense, but a finite neural model does not necessarily learn every useful transform equally efficiently.

Therefore STEP 06 tests:

\[
\boxed{
\text{information-equivalent representation}
\neq
\text{learning-equivalent representation}
}
\]

and asks whether explicit transforms improve:

- sample efficiency;
- robustness;
- long-horizon forecasting;
- phase/timing accuracy;
- cross-feature dependency modeling;
- performance under finite model capacity.

This is already an active research area. Fourier-domain models such as FEDformer, FreTS and FITS, multi-period models such as TimesNet, multiscale models such as TimeMixer, wavelet methods such as WPMixer/WaveletMixer and WaveToken, and recent phase-aware architectures provide strong prior art. The project should therefore **benchmark and integrate**, not reinvent these transformations.

The strongest project-specific opportunity is a controlled, branch-wise comparison in which **amplitude, phase, frequency, time-frequency and cross-spectral representations are isolated experimentally**, with causal train-only transformations and falsifiable ablations.

---

# 1. Correct interpretation of the communications analogy

In a communication system, modulation exploits properties of a carrier such as:

- amplitude;
- phase;
- frequency;
- symbol timing.

In the present project, there is no requirement to construct an artificial carrier.

Instead, the useful analogy is:

> A measured time series can be represented in mathematical coordinate systems that make different structures explicit.

For a real discrete sequence:

\[
x[n],
\]

we may use:

\[
x[n]
\]

directly, or transform it to:

\[
X[k]
=
\sum_{n=0}^{N-1}
x[n]
e^{-j2\pi kn/N}.
\]

The DFT is invertible, so no information is lost if the complete complex spectrum is retained.

However, the coordinate representation changes radically:

- time domain localizes events in time;
- Fourier domain localizes energy in frequency;
- phase encodes timing/alignment information;
- wavelets localize jointly in time and scale;
- STFT localizes approximately in time and frequency;
- Hilbert-based representations expose instantaneous amplitude and phase under suitable conditions;
- cross-spectral measures expose frequency-dependent inter-series relationships.

This is the formal basis for STEP 06.

---

# 2. Core methodological principle

If transform:

\[
T
\]

is invertible, then mathematically:

\[
I(X;T(X)) = H(X)
\]

under an ideal deterministic/discrete information-theoretic interpretation where no numerical loss occurs.

Nevertheless, a finite model class:

\[
\mathcal M
\]

may satisfy:

\[
\min_{M\in\mathcal M}
L(M(T(X)),Y)
<
\min_{M\in\mathcal M}
L(M(X),Y).
\]

That is:

> an invertible representation can improve learnability without creating new information.

This distinction is fundamental.

STEP 06 is therefore about **inductive bias and representational accessibility**, not magical creation of information.

---

# 3. Fourier representation

For a length-\(N\) real-valued sequence:

\[
x[n],
\quad
n=0,\ldots,N-1,
\]

the DFT is:

\[
X[k]
=
\sum_{n=0}^{N-1}
x[n]
e^{-j2\pi kn/N}.
\]

Write:

\[
X[k]
=
A[k]e^{j\phi[k]},
\]

where:

\[
A[k]=|X[k]|
\]

is magnitude and:

\[
\phi[k]
=
\arg X[k]
\]

is phase.

Equivalently:

\[
X[k]
=
R[k]+jI[k].
\]

Thus complete representations include:

## Cartesian

\[
(R[k],I[k]).
\]

## Polar

\[
(A[k],\phi[k]).
\]

Both contain the complete complex Fourier information.

---

# 4. Magnitude / power spectrum

The magnitude spectrum:

\[
A[k]
=
|X[k]|
\]

describes the strength of components at each Fourier frequency.

Power:

\[
P[k]
=
|X[k]|^2.
\]

Possible model inputs:

\[
A[k],
\]

\[
\log(A[k]+\epsilon),
\]

\[
P[k],
\]

\[
\log(P[k]+\epsilon).
\]

The logarithmic representation can stabilize dynamic range but changes the numerical geometry and must be fitted/scaled causally.

Magnitude alone is **not** generally sufficient to reconstruct the original sequence.

---

# 5. Phase is not a minor residual variable

Oppenheim and Lim's classic review established that Fourier phase can carry essential signal structure and, under specific conditions, phase alone can determine a finite-length signal up to scale.

Therefore STEP 06 must not repeat a common engineering shortcut:

\[
\text{FFT}
\rightarrow
|X|
\rightarrow
\text{discard phase}
\]

without an ablation.

For forecasting, phase can encode:

- relative temporal location;
- alignment of oscillatory components;
- turning-point timing;
- lead/lag structure;
- waveform shape.

Recent time-series forecasting work explicitly revisits phase-aware modeling, including complex-spectrum models and Hilbert amplitude-phase architectures.

---

# 6. Phase representation problem: circular topology

Raw phase:

\[
\phi\in(-\pi,\pi]
\]

has a discontinuity:

\[
-\pi
\equiv
+\pi.
\]

Feeding raw phase directly creates an artificial numerical distance:

\[
|\pi-\epsilon - (-\pi+\epsilon)|
\approx 2\pi
\]

even though the two angles are nearly identical.

Therefore the preferred representation is:

\[
\boxed{
(\cos\phi,\sin\phi)
}
\]

or the complex Cartesian representation:

\[
(\Re X,\Im X).
\]

Raw unwrapped phase may be used only where unwrapping is well-defined and causal.

---

# 7. Global Fourier representation versus local nonstationarity

A single FFT over a trailing window assumes that spectral content inside that window is represented globally.

For a nonstationary process:

\[
x_t,
\]

spectral structure may evolve over time:

\[
P(f,t).
\]

Financial data frequently exhibit:

- volatility regimes;
- temporary cycles;
- structural breaks;
- event shocks;
- variable-frequency oscillations.

Therefore global FFT features must be compared against time-localized alternatives.

---

# 8. Windowing and spectral leakage

Finite-window Fourier analysis implicitly multiplies the underlying sequence by a window:

\[
x_w[n]
=
x[n]w[n].
\]

In frequency:

\[
X_w
=
X*W.
\]

Thus sharp truncation causes spectral leakage.

Candidate windows:

- rectangular;
- Hann;
- Hamming;
- Blackman;
- DPSS/Slepian.

Window choice must be controlled because it changes:

- frequency resolution;
- leakage;
- amplitude estimation;
- effective sample weighting.

Do not interpret every FFT peak as a true oscillation.

---

# 9. Welch spectral estimation

A classical robust PSD baseline is Welch's method:

1. divide data into overlapping windows;
2. window each segment;
3. compute periodograms;
4. average them.

This reduces variance at the cost of spectral resolution.

STEP 06 should include Welch as a simple PSD baseline before introducing more complex methods.

---

# 10. Multitaper spectral estimation

Thomson's multitaper method uses multiple orthogonal DPSS tapers.

For tapers:

\[
v_k[n],
\]

compute:

\[
X_k(f)
=
\sum_n
x[n]v_k[n]e^{-j2\pi fn}
\]

and combine spectral estimates.

Advantages:

- controlled bias/variance tradeoff;
- reduced leakage;
- multiple approximately independent spectral views;
- useful small-sample theory;
- coherent/cross-spectral extensions.

Because the project already has multitaper infrastructure, this should be treated as an existing reusable representation rather than reimplemented from scratch.

---

# 11. Short-Time Fourier Transform

The STFT is:

\[
X(\tau,\omega)
=
\sum_n
x[n]
w[n-\tau]
e^{-j\omega n}.
\]

It yields a time-frequency representation.

Spectrogram:

\[
S(\tau,\omega)
=
|X(\tau,\omega)|^2.
\]

The STFT provides local frequency structure, but its fixed window creates a resolution tradeoff:

- short window:
  - good time resolution;
  - poor frequency resolution;

- long window:
  - good frequency resolution;
  - poor time resolution.

This tradeoff is fundamental and cannot be tuned away completely.

---

# 12. Causal STFT for forecasting

For forecast time \(t\), use only:

\[
x_{t-W+1:t}.
\]

A centered window:

\[
x_{t-W/2:t+W/2}
\]

is invalid in live forecasting because it consumes future samples.

Therefore:

\[
\boxed{
\text{trailing-window STFT only}
}
\]

for production-equivalent experiments.

Padding must not inject future observations.

---

# 13. Wavelet representations

Wavelets provide variable time-frequency resolution.

A continuous wavelet transform can be written:

\[
W_x(a,b)
=
\frac{1}{\sqrt{|a|}}
\int
x(t)
\psi^*
\left(
\frac{t-b}{a}
\right)
dt.
\]

where:

- \(a\): scale;
- \(b\): time position.

Compared with fixed-window STFT:

- high frequencies can receive fine temporal resolution;
- low frequencies can receive coarse temporal resolution.

This is often better aligned with multi-scale time-series behavior.

---

# 14. DWT / multiresolution decomposition

Discrete wavelet decomposition generates:

\[
X
\rightarrow
\{
A_J,
D_J,
D_{J-1},
\ldots,
D_1
\}
\]

where:

- \(A_J\): approximation/coarse component;
- \(D_j\): detail at scale \(j\).

This naturally supports the project's branch architecture:

\[
B_j=D_j.
\]

Possible architecture:

\[
D_1\rightarrow E_1,
\quad
D_2\rightarrow E_2,
\quad
\ldots,
\quad
A_J\rightarrow E_A.
\]

Then:

\[
Z_{core}
=
C(E_A,E_1,\ldots,E_J).
\]

---

# 15. Boundary effects in wavelets are a major leakage risk

Many standard wavelet implementations use:

- symmetric extension;
- reflection;
- periodic extension.

Near the right boundary, these methods may implicitly use samples that would not exist at live forecast time.

Therefore every wavelet transform must be audited for:

- extension mode;
- filter support;
- output alignment;
- causal availability.

A transform is not production-valid merely because it was computed separately inside training.

---

# 16. Wavelet coherence for multivariate signals

For two series:

\[
X(t),Y(t),
\]

cross-wavelet transform:

\[
W_{XY}(a,b)
=
W_X(a,b)W_Y^*(a,b).
\]

Wavelet coherence is a localized measure of association in time-scale space.

Phase difference:

\[
\Delta\phi_{XY}(a,b)
=
\arg
W_{XY}(a,b)
\]

can expose scale-dependent lead/lag behavior.

This is substantially richer than a single Pearson correlation.

However:

> coherence is not causality.

The project must label it as dependence/alignment.

---

# 17. Fourier cross-spectrum and coherence

For two stationary processes:

\[
X_t,Y_t,
\]

cross-spectrum:

\[
S_{XY}(f).
\]

Magnitude-squared coherence:

\[
C_{XY}(f)
=
\frac{
|S_{XY}(f)|^2
}{
S_{XX}(f)S_{YY}(f)
}.
\]

Range:

\[
0\leq C_{XY}(f)\leq1.
\]

Cross-spectral phase:

\[
\phi_{XY}(f)
=
\arg S_{XY}(f).
\]

If a meaningful approximately linear phase relation exists:

\[
\phi_{XY}(f)
\approx
-2\pi f\tau,
\]

then:

\[
\tau
\]

can be interpreted as a delay under appropriate assumptions.

This is attractive for financial exogenous features where relationships may be frequency-specific.

---

# 18. Multivariate spectral matrix

For:

\[
\mathbf X_t
=
(X_{1,t},\ldots,X_{m,t}),
\]

define spectral density matrix:

\[
\mathbf S(f)
=
[S_{ij}(f)].
\]

This contains frequency-dependent:

- auto-spectra;
- cross-spectra;
- coherence;
- phase relationships.

The project's multiple asset and fundamental series make this representation a high-priority candidate.

A learned branch can consume:

\[
\mathbf S(f)
\]

or compressed summaries thereof.

Full:

\[
m\times m\times F
\]

representations may be computationally expensive, so sparsity/top-\(K\) selection should be investigated.

---

# 19. Analytic signal and Hilbert transform

For real signal:

\[
x(t),
\]

define analytic signal:

\[
z(t)
=
x(t)
+
j\mathcal H\{x(t)\}
\]

where:

\[
\mathcal H
\]

is the Hilbert transform.

Then:

\[
z(t)
=
A(t)e^{j\phi(t)}.
\]

Instantaneous amplitude:

\[
A(t)
=
|z(t)|
\]

and phase:

\[
\phi(t)
=
\arg z(t).
\]

Instantaneous angular frequency:

\[
\omega_i(t)
=
\frac{d\phi(t)}{dt}.
\]

This gives a time-aligned amplitude-phase-frequency representation.

---

# 20. Critical Hilbert-transform limitation

Instantaneous frequency is not automatically meaningful for an arbitrary multicomponent signal.

Boashash's classical analysis emphasizes the importance of the monocomponent/multicomponent distinction.

Therefore:

\[
x(t)
=
\sum_k x_k(t)
\]

should often be decomposed first into interpretable modes/bands:

\[
x_k(t)
\]

before computing:

\[
A_k(t),
\phi_k(t),
f_k(t).
\]

Do not calculate a Hilbert instantaneous frequency on arbitrary noisy financial price levels and assume physical meaning.

---

# 21. Hilbert–Huang transform / EMD

Huang et al. proposed:

1. Empirical Mode Decomposition;
2. intrinsic mode functions;
3. Hilbert spectral analysis.

Represent:

\[
x(t)
=
\sum_{k=1}^{K}
c_k(t)+r(t)
\]

where:

\[
c_k(t)
\]

are IMFs.

For each IMF:

\[
z_k(t)
=
c_k(t)
+
j\mathcal H\{c_k(t)\}
\]

and derive:

\[
A_k(t),
\phi_k(t),
f_k(t).
\]

This creates an adaptive:

\[
\text{time}
\times
\text{frequency}
\times
\text{energy}
\]

representation.

Risks:

- mode mixing;
- endpoint effects;
- algorithmic instability;
- causality complications.

Therefore HHT is an experimental branch, not a default preprocessing step.

---

# 22. Synchrosqueezed transforms

Synchrosqueezing sharpens time-frequency representations by reallocating wavelet energy using local instantaneous-frequency estimates.

Daubechies, Lu and Wu provided a mathematically analyzable alternative inspired by EMD.

Potential benefit:

- sharper separation of time-varying oscillatory modes.

Potential risk:

- increased complexity;
- parameter sensitivity;
- questionable benefit over ordinary wavelets for forecasting.

Only promote if simpler representations fail or leave unresolved mode overlap.

---

# 23. Wavelet scattering transform

Wavelet scattering cascades:

\[
\text{wavelet convolution}
\rightarrow
|\cdot|
\rightarrow
\text{wavelet convolution}
\rightarrow
|\cdot|
\rightarrow
\text{averaging}.
\]

It provides stable multiscale descriptors and can capture higher-order structure beyond power spectra.

This is highly relevant to the project because it provides a mathematically structured alternative to learning all early convolutional filters from scratch.

Potential comparison:

\[
\text{raw}
\rightarrow
\text{Conv1D}
\]

versus:

\[
\text{wavelet scattering}
\rightarrow
\text{smaller learned network}.
\]

---

# 24. Phase-randomized surrogate controls

A particularly powerful STEP-06 falsification tool is surrogate data.

Given:

\[
X[k]
=
A[k]e^{j\phi[k]},
\]

construct:

\[
\tilde X[k]
=
A[k]e^{j\tilde\phi[k]}
\]

where phases are randomized while conjugate symmetry is maintained.

Inverse FFT produces:

\[
\tilde x[n].
\]

This approximately preserves the power spectrum but destroys specific phase organization.

Theiler et al. established surrogate-data methodology for testing nonlinear temporal structure.

Schreiber and Schmitz later developed improved amplitude-adjusted/iterative surrogates that better preserve both autocorrelation structure and marginal distribution.

This allows a direct test:

> How much predictive behavior remains if spectral magnitude is preserved but phase organization is destroyed?

This is one of the strongest experiments in STEP 06.

---

# 25. Multivariate surrogate controls

For multivariate data, independently randomizing every phase destroys cross-series relationships.

Prichard and Theiler developed multivariate phase-randomized surrogates designed to preserve linear auto- and cross-correlation structure.

Therefore STEP 06 can distinguish:

1. univariate spectral structure;
2. cross-series linear spectral relationships;
3. higher-order/nonlinear structure.

This should be used carefully as a statistical control rather than data augmentation by default.

---

# 26. State of the art — Fourier-domain forecasting

Several modern forecasting architectures already exploit the frequency domain.

---

# 27. FEDformer

FEDformer combines:

- decomposition;
- Transformer modeling;
- frequency-domain enhanced blocks.

The original ICML 2022 work explicitly exploits sparse representations in Fourier/wavelet bases.

Project action:

- do not create a generic “Fourier Transformer” without benchmarking FEDformer ideas;
- use FEDformer as frequency-domain architecture prior art.

---

# 28. FreTS

FreTS, NeurIPS 2023, performs MLP learning in the frequency domain and explicitly processes:

\[
\Re X
\]

and:

\[
\Im X.
\]

This is directly relevant because it preserves the complete complex spectrum and thus implicitly both amplitude and phase.

Project action:

> Use FreTS as a key baseline for testing whether explicit complex-domain learning improves over raw Conv/LSTM processing.

---

# 29. FITS

FITS, ICLR 2024, operates directly in the complex frequency domain using interpolation and reports competitive performance with a very small parameter count.

This is especially relevant to the user's hypothesis that a good representation may reduce the network complexity required to learn the data.

Experimental question:

\[
\text{better representation}
\stackrel{?}{\Rightarrow}
\text{smaller model for same performance}.
\]

---

# 30. TimesNet

TimesNet discovers dominant periods and transforms 1D series into 2D structures representing:

- intraperiod variation;
- interperiod variation.

This is an important warning:

> frequency information does not have to be presented as raw FFT coefficients.

Period-derived geometric restructuring is also established prior art.

---

# 31. TimeMixer

TimeMixer, ICLR 2024, explicitly models multiple temporal scales and separates fine-scale/macro-scale variation.

This overlaps with the project's intended multiscale branches.

Project action:

- use TimeMixer as a multiscale baseline;
- do not assume wavelets are the only legitimate multiscale representation.

---

# 32. WPMixer and WaveletMixer

AAAI 2025 includes wavelet-based multi-resolution forecasting methods such as WPMixer and WaveletMixer.

These combine:

- wavelet decomposition;
- multiresolution learning;
- efficient MLP/mixer-style architectures.

Project action:

> benchmark against these before claiming novelty for “wavelet branches + neural mixing.”

---

# 33. WaveToken

WaveToken, ICML 2025, uses:

\[
\text{scaling}
\rightarrow
\text{wavelet decomposition}
\rightarrow
\text{thresholding}
\rightarrow
\text{quantization}
\rightarrow
\text{autoregressive forecasting}.
\]

This directly spans STEP 03, STEP 04, STEP 05 and STEP 06.

It is one of the most important prior-art systems for the overall project.

Project differentiation should therefore focus on controlled decomposition across multiple signal dimensions, SNR-aware decisions and multi-branch financial integration.

---

# 34. FreDF

FreDF, ICLR 2025, modifies the learning objective by forecasting in the frequency domain to address correlation among future labels.

This matters because frequency-domain representation may be useful not only for inputs but also for:

\[
Y_{t+1:t+H}.
\]

A later extension should test:

\[
\text{time-domain target loss}
\]

versus:

\[
\text{frequency-domain target loss}
\]

or a joint loss.

---

# 35. TimeKAN

TimeKAN, ICLR 2025, explicitly decomposes frequency components and learns different frequency bands with specialized blocks.

This independently supports the project's branch principle:

\[
\boxed{
\text{different frequency bands may deserve different model capacity}
}
\]

rather than one uniform network for the whole spectrum.

---

# 36. Phase-aware forecasting is now explicit current prior art

Recent 2026 work makes phase a direct modeling target.

PARCNet uses Hilbert analytic-signal amplitude and phase descriptors in multivariate forecasting.

Other recent complex-frequency-domain approaches explicitly preserve or model real/imaginary spectral components.

Therefore:

> **“Use phase explicitly in time-series forecasting” is no longer a novel claim by itself.**

The project-specific research question is instead:

\[
\boxed{
\text{which phase representation}
+
\text{which scale}
+
\text{which branch}
+
\text{which SNR regime}
}
\]

actually improves the project's targets.

---

# 37. Proposed representation families

STEP 06 should compare the following families independently.

## R0 — Raw time domain

\[
X_t.
\]

## R1 — Fourier magnitude

\[
|FFT(X)|.
\]

## R2 — Fourier phase

\[
(\sin\phi,\cos\phi).
\]

## R3 — Full complex Fourier

\[
(\Re X_f,\Im X_f).
\]

## R4 — Magnitude + phase

\[
(|X_f|,\sin\phi,\cos\phi).
\]

## R5 — PSD / multitaper PSD

\[
P(f).
\]

## R6 — STFT complex

\[
(\Re X(t,f),\Im X(t,f)).
\]

## R7 — STFT magnitude

\[
|X(t,f)|.
\]

## R8 — DWT coefficients

\[
(A_J,D_J,\ldots,D_1).
\]

## R9 — CWT scalogram

\[
|W_x(a,b)|.
\]

## R10 — Complex wavelet

\[
(\Re W,\Im W)
\]

or magnitude/phase.

## R11 — Hilbert amplitude/phase

\[
(A_t,\sin\phi_t,\cos\phi_t).
\]

## R12 — IMF/Hilbert modes

\[
(A_k,f_k,\phi_k).
\]

## R13 — Cross-spectrum/coherence

\[
C_{ij}(f),\phi_{ij}(f).
\]

## R14 — Wavelet coherence

\[
C_{ij}(a,t),\Delta\phi_{ij}(a,t).
\]

## R15 — Wavelet scattering

\[
S(X).
\]

---

# 38. Mandatory amplitude/phase ablation

For a Fourier branch, use identical downstream capacity for:

## A. Magnitude only

\[
A.
\]

## B. Phase only

\[
(\sin\phi,\cos\phi).
\]

## C. Real/imaginary

\[
(R,I).
\]

## D. Magnitude + phase

\[
(A,\sin\phi,\cos\phi).
\]

## E. Raw

\[
X.
\]

## F. Raw + magnitude

\[
[X,A].
\]

## G. Raw + phase

\[
[X,\sin\phi,\cos\phi].
\]

## H. Raw + full complex

\[
[X,R,I].
\]

This isolates exactly which Fourier information is useful.

---

# 39. Phase-destruction experiment

Take training/validation data and generate surrogates that preserve approximately:

\[
|X[k]|
\]

while altering:

\[
\phi[k].
\]

Train/evaluate equivalent models.

Compare:

\[
P(X)
\]

against:

\[
P(X_{\text{phase-randomized}}).
\]

If performance collapses while power spectra remain matched, there is evidence that phase organization carries useful temporal structure.

This is a much stronger statement than merely observing that a phase branch helps.

---

# 40. Magnitude-destruction control

Construct carefully defined experimental controls where phase is preserved but spectral magnitudes are modified or normalized.

Examples:

\[
X_{\text{phase-only}}[k]
=
e^{j\phi[k]}.
\]

After inverse transform and normalization, compare predictive performance.

This tests the relative contribution of magnitude versus phase.

Do not interpret phase-only reconstruction scale directly without normalization.

---

# 41. Causal rolling-spectrum experiment

At each time:

\[
t,
\]

construct trailing window:

\[
W_t
=
[x_{t-L+1},\ldots,x_t].
\]

Compute:

\[
FFT(W_t).
\]

Thus features:

\[
F_t
=
T(W_t).
\]

No future leakage occurs if:

\[
W_t\subseteq(-\infty,t].
\]

The same principle applies to:

- STFT;
- multitaper;
- wavelets;
- Hilbert modes.

Every implementation must document exact effective support.

---

# 42. Window-length sweep

For each representation:

\[
L
\in
\{L_1,\ldots,L_K\}.
\]

Because:

\[
\Delta f
\approx
\frac{f_s}{L},
\]

longer windows give finer frequency spacing.

But longer windows:

- reduce locality;
- cross more regimes;
- increase model input size.

Therefore:

\[
P=P(L)
\]

must be measured.

A single arbitrary FFT window is unacceptable.

---

# 43. Frequency-bin selection

Full FFT branches may be high-dimensional.

Candidate reductions:

1. fixed low-frequency subset;
2. top-\(K\) magnitude bins;
3. frequency bands;
4. learned frequency masks;
5. SNR-filtered bins;
6. multitaper-significant spectral lines.

All selection criteria must be training-only.

A per-window top-\(K\) selection is causal if it uses only that trailing window, but the changing bin identity must be encoded explicitly.

---

# 44. SNR-aware spectral selection

STEP 03 gives estimated signal/noise structure.

For spectral bin:

\[
f_k,
\]

estimate:

\[
\widehat{\mathrm{SNR}}(f_k).
\]

Candidate rule:

\[
\mathcal F_{keep}
=
\{
f_k:
\widehat{\mathrm{SNR}}(f_k)>\tau
\}.
\]

Hypothesis:

> discarding bins dominated by estimated noise may improve finite-capacity forecasting.

Counter-hypothesis:

> weak individually noisy bins may jointly contain predictive information.

Therefore raw/full-spectrum control remains mandatory.

---

# 45. Frequency-band branch specialization

Partition:

\[
[0,f_N]
\]

into:

\[
B_1,\ldots,B_K.
\]

For each band:

\[
X^{(k)}.
\]

Then:

\[
Z_k
=
E_k(X^{(k)}).
\]

Core:

\[
Z
=
C(Z_1,\ldots,Z_K).
\]

This maps directly onto the project's modular multi-branch design.

Compare:

- equal encoder architecture per band;
- capacity proportional to energy;
- capacity proportional to SNR;
- capacity selected by validation.

---

# 46. Wavelet branch specialization

Wavelet scales already form natural branches:

\[
D_1,D_2,\ldots,D_J,A_J.
\]

Hypothesis:

> specialized models per scale outperform one model that must jointly infer all scales.

Possible branch specialization:

- high-frequency detail:
  - Conv1D / small TCN;

- intermediate oscillatory scales:
  - Conv1D + BiLSTM;

- low-frequency approximation:
  - LSTM / trend model / attention.

This is an architectural hypothesis and must be compared against an equal-parameter shared model.

---

# 47. Equal-capacity fairness

If branch decomposition increases parameter count, performance gain cannot be attributed uniquely to representation.

Therefore compare under two conditions:

## Unconstrained

natural architecture.

## Parameter-matched

\[
\#\theta_{\text{multi-branch}}
\approx
\#\theta_{\text{raw baseline}}.
\]

The parameter-matched experiment is mandatory for causal attribution.

---

# 48. Cross-spectral branch

For selected feature pairs/groups, compute:

\[
C_{ij}(f)
\]

and:

\[
\phi_{ij}(f).
\]

Candidate summaries:

- mean coherence per band;
- max coherence;
- frequency of max coherence;
- phase at max coherence;
- bandwise phase lag;
- rolling coherence change.

Feed:

\[
B_{cross}
=
[
C_{ij,b},
\sin\phi_{ij,b},
\cos\phi_{ij,b}
].
\]

Avoid all \(O(m^2)\) pairs initially.

---

# 49. Pair selection for multivariate spectral features

Candidate training-only strategies:

- prior economic grouping;
- top absolute correlation;
- top mutual information;
- top conditional code gain from STEP 05;
- top coherence;
- causal-screening candidates from separate protocol.

No pair selection may use test performance.

---

# 50. Phase slope and delay hypothesis

For frequency band where phase is approximately linear:

\[
\phi_{ij}(f)
\approx
-2\pi f\tau_{ij}.
\]

Estimate:

\[
\hat\tau_{ij}.
\]

Potential feature:

\[
B_{\tau}
=
\hat\tau_{ij}(t).
\]

Hypothesis:

> dynamic spectral lead/lag may be more informative than zero-lag correlation.

Risk:

- unstable phase;
- wrap ambiguity;
- low-coherence regions;
- spurious delay estimates.

Only estimate phase delay where coherence exceeds a predeclared threshold.

---

# 51. Time-frequency image-like models

STFT/CWT produce 2D tensors:

\[
T
\times
F.
\]

Possible encoders:

- Conv2D;
- lightweight vision Transformer;
- separable convolutions;
- axial attention.

But the project should first compare simple models.

A large image model can hide the value of the representation behind increased capacity.

---

# 52. 2D representation prior art

TimesNet already demonstrates a related principle by converting 1D time series into 2D structures according to detected periods.

Therefore:

> “Convert time series into 2D structure and use 2D kernels” is prior art.

Project novelty should lie in systematic representation comparison and branch integration, not merely 2D conversion.

---

# 53. Full complex input representation

Complex-valued spectrum can be represented with real tensors:

\[
[\Re X,\Im X].
\]

This avoids circular phase issues and preserves exact invertibility.

It should be the primary Fourier representation baseline.

Only after this baseline should the project evaluate:

- magnitude-phase;
- learned complex layers;
- complex-valued neural networks.

---

# 54. Complex-valued neural networks — deferred option

A mathematically natural model would operate directly on:

\[
X\in\mathbb C^F.
\]

However, complex networks introduce:

- specialized layers;
- initialization issues;
- complex activations;
- framework limitations;
- difficult parameter-matching.

FreTS/FITS already demonstrate that much of the value can be captured with ordinary real networks over real/imaginary components.

Therefore fully complex neural layers are not required for first-stage STEP 06.

---

# 55. Target-domain transforms

For multi-output forecast:

\[
Y
=
[y_{t+1},\ldots,y_{t+H}],
\]

compute:

\[
Y_f=FFT(Y).
\]

Potential loss:

\[
\mathcal L
=
\lambda_t
\mathcal L_t(
\hat Y,Y
)
+
\lambda_f
\mathcal L_f(
\hat Y_f,Y_f
).
\]

Possible frequency loss:

\[
\mathcal L_f
=
\|\hat Y_f-Y_f\|_2^2.
\]

This tests whether preserving future waveform structure matters beyond pointwise error.

FreDF provides recent prior art for frequency-domain forecast objectives.

This should be a secondary STEP-06 extension, not mixed into initial input ablations.

---

# 56. Phase-aware target loss

Separate:

\[
A_Y,
\phi_Y.
\]

Potential:

\[
\mathcal L_\phi
=
1-
\cos(
\hat\phi_Y-\phi_Y
).
\]

This avoids circular discontinuity.

A phase-aware loss may penalize timing shifts that ordinary MSE undercharacterizes.

This is particularly relevant for turning-point timing.

However, phase is unreliable where:

\[
A_Y\approx0.
\]

Therefore phase loss should be magnitude-weighted:

\[
\mathcal L_{\phi,w}
=
\sum_k
w_k
\left[
1-\cos(
\hat\phi_k-\phi_k
)
\right]
\]

with:

\[
w_k
=
g(A_k).
\]

---

# 57. Financial-domain caution: spectral peaks are not stationary laws

A historical peak at frequency:

\[
f^*
\]

does not imply a permanent market cycle.

Therefore spectral features must be:

- rolling;
- causal;
- stability-tested;
- regime-aware;
- validated OOS.

The project must avoid “cycle discovery” narratives based on full-sample FFT peaks.

---

# 58. Synthetic validation before financial interpretation

Construct known signals.

## S1 — Pure sinusoid

\[
x_t
=
A\cos(2\pi ft+\phi).
\]

## S2 — Multiple sinusoids

\[
x_t
=
\sum_k
A_k\cos(2\pi f_kt+\phi_k).
\]

## S3 — Chirp

\[
f=f(t).
\]

## S4 — Amplitude modulation

\[
x_t
=
A(t)\cos(\omega t).
\]

## S5 — Phase shift

\[
y_t
=
x_{t-\tau}.
\]

## S6 — Noisy oscillation

\[
x_t=s_t+n_t.
\]

## S7 — Regime change

frequency or amplitude changes at:

\[
t_0.
\]

The representation pipeline must recover expected behavior before financial conclusions are drawn.

---

# 59. Phase ablation on controlled synthetic signals

For fixed:

\[
A_k
\]

vary only:

\[
\phi_k.
\]

Train equivalent forecasting models.

This tests whether representation captures phase effects independently of spectral energy.

Then reverse:

- fixed phase;
- varying magnitude.

This makes the amplitude/phase roles experimentally identifiable.

---

# 60. Public benchmark protocol

Use at least:

- ETT;
- Electricity;
- Traffic;
- Weather;
- Exchange Rate.

For fairness reuse published splits and reference preprocessing where possible.

Models:

- DLinear;
- PatchTST;
- FreTS or FITS;
- one wavelet/multiscale baseline;
- project modular model.

---

# 61. Project-data protocol

After public validation:

- EURUSD;
- ETHUSDT;
- other assets;
- 1 h;
- 4 h;
- technical;
- fundamental;
- cross-asset;
- calendar/macro.

Respect:

\[
4\text{ years train}
+
1\text{ year validation}
+
1\text{ year test}.
\]

All frequency-bin, scale, window and pair-selection decisions are training/validation only.

---

# 62. Final falsifiable hypotheses

## H6.1 — Explicit transform representation improves finite-model learnability

There exists at least one invertible or near-invertible transform:

\[
T
\]

such that:

\[
P(T(X))
>
P(X)
\]

under parameter-matched models.

**Falsified if:** raw input matches or outperforms all transforms consistently.

---

## H6.2 — Full complex Fourier representation outperforms magnitude-only

\[
P(\Re X_f,\Im X_f)
>
P(|X_f|).
\]

**Falsified if:** magnitude-only is consistently equivalent or superior.

---

## H6.3 — Phase contains incremental forecasting information

\[
P([X,\phi])
>
P(X)
\]

or:

\[
P(A,\phi)
>
P(A).
\]

**Falsified if:** phase never produces reproducible incremental value.

---

## H6.4 — Phase randomization degrades performance beyond power-spectrum preservation

For surrogates preserving approximately:

\[
|X_f|,
\]

forecast performance decreases when phase organization is randomized.

**Falsified if:** phase-randomized data preserve equivalent predictive structure.

---

## H6.5 — Time-frequency representations outperform global spectrum under nonstationarity

For regime-changing/nonstationary datasets:

\[
P(TF(X))
>
P(FFT(X)).
\]

**Falsified if:** global FFT consistently matches or dominates.

---

## H6.6 — Wavelet/multiscale representation is more robust than fixed-window STFT when characteristic scales vary

\[
P(Wavelet(X))
>
P(STFT(X))
\]

in variable-frequency synthetic/public regimes.

**Falsified if:** no reproducible advantage exists.

---

## H6.7 — Different frequency bands require different modeling capacity

Specialized band models outperform a shared model under equal total parameters.

**Falsified if:** shared model is equal or superior.

---

## H6.8 — Noise-aware spectral selection improves rate/performance efficiency

Selecting spectral components using training-derived SNR improves or preserves performance with fewer coefficients.

**Falsified if:** SNR filtering removes useful predictive components.

---

## H6.9 — Phase-aware representations improve timing accuracy

Metrics sensitive to turning-point or temporal alignment improve even when aggregate MAE improvement is small.

**Falsified if:** phase-aware models do not improve timing-sensitive metrics.

---

## H6.10 — Cross-spectral coherence adds information beyond time-domain correlation

\[
P([X,C_{ij}(f),\phi_{ij}(f)])
>
P(X)
\]

for selected multivariate pairs.

**Falsified if:** cross-spectral features provide no incremental value.

---

## H6.11 — Dynamic spectral lag improves cross-series modeling

Estimated:

\[
\tau_{ij}(f,t)
\]

adds incremental predictive value over static/zero-lag relationships.

**Falsified if:** lag estimates are unstable or uninformative.

---

## H6.12 — Hilbert amplitude/phase features are useful only after appropriate component separation

\[
P(Hilbert(IMF/band))
>
P(Hilbert(raw))
\]

for multicomponent signals.

**Falsified if:** direct raw Hilbert representation performs equally well or better.

---

## H6.13 — Representation value depends on SNR

Benefit of high-resolution phase/frequency features should decrease as noise rises.

Formally:

\[
\Delta P_T
=
f(\mathrm{SNR}).
\]

**Falsified if:** representation benefit is unrelated to controlled SNR.

---

## H6.14 — Representation + raw can outperform either individually

\[
P([X,T(X)])
>
\max(
P(X),
P(T(X))
).
\]

**Falsified if:** transformed branches are always redundant once raw is available.

---

## H6.15 — Better representation can reduce required model size

For target performance:

\[
P_0,
\]

there exists:

\[
|\theta_T|
<
|\theta_{raw}|
\]

such that:

\[
P_T\geq P_0.
\]

**Falsified if:** transformed models require equal or greater capacity.

---

## H6.16 — Wavelet scattering offers a useful fixed front-end

A scattering representation reaches comparable forecasting quality with reduced learned front-end complexity.

**Falsified if:** learned Conv1D consistently dominates at equal resource budget.

---

## H6.17 — Frequency-domain target loss improves multi-horizon structural coherence

Joint time/frequency objective improves horizon consistency.

**Falsified if:** time-domain loss alone is equal or superior.

---

## H6.18 — Frequency-specific cross-series dependence is not captured completely by scalar correlation

Spectral/coherence features reveal stable OOS relationships not reflected in simple correlation.

**Falsified if:** all incremental spectral dependence disappears OOS.

---

# 63. Metrics beyond MAE

STEP 06 needs metrics sensitive to representation-specific behavior.

## Standard

- MAE;
- RMSE;
- \(R^2\);
- per-horizon error.

## Spectral

\[
D_{PSD}
\]

between true and predicted PSD.

## Complex spectral error

\[
\|Y_f-\hat Y_f\|_2.
\]

## Phase error

Circular:

\[
E_\phi
=
1-\cos(
\hat\phi-\phi
).
\]

## Coherence error

Difference between true and predicted cross-spectral coherence where applicable.

## Timing

- peak/turning-point lag;
- zero-crossing timing;
- directional transition lag.

## Calibration

for probabilistic/Bayesian heads.

---

# 64. Statistical validation

Use:

- Diebold–Mariano for forecast-loss comparisons where assumptions are appropriate;
- block/stationary bootstrap;
- seed distributions;
- FDR/multiple-comparison control;
- predeclared primary hypotheses.

Spectral feature selection must be nested inside training/validation procedures.

---

# 65. Minimal experiment matrix

| ID | Representation | Phase | Localized | Multivariate | Model |
|---|---|---:|---:|---:|---|
| R00 | Raw | implicit | No | Native | DLinear |
| R01 | FFT magnitude | No | No | No | MLP |
| R02 | FFT phase | Yes | No | No | MLP |
| R03 | FFT real+imag | Yes | No | No | MLP/FreTS |
| R04 | Raw + FFT complex | Yes | No | No | Multi-branch |
| R05 | Multitaper PSD | No | Rolling | No | MLP |
| R06 | STFT complex | Yes | Yes | No | Conv2D/light |
| R07 | DWT | implicit | Multiscale | No | Branch model |
| R08 | CWT complex | Yes | Yes | No | Conv2D/light |
| R09 | Hilbert amp/phase | Yes | Yes | No | Conv1D |
| R10 | IMF + Hilbert | Yes | Yes | No | Branch model |
| R11 | Cross-spectrum | Yes | Rolling | Yes | Branch model |
| R12 | Wavelet coherence | Yes | Yes | Yes | Branch model |
| R13 | Scattering | implicit | Multiscale | No | Linear/MLP |
| R14 | Best representation | Best | Best | Best | PatchTST |
| R15 | Best multi-branch | Best | Best | Best | Project predictor |

---

# 66. Recommended staged implementation order

1. Synthetic sinusoid/chirp validation.
2. Raw baseline.
3. FFT magnitude.
4. FFT real/imag.
5. Phase-only and magnitude+phase.
6. Phase-randomized surrogate controls.
7. Rolling FFT window sweep.
8. Multitaper PSD.
9. DWT.
10. STFT.
11. CWT.
12. Wavelet scattering.
13. Hilbert on controlled narrowband components.
14. EMD/HHT only if justified.
15. Cross-spectrum/coherence.
16. Wavelet coherence.
17. Frequency-band specialized branches.
18. Public benchmark.
19. Parameter-matched multi-branch comparison.
20. Project predictor integration.
21. Target-frequency loss extension.

---

# 67. Reuse matrix

| Requirement | Prior art / implementation family | Recommended action |
|---|---|---|
| FFT/DFT | NumPy/SciPy/PyTorch FFT | Reuse |
| PSD | Welch | Reuse SciPy |
| Multitaper | Thomson / MNE / spectrum libraries | Reuse validated implementation |
| STFT | SciPy/PyTorch | Reuse |
| DWT/CWT | PyWavelets | Reuse |
| Wavelet coherence | established CWT coherence algorithms | Reuse/reference |
| EMD/HHT | PyEMD or validated library | Audit endpoint/causality |
| Synchrosqueezing | ssqueezepy or validated implementation | Optional |
| Scattering | Kymatio | Benchmark before custom implementation |
| Fourier forecasting | FEDformer/FreTS/FITS | Benchmark/reference |
| Multi-period | TimesNet | Benchmark/reference |
| Multiscale | TimeMixer | Benchmark/reference |
| Wavelet forecasting | WPMixer/WaveletMixer | Benchmark/reference |
| Wavelet tokenization | WaveToken | High-priority prior art |
| Frequency target objective | FreDF | Benchmark/reference |
| Phase-aware modeling | PARCNet / complex-spectrum literature | Benchmark/reference |
| Surrogates | Theiler / Schreiber methods | Use established algorithms |

---

# 68. Major failure modes

## 68.1. Leakage through centered transforms

Most dangerous operational failure.

## 68.2. Boundary contamination

Wavelet/Hilbert/EMD endpoints can be unreliable.

## 68.3. Spectral leakage

Finite windows create false spread.

## 68.4. Phase instability near zero amplitude

When:

\[
|X[k]|\approx0,
\]

phase is numerically unstable.

## 68.5. Phase wrapping

Use circular encoding.

## 68.6. False spectral cycles

Historical peaks may not persist.

## 68.7. Regime mixing

Long FFT windows mix distinct market regimes.

## 68.8. Branch parameter inflation

Representation benefit can be confounded by model size.

## 68.9. Pair explosion

Cross-spectral features scale approximately:

\[
O(m^2F).
\]

## 68.10. Coherence interpreted as causality

Forbidden.

## 68.11. HHT overinterpretation

Instantaneous frequency is not universally meaningful.

## 68.12. Multiple-comparison mining

Large transform grids create false discoveries.

---

# 69. Audit checklist

- [ ] Complete complex spectrum is distinguished from magnitude-only spectrum.
- [ ] Phase is represented circularly or by real/imaginary components.
- [ ] Phase instability at low magnitude is handled.
- [ ] Every rolling transform uses only data available at forecast time.
- [ ] Wavelet extension/padding mode is documented.
- [ ] Hilbert transform endpoint handling is documented.
- [ ] EMD/HHT is not used without mode/endpoint stability tests.
- [ ] FFT window function is logged.
- [ ] FFT/STFT window length is tuned only with training/validation.
- [ ] Multitaper parameters are training/validation only.
- [ ] Spectral bin selection never uses test.
- [ ] Phase-randomized surrogate implementation preserves required conjugate symmetry.
- [ ] Multivariate surrogate controls preserve specified cross-correlations when required.
- [ ] Cross-spectral coherence is not labeled causal.
- [ ] Low-coherence phase lags are masked.
- [ ] Parameter-matched architecture controls are included.
- [ ] Raw branch is preserved in early ablations.
- [ ] Public benchmark is run before financial claims.
- [ ] Synthetic known-frequency tests pass.
- [ ] SNR interaction is measured.
- [ ] Tail/event behavior is separately reported.
- [ ] Spectral prediction metrics supplement ordinary MAE.
- [ ] Multiple-comparison correction policy is predefined.
- [ ] Existing FEDformer/FreTS/FITS/TimesNet/TimeMixer/WPMixer/WaveToken prior art is audited before custom architecture work.
- [ ] Recent phase-aware prior art is acknowledged.
- [ ] Claims distinguish representation accessibility from creation of new information.

---

# 70. Deliverables

Recommended artifacts:

1. `step06_representation_spec.json`
2. `transform_causality_manifest.json`
3. `fft_window_sweep.parquet`
4. `magnitude_phase_ablation.parquet`
5. `phase_surrogate_results.parquet`
6. `multitaper_spectra.parquet`
7. `stft_ablation.parquet`
8. `wavelet_scale_metrics.parquet`
9. `hilbert_phase_metrics.parquet`
10. `cross_spectral_features.parquet`
11. `wavelet_coherence_features.parquet`
12. `representation_model_size_frontier.parquet`
13. `forecast_metrics.parquet`
14. `spectral_prediction_metrics.parquet`
15. `statistical_tests.json`
16. `step06_audit_report.md`
17. reproducibility manifest with:
    - data hashes;
    - split timestamps;
    - transform parameters;
    - padding modes;
    - random seeds;
    - software versions;
    - git commits.

---

# 71. Decision gates

## Gate 6A — Synthetic correctness

Transforms recover known amplitude/frequency/phase behavior.

## Gate 6B — No leakage

Causal support is verified.

## Gate 6C — Complex-spectrum value

Real/imag or magnitude+phase beats or matches magnitude-only sufficiently to justify phase modeling.

## Gate 6D — Local time-frequency value

STFT/wavelets improve nonstationary cases.

## Gate 6E — SNR robustness

Representation benefit survives controlled noise analysis.

## Gate 6F — Multivariate spectral value

Cross-spectral features add stable validation information.

## Gate 6G — Parameter-matched benefit

Transform advantage remains under equal model capacity.

## Gate 6H — Public benchmark transfer

Effect survives at least one public dataset family.

## Gate 6I — Architecture transfer

Effect survives more than one model family.

## Gate 6J — Held-out confirmation

Only final preselected configuration reaches test.

---

# 72. Possible scientifically valid outcomes

## Outcome A

Raw time-domain representation remains optimal.

## Outcome B

Fourier complex representation improves long-horizon forecasting.

## Outcome C

Phase adds incremental timing information.

## Outcome D

Phase is useful only in selected SNR regimes.

## Outcome E

Wavelets dominate global Fourier for nonstationary series.

## Outcome F

Multi-scale branches outperform a monolithic branch.

## Outcome G

Cross-spectral features improve multivariate forecasting.

## Outcome H

Spectral features compress model-size requirements without improving absolute peak accuracy.

## Outcome I

Representation benefits disappear once model capacity is sufficiently large.

Each result is useful.

---

# 73. State-of-the-art synthesis

The literature strongly confirms the broad premise that time-series learning can benefit from representations beyond raw time samples.

However, the field has already covered much of the obvious territory:

- Fourier-domain forecasting;
- frequency-domain MLPs;
- complex spectra;
- multi-period 2D restructuring;
- multi-scale decomposition;
- wavelet forecasting;
- wavelet tokenization;
- frequency-domain prediction losses;
- phase-aware forecasting.

Therefore the project should not claim novelty for these components individually.

The stronger research contribution to investigate is the **systematic information-processing chain**:

\[
\boxed{
\text{sampling constraints}
\rightarrow
\text{noise/SNR}
\rightarrow
\text{denoising}
\rightarrow
\text{quantization}
\rightarrow
\text{source/context structure}
\rightarrow
\text{amplitude/frequency/phase/time-frequency exposure}
\rightarrow
\text{branch-specific models}
}
\]

under controlled causal ablations.

---

# 74. Project-specific candidate architecture

A possible future architecture, **not yet a recommendation**, is:

\[
X
\rightarrow
\begin{cases}
B_{raw}(X)\\
B_{innovation}(X)\\
B_{FFT}(X)\\
B_{wavelet}(X)\\
B_{phase}(X)\\
B_{cross}(X)
\end{cases}
\]

with:

\[
Z_i=E_i(B_i)
\]

then:

\[
Z_{core}
=
C(Z_1,\ldots,Z_K)
\]

and:

\[
\hat Y_h
=
H_h(Z_{core})
\]

for multiple predictive heads.

The project's existing Conv1D + BiLSTM + Bayesian-head pattern can remain unchanged initially while the upstream representations are ablated.

Only after representation value is established should branch-specific neural architectures be optimized.

---

# 75. Recommended immediate research priority

The highest-information low-cost experiment is:

\[
\boxed{
\text{raw}
\quad vs \quad
|FFT|
\quad vs \quad
\phi
\quad vs \quad
(\Re FFT,\Im FFT)
\quad vs \quad
\text{raw + complex FFT}
}
\]

combined with:

\[
\boxed{
\text{phase-randomized surrogate control}
}
\]

and a rolling causal window sweep.

Why this first:

- mathematically clean;
- inexpensive;
- directly tests amplitude versus phase;
- directly addresses the communications analogy;
- avoids immediate architectural complexity;
- establishes whether more expensive STFT/wavelet/Hilbert branches are justified.

---

# 76. References — IEEE style

[1] A. V. Oppenheim and J. S. Lim, “The Importance of Phase in Signals,” *Proceedings of the IEEE*, vol. 69, no. 5, pp. 529–541, May 1981, doi: 10.1109/PROC.1981.12022. Available: https://doi.org/10.1109/PROC.1981.12022

[2] D. J. Thomson, “Spectrum Estimation and Harmonic Analysis,” *Proceedings of the IEEE*, vol. 70, no. 9, pp. 1055–1096, Sep. 1982, doi: 10.1109/PROC.1982.12433. Available: https://doi.org/10.1109/PROC.1982.12433

[3] L. Cohen, “Time-Frequency Distributions—A Review,” *Proceedings of the IEEE*, vol. 77, no. 7, pp. 941–981, Jul. 1989. Available via IEEE/DOI bibliographic services.

[4] B. Boashash, “Estimating and Interpreting the Instantaneous Frequency of a Signal—Part 1: Fundamentals,” *Proceedings of the IEEE*, vol. 80, no. 4, pp. 520–538, Apr. 1992, doi: 10.1109/5.135376. Available: https://doi.org/10.1109/5.135376

[5] B. Boashash, “Estimating and Interpreting the Instantaneous Frequency of a Signal—Part 2: Algorithms and Applications,” *Proceedings of the IEEE*, vol. 80, no. 4, pp. 540–568, Apr. 1992, doi: 10.1109/5.135378. Available: https://doi.org/10.1109/5.135378

[6] N. E. Huang *et al.*, “The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and Non-Stationary Time Series Analysis,” *Proceedings of the Royal Society A*, vol. 454, no. 1971, pp. 903–995, 1998, doi: 10.1098/rspa.1998.0193. Available: https://doi.org/10.1098/rspa.1998.0193

[7] C. Torrence and G. P. Compo, “A Practical Guide to Wavelet Analysis,” *Bulletin of the American Meteorological Society*, vol. 79, no. 1, pp. 61–78, 1998, doi: 10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2. Available: https://doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2

[8] I. Daubechies, J. Lu, and H.-T. Wu, “Synchrosqueezed Wavelet Transforms: An Empirical Mode Decomposition-like Tool,” *Applied and Computational Harmonic Analysis*, vol. 30, no. 2, pp. 243–261, 2011, doi: 10.1016/j.acha.2010.08.002. Available: https://doi.org/10.1016/j.acha.2010.08.002

[9] J. Bruna and S. Mallat, “Invariant Scattering Convolution Networks,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 35, no. 8, pp. 1872–1886, 2013, doi: 10.1109/TPAMI.2012.230. Available: https://doi.org/10.1109/TPAMI.2012.230

[10] A. Grinsted, J. C. Moore, and S. Jevrejeva, “Application of the Cross Wavelet Transform and Wavelet Coherence to Geophysical Time Series,” *Nonlinear Processes in Geophysics*, vol. 11, pp. 561–566, 2004, doi: 10.5194/npg-11-561-2004. Available: https://doi.org/10.5194/npg-11-561-2004

[11] J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, and J. D. Farmer, “Testing for Nonlinearity in Time Series: The Method of Surrogate Data,” *Physica D*, vol. 58, no. 1–4, pp. 77–94, 1992, doi: 10.1016/0167-2789(92)90102-S. Available: https://doi.org/10.1016/0167-2789(92)90102-S

[12] T. Schreiber and A. Schmitz, “Improved Surrogate Data for Nonlinearity Tests,” *Physical Review Letters*, vol. 77, no. 4, pp. 635–638, 1996, doi: 10.1103/PhysRevLett.77.635. Available: https://doi.org/10.1103/PhysRevLett.77.635

[13] D. Prichard and J. Theiler, “Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables,” *Physical Review Letters*, vol. 73, pp. 951–954, 1994, doi: 10.1103/PhysRevLett.73.951. Available: https://doi.org/10.1103/PhysRevLett.73.951

[14] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting,” in *Proceedings of the 39th International Conference on Machine Learning*, PMLR, vol. 162, pp. 27268–27286, 2022. Available: https://proceedings.mlr.press/v162/zhou22g.html

[15] K. Yi *et al.*, “Frequency-domain MLPs are More Effective Learners in Time Series Forecasting,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023. Available: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html

[16] Z. Xu, A. Zeng, and Q. Xu, “FITS: Modeling Time Series with 10k Parameters,” in *International Conference on Learning Representations*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html

[17] H. Wu *et al.*, “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis,” in *International Conference on Learning Representations*, 2023. Available: https://iclr.cc/virtual/2023/poster/11976

[18] S. Wang *et al.*, “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting,” in *International Conference on Learning Representations*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html

[19] M. M. N. Murad, M. Aktukmak, and Y. Yilmaz, “WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 18, pp. 19581–19588, 2025, doi: 10.1609/aaai.v39i18.34156. Available: https://doi.org/10.1609/aaai.v39i18.34156

[20] Z. Zhang *et al.*, “WaveletMixer: A Multi-Resolution Wavelets Based MLP-Mixer for Multivariate Long-Term Time Series Forecasting,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 21, 2025. Available: https://ojs.aaai.org/index.php/AAAI/article/view/34434

[21] L. Masserano *et al.*, “Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization,” in *Proceedings of the 42nd International Conference on Machine Learning*, PMLR, vol. 267, pp. 43248–43275, 2025. Available: https://proceedings.mlr.press/v267/masserano25a.html

[22] H. Wang *et al.*, “FreDF: Learning to Forecast in the Frequency Domain,” in *International Conference on Learning Representations*, 2025. Available: https://iclr.cc/virtual/2025/poster/31031

[23] S. Huang, Z. Zhao, C. Li, and L. Bai, “TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting,” in *International Conference on Learning Representations*, 2025. Available: https://iclr.cc/virtual/2025/poster/27844

[24] Z. Li, H. Chai, Z. Song, S. Liu, and X. Liu, “PARCNet: Phase-aware Residual Correction Network for Efficient Multivariate Time Series Forecasting,” *Knowledge-Based Systems*, vol. 348, art. 116370, 2026, doi: 10.1016/j.knosys.2026.116370. Available: https://doi.org/10.1016/j.knosys.2026.116370

[25] R. T. Krafty and M. Hall, “Nonparametric Spectral Analysis of Multivariate Time Series,” *Annual Review of Statistics and Its Application*, 2020, doi: 10.1146/annurev-statistics-031219-041138. Available: https://doi.org/10.1146/annurev-statistics-031219-041138

---

# 77. Final status

**STEP 06 is theoretically specified, grounded in classical signal-processing theory and current forecasting research, and organized as a falsifiable experimental protocol.**

The strongest immediate hypothesis is not simply:

> “frequency features help.”

It is the more precise claim:

\[
\boxed{
\text{explicit separation of amplitude, phase and localized spectral structure}
}
\]

may reduce the burden placed on a finite-capacity forecasting model.

The highest-priority direct test is:

\[
\boxed{
\text{raw}
\leftrightarrow
\text{magnitude}
\leftrightarrow
\text{phase}
\leftrightarrow
\text{full complex spectrum}
\leftrightarrow
\text{raw + complex spectrum}
}
\]

with controlled phase-randomized surrogates and parameter-matched models.

If agent audit approves this protocol, proceed to:

**STEP 07 — matched filtering / template detection / convolution / event detection as the analogue of receiver-side signal matching.**
