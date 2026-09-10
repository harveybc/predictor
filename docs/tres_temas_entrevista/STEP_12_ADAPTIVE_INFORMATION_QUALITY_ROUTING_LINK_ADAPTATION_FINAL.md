# STEP 12 — Adaptive Information-Quality Routing / Link Adaptation
## Causal Configuration Selection from SNR, Uncertainty, OOD, Missingness, Regime and Synchronization State

**Status:** Research protocol — final draft for independent agent audit  
**Date:** 2026-09-05  
**Scope:** Dynamic selection among already validated preprocessing/representation/detector/model configurations according to current information quality. Initial domain: multivariate financial time series; intended abstraction: general multisource sequential systems.  
**Prerequisites:** STEPS 01–11.  
**Critical boundary:** STEP 12 chooses **which validated mode/configuration to use now**. STEP 13 will address **how to allocate a limited continuous/discrete resource budget across multiple active branches**.  
**Implementation status:** research protocol only; no router implementation or campaign is authorized by this document.

---

# 0. Executive conclusion

STEP 12 asks:

> **If the quality and structure of the incoming information change over time, should the pipeline continue using one fixed processing configuration, or should it adaptively choose the representation/detector/model path best suited to the current state?**

The classical communications analogue is adaptive modulation and coding (AMC) / link adaptation.

A transmitter/receiver observes or estimates channel state:

\[
q_t
\]

and chooses a mode:

\[
m_t
=
\pi(q_t)
\]

from a finite set:

\[
\mathcal M
=
\{
m_1,\ldots,m_K
\}.
\]

Different modes trade:

- throughput;
- robustness;
- spectral efficiency;
- error probability;
- power.

The STEP-12 ML analogue is:

\[
\boxed{
q_t
=
[
\widehat{SNR}_t,\;
U_t,\;
OOD_t,\;
M_t,\;
H_{regime,t},\;
S_t,\;
Q_{sync,t},\;
Q_{det,t},
\ldots
]
}
\]

followed by:

\[
\boxed{
a_t
=
\pi(q_t)
}
\]

where \(a_t\) selects one of a **small bank of previously validated pipeline modes**.

The router is not permitted to invent a new denoiser, a new feature extractor, a new detector or a new forecasting architecture. It only chooses among already qualified alternatives.

The strongest state-of-the-art lesson is that adaptive routing is already mature in several adjacent fields:

- communications link adaptation;
- conditional computation;
- sparse Mixture-of-Experts;
- early-exit/selective prediction;
- uncertainty/risk-controlled abstention;
- adaptive pathway time-series models;
- time-series MoE specialization;
- quality-driven routing under missingness;
- OOD frequency routing;
- dynamic time/frequency expert routing.

Therefore the project should begin with the **simplest interpretable causal router** and treat large learned routing systems as late-stage comparators.

The recommended ladder is:

\[
\boxed{
\text{static best mode}
\rightarrow
\text{oracle upper bound}
\rightarrow
\text{threshold/rule router}
\rightarrow
\text{small supervised meta-router}
\rightarrow
\text{soft gating}
\rightarrow
\text{MoE-style router}
\rightarrow
\text{RL routing only if sequential effects require it}.
}
\]

---

# 1. Classical adaptive modulation/coding principle

For a fading/noisy communications channel, fixed modulation/coding wastes capacity:

- conservative mode is reliable but slow in good conditions;
- aggressive mode is efficient but error-prone in poor conditions.

Adaptive systems estimate channel quality and switch among modes.

Abstractly:

\[
m_t
=
\pi(
CSI_t
).
\]

The objective can be written:

\[
\boxed{
m_t
=
\arg\max_{m\in\mathcal M}
U(m\mid CSI_t)
}
\]

where utility balances:

\[
U
=
\text{rate}
-
\lambda_1\text{error risk}
-
\lambda_2\text{resource cost}.
\]

This decision structure—not MQAM itself—is the STEP-12 homology.

---

# 2. Goldsmith–Chua adaptive modulation precedent

Goldsmith and Chua's variable-rate/variable-power MQAM work established a canonical adaptive strategy for fading channels:

- observe channel state;
- alter transmission rate/power/modulation according to that state;
- exploit favorable conditions;
- become more conservative when conditions degrade.

The transferable project principle is:

\[
\boxed{
\text{one fixed configuration need not be optimal over all information-quality states}.
}
\]

---

# 3. Current communications state of the art

Modern AMC research still revolves around:

- Channel State Information (CSI);
- reliability/throughput trade-offs;
- stale/noisy state estimates;
- temporal channel evolution;
- rule-based versus learned policies.

A 2026 survey/benchmark of adaptive modulation and coding for underwater acoustic communications compares:

- SNR thresholding;
- boosted-tree selection;
- multi-armed bandits;
- LSTM-DQN policies;

and reports that sequential channel information can materially matter in rapidly changing channels.

The project does **not** inherit the underwater channel model.

It inherits the experimental structure:

\[
\boxed{
\text{simple threshold baseline}
\rightarrow
\text{supervised router}
\rightarrow
\text{sequential policy only if needed}.
}
\]

---

# 4. STEP 12 versus STEP 13

These steps must remain distinct.

## STEP 12 — link adaptation / mode selection

Choose:

\[
a_t
\in
\mathcal A
\]

such as:

- baseline mode;
- robust/noisy-input mode;
- spectral mode;
- cross-series mode;
- fallback mode.

This is primarily **conditional selection**.

## STEP 13 — multiplexing/MIMO/resource allocation

Given active branches, solve:

\[
\max_{\{c_i\}}
P
\]

subject to:

\[
\sum_i c_i
\leq
C_{max}.
\]

Here \(c_i\) can be:

- latent width;
- bits;
- parameters;
- FLOPs;
- attention budget;
- sampling/resource allocation.

Thus:

\[
\boxed{
\text{STEP 12: Which mode?}
}
\]

\[
\boxed{
\text{STEP 13: How much resource to each active mode/branch?}
}
\]

---

# 5. STEP 12 is not test-time adaptation

STEP 08 allowed carefully governed test-time parameter adaptation.

STEP 12 instead chooses among **frozen** configurations.

At time \(t\):

\[
\theta_a
\]

for every eligible action \(a\) remains unchanged.

Only:

\[
a_t
\]

changes.

This separation is important for attribution and safety.

---

# 6. STEP 12 is not a new MoE model by default

Sparse MoE is relevant prior art, but STEP 12 does not begin by replacing the project core with a Transformer MoE.

The first question is narrower:

> can existing pipeline alternatives be routed better than choosing one global champion?

If the answer is no, a larger MoE is unnecessary.

---

# 7. Causal Information Quality State (IQS)

Define:

\[
\boxed{
q_t
=
IQS_t
}
\]

as a vector of quantities available by decision time \(t\).

Candidate components:

## From STEP 03

\[
q^{snr}_t
=
\widehat{SNR}_t
\]

or band/family summaries.

## From current predictive models

\[
q^{unc}_t
=
U_t
\]

such as MC predictive uncertainty.

## OOD / distribution state

\[
q^{ood}_t.
\]

## Missingness / age

\[
q^{miss}_t.
\]

## STEP 05

\[
q^{surp}_t
=
-\log p(X_t\mid context).
\]

## Regime uncertainty

\[
q^{reg}_t
=
H(regime_t).
\]

## STEP 10

\[
q^{sync}_t
=
[
LOCK_t,
Q_{sync,t},
H(p(\tau_t))
].
\]

## STEP 07

\[
q^{det}_t
=
\text{detector confidence}.
\]

---

# 8. Minimal IQS for the first experiment

Do not begin with every prior signal.

Recommended minimal quality vector:

\[
\boxed{
q^{(0)}_t
=
[
SNR_t,\;
U_t,\;
missingness_t,\;
OOD_t,\;
Q_{sync,t}
].
}
\]

Additional variables are introduced only through ablation.

Reason:

> a router with dozens of quality indicators can overfit more easily than the experts it routes.

---

# 9. Existing project signals to reuse

The current `predictor` infrastructure already exposes a `predict_with_uncertainty(...)` interface and multiple pipelines use configurable MC sampling.

Therefore STEP 12 should reuse existing prediction-uncertainty outputs before creating a separate uncertainty model.

The current `agent-multi` state-profile plumbing also contains regime-probability and `state_regime_entropy` feature contracts.

These are existing candidate quality/state signals.

STEP 12 should consume them only if the corresponding source pipeline is already validated; it should not duplicate their estimation logic.

---

# 10. Quality values must not depend on current future target

At routing time:

\[
q_t
=
f(X_{\le t})
\]

only.

Forbidden:

\[
q_t
=
f(Y_{t+h})
\]

or any statistic constructed from an outcome that has not matured.

Router training labels may use historical realized future losses; online routing inputs may not.

---

# 11. Eligible configuration bank

Let:

\[
\mathcal A
=
\{
a_0,\ldots,a_K
\}.
\]

An action is eligible only if its underlying mechanism already passed its own upstream gate.

Examples are conditional, not guaranteed:

- raw/current baseline;
- STEP-03 denoised mode **if STEP 03 shows benefit**;
- STEP-06 spectral/phase branch **if STEP 06 shows benefit**;
- STEP-07 detector-assisted mode **if STEP 07 passes**;
- STEP-08 equalized mode **if STEP 08 passes**;
- STEP-09 common/private mode **if STEP 09 passes**;
- STEP-10 lag-aware mode **if STEP 10 passes**;
- STEP-11 robust latent **if STEP 11 passes**.

No failed/null module becomes an “expert” merely to populate the router.

---

# 12. Configuration action, not arbitrary parameter mutation

The first router chooses among **frozen packages**.

Example:

\[
a_1
=
\texttt{baseline}
\]

\[
a_2
=
\texttt{robust\_low\_snr}
\]

\[
a_3
=
\texttt{spectral\_high\_quality}
\]

\[
a_4
=
\texttt{channel\_independent\_sync\_fallback}.
\]

Each package has:

- frozen transforms;
- frozen encoder;
- frozen detector;
- frozen predictor;
- known compute cost.

The router cannot independently tweak 30 parameters at each timestamp.

That would collapse STEP 12 into online hyperparameter optimization.

---

# 13. The no-trade / abstain / fallback mode

A communications link can reduce rate or effectively refuse an aggressive mode when conditions are poor.

The ML analogue may include:

\[
a_{safe}
\]

such as:

- baseline-only prediction;
- channel-independent mode;
- conservative predictor;
- abstain/no-decision where the application contract permits.

For a trading application, “abstain” can naturally map to:

\[
\boxed{
\text{no new trade / no new high-confidence signal}
}
\]

rather than forcing a speculative prediction-driven action.

STEP 12 itself evaluates predictive routing first; trading policy consequences remain downstream.

---

# 14. Routing utility

For sample/time \(t\) and action \(a\), define:

\[
\boxed{
J_t(a)
=
-L_t(a)
-\lambda_C C(a)
-\lambda_R R_t(a)
-\lambda_S S(a_{t-1},a)
}
\]

where:

- \(L_t(a)\): realized predictive loss;
- \(C(a)\): compute/latency cost;
- \(R_t(a)\): risk/uncertainty penalty;
- \(S(a_{t-1},a)\): switching penalty.

A pure accuracy experiment sets:

\[
\lambda_C
=
\lambda_R
=
\lambda_S
=
0.
\]

Cost/risk terms are added only after predictive routing works.

---

# 15. Switching cost and route chatter

If mode selection changes every timestamp:

\[
a_t
\neq
a_{t-1}
\]

frequently, the system may become:

- computationally expensive;
- unstable;
- hard to audit.

Define:

\[
N_{switch}
=
\sum_t
\mathbb{1}
[
a_t\neq a_{t-1}
].
\]

A switching penalty or hysteresis can be used after the basic router works.

---

# 16. State-of-the-art conditional computation — Adaptive Computation Time

Graves' Adaptive Computation Time (ACT) lets a sequence model learn how much computation to allocate to each input.

The key observation for STEP 12 is:

\[
\boxed{
\text{harder/less predictable inputs may justify more or different computation}.
}
\]

ACT is not directly implemented here; it establishes conditional computation as a principled paradigm.

---

# 17. Sparse Mixture-of-Experts

Shazeer et al.'s sparsely gated MoE established:

\[
\text{input}
\rightarrow
\text{gate}
\rightarrow
\text{subset of experts}.
\]

Conditional activation increases total model capacity without activating all parameters on every sample.

This is a direct architectural relative of adaptive link modes.

---

# 18. Switch Transformer lesson

Switch Transformers simplify sparse routing to a single selected expert per token in many configurations and highlight several practical problems:

- expert imbalance;
- router instability;
- communication overhead;
- capacity overflow.

STEP 12 should inherit the warnings, not the trillion-parameter scale.

A tiny router with four modes can suffer the same conceptual failure:

\[
\boxed{
\text{route collapse}
}
\]

where one action receives almost every sample.

---

# 19. Router-collapse diagnostic

Expert/action utilization:

\[
p(a)
=
\frac{
N_a
}{
N
}.
\]

Entropy:

\[
H(A)
=
-\sum_a
p(a)\log p(a).
\]

Low entropy is not automatically bad if one mode truly dominates.

But a learned router that collapses during training while an oracle shows strong sample-specific specialization is defective.

---

# 20. Selective prediction and reject option

SelectiveNet formalizes the idea that a model can decide:

\[
\text{predict}
\quad\text{or}\quad
\text{reject}.
\]

The main metric becomes a risk–coverage trade-off.

This is highly relevant for a STEP-12 safe mode.

The router need not always select a more elaborate expert.

Sometimes the correct adaptive response is:

\[
\boxed{
\text{do less / abstain / use safe fallback}.
}
\]

---

# 21. Conformal risk control

Conformal Risk Control provides a model-agnostic method for controlling expected monotone losses under calibration assumptions/protocols.

STEP 12 does not require conformal routing in the first experiment.

But if the router is later used to make a safety claim such as:

> only use aggressive mode when expected risk is below \(\alpha\),

then conformal/risk-control methods are a much stronger basis than arbitrary uncertainty thresholds.

---

# 22. Early-exit networks

Early-exit systems adapt computation depth according to sample difficulty.

Modern work also emphasizes that routing should be uncertainty-aware and calibrated.

STEP-12 relevance:

- branch activation can be viewed as an exit/depth decision;
- “more compute” should be justified by expected marginal benefit.

This becomes especially important in STEP 13.

---

# 23. State of the art — Pathformer

Pathformer (ICLR 2024) uses adaptive pathways across multiple temporal scales.

Its router adapts the multi-scale modeling process to the varying temporal dynamics of each input.

This is one of the closest established time-series analogues to STEP 12.

Important lesson:

\[
\boxed{
\text{different windows may require different temporal-scale pathways}.
}
\]

Project action:

> compare any future scale-routing idea against Pathformer's adaptive-pathway principle rather than claiming novelty for input-dependent scale selection.

---

# 24. Time-MoE — ICLR 2025

Time-MoE uses sparse mixture-of-experts for large-scale time-series forecasting.

It activates only a subset of experts for each prediction, increasing model capacity without proportionally increasing inference computation.

Its relevance is:

- sparse conditional computation is viable in time series;
- large heterogeneous time-series corpora benefit from specialization.

Its irrelevance to the first project experiment is equally important:

> STEP 12 does not need a billion-parameter time-series foundation model.

---

# 25. Moirai-MoE — ICML 2025

Moirai-MoE makes an especially important observation.

Human-defined frequency grouping is often too coarse because:

- series with different frequencies can share patterns;
- one series can change distribution even within a short window.

It therefore performs sparse **token-level expert specialization**.

This directly challenges a simplistic STEP-12 rule such as:

\[
\text{“4h data always go to expert A.”}
\]

The correct routing variable should come from current information/pattern quality, not merely metadata such as timeframe.

---

# 26. TimeRouter — 2026 quality-driven dynamic routing

TimeRouter is exceptionally relevant to STEP 12.

It addresses multivariate forecasting under varying missing-data patterns and dynamically adjusts fusion structure according to input quality.

Its reported design includes:

- hierarchical expert/fusion space;
- dynamic routing;
- real-time input-quality assessment;
- reduced reliance on badly missing variables.

This is nearly a direct implementation of:

\[
\boxed{
\text{quality state}
\rightarrow
\text{fusion/routing configuration}.
}
\]

Project action:

> before inventing a missingness-driven router, evaluate/reuse TimeRouter concepts.

---

# 27. FreqMoE — 2026 OOD frequency routing

FreqMoE targets time-series forecasting under temporal distribution shift.

It combines:

- masking;
- frequency-domain representation;
- MoE;
- spectrum-embedding-based dynamic routing.

This creates a direct bridge across project steps:

\[
\text{STEP 06 frequency}
+
\text{STEP 11 masking}
+
\text{STEP 12 routing}.
\]

FreqMoE is strong evidence that routing based on current spectral state/OOD conditions is legitimate prior art.

It also reports limitations: OOD evaluation relies on temporal holdout and does not yet provide formal invariance guarantees.

The project should therefore retain controlled synthetic-shift experiments.

---

# 28. Dual-domain dynamic MoE routing — 2026

A 2026 Scientific Reports paper routes through separate time-domain and frequency-domain MoE modules and optimizes routing with reinforcement learning.

It reports:

- dynamic expert specialization;
- routing stability/diversity objectives;
- sparse top-1 activation;
- lower inference memory/latency versus a dense comparator in its benchmark.

This is relevant as an **advanced comparator**.

It does **not** justify beginning STEP 12 with RL.

---

# 29. Why RL routing is late-stage

RL is justified only if routing has genuinely sequential consequences:

\[
a_t
\rightarrow
q_{t+1}
\]

or:

\[
a_t
\rightarrow
\text{future switching/compute/state cost}.
\]

If the best action at time \(t\) depends only on:

\[
q_t
\]

and immediate predictive loss, supervised contextual routing is simpler and statistically easier.

Therefore:

\[
\boxed{
\text{no RL until a myopic router is demonstrably insufficient}.
}
\]

---

# 30. Building-MoE 2026 — closed-loop router-health monitoring

Building-MoE includes a Closed-Loop Routing Scheduler that monitors:

- routing entropy;
- maximum expert share;
- inactive experts;

and adjusts routing conditions to prevent expert imbalance/collapse.

This suggests an important STEP-12 engineering concept:

> the router itself has a health state that must be audited.

Possible telemetry:

\[
[
H(A),
p_{max},
N_{inactive},
switch\ rate
].
\]

---

# 31. Dynamic model selection prior art

Dynamic model-selection work in time-series classification has used meta-learning to choose different models for changing time-series conditions.

This supports a cheap STEP-12 alternative to an end-to-end MoE:

\[
\boxed{
\text{quality/meta-features}
\rightarrow
\text{choose frozen model}.
}
\]

This is much closer to the intended first project implementation.

---

# 32. Core state-of-the-art conclusion

The literature supports at least four different routing levels:

## Level 1 — rule/threshold adaptation

\[
q
\rightarrow
a.
\]

## Level 2 — supervised meta-routing

Predict best expert/configuration.

## Level 3 — differentiable sparse gating

MoE/Pathformer-style.

## Level 4 — sequential/RL routing

Action policy with delayed consequences.

STEP 12 should test them in this order.

---

# 33. Router R0 — static global champion

Select one action:

\[
a^*
=
\arg\min_a
E[L(a)].
\]

Use it for every sample.

This is the mandatory baseline.

If a dynamic router cannot beat the global champion:

\[
\boxed{
\text{adaptive routing is unnecessary}.
}
\]

---

# 34. Router R1 — hindsight oracle

For diagnostic purposes only:

\[
a_t^{oracle}
=
\arg\min_a
L_t(a).
\]

This uses realized target outcome and is **not deployable**.

It estimates the maximum possible value of per-sample routing.

Oracle gain:

\[
\boxed{
G_{oracle}
=
P_{oracle}
-
P_{static}.
}
\]

If:

\[
G_{oracle}
\approx0,
\]

there is little reason to train a router.

This is a crucial early gate.

---

# 35. Oracle action separability

Compute:

\[
P(
a_t^{oracle}=a
\mid
q_t
).
\]

If oracle choices are nearly independent of available quality state:

\[
q_t,
\]

even a large router cannot predict them reliably.

This can close STEP 12 early.

---

# 36. Router R2 — threshold/rule baseline

Example:

\[
a_t
=
\begin{cases}
a_{robust}, & SNR_t<\tau_S\\
a_{independent}, & Q_{sync,t}<\tau_Q\\
a_{safe}, & OOD_t>\tau_O\\
a_{baseline}, & \text{otherwise}.
\end{cases}
\]

Thresholds are selected with train/development data only.

The rules are deliberately interpretable.

---

# 37. Phase-diagram router

For two quality variables such as:

\[
SNR
\]

and:

\[
missingness,
\]

construct empirical regions:

\[
\mathcal R_a
\]

where each action wins.

This produces an adaptive-modulation-like mode map.

Example:

\[
(SNR,missingness)
\rightarrow
a.
\]

This is one of the cleanest communications analogues.

---

# 38. Router R3 — supervised loss predictor

Instead of classifying the best action directly, train for each action:

\[
\hat L_a(q_t)
\approx
L_t(a).
\]

Then choose:

\[
\boxed{
a_t
=
\arg\min_a
\hat L_a(q_t).
}
\]

Recommended low-cost models:

- Ridge;
- logistic/linear;
- shallow decision tree;
- gradient boosting.

This provides interpretable expected-regret estimates.

---

# 39. Why loss prediction may beat best-action classification

If actions have similar performance:

\[
L_1\approx L_2,
\]

a hard best-action label is unstable.

Loss prediction preserves magnitude:

\[
\Delta L.
\]

It can also incorporate cost:

\[
\hat J_a
=
-\hat L_a
-\lambda C_a.
\]

---

# 40. Router R4 — soft gating

Convert predicted utilities into weights:

\[
w_a(q)
=
\frac{
e^{\beta \hat J_a(q)}
}{
\sum_j e^{\beta \hat J_j(q)}
}.
\]

Then combine predictions:

\[
\hat Y
=
\sum_a
w_a
\hat Y_a.
\]

This creates a small mixture-of-experts **without modifying the experts**.

It is a natural bridge between meta-routing and full MoE.

---

# 41. Soft routing cost problem

Soft routing may require executing every expert:

\[
C_{soft}
=
\sum_a C(a),
\]

destroying the compute advantage of conditional routing.

Therefore report:

- prediction quality;
- total executed experts;
- latency.

Sparse top-\(k\) soft routing is a later variant.

---

# 42. Router R5 — learned sparse gate

A small neural gate maps:

\[
q_t
\rightarrow
p(a|q_t).
\]

Select:

\[
top\text{-}k.
\]

This is the first genuine MoE-style controller.

It is eligible only after the smaller meta-router provides evidence that quality-conditioned specialization exists.

---

# 43. Router R6 — RL policy

State:

\[
q_t.
\]

Action:

\[
a_t.
\]

Reward:

\[
r_t
=
-L_t(a_t)
-\lambda_C C(a_t)
-\lambda_Sswitch_t
+\ldots
\]

Only justified after a sequential routing effect is identified.

Do not use trading PnL reward to answer the initial STEP-12 representation-routing question.

---

# 44. Router training leakage problem

If expert \(a\) is trained on sample \(t\), its in-sample loss:

\[
L_t(a)
\]

is optimistically biased.

Training a router from those losses creates leakage.

Therefore routing labels/utilities must be generated from:

\[
\boxed{
\text{out-of-fold / rolling-origin expert predictions}.
}
\]

---

# 45. Chronological cross-fitting for router labels

Recommended procedure inside training data:

1. partition training chronology into \(K\) forward blocks;
2. for each block \(k\):
   - fit each eligible expert/configuration on earlier training history;
   - predict block \(k\);
   - store:
     \[
     q_t,
     \hat y_{a,t},
     L_t(a);
     \]
3. concatenate out-of-fold routing table;
4. train router on this table.

Thus:

\[
q_t
\rightarrow
L_t(a)
\]

uses genuinely OOS expert behavior.

---

# 46. Validation protocol

Validation is used to select:

- router class;
- quality-feature subset;
- thresholds;
- switching penalty;
- top-\(k\).

The expert/configuration bank itself is frozen before STEP-12 router selection.

Do not retune upstream experts jointly with the router during the first protocol.

---

# 47. Test protocol

After router choice is frozen:

- every expert stays frozen;
- quality-state computation stays frozen;
- router stays frozen.

Run test once.

No threshold change after test inspection.

---

# 48. Synthetic benchmark A — SNR-dependent mode superiority

Generate a forecasting/detection problem where:

- high-resolution/raw expert is best at high SNR;
- robust/smoothed expert is best at low SNR.

Sweep:

\[
SNR.
\]

Verify the router recovers the correct mode transition.

This is the most direct AMC analogue.

---

# 49. Synthetic benchmark B — missingness-driven mode superiority

Create multivariate data where:

- cross-channel expert is best when all channels exist;
- channel-independent expert is best under severe missingness.

Quality variable:

\[
m_t
=
\text{missing fraction}.
\]

This directly mirrors TimeRouter-style adaptation.

---

# 50. Synthetic benchmark C — synchronization lock

Construct:

- stable lead–lag state;
- unlocked/uncertain lag state.

Modes:

- lag-aware cross-series model;
- channel-independent baseline.

Test whether:

\[
Q_{sync}
\]

correctly gates the lag-aware mode.

---

# 51. Synthetic benchmark D — OOD shift

Train on one regime.

Create OOD regimes where different frozen experts degrade differently.

Router input includes:

\[
OOD_t.
\]

Test:

\[
\text{OOD-gated fallback}
\]

versus:

\[
\text{always use global champion}.
\]

---

# 52. Synthetic benchmark E — useless quality state

Construct a dataset where:

\[
a_t^{oracle}
\]

is independent of \(q_t\).

Expected:

\[
\boxed{
\text{router should not beat static champion materially}.
}
\]

This is a critical negative control.

---

# 53. Synthetic benchmark F — route chatter

Create quality state fluctuating around a threshold.

Compare:

- no hysteresis;
- hysteresis/switch penalty.

Measure:

\[
N_{switch}
\]

and performance.

---

# 54. Public time-series benchmark

STEP 12 does not need a huge foundation-model campaign.

Recommended subset:

- ETTh1;
- ETTm1;
- Weather;
- Electricity.

These support comparison with:

- Pathformer;
- MoE literature;
- frequency/OOD routing literature.

Missingness stress can be applied synthetically where appropriate.

---

# 55. TimeRouter-style missingness benchmark

For clean multivariate benchmark, create train/validation/test missingness stress levels:

\[
m
\in
\{0,0.1,0.3,0.5\}.
\]

Important:

- primary natural clean test remains separate;
- artificial missingness is labeled stress;
- router does not see future missingness.

---

# 56. OOD benchmark

Use controlled shifts rather than only chronological holdout.

Examples:

- scale shift;
- variance shift;
- frequency shift;
- channel dropout;
- noise/SNR shift.

This addresses a limitation acknowledged by FreqMoE's naturalistic temporal-holdout OOD evaluation.

---

# 57. Project financial experiment — only after upstream steps produce eligible modes

The project-specific STEP-12 experiment cannot be fully specified until STEPS 03–11 have empirical verdicts.

The configuration bank is constructed from their validated outcomes.

A hypothetical eligible set may look like:

\[
\mathcal A
=
\{
a_{base},
a_{robust},
a_{spectral},
a_{cross},
a_{safe}
\}
\]

but any action whose source STEP failed its gate is removed.

---

# 58. Quality-state ablation

Begin with:

\[
q^{(0)}
=
[
SNR,U,missing,OOD,Q_{sync}
].
\]

Then add one family at a time:

\[
+surprisal
\]

\[
+regime\ entropy
\]

\[
+detector\ confidence.
\]

Report incremental router value.

---

# 59. No raw price as a router shortcut initially

The first router should consume quality/state variables, not the entire raw market window.

Reason:

> otherwise the router becomes a second forecasting model and the AMC hypothesis becomes unidentifiable.

After quality-routing value is established, contextual raw embeddings can be considered.

---

# 60. Router complexity ladder

Recommended:

## R2

rules.

## R3a

Ridge/logistic.

## R3b

small decision tree.

## R3c

gradient boosting.

## R5

small MLP gate.

No Transformer router in the first pass.

---

# 61. Static versus adaptive fairness

All static and dynamic comparisons must have access to the same configuration bank.

Static baseline chooses:

\[
a^*.
\]

Adaptive router chooses:

\[
a_t.
\]

Do not give the adaptive system extra experts unavailable to the static baseline.

---

# 62. Compute-matched routing

If router adds cost:

\[
C_{router},
\]

report total:

\[
C_{total}
=
C_{router}
+
C(a_t).
\]

Compare with static mode at:

- equal accuracy;
- equal compute;
- Pareto frontier.

---

# 63. Risk–coverage evaluation

If safe/abstain mode is allowed, define coverage:

\[
coverage
=
\frac{
N_{\text{active predictions}}
}{
N
}.
\]

Risk on covered set:

\[
R(c).
\]

Plot:

\[
\boxed{
risk
\leftrightarrow
coverage.
}
\]

This is more informative than accuracy alone.

---

# 64. Router regret

Oracle utility:

\[
J_t^{oracle}
=
\max_a J_t(a).
\]

Router regret:

\[
\boxed{
Regret
=
\frac1N
\sum_t
[
J_t^{oracle}
-
J_t(a_t)
].
}
\]

This is a core STEP-12 metric.

---

# 65. Routing accuracy is secondary

Best-action accuracy:

\[
Acc_{route}
=
P(
a_t
=
a_t^{oracle}
).
\]

But two actions may have almost identical losses.

Therefore regret is more meaningful than exact action match.

---

# 66. Specialization gain

Define:

\[
G_{spec}
=
P_{adaptive}
-
P_{static}.
\]

Compare with oracle specialization:

\[
G_{oracle}.
\]

Efficiency ratio:

\[
\boxed{
\eta_{route}
=
\frac{
G_{spec}
}{
G_{oracle}
}
}
\]

when denominator is positive.

This measures how much available specialization value the router captures.

---

# 67. Expert utilization

Report:

\[
p(a)
\]

for every mode.

Also report utilization conditional on quality bins:

\[
p(a|SNR\ bin),
\]

\[
p(a|OOD\ bin),
\]

etc.

This reveals whether routing semantics make sense.

---

# 68. Router calibration

If the router outputs:

\[
\hat P(
a=a^{oracle}|q
),
\]

or predicted losses, evaluate calibration.

For predicted loss:

\[
\hat L_a
\]

compare residual:

\[
L_a-\hat L_a.
\]

Overconfident routers are especially dangerous under OOD.

---

# 69. OOD fallback

Define conservative policy:

\[
a_t
=
a_{safe}
\]

when:

\[
OOD_t>\tau_O
\]

and router confidence is low.

Compare against unrestricted learned routing.

This is a primary safety ablation.

---

# 70. Confidence disagreement

If quality signals conflict:

- high SNR;
- high OOD;
- low sync lock;

the router should not rely on one scalar “quality score” unless validated.

Preserve a vector:

\[
q_t
\]

and let the routing rule/model learn trade-offs.

---

# 71. Hysteresis

For discrete mode map, define separate enter/exit thresholds.

Example:

\[
SNR<\tau_{enter}
\]

switches to robust mode, but return requires:

\[
SNR>\tau_{exit}
\]

with:

\[
\tau_{exit}>\tau_{enter}.
\]

This reduces chatter.

---

# 72. Minimum dwell time

Optional:

\[
dwell(a)
\geq
D_{min}.
\]

Use only if frequent mode switching has actual cost.

Not a first-line constraint.

---

# 73. Route health telemetry

Persist:

- action;
- action probability;
- predicted loss;
- quality vector;
- switch flag;
- router confidence;
- route entropy;
- OOD status;
- fallback status.

This makes dynamic routing auditable.

---

# 74. Falsifiable hypotheses

## H12.1 — One global configuration is not always optimal

There exist controlled information-quality regimes where:

\[
a_t^{oracle}
\]

changes systematically with:

\[
q_t.
\]

**Falsified if:** one mode dominates nearly every sample/regime.

---

## H12.2 — A causal quality-aware router can improve over the static global champion

\[
\boxed{
P_{router}
>
P_{static}
}
\]

with the same frozen action bank.

**Falsified if:** adaptive selection gives no stable OOS improvement.

---

## H12.3 — The quality state predicts relative expert performance

\[
I(
q_t;
a_t^{oracle}
)
>
0
\]

or equivalent predictive evidence.

**Falsified if:** router cannot predict relative losses beyond chance/baseline.

---

## H12.4 — Simple rules capture a meaningful fraction of available specialization

A threshold/rule router captures:

\[
\eta_{route}>0
\]

relative to oracle.

**Falsified if:** simple quality thresholds have no value.

---

## H12.5 — Learned routing is justified only if it beats rule-based routing

\[
P_{learned}
>
P_{rules}
\]

at matched cost.

**Falsified if:** complex gate adds no value.

---

## H12.6 — Missingness-driven routing improves robustness

When cross-channel information availability degrades:

\[
P_{quality-router}
>
P_{static-cross-channel}.
\]

**Falsified if:** missingness state does not help mode selection.

---

## H12.7 — OOD-gated fallback reduces negative-transfer risk

Under controlled OOD:

\[
Risk_{gated}
<
Risk_{ungated}
\]

at declared coverage/cost.

**Falsified if:** fallback provides no safety benefit.

---

## H12.8 — Synchronization confidence should gate lag-sensitive modes

Low:

\[
Q_{sync}
\]

should reduce routing to synchronization-dependent experts.

**Falsified if:** sync quality is unrelated to relative performance.

---

## H12.9 — Router hysteresis can reduce switching without materially harming prediction

\[
N_{switch}^{hyst}
<
N_{switch}^{raw}
\]

while:

\[
P^{hyst}
\approx
P^{raw}.
\]

**Falsified if:** hysteresis destroys useful adaptation or does not reduce chatter.

---

## H12.10 — Sparse/conditional routing can improve the compute–accuracy frontier

There exists a router configuration such that:

\[
C_{adaptive}
<
C_{dense/all}
\]

with comparable or superior \(P\).

**Falsified if:** dynamic routing never improves the Pareto frontier.

---

## H12.11 — RL routing is unnecessary when routing reward is myopic

If:

\[
J_t(a)
\]

depends only on current state/action, supervised/meta-routing should match RL.

**Falsified if:** RL yields reproducible benefit despite no meaningful sequential dependency, requiring investigation.

This is deliberately a restraint hypothesis.

---

## H12.12 — No universal quality metric is sufficient

Different failure states require different signals:

- SNR;
- missingness;
- OOD;
- uncertainty;
- sync confidence.

**Falsified if:** one scalar quality indicator consistently matches the full-vector router across all controlled shift classes.

---

# 75. Gate 12A — action-bank heterogeneity

Before training any router:

\[
G_{oracle}
\]

must be measurably positive.

If one action globally dominates:

\[
\boxed{
\text{STOP.}
}
\]

Dynamic routing is unnecessary.

---

# 76. Gate 12B — quality-state predictability

Quality features must predict relative action loss better than a constant/global baseline.

If not:

\[
\boxed{
\text{STOP learned routing.}
}
\]

The oracle advantage is not causally/operationally accessible from the allowed state.

---

# 77. Gate 12C — rule router

A preregistered/simple rule or small tree must show stable value before large gates are considered.

If rules fail and learned gate also cannot clearly exceed them, STEP 12 closes null.

---

# 78. Gate 12D — learned-router incremental value

A learned router is promoted only if it exceeds:

- static champion;
- rule router;

under matched action bank and declared compute.

---

# 79. Gate 12E — OOD/fallback safety

Under controlled OOD/missingness/sync-failure stress:

- catastrophic negative transfer must not increase;
- fallback behavior must be auditable.

---

# 80. Gate 12F — route stability

Router must not exhibit unexplained collapse/chatter.

Report:

- utilization entropy;
- maximum expert share;
- switch rate.

There is no requirement for perfectly balanced utilization.

---

# 81. Gate 12G — held-out confirmation

Only one frozen routing policy reaches final test.

No test-driven threshold edits.

---

# 82. Synthetic experiment matrix

| ID | Quality change | Static alternatives | Router input | Expected test |
|---|---|---|---|---|
| A00 | none | multiple | none | static should win/tie |
| A01 | SNR | raw vs robust | SNR | mode transition |
| A02 | missingness | mixed vs independent | missingness | quality routing |
| A03 | sync lock | lag-aware vs independent | \(Q_{sync}\) | gated relation |
| A04 | OOD | specialized vs safe | OOD | fallback |
| A05 | multi-factor | 3–4 modes | full \(q\) | learned > single threshold |
| A06 | chatter | 2 modes | noisy state | hysteresis |

---

# 83. Router benchmark matrix

| Router | Deployable | Learns | Cost | Role |
|---|---:|---:|---:|---|
| Static champion | Yes | No | Minimal | mandatory baseline |
| Hindsight oracle | No | No | diagnostic | upper bound |
| Threshold rules | Yes | thresholds | tiny | first adaptive baseline |
| Decision tree | Yes | Yes | tiny | interpretable meta-router |
| GBDT | Yes | Yes | low | strong tabular router |
| Loss-predictor ensemble | Yes | Yes | low | expected-regret router |
| Soft gate | Yes | Yes | potentially high expert cost | bridge |
| Sparse MLP gate | Yes | Yes | low | MoE-style |
| RL policy | Yes | Yes | higher | last-stage only |

---

# 84. Public benchmark protocol

Recommended:

- ETTh1;
- ETTm1;
- Weather;
- Electricity.

Configurations can be intentionally constructed from:

- raw baseline;
- robust/masked variant;
- frequency branch;
- channel-independent/mixed mode.

The purpose is routing validation, not SOTA model competition.

---

# 85. Project benchmark protocol

Use project chronology:

\[
4y\ train
+
1y\ validation
+
1y\ test.
\]

Router training labels come from **internal training cross-fit**, not validation/test.

Quality-state computation uses training-fitted statistics.

---

# 86. Financial routing mode examples

These are illustrative only.

## High-quality synchronized state

Potentially allow:

- higher-resolution spectral/phase information;
- cross-series lag-aware branch.

## Low-SNR state

Potentially prefer:

- robust/denoised representation;
- robust STEP-11 latent.

## Missing/unlocked cross-series state

Potentially prefer:

- channel-independent/raw local branch.

## High-OOD/uncertainty state

Potentially route to:

- safe baseline;
- abstain/no-new-signal.

Actual mode contents are determined only by upstream empirical results.

---

# 87. Horizon-specific routing

Different forecast horizons may prefer different modes.

Option:

\[
a_{t,h}.
\]

But first STEP-12 experiment should prefer one common action per timestamp unless upstream evidence strongly supports horizon-specific routing.

Otherwise the action space grows rapidly.

---

# 88. Routing frequency

Do not assume router must decide every bar.

Compare:

- every timestamp;
- regime-change only;
- fixed refresh period.

Frequent routing is only useful if quality state changes sufficiently fast.

---

# 89. Quality-state smoothing

Some state estimates are noisy.

Possible causal smoothing:

\[
\tilde q_t
=
\alpha q_t
+
(1-\alpha)\tilde q_{t-1}.
\]

Smoothing is optional and must be compared against raw quality signals.

It must not hide rapid genuine failures.

---

# 90. Tail/event routing

Report routing behavior during:

- high-volatility windows;
- large-return windows;
- macro events if available;
- synchronization loss;
- OOD spikes.

A router that improves average loss by making poor decisions during rare important states can be unacceptable.

---

# 91. Mode-switch attribution

For every switch:

\[
a_{t-1}
\rightarrow
a_t
\]

store the dominant quality variables that changed.

For rule/tree router this is transparent.

For neural gates use:

- feature attribution;
- local sensitivity

only as diagnostics.

---

# 92. Action-bank versioning

Each action should have immutable identifier:

```text
mode_id
upstream_step_versions
artifact_hashes
compute_cost
input_contract
output_contract
```

The router selects versions, not mutable labels such as “best spectral model.”

---

# 93. No hidden retraining

Router decision:

\[
a_t
\]

must not trigger automatic expert retraining in the first protocol.

That would mix STEP 12 with TTA/curriculum/model management.

---

# 94. No route to unavailable data

If a mode requires:

- cross-series feature \(X_j\);
- synchronization lock;
- event stream;

and that requirement is unavailable, the action is masked:

\[
a\notin\mathcal A_t.
\]

The router must honor action feasibility.

---

# 95. Feasible-action mask

Define:

\[
M_t(a)\in\{0,1\}.
\]

Selection:

\[
a_t
=
\arg\max_{a:M_t(a)=1}
\hat J_a(q_t).
\]

This is better than hoping the router learns not to select impossible modes.

---

# 96. Failure modes

## 96.1. Oracle illusion

Oracle gain exists but cannot be predicted from causal quality state.

## 96.2. Router becomes second forecaster

Feeding full raw window into gate obscures the hypothesis.

## 96.3. Expert overfit leakage

Router labels built from in-sample expert errors.

## 96.4. Route collapse

One expert selected regardless of state.

## 96.5. Forced load balancing

Artificially uses weak experts merely to increase entropy.

## 96.6. Chattering

Frequent unstable switching.

## 96.7. OOD overconfidence

Router becomes most confident exactly where state is unfamiliar.

## 96.8. Compute laundering

“Sparse” router actually executes every expert.

## 96.9. Upstream-failure routing

Failed STEP modules remain in action bank.

## 96.10. Test threshold tuning

Route thresholds edited after test.

## 96.11. Future-target quality feature

Unmatured outcome leaks into router state.

## 96.12. Step-13 overlap

Router jointly optimizes continuous branch budgets instead of discrete modes.

---

# 97. Audit checklist

- [ ] Every action comes from a qualified upstream configuration.
- [ ] Static global champion is included.
- [ ] Hindsight oracle is diagnostic only.
- [ ] Oracle specialization gain is measured before router training.
- [ ] Router inputs are available causally at decision time.
- [ ] Expert losses used to train router are out-of-fold/rolling-origin.
- [ ] First router uses quality/meta-state, not full raw market window.
- [ ] Rule/threshold baseline precedes learned gate.
- [ ] Learned gate must beat rule baseline.
- [ ] RL is not used unless delayed routing effects are documented.
- [ ] Action feasibility mask is explicit.
- [ ] Route utilization and collapse are logged.
- [ ] Switching rate is logged.
- [ ] OOD fallback is tested.
- [ ] Missingness stress is separate from clean primary evaluation.
- [ ] Safe/abstain mode risk–coverage is reported when applicable.
- [ ] Compute includes router + executed experts.
- [ ] Soft gating does not hide all-expert execution cost.
- [ ] Tail/event routing is audited.
- [ ] Upstream expert parameters remain frozen.
- [ ] STEP 13 resource allocation is not folded into STEP 12.
- [ ] Test is used once after router freeze.

---

# 98. Reuse matrix

| Requirement | Prior art | STEP-12 action |
|---|---|---|
| Adaptive channel-state modes | Goldsmith/Chua AMC | Historical/theoretical basis |
| Simple CSI thresholds | classical AMC | mandatory rule baseline |
| Conditional compute | ACT | conceptual baseline |
| Sparse expert routing | Shazeer MoE | reference |
| Simple sparse top-1 routing | Switch Transformer | router-stability lessons |
| Adaptive temporal-scale pathways | Pathformer | benchmark/reference |
| Sparse TS experts | Time-MoE | reference; no large model needed |
| Token-level TS specialization | Moirai-MoE | important specialization prior |
| Missingness quality routing | TimeRouter | high-priority comparator |
| OOD spectral routing | FreqMoE | STEP-06/11/12 bridge |
| Dynamic time/frequency routing | Ji et al. 2026 | advanced comparator |
| Closed-loop router health | Building-MoE | telemetry/reference |
| Reject/fallback | SelectiveNet | safety reference |
| Statistical risk control | Conformal Risk Control | later risk-guarantee layer |
| Dynamic model selection | 2024 meta-learning model-selection work | cheap router precedent |

---

# 99. Recommended implementation order

1. Freeze eligible action bank.
2. Generate cross-fitted per-action training losses.
3. Compute static global champion.
4. Compute hindsight oracle and \(G_{oracle}\).
5. Test whether IQS predicts oracle/relative losses.
6. Build threshold/rule router.
7. Build shallow decision tree / GBDT loss router.
8. Add action feasibility mask.
9. Add OOD fallback.
10. Add hysteresis only if chatter appears.
11. Evaluate clean public/project validation.
12. Evaluate controlled SNR/missingness/OOD stress.
13. Only if warranted, test sparse neural gate.
14. Only if sequential action consequences are proven, test RL router.
15. Freeze one router for held-out test.

---

# 100. Decision logic

If:

\[
G_{oracle}
\approx0,
\]

verdict:

\[
\boxed{
\text{one static mode is sufficient}.
}
\]

If oracle gain is high but quality state cannot predict it:

\[
\boxed{
\text{specialization exists but is not operationally routable from current IQS}.
}
\]

If rules beat static:

\[
\boxed{
\text{adaptive link-style routing is supported}.
}
\]

If learned router beats rules:

\[
\boxed{
\text{nonlinear/contextual routing is justified}.
}
\]

If a large MoE/RL router adds no gain:

\[
\boxed{
\text{retain simple router}.
}
\]

---

# 101. Recommended artifacts

1. `step12_action_bank.json`
2. `step12_information_quality_state.parquet`
3. `step12_crossfit_expert_losses.parquet`
4. `step12_static_champion.json`
5. `step12_oracle_routing.parquet`
6. `step12_oracle_gain.json`
7. `step12_rule_router.json`
8. `step12_loss_router_model.*`
9. `step12_route_trace.parquet`
10. `step12_route_utilization.parquet`
11. `step12_switching_metrics.parquet`
12. `step12_risk_coverage.parquet`
13. `step12_compute_frontier.parquet`
14. `step12_ood_stress.parquet`
15. `step12_statistical_tests.json`
16. `step12_audit_report.md`
17. reproducibility manifest:
    - action artifact hashes;
    - upstream STEP versions;
    - quality-feature definitions;
    - causal availability rules;
    - cross-fitting blocks;
    - router seed;
    - thresholds;
    - switching policy;
    - compute measurements.

---

# 102. Project code-location recommendation if later implemented

The router spans multiple validated pipeline alternatives, so it should **not** live inside:

- `feature-extractor`;
- one preprocessor plugin;
- one detector plugin.

It belongs at the orchestration/fusion boundary that already knows which representation/predictor mode is available.

The exact repository location should be decided only after the experiment defines the minimum action contract.

No code move is required by this protocol.

---

# 103. State-of-the-art synthesis

The literature strongly supports the general adaptive-routing concept but changes how the project should implement it.

## 103.1. Adaptive routing is established

Communications has used channel-state-based mode switching for decades.

## 103.2. Conditional computation is established

ACT, MoE and Switch show that computation/model parameters can be input-dependent.

## 103.3. Time-series routing is now explicit

Pathformer, Time-MoE and Moirai-MoE demonstrate sample/token-dependent specialization.

## 103.4. Quality-driven time-series routing now exists

TimeRouter routes based on missing-data quality.

FreqMoE routes according to spectral embeddings under OOD shifts.

## 103.5. Dynamic RL routing exists but is not the default

2026 dual-domain MoE work shows RL routing can be useful, but it is much more complex and should be treated as an advanced comparator.

## 103.6. Safe fallback has its own literature

Selective prediction and conformal risk control provide stronger foundations than ad-hoc “confidence < 0.5” rules.

---

# 104. Project-specific research opportunity

The potential contribution is not:

> “use Mixture of Experts for time series.”

That already exists.

The stronger project question is:

\[
\boxed{
\text{Can the explicit information-quality measurements produced by STEPS 03–11 drive a small causal router that selects among independently validated processing modes?}
}
\]

That is more interpretable and experimentally cleaner than learning routing directly from raw inputs.

---

# 105. General-domain interpretation

For any multisource sequential application:

\[
q_t
=
[
\text{signal quality},
\text{uncertainty},
\text{missingness},
\text{OOD},
\text{synchronization},
\text{detector confidence}
].
\]

Then:

\[
a_t
=
\pi(q_t).
\]

Examples:

## Industrial sensing

sensor degradation chooses robust/local modes.

## Biomedical monitoring

missingness/uncertainty chooses safe or multimodal modes.

## Energy

weather/data quality routes scale-specific experts.

## Robotics

sensor confidence routes visual/inertial/localization modes.

## Finance

SNR/OOD/regime/sync state routes robust, spectral, cross-asset or safe representations.

The domain changes the IQS estimators and action bank, not the controller contract.

---

# 106. References — IEEE style

[1] A. J. Goldsmith and S.-G. Chua, “Variable-Rate Variable-Power MQAM for Fading Channels,” *IEEE Transactions on Communications*, vol. 45, no. 10, pp. 1218–1230, Oct. 1997, doi: 10.1109/26.634685. Available: https://doi.org/10.1109/26.634685

[2] S. G. Chua and A. J. Goldsmith, “Adaptive Coded Modulation for Fading Channels,” in *Proc. IEEE International Conference on Communications (ICC)*, 1997, pp. 1488–1492, doi: 10.1109/ICC.1997.595036. Available: https://doi.org/10.1109/ICC.1997.595036

[3] P. S. Chow, J. M. Cioffi, and J. A. C. Bingham, “A Practical Discrete Multitone Transceiver Loading Algorithm for Data Transmission over Spectrally Shaped Channels,” *IEEE Transactions on Communications*, vol. 43, pp. 773–775, 1995, doi: 10.1109/26.380108. Available: https://doi.org/10.1109/26.380108  
**Boundary note:** fine-grained bit/resource loading belongs primarily to STEP 13.

[4] Z. Cooper-Baldock, E. Panteli, and P. E. Santos, “A State-of-the-Art Survey and Benchmarking of Adaptive Modulation and Coding for Underwater Acoustic Communications,” *Applied Ocean Research*, vol. 170, art. 105018, 2026, doi: 10.1016/j.apor.2026.105018. Available: https://doi.org/10.1016/j.apor.2026.105018

[5] A. Graves, “Adaptive Computation Time for Recurrent Neural Networks,” arXiv:1603.08983, 2016. Available: https://arxiv.org/abs/1603.08983

[6] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. Hinton, and J. Dean, “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” in *International Conference on Learning Representations (ICLR)*, 2017. Available: https://arxiv.org/abs/1701.06538

[7] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022. Available: https://www.jmlr.org/papers/v23/21-0998.html

[8] Y. Geifman and R. El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option,” in *Proceedings of the 36th International Conference on Machine Learning*, PMLR vol. 97, pp. 2151–2159, 2019. Available: https://proceedings.mlr.press/v97/geifman19a.html

[9] A. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal Risk Control,” in *International Conference on Learning Representations (ICLR)*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html

[10] M. Jazbec, P. Forré, S. Mandt, D. Zhang, and E. Nalisnick, “Early-Exit Neural Networks with Nested Prediction Sets,” in *Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence*, PMLR vol. 244, pp. 1780–1796, 2024. Available: https://proceedings.mlr.press/v244/jazbec24a.html

[11] P. Chen, Y. Zhang, Y. Cheng, Y. Shu, Y. Wang, Q. Wen, B. Yang, and C. Guo, “Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting,” in *International Conference on Learning Representations (ICLR)*, 2024. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html

[12] X. Shi, S. Wang, Y. Nie, D. Li, Z. Ye, Q. Wen, and M. Jin, “Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts,” in *International Conference on Learning Representations (ICLR)*, 2025. Available: https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html

[13] X. Liu, J. Liu, G. Woo, T. Aksu, Y. Liang, R. Zimmermann, C. Liu, J. Li, S. Savarese, C. Xiong, and D. Sahoo, “Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts,” in *Proceedings of the 42nd International Conference on Machine Learning*, PMLR vol. 267, pp. 38940–38962, 2025. Available: https://proceedings.mlr.press/v267/liu25an.html

[14] Q. Hua, Y. Chen, C. Dong, Y. Zhang, and L. Xu, “TimeRouter: A Unified Dynamic Routing Framework for Handling Missing Data in Time Series Forecasting,” *Knowledge-Based Systems*, vol. 341, art. 115782, 2026, doi: 10.1016/j.knosys.2026.115782. Available: https://doi.org/10.1016/j.knosys.2026.115782

[15] A. Shen, Z. Lai, and T. Wang, “FreqMoE: Robust Time Series Forecasting via Frequency-Domain Mixture of Experts for Out-of-Distribution Scenarios,” *Electronics*, vol. 15, no. 13, art. 2865, 2026, doi: 10.3390/electronics15132865. Available: https://doi.org/10.3390/electronics15132865

[16] Q. Ji, J. Wang, H. He, S. He, et al., “Learning to Route in Time and Frequency Domains: A Dual-Domain MoE Transformer for Multi-Horizon Forecasting,” *Scientific Reports*, vol. 16, art. 21574, 2026, doi: 10.1038/s41598-026-50232-8. Available: https://doi.org/10.1038/s41598-026-50232-8

[17] X. Liu, Q. Fu, J. Chen, K. Liu, L. Liu, Y. Wang, and Y. Lu, “Building-MoE: A Closed-Loop Routing Sparse Mixture-of-Experts Time-Series Foundation Model for Building Short-Term Load Forecasting,” *Building Simulation*, vol. 19, pp. 1007–1030, 2026, doi: 10.1007/s12273-026-1403-6. Available: https://doi.org/10.1007/s12273-026-1403-6

[18] “Dynamic Selection of Machine Learning Models for Time-Series Data,” *Information Sciences*, vol. 665, art. 120360, 2024, doi: 10.1016/j.ins.2024.120360. Available: https://doi.org/10.1016/j.ins.2024.120360

---

# 107. Verified project resources relevant to IQS

## `harveybc/predictor`

Existing predictor/pipeline code exposes predictive uncertainty through `predict_with_uncertainty(...)` and configurable MC sampling.

## `harveybc/agent-multi`

Existing state-profile plumbing includes regime probability columns and `state_regime_entropy` in tested feature contracts.

These existing signals should be consumed before creating duplicate uncertainty/regime estimators.

---

# 108. Final status

**STEP 12 is theoretically specified after a dedicated state-of-the-art review and is ready for independent agent audit.**

The central falsifiable idea is:

\[
\boxed{
\text{information quality}
\rightarrow
\text{pipeline mode}
}
\]

not:

\[
\boxed{
\text{raw input}
\rightarrow
\text{giant new MoE}.
}
\]

The first decisive quantity is not model accuracy.

It is:

\[
\boxed{
G_{oracle}
=
P_{oracle}
-
P_{static}.
}
\]

If the hindsight oracle cannot beat one global mode, adaptive routing has no job to do.

If the oracle can beat it but causal quality state cannot predict which mode wins, the specialization is not operationally usable.

Only if both conditions pass should increasingly sophisticated routers be tested.

After STEP 12 audit/experimentation, proceed to:

**STEP 13 — Multiplexing / MIMO / optimal allocation of finite information/model capacity across active branches.**
