# FINAL MASTER REVIEW
## From Multivariate Time-Series Information to Usable Knowledge
### State-of-the-Art Audit, Autocritique, Cross-Domain Architecture, Causal Event Lane, and Level-3 Meta-Optimization

**Date:** 2026-09-05  
**Status:** Final consolidation/review document for independent agent audit  
**Scope:** STEPS 01–13, compression transversal lane, project ML repositories, economic-event database, causal-inference capability, simulation/backtesting, and Level-3 meta-optimization.  
**Normative patches:** `WORKPLAN_PATCH_001`, `WORKPLAN_PATCH_002`, and the accompanying `WORKPLAN_PATCH_003_FINAL_REVIEW_CAUSAL_EVENTS_METAOPT.md`.

---

# 0. Executive conclusion

After reviewing the full chain, current literature, and the newly clarified project capabilities, the strongest version of the project is **not** a linear preprocessing pipeline in which every STEP is permanently enabled.

It is a **modular evidence-driven information-processing system**:

\[
\boxed{
\text{raw observations}
\rightarrow
\text{quality/evidence characterization}
\rightarrow
\text{candidate representations}
\rightarrow
\text{candidate detectors/decompositions}
\rightarrow
\text{validated branches}
\rightarrow
\text{task-specific heads}
}
\]

with two additional cross-cutting intelligence layers:

\[
\boxed{
\text{causal event-response analysis}
}
\]

and

\[
\boxed{
\text{Level-3 meta-optimization from historical experiment evidence}.
}
\]

The communications analogy remains useful because it provides a disciplined sequence of questions:

1. what can be observed?
2. what is noise?
3. what resolution is justified?
4. what redundancy exists?
5. what representation exposes structure?
6. what patterns can be detected?
7. what distortions can be compensated?
8. what information is shared or interfering?
9. what is temporally misaligned?
10. what redundancy improves robustness?
11. when should the operating mode change?
12. how should finite resources be allocated?

But the analogy must remain **functional, not literal**.

The final project should never assume:

\[
\boxed{
\text{communications block}
=
\text{mandatory ML block}.
}
\]

Every block remains conditional on empirical gates.

---

# 1. Final architectural thesis

The strongest generalizable architecture is:

\[
X_t
\rightarrow
Q_t
\rightarrow
\{R_i(X_{\le t})\}
\rightarrow
\{D_i(R_i)\}
\rightarrow
\{Z_i\}
\rightarrow
C
\rightarrow
Z
\rightarrow
\{H_j\}
\]

where:

- \(X_t\): raw multivariate observations;
- \(Q_t\): information-quality/evidence state;
- \(R_i\): representation transforms;
- \(D_i\): representation-matched detectors/decomposers;
- \(Z_i\): branch latent representations;
- \(C\): fusion/core;
- \(Z\): consolidated task representation;
- \(H_j\): task heads.

Cross-cutting planes add:

\[
E_t
=
\text{event/causal-response context}
\]

and:

\[
M
=
\text{meta-knowledge from historical optimization/evaluation runs}.
\]

Thus the complete conceptual system is:

\[
\boxed{
(X_t,E_t,M)
\rightarrow
\text{evidence-conditioned representation system}
\rightarrow
\text{task knowledge}.
}
\]

---

# 2. Final governance principle

The unit of progress is **not implementation count**.

The unit of progress is a falsified or surviving hypothesis.

A STEP can close with:

- `SUPPORTED`;
- `NULL / NO VALUE`;
- `HARMFUL`;
- `UNIDENTIFIABLE`;
- `DEFERRED BY PREREQUISITE`.

A null STEP is scientific progress.

---

# 3. Repository capability map

The project ecosystem is now broad enough that the final architecture should **reuse existing specialized repositories rather than creating another monolith**.

Observed project capabilities include:

- `predictor`: regression/forecasting, binary/directional variants, multi-branch models, Bayesian uncertainty;
- `feature-extractor`: ANN/CNN/LSTM/Transformer/VAE encoder-decoder representation learning;
- `preprocessor` / `feature-eng`: data transformations and feature engineering;
- `causal-inference`: experimental NOTEARS / Invariant Causal Prediction / DoWhy work;
- `gym-fx`, RL/DOIN stack: reinforcement-learning simulation/evaluation;
- `synthetic-datagen`, `timeseries-gan`, VAE/GAN-related repos: synthetic-data work;
- DEAP/NEAT/DOIN optimization infrastructure;
- result/evidence infrastructure and metric persistence;
- event-token Transformer work in `agent-multi`;
- event/economic-calendar data in the broader financial-data pipeline;
- backtesting/heuristic strategy/live-serving layers.

The final system therefore needs **contracts between capabilities**, not duplicated algorithms.

---

# 4. Task-assay matrix

Each ML paradigm should answer the question it is best suited to answer.

| Capability | Primary role in this research |
|---|---|
| Regression/forecasting | Main assay for future-value utility |
| Classification | Pattern/event/rush/onset detector assay |
| Unsupervised learning | Regime, motif, redundancy and clustering diagnostics |
| Causal inference | Treatment/event-effect estimation under explicit identification assumptions |
| Autoencoder/representation learning | Compress/expose structure for downstream tasks |
| RL | Late downstream action-policy utility; never first proof of preprocessing value |
| Simulation/synthetic data | Ground-truth identifiability/falsification |
| Backtesting | Late application utility after predictive validity |
| DEAP/NEAT | Level-2 search over candidate configurations |
| Level-3 meta-learning | Warm-start/recommend Level-2 search from accumulated experiment history |

No one paradigm should be asked to substitute for all others.

---

# 5. Autocritique: the largest conceptual risk in the original roadmap

The early roadmap risked becoming:

\[
\text{STEP 1}
\rightarrow
\text{STEP 2}
\rightarrow
\cdots
\rightarrow
\text{STEP 13}
\]

as if all transforms should be serially stacked.

That would be a mistake.

Many STEPS are **alternative views or decision layers**, not serial filters.

The improved structure is a directed acyclic research graph:

```text
sampling / availability
        ↓
noise-quality characterization
   ↙             ↘
denoising      resolution/quantization
   ↓              ↓
source/context modeling
        ↓
representation bank
        ↓
detector/decomposition bank
   ↙      ↓       ↘
equalize shared/private synchronize
      \     |      /
        validated branches
              ↓
        robust extractor
              ↓
        adaptive router
              ↓
        budget allocator
              ↓
        task heads
```

Economic-event causal evidence and Level-3 meta-optimization act across this graph.

---

# 6. STEP 01 — Sampling / Nyquist
## Final verdict: KEEP, but strengthen the observation model

The classical Nyquist result remains foundational.

However, the original formulation is incomplete for financial bars.

A 4-hour OHLC bar is **not a simple instantaneous sample** of a continuous signal. It is an aggregate/statistic over an interval.

Therefore distinguish:

\[
\text{point sampling}
\]

from:

\[
\text{window aggregation}.
\]

For bar-derived data, the effective observation operator is closer to:

\[
Y_k
=
\mathcal A(
X(t),
t\in[k\Delta,(k+1)\Delta)
)
\]

where \(\mathcal A\) may be:

- open;
- max;
- min;
- close;
- sum/volume;
- average.

### Improvement

Add an **observation-operator contract**:

- sampling/aggregation interval;
- timestamp semantics;
- availability/finalization time;
- anti-aliasing/aggregation assumptions;
- missing/asynchronous observation semantics.

This links STEP 01 directly to STEP 10.

---

# 7. STEP 02 — Noise / SNR
## Final verdict: KEEP as characterization, not a universal scalar

The major risk is treating SNR as if finance has one true physical signal/noise decomposition.

It does not.

SNR must remain conditional on a declared model:

\[
X=S+N.
\]

### Improvement

Maintain several labeled quantities:

- measurement-noise SNR;
- microstructure-noise SNR;
- decomposition-residual SNR;
- task-relative predictability ratio;
- spectral SNR.

Do not collapse them into one master quality number until STEP 12 proves such compression is valid.

---

# 8. STEP 03 — Denoising
## Final verdict: KEEP with stronger “do no harm” tests

The current protocol already correctly rejects:

\[
\text{high frequency}=\text{noise}.
\]

### Improvement

Every denoiser must report:

1. clean OOS task utility;
2. residual predictability;
3. tail/event preservation;
4. phase/spectral distortion;
5. latency/cost.

Economic-event windows should become an important tail/event stress subset:

> a denoiser that removes post-announcement transients may improve average MAE while destroying the exact information the trading application needs.

---

# 9. STEP 04 — Quantization / companding
## Final verdict: KEEP, but demote to evidence-gated optional branch

This STEP is theoretically sound but should not become an implementation priority merely because communications uses quantization.

For conventional float-valued forecasting, learned neural models may already tolerate excessive numerical precision.

Quantization becomes more promising when:

- tokenizing time series;
- compressing edge/deployment representations;
- enforcing finite resolution under low SNR;
- learning discrete latent states.

### Improvement

Prioritize:

\[
\text{raw}
\leftrightarrow
\text{quantized}
\leftrightarrow
[\text{raw},\text{quantized}]
\]

and only promote quantization if downstream value or efficiency is measurable.

---

# 10. STEP 05 — Source coding / entropy / context
## Final verdict: STRONG; reinterpret as source modeling, not bitstream generation

This remains one of the strongest conceptual STEPS.

Key outputs:

\[
-\log p(X_t|context)
\]

and:

\[
G_{i\to j}
=
L_j^{self}
-
L_j^{cond(i)}.
\]

### Improvement

The conditional-code-gain matrix should become a **shared evidence artifact** consumed by:

- STEP 09 grouping;
- STEP 10 pair prioritization;
- STEP 13 redundancy-aware allocation.

No coder bitstream is fed to the predictor.

---

# 11. STEP 06 — amplitude / frequency / phase / time-frequency
## Final verdict: STRONG, but avoid transform zoo

The biggest risk is combinatorial experimentation.

The correct final rule is:

> use STEP-06 diagnostics to identify which representations are justified for a source/feature family, then open only a small branch set.

### Improvement

Require a representation manifest:

```text
representation
causality_mode
invertible?
boundary_rule
phase_validity
frequency_support
SNR_support
expected detector
compute_cost
```

Economic-event causal responses can later produce response-kernel or event-phase representations, but they should not be mixed blindly into standard spectral transforms.

---

# 12. STEP 07 — matched filtering / pattern detection
## Final verdict: STRONG

This is where the user's classification infrastructure becomes explicitly useful.

The pattern-recognition assay can use:

- classification models;
- shapelets;
- MiniRocket/MultiRocket/HYDRA;
- matched filters;
- current predictor binary/directional variants.

### Improvement

Use the **classification stack as an assay**, not as a replacement for the regression system.

For a candidate pattern:

1. can it be detected?
2. is detection calibrated?
3. does detection predict future target utility?

Only then does it become a predictor branch.

---

# 13. STEP 08 — equalization
## Final verdict: NARROW THE CORE

This STEP became too broad when it included nearly all of:

- normalization;
- domain adaptation;
- test-time adaptation;
- optimal transport;
- adversarial domain alignment.

PATCH 002 already identified the problem.

### Normative final interpretation

STEP 08 opens only when an operational channel/distortion is declared:

\[
Y^{(d)}
=
G_d(S)+N.
\]

Examples:

- known source/vendor transform;
- instrument scale/response;
- controlled spectral distortion;
- volatility-dependent observation response;
- specific domain shift.

### Improvement

Move generic DANN/OT/TTA into an **optional domain-shift extension** rather than the equalization core.

The core baseline should remain:

- known affine inverse;
- Wiener/MMSE;
- RevIN/SAN/FAN where the distortion model matches.

---

# 14. STEP 09 — interference / shared-private separation
## Final verdict: STRONG, now improved by causal evidence

The safe decomposition remains:

\[
X=C+U
\]

with:

\[
[X,C,U]
\]

retained initially.

### New causal-inference improvement

Use causal inference **only to answer a declared causal question**, such as:

> Is the effect of a common macro/event driver on \(X_i\) different across regimes?

Do not use NOTEARS/DML as a magical filter deciding what should be subtracted.

Causal evidence can label a shared component as:

- likely confounder;
- mediator;
- treatment response;
- merely associative.

But destructive cancellation still requires downstream target evidence.

---

# 15. STEP 10 — synchronization
## Final verdict: STRONG; economic events add a natural timing anchor

The point-in-time availability contract remains essential.

### Improvement

Economic event publication times provide exogenous-ish anchors for studying:

- response onset;
- time to peak;
- decay;
- lag across assets;
- lag across countries;
- lag across representations.

This is a better use of the event table than forcing it into a regular time series.

---

# 16. STEP 11 — controlled redundancy
## Final verdict: CORRECTLY NARROWED

PATCH 002 is exactly right.

No new AE.

Use the existing `feature-extractor`.

The question remains:

\[
\text{train-only corruption}
\rightarrow
\text{same latent interface}
\rightarrow
P_{OOS}.
\]

No conceptual expansion is needed before results.

---

# 17. STEP 12 — adaptive routing
## Final verdict: STRONG, but dependent on upstream heterogeneity

The most important quantity remains:

\[
G_{oracle}
=
P_{oracle}
-
P_{static}.
\]

### Causal-inference critique

Because all frozen experts/configurations can be run historically on the same sample, their counterfactual predictive losses are often **computationally observable**:

\[
L_t(a_1),\ldots,L_t(a_K).
\]

Therefore DML/causal inference is usually unnecessary for the first router.

Use causal inference only if historical action choice prevented observing alternative outcomes or if routing itself changes future state.

This avoids adding causal machinery where exact counterfactual evaluation is already available.

---

# 18. STEP 13 — resource allocation
## Final verdict: STRONG and highly compatible with Level-3 meta-optimization

Again, when every candidate allocation can be trained/evaluated offline, direct response-surface evidence is preferable to causal inference.

### Improvement

STEP 13 should export a standardized allocation-response table:

\[
(
task,\;
branch\ budget,\;
genome,\;
cost,\;
metrics
)
\]

which becomes one of the richest inputs to Level-3 meta-optimization.

---

# 19. Compression transversal lane — final placement

The compression concepts remain cross-cutting.

## Sparse/CSC

Only after STEP-07 cheap detector gates.

## Successive refinement

Potentially bridges STEP 06 and STEP 12/13.

## Side information

Already absorbed into STEP 05/09.

## Hierarchical residual coding

Mainly STEP 03/05/06.

## C6 latent rate–distortion

Belongs to the existing feature extractor/core latent after SNR and representation evidence.

## Duration coding

Useful for event/regime token streams.

No additional main-chain STEP is required.

---

# 20. NEW CROSS-CUTTING LANE A
# Causal Economic-Event Response Modeling

This is the largest improvement made possible by the newly clarified project capability.

The event database is not a regular time series, and it should **not be forced into one**.

Instead treat each event as an indexed intervention/context object.

Event \(e\):

\[
e
=
(
t_e,\;
type_e,\;
country_e,\;
importance_e,\;
consensus_e,\;
actual_e,\;
metadata_e
).
\]

Define surprise:

\[
S_e
=
actual_e
-
consensus_e
\]

or a training-estimated standardized surprise:

\[
Z_e
=
\frac{
actual_e-consensus_e
}{
\sigma_{type,country}^{train}
}.
\]

The denominator is fitted training-only.

---

# 21. Event-study data representation

Construct an **event panel**, not a regular feature matrix.

One row per event:

```text
event_id
publication_time
event_family
country/currency
importance
consensus
actual
standardized_surprise
pre-event market state
pre-event volatility
pre-event regime
pre-event trend/spectrum
post-event outcomes h=...
```

Outcome examples:

\[
Y_{e,h}
=
r(
t_e,t_e+h
)
\]

or:

- realized volatility;
- maximum favorable/adverse excursion;
- trend slope change;
- spectral-power change;
- phase/coherence change;
- time to response peak.

---

# 22. Local projections as the first causal/dynamic event tool

Jordà local projections estimate horizon-specific responses without committing to a full VAR dynamic system.

For horizon \(h\):

\[
\boxed{
Y_{e,h}
=
\alpha_h
+
\beta_h S_e
+
\gamma_h^T W_e
+
\epsilon_{e,h}.
}
\]

This is highly appropriate for:

- event transients;
- impulse-response shape;
- onset/peak/decay;
- nonlinear/heterogeneous extensions.

It should be benchmarked before building a bespoke neural “causal event response network.”

---

# 23. Double Machine Learning for event heterogeneity

DML uses orthogonalization/cross-fitting to estimate treatment effects with high-dimensional nuisance controls.

A possible event formulation:

\[
T_e
=
Z_e
\]

continuous treatment = standardized surprise.

Outcome:

\[
Y_{e,h}.
\]

Controls:

\[
W_e
=
[
pre\ market\ state,\;
volatility,\;
calendar,\;
event\ family,\;
country,\;
related\ assets,\ldots
].
\]

Heterogeneity features:

\[
X_e
=
[
event\ family,\;
country,\;
regime,\;
pre\ volatility,\;
surprise\ sign/magnitude,\ldots
].
\]

Candidate tools:

- `LinearDML`;
- `NonParamDML`;
- `CausalForestDML`;
- DoubleML equivalents.

---

# 24. Critical causal warning

DML does **not** make an observational treatment causal automatically.

It still requires identification assumptions such as adequate control of confounding/unconfoundedness.

For scheduled economic releases, causal credibility can be stronger when:

- publication time is exact;
- treatment is unanticipated surprise;
- outcome window is narrow;
- pre-event controls are point-in-time;
- overlapping events are handled;
- anticipation/leakage is tested.

Where appropriate, high-frequency identification / instrumental-variable local projections can be considered.

---

# 25. 2026 time-series DML update

A 2026 paper extends Double Machine Learning specifically to time-series settings via Reverse Cross-Fitting under stated stationarity/time-reversibility conditions.

This is relevant if the causal unit becomes a continuous macro time series.

For the project's **event panel**, ordinary event-level cross-fitting may be simpler.

Do not use Reverse Cross-Fitting merely because it is new.

---

# 26. Heterogeneous treatment-effect outputs

The useful output is not only:

\[
ATE_h.
\]

Estimate, when identified:

\[
CATE_h(x)
=
E[
Y_h(1)-Y_h(0)
\mid X=x
].
\]

For continuous surprise:

\[
\tau_h(x)
=
\frac{
\partial E[Y_h|do(T=t),X=x]
}{
\partial t
}.
\]

Potential heterogeneity:

- event family;
- country/currency;
- positive versus negative surprise;
- high/low volatility regime;
- risk-on/off state;
- session;
- liquidity.

---

# 27. Event-response knowledge artifacts

The causal lane can generate train-only/frozen artifacts:

```text
event_family_response_curve
country_response_curve
CATE model
time_to_peak prior
decay/half_life prior
sign_asymmetry
volatility_response prior
trend_shift response
spectral_shift response
uncertainty/confidence interval
```

These are **knowledge artifacts**, not raw event labels.

---

# 28. Where event causal outputs enter STEPS

## STEP 05

Event surprisal/context.

## STEP 06

Event-conditioned spectral/phase change.

## STEP 07

Response templates / event detectors.

## STEP 09

Separate common macro driver from asset-private residual.

## STEP 10

Event-aligned response timing.

## STEP 12

Information-quality state: proximity to event, response uncertainty.

## Trading policy

Risk/exposure modulation only after predictive validation.

---

# 29. Event-token Transformer and causal lane are complementary

The existing event-token work already provides a learned variable-length representation path.

The correct division is:

\[
\boxed{
\text{event token model}
=
\text{representation/context}
}
\]

while:

\[
\boxed{
\text{causal event lane}
=
\text{effect estimation under explicit assumptions}.
}
\]

Do not replace one with the other.

A useful combined branch is:

\[
[
Z_{event-token},
\hat\tau_{event,h},
CI_{event,h}
].
\]

---

# 30. Causal discovery repository — final role

The current `causal-inference` repo contains exploratory NOTEARS, Invariant Causal Prediction and DoWhy analyses, but is explicitly marked experimental/unverified and still carries inherited `rl-optimizer` packaging identity.

Therefore:

\[
\boxed{
\text{do not make it a production dependency yet}.
}
\]

Use it as:

- experimental research workspace;
- hypothesis generator;
- causal-graph sensitivity tool.

Before it becomes shared infrastructure:

1. repair package identity;
2. pin causal dependencies;
3. add reproducible data contracts;
4. verify tests;
5. separate discovery from effect-estimation APIs.

---

# 31. NEW CROSS-CUTTING LANE B
# Level-3 Meta-Optimization from the OLAP Experiment Cube

This capability is unusually well aligned with current 2025–2026 research.

The project already conceives:

- Level 1: individual candidate training/evaluation;
- Level 2: DEAP/NEAT/DOIN search;
- Level 3: meta-learning from accumulated optimization history.

The final recommendation is to formalize Level 3 as:

\[
\boxed{
\text{a surrogate/warm-start system for Level 2},
}
\]

not as an autonomous champion selector.

---

# 32. Why this is state of the art, not speculative

Current literature directly supports learning from historical HPO runs.

Relevant lines include:

- OptFormer: Transformer trained on large HPO histories;
- Meta-Black-Box Optimization / Learn-to-Optimize;
- warm-started HPO;
- 2026 meta-learning for architecture/hyperparameter selection on time-series tasks;
- 2026 recurrent HPO systems using parameter/performance history.

A 2026 AAAI paper specifically meta-learns architecture + hyperparameter recommendations for time-series forecasting/classification and reports substantial search-cost reductions on subsets of its benchmarks.

This is extremely close to the project's Level-3 idea.

---

# 33. Level-3 meta-dataset

One record should represent one candidate evaluation:

\[
r
=
(
T,\;
G,\;
F,\;
O,\;
M,\;
C,\;
S
)
\]

where:

- \(T\): task/dataset meta-features;
- \(G\): genome/configuration;
- \(F\): fidelity/training budget;
- \(O\): optimizer context/history;
- \(M\): resulting metric vector;
- \(C\): compute/cost;
- \(S\): status/failure metadata.

---

# 34. Required task meta-features

Examples:

- asset;
- timeframe;
- sample count;
- feature count;
- target/horizon;
- algorithm family;
- representation family;
- estimated SNR;
- regime entropy;
- missingness;
- volatility distribution;
- autocorrelation/spectral summaries;
- class imbalance where applicable;
- training-window dates;
- domain ID.

These let Level 3 distinguish tasks rather than learning a single global parameter prior.

---

# 35. Genome encoding

The genome is heterogeneous and conditional.

Represent:

```text
parameter_name
typed value
active/inactive mask
model family
search-space version
bounds/schema
```

Do not flatten inactive conditional parameters as if they were meaningful zeros.

---

# 36. Metric vector, not only scalar fitness

Store/predict:

\[
\mathbf M
=
[
MAE,
R^2,
Sharpe,
drawdown,
trades,
calibration,
runtime,\ldots
].
\]

Level 2 may collapse this to a scalar fitness, but Level 3 should preserve the metric vector.

Why?

A future objective may change.

Historical runs should remain reusable.

---

# 37. Failed candidates are valuable

Do not remove:

- NaNs;
- divergence;
- timeout;
- OOM;
- no-trade;
- constraint failure.

Instead learn:

\[
P(feasible|T,G)
\]

separately from:

\[
E[M|T,G,feasible].
\]

This can dramatically improve warm-start safety.

---

# 38. Optimizer-history bias

DEAP/NEAT samples are **not IID**.

Later candidates are descendants of earlier high-fitness candidates.

Therefore the OLAP cube includes selection bias.

Record:

- optimizer;
- generation;
- parent IDs;
- mutation/crossover provenance;
- population/campaign ID;
- acquisition policy if applicable.

Level 3 should not pretend the history is a random factorial design.

---

# 39. Level-3 first baseline

Before OptFormer/Transformer/meta-RL:

\[
\boxed{
\text{GBDT/CatBoost surrogate}
}
\]

or another strong tabular model.

Input:

\[
[T,G,F,O].
\]

Outputs:

- predicted metric vector;
- uncertainty/ensemble dispersion;
- feasibility.

Use it to recommend:

\[
top-k
\]

initial genomes for Level 2.

---

# 40. Nearest-task warm start baseline

For new task \(T^*\):

1. compute task meta-features;
2. find nearest historical tasks;
3. retrieve their top robust genomes;
4. seed DEAP/NEAT population.

This is cheap, interpretable and a mandatory baseline.

---

# 41. 2026 time-series meta-HPO comparator

The AAAI 2026 architecture/hyperparameter meta-learning framework should be treated as a direct reference for:

\[
\text{task representation}
+
\text{architecture/config representation}
\rightarrow
\text{performance prediction}.
\]

The project can exceed this only if its richer OLAP evidence adds value.

---

# 42. OptFormer-style advanced model

If the meta-dataset becomes large/heterogeneous enough, a Transformer over:

- task metadata;
- parameter schema;
- historical trials

becomes plausible.

But do not begin there.

A learned universal optimizer requires far more evidence than a useful warm-start surrogate.

---

# 43. Level-3 primary experiment

Compare under the **same Level-2 evaluation budget**:

## Cold start

standard current DEAP/NEAT initialization.

## Random-history warm start

control.

## Nearest-task warm start

meta-feature baseline.

## Surrogate top-k warm start

Level-3 baseline.

Measure best-so-far:

\[
f^*(n)
\]

versus number of Level-1 evaluations:

\[
n.
\]

---

# 44. Level-3 metrics

Required:

- best-so-far fitness vs evaluations;
- time-to-threshold;
- regret versus eventual champion;
- diversity of seeded population;
- fraction of infeasible candidates;
- surrogate calibration;
- top-k recall;
- transfer success across tasks;
- wall-clock savings.

The Level-3 claim is:

\[
\boxed{
\text{better search efficiency},
}
\]

not merely high surrogate \(R^2\).

---

# 45. Level-3 split strategy

Never randomly split candidate rows from the same campaign into train/test and call it meta-generalization.

Use holdouts by:

- campaign;
- asset;
- timeframe;
- model family;
- time period;
- entire task.

The strongest evaluation is:

\[
\boxed{
\text{leave-one-task-family-out}.
}
\]

---

# 46. Held-out firewall propagation

A protected test/Stage-C period remains protected at **every optimization level**.

It cannot enter:

- Level-1 training;
- Level-2 fitness tuning;
- Level-3 surrogate training;
- task meta-features;
- prompts;
- event-effect fitting.

Meta-learning does not create an exemption from leakage governance.

---

# 47. Level-3 should recommend priors, not hard truth

Output:

\[
P(G|T)
\]

or top-k candidates, not:

\[
\boxed{
\text{“these are the optimal hyperparameters.”}
}
\]

Level 2 remains responsible for local search because:

- tasks shift;
- objective changes;
- interactions are nonstationary.

---

# 48. Level-3 and STEP 13

STEP 13 produces:

\[
(\mathbf c,B,P,C)
\]

allocation response surfaces.

These should enter the Level-3 cube.

Then Level 3 can warm-start:

- branch widths;
- budget allocations;
- router thresholds;
- representation choices.

This makes the final pipeline recursively self-improving **without violating the hierarchy of evidence**.

---

# 49. NEW CROSS-CUTTING LANE C
# Multi-paradigm verification

A transformation should not be accepted based on one downstream model.

Recommended escalating assays:

## Assay A — simple regression

Does information improve a low-capacity predictor?

## Assay B — classification

Can expected patterns/events be detected?

## Assay C — unsupervised

Is claimed structure stable without labels?

## Assay D — causal

Does a declared intervention have identified effect?

## Assay E — complex predictor

Does the project model gain?

## Assay F — simulator/backtest

Does predictive gain survive costs/action rules?

## Assay G — RL

Only if the representation becomes state input for policy learning.

This prevents architecture-specific false conclusions.

---

# 50. Hierarchical clustering / unsupervised role

Hierarchical/agglomerative clustering and other regime algorithms should be used to:

- propose groups;
- summarize states;
- test stability;
- define strata.

They should not be treated as ground-truth regimes.

Every cluster output requires:

- temporal stability;
- interpretability;
- downstream utility;
- leakage audit.

---

# 51. Classification role

Classification infrastructure is particularly valuable in:

- STEP 07 pattern detection;
- STEP 10 synchronization lock/failure classification;
- STEP 12 mode/failure-state diagnostics;
- economic-event response classes.

A classifier can be a measurement instrument even when the final application is regression.

---

# 52. Causal inference role — what it should NOT do

Do not use causal tools simply because a transformation feels “causal.”

Causal inference requires:

1. treatment/intervention;
2. outcome;
3. identification assumptions;
4. temporal ordering;
5. confounder argument;
6. overlap/positivity where relevant.

Examples where causal inference is **not** the default tool:

- choosing FFT bins;
- denoising;
- quantization;
- reconstructing latents;
- selecting a router when every expert can be replayed counterfactually;
- branch-budget search when all allocations can be directly evaluated.

---

# 53. Simulation and synthetic generation — final role

Synthetic data is essential when a quantity is otherwise unidentifiable.

Use it for:

- known SNR;
- known delay;
- known common/private sources;
- known causal treatment effects;
- known detector templates;
- known corruption;
- known branch utility curves.

But synthetic success never substitutes for public/financial OOS validation.

---

# 54. Backtesting — final role

Backtesting answers:

> does validated predictive/state information survive the application policy and costs?

It does **not** prove:

- causal validity;
- representation correctness;
- denoising correctness.

Backtest sits downstream of the evidence chain.

---

# 55. RL — final role

RL is appropriate when the problem is genuinely sequential and actions change future state/reward.

It is a poor first assay for preprocessing because reward noise/credit assignment can hide whether representation improved.

Therefore:

\[
\boxed{
\text{representation validity}
\rightarrow
\text{supervised/statistical evidence}
\rightarrow
\text{RL utility}.
}
\]

---

# 56. Final priority order

The original 13 STEPS are too large to execute exhaustively in sequence.

Recommended implementation priority:

## Tier 0 — contracts and integrity

- point-in-time availability;
- data hashes;
- train-only fit rules;
- representation manifest;
- repository test health;
- event-vintage contract.

## Tier 1 — cheap evidence

- SNR characterization;
- denoising controls;
- conditional code-gain table;
- simple spectral diagnostics;
- causal event panel;
- event local projections;
- branch subset tests;
- baseline feature-extractor reproducibility.

## Tier 2 — representation candidates

- selected STEP-06 transforms;
- STEP-07 detector baselines;
- narrow STEP-08 equalizers;
- STEP-09 common/private decomposition;
- STEP-10 lag evidence.

## Tier 3 — robustness

- STEP 11 corruption-hardening.

## Tier 4 — adaptive system

- STEP 12 oracle/routing;
- STEP 13 allocation frontier.

## Tier 5 — meta-level acceleration

- Level-3 warm-start/meta-surrogate.

## Tier 6 — application

- complex predictor;
- backtest;
- RL;
- live/simulator evaluation.

---

# 57. Stop rules

The project should actively kill branches.

Examples:

\[
G_{oracle}^{STEP12}\approx0
\Rightarrow
\text{no router}.
\]

\[
\Delta P_{quantization}\approx0
\Rightarrow
\text{no quantized branch}.
\]

\[
\Delta P_{detector}\approx0
\Rightarrow
\text{no detector branch}.
\]

\[
P_{equalized}<P_{raw}
\Rightarrow
\text{do not canonicalize}.
\]

\[
P_{robust-latent}\leq P_{baseline}
\Rightarrow
\text{STEP 11 closes}.
\]

The final architecture should be **smaller than the research space**.

---

# 58. Final “knowledge contract” for every branch

A production-eligible branch must publish:

```text
branch_id
source_step
input_semantics
point_in_time_rule
fit_scope
transform_parameters
output_shape
output_units
causality_mode
noise assumptions
quality metrics
OOD behavior
missingness behavior
compute cost
latency
downstream incremental utility
tail/event utility
artifact hash
training data hash
```

This is the generalization mechanism across domains.

---

# 59. Final evidence cube

The OLAP/meta-evidence system should eventually store not just model metrics but **research evidence**:

```text
task_id
dataset_id
step_id
branch_id
representation_id
genome
fit_scope
quality_state_summary
metrics
tail_metrics
cost
latency
seed
failure_reason
parent/genetic lineage
artifact hashes
protocol hash
```

This becomes the substrate for Level 3.

---

# 60. Final master optimization hierarchy

\[
\boxed{
\text{Level 0}
=
\text{data/representation contract}
}
\]

\[
\boxed{
\text{Level 1}
=
\text{individual candidate training}
}
\]

\[
\boxed{
\text{Level 2}
=
\text{DEAP/NEAT/DOIN candidate search}
}
\]

\[
\boxed{
\text{Level 3}
=
\text{meta-learned warm start / surrogate / algorithm selection}
}
\]

Potential future Level 4, only much later:

\[
\text{automatic experiment-program design}.
\]

Do not open Level 4 now.

---

# 61. Final novelty/autocritique assessment

Individually, almost none of the component methods are novel:

- wavelets;
- entropy models;
- matched filtering;
- RevIN;
- factor models;
- lag detection;
- masked reconstruction;
- MoE routing;
- branch budget allocation;
- DML;
- HPO meta-learning.

The potentially distinctive contribution is the **governed integration and falsification framework**:

\[
\boxed{
\text{communications/information-theory questions}
+
\text{representation-matched ML}
+
\text{causal event-response evidence}
+
\text{multi-paradigm assays}
+
\text{hierarchical optimization evidence}.
}
\]

Any future novelty claim should be stated at that systems/methodology level only after a systematic literature review.

---

# 62. Final research question

The overall research question can now be stated cleanly:

> Can a modular, causally governed, information-quality-aware preprocessing and representation system expose task-relevant structure in heterogeneous multivariate sequential data so that downstream models learn more accurately, robustly, and efficiently than from a fixed raw representation?

Secondary question:

> Can accumulated optimization evidence be meta-learned to reduce the cost of finding those representations on new tasks without leaking held-out information?

---

# 63. Final validation strategy

A result should ideally survive the following ladder:

\[
\boxed{
\text{known synthetic truth}
\rightarrow
\text{public benchmark}
\rightarrow
\text{project validation}
\rightarrow
\text{held-out confirmation}
\rightarrow
\text{application/backtest/RL}.
}
\]

Causal-event claims additionally require:

\[
\boxed{
\text{identification argument}
+
\text{causal robustness/refutation}.
}
\]

---

# 64. Proposed final implementation work packages

## WP-A — Evidence contracts

Implement shared schemas for:

- quality;
- representation;
- branch;
- protocol;
- causal event;
- meta-optimization trial.

## WP-B — Cheap diagnostics

Implement low-cost STEP-03/05/06/09/10 evidence generation.

## WP-C — Event causal lane

Build event panel + local projections + DML/CATE experiments.

## WP-D — Branch qualification

Run STEP-06/07/08/09/10 candidate gates.

## WP-E — Existing extractor robustness

Execute STEP 11 exactly as specified.

## WP-F — Adaptive control

STEP 12.

## WP-G — Budget allocation

STEP 13.

## WP-H — Level-3 meta-optimization

Warm-start DEAP/NEAT/DOIN from OLAP evidence.

## WP-I — Application validation

Predictor/backtester/RL/live simulator.

---

# 65. Work-package dependencies

```text
WP-A
 ├─ WP-B
 ├─ WP-C
 │   └─ WP-D
 └─ WP-D
      └─ WP-E
           └─ WP-F
                └─ WP-G
                     └─ WP-H

WP-D/E/F/G → WP-I
WP-C       → WP-I
```

Level 3 can begin earlier using historical data, but it should not control the new chain until its own validation gates pass.

---

# 66. Event-causal lane gates

## CE-A — point-in-time event integrity

Exact publication timestamps and vintages valid.

## CE-B — overlap/anticipation audit

No obvious contamination by overlapping releases/future revisions.

## CE-C — baseline event study

Local projection/event-study response survives.

## CE-D — DML/HTE stability

Heterogeneous effects reproducible across folds/time blocks.

## CE-E — incremental predictive utility

Causal-response artifacts add to event-token/raw features.

If CE-C fails, do not force CATE models.

---

# 67. Level-3 gates

## L3-A — sufficient evidence volume

Enough comparable historical evaluations.

## L3-B — task split validity

Meta-generalization survives whole-task holdouts.

## L3-C — simple surrogate

GBDT/nearest-task warm start beats cold/random start.

## L3-D — search-efficiency benefit

Best-so-far curve improves under identical Level-2 evaluation budget.

## L3-E — no diversity collapse

Warm start does not prematurely concentrate search.

## L3-F — advanced meta-model

Only if simple L3 works.

---

# 68. Strong warnings

1. **Do not let Level 3 optimize against test leakage.**
2. **Do not interpret causal discovery graphs as identified causal effects.**
3. **Do not treat event-token attention weights as causal effects.**
4. **Do not convert every STEP into a permanent feature branch.**
5. **Do not use RL to rescue weak predictive evidence.**
6. **Do not optimize one scalar fitness and discard the metric vector.**
7. **Do not use revised macro data as historical event truth.**
8. **Do not benchmark giant SOTA models before cheap falsifiers.**
9. **Do not confuse representational bits with stored float bits or neuron capacity.**
10. **Do not claim novelty from the communications analogy alone.**

---

# 69. High-value improvements identified in this final review

The most valuable changes relative to the earlier roadmap are:

### Improvement 1

Treat the pipeline as a **branch graph**, not a serial 13-stage transform.

### Improvement 2

Narrow STEP 08 to operational equalization.

### Improvement 3

Add a **causal event-response lane** rather than forcing event data into a regular time series.

### Improvement 4

Use local projections + DML/CausalForest as event-effect tools.

### Improvement 5

Keep event-token representation and causal event effect estimation separate but composable.

### Improvement 6

Formalize the **Level-3 OLAP meta-optimization lane**.

### Improvement 7

Use all candidate metrics/genomes/failures as meta-data, not only champions.

### Improvement 8

Evaluate Level 3 on **search efficiency**, not surrogate fit alone.

### Improvement 9

Use classification, unsupervised, causal, regression and RL as different **assays**, not competing universal paradigms.

### Improvement 10

Make stop/null decisions an explicit product of every STEP.

---

# 70. Selected state-of-the-art references added by final review

## Causal inference / events

1. V. Chernozhukov et al., “Double/debiased machine learning for treatment and structural parameters,” *The Econometrics Journal*, 2018.  
   https://doi.org/10.1111/ectj.12097

2. Ò. Jordà, “Estimation and Inference of Impulse Responses by Local Projections,” *American Economic Review*, 2005.  
   https://doi.org/10.1257/0002828053828518

3. EconML, Double Machine Learning / CausalForestDML documentation.  
   https://econml.azurewebsites.net/spec/estimation/dml.html

4. DoubleML project/user guide.  
   https://docs.doubleml.org/stable/

5. M. Ciganovic, F. D'Amario, M. Tancioni, “Double Machine Learning for Time Series,” 2026.  
   arXiv:2603.10999

6. L. Khalaf, Z. Lin, H. Tang, “Monetary policy surprises: Robust dynamic causal effects,” *Journal of Economic Dynamics and Control*, 2026.  
   https://doi.org/10.1016/j.jedc.2026.105309

## Meta-optimization / HPO

7. Y. Chen et al., “Towards Learning Universal Hyperparameter Optimizers with Transformers (OptFormer),” 2022.  
   https://arxiv.org/abs/2205.13320

8. X. Yang, R. Wang, K. Li, “Meta-Black-Box Optimization for Evolutionary Algorithms: Review and Perspective,” *Swarm and Evolutionary Computation*, 2025.  
   https://doi.org/10.1016/j.swevo.2024.101838

9. Z. Ma et al., “Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization,” *IEEE Transactions on Evolutionary Computation*, 2025/2026.  
   https://doi.org/10.1109/TEVC.2025.3568053

10. E. Moeini et al., “Neural Architecture and Hyperparameter Selection Through Meta-Learning on Time Series,” *AAAI*, 2026.  
    https://doi.org/10.1609/aaai.v40i29.39622

11. “Automated deep learning by recurrent hyperparameter optimization,” *Nature Communications*, 2026.  
    https://doi.org/10.1038/s41467-026-72413-9

12. P. Ram, “On the Optimality Gap of Warm-Started Hyperparameter Optimization,” *AutoML Conference*, 2022.  
    https://proceedings.mlr.press/v188/ram22a.html

13. DEHB: Evolutionary Hyperband for scalable HPO.  
    https://www.automl.org/dehb/

---

# 71. Final project stance

The project should now stop expanding the conceptual chain.

The next stage is **execution and evidence accumulation**, with two new cross-cutting research lanes:

\[
\boxed{
\text{Causal Economic-Event Response}
}
\]

and:

\[
\boxed{
\text{Level-3 Meta-Optimization}.
}
\]

The final architecture should emerge from surviving gates rather than from the full theoretical roadmap.

The target is not the largest system.

The target is:

\[
\boxed{
\text{the smallest validated set of representations, detectors and adaptive controls that preserves the useful information for the task.}
}
\]

---

# 72. Final status

**The communications-derived main roadmap is complete through STEP 13.**

**Compression-derived extensions are formally placed.**

**Causal event-response analysis is added as a cross-cutting lane.**

**Level-3 meta-optimization is added as a cross-cutting optimization lane.**

**The final implementation phase should now be governed by the work packages, gates, stop rules and evidence contracts defined in this document and PATCH 003.**
