# STEP 13 — Multiplexing / MIMO / Optimal Multi-Branch Budget Allocation
## Capacity-Constrained Allocation of Latent Width, Compute, Features, Experts and Resolution Across Validated Information Branches

**Status:** Research protocol — final draft for independent agent audit  
**Date:** 2026-09-05  
**Scope:** Allocation of finite representational/computational resources across already validated branches, modalities, experts and transformed views. Initial domain: multivariate financial time series; intended abstraction: general multisource sequential systems.  
**Prerequisites:** STEPS 01–12.  
**Critical boundary:** STEP 12 decides **which mode/configuration should be active** under the current information-quality state. STEP 13 decides **how a finite budget should be distributed among the branches/components that are eligible and active**.  
**Implementation status:** research protocol only. No large architecture search, no dynamic-width refactor and no production mutation are authorized by this document.

---

# 0. Executive conclusion

STEP 13 asks:

> **Given several useful information branches and a finite model/inference budget, how should capacity be allocated among them rather than assigning every branch the same width, compute or representation budget?**

The classical communications analogues are:

1. **multiplexing** — combine several information streams over limited transmission resources;
2. **parallel-channel bit loading** — allocate bits/power preferentially to better subchannels;
3. **MIMO** — exploit multiple correlated spatial streams/eigenmodes;
4. **water-filling** — allocate continuous power so that useful channels receive more resources while weak channels may receive little or none.

The project analogue is not to call neural latent dimensions “Shannon bits.”

Instead define a real engineering budget:

\[
\boxed{
B
}
\]

and branch allocations:

\[
\boxed{
c_1,\ldots,c_K
}
\]

subject to:

\[
\boxed{
\sum_{i=1}^{K} C_i(c_i)
\leq
B
}
\]

where \(c_i\) may control:

- latent dimension;
- Conv/filter width;
- number of retained spectral bins;
- number of detector templates;
- number of active experts;
- token count;
- attention budget;
- sampling/representation resolution;
- FLOPs;
- latency;
- memory.

The optimization target is empirical task utility:

\[
\boxed{
\max_{\mathbf c}
P(\mathbf c)
\quad
\text{s.t.}
\quad
C(\mathbf c)\leq B.
}
\]

The state-of-the-art review yields four strong constraints.

First, classical water-filling/bit-loading shows why **uniform resource allocation is not generally optimal** when parallel channels differ in quality.

Second, MIMO theory shows that correlated channels should often be transformed into more independent eigenmodes before allocating power; analogously, STEP 09 redundancy/shared-private analysis should inform branch allocation.

Third, modern ML already studies budget-aware allocation at several granularities:

- modality selection;
- dynamic width/depth;
- token pruning/merging;
- expert-activation budgets;
- resource-aware inference;
- latency-aware model configuration.

Fourth, theoretical FLOP savings frequently fail to become real latency savings because routing, dispatch, tensor compaction and irregular shapes have overhead.

Therefore STEP 13 should begin with an **offline static budget-allocation frontier**, not a dynamic neural allocator.

Recommended ladder:

\[
\boxed{
\text{equal allocation}
\rightarrow
\text{single-branch baselines}
\rightarrow
\text{marginal-utility curves}
\rightarrow
\text{greedy allocation}
\rightarrow
\text{knapsack / DP allocation}
\rightarrow
\text{interaction-aware allocation}
\rightarrow
\text{dynamic state-conditioned allocation only after STEP 12}.
}
\]

---

# 1. Historical basis — parallel channels

Consider \(K\) independent Gaussian subchannels:

\[
Y_i
=
h_i X_i
+
N_i.
\]

With power allocation:

\[
P_i
\]

and total constraint:

\[
\sum_i P_i
\leq
P.
\]

The aggregate capacity under the classical idealized model is:

\[
C
=
\sum_i
\log_2
\left(
1+\frac{|h_i|^2P_i}{N_i}
\right)
\]

up to bandwidth/normalization conventions.

The crucial principle is:

\[
\boxed{
\text{different subchannels can have different marginal value per unit resource}.
}
\]

---

# 2. Water-filling

For separable concave channel utilities, optimal allocation has the form:

\[
\boxed{
P_i
=
\left(
\mu
-
\frac{N_i}{|h_i|^2}
\right)^+
}
\]

where:

\[
(x)^+
=
\max(x,0)
\]

and \(\mu\) is chosen to satisfy total power.

Interpretation:

- good channel:
  more power;
- poor channel:
  less power;
- sufficiently poor channel:
  potentially zero power.

This does **not** directly determine neural branch width.

It motivates a falsifiable ML hypothesis:

> if branch utility versus resource is concave enough and approximately separable, allocate resources until marginal utility per cost is approximately equalized.

---

# 3. Marginal-utility equalization

Let branch utility contribution under budget \(c_i\) be:

\[
U_i(c_i).
\]

For differentiable separable optimization:

\[
\max_{\mathbf c}
\sum_i U_i(c_i)
\]

subject to:

\[
\sum_i c_i
\leq
B.
\]

At an interior optimum:

\[
\boxed{
\frac{\partial U_i}{\partial c_i}
=
\lambda
}
\]

for active branches, up to branch-specific cost scaling.

This becomes the STEP-13 “water-filling” analogue.

But the separability assumption must be tested.

---

# 4. Discrete bit loading

Real communication systems cannot always assign continuous resources.

Discrete Multitone/DSL uses integer bit loading:

\[
b_i\in\{0,1,2,\ldots\}.
\]

Chow–Cioffi–Bingham developed a practical finite-granularity loading method with near-optimal performance relative to water-pouring in their setting.

STEP-13 analogue:

\[
c_i
\in
\{0,8,16,32,64,\ldots\}
\]

for branch width or another discrete resource.

Thus the project should compare:

- continuous relaxation;
- discrete feasible allocation.

---

# 5. MIMO model

A classical MIMO channel is:

\[
\boxed{
\mathbf y
=
\mathbf H\mathbf x
+
\mathbf n.
}
\]

The channel matrix:

\[
\mathbf H
\]

couples multiple input/output streams.

The singular-value decomposition:

\[
\mathbf H
=
U\Sigma V^H
\]

can transform the channel into approximately parallel eigenmodes.

Power can then be allocated across singular modes.

The key project analogy is:

\[
\boxed{
\text{do not allocate capacity independently to highly redundant branches before examining their joint information geometry}.
}
\]

STEP 09 common/private decomposition and STEP 05 conditional code gain therefore precede STEP 13 for a reason.

---

# 6. MIMO warning — multiple branches are not automatically independent channels

If:

\[
Z_1
\approx
Z_2,
\]

doubling both widths may waste capacity.

If:

\[
Z_1
\perp
Z_2
\]

and both contain target information, parallel capacity may be valuable.

Therefore define branch complementarity diagnostics before allocation:

- correlation;
- conditional code gain;
- canonical correlation;
- redundancy;
- incremental target utility;
- shared/private decomposition.

---

# 7. STEP 13 is not STEP 04

STEP 04 asks:

> what numerical resolution/quantization is useful for a feature?

STEP 13 asks:

> given finite total system resources, how much resource should each already-defined branch receive?

A STEP-13 budget may include quantization bits, but branch allocation is broader.

---

# 8. STEP 13 is not compression lane C6

C6 asks about latent information rate:

\[
R_Z
=
E[-\log p(Z)].
\]

STEP 13 can allocate latent dimensions or compute without claiming those dimensions equal information bits.

C6 may later provide a more principled branch cost.

For the first STEP-13 experiment:

\[
\boxed{
\text{use measurable engineering cost, not theoretical latent bit claims}.
}
\]

---

# 9. STEP 13 is not STEP 12

STEP 12:

\[
q_t
\rightarrow
a_t.
\]

STEP 13:

\[
a_t,B_t
\rightarrow
\mathbf c_t.
\]

The initial STEP-13 experiment freezes:

\[
a_t=a
\]

and studies **static allocation**.

Dynamic state-conditioned allocation is a later extension.

---

# 10. Existing project architecture gives a concrete starting point

The current `predictor_plugin_composite.py` already contains a real multi-branch architecture.

In its `close_window_only` path, the implemented branches include:

- full-window CLOSE branch with 64-channel Conv1D output;
- 15-minute HF branch with 32 channels;
- 30-minute HF branch with 32 channels;
- point/context branch with 32 channels;

followed by concatenation and shared fusion.

Thus the current allocation is effectively:

\[
\boxed{
64:32:32:32
}
\]

before the fusion layer.

STEP 13 can directly ask:

> Is this fixed manually chosen branch capacity near a Pareto-optimal allocation?

This is substantially more concrete than inventing a synthetic multi-branch architecture solely for this step.

---

# 11. Fixed branch widths are a baseline, not a defect

The current branch widths provide:

\[
A_{current}.
\]

Do not assume a new allocator must improve them.

Possible valid result:

\[
\boxed{
64:32:32:32
\text{ is already near the useful frontier.}
}
\]

---

# 12. Resource types

STEP 13 must choose **one primary budget unit per experiment**.

Possible units:

## Latent/channel width

\[
c_i=d_i.
\]

## Parameters

\[
c_i=\#\theta_i.
\]

## FLOPs/MACs

\[
c_i=F_i.
\]

## Measured latency

\[
c_i=L_i.
\]

## Memory

\[
c_i=M_i.
\]

## Input representation count

\[
c_i=\text{features/bins/tokens retained}.
\]

Do not optimize all resource types simultaneously in the first experiment.

---

# 13. Recommended first budget unit

Use:

\[
\boxed{
\text{branch output channel width}
}
\]

because:

- it maps directly to the existing composite architecture;
- it is easy to control;
- total latent/fusion width is interpretable;
- it does not require a hardware-dependent timing loop to establish the hypothesis.

Measured latency/FLOPs are reported as diagnostics.

---

# 14. Resource-cost function

Width is not proportional to actual compute in every branch.

Define:

\[
C_i(c_i)
\]

from measured or analytical branch cost.

For Conv1D:

\[
FLOPs
\propto
T
\cdot
K
\cdot
C_{in}
\cdot
C_{out}.
\]

Thus giving 32 channels to a long-window branch may cost more than 32 channels in a short vector branch.

The allocation optimizer should ultimately use:

\[
\boxed{
\text{cost}
}
\]

not only width.

---

# 15. Real latency versus theoretical FLOPs

Modern dynamic-compute research repeatedly shows:

\[
\boxed{
FLOPs\downarrow
\not\Rightarrow
latency\downarrow.
}
\]

Reasons include:

- dispatch overhead;
- memory access;
- irregular tensor shape;
- indexing;
- batching;
- kernel inefficiency.

Therefore every promoted allocation must report measured latency on the intended hardware class.

---

# 16. State of the art — Dynamic Slimmable Networks

Dynamic Slimmable Networks adjust network width per input while preserving hardware-friendly contiguous execution.

Their motivation is directly relevant:

> dynamic pruning may reduce theoretical operations but not real runtime because sparse execution overhead dominates.

STEP 13 should treat **structured width choices** as preferable to arbitrary irregular pruning for first implementation.

---

# 17. Once-for-All networks

Once-for-All trains one supernetwork that supports many subnetwork choices across:

- depth;
- width;
- kernel size;
- resolution.

It decouples supernetwork training from later deployment specialization.

Project lesson:

\[
\boxed{
\text{one trained architecture can potentially expose multiple budget configurations without retraining every allocation from scratch}.
}
\]

However this requires a special training scheme and is a late-stage engineering optimization.

The first STEP-13 evidence should use separately trained/frozen width configurations.

---

# 18. State of the art — modality selection under budget

Efficient Modality Selection (JMLR 2024) formalizes selection of the most useful subset of modalities under a cardinality constraint.

Important findings/concepts include:

- not all modalities are useful enough to justify their cost;
- redundant modalities can waste budget;
- marginal contribution can guide selection;
- under suitable dependence assumptions, greedy submodular selection has approximation guarantees.

This is extremely relevant to branch allocation.

Project interpretation:

\[
\boxed{
\text{branch inclusion itself is a budget-allocation variable}.
}
\]

Thus:

\[
c_i=0
\]

must be allowed.

---

# 19. MOSEL — dynamic modality serving

MOSEL (EMNLP 2024) dynamically chooses input modality configurations under application accuracy/performance requirements.

It demonstrates that:

- modality subsets can be chosen dynamically;
- using all modalities is not always necessary;
- serving/inference cost is part of the decision.

This is close to the STEP-12/13 boundary:

- STEP 12 chooses configuration;
- STEP 13 determines budget/subset under a resource constraint.

---

# 20. State of the art — Alloc-MoE 2026

Alloc-MoE explicitly introduces an **activation budget** and allocates expert activations at:

- layer level;
- token level.

It uses:

- sensitivity profiling;
- dynamic programming;
- routing scores;

to reduce performance degradation under a fixed activation budget.

This is one of the strongest modern analogues to STEP 13.

The central reusable idea is:

\[
\boxed{
\text{profile marginal sensitivity, then allocate finite activation budget where it hurts least / helps most}.
}
\]

---

# 21. Expert-attention compute allocation — ACL 2026

Recent work studies the optimal fraction of total compute assigned to:

- expert layers;
- attention layers

in MoE systems.

It finds that the optimal ratio changes with:

- total compute budget;
- sparsity.

This independently supports STEP-13's premise:

\[
\boxed{
\text{optimal internal resource proportions can depend on the total available budget}.
}
\]

Thus one fixed 64:32:32:32 split need not remain optimal at every total budget.

---

# 22. Dynamic token-computation literature

A 2026 survey of dynamic token computation unifies:

- pruning;
- merging;
- routing;
- early exit;
- adaptive depth.

It emphasizes:

- budget control;
- decision stability;
- realized token counts;
- measured latency;
- routing overhead.

This produces a mandatory STEP-13 reporting rule:

> always report the **realized** budget and runtime, not only the requested theoretical budget.

---

# 23. Time-series MoE allocation prior art

Recent time-series models increasingly allocate modeling capacity across:

- frequency bands;
- resolutions;
- experts.

M²FMoE (AAAI 2026) uses:

- multi-view Fourier/wavelet expert groups;
- multiple resolutions;
- adaptive fusion.

ABF-MoE (2026) learns adaptive spectral bands and routes information across frequency-specific experts.

This means:

> “allocate separate expert capacity to different frequency bands” is already prior art.

The project-specific opportunity is not to rediscover frequency experts, but to allocate budget using the information diagnostics from STEPS 03–12.

---

# 24. Multimodal/resource-budget principle

In multimodal ML:

\[
\text{more modalities}
\]

can improve quality but also:

- increase compute;
- add redundancy;
- create missing-data fragility.

This is structurally equivalent to the project branch bank.

Thus branch allocation should consider both:

\[
\boxed{
\text{marginal utility}
}
\]

and:

\[
\boxed{
\text{redundancy}.
}
\]

---

# 25. Branch budget vector

Let:

\[
\mathbf c
=
[c_1,\ldots,c_K].
\]

Feasible set:

\[
\mathcal C_B
=
\{
\mathbf c:
C(\mathbf c)\leq B
\}.
\]

Optimization:

\[
\boxed{
\mathbf c^*
=
\arg\max_{\mathbf c\in\mathcal C_B}
P(\mathbf c).
}
\]

---

# 26. Static budget frontier

Evaluate multiple total budgets:

\[
B
\in
\{
B_1,\ldots,B_M
\}.
\]

For each:

\[
\mathbf c^*(B).
\]

Build:

\[
\boxed{
P^*(B)
}
\]

the best achievable validation performance at each budget.

This is the primary STEP-13 artifact.

---

# 27. Equal-allocation baseline

For \(K\) branches:

\[
c_i
=
\frac{B}{K}
\]

subject to discrete rounding.

This is mandatory.

---

# 28. Current-allocation baseline

For the existing four-branch composite:

\[
\mathbf c_{current}
=
[64,32,32,32].
\]

Scale proportionally for different budgets:

\[
\mathbf c_B
=
\alpha_B
[64,32,32,32].
\]

This separates:

- equal allocation;
- existing engineering prior;
- learned/optimized allocation.

---

# 29. Single-branch baselines

For each branch \(i\):

\[
c_i=B,
\]

others zero.

These determine whether one branch alone dominates under low budgets.

They also reveal genuine complementarity.

---

# 30. Leave-one-branch-out baselines

Evaluate:

\[
P(\mathbf c_{-i})
\]

where branch \(i\) is removed and freed budget redistributed.

This estimates branch necessity under budget.

---

# 31. Marginal utility curve

For branch \(i\), hold others at a reference allocation and vary:

\[
c_i.
\]

Estimate:

\[
U_i(c_i)
=
P(c_i,\mathbf c_{-i}^{ref}).
\]

Marginal gain:

\[
\boxed{
\Delta_i(c)
=
P(c_i+\delta)-P(c_i).
}
\]

Normalize by cost:

\[
\boxed{
g_i
=
\frac{\Delta P_i}{\Delta C_i}.
}
\]

This is the empirical water-filling score.

---

# 32. Diminishing returns test

Test whether:

\[
\Delta_i(c)
\]

decreases as:

\[
c_i
\]

grows.

If approximately true:

\[
\boxed{
\text{greedy marginal allocation is plausible}.
}
\]

If not, allocation may contain thresholds/synergies requiring interaction-aware search.

---

# 33. Greedy marginal allocation

Initialize minimum allocation:

\[
\mathbf c^{(0)}.
\]

Repeatedly assign next resource quantum:

\[
\delta
\]

to branch:

\[
i^*
=
\arg\max_i
\frac{
\Delta P_i
}{
\Delta C_i
}.
\]

Continue until budget exhausted.

This is the simplest water-filling analogue.

---

# 34. Knapsack formulation

For each branch, define discrete capacity choices:

\[
c_i
\in
\mathcal D_i.
\]

Each choice has cost:

\[
C_{ij}
\]

and estimated value:

\[
V_{ij}.
\]

Then solve a multiple-choice knapsack:

\[
\max
\sum_i
V_{i,j(i)}
\]

subject to:

\[
\sum_i
C_{i,j(i)}
\leq
B.
\]

This is appropriate when branch utilities are approximated independently.

---

# 35. Dynamic programming baseline

The STEP-13 problem sizes are likely small enough that exact/near-exact dynamic programming can often solve the discrete independent approximation.

Use DP before invoking:

- Bayesian optimization;
- RL;
- evolutionary search.

---

# 36. Interaction term

Actual performance is not necessarily separable.

Define pairwise synergy:

\[
\boxed{
S_{ij}
=
P(i,j)
-
P(i)
-
P(j)
+
P(\emptyset)
}
\]

under matched budgets/reference conditions.

Positive:

\[
S_{ij}>0
\]

means complementarity.

Negative:

\[
S_{ij}<0
\]

means redundancy/interference.

---

# 37. Interaction-aware objective

Approximate:

\[
P(\mathbf c)
\approx
P_0
+
\sum_i U_i(c_i)
+
\sum_{i<j}
S_{ij}(c_i,c_j).
\]

If pairwise interactions explain enough variance, use them in allocation.

Otherwise continue with separable/greedy model.

---

# 38. Shapley/marginal contribution

For a small number of branches, Shapley-style contribution estimates can quantify average marginal utility across subsets.

This is related to the 2024 efficient modality-selection literature.

But exact Shapley cost is exponential.

Use:

- permutation approximation;
- small branch sets only.

Do not create a large Shapley campaign.

---

# 39. Submodularity test

A set function \(F(S)\) is submodular if marginal gain decreases as the selected set grows.

Informally:

\[
\Delta(i|A)
\geq
\Delta(i|B)
\quad
\text{for }
A\subseteq B.
\]

If branch utility is approximately submodular, greedy branch selection has strong theoretical motivation.

This should be empirically checked on small subsets.

---

# 40. Branch elimination

Allow:

\[
\boxed{
c_i=0.
}
\]

If a branch's marginal contribution is consistently negative or negligible under constrained budgets, removing it is a valid outcome.

STEP 13 is not a fairness policy among branches.

---

# 41. Minimum viable branch budget

Some branches require:

\[
c_i\geq c_i^{min}
\]

to function.

Example:

- multihead attention divisibility;
- minimum Conv channels;
- latent shape contracts.

Feasible choices must honor implementation constraints.

---

# 42. Budget quanta

For initial Conv-width experiment:

\[
c_i
\in
\{0,8,16,32,64,96,128\}
\]

as a candidate grid.

This is illustrative.

Actual set should reflect:

- current architecture;
- memory;
- divisibility;
- reasonable runtime.

Do not use a dense integer sweep.

---

# 43. Two-stage experiment to limit combinatorial explosion

## Stage 13A — one-dimensional sensitivity

Per-branch width sweep with others fixed.

## Stage 13B — allocation search

Use sensitivity curves to define a small discrete candidate set.

Then run:

- equal;
- current ratio;
- greedy;
- DP/knapsack;
- selected interaction-aware allocations.

This prevents full Cartesian search.

---

# 44. Width supernetwork is deferred

A Once-for-All/slimmable implementation could later allow many widths from one training run.

But it changes training semantics.

Therefore the primary evidence should come from explicit trained configurations.

Only after STEP-13 usefulness is established should a supernetwork be considered for efficient deployment/search.

---

# 45. Weight sharing confound

Comparing independently trained widths measures:

\[
\text{best representation at each width}
\]

but costs more training.

Comparing widths inside one shared supernetwork measures:

\[
\text{conditional subnet performance}
\]

but introduces weight-sharing bias.

These are different experiments.

Do not mix their results.

---

# 46. Budget types should be tested sequentially

Recommended order:

1. branch channel width;
2. measured FLOPs/latency;
3. retained representation dimensions/bins;
4. expert activations/tokens;
5. C6 rate-based budget if later available.

Do not construct one weighted “resource score” prematurely.

---

# 47. Latent budget allocation

For branch encoder output:

\[
Z_i
\in
\mathbb R^{d_i},
\]

constraint:

\[
\sum_i d_i
\leq
D.
\]

This is the cleanest general-domain formulation.

Project-specific concern:

current branches may emit time-indexed channel tensors rather than one flat latent vector.

Use equivalent channel-width budget at fusion input.

---

# 48. Frequency-bin budget allocation

For spectral branch:

\[
F_i
\]

retained bins/bands.

Constraint:

\[
\sum_i F_i
\leq
F_{max}.
\]

Possible allocation signal:

- STEP-03 spectral SNR;
- STEP-06 predictive importance;
- STEP-12 current quality.

This is similar to classical bit loading but must be empirically validated.

---

# 49. Detector-budget allocation

STEP-07 detector bank may contain:

\[
K_i
\]

templates/kernels.

Budget:

\[
\sum_i K_i
\leq
K_{max}.
\]

Allocate more detectors to representations where additional templates produce higher marginal validation value.

---

# 50. Expert-activation budget

For MoE-like branches:

\[
k_i
\]

active experts.

Budget:

\[
\sum_i k_i
\leq
K_{active}.
\]

Alloc-MoE becomes a strong implementation/reference baseline for this later case.

---

# 51. Token/time-step budget

If a branch uses a Transformer:

\[
T_i
\]

processed tokens/time points.

Budget:

\[
\sum_i T_i
\leq
T_{budget}.
\]

Dynamic token-computation literature provides prior art for:

- pruning;
- merging;
- routing.

This is not first-line for the current Conv/LSTM predictor.

---

# 52. Modality/branch cardinality budget

Simplest budget:

\[
|S|
\leq
K.
\]

Select subset of branches.

Efficient Modality Selection provides the closest theoretical baseline.

This is useful before fine-grained width allocation.

---

# 53. Static branch-selection experiment

For \(K\) current branches, evaluate all subsets if:

\[
2^K
\]

is small.

For four branches:

\[
2^4=16.
\]

This is tractable and should be done.

It provides exact branch-subset evidence before any heuristic allocator.

---

# 54. Current four-branch exhaustive subset test

For:

- CLOSE;
- HF15;
- HF30;
- point/context;

evaluate all 16 inclusion subsets under a matched downstream fusion budget where practical.

This directly estimates:

- complementarity;
- redundancy;
- branch necessity.

---

# 55. Fixed total fusion width

To ensure fairness, keep shared fusion/output budget fixed while branch inclusion/width changes.

Otherwise removing a branch can indirectly shrink the entire model.

---

# 56. Parameter-matched design

For each allocation:

\[
\#\theta(\mathbf c)
\approx
B_\theta
\]

within tolerance.

If width changes also alter downstream head size, compensate or freeze head architecture.

The first experiment should isolate pre-fusion branch capacity.

---

# 57. Current composite implementation concern

In the existing composite plugin, branch outputs concatenate and are then fused to a shared width before horizon-specific heads.

Therefore STEP 13 should initially change:

- branch output filters;

while keeping:

- fused width;
- head widths;
- BiLSTM units;
- Bayesian head structure

fixed.

This isolates the allocation question.

---

# 58. MIMO-inspired orthogonalized allocation

Advanced experiment only.

Let branch latents:

\[
Z
=
[Z_1,\ldots,Z_K].
\]

Estimate covariance/canonical correlations.

Transform into approximately orthogonal components:

\[
\tilde Z
=
V^T Z.
\]

Allocate capacity to components rather than original branches.

This is conceptually similar to MIMO eigenmode decomposition.

But interpretability decreases and it overlaps STEP 09.

Therefore it is a late-stage diagnostic, not primary architecture.

---

# 59. Redundancy penalty

A practical branch allocation score may be:

\[
Score_i
=
\frac{
\Delta P_i
}{
\Delta C_i
}
-
\lambda_R Redundancy_i.
\]

Redundancy may come from:

- conditional code gain;
- CCA;
- latent correlation.

Do not add this penalty until plain marginal utility is characterized.

---

# 60. SNR-aware allocation

From STEP 03:

\[
SNR_i.
\]

Hypothesis:

higher-quality branches may justify more resolution/width.

But:

\[
SNR_i
\]

is not equivalent to predictive utility.

Therefore compare:

\[
\text{SNR allocation}
\]

against:

\[
\text{validation marginal-utility allocation}.
\]

---

# 61. Information-aware allocation

Possible branch evidence:

\[
I_i
=
[
SNR_i,\;
G_i^{conditional},\;
\Delta P_i,\;
OOD_i,\;
stability_i
].
\]

A later allocator can learn resource scores from these.

First experiment uses direct validation utility.

---

# 62. Static versus state-conditioned allocation

## Static STEP-13A

\[
\mathbf c_t
=
\mathbf c^*
\]

for all \(t\).

## Dynamic STEP-13B

\[
\mathbf c_t
=
g(
q_t,B_t
).
\]

Dynamic allocation is eligible only if:

- STEP 12 routing is supported;
- multiple static allocations dominate in different quality regimes.

---

# 63. Dynamic water-filling analogue

Suppose branch quality changes:

\[
q_{i,t}.
\]

A state-conditioned allocator could approximate:

\[
c_{i,t}
=
g(
q_{i,t},
\lambda_t
).
\]

But it must select from hardware-friendly widths.

No continuous arbitrary channel count is required at runtime.

---

# 64. Budget-conditioned subnetwork family

If dynamic allocation proves useful, a slimmable/OFA-style family can support:

\[
c_i
\in
\mathcal D_i
\]

without storing a separate model for every allocation.

This is a deployment optimization, not the first scientific experiment.

---

# 65. Missing branch/action feasibility

If a branch is unavailable:

\[
A_i(t)=0,
\]

force:

\[
c_i(t)=0.
\]

Freed budget can be:

- left unused;
- reallocated among available branches.

This belongs to dynamic STEP-13B.

---

# 66. Budget reserve

Not all resource must be allocated.

Allow:

\[
\sum_i C_i(c_i)
<
B.
\]

If additional capacity has no utility, spare compute is a legitimate optimum.

This mirrors channels receiving zero power under water-filling.

---

# 67. Training budget versus inference budget

Distinguish:

## Model capacity budget

parameters/width.

## Runtime budget

FLOPs/latency.

## Data/input budget

features/bins/tokens.

A configuration can be parameter-efficient but runtime-expensive or vice versa.

Every experiment names its budget class explicitly.

---

# 68. Multi-objective Pareto frontier

For configuration \(\mathbf c\), record:

\[
(
P,
latency,
memory,
FLOPs,
parameters
).
\]

Compute non-dominated frontier.

There is no single “best” allocation without a deployment constraint.

---

# 69. Budget robustness

An allocation optimized for:

\[
B_1
\]

may be poor at:

\[
B_2.
\]

Measure:

\[
\mathbf c^*(B)
\]

over several budgets.

This tests whether proportions remain stable.

---

# 70. Allocation stability across time blocks

Fit allocation using training/development blocks.

Compare preferred allocation across:

- years;
- regimes;
- seeds.

A wildly unstable static allocation suggests dynamic STEP-12/13B or insufficient evidence.

---

# 71. Branch marginal-utility stability

For branch \(i\):

\[
g_i^{(block)}
=
\frac{
\Delta P_i
}{
\Delta C_i
}.
\]

Measure rank stability of branches across blocks.

This is more informative than one global importance score.

---

# 72. Allocation overfitting risk

Searching many allocations on one validation year can overfit.

Controls:

- small candidate set derived from Stage 13A;
- nested/blocked internal development split;
- predeclared search algorithm;
- final validation selection;
- one test.

Do not use unconstrained NAS.

---

# 73. No full NAS in STEP 13

Neural Architecture Search literature covers resource-constrained architecture optimization.

But STEP 13's hypothesis is narrower:

> does **branch budget allocation** matter?

Full NAS changes:

- depth;
- operators;
- connectivity;
- activation;
- width;
- fusion.

That would destroy attribution.

---

# 74. Hardware-aware NAS as late-stage reference

If deployment becomes primary, hardware-aware NAS literature provides methods for optimizing:

- latency;
- FLOPs;
- energy;
- memory.

Use only after branch-allocation value is established.

---

# 75. Allocation under uncertainty

Let marginal utility estimates have uncertainty:

\[
\hat g_i
\pm
\sigma_i.
\]

Conservative allocation can use:

\[
g_i^{LCB}
=
\hat g_i
-
\kappa\sigma_i.
\]

This prevents spending most capacity on an unstable apparent winner.

This is an optional robust allocator.

---

# 76. Exploration is not needed for offline STEP 13A

Every candidate allocation is evaluated offline.

No contextual bandit/RL is needed.

Online adaptive allocation may later use bandit methods if full counterfactual expert performance is unavailable.

That belongs beyond the first protocol.

---

# 77. Branch-budget regularization

Instead of hard allocation, one may train with penalty:

\[
\mathcal L
=
\mathcal L_{task}
+
\lambda
C(\mathbf c).
\]

This makes cost part of training.

But the primary experiment should use hard matched budgets so results are easy to interpret.

---

# 78. Lagrangian view

Constrained problem:

\[
\max P(\mathbf c)
\quad
s.t.
\quad
C(\mathbf c)\leq B.
\]

Equivalent candidate Lagrangian:

\[
\boxed{
\max_{\mathbf c}
P(\mathbf c)
-
\lambda C(\mathbf c).
}
\]

Sweep \(\lambda\) to approximate Pareto frontier.

This is a later optimization method.

---

# 79. Extreme-event preservation

Low-budget allocation can preferentially preserve common smooth signals while dropping rare branches.

Therefore report performance on:

- high-volatility windows;
- event windows;
- tail targets.

An allocation that improves average MAE by starving the rare-event branch may be unacceptable.

---

# 80. Diversity reserve

A possible policy is minimum branch allocation:

\[
c_i\geq c_{min}
\]

for selected critical branches.

But this is an application constraint, not a scientific assumption.

First experiment allows zero unless architecture prohibits it.

---

# 81. Head-specific branch value

Different forecast horizons may use different branches.

Define:

\[
P_h(\mathbf c).
\]

A future extension can allocate per horizon.

First experiment uses one shared allocation across all heads to keep scope controlled.

---

# 82. Task-specific allocation

If multiple applications consume the same latent:

\[
Y^{(1)},\ldots,Y^{(T)},
\]

allocation may optimize weighted utility:

\[
P
=
\sum_t
w_t P_t.
\]

This is a domain-general extension after the forecasting case.

---

# 83. Minimum public benchmark

Use a small set only:

- ETTh1;
- Weather;
- Electricity.

Purpose:

- test allocation methodology;
- demonstrate it is not finance-specific.

No need to rerun every STEP benchmark.

---

# 84. Project benchmark

Primary project host:

\[
\boxed{
\text{existing composite predictor}
}
\]

because it already has unequal fixed branch widths and clear branch semantics.

Run on the project's governed train/validation/test split.

---

# 85. Static experiment matrix

| ID | Allocation strategy | Budget | Interaction-aware | Hardware-aware |
|---|---|---:|---:|---:|
| B00 | equal | \(B\) | No | No |
| B01 | current ratio | \(B\) | No | No |
| B02 | single branch | \(B\) | No | No |
| B03 | leave-one-out | \(B\) | diagnostic | No |
| B04 | greedy marginal | \(B\) | No | cost-aware |
| B05 | knapsack/DP | \(B\) | No | cost-aware |
| B06 | pairwise interaction | \(B\) | Yes | cost-aware |
| B07 | SNR-based | \(B\) | No | No |
| B08 | redundancy-aware | \(B\) | Yes | No |
| B09 | selected Pareto champion | multiple \(B\) | best | measured |

---

# 86. Budget sweep

Recommended normalized budgets relative to current branch-output budget:

\[
\boxed{
B/B_{current}
\in
\{0.5,\;0.75,\;1.0,\;1.25\}
}
\]

provided architecture supports them.

Do not automatically expand beyond this grid.

---

# 87. Current budget definition

Current branch channels:

\[
64+32+32+32
=
160.
\]

Thus illustrative total channel budgets:

\[
80,\;120,\;160,\;200.
\]

Actual allocation candidates must respect minimum channel constraints.

---

# 88. Example equal allocations

For:

\[
B=160
\]

equal:

\[
[40,40,40,40]
\]

or nearest architecture-valid multiples.

Current:

\[
[64,32,32,32].
\]

The experiment directly tests whether the existing CLOSE-heavy prior is useful.

---

# 89. Candidate width set

For each branch:

\[
c_i
\in
\{0,16,32,48,64,80,96\}
\]

as an illustrative structured grid.

Agents should prune infeasible values before experiment.

---

# 90. Core/head freeze

To isolate branch allocation, freeze:

- fusion output width;
- head Conv stack;
- BiLSTM units;
- Bayesian head;
- optimizer;
- target;
- horizon set;
- sequence length;
- feature set.

Only branch widths change.

---

# 91. Reinitialization and retraining

Each allocation should be trained from scratch under the same seed set.

Do not transplant weights from wider models in the primary experiment.

Weight inheritance is a later deployment optimization.

---

# 92. Seed protocol

Use identical deterministic seed set across allocations.

Recommended minimum:

\[
3
\]

seeds.

Allocation claims require stability beyond seed noise.

---

# 93. Primary metric

Choose one preregistered task metric:

\[
P.
\]

Secondary:

- MAE;
- RMSE;
- \(R^2\);
- uncertainty/calibration;
- tail/event performance.

The allocation optimizer itself uses only the primary development metric plus declared cost.

---

# 94. Allocation efficiency metric

For two allocations:

\[
a,b
\]

define:

\[
\boxed{
\eta_{ab}
=
\frac{
P_a-P_b
}{
C_a-C_b
}
}
\]

when denominators/signs are meaningful.

More useful is the full Pareto frontier than one efficiency number.

---

# 95. Regret relative to best searched allocation

For budget \(B\):

\[
P^*(B)
=
\max_{\mathbf c\in\mathcal C_B}
P(\mathbf c).
\]

Algorithm regret:

\[
\boxed{
R_{alloc}
=
P^*(B)
-
P(\hat{\mathbf c}(B)).
}
\]

On small search spaces, exact enumeration can approximate \(P^*\).

---

# 96. Four-branch subset oracle

For current 4 branches, all inclusion subsets are cheap enough to enumerate.

This gives an exact **subset oracle**, though not exact width oracle.

Use it to validate greedy selection assumptions.

---

# 97. Dynamic extension after STEP 12

If STEP 12 establishes quality-conditioned mode differences, dynamic STEP-13 allocation can use:

\[
q_t
\]

and budget:

\[
B_t.
\]

Output:

\[
\mathbf c_t.
\]

But allocations should come from a small hardware-supported catalog:

\[
\mathcal C
=
\{
\mathbf c^{(1)},\ldots,\mathbf c^{(M)}
\}.
\]

Do not resize arbitrary TensorFlow layers every bar.

---

# 98. Catalog-based dynamic allocation

Train/evaluate a set of feasible submodels:

\[
\mathcal C.
\]

STEP-12-like router selects allocation catalog entry conditional on:

\[
(q_t,B_t).
\]

This is more practical than continuous runtime architecture mutation.

---

# 99. State-conditioned allocation versus mode routing

A mode can contain one allocation.

Example:

\[
a_{lowSNR}
=
(
\text{robust representation},
\mathbf c^{robust}
).
\]

The conceptual distinction remains:

- STEP 12 determines operating mode;
- STEP 13 determines/optimizes capacity layout within modes.

---

# 100. Falsifiable hypotheses

## H13.1 — Equal branch allocation is not generally optimal

For at least one budget:

\[
P(\mathbf c^*)
>
P(\mathbf c_{equal})
\]

under matched total cost.

**Falsified if:** equal allocation matches every optimized candidate within equivalence margin.

---

## H13.2 — The current fixed ratio is empirically testable and may or may not be optimal

\[
[64,32,32,32]
\]

should be compared directly against optimized allocations at:

\[
B=160.
\]

**Falsified as an improvement thesis if:** current ratio remains Pareto-optimal.

---

## H13.3 — Branches exhibit diminishing marginal returns

For at least some branches:

\[
\Delta_i(c+\delta)
<
\Delta_i(c).
\]

**Falsified if:** utility is strongly non-concave/thresholded.

This determines whether water-filling-style greedy allocation is appropriate.

---

## H13.4 — Marginal utility per cost predicts good allocations

Greedy:

\[
\Delta P/\Delta C
\]

allocation approaches the best searched allocation.

**Falsified if:** interactions dominate and greedy regret is large.

---

## H13.5 — Branch redundancy changes optimal allocation

Highly redundant branches should receive less joint capacity than equally useful but complementary branches.

**Falsified if:** redundancy diagnostics do not predict allocation shifts.

---

## H13.6 — Allowing zero allocation can improve constrained performance

Under tight budget:

\[
c_i=0
\]

for low-value branches can outperform forcing all branches active.

**Falsified if:** every branch consistently requires positive capacity.

---

## H13.7 — Optimal allocation depends on total budget

\[
\frac{\mathbf c^*(B_1)}{B_1}
\neq
\frac{\mathbf c^*(B_2)}{B_2}
\]

for at least some budgets.

**Falsified if:** one fixed proportion scales optimally across all budgets.

---

## H13.8 — SNR alone is insufficient for allocation

A pure SNR-based allocation should be inferior to target-utility-aware allocation in at least some conditions.

**Falsified if:** SNR ranking fully predicts branch value.

---

## H13.9 — Static optimized allocation can reduce required capacity

There exists:

\[
B'<B_{current}
\]

with:

\[
P^*(B')
\approx
P(\mathbf c_{current}).
\]

**Falsified if:** current budget is already necessary.

---

## H13.10 — Theoretical FLOP savings do not guarantee latency savings

Some lower-FLOP allocations will fail to achieve proportional measured-latency improvement.

This is a protective systems hypothesis.

---

## H13.11 — Dynamic allocation only helps if preferred static allocations differ by information state

If quality regimes favor distinct:

\[
\mathbf c^*_{regime},
\]

dynamic allocation may improve over one global allocation.

**Falsified if:** one static allocation dominates all states.

---

## H13.12 — Resource allocation can be generalized across domains at the contract level

The same constrained optimization framework should apply to different branch semantics, even if utility/cost curves differ.

**Falsified if:** no stable budget abstraction can be defined outside the financial host.

---

# 101. Gate 13A — branch usefulness

At least two branches must have positive incremental validation value.

If one branch dominates and others add nothing:

\[
\boxed{
\text{STOP multi-branch allocation; use the dominant branch.}
}
\]

---

# 102. Gate 13B — budget sensitivity

Performance must change meaningfully with allocation under fixed total budget.

If all feasible allocations tie:

\[
\boxed{
\text{allocation optimization is unnecessary}.
}
\]

---

# 103. Gate 13C — marginal-utility stability

Per-branch sensitivity curves must be reproducible enough across seeds/blocks to guide search.

If not:

\[
\boxed{
\text{do not fit a complex allocator to noise}.
}
\]

---

# 104. Gate 13D — simple allocator

Greedy/DP allocation must beat or match equal/current baselines before advanced interaction-aware methods.

---

# 105. Gate 13E — interaction justification

Only introduce pairwise/Shapley/submodular machinery if:

\[
P(\mathbf c)
\]

cannot be predicted sufficiently from branchwise utility curves.

---

# 106. Gate 13F — real resource benefit

A “budget-saving” configuration must improve at least one measured deployment quantity:

- parameters;
- memory;
- FLOPs;
- latency.

If only theoretical width changes but runtime does not:

\[
\boxed{
\text{deployment claim rejected}.
}
\]

---

# 107. Gate 13G — tail/event safety

Resource reduction/allocation must not materially destroy critical tail/event performance.

---

# 108. Gate 13H — held-out confirmation

Only one static allocation per declared budget, or one frozen catalog/router if dynamic extension is later approved, reaches final test.

---

# 109. Allocation search budget

The allocation search itself has cost.

Set maximum candidate count:

\[
N_{alloc}^{max}.
\]

Recommended initial target:

\[
\boxed{
N_{alloc}^{max}\leq 30
}
\]

per total budget after one-dimensional sensitivity sweeps.

This prevents architecture-search creep.

---

# 110. Recommended implementation order

1. Freeze current four-branch composite architecture.
2. Freeze core/head and experiment protocol.
3. Enumerate 16 branch subsets.
4. Run one-dimensional width sensitivity per branch.
5. Build marginal gain/cost curves.
6. Compare equal vs current ratio.
7. Run greedy allocation.
8. Run DP/knapsack allocation.
9. Measure pairwise interactions.
10. Add interaction-aware allocation only if needed.
11. Sweep 3–4 total budgets.
12. Measure FLOPs/params/latency.
13. Audit tail/event performance.
14. Validate on one small public benchmark.
15. Freeze best static allocation(s).
16. Only after STEP 12 support, test dynamic allocation catalog.
17. Only later consider slimmable/OFA implementation.

---

# 111. Required artifacts

1. `step13_branch_contract.json`
2. `step13_budget_definition.json`
3. `step13_subset_ablation.parquet`
4. `step13_branch_sensitivity.parquet`
5. `step13_marginal_utility.parquet`
6. `step13_pairwise_synergy.parquet`
7. `step13_allocation_candidates.parquet`
8. `step13_greedy_allocations.json`
9. `step13_dp_allocations.json`
10. `step13_budget_frontier.parquet`
11. `step13_compute_latency.parquet`
12. `step13_tail_event_metrics.parquet`
13. `step13_allocation_stability.parquet`
14. `step13_statistical_tests.json`
15. `step13_audit_report.md`
16. reproducibility manifest:
    - branch definitions;
    - widths;
    - parameter counts;
    - fusion/head freeze;
    - dataset hashes;
    - split dates;
    - seeds;
    - hardware;
    - precision;
    - batch size;
    - latency protocol;
    - commit hashes.

---

# 112. Latency measurement protocol

Report:

- hardware;
- batch size;
- warm-up runs;
- measured repetitions;
- median;
- p95;
- preprocessing included/excluded;
- router overhead if applicable.

Do not compare latency numbers obtained under different timing protocols.

---

# 113. Compute-accounting boundary

Include:

- branch transforms if executed at inference;
- branch encoder;
- fusion;
- routing/allocation overhead.

Do not count only core model FLOPs while ignoring expensive spectral/wavelet preprocessing.

---

# 114. Public benchmark philosophy

The public benchmark is not intended to prove global SOTA.

It tests whether:

\[
\boxed{
\text{branch allocation methodology}
}
\]

transfers outside finance.

Use small reproducible models.

---

# 115. Project-specific scientific question

The current composite predictor already assumes a prior:

\[
\text{CLOSE branch deserves twice the channel width of each other branch.}
\]

STEP 13 converts that engineering decision into a falsifiable research question.

This is exactly the kind of hypothesis the larger pipeline was designed to expose.

---

# 116. Strongest connection to previous steps

## STEP 03

SNR can be a quality signal but not the allocation objective.

## STEP 05

Conditional code gain identifies redundant branch information.

## STEP 06

Different representation branches expose different signal domains.

## STEP 07

Detector banks consume finite template/compute budgets.

## STEP 09

Common/private decomposition reduces duplicated capacity.

## STEP 12

Information quality can decide which allocation catalog/mode is appropriate.

Thus STEP 13 is where the preceding information diagnostics become an explicit **resource economics** problem.

---

# 117. General domain contract

For domain \(D\), define branches:

\[
B_1,\ldots,B_K.
\]

Each has:

- representation contract;
- resource-cost function:
  \[
  C_i(c_i);
  \]
- validation utility;
- redundancy with other branches;
- minimum/maximum capacity.

Then solve:

\[
\boxed{
\max_{\mathbf c}
P_D(\mathbf c)
\quad
s.t.
\quad
C_D(\mathbf c)
\leq B.
}
\]

This contract can apply to:

- finance;
- biomedical sensors;
- robotics;
- industrial monitoring;
- weather;
- energy;
- communications;
- multimodal ML.

---

# 118. State-of-the-art synthesis

The review produces six major conclusions.

## 118.1. Uniform allocation is not the default optimum

Classical parallel-channel theory and practical bit loading already established this in communications.

## 118.2. Correlation changes allocation

MIMO/eigenmode theory shows why resource allocation should account for coupled/redundant channels.

## 118.3. ML now treats inference as a budgeted allocation problem

Modern systems allocate:

- modalities;
- expert activations;
- tokens;
- width;
- depth.

## 118.4. Marginal sensitivity profiling is current practice

Alloc-MoE 2026 uses sensitivity profiling and dynamic programming under activation budgets.

## 118.5. Time-series expert allocation already exists

Frequency/resolution MoE work means generic spectral-expert allocation is not novel.

## 118.6. Realized latency is mandatory

Dynamic-compute literature shows theoretical compute reduction frequently overstates actual deployment gain.

---

# 119. Project-specific opportunity

The likely distinctive contribution is not:

> “allocate more neurons to useful branches.”

It is the integrated methodology:

\[
\boxed{
\text{SNR}
+
\text{conditional redundancy}
+
\text{representation value}
+
\text{detector value}
+
\text{synchronization confidence}
+
\text{OOS marginal utility}
\rightarrow
\text{budgeted branch allocation}.
}
\]

That connects the entire communications/information-processing chain to an explicit finite-resource design problem.

---

# 120. References — IEEE style

[1] I. E. Telatar, “Capacity of Multi-antenna Gaussian Channels,” *European Transactions on Telecommunications*, vol. 10, no. 6, pp. 585–595, 1999, doi: 10.1002/ett.4460100604. Available: https://doi.org/10.1002/ett.4460100604

[2] G. J. Foschini and M. J. Gans, “On Limits of Wireless Communications in a Fading Environment when Using Multiple Antennas,” *Wireless Personal Communications*, vol. 6, pp. 311–335, 1998, doi: 10.1023/A:1008889222784. Available: https://doi.org/10.1023/A:1008889222784

[3] P. S. Chow, J. M. Cioffi, and J. A. C. Bingham, “A Practical Discrete Multitone Transceiver Loading Algorithm for Data Transmission over Spectrally Shaped Channels,” *IEEE Transactions on Communications*, vol. 43, pp. 773–775, 1995, doi: 10.1109/26.380108. Available: https://doi.org/10.1109/26.380108

[4] Y. He, R. Cheng, G. Balasubramaniam, Y.-H. H. Tsai, and H. Zhao, “Efficient Modality Selection in Multimodal Learning,” *Journal of Machine Learning Research*, vol. 25, no. 47, pp. 1–39, 2024. Available: https://www.jmlr.org/papers/v25/23-0439.html

[5] B. Hu, L. Xu, J. Moon, N. J. Yadwadkar, and A. Akella, “MOSEL: Inference Serving Using Dynamic Modality Selection,” in *Proceedings of EMNLP*, 2024, pp. 8872–8886, doi: 10.18653/v1/2024.emnlp-main.501. Available: https://aclanthology.org/2024.emnlp-main.501/

[6] C. Li, G. Wang, B. Wang, X. Liang, Z. Li, and X. Chang, “Dynamic Slimmable Network,” in *Proceedings of CVPR*, 2021, pp. 8607–8617. Available: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dynamic_Slimmable_Network_CVPR_2021_paper.html

[7] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, “Once-for-All: Train One Network and Specialize It for Efficient Deployment,” in *International Conference on Learning Representations*, 2020. Available: https://hanlab.mit.edu/projects/ofa

[8] B. Liu, K. Tian, W. Wang, Z. Zhang, L. Qiao, and D. Li, “Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference,” in *Proceedings of ACL*, 2026, pp. 9653–9667, doi: 10.18653/v1/2026.acl-long.437. Available: https://aclanthology.org/2026.acl-long.437/

[9] J. Li, P. Jiang, C. Tian, J. Liu, Z. Zhang, and X. Hu, “Optimal Expert-Attention Allocation in Mixture-of-Experts: A Scalable Law for Dynamic Model Design,” in *Proceedings of ACL Industry Track*, 2026, pp. 1406–1418, doi: 10.18653/v1/2026.acl-industry.98. Available: https://aclanthology.org/2026.acl-industry.98/

[10] “A Survey of Dynamic Token Computation in Transformers: Taxonomy, Stability, and Budget-Aware Evaluation,” *IEEE Access*, 2026, doi: 10.1109/ACCESS.2026.3720106. Available: https://doi.org/10.1109/ACCESS.2026.3720106

[11] Y. Huang, R. Zou, Y. Wang, L. Aslam, and R. Dong, “M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 26, pp. 22075–22083, 2026, doi: 10.1609/aaai.v40i26.39362. Available: https://doi.org/10.1609/aaai.v40i26.39362

[12] “Adaptive Frequency-Band Mixture of Experts for Long-Horizon Multivariate Time-Series Forecasting,” *Computers & Electrical Engineering*, vol. 139, art. 111465, 2026, doi: 10.1016/j.compeleceng.2026.111465. Available: https://doi.org/10.1016/j.compeleceng.2026.111465

[13] “Load-Balancing Strategies for Forecasting with Mixture-of-Experts Architecture,” *Procedia Computer Science*, vol. 272, pp. 155–162, 2025, doi: 10.1016/j.procs.2025.10.191. Available: https://doi.org/10.1016/j.procs.2025.10.191

[14] Z. Xu, K. D. Nguyen, P. Mukherjee, S. Bagchi, S. Chaterji, Y. Liang, and Y. Li, “Learning to Inference Adaptively for Multimodal Large Language Models,” in *Proceedings of ICCV*, 2025, pp. 3552–3563. Available: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Learning_to_Inference_Adaptively_for_Multimodal_Large_Language_Models_ICCV_2025_paper.html

[15] L. Yang, Y. Han, X. Chen, S. Song, J. Dai, and G. Huang, “Resolution Adaptive Networks for Efficient Inference,” in *Proceedings of CVPR*, 2020, pp. 2369–2378. Available: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Resolution_Adaptive_Networks_for_Efficient_Inference_CVPR_2020_paper.html

---

# 121. Final status

**STEP 13 is theoretically specified after a dedicated classical and current state-of-the-art review and is ready for independent agent audit.**

The central claim is deliberately modest and falsifiable:

\[
\boxed{
\text{useful multi-branch systems should allocate finite capacity according to marginal task value and redundancy, rather than assuming uniform or hand-fixed allocation is always optimal.}
}
\]

The first experiment is not a learned allocator.

It is:

\[
\boxed{
\text{measure the branch utility curves and the budget frontier}.
}
\]

Only after that evidence exists should the project consider:

- greedy water-filling analogues;
- knapsack/DP;
- interaction-aware allocation;
- dynamic catalog selection;
- slimmable/OFA deployment.

This completes the planned main communications-derived chain through STEP 13.
