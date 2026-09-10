# WORKPLAN PATCH 003
## Final Review: Causal Event Lane, Level-3 Meta-Optimization, and Normative Scope Corrections

**Date:** 2026-09-05  
**Status:** Normative final roadmap patch  
**Applies to:** Master Work Plan v2, PATCH 001, PATCH 002, STEPS 01–13.

---

# 1. Main-chain completion

The communications-derived main chain ends at STEP 13.

No STEP 14 is added merely to extend the analogy.

Future work is organized as:

- implementation/evidence work packages;
- compression transversal lane;
- causal economic-event lane;
- Level-3 meta-optimization lane.

---

# 2. Pipeline is not strictly serial

STEPS 01–13 define research questions/modules, not a mandate to compose all transforms in sequence.

Only configurations surviving their own gates may enter downstream branch banks.

---

# 3. STEP 01 amendment

Add an observation-operator contract distinguishing:

- point sampling;
- interval aggregation;
- timestamp label;
- finalization/availability time;
- asynchronous observation.

OHLC bars are interval aggregates, not ordinary point samples.

---

# 4. STEP 08 narrowing

STEP 08 core requires an operational distortion/channel.

Generic:

- DANN;
- optimal transport domain adaptation;
- broad TTA

are optional domain-shift extensions, not default equalization.

Whitening that exists only for STEP-07 detector optimality remains STEP 07.

---

# 5. Causal Event Response Lane — new cross-cutting lane

Economic-calendar/event records remain event-indexed data.

Do not force them into a uniformly sampled series.

Canonical event panel:

```text
event_id
publication_time
event_family
country/currency
importance
consensus
actual
standardized_surprise
pre-event covariates
post-event outcomes by horizon
```

Initial methods:

1. event-study/local projection;
2. DML;
3. CausalForestDML / heterogeneous treatment effects;
4. LP-IV / high-frequency identification only where identification is defensible.

Outputs are causal-response knowledge artifacts/priors, not raw causal labels.

---

# 6. Event-token boundary

Existing event-token Transformer:

\[
\text{representation/context}.
\]

Causal event lane:

\[
\text{effect estimation}.
\]

They may be concatenated but are not substitutes.

---

# 7. Causal-inference usage boundary

Causal methods require:

- explicit treatment;
- outcome;
- identification assumptions;
- point-in-time confounders;
- temporal ordering.

Do not use DML/NOTEARS simply to choose:

- FFT bins;
- denoising strength;
- quantization;
- routing when all expert counterfactual losses can be replayed;
- allocation when all candidates can be directly evaluated.

---

# 8. `causal-inference` repository status

The committed repository is experimental/unverified and retains inherited `rl-optimizer` packaging identity.

Before becoming common infrastructure:

- repair package identity;
- pin dependencies;
- add tests;
- add point-in-time data contracts;
- separate discovery from effect estimation.

---

# 9. Level-3 Meta-Optimization Lane — new cross-cutting lane

Formal hierarchy:

- L1 = individual candidate train/evaluate;
- L2 = DEAP/NEAT/DOIN search;
- L3 = meta-learned warm start/surrogate/algorithm selection from historical evidence.

L3 recommends priors/top-k seeds to L2.

L3 does not bypass L2 or held-out validation.

---

# 10. L3 canonical record

Every candidate evaluation should preserve:

```text
task meta-features
typed genome
parameter active mask
search-space version
optimizer/campaign/generation
parentage/provenance
fidelity/budget
full metric vector
compute/runtime
failure/status
artifact/protocol hashes
```

Failed candidates remain evidence.

---

# 11. L3 validation

Do not random-split candidate rows from one optimization campaign.

Hold out entire:

- campaigns;
- tasks;
- assets;
- timeframes;
- model families.

Primary success metric:

\[
\text{best-so-far performance vs L1 evaluations}
\]

under the same L2 budget.

---

# 12. L3 first models

Mandatory baselines before Transformer/meta-RL:

1. nearest-task warm start;
2. GBDT/CatBoost-type surrogate;
3. feasibility classifier + metric regressor.

Advanced OptFormer-style/metaBBO model only after simple L3 passes.

---

# 13. Held-out firewall propagates to L3

Protected test/Stage-C information must not enter:

- task meta-features;
- candidate metrics;
- causal event fitting;
- L3 surrogate training;
- prompts used to recommend parameters.

---

# 14. Multi-paradigm assay rule

Use each project ML capability as an assay:

- regression → forecasting utility;
- classification → detection;
- unsupervised → structural stability;
- causal → interventions/effects;
- RL → late sequential policy utility;
- backtest → application utility.

No single paradigm is a universal validator.

---

# 15. STEP 12 causal boundary

If every frozen expert can be evaluated historically for every sample, per-expert losses are directly observed counterfactual computation.

Do not add DML merely to select the router.

Causal/off-policy methods become relevant only when alternatives are genuinely unobserved or actions change future state.

---

# 16. STEP 13 causal boundary

If allocations can be trained/evaluated offline, optimize the measured response surface directly.

Causal inference is not required to infer candidate allocation effects.

---

# 17. Final implementation priority

1. contracts/integrity;
2. cheap evidence;
3. causal event panel;
4. qualified representations/detectors/decompositions;
5. STEP 11;
6. STEP 12;
7. STEP 13;
8. L3 meta-optimization;
9. backtest/RL/application.

---

# 18. Stop rule

The final deployed architecture must contain only modules with demonstrated incremental value or explicit safety/robustness justification.

The research roadmap is intentionally larger than the final model.
