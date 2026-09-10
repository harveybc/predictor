# WORKPLAN PATCH 002
## Agent-Audit Constraints and Boundary Clarifications

**Status:** Normative clarification patch  
**Date:** 2026-09-05  
**Applies to:** Master Work Plan v2 and STEPS 07–13.

---

# 1. STEP 08 opening condition

STEP 08 “equalization” is not equivalent to ordinary z-score normalization.

A STEP-08 experiment must identify an operational channel/distortion, such as:

- source/vendor;
- instrument;
- venue;
- volatility regime;
- controlled transfer function;
- frequency response;
- covariance/domain shift.

Existing project z-score normalization remains a baseline only.

---

# 2. Whitening boundary

Whitening used only to maximize matched-filter/detector performance remains within STEP 07.

Whitening becomes STEP 08 only when it is part of an operational source/channel canonicalization objective.

---

# 3. STEP 09 side information

Compression lane C4 begins with the existing STEP-05 conditional-code-gain concept:

\[
G_{i\to j}
=
L_j^{self}
-
L_j^{cond(i)}.
\]

The first deliverable is a table/graph.

No new side-information neural architecture is authorized before that evidence exists.

C4 is operationally shared between:

- STEP 05;
- STEP 09;
- later STEP 13 allocation.

---

# 4. Sparse/CSC gate

Compression items C1–C2 remain deferred until STEP-07 detector gate evidence exists.

Any sparse/CSC detector must beat or complement a strong inexpensive baseline such as MiniRocket under a fair compute/performance comparison.

No custom K-SVD implementation is planned.

---

# 5. C6 latent rate–distortion

C6 belongs to the existing feature-extractor/core latent representation after SNR/representation experiments.

It is not interpreted as:

- a MacKay neuron-capacity bound;
- the information quantity of STEP 02;
- network Kolmogorov complexity.

---

# 6. STEP 11 scope reduction

STEP 11 does not begin by constructing another autoencoder.

The project already has feature-extractor/reconstruction infrastructure.

The initial STEP-11 question becomes:

> Does controlled corruption/masking/noise applied while training the existing extractor produce a latent representation that is more robust OOS?

Only modifications/extensions to the existing extractor architecture are eligible initially.

---

# 7. STEP 09 cancellation semantics

STEP 09 does not classify shared/common components as noise.

Its safe initial decomposition is:

\[
X=C+U
\]

and the initial model comparison must retain:

\[
[X,C,U].
\]

Destructive cancellation requires target-preservation evidence.

---

# 8. STEP 10 boundary

Dynamic timing/lag optimization belongs to STEP 10.

STEP 09 may use fixed causal lag models such as VAR but must not hide dynamic synchronization/timing recovery inside the cancellation optimizer.

---

# 9. Governance

This patch does not invalidate prior STEP documents.

Where wording conflicts, this patch governs the boundaries of future implementation.
