# WORKPLAN PATCH 001
## Historical Communications-Chain Correction and Compression-Derived Extensions

**Status:** Normative patch for documents STEP 01–STEP 07  
**Purpose:** Correct the forward roadmap without invalidating completed STEP documents, and formalize the compression-derived research lane before continuing to STEP 08.

---

# 1. Why this patch exists

The first seven documents were developed incrementally while reconstructing the telecommunications analogy. Their individual experimental content remains valid, but the **forward historical ordering** should now be normalized.

This patch therefore changes the roadmap, not the scientific results already specified.

---

# 2. Completed main-chain steps

The following steps remain unchanged:

1. **STEP 01 — Sampling / Nyquist**
2. **STEP 02 — Noise / SNR estimation**
3. **STEP 03 — Denoising / signal-preservation validation**
4. **STEP 04 — Quantization / non-uniform quantization / companding**
5. **STEP 05 — Source coding / entropy / contexts / innovation / surprisal / MDL**
6. **STEP 06 — Amplitude / frequency / phase / time-frequency representations**
7. **STEP 07 — Matched filtering / representation-aware pattern detection**

These are considered the completed conceptual/protocol layer, pending agent audit and later experimental execution.

---

# 3. Historical debt: corrected remaining communications chain

The recommended continuation is now:

8. **STEP 08 — Equalization / canonicalization / inverse channel compensation**
9. **STEP 09 — Interference, echo and crosstalk cancellation**
10. **STEP 10 — Synchronization / temporal alignment / timing recovery**
11. **STEP 11 — Channel coding / controlled redundancy / error robustness**
12. **STEP 12 — Adaptive modulation and coding / information-quality-aware routing**
13. **STEP 13 — Multiplexing / MIMO / optimal multi-branch information allocation**

This replaces any earlier informal suggestion that STEP 08 should directly begin with pulse shaping or another receiver block.

Pulse shaping and closely related filtering concepts should be treated inside STEP 08/STEP 10 where their actual role can be analyzed against the project's time-series data model.

---

# 4. Main-chain interpretation

The corrected main chain is:

\[
\text{Sampling}
\rightarrow
\text{Noise/SNR}
\rightarrow
\text{Denoising}
\rightarrow
\text{Quantization}
\rightarrow
\text{Source modeling/coding}
\rightarrow
\text{Representation domains}
\rightarrow
\text{Pattern detection}
\rightarrow
\text{Equalization}
\rightarrow
\text{Interference cancellation}
\rightarrow
\text{Synchronization}
\rightarrow
\text{Controlled redundancy}
\rightarrow
\text{Adaptive configuration}
\rightarrow
\text{Multi-branch allocation}
\]

This is the normative roadmap until superseded by a later patch.

---

# 5. Compression-derived transversal lane

A separate research lane is now formally recognized.

It is **not** inserted between STEP 07 and STEP 08 because several of its concepts operate across multiple stages.

The lane contains:

1. **Sparse coding**
2. **Convolutional sparse coding**
3. **Successive refinement / progressive representation**
4. **Conditional coding with side information**
5. **Hierarchical residual coding**
6. **Latent rate-distortion / learned entropy models**
7. **Duration / run-length / event coding**

These concepts should be audited now, but implementation should be scheduled only where they naturally attach to the main-chain experiments.

---

# 6. Integration points

## Sparse coding

Primary attachment points:

- STEP 06 representations
- STEP 07 pattern detector banks

## Convolutional sparse coding

Primary attachment points:

- STEP 07 template/pattern detection
- STEP 08 canonicalized inputs

## Successive refinement

Primary attachment points:

- STEP 06 multiscale representation
- STEP 12 adaptive information-depth routing

## Conditional / side-information coding

Primary attachment points:

- STEP 05 conditional entropy
- STEP 09 shared-versus-unique component cancellation
- STEP 13 branch allocation

## Hierarchical residual coding

Primary attachment points:

- STEP 03 denoising
- STEP 05 innovation
- STEP 06 multiscale decomposition

## Latent rate-distortion

Primary attachment points:

- feature extractor
- branch latent outputs
- core input bottleneck

## Duration/event coding

Primary attachment points:

- STEP 05 token streams
- STEP 07 event detection
- agent-multi event-token Transformer

---

# 7. Governance rule

No completed STEP document needs to be rewritten immediately.

Instead:

- this patch is authoritative for roadmap ordering;
- individual STEP documents may be amended later only when agent audit identifies a substantive scientific or implementation issue;
- experimental results must reference both the relevant STEP document and all applicable patches.

---

# 8. Decision

Before continuing with STEP 08:

1. formally register the compression-derived lane;
2. update the master work plan;
3. keep STEP 01–07 unchanged except for roadmap references;
4. proceed to STEP 08 only after this patch is acknowledged by the orchestrating agents.
