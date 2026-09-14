# D2-R4 — numerical portability of the SNR estimators: measurement and prospective policy

Date: 2026-09-14. Subset, seeds and criterion frozen **before** the replay
(`R4_DIAGNOSTIC_SUBSET.json`, built by `tools/df_d2_r4_margins.py` from the published
decisions and the sealed rules). The historical tolerance is not widened anywhere in
this document: AT9's criterion remains exactness, and the deviation observed here is
reported against it, never adopted as a new limit.

## 1. What was executed

`tools/df_d2_r4_replay.py --replay` calls the **same** entry point that produced the
historical observations (`tools/df_snr.py --d2-unit-facts`) in a child process, one unit
at a time, one linear-algebra thread, 2 GiB hard limit. No file of the D2 code was
modified: on the worker roles the replay runs from a *copy* of that role's own `tools/`
directory with the single new file added, and the reserve is read-only.

| role | units | CPU seconds | peak RSS | `df_snr.py` sha256 |
|---|---|---|---|---|
| COORDINATOR | 32 | 133.1 | 189 MB | `a5ad155f…`, identical on the three roles |
| WORKER_A | 32 | 192.4 | 190 MB | identical |
| WORKER_B | 32 | 78.1 | 191 MB | identical |

Aggregate cost far below the R5 budget (≤ 6 CPU-h, ≤ 4 h wall): the three replays
together are ~7 CPU-minutes. Environments are identical except the processor:
Python 3.12.13, numpy 2.5.1, scipy 1.18.0, statsmodels 0.14.6, scipy-openblas,
glibc 2.43, float64, one BLAS thread; CPUs are AMD Ryzen 7 (COORDINATOR),
Intel Core i9 (WORKER_A) and AMD Ryzen 9 (WORKER_B).

## 2. Three questions kept apart

| question | answer |
|---|---|
| **byte equality** | 1,716 (unit × variable × estimator × partition × role) cells; **four** estimators reproduce the historical value exactly on the three CPUs (`mad_first_difference`, `wavelet_mad`, `spectral_floor`, `trailing_median_residual`: max \|Δ\| = 0.0 dB). A **fifth**, `ar_residual`, stays inside the observed tolerance but is **not** exact: max \|Δ\| = 1.9184653865522705e-13 dB. Corrected on 2026-09-14 (owner's finding); per-estimator counts in `R4_PORTABILITY_REPORT.json.per_estimator` |
| **numerical tolerance** | 83 cells deviate beyond 1e-9 dB, **all of them `local_level_kalman`**: confirmation partition max 2.44e-05 dB (median 2.3e-07), calibration partition max 1.099 dB |
| **decision stability** | the subset's regimes were re-adjudicated with the replayed values substituted, and the comparison now states its own coverage (32/32 units, 1,716 cells, 639 substituted facts, 16/16 regimes, 0 units or regimes missing; verdict `MEASURED`): **288 decisions compared, 0 changed**; 0 identifiability changes; no convergence failure (the only reasons recorded are `signal_variance_nonpositive`, a contract state, identical in both runs) |

## 3. The pattern behind AT9

`local_level_kalman`, producing role → replaying role (cells / exact / outside 1e-9 / max |Δ| dB):

| producing → replaying | COORDINATOR | WORKER_A | WORKER_B |
|---|---|---|---|
| COORDINATOR (2 units) | 4/4/0/0 | 4/4/0/0 | 4/0/0/0 |
| WORKER_A (16 units) | 37/37/0/0 | 37/37/0/0 | 37/0/25/2.44e-05 |
| WORKER_B (14 units) | 37/0/29/**1.099** | 37/0/29/**1.099** | 37/37/0/0 |

COORDINATOR and WORKER_A reproduce each other **bit for bit** in both directions;
WORKER_B is the one that differs, in both directions, with the same code digests and the
same package version strings. AT9 (up to 0.006 dB between two CPUs on one unit) is a case
of the same phenomenon: the local-level Kalman MLE follows a different optimisation path
on that host.

**Attribution is a hypothesis, not a result (corrected 2026-09-14).** What is measured is
that one of three hosts differs; what is *not* measured is why. Matching version strings
do not isolate the processor: the same wheel dispatches different kernels by CPU feature
detection, and the build identity of the linear-algebra library, its runtime dispatch and
the optimiser's own convergence path were never recorded. Processor-only attribution needs
an intervention (pin the dispatch, or run the same host under different kernels) or
controls that exclude the alternatives. Until that diagnostic runs, the finding is: *this
estimator is not reproducible across these hosts*, cause unisolated. The proposal is
`docs/integracion_workplan_2026_09_10/10_DIAGNOSTICO_KALMAN_…`.

The largest deviation, 1.099 dB, is in a **calibration** cell of a high-SNR regime
(≈ 85 dB) where the likelihood is nearly flat; governing confirmation cells stay at
2.44e-05 dB. **These are observations, not bounds.** Nothing here licenses treating
1.099 dB or 2.44e-05 dB as a maximum for future runs, on these hosts or any other: they are
the largest values seen in 32 units of one frozen subset.

## 4. Prospective policy per estimator

**Reference environment** (for any governing use): Python 3.12.13, numpy 2.5.1,
scipy 1.18.0, statsmodels 0.14.6, scipy-openblas, one BLAS thread, float64; the role and
the environment are recorded with the result.

| estimator | portability | policy |
|---|---|---|
| `mad_first_difference` | exact on the three CPUs | no restriction from portability; environment recorded |
| `wavelet_mad` | exact | idem |
| `spectral_floor` | exact | idem |
| `trailing_median_residual` | exact | idem |
| `ar_residual` | **not exact**; within the observed tolerance (max \|Δ\| = 1.918e-13 dB) | no restriction from portability; environment recorded, and the non-zero deviation is stated wherever exactness is claimed |
| `local_level_kalman` | **not reproducible across these hosts** (cause unisolated): ≤ 2.44e-05 dB on confirmation, up to 1.099 dB on a near-flat calibration cell | restricted scope: (a) every governing result records the producing role and environment; (b) no decision may rest on a Kalman margin smaller than the deviation observed for that partition class — an observed value used as a caution, never as a proven bound; today none does — 0/288 decisions move; (c) before any use in selection, live or a public claim, replace it with a deterministic algorithm (fixed iteration budget and declared convergence tolerance, or a closed-form estimator) proven independently; (d) **AT9 stays open** under its original criterion |

The four exact estimators and `ar_residual` (exact to within 1.918e-13 dB) can be submitted separately; `local_level_kalman` does not hold
the rest of the project blocked. Nothing here grants eligibility: the D2 gate keeps
refusing every subject without an external review record.

## 5. Limits of this measurement

Thirty-two units of sixteen regimes, chosen by the frozen rule — not the population.
Three roles of this fleet, three CPU models, one BLAS build: portability beyond them is
not established. The replay reproduces the estimator, not the whole unit worker, so it
speaks about the SNR estimates and their intervals, not about denoising outputs. It is a
post-result sensitivity study, not a fresh confirmation.
