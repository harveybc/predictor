# Addendum — D2-R3 executed under governance, D2-R4 measured, orders incorporated into the work plan

**Date:** 2026-09-14 (after `SATOSHI_GOV_N1_N8_D2_R1_R8_RETURN_PACKET_2026_09_14.md`)
**Trigger:** Musashi's acta `docs/handoffs/MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md`
(N3 executed) and the owner's instruction: incorporate the updated orders into the work
plan; previous experiment closures continue **on the running services, without
interrupting them**.
**Stopping at:** `D2_SUPPORT_READJUDICATED_PORTABILITY_SCOPED_D3_DESIGN_READY_FOR_REVIEW`
(the D2-R8 stop; the review of these outputs is Musashi's, and nothing here grants
eligibility or opens D3).

## 1. What the acta changed, read from the repository

N3 is `PROVEN_PRODUCTION`. I did not restart anything: the services were already serving
from detached runtime worktrees when I began, and I verified what they run rather than
assuming it — data-gov `ff4503a` (N1 + N2 + N4 + operator console), the financial lake at
content **byte-identical** to my tested `f00bc6c15`, the warehouse at predictor `a7a86e9`.
So the N2 and N4 corrections are in production, which the acta could not yet state because
it was written before the 06:35Z restart.

**Declared provenance deficit — resolved, and my recommendation withdrawn (2026-09-14):**
`financial-data/master` carries that content by copy, not by ancestry — `7f77e3ce6 →
13e6b1f47 → f00bc6c15` are not ancestors of `master`. Equality holds by digest, not by
history. I recommended merging `satoshi/c122-c145-20260912` into `master` "without changing
a byte"; that was wrong twice over. The owner's disposition is to document provenance **per
file**, and the published record does so:
`financial-data/docs/ADAPTER_PUBLICATION_PROVENANCE_2026_09_14.md` (`2dbe92286`) maps
`cc0f15e6e` ← `f00bc6c15` for the three adapter files with their SHA-256. I re-executed its
reproduction command: `git diff --exit-code` over those three paths is clean, while the two
full revisions differ in **80 paths** (research scripts, census artefacts, documentation) —
so a whole-branch merge would not have been byte-neutral, and none is needed. No merge is
pending on anyone.

The operator console is incorporated as what it is: a **pending** configuration is neither
an authorization nor a deployment; activation remains the service's configuration load.
That is the route for any future lake (for example downloadable D2 evidence resources),
in a deliberate window, never by editing a detached worktree.

Work plan: `docs/integracion_workplan_2026_09_10/09_INCORPORACION_ORDENES_ACTUALIZADAS_2026_09_14.md`
(binding rules, identity of what serves today, sequence) and section 9 of document 06.

## 2. D2-R3 — the re-adjudication is now governed (production)

Campaign `d2-support-readjudication-r3`
`733736b795e075bffc8c1331a0ce7c910e0ecf721b91123f19990eb1e577ab66`, unit
`readjudication`, classification GOVERNING, terminal lake `olap_cube`, registered against
the live data-gov before any evidence was read. Input mode SYNTHETIC: the evidence spec
(`synthetic_spec_sha256` in the receipt) binds the generator, design `3577c154…`, tape
`230a0a44…`, the reserve manifest, the three conserved tables by digest (re-hashed and
checked against their sealed manifest), the lab code digests at creation and now, and the
re-adjudication code. The receipt is current; the bytes keep their original production
date and provenance, and no governed download of the original production is claimed.

| evidence | result |
|---|---|
| terminal | `COMPLETED`, one row in `gov_terminal` |
| metrics | 25 in `gov_terminal_metric` (every transition plus the universe counts) |
| artifacts | 7 in `gov_terminal_artifact` (successor decisions, impact table and rows, universe check, support summary and table, preview) |
| reconciliation | `missing_units=[] accounting_only=[] lake_only=[]` |
| successor rows | 3,591 loaded additively under `d2r3_733736b795e075bffc8c1331`, throwaway database first, then the cube |
| history | `d2v2_fresh_1fd0e710bc0689b3bf9d7162` keeps its 3,591 rows; only `df_fact_d2_decision`, `df_dim_run` and `df_fact_load_receipt` changed |
| loader | `ActiveState=active`, `NRestarts=0` throughout |

Scientific content (unchanged from the preview, now under governance): **138 of 3,591
decisions change and none gains a pass**. Seven lose one — the five cases Musashi found
and two identity controls on flat noise-free windows — all to `NOT_IDENTIFIABLE` with
`COMPLETE_SEEDS n < DESIGN 30`. 95 `LAB_REJECTED` become `NOT_IDENTIFIABLE` (they rested
on incomplete seeds without measured damage), 10 `NOT_IDENTIFIABLE` become `LAB_REJECTED`
(damage measured on an observed seed now rejects regardless of another seed's
incompleteness), 9 become `UNDERPOWERED`, and 17 SNR decisions become
`SNR_NOT_IDENTIFIABLE`. Candidate passes (arm role `CANDIDATE` only): 51 `LAB_CALIBRATED` + 7 `REGIME_LIMITED`
→ **47 `LAB_CALIBRATED` + 6 `REGIME_LIMITED`**; SNR calibrations 39 → 39 and
`SNR_REGIME_LIMITED` 171 → 167. Corrected on 2026-09-14 after the owner's finding: the
earlier "48 plus 6" was wrong. Recount executed against the cube
(`SELECT run_id, arm_role, decision, count(*) FROM df_fact_d2_decision …`, read-only),
recorded in `d2_support_r1/r3/R3_CANDIDATE_RECOUNT.json`. Evidence:
`docs/audits/evidence/repro_runs/d2_support_r1/r3/`.

## 3. D2-R4 — portability measured, scope restricted, tolerance untouched

The subset was frozen before the replay. `tools/df_d2_r4_replay.py` calls the same entry
point that produced the historical observations (`df_snr.py --d2-unit-facts`), one unit at
a time, one BLAS thread, 2 GiB cap; on each worker role it runs from a **copy** of that
role's own `tools/` directory with the single new file added, so no D2 code file was
modified on any machine. Cost: 403 CPU-seconds in total across the three roles (budget
≤ 6 CPU-h), peak 191 MB, one process per host.

| question | answer |
|---|---|
| byte equality | 1,716 cells; **four** estimators reproduce **exactly** (`mad_first_difference`, `wavelet_mad`, `spectral_floor`, `trailing_median_residual`: max \|Δ\| = 0.0 dB). `ar_residual` meets the observed tolerance but is **not** exact: 1.9184653865522705e-13 dB. Corrected 2026-09-14 (owner's finding); counts per estimator in `R4_PORTABILITY_REPORT.json.per_estimator` |
| numerical tolerance | 83 cells deviate beyond 1e-9 dB, **all `local_level_kalman`**: ≤ 2.44e-05 dB on confirmation (median 2.3e-07), 1.099 dB on one calibration cell of a ~85 dB regime where the likelihood is nearly flat |
| decision stability | 0 identifiability changes; the subset's regimes re-adjudicated with the replayed values leave **288 of 288 decisions unchanged** — now stated with its coverage: 32/32 units, 1,716 cells, **639 substituted facts**, 16/16 regimes, nothing missing, verdict `MEASURED`. The comparator that produced the first version of this line could report `changed: 0` from empty inputs; repaired and re-run on 2026-09-14 (`d2_support_r1/r4_guard/`, PRE 7/7 failing → POST 7/7) |
| convergence | no failure; the only reasons recorded are `signal_variance_nonpositive`, a contract state, identical in both runs |

The pattern behind AT9: COORDINATOR and WORKER_A reproduce each other **bit for bit** in
both directions; WORKER_B differs in both, with identical Python 3.12.13, numpy 2.5.1,
scipy 1.18.0, statsmodels 0.14.6, scipy-openblas and glibc 2.43. **Corrected 2026-09-14:**
one host differs; *why* is not measured. Matching version strings do not isolate the
processor — the same wheel dispatches different kernels by CPU feature detection — and the
library build identity, the runtime dispatch and the optimiser's convergence path were never
recorded. The finding is that this estimator is not reproducible across these hosts, cause
unisolated; AT9's 0.006 dB is one case of it. The observed deviations are not bounds for
future runs. Prospective policy per estimator is in
`d2_support_r1/r4/R4_PORTABILITY_POLICY.md`: the four exact estimators and `ar_residual` carry no
portability restriction beyond recording the reference environment; `local_level_kalman`
is restricted — producing role and environment recorded with every governing result, no
decision may rest on a margin below the deviation observed for its partition class (none
does today), and a deterministic algorithm proven independently is required before
selection, live or a public claim. AT9 remains **open** under its original criterion:
neither 0.006 nor 1.099 dB is adopted as a tolerance. `local_level_kalman` does not hold
the other five estimators back; they can be submitted separately.

## 4. States after this addendum

| block | state |
|---|---|
| GOV-N1, N2, N4 | `IMPLEMENTED` + `PROVEN_DISPOSABLE` + in production (verified, not assumed) |
| GOV-N3 | `PROVEN_PRODUCTION` (Musashi) |
| GOV-N5 feature-eng | `PROVEN_DISPOSABLE`; deployment not part of this order |
| GOV-N5 feature-extractor | pending Musashi's decision on the `stl_preprocessor` API drift |
| D2-R1/R2 | `IMPLEMENTED` (PRE 10/10 failing → POST 26/26) |
| **D2-R3** | **`PROVEN_PRODUCTION`** |
| **D2-R4** | **executed and scoped**; AT9 open |
| D2-R5 | respected: one process per host, one BLAS thread, ≤ 2 GiB, no deliberate OOM, 403 CPU-seconds |
| D2-R6 | rehearsed on a throwaway database, not applied; proposal superseded |
| D2-R7 / N7 | design and plan (documents 07 and 08) |

## 5. Commits of this round (read from the repositories)

| repo | branch | commits |
|---|---|---|
| predictor | `satoshi/c166-c184-20260913` | `ddf621e` merge of Musashi's acta by identity, `3ac95e1` R3 campaign tool + successor rows, `0600196` R3 executed + work-plan incorporation, `5f67906` R4 replay, policy and evidence |
| data-gov | `master` | fast-forwarded to `2afabf6` (operator console and publication by the owner; my N1/N2/N4 are its ancestors) |
| financial-data, feature-eng, preprocessor, feature-extractor | unchanged this round | see the previous packet |

## 6. Open, with owner and minimum action

| pending | owner | minimum action |
|---|---|---|
| ~~`financial-data/master` lineage~~ | — | **closed 2026-09-14**: provenance documented per file in `ADAPTER_PUBLICATION_PROVENANCE_2026_09_14.md`; no branch merge required |
| ~~feature-extractor API drift~~ | — | **decided by the owner 2026-09-14**: port feature-extractor to predictor's `run_preprocessing(self, target_plugin, config)`; no further authorization awaited |
| R6 coverage views | adoption route | pending configuration + deliberate activation window; never a direct edit of the cube |
| `local_level_kalman` | design | deterministic algorithm with independent proof before selection, live or public claim |
| financial resource contracts | data producer/integrator | provider time semantics before any scientific use |
| DOIN / live | Satoshi designs, Musashi reviews | `doin_governed_result.v1` and fixtures; live only through offline replay |

---

## 8. Second addendum, 2026-09-14: the owner's corrections and the two store hosts

**The three corrections.** (1) The successor run leaves **47 `LAB_CALIBRATED` + 6
`REGIME_LIMITED`** in the `CANDIDATE` arm, against 51 + 7 published — not "48 plus 6";
recount read-only against the cube in `d2_support_r1/r3/R3_CANDIDATE_RECOUNT.json`.
(2) **Four** estimators are exactly equal on the three CPUs and a **fifth**, `ar_residual`,
only meets the observed tolerance (1.9184653865522705e-13 dB). (3) The comparator could
report `changed: 0` from inputs that covered nothing; it now accounts for coverage, emits
`MEASURED` | `INCONCLUSIVE` and exits 4 rather than serving an empty comparison as
stability (PRE 7/7 failing → POST 7/7, `d2_support_r1/r4_guard/`). Re-run: `MEASURED`,
32/32 units, 1,716 cells, 639 substituted facts, 16/16 regimes, **288 decisions compared,
0 changed** — the conclusion survives, now with its scope.

**The two repositories, created and in use.** `data-lake` and `data-warehouse` exist with
the providers packaged separately (`financial-data-store`, `predictor-olap-store`). The
design's first deliverable — an installed external provider passing the unchanged
governance contract — is proven three ways: real installs into throwaway virtual
environments with named refusals; route parity against both legacy hosts (8/8 and 11/11);
and the full Flow v3 campaign through both new hosts with the governance configuration
untouched (same availability contract, same dataset digest, exact reconciliation). The
parity harness found a real defect on its first run — a provider's own `HoldoutError` is
not the host's class, so a holdout became a 500 — now classified and regression-tested.
Nothing is deployed.

**feature-extractor** is ported (`5d10f0d`) and proven under governance
(`flow_v3_tools/p6_feature_extractor_throwaway.out`). The port exposed a second drift: the
preprocessor trims its windows to the targets but not the parallel date vectors; the
alignment rule and its refusal are explicit.

**financial-data**: closed by the owner's disposition — provenance per file, no merge. My
recommendation is withdrawn (§2 above).

Services `:5055/:5056/:5057` untouched, loader `active` with `NRestarts=0`, no throwaway
database left, `gov_terminal` = 3 rows in the production cube.

Stop: `CORRECTIONS_APPLIED_STORE_HOSTS_PROVEN_FEATURE_EXTRACTOR_PORTED_READY_FOR_REVIEW`.
