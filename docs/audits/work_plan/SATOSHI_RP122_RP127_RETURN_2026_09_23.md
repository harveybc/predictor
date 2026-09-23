# Satoshi — RP122–RP127 return: the declared catalog checked, the accepted relationship bound, the bypass removed

Orders: [MUSASHI_SOTA_RP122_RP127_2026_09_23](../../handoffs/MUSASHI_SOTA_RP122_RP127_2026_09_23.md) (37df73b8), on the
[RP114–RP121 review](MUSASHI_RP114_RP121_REVIEW_2026_09_23.md) of 316c67af. Commits after 37df73b8: eccd7828 … 0d35061d.

**Placement and cost.** Every new GPU execution ran on WORKER_B's EXTERNAL RTX 5090 (UUID a9f35631…), admitted before each
dispatch and asserted inside the child; WORKER_A's GPU, WORKER_B's internal RTX 5070 Ti and the travelling coordinator stayed
held with no fallback. Host CPU work ran single-threaded inside governed slices with the host's own temperature watched beside
the GPU's. This round measured 2 h 25 min of admitted worker
wall time (19 GPU inferences and 24 bounded CPU acceptance passes), inside the standing 14,400 CPU-second round ceiling; the
5090 stayed at 35–41 °C and the host at 41–68 °C, and the reproduction root ended at 2.4 GB with 211 GB free.

## Owner-facing table — official normalized MSE / MAE first

Population: 12 cells (L = 96 × T ∈ {96, 192, 336, 720} × seeds {2021, 2022, 2023}), design 9b49010d…, official processed ECL
(TSL, electricity.csv sha256 7e45845d…), TimeFilter @ dffde87e, the author's recipe and epochs, metrics in the normalized space
through the author's own float32 reduction (route `df_sota_author_metric_exact.v2`). Frozen operational margin:
|mean − published| ≤ 2·σ_paper + 0.0005 = 0.0105 MSE / 0.0125 MAE — a predeclared operational band, **not** statistical
equivalence and not exact equality to a rounded paper table. Scores are unchanged from the previous return; what changed is what
has been independently checked about them. Closure report sha256 211e6231…
([REPORT](../evidence/d3_k5_20260917/RP127/REPORT.gamma.json), [table](../evidence/d3_k5_20260917/RP127/SOTA_TABLE.gamma.md)),
and this narrative's numbers are generated from that same file.

| T | published MSE / MAE | official author-float32 per seed 2021 / 2022 / 2023 | mean (SD across seeds, n) | difference | matched persistence / seasonal-24 (z-MAE, same windows) | measurement | custody | original scorer | catalog acceptance (14 declared families) | replay | agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.133 / 0.230 | 0.13332 / 0.14008 / 0.13309 ; 0.23055 / 0.23778 / 0.23032 | **not pooled** (measured mean 0.13550 / 0.23288, shown apart) | — | 0.9455 / 0.3258 (5165 windows) | **MEASURED_REPLAY_UNVERIFIED** | ACCEPTED_ARTIFACT_CHAIN | author float32, exact route | **all 14 families agree with the independent reference, none unchecked** (159,164,640 elements per cell) | FAIL cross-device on the 5090 (max \|Δ\| 1.9e-4 / 2.2e-4 / 2.7e-4 > 1e-4), replayed author metrics bit-identical | UNVERIFIED: the frozen rule is not widened |
| 192 | 0.154 / 0.248 | 0.16086 / 0.15863 / 0.15345 ; 0.25599 / 0.25243 / 0.24807 | **0.157645 (0.00380) / 0.252163 (0.00396), n = 3** | +0.00365 / +0.00416 | 0.9507 / 0.3237 (5069 windows) | VERIFIED | accepted; predictions deleted 22-sep | author float32 of the accepted original closure (seed 2021) and of bit-identical regenerations (seeds 2022, 2023) | 14/14 families for seeds 2022 and 2023; **seed 2021 REFUSED** — trained on the coordinator's 4070, it does not regenerate identically on the 5090 | historical: same-device PASS before deletion (2022, 2023), cross-device PASS (2021) | **OPERATIONAL_AGREEMENT** |
| 336 | 0.162 / 0.261 | 0.16259 / 0.16817 / 0.16267 ; 0.26067 / 0.26741 / 0.26087 | **0.164480 (0.00320) / 0.262985 (0.00384), n = 3** | +0.00248 / +0.00198 | 0.9613 / 0.3427 (4925 windows) | VERIFIED | accepted; predictions deleted 22-sep | author float32 from bit-identical regenerations | **14/14 families on all three seeds** | historical: same-device PASS before deletion | **OPERATIONAL_AGREEMENT** |
| 720 | 0.184 / 0.284 | 0.18714 / 0.19817 / 0.18537 ; 0.28752 / 0.29865 / 0.28568 | **0.190226 (0.00694) / 0.290617 (0.00702), n = 3** | +0.00623 / +0.00662 | 0.9754 / 0.3733 (4541 windows) | VERIFIED | accepted; predictions deleted 23-sep | author float32 from bit-identical regenerations | **14/14 families on all three seeds** | replays bit-identical before deletion | **OPERATIONAL_AGREEMENT** |
| avg | 0.158 / 0.256 | — | **NOT_COMPUTED** over a denominator of four; T = 96 is the missing horizon for every seed and is named, not dropped | — | — | — | — | — | — | — | — |

Seed dispersion is the sample SD across the three seeds of each horizon, with its host scope stated: T = 192 mixes the
coordinator's RTX 4070 Laptop (seed 2021) with WORKER_B's RTX 5090, so its 0.0038 / 0.0040 is not pure seed variability;
T = 336 and T = 720 are single-host. Paired seed contrasts from the persisted per-window series are DONE for T = 192, 336 and
720 and NOT_APPLICABLE for T = 96. The independently checked float64 reductions remain separately named in every catalog and
never appear under the author's field.

## What was done, per block

* **RP122 — the three counterexamples frozen, then repaired.** Musashi's probe is preserved unchanged
  ([musashi_probe_unchanged.py](../evidence/d3_k5_20260917/RP122/musashi_probe_unchanged.py)) and run at the reviewed base as the
  PRE ([PROBE_RESULTS_PRE.json](../evidence/d3_k5_20260917/RP122/PROBE_RESULTS_PRE.json)): the catalog with zeroed
  autocorrelations, zeroed quantiles and a nulled correlation was accepted; a local relabel turned a diagnostic attachment into an
  accepted closure; `require_acceptance=False` deleted the array. On the repaired head his probe stops at its own third scenario
  because the bypass parameter no longer exists, so the POST
  ([reproduce_post.py](../evidence/d3_k5_20260917/RP122/reproduce_post.py) →
  [REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP122/REVIEW_REPRODUCED_POST.json)) records that refusal explicitly
  alongside the other two: the catalog is refused naming quantiles (2.41), correlation (undefined against −0.1979) and
  autocorrelation (0.943); the relabel is refused before and after, with the warehouse payload byte-for-byte unchanged; the
  deletion is refused by default and the bypass raises a TypeError at the interface with the array still on disk; and the positive
  control passes with fourteen families complete and none unchecked. Each scenario is also a frozen test, one per estimator
  family rather than one combined example.
* **RP123 — the accepted relationship, not merely accepted bytes.** `accepted_artifact` now reads the evidence kind, the subject,
  the artifact's role and the design from the ACCEPTED TERMINAL itself — the acceptance units the receipts name, queried in the
  warehouse — and the local registry is only a hint about which unit to ask about first. A hint that disagrees with the terminal
  is reported and ignored. A digest accepted as a diagnostic attachment, for another subject, or in another role does not certify
  a closure, a catalog or a regeneration, and a regeneration record and its acceptance must come from the same accepted terminal.
* **RP124 — a declared catalog and a reference that implements it.** `CATALOG_ESTIMATORS` states, before measurement, each
  family's input population, axis, orientation, reduction, parameters, undefined condition and tolerance: fourteen families from
  the errors and the matched baselines to the histogram, the residual moments, the entropy, the histogram-CDF quantiles, the
  correlation and R², the joint mutual information, the autocorrelations, the per-step, per-channel, per-window and time-block
  series. The independent implementation follows those declarations on the SAME population: the autocorrelation is now taken over
  window ORIGINS at each fixed forecast step (not concatenated steps of sampled windows), the quantiles are read from the same
  declared histogram approximation rather than silently replaced by exact order statistics, and the baselines are recomputed from
  the author's own loader inputs. The comparison is field by field under each family's tolerance; a value present on one side and
  absent or null on the other is a DISAGREEMENT, never a zero difference; booleans and non-finite values are typed failures; and
  coverage is reported per family, so no claim of full independent acceptance survives an unchecked field.
* **RP125 — the destructive bypass is gone.** `require_acceptance` is removed from `delete_predictions` and
  `deletion_preflight`. Every callable path that can unlink a prediction array — including resumption — requires the accepted
  closure report, the accepted catalog acceptance of the catalog on disk, and the verified durable backup. Dry runs remain
  non-destructive and report the same failed prerequisites. The tests no longer disable a production precondition: they publish a
  faithful accepted chain through the governed fixture.
* **RP126 — the corrected checks applied to the twelve model cells.** Each cell's declared families were compared with the
  independent reference on real bytes: the three retained T = 96 arrays in place, and every deleted cell by regenerating it from
  its retained checkpoint on the admitted 5090, one bounded cell at a time, with its temporaries removed before the next.
  Result, per cell: the three T = 96 catalogs pass **all fourteen declared families with
  nothing unchecked and no disagreement** (159,164,640 elements each); their only refusal is the missing accepted replay. Eight
  of the nine deleted cells regenerate BIT-IDENTICALLY on the admitted device and pass all fourteen families with no
  disagreement — the largest difference anywhere in those catalogs is 3.6e-12, in the raw kurtosis — after which their
  temporaries are removed with per-path receipts. The ninth, **T = 192 seed 2021, was trained on the coordinator's RTX 4070
  Laptop and does NOT regenerate identically on the RTX 5090**: the same cross-device numerical difference that fails the T = 96
  replay. Its acceptance refuses with nine family disagreements (1e-9 to 1e-7, plus the near-zero-denominator ratio estimator and
  23 of 312,412,608 histogram counts), and I do not present those bytes as a certification of its catalog. Eleven of twelve cells
  therefore carry a full independent estimator acceptance; the twelfth needs an inference run on the device that produced it.
* **RP127 — the scoped closure, the errata and the ledger.** The successor closure re-verified all twelve cells against the live
  accepted chain with the replays restricted to the admitted device, and the owner table is generated from that verified source.
  The previous return keeps its text and carries a **dated erratum** for its three over-broad claims — "every estimator family
  agrees", "prerequisites that cannot be skipped" and "acceptance, not resolution" — each naming exactly what it did and did not
  cover; no history was deleted and no measured score changed. The scores are unchanged: T = 192 0.157645 / 0.252163, T = 336 0.164480 / 0.262985
  and T = 720 0.190226 / 0.290617 in OPERATIONAL_AGREEMENT, T = 96 measured and replay-unverified, the four-horizon average
  NOT_COMPUTED over a denominator of four. The accepted-evidence registry now holds 56 entries (4 closure reports, 18 catalog
  acceptances, 34 regeneration records). The ledger stands at 19 attempts, 5.37 h of measured training wall time, and 208–211 GiB
  free disk after the temporaries were removed.


## Exact remaining deficits

1. **One cell cannot be independently certified on the admitted device.** T = 192 seed 2021 was trained on the coordinator's
   RTX 4070 Laptop. Regenerating it by inference on the admitted RTX 5090 returns REGENERATED_NOT_IDENTICAL — the same
   cross-device numerical difference that fails the T = 96 replay — so those bytes cannot certify its retained catalog, and the
   acceptance refuses rather than pretending otherwise (nine families differ, at 1e-9 to 1e-7 except the near-zero-denominator
   ratio estimator and 23 of 312,412,608 histogram counts). Certifying that cell's catalog needs an inference run on the device
   that produced it, which is the coordinator's GPU, held. Its scores stand as measured and accepted; what is missing is the
   independent recomputation of its catalog families.
2. **T = 96 is measured, not verified.** Custody accepted, the author's own float32 scores, and now all fourteen declared
   estimator families agreeing with the independent reference over 159,164,640 elements — its only missing property is an
   accepted replay under the frozen rule. On the admitted 5090 the pointwise rule fails cross-device while the replayed metrics
   are bit-identical; the rule is not widened, and the original-device replay waits for an explicit cooling confirmation for
   WORKER_A. The estimate of about 66 s is a replay step, not a guarantee of acceptance.
3. **The four-horizon average stays NOT_COMPUTED** over a denominator of four, with T = 96 named as missing per seed.
4. **Legacy device attribution stays INFERRED** for the nine GPU-trained cells; no replay or regeneration measures the device
   that trained them, and nothing was retrained to manufacture a UUID.
5. **Protocol B remains at resource-planning scope**: the pilot was not repeated and no cell was trained or scored.


## Suites, with scope and accounting

| suite | result | wall | peak RSS |
|---|---|---|---|
| `tests/test_df_sota_repro.py` (own process, WORKER_A) | 98 passed, 1 skipped | 76.5 s | 1.37 GB |
| closure table + benchmark contract + block + financial acceptance | 117 passed, 2 deselected | 165.4 s | 2.96 GB |
| `olap/store/tests` on SQLite **and** DuckDB | 106 passed, 1 skipped | 3.6 s | 0.22 GB |
| lake adopters | 4 skipped | 0.2 s | — |

Interpreter 3.12.13 with numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cu130, scikit-learn 1.9.0 on WORKER_A; the warehouse store
suite in a throwaway venv with duckdb 1.5.5 and SQLAlchemy 2.0.54. Skips named rather than hidden: one sota test needs the
benchmark store receipt only the coordinator holds; two financial tests (FL08) need the coordinator's governance stack; the
four lake-adopter tests need the coordinator's deployed data-gov configuration. No test ran on the coordinator. The mutation
tests exercise one estimator family at a time — autocorrelation, quantiles, correlation, entropy, per-step errors, baselines and
time blocks — plus the relabelled acceptance, the wrong subject, the wrong role and the absent bypass.


## Review request

One consolidated audit request: (a) the declared estimator inventory and whether the independent implementation matches each
declaration — in particular the autocorrelation over window origins per fixed step, the histogram-CDF quantiles and the
loader-derived baselines; (b) the missing-value semantics and the per-family coverage report, including the refusal to claim
full independent acceptance when a field is unchecked; (c) the acceptance relationship read from the terminal's own kind,
subject, role and design, and whether any path still lets a local hint decide; (d) the removal of the deletion bypass and the
proof that no public invocation unlinks without the accepted chain; (e) the per-cell coverage actually achieved on the twelve
model cells, including the cell whose regeneration on the admitted device is NOT bit-identical and what that limits;
(f) the dated errata on the previous return.


## Errata, 2026-09-23 (added after Musashi's RP127 review; nothing above is deleted)

1. **"eleven of twelve cells carry a full independent estimator acceptance" was true of the checks that ran, but the acceptance
   CLASS was not enforced where it mattered.** An acceptance produced without the numerical reference — no data path, the
   reference disabled, or arrays unavailable — still emitted `pass`, and the deletion read that flag. RP128 gives every
   acceptance an explicit class and certificate derived from its own content, and the deletion now reads the certificate
   (class, catalog digest, population, inventory) instead of the flag. The eleven checks themselves stand and were revalidated
   read-only, not repeated.
2. **The scientific design was not required where acceptance is consumed.** `accepted_artifact` could check it when a caller
   passed it, but the deletion and regeneration consumers did not, and a terminal that declared no design passed even when the
   expected design was given. RP129 makes the design mandatory at those consumers, read from the terminal's own field, with a
   contradicting campaign-config digest refused.
3. **A boolean compared equal to a number.** `False` matched the count zero in a declared numeric field. RP130 rejects booleans
   in numeric fields at any depth.

No measured score changes, and no production array was deleted through any of these paths.
