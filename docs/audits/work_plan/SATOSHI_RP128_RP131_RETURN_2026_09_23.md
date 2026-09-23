# Satoshi — RP128–RP131 return: the checks now bind the destructive decision

Orders: [MUSASHI_SOTA_RP128_RP131_2026_09_23](../../handoffs/MUSASHI_SOTA_RP128_RP131_2026_09_23.md) (04022f89), on the
[RP122–RP127 review](MUSASHI_RP122_RP127_REVIEW_2026_09_23.md) of a73e55b2. Commits after 04022f89: 93a9483d … 0ab2f7ed.

**Placement and cost.** No GPU was used this round: the repairs and their fixtures are bounded CPU work, run on WORKER_A and
WORKER_B inside governed single-threaded slices. WORKER_A's GPU, WORKER_B's internal RTX 5070 Ti and the travelling coordinator
stay held; the external RTX 5090 remains the only eligible device and was not needed. Measured cost: about 22 minutes of CPU in
admitted scopes (three suite runs of 96 s, 164 s and 3.6 s, three catalog re-acceptances of ~4 min each, two revalidation passes
of a few seconds, two probe runs), against the standing 14,400 CPU-second round ceiling. Host temperatures stayed at 20–56 °C on
WORKER_B and 20–48 °C on WORKER_A; one probe run refused to start on WORKER_A because its own thermal guard saw a hot zone after
back-to-back suites, so that run was moved to the cooler worker rather than forced.

## What was wrong, and what each repair does

* **RP128 — an acceptance that never ran the numerical reference could still authorise a deletion.** `accept_catalog` skips the
  reference when the data path is absent, when it is disabled, or when the arrays are gone, and the deletion read only its
  `pass`. Now every acceptance carries an explicit **class** and a **certificate derived from its own recorded content**:
  FULL_INDEPENDENT_NUMERIC only when every declared family was compared, nothing was left unchecked and nothing disagreed;
  otherwise DOMAIN_AND_INTERNAL_ONLY, kept as a diagnostic with a restricted-scope note rather than discarded. The destructive
  consumer validates that certificate — its class, the catalog digest it covers, the population against the record's own shape,
  and the estimator inventory version and digest it was issued under — and no caller-declared boolean can stand in for it. The
  same condition applies on resumption. A second separation follows from the same review: the catalog acceptance certifies **the
  catalog** (the closure recomputed and read back that very catalog), while whether the **cell** is verified — its custody, its
  replay — is a separate fact the deletion gate enforces on its own.
* **RP129 — the destructive callers did not require the accepted design.** The design is now mandatory wherever acceptance is
  consumed, and it is read from the terminal's own `design_sha256` tag, the authoritative scientific-design field. A terminal
  that declares no design refuses even when the caller passes the expected one, and a campaign-config digest that disagrees with
  the declared design is treated as a contradiction rather than a match.
* **RP130 — a boolean compared equal to a number.** `False` matched the count zero in a declared numeric field. Booleans are now
  only comparable to booleans, at any depth of a list or mapping, and the real catalog acceptance refuses the producer's boolean
  count with a typed message.

## PRE and POST, on Musashi's unchanged probe

His probe is preserved ([musashi_probe_unchanged.py](../evidence/d3_k5_20260917/RP128/musashi_probe_unchanged.py)) and run
before and after the repairs:

| scenario | PRE (a73e55b2) | POST (repaired head) |
|---|---|---|
| acceptance without the numerical reference, then the normal deletion API | COMPLETE, array removed | **REFUSED, array intact** |
| acceptance declaring a foreign design, then deletion | COMPLETE, array removed | **REFUSED, array intact** |
| terminal declaring no design, with the expected design passed explicitly | accepted | **refused** |
| producer emitting `False` where a count belongs | accepted, difference 0.0 | **refused: a boolean cannot be compared with int** |
| valid catalog control | accepted | accepted, fully independent |

Each is also a frozen test on the real consumers, with the additional cases the orders name: a stale `pass` left by an
incomplete comparison, one unchecked field, a foreign inventory, a population that is not the record's, contradictory design
declarations, a right digest under another subject, dry runs, interrupted and resumed deletions, and the boolean cases at depth.

## The twelve cells, revalidated read-only

The eleven successful numerical reductions were **not repeated**. They were re-read under the repaired gates, together with
the accepted chain, and judged again from their own recorded content
([REVALIDATION.json](../evidence/d3_k5_20260917/RP131/REVALIDATION.json), published as accepted evidence):

| cell group | numerical acceptance | where it comes from | predictions on disk | deletion eligible today |
|---|---|---|---|---|
| T = 96, three seeds | FULL_INDEPENDENT_NUMERIC | their own catalog certificates (14/14 families, nothing unchecked) | yes | **no** — the cell is not verified by the current closure: its replay is not accepted |
| T = 192 seeds 2022, 2023; T = 336 ×3; T = 720 ×3 | FULL_INDEPENDENT_NUMERIC | their regeneration certificates, on bytes bit-identical to the deleted originals | no | no (nothing to delete) |
| T = 192 seed 2021 | DOMAIN_AND_INTERNAL_ONLY | its regeneration on the admitted device is not the original, so it certifies nothing numeric | no | no |

The three T = 96 catalog certificates were re-issued and published, not recomputed from scratch for a gate: the contract changed
(the catalog's certification is now separate from the cell's verification), so their acceptance had to be re-expressed under it.
Their numbers are identical. No array was deleted this round, and the only predictions on disk — the three T = 96 arrays —
remain retained, with the reason recorded per cell.

## Owner-facing table — official normalized MSE / MAE first

Unchanged from the previous return; repeated here from the same accepted source
([REPORT](../evidence/d3_k5_20260917/RP127/REPORT.gamma.json), sha256 211e6231…), with the comparability scope stated. Frozen
operational margin |mean − published| ≤ 0.0105 MSE / 0.0125 MAE: a predeclared operational band, **not** statistical equivalence
and not exact equality to a rounded paper table. No value here is a new measurement.

| T | published MSE / MAE | reproduced mean (SD across seeds, n) | difference | matched persistence / seasonal-24 (z-MAE, same windows) | score status | replay | catalog | agreement |
|---|---|---|---|---|---|---|---|---|
| 96 | 0.133 / 0.230 | not pooled (measured 0.13550 / 0.23288) | — | 0.9455 / 0.3258 (5165 windows) | MEASURED | not accepted (cross-device failure preserved) | 14/14 families independently checked | UNVERIFIED |
| 192 | 0.154 / 0.248 | 0.157645 (0.00380) / 0.252163 (0.00396), n = 3 | +0.00365 / +0.00416 | 0.9507 / 0.3237 (5069 windows) | accepted historically | same-device PASS before deletion (2022, 2023), cross-device PASS (2021) | 14/14 for seeds 2022, 2023; seed 2021 incomplete | **OPERATIONAL_AGREEMENT** |
| 336 | 0.162 / 0.261 | 0.164480 (0.00320) / 0.262985 (0.00384), n = 3 | +0.00248 / +0.00198 | 0.9613 / 0.3427 (4925 windows) | accepted historically | same-device PASS before deletion | 14/14 on all three seeds | **OPERATIONAL_AGREEMENT** |
| 720 | 0.184 / 0.284 | 0.190226 (0.00694) / 0.290617 (0.00702), n = 3 | +0.00623 / +0.00662 | 0.9754 / 0.3733 (4541 windows) | accepted historically | replays bit-identical before deletion | 14/14 on all three seeds | **OPERATIONAL_AGREEMENT** |
| avg | 0.158 / 0.256 | **NOT_COMPUTED** over a denominator of four | — | — | — | — | — | T = 96 is the missing horizon for every seed |

T = 192's dispersion mixes two hosts (the coordinator's RTX 4070 Laptop for seed 2021, WORKER_B's RTX 5090 for the others) and is
not pure seed variability; T = 336 and T = 720 are single-host.

## Exact remaining deficits

1. **T = 96 remains measured with its replay unaccepted.** Its catalogs are fully checked and its arrays are retained; the
   original-device replay waits for an explicit cooling confirmation for WORKER_A. The estimate of about 66 s is a replay step,
   not a guarantee of acceptance, and the cross-device failure is preserved rather than tolerated.
2. **T = 192 seed 2021's catalog is not independently certified.** It was trained on the coordinator's RTX 4070 Laptop and its
   regeneration on the admitted 5090 is not the original; per the orders I did NOT repeat that failing inference. Certifying it
   needs an inference run on the device that produced it, which is held.
3. **The four-horizon average stays NOT_COMPUTED** over a denominator of four, with T = 96 named per seed.
4. **Legacy device attribution stays INFERRED** for the nine GPU-trained cells; nothing was retrained to manufacture a UUID.
5. **Certificates issued before this round** carry no inventory digest; they are read by content and their coverage is checked
   against the current declared families. Where that content was insufficient the evidence is marked DOMAIN_AND_INTERNAL_ONLY
   rather than upgraded.
6. **Protocol B stays at measured resource-planning scope**; its pilot was not repeated and no cell was trained or scored.

## Suites, with scope and accounting

| suite | result | wall | peak RSS |
|---|---|---|---|
| `tests/test_df_sota_repro.py` (own process, WORKER_A) | 104 passed, 1 skipped | 96.4 s | 1.36 GB |
| closure table + benchmark contract + block + financial acceptance | 117 passed, 2 deselected | 163.8 s | 2.95 GB |
| `olap/store/tests` on SQLite **and** DuckDB | 106 passed, 1 skipped | 3.6 s | 0.22 GB |
| lake adopters | 4 skipped | — | — |

Interpreter 3.12.13 with numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cu130, scikit-learn 1.9.0; the store suite in a throwaway venv
with duckdb 1.5.5 and SQLAlchemy 2.0.54. Skips named rather than hidden: one sota test needs the benchmark store receipt only the
coordinator holds; two financial tests need the coordinator's governance stack; the four lake-adopter tests need its deployed
data-gov configuration. No test ran on the coordinator.

## Review request

One consolidated audit request: (a) the acceptance certificate and whether the destructive consumer now establishes what it
must — class, catalog, population, inventory, complete comparison — without any caller-declared boolean; (b) the separation
between certifying a catalog and verifying a cell, and whether the deletion gate still enforces both; (c) the design identity
read from the terminal's own field, including the contradiction case; (d) the boolean typing at depth; (e) the read-only
revalidation as a way to reuse valid evidence instead of recomputing it, and whether the re-issued T = 96 certificates are
scoped honestly; (f) the dated errata on the previous return.
