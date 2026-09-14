# Return packet — GOV-N1..N8 (Flow v3 deployment and adoption) and D2-R1..R8 (support, portability, D3 design)

**Date:** 2026-09-14
**Orders:** `docs/handoffs/MUSASHI_TO_SATOSHI_FLOW_V3_DEPLOYMENT_ADOPTION_AND_D2_ORDER_2026_09_13.md` (GOV-N1..N8),
`docs/handoffs/MUSASHI_TO_SATOSHI_D2_SUPPORT_PORTABILITY_NEXT_ORDER_2026_09_13.md` (D2-R1..R8)
**Review answered:** `MUSASHI_REVIEW_C166_C184_AND_DATA_GOV_2026_09_13.md` (`REVISE_D2_ADJUDICATION_BEFORE_CONSUMPTION`)
**State file (kept after every block):** `docs/audits/work_plan/FLOW_V3_GOV_N1_N8_STATE_2026_09_13.md`
**Stopping at:** `D2_SUPPORT_REPAIRED_PREVIEW_NON_GOVERNING_R3_PENDING_PRODUCTION_RUN` — the D2-R8 stop
`D2_SUPPORT_READJUDICATED_PORTABILITY_SCOPED_D3_DESIGN_READY_FOR_REVIEW` is **not** reached: R3 and R4's replay need the reconciled production micro-run (N3), which my execution environment cannot perform.

CPU only. No GPU, scientific training, D3 execution, selection, DOIN publication or live. No historical result was regenerated or rewritten; the 3,972 units and the 3,591 published decisions stand. Every state below is one of `IMPLEMENTED`, `PROVEN_DISPOSABLE`, `DEPLOYED`, `PROVEN_PRODUCTION`, or pending with the missing object, the owner and the minimum action.

---

## 0. States, up front

| block | state | missing object / owner / minimum action |
|---|---|---|
| N1 lake/warehouse integration | `IMPLEMENTED`, `PROVEN_DISPOSABLE` | — |
| N2 executable contract scope | `IMPLEMENTED`, `PROVEN_DISPOSABLE` | — |
| N3 bounded deployment + production micro-run | **pending** | restart of `:5057/:5056/:5055` with the integrated code; Musashi in a permitted environment or the operator; then one CPU micro-run (§3). Recorded once; the previous packet's Appendix A is a description, its pids an old observation |
| N4 permanent-rejection disposition | `IMPLEMENTED`, `PROVEN_DISPOSABLE` | — |
| N5 feature-eng | `IMPLEMENTED`, `PROVEN_DISPOSABLE` | — |
| N5 feature-extractor | `IMPLEMENTED` (wrapper, environment) — **pending** | API drift between feature-extractor and predictor's `stl_preprocessor` (§5); Musashi decides pin vs port; not modernised here |
| N6 / D2-R1 support contract, D2-R2 repair | `IMPLEMENTED` (PRE 10/10 failing → POST 26/26) with a **non-governing preview** of the re-adjudication | — |
| D2-R3 governed re-adjudication | **pending** | reconciled production micro-run (N3); then register the review campaign |
| D2-R4 AT9 portability | `IMPLEMENTED` (margins, inventory, frozen subset), replay **pending** | same dependency as R3 |
| D2-R5 resources | respected (one process per host, one BLAS thread, ≤2 GiB, no deliberate OOM) | — |
| D2-R6 coverage views + proposal update | `IMPLEMENTED`, `PROVEN_DISPOSABLE` (SQL rehearsed, not applied) | apply through the adoption route after review |
| D2-R7 D3 design | `IMPLEMENTED` (design and acceptance tests, not executed) | — |
| N7 plans agent-multi / DOIN / live | `IMPLEMENTED` (plan) | — |
| N8 packet | this document | — |

Production: `:5055/:5056/:5057` still run the pre-v3 code (probed: `/api/v2/*` → 404); no `gov_*` table exists in the real cube; the OLAP loader is `active`, `NRestarts=0`; no throwaway database remains.

## 1. My own faults, first

1. **N3 not executed.** Same policy limit as the previous packet; it is recorded once (state file, N3) and the block passes to Musashi or the operator. I did not look for another way around the refusal and did not stop the independent blocks.
2. **The first throwaway proof of N4 mis-classified an outage as a server refusal**: data-gov answered `409` when the terminal lake was unreachable (a `RuntimeError` from the transport wrapper). Fixed in data-gov (`ade6471`, `LakeUnreachable` → `503`) with a route test; the proof was re-run.
3. **feature-eng needed three passes on the throwaway stack**: an undeclared dependency set (`pandas_ta`, `keras`, `scikit-learn`), a pipeline that cannot run without an additional dataset, and a loader that parsed ISO 8601 with `dayfirst=True` (months and days swapped, days above 12 → `NaT`, every governed run silently emptied). Each is now a committed fix or fixture with a test.
4. **My first re-adjudication preview flipped 383 decisions**, because I had made `residual_signal_share` required in noise-free regimes (where it is undefined by construction) and let incompleteness precede measured damage. Both corrected before publishing the preview (now 138, §6).

## 2. N1 — lake/warehouse integration (done)

data-gov `2470e4b`: merge of `musashi/store-kinds-20260913 @f83f676` over master `8cd45f5`; `files_lake.py` resolved by content (kind metadata **and** the string-storage pin both kept). Suite 145 → 156 passed at the final tip; real-server regressions and store tests 29 passed; disposable three-service E2E exact on the combined tip. `/api/v1/lakes` over the deployed config: `financial_files = lake / files_inventory / http`, `olap_cube = warehouse / sql_olap / http`, `predictor_examples = lake / files_inventory / local`. README unchanged (no command changed). Data-gov master at `956dec7` includes everything below.

## 3. N2 — the contract's scope is executed, not noted (done)

A contract may declare `availability = {label, completion_lag_max, timezone_evidence, use_class}` and both lakes execute it: a day cut keeps a row only when `label + completion_lag_max < range end`; an `AS_IS` delivery under holdout needs `max(label) + lag < holdout`; ranges stay calendar days (intrabar requests refused); `LIVE_EQUIVALENT` needs a known label, zero lag and a producer time-zone statement. The four facts are published separately from the digest (`X-Availability-Label`, `X-Availability-Completion-Lag-Max`, `X-Timezone-Evidence`, `X-Availability-Use`), forwarded by `http_lake`, recorded per input by both consumers and tagged on every terminal (`availability_use`, the weakest scope among the inputs).

Tests (data-gov `da8b881`, financial-data `f00bc6c15`): scope validation; a 23:30 row excluded from its day's cut under a 1h bound and kept under 0s; interval extremes and incomplete days; train/calibration/confirmation cuts disjoint and **byte-identical after altering a future observation**; holdout with the bound; headers on both lakes. The three toy contracts now declare `WINDOW_END / 1h / UNKNOWN / OFFLINE_DAY_GRANULAR` (digest `5a521473…`), re-validated against bytes (`p02_validate_contracts.n2.out`), and the throwaway runs carry that scope into the cube (`p03_throwaway_governed_runs.n2.out`). The micro-run is `ARCHIVAL_REPLAY_NON_AUTHORITATIVE`: it proves transport and accounting, not causality or model quality; no `UNKNOWN` resource was promoted by inference (`data-gov/docs/07_…`, §2 and §6).

## 4. N4 — a permanent refusal is visible, traceable and disposable (done)

predictor `08a4c04`, `ef65e9b`, data-gov `79a2397`, `ade6471`: a refused send keeps its envelope with a sidecar (attempts, last error, class `TRANSIENT | CONFIGURATION | REFUSED_BY_SERVER | UNRESOLVED`; a 4xx alone never decides invalidity); `--status` separates recoverable, awaiting-adjudication, unresolved and adjudicated; `--dispose` moves an envelope unchanged to `adjudicated/` with a write-once record (`INVALID_ENVELOPE`); `--supersede` sends a corrected terminal as the next generation of the same campaign and unit (same outcome, same deliveries — a `FAILED` never becomes `COMPLETED`) and links it. Mirrored in `governed_exec` for the other repositories.

Throwaway proof (`p4_outbox_disposition_throwaway.out`, throwaway PostgreSQL created and dropped): `400` → awaiting adjudication; wrong key `401` → configuration, right key recovers; supersede accepted as generation 3, replay `200 already_stored`, counts unchanged; explicit `INVALID_ENVELOPE` with bytes preserved; a second campaign pending through a lake outage (`TRANSIENT`), recovered by one flush; both reconciliations exact. Adjudicated envelopes no longer block new governing runs; pending ones still do.

## 5. N5 — consumers that were only wrappers

**feature-eng — `PROVEN_DISPOSABLE`** (`p5_feature_eng_throwaway.out`, predictor `174cb88`): the real `tech_indicator` pipeline in feature-eng's own environment (`requirements-governed.txt`) from a clean worktree (`c086367`), two governed inputs (synthetic hourly OHLC and a daily series with explicit provenance and contracts: `tests/data/governed/`, seeded generator, manifest with hashes), `COMPLETED` with `rows/columns` metrics of `output_file`, six produced CSVs and the plot hashed as artifacts (`hourly_dataset_aligned`, `indicators_output`, `merged_features`, `seasonality_dataset`, `technical_indicators_aligned`, `vix_aligned`), lineage changes with input B (different delivery hash and campaign identity), stale outputs `REFUSED` without download, bogus plugin `FAILED` with cost and deliveries, a second flush sends nothing. Synthetic proves mechanics only.

**feature-extractor — `IMPLEMENTED`, pending**: wrapper, profile test, per-application environment (`requirements-governed.txt`: tensorflow-cpu, predictor installed non-editably as the *declared* provider of `stl_preprocessor`, no other checkout on the path). The small real run fails at `app/data_processor.py:72`: feature-extractor calls `run_preprocessing(config)` while predictor's plugin has been `run_preprocessing(self, target_plugin, config)` since `9b7d611` (2026-02-18). Minimum action (Musashi decides): pin a predictor commit before the drift as the provider, or port feature-extractor to the target-plugin API. Metric key names and the suffixed loss plot remain unverified until a run completes.

## 6. D2-R1/R2 — support contract and repair (done), preview of R3 (non-governing)

Frozen: Musashi's reproducer on the published decisions (`d2_support_r1/musashi_reproducer_frozen.out`, sha256 `f4958c88…`). The five cases physically: the seeds the reproducer flags have conserved rows with `status = INCONCLUSIVE`, reason `undefined for this unit`, for the primary metrics on the confirmation window — the clean signal has no variance there (no motif/step event inside the window), so the metric is undefined by its estimator; the adjudicator's `_seed_table` dropped every non-`COMPLETED` row and the seed still counted as valid.

Contract (`d2_support_r1/D2_SUPPORT_CONTRACT_R1.md`): per arm × unit × variable × metric — applicability from the contract (noisy vs noise-free, event kinds in the window from the unit's own event list, extreme geometry), status (`OBSERVED | INCONCLUSIVE | UNAVAILABLE | REFUSED | FAILED | MISSING_ROW | NOT_APPLICABLE`), reason, support, responsible component. Eight rules declared as tests before the repair: PRE 10/10 failing on the unrepaired adjudicator (`PRE_declared_tests.out`), POST 26/26 with the existing D2 lab suite (`POST_declared_tests.out`).

Repair (`41fee5b`, `91de809`, `a318668`): a valid seed is a **complete** seed; applicability is derived, never inferred from absence; floors and non-inferiority report applicable / unmeasured seeds; published `n_seeds_valid` = complete seeds; the evidence carries planned / observed / complete / abstained and unsupported / inapplicable per metric; `decide_snr` averages all required variables of a seed or declares it incomplete; measured damage on any observed seed still rejects (a post-result clarification of the sealed precedence, conservative). Margins, alpha, thresholds, seeds, partitions and metrics are unchanged. The loader (`df_load_d0_d2 --table-dir`) now refuses without a clean `--universe-check`.

Preview (`tools/df_d2_support.py` → `d2_support_r1/IMPACT_TABLE.json`, `UNIVERSE_CHECK.json`; full tables under the state root `d2_support_r1_v1/`): 3,591 → 3,591, **138 change, no decision gains a pass**:

| transition | n |
|---|---|
| `LAB_CALIBRATED → NOT_IDENTIFIABLE` (the five reviewed cases + two identity controls on flat noise-free windows) | 6 + 1 `REGIME_LIMITED → NOT_IDENTIFIABLE` |
| `LAB_REJECTED → NOT_IDENTIFIABLE` (rejections that rested on incomplete seeds, no measured damage) | 95 |
| `NOT_IDENTIFIABLE → LAB_REJECTED` (measured damage now rejects regardless of incompleteness) | 10 |
| `NOT_IDENTIFIABLE → UNDERPOWERED` (sealed order: underpowered before completeness) | 9 |
| `SNR_REJECTED / SNR_REGIME_LIMITED → SNR_NOT_IDENTIFIABLE` (seeds with an unestimated variable) | 13 + 4 |
| unchanged | 3,453 (incl. `LAB_CALIBRATED` 48 → 48 kept, `SNR_CALIBRATED_FOR_REGIME` 39 → 39 kept) |

Candidate pass count (arm role `CANDIDATE`): 51 `LAB_CALIBRATED` + 7 `REGIME_LIMITED` → **47 `LAB_CALIBRATED` + 6 `REGIME_LIMITED`** kept — corrected 2026-09-14 after the owner's finding; the earlier "48 + 6" was wrong, recount in `d2_support_r1/r3/R3_CANDIDATE_RECOUNT.json` (the 7 lost are listed with `COMPLETE_SEEDS n < DESIGN 30` reasons). SNR calibrations: 39 → 39. Universe: 416,873 expected applicable cells, 0 missing rows; one event-count disagreement, named: the non-causal oracle control on an impulse at index 2046 of a 2048-sample window (its window is not finite there) — never a decided arm. **This preview is not governing**: R3 registers it as a Flow v3 review campaign that consumes the conserved rows as evidence resources with a current receipt once N3 is done; nothing here overwrites a published row.

## 7. D2-R4 — portability, prepared, not replayed

`d2_support_r1/r4/`: margins of all 1,026 SNR decisions against every threshold (upper bound vs 1 dB, lower bound, coverage vs 0.90, not-identifiable rate vs 0.10, identifiable seeds vs design; `R4_SNR_MARGINS.json`); coordinator environment inventory by role (`R4_ENVIRONMENT_INVENTORY.json`; worker roles to be inventoried by the same tool when the replay runs); diagnostic subset frozen by rule before any replay (`R4_DIAGNOSTIC_SUBSET.json`: AT9 case, non-iterative controls of the 5 calibrated Kalman regimes, the 3 closest decisions to each limit; 16 regimes, 32 units by lowest seed ids; ≤6 CPU-h, ≤4 h wall, 2 GiB/process, one process per host, three roles only for the AT9 comparison). AT9 stays open under its original tolerance; 0.006 dB is not adopted.

## 8. D2-R6, D2-R7, N7

- **R6** `d2_support_r1/R6_COVERAGE_VIEWS.sql` + `R6_THROWAWAY_TEST.out`: the cube's `df_fact_coverage` holds one run id (`c140_…`) with two code digests (`31e3376d…`, `b4c9157c…`, 220,347 rows each); `df_fact_coverage_v2` holds `c170_…`/`b4c9157c…` (633,189). Proposed: a selection table (governed write) + `df_coverage_current` (explicit run + code) + `df_coverage_history` (every version with provenance) + a denominator view; rehearsed on a throwaway database: current 6 = the selected matrix, history 14 = 8 + 6, denominator not doubled, nothing deleted. Not applied to production. The v2 proposal carries a supersession header (data-gov `956dec7`).
- **R7** `docs/integracion_workplan_2026_09_10/07_DISENO_D3_…_2026_09_14.md`: plugin contract (state bytes, lookback, availability, warm-up, delay, cost, applicability, chunk/restart), nine operators with deliberate non-causal controls, ten acceptance tests, bank and dependencies. Not executed.
- **N7** `docs/integracion_workplan_2026_09_10/08_PLAN_INTEGRACION_FLOW_V3_…_2026_09_14.md`: tasks/components/tests for agent-multi (profile, read-once, freshness, gate on dispatchers, terminals, OLAP as artifacts), DOIN (`doin_governed_result.v1` referencing the terminal, ETL keeps identity, fixture before real publishers) and live (operational calls separated from governance; adoption only through offline replay; the `next()` API call is not removed from a live system).

## 9. Commits per repository (read from the repositories, all pushed)

| repo | branch | commits in this order |
|---|---|---|
| data-gov | `master` → `956dec7` | `2470e4b` N1 merge, `da8b881` N2 scope, `ade6471` unreachable lake → 503, `79a2397` N4 mirror + columns metric + glob artifacts, `beb2673` private `app` package loading, `956dec7` proposal supersession |
| financial-data | `satoshi/c122-c145-20260912` → `f00bc6c15` | N2 scope executed and published by the lake |
| predictor | `satoshi/c166-c184-20260913` → `3e2a18a` | `2470e4b`-era merges `93e87bb`/`39b1d2f` (orders and review by identity), `fa78edf` N2 consumer + R1 freeze, `06edee9` N2 proof, `ef65e9b` N4 + proof, `95c8d4e` docs/state, `174cb88` N5 proof, `41fee5b` R1/R2, `91de809` R2 tooling + loader gate, `a318668` R2/R4, `3e2a18a` R6/R7/N7 docs |
| preprocessor | `satoshi/crispdm-census-gate-20260910` → `20db3fb` | (unchanged this order) |
| feature-eng | `docs/agent-onboarding-20260816` → `c086367` | `7ceb711`/`c7319a6` fixtures, `c526b63`/`799af5a` environment pins, `c086367` ISO-first wall clock |
| feature-extractor | `docs/agent-onboarding-20260816` → `675b0bf` | environment pins |

Final suites: data-gov 156 passed; financial lake 61 passed; predictor (consumer, outbox, gate, dispatcher, D2 support, D2 lab, consumption gate, SNR isolation) 98 passed; feature-eng profile + wall-clock 3 passed; preprocessor and feature-extractor profile 1 each.

## 10. Coverage and evidence of loader/outbox

- Coverage per project (`tools/flow_v3_coverage.py`): production has no `gov_*` table yet, so no production matrix exists; on the throwaway stacks the cube received every state produced by the runners (`COMPLETED`, `FAILED`, `REFUSED`; `INCONCLUSIVE` only by the generic consumer's metrics-missing path in its tests; `QUARANTINED` not producible by these runners).
- Loader: `crispdm-olap-loader` `ActiveState=active`, `NRestarts=0` throughout; the real cube's only row changes since the previous packet are its own (`dim_campaign*`, `fact_campaign_unit`).
- Outbox: throwaway proofs in §4; production outbox untouched (no production run).

## 11. Blockers and requests (no new owner decision)

| pending | owner | minimum action |
|---|---|---|
| restart of the three services and the production micro-run (N3), then R3 and the R4 replay | Musashi (permitted environment) or the operator | N3 as ordered; the runbook is in the previous packet's Appendix A as a description; verify supervisor, environment, effective config, backups (`flow_v3_deploy_2026_09_13/backups/`), rollback, current pids |
| feature-extractor preprocessor API drift | Musashi (decision), Satoshi (execution) | pin or port (§5) |
| financial resource contracts | data producer/integrator | provider time semantics before any scientific use; not needed for synthetic trials |
| DOIN result contract `doin_governed_result.v1` | Satoshi designs, Musashi reviews | §8 plan, fixture before real publishers |
