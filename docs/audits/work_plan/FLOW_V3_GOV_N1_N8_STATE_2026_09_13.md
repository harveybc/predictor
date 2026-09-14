# Flow v3 — GOV-N1..N8 and D2-R1..R8: stage state and requirement → test → evidence matrix

Orders: `docs/handoffs/MUSASHI_TO_SATOSHI_FLOW_V3_DEPLOYMENT_ADOPTION_AND_D2_ORDER_2026_09_13.md`
(GOV-N1..N8) and `docs/handoffs/MUSASHI_TO_SATOSHI_D2_SUPPORT_PORTABILITY_NEXT_ORDER_2026_09_13.md`
(D2-R1..R8). Updated after every block; this file is the persistent state, not chat memory.

States: `NOT_STARTED` | `IN_PROGRESS` | `IMPLEMENTED` | `PROVEN_DISPOSABLE` | `DEPLOYED` |
`PROVEN_PRODUCTION` | `BLOCKED(<object>, <owner>, <minimum action>)`.

## Stage state

| block | state | evidence |
|---|---|---|
| N1 lake/warehouse integration | `IMPLEMENTED` + `PROVEN_DISPOSABLE` | data-gov `2470e4b` (merge of `f83f676` over `8cd45f5`, auto-resolved by content in `files_lake.py`: kind metadata + string-storage pin both kept); suite 145 passed; real-server + store tests 29 passed; E2E exact on the combined tip; `/api/v1/lakes` over the deployed config: `financial_files=lake/files_inventory/http`, `olap_cube=warehouse/sql_olap/http`, `predictor_examples=lake/files_inventory/local` |
| N2 temporal contract scope | `IMPLEMENTED` + `PROVEN_DISPOSABLE` | data-gov `da8b881`, financial-data `f00bc6c15`, predictor `fa78edf`/`06edee9`: `availability` block executed by both lakes (completion lag in cuts and holdout, calendar-day ranges), published in headers, recorded by consumers, tagged on terminals; toy contracts re-validated against bytes (`p02_validate_contracts.n2.out`, digest `5a521473…`); throwaway runs carry `OFFLINE_DAY_GRANULAR / WINDOW_END / 1h / UNKNOWN` into the cube (`p03_throwaway_governed_runs.n2.out`) |
| N3 bounded deployment | `PROVEN_PRODUCTION` (executed by Musashi, 2026-09-14) | acta `docs/handoffs/MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md`; services now run `.worktrees/musashi-n3-{data-gov,financial-data,predictor}-20260914T063541Z` at data-gov `ff4503a` (N1+N2+N4+operator console), financial-data `cc0f15e6e` (content byte-identical to my tested `f00bc6c15`), predictor `a7a86e9`; do not restart again |
| N4 permanent-rejection disposition | `IMPLEMENTED` + `PROVEN_DISPOSABLE` | predictor `tools/governed_run.py` + `flush_governed_terminals.py` (`--status/--dispose/--supersede`), data-gov `app/outbox.py` + `governed_exec` mirror; defect found and fixed: an unreachable terminal lake answered 409 (RuntimeError) and was classified as a server refusal → `LakeUnreachable` → 503 (data-gov `ade6471`); proof `p4_outbox_disposition_throwaway.out`: 400 → awaiting adjudication, 401 → configuration then recovery, supersede as generation 3 (same outcome/deliveries, replay already_stored, counts unchanged), explicit `INVALID_ENVELOPE`, isolation of a second campaign through an outage, both reconciliations exact |
| N5 feature-eng real run | `IMPLEMENTED` + `PROVEN_DISPOSABLE` | feature-eng `c7319a6`/`c086367`/`799af5a` (synthetic fixtures with provenance and contracts; ISO-8601-first wall clock — `dayfirst=True` swapped months and days on governed deliveries; `requirements-governed.txt` per-application environment: pandas_ta 0.4 + keras/tensorflow-cpu/scikit-learn undeclared in requirements.txt); proof `p5_feature_eng_throwaway.out` (predictor `174cb88`): COMPLETED with two governed inputs, six CSVs + plot hashed, lineage changes with input B, stale REFUSED, bogus plugin FAILED with cost, retry sends nothing |
| N5 feature-extractor real run | `IMPLEMENTED` (wrapper, profile test, `requirements-governed.txt`) — `BLOCKED(API drift: feature-extractor calls stl_preprocessor.run_preprocessing(config) but predictor's plugin is run_preprocessing(self, target_plugin, config) since predictor 9b7d611 (2026-02-18); Musashi decides: pin an older predictor or port feature-extractor; not modernised here)` | smoke in its own environment fails at `app/data_processor.py:72` with that TypeError; predictor installed non-editably as the entry-point provider, no other checkout on the path |
| N6 / D2-R1 PRE and support contract | `IMPLEMENTED` | `d2_support_r1/D2_SUPPORT_CONTRACT_R1.md`, `musashi_reproducer_frozen.out`, `PRE_declared_tests.out` (10/10 failing), `tests/test_df_d2_support.py` |
| D2-R2 adjudicator repair | `IMPLEMENTED` + preview | predictor `41fee5b`, `91de809`, `a318668`; POST 26/26; `tools/df_d2_support.py` → `IMPACT_TABLE.json` (138 of 3,591 change, no decision gains a pass), `UNIVERSE_CHECK.json` ok (1 named oracle-edge disagreement); `df_load_d0_d2 --table-dir` needs `--universe-check` |
| D2-R3 governed re-adjudication | `PROVEN_PRODUCTION` | campaign `d2-support-readjudication-r3` (`733736b7…`), unit `readjudication`, SYNTHETIC evidence spec binding generator/design/tape/reserve/conserved tables/code; terminal COMPLETED with 25 metrics and 7 artifacts, reconciliation exact; successor rows loaded additively under `d2r3_733736b795e075bffc8c1331` (3,591) with the published run untouched; evidence `d2_support_r1/r3/` |
| D2-R4 AT9 portability | `IMPLEMENTED` (preparation), replay **next** — its operational prerequisite now exists | `d2_support_r1/r4/R4_SNR_MARGINS.json` (1,026 decisions vs every threshold), `R4_ENVIRONMENT_INVENTORY.json` (coordinator role), `R4_DIAGNOSTIC_SUBSET.json` (16 regimes, 32 units, fixed rule; ≤6 CPU-h, ≤4 h wall, 2 GiB/process); tolerance unchanged |
| D2-R6 OLAP coverage view + proposal update | `IMPLEMENTED` + `PROVEN_DISPOSABLE` (not applied) | `d2_support_r1/R6_COVERAGE_VIEWS.sql` + `R6_THROWAWAY_TEST.out`; data-gov `05_PROPUESTA…` supersession header |
| D2-R7 D3 design | `IMPLEMENTED` (design only) | `docs/integracion_workplan_2026_09_10/07_DISENO_D3_…_2026_09_14.md` |
| N7 plans agent-multi / DOIN / live | `IMPLEMENTED` (plan only) | `docs/integracion_workplan_2026_09_10/08_PLAN_INTEGRACION_FLOW_V3_…_2026_09_14.md` |
| N8 packet | `IMPLEMENTED` (2026-09-13) + addendum for this round | `SATOSHI_GOV_N1_N8_D2_R1_R8_RETURN_PACKET_2026_09_14.md` and its addendum |

## Requirement → test → evidence

| id | requirement | test | evidence |
|---|---|---|---|
| N1-1 | files are `lake`, cube is `warehouse`, HTTP is transport | `tests/unit/test_store_kinds.py`, `tests/user/test_store_labels.py` | 29 passed on `2470e4b` |
| N1-2 | string-storage fix survives the merge | `tests/user/test_threaded_server_downloads.py` | passed on `2470e4b` |
| N1-3 | combined tip E2E | `tools/verify_flow_v3_e2e.py` | exact, contract `139f3adc…` |
| N2-1 | contract scope declared separately: label, completion bound, time-zone evidence, use class; `LIVE_EQUIVALENT` strict | `data-gov/tests/unit/test_availability_scope.py::test_scope_block_is_validated…`, financial `tests/test_availability_scope_v2.py` | |
| N2-2 | offline-by-day is executable: cut keeps `label + lag < range end`; AS_IS under holdout needs `max + lag < holdout` | `…::test_completion_lag_excludes_rows…`, `…::test_as_is_under_holdout_applies_the_completion_bound` | 23:30 row excluded under 1h, kept under 0s |
| N2-3 | intrabar use impossible; interval extremes; incomplete days | `…::test_ranges_are_calendar_days_never_intrabar`, `…::test_interval_extremes_and_incomplete_days` | |
| N2-4 | train/cal/test boundaries disjoint; an altered future observation leaves every cut byte-identical | `…::test_partition_boundaries_and_an_altered_future_observation` | |
| N2-5 | consumers record scope per input and tag terminals `availability_use` | predictor `tools/governed_run.py`, data-gov `tools/governed_exec.py` | governed suites |
| N2-6 | toy contracts carry the scope and re-validate against bytes | `p02_validate_contracts.n2.out` | ok, digest `5a521473…` |

## Operating rule after the production restart (owner, 2026-09-14)

Previous experiment closures continue **on the running services, without
interrupting them**: no further restart of `:5055/:5056/:5057`. A configuration
change (for example an evidence lake for downloadable D2 resources) is prepared
as a *pending* configuration through the operator console and activated in a
deliberate window — a pending file is neither an authorization nor a deployment.
Editing `master` does not update a detached runtime worktree, and the worktrees
in use are not modified or removed while they serve traffic. Incorporated into
the work plan in `docs/integracion_workplan_2026_09_10/09_INCORPORACION_ORDENES_ACTUALIZADAS_2026_09_14.md`.
