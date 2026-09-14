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
| N2 temporal contract scope | `IN_PROGRESS` | |
| N3 bounded deployment | `BLOCKED(restart of :5057/:5056/:5055, Musashi in a permitted environment or operator, N3)` | recorded once here; Appendix A of the previous packet is a description, not a script to run literally |
| N4 permanent-rejection disposition | `NOT_STARTED` | |
| N5 feature-eng / feature-extractor real runs | `NOT_STARTED` | |
| N6 / D2-R1 PRE and support contract | `IN_PROGRESS` | |
| D2-R2 adjudicator repair | `NOT_STARTED` | |
| D2-R3 governed re-adjudication | `BLOCKED(reconciled production micro-run, N3)` | |
| D2-R4 AT9 portability | `NOT_STARTED` (preparation allowed) | |
| D2-R6 OLAP coverage view + proposal update | `NOT_STARTED` | |
| D2-R7 D3 design | `NOT_STARTED` | |
| N7 plans agent-multi / DOIN / live | `NOT_STARTED` | |
| N8 packet | `NOT_STARTED` | |

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
