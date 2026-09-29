# Registered TSL benchmarks

Verified through the live governed route on 2026-09-29 02:01 UTC
(2026-09-28 local time). No scientific experiment was run.

Lake identifier: `sota_benchmarks`.

| Resource | Variables | CSV columns including date | Rows | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `thuml_tsl_electricity/electricity.csv` | 321 | 322 | 26304 | 95581762 |
| `thuml_tsl_weather/weather.csv` | 21 | 22 | 52696 | 7235425 |
| `thuml_tsl_traffic/traffic.csv` | 862 | 863 | 17544 | 136478119 |

Distributor: `thuml/Time-Series-Library`, revision
`2b66e59ee19dac8f6f19fb5d4997f289fdfea357`.
The local CSVs are unchanged distributor bytes. Electricity and Traffic match
the upstream LFS SHA256; Weather matches the upstream Git blob identity and
has an independently recorded SHA256.

## Access and semantics

Register a campaign naming the lake, exact resource, and a dataset role;
request a whole-resource governed delivery with its campaign and unit IDs.
These are retrospective public benchmarks: `AS_IS`, availability `UNDECLARED`,
availability label `UNKNOWN`. Their date column is not proof of publication
or reception time. Date-ranged requests are refused. They are not authorized
as point-in-time/live financial data by this registration.

Nominal sample intervals are one hour for Electricity and Traffic and ten
minutes for Weather. No preprocessing, imputation, normalization or split
selection was performed as part of the registration.

## Acceptance

All three resources returned `VERIFIED_TRANSFER`, their delivered files
matched the retained SHA256, campaign reconciliation had no missing units,
and the warehouse content matched the emitted terminals. Refusal checks
passed for a date range, an undeclared resource and a download without a
campaign. Tests were first run through disposable services and then live.
Each route test deliberately records one completed unit and one failed probe;
these are operational tests, not scientific training results.

Only the existing benchmark lake service restarted. The data-gov config was
unchanged; data-gov and warehouse were not restarted. Electricity's bytes and
its existing resource contract were preserved. The original BUILD_RECEIPT
was not rewritten: new resource provenance is in the additive
`WEATHER_TRAFFIC_EXTENSION_20260928.json` beside it.

Operator receipts are retained under
`~/.local/state/crispdm-data-foundation/tsl-extension-20260928/`:
`PLAN.json`, `RESOURCES.json`, `REHEARSAL.json`,
`REHEARSED_BINDING.json`, `ADOPTION.json`, and the original host config backup.
The first rehearsal stopped on the clean-checkout requirement before any
production change; its receipt is preserved as `REHEARSAL.attempt1.json`.

SHA256 identities:

```text
electricity 7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad
weather     34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63
traffic     cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16
```
