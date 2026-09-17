# Return: D3 J1–J3 — the temporal amendment, the proved contract, the nine operators, the governed mechanics

Order: `docs/handoffs/MUSASHI_R6_ACCEPTANCE_AND_D3_J1_J3_2026_09_16.md` (`810d4c9`), over the
acceptance of `54f2df0`. Executed without pausing between blocks. Nothing here is scientific
utility; every run is `NON_GOVERNING` mechanical evidence and the readiness states are kept
apart: **INFRASTRUCTURE_PRESENT** (R2, R6, N3), **TEMPORAL_BATTERY_ACCEPTED** (measured per
operator and per unit by the run below), **SCIENTIFIC_UTILITY** (not claimed).

## J1 — the amendment, sealed before any candidate

`docs/integracion_workplan_2026_09_10/07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md` supersedes §1 and
§3 of design 07. The original bytes are untouched and cited by digest
(`45959063e01895c6…`, commit `3e2a18a`); `tools/df_d3_design.py` holds the same amendment as
data, sealed by `design_sha256` and re-validated against the original file on every load — if
the original ever changes, the amendment refuses to validate.

The reviewer's four findings, each answered by a rule rather than a paragraph:

| finding | rule now |
|---|---|
| `delay + lookback` "made" a late input available | four instants are separate; `emitted_at[t] >= max(available_at)` over the inputs the output consumes, plus the declared emission delay; past lookback never enters availability |
| `'0s'` refused, `int(lag)` truncates | durations parsed with the producer's own parser (`pandas.Timedelta`, as `data-gov files_lake`); converted to samples **only** by exact division under a declared `frequency`; a fractional offset is `FRACTIONAL_SAMPLE_OFFSET`, refused, never truncated; `UNKNOWN` stays `UNKNOWN` |
| the centred twin passed test 1 | every output available at the cut is compared — value, mask, `emitted_at` — with **no** lookback exemption; a delayed output is compared after its emission; an empty population is `INSUFFICIENT_TEST` |
| two operators had no twin and passed by `undecided` | a twin is required where meaningful; `NOT_APPLICABLE` only with a design reason; absence is a **refusal** |

Plus: predetermined seeded cuts (`2^j±1`, warm-up±1, window multiples, edges, seeded random)
at two lengths and three missingness regimes; fit and restart tested apart; state instantiated
fresh per branch; impulse onset distinguished from group delay with a declared probe and an
`UNIDENTIFIED` outcome allowed, never a fabricated zero; wavelet support derived from the
library (`(dec_len−1)·(2^L−1)+1`); recursive filters carry `zi` state, not a memory of order p.

## J2 — the contract proved before candidates

`tools/df_d3_contract.py` v2 (schema `d3_operator_spec.v2`): `response_probe`,
`non_causal_twin`, `support`, `delay_samples` as the emission delay, and an input/output
contract that carries `timestamps`, `available_at`, `period_seconds` and `emitted_at`. An
output emitted before its own input was available is refused at the contract, before any test.

`tools/df_d3_acceptance.py` v2: the twelve tests of the sealed required-test matrix, with
`review_ready` true only when every mandatory test ran and passed; a scoped test is reported
as scoped, an undecided one makes the verdict `INCONCLUSIVE`.

`tests/test_d3_temporal_contract.py` — **40 rules**, every one against the real harness
callables: the four findings as negatives; a causal zero-lag control and a delayed-output
control as positives; fractional durations, late arrivals, missing timestamps, a real `'0s'`
contract; window padding and centring; whole-series normalisation caught and train-only
normalisation accepted; a transform that mutates its state caught; changed future fit data;
restart by re-read lookback and by checkpoint; probes; wavelet support; recursive support.

One defect found in my own v2 on the way, fixed and frozen: the availability test recomputed
`available_at` from the contract's nominal lag and **erased a late arrival the snapshot had
recorded**. The later of the two now wins.

## J3 — the nine operators, and their twins

`tools/df_d3_operators.py`, on numpy 2.5.1, scipy 1.18.0, pywt 1.8.0 (versions recorded in
every declaration):

| operator | group | fit | twin | probe |
|---|---|---|---|---|
| `uniform_decile_quantizer` | quantization | train | NOT_APPLICABLE — pointwise codec | step 0 |
| `sax_paa_trailing` | quantization | train | `sax_paa_centred` | step 0 |
| `delta_run_length` | quantization | none | NOT_APPLICABLE — only `x[t−1]` | impulse 0 |
| `stft_trailing` | time-frequency | none | `stft_centred` | impulse **1** |
| `wavelet_trailing` (db4, L=3, support 50, zero boundary) | time-frequency | none | `wavelet_centred` | impulse 0 |
| `butterworth_causal` (order 2, `lfilter` + `zi`, checkpoint) | time-frequency | none | `butterworth_filtfilt` | impulse 0 |
| `cusum_causal` (k, h from train, checkpoint) | detectors | train | `cusum_lookahead` | level shift 0 |
| `mad_extremes_trailing` | detectors | train | `mad_extremes_centred` | impulse 0 |
| `variance_regime_trailing` | detectors | none | `variance_regime_centred` | variance shift 0 |

**Measured, not assumed.** The Hann window weights its newest sample by exactly zero, so the
trailing STFT for `t` does not see `x[t]` and an impulse first moves it one sample later. The
operator declares onset **1**. Reshaping the window to make it 0 would be tuning against the
evaluation fixture, which the order forbids.

`tests/test_d3_operators.py` — **37 rules**: each operator review-ready with its twin; each
twin fails causality on its own; the two codecs scoped with the amendment's reason; the STFT
onset; wavelet support from the library; recursive declarations; checkpoint/resume for the
two stateful operators; a missing value yields an unavailable output for every operator.

## J3 — the governed mechanics

| piece | what it does |
|---|---|
| `tools/df_d3_unit_worker.py` | one unit per process under the D2 child protocol (`result.json`, re-hashed by the parent); rows `df_fact_d3_mechanics.v1` per unit × variable × operator × test, plus a verdict row; `--pilot` for cost |
| `tools/df_d3_toy_resources.py` | the seven toy resources **delivered** under a DATASETS campaign (`X-Delivery-ID`, confirmed `VERIFIED_TRANSFER`) and materialised with the lake's own availability (`WINDOW_START`, `4h`/`1h`) and frequency; backwards timestamps refuse |
| `tools/df_d3_campaign.py` | `freeze` (population + budgets + cost pilot, sealed), `toys`, `shards` + jobs, `sync` (clean detached worktree per worker), `collect` (rsync + digest verification), the MECHANICAL envelope |
| `tools/df_d3_report.py` | two data-gov campaigns under one run id — SYNTHETIC for bank units, DATASETS for toys — terminals through the durable outbox and reconciled one by one; one MECHANICAL envelope to the OLAP outbox for the running loader |

`tests/test_d3_mechanics_pipeline.py` — **10 rules** on a throwaway bank: unit loading under
both semantics, row identities, deterministic population selection, write-once freeze, shards
and jobs after the D2 precedent, terminal metric identities unique, the envelope valid as
`MECHANICAL`.

### The frozen run: `d3mech-v1`

* Population: **504** bank units (lengths 512 and 2048; none/mcar/blocks) + **7** toy units;
  `FREEZE.json` `5f6c1ca564b57b7d…`.
* Budget from the pilot and a three-unit smoke: ~25 s per variable at N=2048, 170 MB peak →
  **120 s per variable**, 2 GiB per task, one unit per process; 43 shards of 12.
* Cost pilot: every operator within its declaration; the costliest is Butterworth at
  1.08 s/1k on the 512-unit.
* Roles: WORKER_A (cap 6) and WORKER_B (cap 2); the COORDINATOR is **excluded** (cap 0)
  because its preflight shows one GPU compute process — the owner's — and this work is
  CPU-only; it stays the governance and collection host. Both workers preflighted
  `dispatchable` at commit `44066b1`, clean, code digest equal, C161 data manifest verified.

Two defects found by the first dispatch, each frozen as a rule and fixed before the second:

1. `df_dispatch` refused **every first dispatch under a sealed Flow v3 gate**: the gate writes
   `DISPATCH_GATE.json` into the root before the dispatcher's write-once check, which then saw
   "root exists". A root holding only the gate's own files is now a fresh root.
2. My jobs file wrote the interpreter and the output root as bare relative paths; the launch
   script expands `~/` and nothing else, so every shard died with exit 127. Kept beside the run
   as `dispatch.attempt-1-exit127/` and `JOBS.attempt-1-exit127.json`.

### Matrix outcome

MATRIX_PLACEHOLDER

## Suites

| suite | scope | result |
|---|---|---|
| D3 contract + operators + pipeline | trading-stack | **87 passed** |
| migration + reconciler + watch + stack + R6 + dispatch + gate + `olap/store/tests` | three engines | **330 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, with the store environment | SUITE_PLACEHOLDER |

## Open, with owners

| item | owner |
|---|---|
| what made `gov_terminal_metric_sha_idx` lose four entries — four mechanisms reproduced and excluded | Satoshi; unknown, not named; independent of this work |
| Metabase driver decision, historical terms research | Satoshi; nonblocking |
