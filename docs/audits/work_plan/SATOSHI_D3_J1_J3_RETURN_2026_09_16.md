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

`tests/test_d3_mechanics_pipeline.py` — **13 rules** (+ `test_d3_matrix.py`, 1) on a throwaway bank: unit loading under
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

**Dispatch.** 43 shards: WORKER_A 444 units, WORKER_B 67. 42 shards `COMPLETED` on the first
attempt; shard_42 `FAILED` because its three 26-variable toy feature units hit `WALL_TIME_LIMIT`.
Cause, mine: the frozen budget is 120 s **per variable**, and I scaled the shard wall by variables
but handed every child the flat figure. Fixed in `d2bbf79` (the runner multiplies wall and CPU by
the unit's own variable count; a retry takes the next attempt number; the killed attempt stays on
disk), frozen as two rules, workers re-synced and re-preflighted, and the three units re-run as a
versioned retry (`JOBS.retry-1.json`, `dispatch.retry-1/`, `COLLECT.retry-1.json`) on the same
worker: each took 133 s, so the flat budget had been short by 13 s. The first receipt is kept as
`COLLECT.attempt-1.json` (508 verified, 3 `RESOURCE_EXCEEDED`). Collect on the coordinator
returns rsync 23 — it ran nothing (cap 0) — and is recorded as such.

**Population and cost.** 511 units verified of 511 (digest and row count re-checked by the
collector), 0 mismatches, 83,070 rows, 2.91 h wall / 2.89 h CPU over the two workers. 486
units without missingness, 9 `blocks`, 9 `mcar` (15 variables), 7 toys (33 variables).

**Matrix** (`MATRIX.json` / `MATRIX.md` under the run root, from `tools/df_d3_matrix.py`; every
figure is a count of what the battery said):

Run `d3mech-v1`: **511** units verified of 511 collected, 0 digest mismatches, 83,070 rows.

| operator | group | units × vars | verdicts | causal tests failed | restart | availability | probe onset | cost s/1k (median, max) |
|---|---|---|---:|---|---|---|---|---|
| `butterworth_causal` | time_frequency | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 0.0 710 | 0.0059, 0.0098 |
| `cusum_causal` | detectors | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 0.0 710 | 0.0063, 0.0088 |
| `delta_run_length` | quantization_compression | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 0.0 710 | 0.0005, 0.002 |
| `mad_extremes_trailing` | detectors | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 0.0 710 | 0.0005, 0.002 |
| `sax_paa_trailing` | quantization_compression | 511 × 710 | MECHANICALLY_ACCEPTED 704 / MECHANICALLY_REFUSED 6 | none | PASSED 710 | PASSED 710 | 0.0 704 / 1.0 1 | 0.0005, 0.002 |
| `stft_trailing` | time_frequency | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 1.0 710 | 0.0005, 0.002 |
| `uniform_decile_quantizer` | quantization_compression | 511 × 710 | MECHANICALLY_ACCEPTED 647 / MECHANICALLY_REFUSED 63 | none | PASSED 710 | PASSED 710 | 0.0 647 / 1.0 58 | 0.0005, 0.002 |
| `variance_regime_trailing` | detectors | 511 × 710 | MECHANICALLY_ACCEPTED 710 | none | PASSED 710 | PASSED 710 | 0.0 710 | 0.0005, 0.002 |
| `wavelet_trailing` | time_frequency | 511 × 710 | INCONCLUSIVE 6 / MECHANICALLY_ACCEPTED 696 / MECHANICALLY_REFUSED 8 | non_causal_twin 8, warm_up_edge 3 | INSUFFICIENT_TEST 7 / PASSED 703 | INSUFFICIENT_TEST 3 / PASSED 707 | 0.0 710 | 0.0011, 0.002 |

**Declared vs measured availability.** Bank units are `SAMPLE_INDEX` with a real `'0s'` lag.
The toys carry the lake's `WINDOW_START` + `completion_lag_max 4h` (or `1h`) under
`frequency 4h` (`1h`): the lag divides exactly into **one sample**, and all nine operators
emitted no output before `available_at` on any of the 33 toy variables (`availability_emission`
PASSED on 900/1800 outputs per variable). Nothing was truncated; no `FRACTIONAL_SAMPLE_OFFSET`
arose because the toys' contracts divide exactly.

**Restart evidence.** `chunk_restart` PASSED on 6,383 of 6,390 operator × variable cases (re-read
lookback for the windowed operators, checkpoint for Butterworth's `zi` and CUSUM's statistic);
the 7 `INSUFFICIENT_TEST` are wavelet cases with no available output after the restart point,
below.

**What was refused or inconclusive, and why — recorded, not tuned:**

1. `uniform_decile_quantizer` refused on **63** variables and `sax_paa_trailing` on **6**, all by
   `response_probe`: 58 + 1 "moved one sample after the step; declares 0", 5 + 5 "the step moved
   no available output". Both codecs are memoryless in time — output *t* consumes only the
   trailing window ending at *t* — so an onset of 1 is not a temporal property they can have. The
   battery's step probe is **+25 on N(0,1) noise, unscaled to the operator's fitted domain**: a
   quantizer whose deciles were fitted on a unit in the thousands (`trend_linear`, `bumps`,
   `impulses`, the OHLC toy) has bins wider than 25, so the step crosses no bin edge at *p* and
   crosses one at *p+1* by the luck of the noise, or crosses none at all. This is a **fixture-scale
   artefact of the probe**, the same class as the reviewer's finding on probes: the honest outcome
   is `UNIDENTIFIED` (onset not measurable at this amplitude), not a fabricated 0 and not a
   measured 1. I did not change the battery after the run; the refusals stand as recorded and the
   rule proposed for the next amendment is: a probe amplitude is declared **relative to the
   operator's resolution** (for a fitted codec, at least one bin width beyond the train range), and
   a step that moves nothing at *p* yields `UNIDENTIFIED`, never a failed onset.
2. `wavelet_trailing` (db4, L=3, support **50**, any-NaN → unavailable) on the 9 `mcar` units
   (15 variables): 8 refused, 6 inconclusive, 1 accepted. Under 10 % MCAR the chance that a
   50-sample window is complete is 0.9⁵⁰ ≈ 0.5 %, so the operator emits almost nothing:
   `warm_up_edge` FAILED 3 ("a warm-up that never ends is not a warm-up"), `prefix`, `restart`,
   `fit_scope` and `availability` `INSUFFICIENT_TEST`. That is a true mechanical inapplicability
   of this operator, as declared, under that missingness — recorded. Beside it, one **battery
   gap** found by the run: `non_causal_twin` FAILED 8 with "the declared twin passed the
   causality tests" — the centred twin also emitted nothing, its causality tests were
   `INSUFFICIENT_TEST`, and the battery read a vacuous pass as a pass. A twin with no output must
   be `INSUFFICIENT_TEST`, not evidence of a wrong twin. Proposed for the next amendment; not
   patched after the fact.
3. `stft_trailing` declares onset **1** and measured **1** on all 710 variables (Hann newest
   weight zero, J3 above). Declared, measured, consistent.

Everything else: 6 of 9 operators `MECHANICALLY_ACCEPTED` on all 710 variables, with every
causal test PASSED, restart PASSED, availability PASSED and cost within declaration
(costliest Butterworth/CUSUM at ~0.006 s per 1,000 samples median).

**Reconciled terminals.** RECONCILE_PLACEHOLDER

## Suites

| suite | scope | result |
|---|---|---|
| D3 contract + operators + pipeline | trading-stack | **87 passed** |
| migration + reconciler + watch + stack + R6 + dispatch + gate + `olap/store/tests` | three engines | **330 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, with the store environment | **1514 passed, 13 skipped**, 0 failed, in 7m28s |

## Open, with owners

| item | owner |
|---|---|
| what made `gov_terminal_metric_sha_idx` lose four entries — four mechanisms reproduced and excluded | Satoshi; unknown, not named; independent of this work |
| Metabase driver decision, historical terms research | Satoshi; nonblocking |
