# Return: R1–R6, what was executed

Order: `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`
(`a9755eb`), over the review `MUSASHI_REVIEW_STANDING_ORDERS_2026_09_14.md`.

Every count below is mine and awaits Musashi's independent verification. No production service
was restarted, no cube history touched, no financial resource installed or licensed.

## Revisions

| repository | branch | head |
|---|---|---|
| predictor | `satoshi/r1-r6-20260914` | `77d55eb` |
| feature-eng | `satoshi/r1-r6-20260914` | `df7ea5b` |
| preprocessor | `satoshi/r1-r6-20260914` | `ad1a313` |
| feature-extractor | `satoshi/r1-r6-20260914` | `4ba9595` |
| agent-multi | `satoshi/moltbook-answer-every-reply-20260914` | `c3ee1880` |
| financial-data | `satoshi/ethusdt-4h-contract-successor-20260914` | `d0f81490a` |

## R1 — column roles, closed

Both counterexamples reproduced before any fix. `allow_target_as_feature` defaulted to True,
so a contract that never mentioned the overlap granted it; a column declared metadata *and*
feature reached the model because it happened to be numeric. Now: a **literal boolean** opt-in
(`"true"`, `1`, `[True]` are refused as configuration accidents), contradictory roles refused
by name, repeated declarations refused rather than deduplicated, malformed role lists refused
as lists.

Connected at each consumer's **real loader**, which is where the round found more than the
contract itself: three of the four coerced to numbers **before** consulting the contract, so a
timestamp declared as a feature became NaN (preprocessor, feature-eng) or **zeros**
(feature-extractor) and the run continued. `feature-extractor` had no contract at all, and its
`load_csv(..., force_date=False)` passed a parameter the function does not have — that path
raised TypeError before reading a byte. feature-eng's `load_and_fix_hourly_data` re-reads the
**main input file**, so a column could enter around the primary gate; it and the auxiliary
loaders are now covered, each file under its own declaration.

Admissibility is decided by identity and clocks, on a deliberately **flat** trajectory where
every value is identical: a window never contains the observation it predicts; equal values
are not a leak; `CLOSE` predicting future `CLOSE` is legitimate; a window shifted onto its own
target is refused although no value changed; a 150-minute publication lag makes the 04:00 bar
unpredictable from a window readable at 04:30, and a longer horizon restores it.

## R2 — the real plugins, closed

The previous battery defined its own `zscore_fit`, `windows` and a three-tap convolution.
Enumerated what the four smoke configurations actually select and ran those entry points
against perturbations of the future — appended rows, a shifted tail, a hole.

* `create_sliding_windows`, the deployed rolling filters (std, EMA, price-minus-EMA) and
  `_align_sliding_windows_with_targets`: invariant, constant inside their window as documented,
  trimming from the end without inventing a window.
* The deployed **normalizer**: A is fitted on D1 alone (a shift from row 200 leaves mean and
  deviation identical; a change *inside* D1 moves them), B is a separate fit on D4.
* feature-eng's fifteen indicator columns: no lookahead under either perturbation, and the
  **warm-up recorded per indicator** — MACD_Histogram and MACD_Signal 33 rows, ADX 27, MACD 25,
  down to EMA 9. A run trimming fewer than 33 rows feeds a model values that do not exist yet.
* The two real decompositions. `stl_pipeline` and `stl_preprocessor` **decompose nothing**;
  the genuine operators are `target_plugins/stl_target.py` (statsmodels STL) and
  `tools/df_snr.py`'s db4 kernel, already guarded as
  `OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL`.

Three measurements contradicted me and are recorded as measured: a *consistent* future barely
moves STL's past estimate (~3.6e-14) while a deviating one moves it, worst at the right edge;
a NaN does not trip the `except` — statsmodels propagates and every component returns all-NaN;
and that `except` does not return zeros either, because `np.zeros_like` of a non-numeric input
returns an array of *that* dtype while printing "Returning 0s".

Every block carries its check on the check, applied to the **productive** path: a mutated
`_add_window_stats_features`, a `Series.rolling` forced to `center=True` inside the deployed
`process`, a centred `pandas_ta.rsi`, an alignment trimmed from the wrong end.

Fixture regeneration repaired: a manifest-only checkout used to die on `open()`. Each
situation now has a name — verified, regenerated into temporary storage with its digest
checked, **mismatched original preserved**, or unreproducible when the generator no longer
matches `generator_sha256`.

## R3 — reproducible replay, closed

Two defects the audit named, both closed. The runner read the repository-global
`config_out.json` as a metric source, so a file left by any earlier run could be reported as
this run's work; only the run's own summary counts now, and one older than the run's start is
recorded `stale_ignored`. `total_timesteps` — a configured budget — travelled as a measured
count; it is now `requested_timesteps`, against `observed_timesteps` / `observed_updates`, and
a missing observation stays missing.

Custody binds template, overlay, resolved config, runner source, entry point and input digest.

**Stranded outbox, really stranded**: the real `TerminalOutbox` and the real `DataGovClient`
against an HTTP server that is stopped and restarted. Destination gone → the envelope stays on
disk and the refusal does not delete it → destination back → **exactly one** terminal received
→ moved to `sent/` → a second flush sends nothing. Delegation proven on the real call, not on
source text.

**The corrected production micro-run**: `doin-offline-replay-prod-11`, COMPLETED,
NON_GOVERNING, campaign `769a498b5ec1…`, reconciliation empty on all three lists, 15.7 s wall.
Its receipt reports `requested_timesteps 64` and **no** `observed_timesteps`, because the
application wrote no summary. The earlier receipt claimed 64 as work done; this one says what
was asked and admits what was never observed.

Getting there exposed a real defect: the governed path was written only under `data`, while
the application reads the flat `input_data_file`, so the replay opened a sample file the
campaign never delivered.

## R4 — the archive class

Nine rules on `ARCHIVE_RETROSPECTIVE`: recognised; its UNKNOWN lag stays `None` rather than
becoming a zero timedelta and travels to the receipt as `"UNKNOWN"`; a ranged delivery is
refused naming the retrospective archive and asking for the whole resource; a contract of that
class carrying a number is refused, and another class borrowing `"UNKNOWN"` is refused too;
`LIVE_EQUIVALENT` and `OFFLINE_DAY_GRANULAR` still resolve, so the change is additive.

Named rather than implied: **the class is not in the provider the lake host imports today**.
The semantics are proven on the candidate implementation, and a rule fails the day the
deployed module gains the class, so the gap cannot rot quietly.

## R5 — primary sources, measured and registered

Three windows declared and sealed **before** any request; then three GETs to the producer's own
public endpoint, 529,443 bytes, no account key, no retry, nothing in the lake opened for
writing. Executed on **dragon**.

3,000 bars matched by `open_time`: **zero material differences** — no revision, including the
21 anomalous bars. What it did find is ours: 145 bars differ in the last digits of
`quote_volume` and `taker_buy_quote_volume`, relative 2e-16, one ULP of float64. The producer
sends decimal strings and the archive stores them as float64, so those two columns do not
round-trip.

Registered through the existing accounting: campaign `d3cf00b6ea06…`, terminal
`2a4b455d7e9a…`, reconciliation clean, metrics `bars_compared 3000` and
`bars_materially_differing 0`.

Agreement at one moment settles neither finality nor the absence of revisions. **Usage rights
remain the owner's only action**: the terms applying to that market data for analysis, derived
artefacts and publication.

## R6 — the fleet, and what is still open

The three machines now take their data **through data-gov**: a governed delivery succeeded from
omega, gamma and dragon, verified against `X-Content-SHA256`, and gamma's second campaign
transferred nothing and confirmed from the content cache — "download once per group" is
deployed behaviour, not a plan. The stores bind `127.0.0.1`, so the workers reach them over a
reverse tunnel opened from omega.

| open item | owner | what exactly |
|---|---|---|
| per-machine principals | operator | the runtime configuration now carries `satoshi-gamma` and `satoshi-dragon`; loading them needs `systemctl --user restart crispdm-data-gov.service`, which this session is not permitted to run |
| capability tokens per delivery | decision pending | replaces the shared static lake token so the lake's own log attributes a download to an experiment |
| usage rights of the market data | owner | the only question this repository cannot answer |
| `p1lr-decision@202` on dragon | Musashi | refuses with `REFUSED_SCREEN_GATE_MISSING`; the gate is doing its job, the verdict file does not exist |

Suites: predictor 81, feature-eng 44, feature-extractor 46, preprocessor 40, agent-multi 11
new rules. Services healthy on all four ports, zero restarts, cube history untouched.

Stop: `R1_R6_EXECUTED_COUNTERS_OBSERVED_ARCHIVE_UNKNOWN_KEPT_AND_THE_FLEET_FED_BY_GOVERNANCE`.
