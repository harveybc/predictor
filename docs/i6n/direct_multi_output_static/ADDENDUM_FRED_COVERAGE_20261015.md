# Addendum: FRED publication-clock coverage wording (2026-10-15)

This addendum corrects one sentence of a published return. It does not edit
that return or any evidence file. The label stays
`PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD`.

## The sentence being corrected

`docs/i6n/direct_multi_output_static/RETURN_MARKET_TIME_PAIR.md` (introduced
in commit `1dbe918d`; file sha256
`a6178dcadbe212dcb0e113b8846a4549ef291d7d2aefa5b12a96191425938e63`), section 7,
says:

> FRED publication clocks: `PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD` — 339 of
> 1,587 observations covered, with the 2020-08-25 back-fill

The phrase "339 of 1,587 observations covered" gets the audit backwards.
**339 were not covered.**

## What the audit measured

Evidence file:
`docs/i6n/direct_multi_output_static/market_time/provenance/PROV01_RESULT.json`
(sha256 `d794a1cc9b13a7139b60708d3a42b83fceefbc55aecdb7141acb15a1f07a4479`).
Its raw ALFRED pull is
`market_time/provenance/ALFRED_PROVENANCE.json` (sha256
`48115d60060ed73ad7542ad7b4835f27d0b53f5cbd4567467d132bf0b2d9b56b`).

The producer declared that each FRED value becomes available at `(D+2) 00:00 UTC`.
The audit compared this declared time with ALFRED `realtime_start`, which is
the date the value entered the FRED/ALFRED database. The results are identical
for both series:

| Series | Compared | `realtime_start` after the declared time (not covered) | Covered by the declared time | Not covered | Same-day check (C1) |
|---|---:|---:|---:|---:|---|
| DGS3MO | 1,587 | 339 | 1,248 | 21.36% | PASS |
| DGS7 | 1,587 | 339 | 1,248 | 21.36% | PASS |

The calendar-day lag histogram gives the same split for each series: 1,176 at
lag 1 and 72 at lag 2 are covered (1,248). There are 269 at lag 3, 68 at lag 4
and 2 at lag 5, for 339 not covered. Most of the late entries are Friday
observations that entered the database on the next Monday
(`C3_lag_by_observation_weekday`). The audit's own counterexamples
(`C4_counterexamples`) caught both the doctored coverage row and the injected
same-day row.

## Corrected sentence

> FRED publication clocks: `PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD`.
> For each of DGS3MO and DGS7, 339 of 1,587 observations (21.36%) have an ALFRED
> `realtime_start` later than the declared `(D+2) 00:00 UTC` instant. They are
> outside the declared coverage. The other 1,248 are covered. The 2020-08-25
> back-fill note is unchanged.

## What this does not claim

The evidence's own `evidence_semantics` field says ALFRED `realtime_start` is an
**upper bound** on public availability. The H.15 release is posted at 4:15 pm ET
on the observation day. A late `realtime_start` refutes coverage *by this
artifact* only. Database ingestion dates alone do not prove look-ahead, so this
addendum claims none.

The independent review
`market_time/REVIEW_MARKET_TIME.md` already states the direction correctly
("for 339 of 1,587 observations (21.4%) the ALFRED availability date is strictly
later than `D+2`").

The market-time run itself, the published metrics and the TEST closure are
unchanged.
