# I6-N plan audit, ADDENDUM, lane calendar (I6N-AUD-07, -08, -10, -11, -29; owner point 6a)

Date: 2026-10-11. This addendum does not replace the original audit (commit
4d3af09d), and no original file was edited. The verdict stays
**REVISE_BEFORE_IMPLEMENTATION**. No model was trained, and no service, timer,
GPU, broker or campaign was touched. Only `ts` arrays dated on or before
2024-12-31 were read, on CPU under `crispdm-run`. Inputs and digests are in
`INPUTS.json`, the exact commands in `COMMANDS.sh`, and file digests in
`MANIFEST.json`.

Classification vocabulary:
- DEMONSTRATED_DEFECT: shown by retained, reproducible evidence on the
  concrete object.
- RISK: a plausible failure not yet tied to the concrete route.
- OWNER_REQUIREMENT: the contract requires it; leaving it out is
  non-compliance.
- METHODOLOGICAL_ALTERNATIVE: a recommendation the owner may decline.
- RETRACTED: an earlier claim that was wrong or overstated.

## Calendars and attribution rule

Every number below names its calendar. A number without a calendar is not
evidence.

- **NY** (assumed): O = Sunday 17:00 America/New_York, C = Friday 17:00
  America/New_York, no holidays. This is the calendar the original counts
  used.
- **OBS-END** (proxy): O and C are taken from the first and last bar of each
  week in the `ts` array, with END labels and the week split at Saturday
  00:00 UTC. MWC:53-55 says that a CSV gap alone does not identify a market
  closure, so OBS is a sensitivity proxy and never the calendar. The week of
  2024-12-29 cannot show its C without reading TEST rows, so its C is taken
  from NY. That week is flagged HYBRID.
- **OBS-OPEN**: the same proxy with OPEN labels.
- **Sensitivity grid**: offsets f = e in {6, 12, 18} h, crossed with an entry
  cut of Wednesday 12:00, Thursday 00:00 or Thursday 12:00 NY. The grid is
  run for NY and OBS-END, which gives 19 calendars in total.

**Calendar-independent results.** Over 2,014 week-instances in all 19
calendars, these identities held with 0 violations
(`calendar_attribution.out` §1):
- management E = max(0, (B−A) − h)
- entry E = max(0, min(cut−A, (B−A) − h))
- the last admissible origin has R(t) = B − t = h + 1, so the inequality
  h < R(t) is strict
- the minimum number of long horizons over entry origins is
  #{h ∈ LONG : h < (B − cut) + 1}

The following also hold for any calendar:
- long horizons sample only the first (B−A) − h hours after A;
- consecutive hourly h = 72 targets overlap by 71 h;
- the two-long minimum binds if and only if B − cut < 24 h.

**Input identity.** The original probes read
`$STATE/.../statistical72_v1/INPUT.npz` (sha256 `897caee1…`). Its `ts`
member (sha256 `a094d141…`) is byte-identical to the `ts` member of the I6-E-M
INPUT (`d9015867…`), so every count also applies to I6-E-M. The three
retained probes (`bar_clock`, `gaps_2024`, `wmc_calendar`) were rerun
unmodified, and their outputs are byte-identical to the retained files
(`rerun_*`). The new scripts are deterministic: each was rerun and the
outputs were byte-identical.

## Findings table

| Claim | Verdict | Class | Evidence (calendar) | Minimal correction |
|---|---|---|---|---|
| 07a: "management = 96−h, entry = min(67, 96−h)" | QUALIFIED | OWNER_REQUIREMENT (support table, MWC:121-122; PROP:50 N0) | The identity is calendar-independent (0 violations). B−A = 96 and A→cut = 67 are NY values (53/53 weeks). Under OBS-END, B−A = 96 in 51/53 weeks, 72 in the Jan-1 reopen week and 95 in the week of 2024-10-27; A→cut is 67 in 49 weeks and 68 in the 3 March DST-gap weeks. With f = e = 6, B−A = 108; with f = e = 18, B−A = 84. | Report the general formula and seal the per-h/week/role table under the versioned calendar. |
| 07b: "all h=72 origins fall Mon 05:00-Tue 04:00 NY" | QUALIFIED | METHODOLOGICAL_ALTERNATIVE (representative choice) | NY: 53/53 weeks. OBS-END: 48/53 weeks; 3 DST-gap weeks fall Mon 04:00-Tue 03:00, 1 week Mon 05:00-Tue 03:00, and the Jan-1 week has no origins. The calendar-independent statement is "the first (B−A)−72 hours after A". | Present the slot as calendar-dependent. The owner may keep h=72 as the representative if he declares the confounding with day of week. |
| 07c: "consecutive targets overlap 71 of 72 h" | CONFIRMED | METHODOLOGICAL_ALTERNATIVE | Holds for hourly origins under any calendar. | Report an effective sample size next to the row counts. |
| 07d: "four 2023 weeks give 96 E / 87 M" | QUALIFIED | RISK | The weeks are illustrative; the plan does not name them (PROP:103-105). Results are 96/87 under NY and OBS-END, 144/131 with f = e = 6, and 48/42 with f = e = 18. | Name the screening weeks in the plan, then generate their counts. |
| 07e: "h=72 VALIDATION 1,272 eligible vs 3,712 in the old run" | QUALIFIED | OWNER_REQUIREMENT (PROP:195, no reuse of old figures) | The original compared E with scored rows. Comparing like with like: M = 1,234 (NY) or 1,222 (OBS-END) against 3,712 old rows. The 3,712 was recomputed with `i6m_data.partitions`, and its rows_sha256 matches DESIGN.json. Also, 2,478 (NY) or 2,490 (OBS-END) of the 3,712 old rows are not WMC-eligible. | State M against M, and that the old rows are not a subset baseline. |
| 07f: "holiday-aware first 2024 week has zero h=72 support" | QUALIFIED: was derived, now executed | RISK (depends on holiday rule), OWNER_REQUIREMENT for the holiday calendar (MWC:53-54) | OBS-END: the week reopens Mon 2024-01-01 17:00 EST, B−A = 72, h=72 E = 0, h=60 E = 12. NY with no holidays: h=72 E = 24, M = 11. | The versioned calendar must fix holiday reopen rules. The support table then decides this week. |
| 08a: "bar timestamp convention not fixed" | QUALIFIED. The leak scenario is RETRACTED for the concrete route. | RISK (plan text silent) | The I6-E route fixes END labels: `tools/i6e_price_metrics.py`:3-4 and 18-28 take a NY-local BAR_START, floor it to the hour and add 1 h; `tools/fs4_weekly_wrapper.py`:271 says "features available at the bar end". PROP:36 and the MWC do not restate this. The open-label leak and the "B−1 closes at B" problems do not arise under the implemented END convention. | Declare END labels in the calendar artifact and test it, so a new executor cannot change the convention silently. |
| 08b: "DST-start week 2024-03-10 opens with a bar labelled at the open itself" | CONFIRMED AND WIDENED | DEMONSTRATED_DEFECT (data clock inconsistency); the cause is a RISK | See `bar_label_check.out` §1 and the detail after this table. | Before N0, resolve the clock against the raw source. Add a calendar test that flags weeks whose first or last bar deviates from O/C. Record the affected weeks in the identity. |
| 10a: "boundary rule missing: MWC:77 does not say whether origin or target decides" | QUALIFIED; this part is RETRACTED | RISK (residual specification gap) | The subject of MWC:76-78 is q = t+h ("también debe pertenecer al split"), so membership by target is specified. I6-E applies origin-in-segment AND t+h < 2025 (`i6m_data.py`:56-58). Still unspecified: whether the straddling market week counts as VALIDATION, and how its origins dated in 2025 are treated. | One rule in the calendar artifact: t and t+h < 2025-01-01T00:00Z within the same week. Typed exclusions TEST_TIME_ORIGIN and TARGET_IN_TEST, with counts sealed. |
| 10b: "58/94 h=2 rows and all h≥48 rows have 2025 targets" | QUALIFIED; the framing is RETRACTED | — | The arithmetic holds under NY, but 56 of the 58 rows have origins in 2025 (TEST time). Only 2 rows are true straddles (origin in 2024, target in 2025). Split per h is given after this table. OBS-END is identical (HYBRID C); OBS-OPEN is shifted by 1. | Report the three categories separately. |
| 11a: "53 segments / Mon 00:00 UTC cutoff in the I6-N proposal contradict WMC06" | QUALIFIED: AMBIGUOUS, not a contradiction | RISK (only if the I6-N executor reuses `weekly_schedule`) | PROP:40-41 adopts a closure cutoff with weekly refits during closure, as MWC:14-19 requires. PROP:43 "53 segmentos" does not define the segments. 53 is also the number of market weeks with A in 2024 under all 19 calendars. | Define the 53 as market weeks of the versioned calendar. Forbid importing `weekly_schedule` into I6-N. |
| 11b: the I6-E segment definition (concrete object) | CONFIRMED | DEMONSTRATED_DEFECT relative to MWC:14-19 (applies to I6-E/I6-E-M, not yet to I6-N) | Segments come from `tools/i6e_horizon_plan.py`:59-83, used through `build_plan` (:118) and `i6m_prepare.py`:52-62; the I6-E-M DESIGN weeks are identical (53 segments, the last is the 2-day fragment Dec 30-31; I6E:131-136). Numbers are given after this table. | I6-N weeks come from the calendar, with fit_cutoff = C(previous week). A test rejects any cutoff later than O. |
| 11c: holiday closures (Dec 25, Jan 1, Good Friday closed in 2023 and open in 2024 in the data) | CONFIRMED (data facts; `gaps_2024` rerun identical) | OWNER_REQUIREMENT (MWC:53-54, versioned calendar with holidays), pending N0 | The facts come from `ts` and do not depend on a calendar. | Add holiday and early-close rows to the calendar artifact. |
| 11d: "lower bound on force-flat offset unspecified" | QUALIFIED | METHODOLOGICAL_ALTERNATIVE | f is configurable by contract. Sensitivity under NY: h=72 VALIDATION M = 1,852 (f=6), 1,234 (f=12), 616 (f=18); h=18 M = 4,649, 4,025, 3,401. | The owner may set a floor. Otherwise each f gets its own support table. |
| 29: "two-long minimum never binds under default offsets/cut" | CONFIRMED for the default only; QUALIFIED | METHODOLOGICAL_ALTERNATIVE | Under the default, B−cut = 29 h (NY; OBS-END 28-29 h), so the minimum is 2 and 0 entry origins have fewer than 2 longs. It binds in 4 of 9 NY settings (detail after this table). The normal NY week has entry origins by number of valid longs {2:7, 3:12, 4:12, 5:12, 6:24}. | State that the rule binds if and only if B−cut < 24 h. Declare how partial long vectors are used. |
| WMC DST note: "DST weeks do NOT change counts (EXECUTED)" | QUALIFIED | RISK | True under NY by construction, because the US switch falls inside the closure. Under OBS-END, the March 2024 DST-gap weeks have A→cut = 68 (+1 entry origin for h ≤ 24) and the h=72 slot shifts 1 h. | Tie the statement to the calendar. |

### Detail for 08b: clock offset in US/EU DST-gap weeks

**What the data shows.** In 2019-2024, 19 of the 21 US/EU DST-gap weeks have
both session ends labelled 1 h earlier in NY time: the first label is Sunday
17:00 and the last label is Friday 16:00. By contrast:
- in 2012-2018, all 22 gap weeks show the normal Sunday 18:00 / Friday 17:00;
- in 2019-2024, 272 aligned weeks show the normal Sunday 18:00 / Friday 17:00.

**Exact rows.**
- Friday 2024-03-08 22:00 UTC is followed by Sunday 2024-03-10 21:00 UTC
  (= 17:00 EDT; under END labels this bar covers 16:00-17:00 EDT, before the
  NY open).
- Friday 2024-03-22 20:00 UTC (16:00 EDT) is followed by Sunday 2024-03-24
  21:00 UTC.
- The week of 2024-03-31 is normal (Sunday 22:00 UTC).

**Why the label convention does not fix it.** No single choice between OPEN
and END labels repairs these weeks alone; the offset is specific to those
weeks.

**Affected weeks.** In VALIDATION 2024: March 10, 17 and 24 in full, and
October 27 partially (119 bars; ambiguous). In 2023: March 12 and October 29;
March 19 has sparse data.

**Known-input control (cause not proven).** Through the audited converter, a
16:00 source stamp maps to a 21:00 UTC label. This matches a source clock
that follows EU DST dates. That cause is unverified without the raw source,
so it is classed as RISK.

**Effect.** A and B under a fixed NY calendar are misaligned by 1 h in those
weeks. As-of joins are also off by 1 h. Pairs of t and t+h within a week are
unaffected.

### Detail for 10b: straddling-week split per h

NY calendar, management role. Format per h:
origin 2024 & target 2024 / origin 2024 & target 2025 / origin 2025.

| h | 2024/2024 | 2024/2025 (straddle) | origin 2025 |
|---|---|---|---|
| 2 | 36 | 2 | 56 |
| 12 | 26 | 12 | 46 |
| 24 | 14 | 24 | 34 |
| 36 | 2 | 36 | 22 |
| 48 | 0 | 38 | 10 |
| 60 | 0 | 36 | 0 |
| 72 | 0 | 24 | 0 |

### Detail for 11b: I6-E segments against MWC:14-19

| Measure | NY | OBS-END |
|---|---|---|
| Segments whose fit_cutoff falls after the market open O | 53/53 | 52/53 (Jan-1 week excepted) |
| Observed post-open bars visible to the fits | 86 | 89 |

Under NY, the last segment holds 38 of the 94 h=2 management origins of the
straddling week.

I6-E scored rows that are not WMC management-eligible (NY):

| h | Not eligible | I6-E scored rows |
|---|---|---|
| 2 | 1,245 | 6,130 |
| 24 | 1,246 | 4,953 |
| 48 | 1,221 | 3,685 |
| 72 | 2,478 | 3,712 |

### Detail for 29: settings where the two-long minimum binds (NY)

| f | Entry cut | Entry origins with fewer than 2 longs | Entry origins with 0 longs |
|---|---|---|---|
| 6 | Thursday 12:00 | 53 | 0 |
| 12 | Thursday 12:00 | 371 | 53 |
| 18 | Thursday 00:00 | 53 | 0 |
| 18 | Thursday 12:00 | 689 | 371 |

## Corrections to the original audit text

**07.** "VALIDATION 2024 h=72 1,272 eligible vs 3,712" compared E with
scored rows. It should read 1,234 M (NY) against 3,712. The zero-support
claim for the first week was derived and had not been executed; it is now
executed and holds under OBS-END only. "All h=72 origins Mon 05:00-Tue
04:00" holds under NY only.

**08.** The convention is fixed in code (END labels). The leak scenario
applies only to OPEN labels, which the concrete route does not use. The
DST-start observation is part of a systematic one-hour offset in the
DST-gap weeks of 2019-2024. It is not a one-off anomaly and not a question of
label convention.

**10.** MWC:76-78 does specify target membership. Of the 58 h=2 rows with
2025 targets, 56 have origins in 2025.

**11.** The I6-N proposal is ambiguous about segments, not contradictory: it
adopts a closure cutoff at PROP:40-41. The demonstrated contradiction is in
the I6-E plan and code.

**29.** "Never binds" holds only for the default B−cut = 29 h.

**WMC note.** WMC_COUNTEREXAMPLES.md calls the scripts "session scratch, not
committed". They are committed under `probes/a`.

## Open items

- The cause of the DST-gap offset needs the raw 5-minute source clock. That
  source was not read in this lane.
- The broker calendar is still not verified.
- The week of 2024-10-27 (119 bars) is ambiguous between a shifted week
  missing its first bar and an unshifted week missing its last bar.
