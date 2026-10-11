# WMC01-WMC10 counterexamples (sub-auditor A/B)

Date: 2026-10-11. Role: sub-auditor A/B (business/calendar and experiment
order) for the I6-N strict independent plan audit. Read-only on the audited
object. No training, GPU, service or TEST read.

## Object audited (hashes verified, all eight match the request)

Worktree `.worktrees/predictor-i6d-weekly`, branch
`codex/i6d-weekly-walk-forward-20261009`, HEAD `269c2d13` plus UNCOMMITTED
draft. Paths below relative to `docs/tres_temas_entrevista/` unless absolute
to the repo.

| File | SHA-256 (verified) |
| --- | --- |
| program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md (PROP) | d4341e72...6eb56487 |
| MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md (MASTER) | c0fc85c3...792a4e59 |
| program_v3/PROJECT_METHOD_STATE.json | ee8a23f2...dd60ec4 |
| program_v3/EXPERIMENT_EXECUTION_QUEUE.json | c9f1559f...6ab672ef |
| program_v3/CURRENT_EXECUTION.md | bbbb09cc...4c133c7b |
| program_v3/I6E_MULTI_HORIZON_RESOLUTION_WORK_PLAN.md (I6E) | 129cea8c...0fee1234 |
| program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md (MWC) | 604f9480...00c01c76 |
| program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md (BWC) | d64206b8...67b5a4c1 |

## Calendar convention used for executed checks (ASSUMPTION)

This is NOT the broker calendar (MWC:41 leaves `market_calendar_id` as a
required placeholder; no versioned calendar exists in the object). For the
checks only: O = Sunday 17:00 America/New_York, C = Friday 17:00
America/New_York (OANDA-style NY Forex convention, MWC:242-246), A = O+12 h,
B = C-12 h, entry cut = Thursday 00:00 America/New_York, elapsed hours via UTC.
Bars are labelled by their END time in UTC (inferred, see WMC01/WMC02).

Executed scripts and raw outputs (session scratch, not committed):
`scratchpad/i6n_a/{bar_clock,gaps_2024,wmc_calendar,strategy_counterexample}.py`
and `.out`. Data read: only the `ts` array of the retained I6-E input
`~/.local/state/canonical_20261003/i6e/statistical72_v1/INPUT.npz`
(sha256 897caee1...; last row 2024-12-31 22:00 UTC; the loader asserts no row
>= 2025-01-01). All scripts ran under `crispdm-run -q -m 1G`.

### Executed support table (normal week, identical for DST weeks)

Weeks 2024-06-09 (normal), 2024-03-10 (US DST start), 2024-11-03 (US DST
end), 2024-03-24 (EU/US DST gap), 2024-11-24 (Thanksgiving): B-A = 96 h,
A->cut = 67 h. E = calendar eligibility; M = E with exact observed t and t+h.

| h | management E | entry E | h | management E | entry E |
| --- | --- | --- | --- | --- | --- |
| 2 | 94 | 67 | 18 | 78 | 67 |
| 3 | 93 | 67 | 24 | 72 | 67 |
| 4 | 92 | 67 | 36 | 60 | 60 |
| 5 | 91 | 67 | 48 | 48 | 48 |
| 6 | 90 | 67 | 60 | 36 | 36 |
| 12 | 84 | 67 | 72 | 24 | 24 |

Management = 96-h. Entry = min(67, 96-h). h >= 96 gives 0 in every week.
Under the NY convention a US DST switch (Sunday 02:00 local) never falls
inside an operational week, so DST weeks do NOT change counts (EXECUTED).

VALIDATION 2024, 53 market weeks with A in 2024 (EXECUTED):

| h | mgmt E | mgmt M | entry E | entry M | mgmt E targets >= 2025-01-01 | weeks with mgmt M = 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4982 | 4885 | 3551 | 3481 | 58 | 0 |
| 12 | 4452 | 4345 | 3551 | 3461 | 58 | 0 |
| 18 | 4134 | 4025 | 3551 | 3453 | 58 | 0 |
| 24 | 3816 | 3707 | 3551 | 3447 | 58 | 0 |
| 36 | 3180 | 3073 | 3180 | 3073 | 58 | 0 |
| 48 | 2544 | 2464 | 2544 | 2464 | 48 | 1 |
| 60 | 1908 | 1856 | 1908 | 1856 | 36 | 1 |
| 72 | 1272 | 1234 | 1272 | 1234 | 24 | 1 |

h=72 scoring population falls from the I6-E compact 3,712 common origins
(CURRENT_EXECUTION.md:55-57) to 1,234: previous figures are not comparable.
All 24 h=72 origins per week lie in Monday 05:00 - Tuesday 04:00 NY with
targets Thursday 05:00 - Friday 04:00 NY: h=72 is a single day-of-week slice.
Four illustrative quarterly 2023 weeks (not named by the plan) give 96 E /
87 M h=72 rows, consecutive targets overlapping 71 of 72 hours.

Long availability over the 67 entry origins of a normal week (EXECUTED):
6 longs at 24, 5 at 12, 4 at 12, 3 at 12, 2 at 7. Minimum is 2, so the
proposed "at least two longs" abstention (MWC:150-152) never binds with the
default offsets and Thursday cut.

## WMC table

Status vocabulary: ESPECIFICADO / AMBIGUO / NO_ESPECIFICADO / NO_IMPLEMENTADO /
CONTRADICTION. EXECUTED = run here; PROPOSED = acceptance test to implement.

### WMC01 - O/C, DST/holidays and source gaps are not confused

- Counterexample (EXECUTED, `gaps_2024.out`): the source clock is not a NY
  calendar. 2023-04-07 Good Friday is a 70 h closure (Fri 00:00 -> Sun 22:00
  UTC) while Good Friday 2024 trades normally; 2023-12-29 22:00 -> 2024-01-01
  23:00 UTC (73 h, New Year Monday closed); 2024-12-25 08:00 -> 23:00 UTC
  (15 h mid-week); many Friday gaps of 51-52 h where the last 1-3 bars before
  the close are missing (e.g. 2023-03-03, 2023-06-16, 2024-11-01). Deriving C
  from the last bar would move B by 1-3 h and flip the h boundary.
- Holiday-aware O changes support: if the first 2024 week reopens Monday
  2024-01-01 17:00 NY (as the data shows), B-A = 72 h and h=72 has ZERO
  E support; with a Sunday O it has 24 E / 11 M rows (EXECUTED for the Sunday
  variant; holiday variant derived). Population depends on calendar version.
- Expected: O/C from a versioned instrument calendar; gaps yield M=false with
  reason SOURCE_GAP and never move A/B.
- Status: principle ESPECIFICADO (MWC:53-57); calendar artifact
  NO_IMPLEMENTADO (MWC:41 placeholder); mid-week holiday closures (positions
  held across Dec 25/Jan 1? does the closure define its own A/B?)
  NO_ESPECIFICADO; Good Friday handling NO_ESPECIFICADO.
- Acceptance (PROPOSED): calendar fixture tests: DST weeks give B-A=96
  (EXECUTED: 4 weeks); deleting the last Friday bar does not move B; a
  holiday-shifted reopen changes A, the support table and the identity digest.

### WMC02 - exact A/B, Thursday entry-only, same-week targets, R=73/72/48

- EXECUTED: E(t,72) holds exactly for t in [A, B-72), i.e. R(t) >= 73;
  R=72 and R=48 fail; shorter heads stay valid at those origins.
- Counterexample: the bar label convention is not fixed by PROP/MWC. Data is
  consistent with END labels (`bar_clock.out`: last bar Fri 22:00 UTC winter /
  21:00 summer = 17:00 NY; first bar Sun 23:00/22:00 UTC = 18:00 NY), but the
  DST-start week 2024-03-10 starts with a bar labelled Sun 21:00 UTC = 17:00
  EDT = O, impossible for an END label. With OPEN labels, "window includes t"
  reads the close at t+1 h (one-hour look-ahead), and a target labelled B-1
  closes at B, violating the exclusion of equality (MWC:131-133).
- Status: E/M formulas ESPECIFICADO (MWC:69-94); bar-label convention
  AMBIGUO (PROP:36 "ventana incluye t" only; FS4 freeze states "available at
  the bar end", not repeated in PROP/MWC).
- Acceptance (PROPOSED): property test over every hourly t of a fixture
  week: E(t,72) iff B-t >= 73 h; deleting bar t+h leaves E unchanged and sets
  M false; a bar whose end label is <= O is rejected; a one-hour label shift
  changes the digest.

### WMC03 - 120/144 h without support are NO_ELIGIBLE_ROWS

- EXECUTED: h >= 96 has E=0 every week; last 2024 week has mgmt M=0 for
  h=48/60/72 if targets must be < 2025-01-01; first 2024 week h=72 is 0 under
  a holiday-aware calendar.
- Status: ESPECIFICADO (MWC:98-99, 115-117, 190-191); disposition name
  differs from I6E:146 `NOT_APPLICABLE` (MINOR naming).
- Acceptance (PROPOSED): zero-support (h, week, role) emits a typed
  NO_ELIGIBLE_ROWS disposition, no MAE/skill row, annual aggregate excludes it
  with an explicit count; N4 budget counts it as a disposition, not a fit.

### WMC04 - non-trading hours present if known at t; future absent

- EXECUTED (`wmc_calendar.out`, 1,258 observed 2024 h=72 management origins):
  N1 anchor L=144, delta=36, n=5: sample t-36 is ALWAYS Sat 17:00 - Sun 16:00
  NY, so at most 4 of 5 samples exist (29.4 % missing, min 2 present). L=432
  hourly: 33.7 % of 433 slots missing (250-291 present). L=432, delta=8:
  33.2 % of 55 missing (32-38 present). Every origin has a missing slot.
- Expected: absent slots carry an explicit mask; observed bars in [O,A) and
  [B,C) stay as inputs (MWC:61-67).
- Status: principle ESPECIFICADO (MWC:61-67); representation of absent slots
  for the new families (Conv1D-only, LSTM adapter, PROP:83-101, 125-126)
  NO_ESPECIFICADO. The old I6-E code zero-fills with a presence channel and
  attention/gate masks (tools/i6e_clock_windows.py:41-48,
  tools/i6m_model.py:19-46); only the Transformer route has an attention mask,
  a Conv1D kernel still reads zeros, and an LSTM would integrate them.
- Acceptance (PROPOSED): for every route, perturbing a placeholder value
  leaves the output bit-identical; perturbing an observed bar in [B,C) before
  t changes it; perturbing anything after t never changes it.

### WMC05 - masks identical across splits/naive; per-head; NaN before residuals; S_h/U_h

- Counterexample: S_h = M AND "prediction verifiable" (MWC:182-183) is
  config-dependent: configs failing on different rows are scored on different
  sets, so MAE is unpaired. PROP:68-69 seals common support only "for the
  largest candidate history". No intersection/coverage rule.
- Counterexample: N4-M long head at t with R=71 h: heads 18..60 valid, 72
  invalid; the old code builds masks with `np.isfinite(y)` and
  `ts+h < inner/cut` (tools/i6m_data.py:51-58), no market-week mask.
- Status: concepts ESPECIFICADO (MWC:96-106, 182-207); common-row rule
  AMBIGUO; which of S_h/U_h is the fitness NO_ESPECIFICADO (PROP:139); whole
  mechanism NO_IMPLEMENTADO.
- Acceptance (PROPOSED): two synthetic configs with disjoint failures are
  scored on the intersection with coverage reported, or the incomplete one is
  typed INCOMPLETE; masked multi-output loss yields finite loss and zero
  gradient for an invalid head while shared weights still update; per-head
  denominators equal per-head M counts.

### WMC06 - release before reopening; no Monday cutoff; four years per week

- Counterexample (EXECUTED from artifacts): the retained I6-E designs use
  `cutoff` = Monday 00:00 UTC (e.g. 2024-01-08T00:00Z in
  `i6e/statistical72_v1/DESIGN.json` weeks), after the Sunday 21:00/22:00 UTC
  reopen; I6E:136 prescribes "cutoff del lunes anterior" for the first 2025
  segment. MWC:14-19 requires cutoff = C and fit between C and O.
- Status: MWC ESPECIFICADO; PROP:43 inherits "53 segmentos" from the
  Monday-UTC I6-E calendar (I6E:131-136) without redefining them as market
  weeks -> CONTRADICTION.
- Acceptance (PROPOSED): every week has fit_cutoff == C(previous) and
  release < O; a Monday-00:00-UTC cutoff fixture is rejected; perturbing bytes
  after C leaves data, scaler and weights unchanged (BW02).

### WMC07 - no entries after cut, pending cancelled, flat at B, failed fills typed

- Counterexample: the effective plugin (heuristic-strategy
  app/plugins/plugin_long_short_predictions.py, blob 3a19b3f) has no calendar.
  Entry checks only position, `max_trades_per_5days` and predictions
  (plugin:340-391); positions close only at TP/SL/early close or `stop()`
  (plugin:491-493). TP/SL are checked at bar close (plugin:293-298).
- Status: MWC:159-165 ESPECIFICADO; plugin NO_IMPLEMENTADO.
- Acceptance (PROPOSED): signal at cut+1 h opens nothing; a position open at
  B-2 h receives a close decision whose next-open fill is < B; a missing next
  bar yields FILL_FAILED, never an ideal close price.

### WMC08 - heuristic and SAC/DQN share constraints, continuous equity

- Plugin swap accrues `duration/24` days without a weekend rule
  (plugin:453-458). The RL environments were not audited here: NOT_VERIFIED.
- Status: ESPECIFICADO (MWC:167-171, BWC:89-95); NO_IMPLEMENTADO in the
  plugin.
- Acceptance (PROPOSED): heuristic adapter and RL env consume the same
  calendar digest; Monday-open equity equals Friday-flat equity minus declared
  costs; positions/pending = 0 on every [B, A_next).

### WMC09 - LITERATURE_STATIC separate; BUSINESS refuses disabling

- Counterexample: the proposed JSON exposes `weekend_positions_allowed` and
  `force_flat_before_close_hours` (MWC:44-45). `force_flat=0` gives B=C: the
  decision at C-1 h fills at C, i.e. not flat before the close. No lower bound.
- Status: mode separation ESPECIFICADO (MWC:30-33, BWC:52-57, BW14); offset
  bounds AMBIGUO.
- Acceptance (PROPOSED): schema rejects BUSINESS with
  `weekend_positions_allowed != false` or a force-flat offset under one bar
  plus execution latency; LITERATURE_STATIC rejects business labels.

### WMC10 - cache/OLAP separate the old protocol from the successor

- Counterexample: the I6-E-M design identity (tools/i6m_prepare.py:94-102)
  has no calendar id, offsets, cut or target policy. A reuse lookup by
  (arm, h, week, seed) would adopt the 266 published old terminals.
- Status: ESPECIFICADO (MWC:209-219); NO_IMPLEMENTADO.
- Acceptance (PROPOSED): identity digest includes calendar id/version,
  offsets, cut, target policy and bar-label convention; adoption of an old
  terminal fails with PROTOCOL_MISMATCH; OLAP rows carry a protocol column and
  never union the two populations.

## Additional boundary counterexample (request 3.2 / 4)

The last 2024 market week (O Sun 2024-12-29) has B = Fri 2025-01-03 05:00 NY.
EXECUTED: 58 of 94 h=2 management E rows and all h >= 48 rows have targets in
2025, which is TEST. Old code truncates at `ts+h < 2025-01-01`
(tools/i6m_data.py:58), so h=72 has zero rows in that week. MWC:77 requires
the target to "belong to the split" without saying whether split membership
follows origin or target. Required rule: target-time split membership,
NO_ELIGIBLE_ROWS for the remainder, and no TEST label read for VALIDATION.
