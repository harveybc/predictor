# Twelve Versus Eighteen Hours: Retained-Prediction Audit

Owner requested investigation of the short/long discontinuity before further
cadence work. This bounded CPU analysis reads authenticated, closed campaign
artifacts only. No model import, fit, TEST, source modification or new winner.

## Requirements And Test Design

| ID | Required behavior and test |
|---|---|
| D01 | Verify fit terminal against PLAN identity and all retained artifact digests; reject mutations before loading arrays |
| D02 | Validate array shapes, unique increasing UTC origin IDs, horizon identity, boolean masks and finite supported prices |
| D03 | Replay stored per-head MAE and naive on identical supported rows and original price units |
| D04 | Report FIT, INNER and DEVELOPMENT separately; constant predictions give undefined correlation, never a fabricated zero |
| D05 | Join 12/18 h by UTC origin, not array position; reject inconsistent origin prices; keep each horizon's own targets and naive |
| D06 | Empty support is typed; report movement amplitude, bias, direction, correlation and near-naive fractions with explicit thresholds |
| D07 | Emit repeatable small JSON and governed OLAP reports; delivery retry never fits or alters the closed campaigns |
| D08 | Independently reconstruct expected open-market-hour ranks, exact raw source inputs/targets, management/entry masks and FIT scaler row union for every split in both families; reject future/origin/target/business mutants |

The diagnostic uses mean absolute predicted movement / own naive MAE, predicted
and actual movement standard deviations, their correlation, residual bias,
directional agreement (nonzero actual moves), and the fraction within 1% of
the split/head naive MAE of the persistence forecast. That 1% is a descriptive
threshold, not an eligibility rule or a tuned trading threshold. FIT and INNER
are diagnostics, not independent holdouts. DEVELOPMENT remains development.

Common-origin results remove origin-population differences, NOT the horizon,
input-grid, context, head-objective or fitted-weight differences. No causal
attribution to any of those factors follows from this audit. Report actual
configuration differences and preserve all original results.

The owner prioritized causal and business verification before any new fit.
Subsequent clarification limits the new work to training/transformation checks,
not reopening feature selection. First propose the small short-input/long-target
R0 bridge in TRAINING_AUDIT.md; the broader three-grid design below is later.
The configured offsets are 12 h after open and 12 h before close, not 8 h.
Historical observed bars in those buffers may be inputs; closed-market hours
are never counted as observations, interpolated or filled. Entry cutoff and
management masks remain distinct, under the retained assumed calendar.

If the discontinuity persists, define a separately sealed bridge contrast at
18 h: 24 open hours / 2 h spacing; 144 open hours / 2 h spacing; 144 open
hours / 12 h spacing. The middle recipe separates history length from input
resolution. Preserve common support, initialization, labels, scaling rules,
head objective and early stopping. Flatten changes readout parameter count
with window length, which must be disclosed or controlled; these are not
already runnable seals. No broad capacity search or cadence experiment yet.

Array timestamp checks do not establish upstream feature publication times,
vendor revisions or producer prefix invariance. Explicitly retain the seven
unverified source features, conservative-clock assumptions, retrospective
selection and closed TEST gate. A negative audit finding is not waived by an
otherwise successful metric replay.
