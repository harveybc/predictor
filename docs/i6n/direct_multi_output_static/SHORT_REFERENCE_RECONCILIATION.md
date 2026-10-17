# Short Result Claims And Reference Identity

This is a read-only reconciliation, not another fit, feature-selection change,
new stopping rule or TEST evaluation. Independent recomputation from retained
predicted/origin/target prices and masks verifies the original receipt hashes
and compares every prediction array with the reference-device reproduction.
Exact machine-readable results are in
[SHORT_REFERENCE_SCORE_AUDIT.json](SHORT_REFERENCE_SCORE_AUDIT.json).

## Three Different Claims

1. **MG2 original, six short outputs:** two of six DEVELOPMENT-2024 heads beat
   their naive: 6h and 12h. All six FIT heads beat their FIT naive. INNER has
   no winning head. FIT, INNER and DEVELOPMENT are different score populations;
   six FIT wins are not six DEVELOPMENT wins.
2. **MG2m matched short control:** three of six DEVELOPMENT-2024 heads beat
   their naive: 4h, 6h and 12h. It is a different retained population/config
   identity, not an alternative name for MG2. See
   [the original matched audit](matched_short/MUSASHI_AUDIT_20261015.md).
3. **Downsampled versus hourly matched control:** MG2m improves all six heads
   relative to MG1. This is six wins over the hourly learned model, NOT six
   wins over persistence. The retained owner-facing report dated 2026-10-15
   22:59 UTC stated both this comparison and the three naive-winning heads.

The earlier report dated 2026-10-15 08:20 UTC likewise identified only 6h
and 12h as naive-winning in original MG2. The reminder dated 2026-10-16
23:52 UTC selected that original model's 6h/12h values and explicitly
distinguished the later matched control. These messages are not evidence
of an all-six annual DEVELOPMENT winner. The reconciliation does not claim
that no other model or shorter period could ever have all-six wins.

## Original MG2 Recalculation And Exact Reproduction

Percentages are `(MAE_h-naive_h)/naive_h * 100`, original price units,
each horizon's own observed origins and targets. TRAIN is named FIT below.

| h | Original FIT | Original INNER | Original DEVELOPMENT 2024 | Reproduced DEVELOPMENT 2024 |
|---|---:|---:|---:|---:|
| 2 | -0.034661% | +0.007556% | +0.025657% | +0.025657% |
| 3 | -0.073001% | +0.022407% | +0.056818% | +0.056818% |
| 4 | -0.121163% | +0.102434% | +0.030727% | +0.030727% |
| 5 | -0.052741% | +0.000078% | +0.051220% | +0.051220% |
| 6 | -0.128724% | +0.202301% | -0.049615% | -0.049615% |
| 12 | -0.236774% | +0.172419% | -0.093170% | -0.093170% |

Reference: I6-DMO-SHORT-MG2-MARKET-EXACT-R0, effective config 920332ae,
source bundle 471ea311, seed6102026, best3 of13 epochs. Source receipt SHA256:
`0f32247f31e01bb1d1c86ac57c44033c3cb1ed8eb996e4572974ee95cd47ea20`.
The original-device reproduction is i6dmo-short-reference-device-20261016,
runtime a2c7fe6d. All 13 epoch weight digests and monitors, selected weights
and all stored FIT/INNER/DEVELOPMENT arrays match exactly. There is no
remaining observed failure to reproduce THIS original experiment.

The 4090 replay remains a separately retained different-device trajectory;
it is not the reference-device reproduction. The latter result cannot be
described as a failed replication merely because the desired claim is six
validation winners rather than the original two. Equally, exact replication
does not establish profitability or independent holdout performance.

## Reporting Correction

Stop using an unqualified "the good short model" as a reference selector.
Name experiment, config/population identity, split/year and comparator.
For this reproduction the reference above is fixed; the matched MG2m result
is kept separate. Any claim of six naive-winning annual DEVELOPMENT heads
requires its own retained result and reference identity before any new fit.
No new fit should be used to search for numbers that a historical artifact
did not contain. Seeds, parameters and saved weights remain retained;
original-device training and saved-weight inference both reproduce them.

This audit found neither changed prediction arrays nor a MAE/naive sign error.
It found no evidence in the cited retained reports that FIT was labeled as
annual DEVELOPMENT. The observed reporting risk is ambiguity of reference
and comparator; the table and machine-readable identities above remove it.
The existing automatic main campaign was not stopped or altered.
