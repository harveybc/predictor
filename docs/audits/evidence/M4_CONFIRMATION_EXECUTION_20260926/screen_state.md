# M4 CONFIRMATION — what each of the 21 eligible slots did

| slot (cell) | contrast it feeds | units planned | attempted | complete | failed | calibration SD | SD UCB95 | precision |
|---|---|---|---|---|---|---|---|---|
| `am::clean::w16` | `intervention_effect::am::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `am::clean::w64` | `intervention_effect::am::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `am::white::w16` | `intervention_effect::am::white` | 144 | 0 | 0 | 0 | 0.973729 | 1.400072 | supported |
| `am::white::w64` | `intervention_effect::am::white` | 144 | 0 | 0 | 0 | 0.666667 | 0.958563 | supported |
| `chirp::clean::w16` | `intervention_effect::chirp::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `chirp::clean::w64` | `intervention_effect::chirp::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `chirp::white::w16` | `intervention_effect::chirp::white` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `chirp::white::w64` | `intervention_effect::chirp::white` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `dnf3::clean::w16` | `intervention_effect::dnf3::clean` | 144 | 0 | 0 | 0 | 3.079201 | 4.427415 | supported |
| `dnf3::clean::w64` | `intervention_effect::dnf3::clean` | 144 | 0 | 0 | 0 | 3.297586 | 4.741418 | supported |
| `identity::clean::w16` | `intervention_effect::identity::clean` | 144 | 0 | 0 | 0 | 2.523959 | 3.629063 | supported |
| `identity::clean::w64` | `intervention_effect::identity::clean` | 144 | 0 | 0 | 0 | 2.360163 | 3.393549 | supported |
| `majority::clean::w16` | `intervention_effect::majority::clean` | 144 | 0 | 0 | 0 | 2.328567 | 3.348118 | supported |
| `majority::clean::w64` | `intervention_effect::majority::clean` | 144 | 0 | 0 | 0 | 2.916111 | 4.192916 | supported |
| `sine::clean::w16` | `intervention_effect::sine::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `sine::clean::w64` | `intervention_effect::sine::clean` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `sine::white::w16` | `intervention_effect::sine::white` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `sine::white::w64` | `intervention_effect::sine::white` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `state_space::clean::w16` | `intervention_effect::state_space::clean` | 144 | 0 | 0 | 0 | 0.973729 | 1.400072 | supported |
| `state_space::white::w16` | `intervention_effect::state_space::white` | 144 | 0 | 0 | 0 | 0.0 | 0.0 | supported |
| `state_space::white::w64` | `intervention_effect::state_space::white` | 144 | 0 | 0 | 0 | 0.973729 | 1.400072 | supported |

**Every slot did nothing.** 21 x 48 x 3 = 3024 units planned, 0 attempted. The SD columns are CALIBRATION dispersion — the evidence that made the slot eligible — not a confirmation result.

## The 7 typed-ineligible slots, never constructed

| slot | typed status |
|---|---|
| `discontinuity::clean::w16` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `discontinuity::clean::w64` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `discontinuity::white::w16` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `discontinuity::white::w64` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `parity4::clean::w16` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `parity4::clean::w64` | `INELIGIBLE_UNDER_PROPOSED_RULE` |
| `state_space::clean::w64` | `INELIGIBLE_UNDER_PROPOSED_RULE` |

## The 16-slot Holm family and which contrasts survived

| # | slot | frozen disposition | eligible widths | p raw | p Holm | survived Holm |
|---|---|---|---|---|---|---|
| 1 | `intervention_effect::identity::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 2 | `intervention_effect::majority::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 3 | `intervention_effect::dnf3::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 4 | `intervention_effect::parity4::clean` | NOT_EVALUABLE_p1 | — | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 5 | `intervention_effect::sine::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 6 | `intervention_effect::sine::white` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 7 | `intervention_effect::chirp::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 8 | `intervention_effect::chirp::white` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 9 | `intervention_effect::am::clean` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 10 | `intervention_effect::am::white` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 11 | `intervention_effect::discontinuity::clean` | NOT_EVALUABLE_p1 | — | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 12 | `intervention_effect::discontinuity::white` | NOT_EVALUABLE_p1 | — | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 13 | `intervention_effect::state_space::clean` | SINGLE_WIDTH_USED_AND_NAMED | w16 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 14 | `intervention_effect::state_space::white` | TWO_WIDTHS_EQUAL_AVERAGE | w16, w64 | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 15 | `checkpoint_effect::primary_pair` | CHECKPOINT_PAIR | — | — | — | **UNDETERMINED_NO_OBSERVATION** |
| 16 | `incremental_prediction::M2_vs_M1` | NON_REJECTING_PLACEHOLDER_p1 | — | — | — | **UNDETERMINED_NO_OBSERVATION** |

**No contrast survived Holm and none failed it.** Holm is a step-down over 16 p-values; zero exist. Topology re-derived: 10 two-width, 1 single-width, 3 NOT_EVALUABLE, 1 checkpoint pair, 1 M2 placeholder; 10*2 + 1 = 21.

## Units that failed, and why

- planned: **3024**
- attempted: **0**
- failed in fitting: **0** (nothing reached a fit, so no unit failed numerically)
- not attempted: **3024**, for three typed reasons:

  - GATE_1_DESIGN_REVIEW_RECORD_ABSENT — the frozen protocol admits only a record whose role token is EXTERNAL_AUDITOR and whose decision token approves execution. Authoring it would have been the executing agent approving its own execution; the host permission system refused that on 2026-09-26 and the refusal was not worked around.
  - GATE_2_OWNER_EXECUTION_RECORD_ABSENT — it chains by digest to the record above, so it cannot exist first.
  - BLOCKER_3_NO_EXECUTION_BODY — independently of the gates, the frozen tools cannot fit a CONFIRMATION unit: the sealed generator bank refuses CONFIRMATION construction unless allow_confirmation is passed, the sealed unit runner never passes it, and execute_confirmation returns right after the pre-result ledger. Both records installed would yield a 3024-unit PENDING ledger and zero fitted units.

## Closure table

| quantity | stage | model error + scale | naive reference, same rows | skill | literature | comparability |
|---|---|---|---|---|---|---|
| M4 CONFIRMATION screen — primary estimand (generator-level paired restricted-endpoint difference, calibration_stop minus matched initialization) | CONFIRMATION | NO_NEW_MEASUREMENT | NO_NEW_MEASUREMENT — the naive reference IS the matched initialization arm of the same generator, tape and compute; neither arm was fitted | NO_NEW_MEASUREMENT | NOT_CARRIED | NOT_COMPARABLE |
| M4 prediction ladder M1 vs M0 — out-of-generator prediction of log1p(restricted endpoint), integrated Brier | CALIBRATION | 0.00993885 integrated Brier (unitless, 0 is perfect), n_groups 224 | M0 = 0.00326826 integrated Brier on the SAME 224 unseen-generator groups (parameter count alone) | -2.041022 | NOT_CARRIED | NOT_COMPARABLE |
| M4 prediction ladder M2 vs M1 — out-of-generator prediction of log1p(restricted endpoint), integrated Brier | CALIBRATION | 0.42976772 integrated Brier (unitless, 0 is perfect), n_groups 224 | M1 = 0.00993885 integrated Brier on the SAME 224 unseen-generator groups (M0 plus checkpoint loss and elapsed updates) | -42.241192 | NOT_CARRIED | NOT_COMPARABLE |
| M4 prediction ladder M2 vs M0 — out-of-generator prediction of log1p(restricted endpoint), integrated Brier | CALIBRATION | 0.42976772 integrated Brier (unitless, 0 is perfect), n_groups 224 | M0 = 0.00326826 integrated Brier on the SAME 224 unseen-generator groups (parameter count alone) | -130.497408 | NOT_CARRIED | NOT_COMPARABLE |

Comparability reasons, per row, in the order above:

1. there is no measurement to compare: zero of 3024 units were fitted. This row is NO_NEW_MEASUREMENT by the standing closure rule, not an unfavourable result.
2. synthetic generator bank, internal endpoint, no external benchmark; and it is a CALIBRATION measurement, so it can never stand in for the confirmation row above
3. synthetic generator bank, internal endpoint, no external benchmark; and it is a CALIBRATION measurement, so it can never stand in for the confirmation row above
4. synthetic generator bank, internal endpoint, no external benchmark; and it is a CALIBRATION measurement, so it can never stand in for the confirmation row above

M2 minus M1 paired gain -0.41982887 (t -20.4579, n_groups 224) is why M2 is carried as a non-rejecting placeholder rather than fitted. The table also shows M1 itself has NEGATIVE skill against M0: on these rows the parameter count alone predicts the endpoint better than the measurement-enriched models.

Generated by `publish_screen_state.py` from `docs/audits/evidence/M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_2026_09_09.json` (sha256 `51247b7853f7da1d5e549810f6b680183e7f4a4c77a4c787188030b6e560ff4c`) and `docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json` (sha256 `0b257a98efca1a946a1c7ca378d8977036a758290d6c2111301bd2574bddf6c2`) at agent-multi@0e99ad1a. No value in this file was typed by hand.
