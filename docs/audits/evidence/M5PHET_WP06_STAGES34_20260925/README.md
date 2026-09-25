# WP06 stages 3-4: the four design candidates fitted, scored and ranked (2026-09-25)

The design job (WP06 stage 2) emitted four candidate representations for the household series. This round fits one
model per candidate, scores every fit on the **same sealed holdout** the earlier stages were scored on, and rebuilds
the closure table over all of them.

## What was held fixed, and why

Every candidate spec (`specs/`) differs from `baseline_hand` in the **representation and nothing else**: the same single
group holding all seven meter columns, the same `tcn` encoder family of `fused_branches`, the same core, no per-feature
preprocessor declared, and the same fitting parameters as the earlier stages — 200 epochs, patience 15, seed 1, batch
256, `enable_op_determinism`, one horizon of 60 steps. A candidate fitted with another grouping or another encoder
would differ from the baseline in three things at once and the table could attribute nothing to the representation.

The window fitted is `max(representation.windows)` — the longest memory the candidate declares. Taking the first, as
the harness did before, would have fitted `short_memory`, `seasonal_lag_1443` and `seasonal_lag_2892` as the *same*
graph (all three list 197 first) and published three candidates as one measurement. Every declared window is written
into the run manifest beside the one used (`windows_declared`, `window_selected_by`).

## The holdout: recomputed, not reused

`population_probe.json` recomputes the population from the same rule (last 20 % of the file by time; origins whose
whole sealing window of 197 rows and whole horizon of 60 rows lie inside it) and gets **9824 origins, protocol
`d0ebd9a4bc75`, seal `33820b552ddf`** — the seal of `baseline_hand`. Both fitted candidates carry that seal in their
own report; the table says so in its conditions block.

The same probe answers what a longer sealing window would do: 1443 leaves **8578** origins under seal `a8c7f07f6d6c`,
2892 leaves **7129** under seal `97155a063e25`. Those are different populations, so the two long candidates cannot be
scored here at all — and they were **not** scored on other rows.

## What was refused, by name

| candidate | attempted | refusal |
|---|---|---|
| `seasonal_lag_1443` | yes (`stages/refusal_seasonal_lag_1443.txt`) | `FEATURE_NOT_IN_DATA`: it declares the calendar feature `hour_of_day`, which the sealed file does not carry. Building it would produce a different file, and the holdout identity is computed over the file the population was sealed from. Its window 1443 also exceeds the sealing window 197. |
| `seasonal_lag_2892` | yes (`stages/refusal_seasonal_lag_2892.txt`) | its window 2892 exceeds the sealing window 197: the sealed population would not hold its history. |

Both enter the table as `NO_NEW_MEASUREMENT` rows carrying those words (`compare_stages --not-measured`), so
"attempted and refused" is not confusable with "never run".

## The decision records (WP23)

Every stage in the table now carries decision records, or the report says why it cannot. The stages a person
configured — `baseline_hand`, `quantile_hand` and the four candidates — carry `chosen_by: HUMAN` records written by
`m5phet.decide.human_choice` (`human_decisions.py`, index in `human_decisions.json`), with no probabilities, from the
**same option sets** the Laya records of the same (kind, question) were chosen from, copied verbatim from those
records. Two questions could not be answered from the declared options and nothing was invented:

- `feature_grouping/grouping_cut`: these pipelines feed every column to one encoder, and there is no `k=1` cut in the
  declared set (`k=2 .. k=6`) — refused `CHOICE_OUTSIDE_OPTIONS`, 6 times;
- `group_extractor/extractor`: their encoder is `tcn` (an inline family of `fused_branches`) or
  `NOT_APPLICABLE_SINGLE_WINDOW_CORE`, and `feature-extractor` declares no such plugin — refused, 6 times.

`link_outcomes.py` links each stage's records to **its** row of this table: 50 linked, 16 refused `NOT_COMPARABLE`
(the records of the two candidates with no measured row). The outcomes live in
`~/.local/state/m5phet/decision_outcomes-wp06-stage34-20260925`, a directory of their own, because the previous
round's outcomes bind rows of the superseded two-stage table and merging them would count the same decisions twice
under two ranks.

`table/calibration.md` is the report. `NO_BEST_RANKED_OPTION` is gone for the two questions the rank-1 stage can
answer (`feature_preprocessing/preprocessing` → `normalizer`, `representation/candidate` → `hand_household_w60`). For
the other two it now names the reason instead: `quantile_hand_95` is rank 1, `COMPARABLE`, and carries no record for
those questions, because its pipeline has no grouping cut among the declared ones and no feature-extractor plugin —
WP23's `COMPARABLE_BUT_NO_DECISION_RECORD`, printed per stage and per question.

## Caveat on the decision store

`~/.local/state/m5phet/decisions` is shared. While this round ran, another session wrote 8 human records of a
*different* shape into it (`backend: "human"`, where `m5phet.decide.human_choice` on branch
`satoshi/wp06-human-decisions-20260925` writes `backend: null`). The inventory table at the foot of the calibration
report counts those records; nothing else here does — every outcome linked above names the record it came from.

## Files

- `specs/` — the four `m5phet.pipeline.v1` specs, each carrying its stage's decision digests and the refusals
- `stages/candidate_*/` — report, config, fit manifest, training history, digests of the artifacts left in the run dir
- `stages/refusal_*.txt` — the two refusals, verbatim
- `table/table.{json,md}` — the closure table over seven stages
- `table/calibration.{json,md}` — the WP23 calibration report
- `population_probe.json`, `link_outcomes.json`, `human_decisions.json` and the scripts that produced them

## A sixth stage, and why it is here (WP07)

`quantile_hand_95` is the same hand representation and the same `quantile_ann` core as `quantile_hand`, refitted with
five quantiles `[0.025, 0.05, 0.5, 0.95, 0.975]` instead of three. The reason is a capability, not a score: the
acceptance harness asks for a **0.95** interval, the WP07 bundle fitted only the pair covering 0.90, and the question
was refused `CONFIDENCE_LEVEL_NOT_FITTED` — correctly, and unusably. Nothing else about the stage changed, and no
hyper-parameter was touched.

It then came first in the table (MAE 0.526294, skill 0.121858), ahead of `baseline_hand` (0.537108). That was not the
purpose and was not tuned for: the quantile head's median is fitted by the pinball loss over five quantiles, and on
these rows that median happens to be a better point forecast than the point head's. Its measured 0.95 interval covers
**0.925998** of the sealed rows (mean width 2.890335 kW), which is the honest coverage claim and is below its nominal
level.

## What the closure table says about the designed representations

No designed representation beat the hand window on this slice. `seasonal_lag_74` (0.545436) and `short_memory`
(0.578034) are both worse than `baseline_hand` (0.537108), and the two long candidates could not be scored here at
all. The designed-representation stage of the doctoral comparison therefore has a negative result on the household
series, recorded as such.

## Reproducibility

Both fitted candidates were refitted after the fitting tool began recording the representation by value; the two runs
produced identical MAE, RMSE, skill and epoch counts (`stages/*/report.json` before and after are equal in every
metric). The artifacts kept here are the second run's, which are the ones the bundles were exported from.

## The exported bundles (WP06 done-when, part 2)

`bundles/` holds the manifest and the parity record of each bundle exported from these fits:
`candidate-short-memory`, `candidate-seasonal-lag-74` (a new directory,
`~/.local/state/m5phet/forecast-bundles-wp06-candidates-20260925`, nothing overwritten) and `quantile-household-95`.
Each manifest carries the WP06 `representation_spec` **by value**. A bundle exported before the tool recorded it —
the owner's retained household model and the direction model — serves `REPRESENTATION_NOT_RECORDED`, which
`acceptance/FINAL-catalog-forecast.json` shows verbatim beside the two that do record one.

The two candidates that were never fitted have no bundle: there is no graph to export, and none was fabricated.

## The owner's instance (WP07 done-when, part 3)

`quantile-household-95` is installed beside the household bundle in the directory the owner's workbench reads. On port
8777, with my own state directory and all three bundles configured:

- `pronostica la potencia y dame un rango` returns a point (0.541226 kW, from the retained household bundle) **and**
  an interval ([0.031671, 2.822953] kW at 0.95, quantiles 0.025/0.975, from the quantile bundle), both with units.
  The interval's answer carries `input_restandardized_from`: the attached window was standardized for the other
  engine and was re-expressed in this one's scale, and the answer says so rather than hiding it;
- `tools/verify_families.py`: examples 11/11, prose 14/14, refusals 2/2, families 5/5 (`acceptance/FINAL-families.json`).
  The example count is 11 and not 9 because each configured bundle contributes its own catalog example, and there is
  now a third;
- `tools/verify_envelopes.py`: 15/15 (`acceptance/FINAL-envelopes.json`), `execution_authorized` false throughout.
