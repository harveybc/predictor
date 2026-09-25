# WP06 stage 5: the representation space searched, with the closure table's own number as the objective (2026-09-25)

Stages 1–4 built one representation by hand, four by a design job, and scored each of them once on a sealed holdout.
This round spends a declared budget of fits over a declared space, makes **every evaluated point a stage of the
table**, and rebuilds the closure table and the calibration report over all of them.

**The finding: the hand window did not survive.** Four searched representations beat `baseline_hand` (0.537108 kW) on
the identical seal, the best by **0.024183 kW** (4.50 % of the hand error), and every one of them declares a window
**shorter** than the hand's 60 rows. The designed candidates of stage 3 had all been longer.

## The objective, and what was held fixed

The objective is one scalar: the **held-out mean absolute error in kW read out of the `m5phet-evaluation-report/1`
that the closure table reads**, minimised. It is not recomputed by the search — `tools/search_representation.py`
opens the report the fit wrote and takes `metric_sets[0].values.mae`.

Every fit ran through `tools/fit_pipeline_spec.py` with everything but the representation held at `baseline_hand`'s
own configuration: the same `fused_branches` core, the same `tcn` encoder, one block, no per-feature preprocessor
declared, 200 epochs, patience 15, batch 256, seed 1, `enable_op_determinism`, one horizon of 60 steps, and the same
sealing window of 197. **No hyper-parameter was moved for any point**, and the fit refuses before a weight is fitted
if it would seal another population (`--expect-seal`, added this round).

All 66 stages fitted here carry `seal 33820b552ddf`, `protocol d0ebd9a4bc75`, **9824 of 9824 sealed rows scored** —
checked over every manifest, not sampled.

### The control: the harness still reproduces the earlier number to the bit

`control_baseline_hand_refit` is `baseline_hand`'s own spec, refitted with this round's harness on this round's GPU.
It returns **MAE 0.5371084250158522, RMSE 0.7927872087974042, 18 epochs** — the same numbers, to every digit
published, as the run of 2026-09-25T08:15. The level/no-differencing path is therefore untouched by the transform and
differencing code this round added, and the table's two rows for it are a reproduction, not a duplicate.

## The declared space (`ledger_*.json` → `space`)

| gene | values | why |
|---|---|---|
| `window` | integer, 3 … 197 | 197 is the sealing window: a longer window seals a different population and **is refused** — the refusal that removed `seasonal_lag_1443` and `seasonal_lag_2892` from the comparison, reused as the bound of the space rather than worked around. The shortest is the encoder's kernel size (3), below which the window is all causal padding |
| `lags` | subset of {1, 74, 197} | the design job's own motivated set: the one-step lag, the ACF peaks outside the ±0.011206 band and the decay lag. **1443 and 2892 are excluded from the space by the same sealing-window refusal**, named in the ledger, not silently dropped |
| `transform` | `level`, `diff`, `log_return` | exactly the values `m5phet.representation.v1` declares |
| `differencing.order` | 0, 1, 2 | `MAX_ORDER` of the spec; an order above 0 under a transform that already differences is refused by the spec itself |
| features | subset of the seven meter columns | the subset the branch gathers is the subset the graph reads |

## The budget, in two declared rounds

| round | window range | draws | generations × individuals | fits dispatched | measured | refused |
|---|---|---|---|---|---|---|
| 1 (`seed 1`) | 3 … 197 | 120 initial | 5 × 24 after it | 47 | 47 | 127 |
| 2 (`seed 2`) | 3 … 73 | 400 initial | 3 × 16 after it | 18 | 18 | 404 |
| **total** | | | | **65** | **65** | **531** |

Round 2 is declared, not improvised, and its reason is a bias round 1 made visible: the space couples the window to
the lags, so a uniform draw over 3…197 is legal only ~16 % of the time and **not one measured point of round 1 had a
window below 74** — the region containing the hand window itself was never tested. Round 2 draws from the declared
sub-range 3…73, where only the lag set {1} can be legal (hence its 404 refusals, all free: a refusal costs no fit).
Both rounds share the objective, the seal, the held-fixed configuration and the ledger format; they are one search
and one decision record.

Fits cost 22.6 s on average (max 51.4 s) on the local RTX 4070, 65 fits in 24.0 minutes of fitting, under
`crispdm-run -m 10G -t 3600 -n wp06s --`, well inside the ~2 hours declared.

## What was refused, by name, and never repaired

| refusal | round 1 | round 2 | what it is |
|---|---|---|---|
| `LAG_EXCEEDS_WINDOW` | 94 | 313 | the genome names a lag longer than its own window; this harness fits one contiguous window, so that history is never read. The same refusal that kept the two long design candidates out of the sealed comparison |
| `NO_LAG_DECLARED` | 15 | 69 | the genome selects no lag at all |
| `AMBIGUOUS_DIFFERENCING` | 18 | 19 | `differencing.order > 0` under a transform that already differences — raised by `feature_eng_m5phet.representation.validate_spec` itself, not by this harness |
| `NO_FEATURE_SELECTED` | 0 | 3 | the genome selects no column; a branch that reads nothing represents nothing |
| **total** | **127** | **404** | 531 of 596 points, 89 % |

No refused point was moved to the nearest legal one. Each is written to the ledger with its genome and its decoded
point, so the space that was searched is the space that is reported.

## The closure table (`table/table.md`), the rows that matter

74 stages: 72 measured on the one seal, 2 `NO_NEW_MEASUREMENT` carrying the refusals of stage 3 verbatim.
Naive reference on the same 9824 rows: `last_value`, **MAE 0.599327 kW**.

| rank | stage | window | model error (kW) | skill | comparability |
|---|---|---|---|---|---|
| 1 | `searched_b373275495e2` | 21 | **0.512925** | 0.144164 | COMPARABLE |
| 2 | `searched_0588f5747b62` | 21 | 0.512970 | 0.144089 | COMPARABLE |
| 3 | `searched_b367cba9b237` | 34 | 0.521704 | 0.129516 | COMPARABLE |
| 4 | `quantile_hand_95` | 60 | 0.526294 | 0.121858 | COMPARABLE |
| 5 | `searched_77a56b176915` | 31 | 0.526542 | 0.121444 | COMPARABLE |
| 6 | `baseline_hand` | 60 | 0.537108 | 0.103813 | COMPARABLE |
| 6 | `control_baseline_hand_refit` | 60 | 0.537108 | 0.103813 | COMPARABLE |
| 8 | `quantile_hand` | 60 | 0.538670 | 0.101208 | COMPARABLE |
| 10 | `candidate_seasonal_lag_74` | 74 | 0.545436 | 0.089919 | COMPARABLE |
| 14 | `laya_chosen` | 60 | 0.557190 | 0.070307 | COMPARABLE |
| 72 | `searched_facdd5812c0f` | 184 | 0.697646 | −0.164049 | COMPARABLE |

The winner: **window 21, lags [1], transform `level`, differencing order 0, columns `Voltage`,
`Global_intensity`, `Sub_metering_3`** — 18 epochs, early stopping on the same patience as every other stage.

Three things about it deserve to be said rather than left to be read off:

1. **It is the window that carries the effect, not the column subset.** Rank 3 declares a window of 34 and **all
   seven** columns and still beats the hand window. At this horizon, on this slice, a memory of 21–34 rows is simply
   better than one of 60. Grouped by the window the stage fitted, over the 66 stages fitted this round: window <= 40 --
   11 stages, best rank **1**, median error 0.557326; window 41-100 -- 16 stages, best rank 6, median 0.573918;
   window > 100 -- 39 stages, best rank 11, median 0.590819.
2. **The two best stages exclude the target column itself** and keep `Global_intensity`, which on a household meter
   is very nearly a rescaling of the active power. That is a fact about this slice; it is not a claim that the target's
   own past is useless, and nothing here tested that question.
3. **The margin is small and single-run.** 0.512925 against 0.537108 is 0.024183 kW on 9824 rows, from one fit per
   point at one seed. No repetition, no interval on the difference, no second dataset. `NO_NEW_MEASUREMENT` is the
   right answer to "is this representation better in general"; what the table says is what these rows measured.

## Two limitations of the harness this search exposed

**The `lags` field decides nothing that the window does not already decide.** `searched_2fbc35eb0452` (lags [1, 74])
and `searched_950e4a6f52e7` (lags [1]) differ only in the declared lag list. Their pipeline specs have different
digests — and their predictor configurations are **byte-identical**, their epoch counts identical, and their MAE
identical to all 16 digits (0.5483918423350391). Under a single contiguous-window harness a lag inside the window is
already in the tensor, so the lag list is a declaration that only bites at the refusal boundary. The search's decision
record says so in its own `why`, and the tie was broken by the smallest candidate id, declared, not by whichever
sorted first.

**`differencing.order` had to be given a reading, and the reading is recorded in every manifest.** The spec says a
transform and an order are not two ways of saying the same thing — the transform is what the target is modelled
under, the order is "extra differences applied on top" — and does not say on top of what. Differencing the **target**
would make the level at the origin plus the horizon unrecoverable from what the origin carries (the d-th difference
at t+h needs the levels at t+h−1 … t+h−d), so every such stage would be incomparable with sealed rows that are levels
in kW. This harness therefore differences the **input channels** and leaves the label to `target.transform`, records
that sentence as `representation_applied.differencing_reading` in every run manifest, and needs `window + order` rows
of history, refusing `WINDOW_PLUS_DIFFERENCING_EXCEEDS_SEALING_WINDOW` past 197.

## The decision records (WP23, third chooser)

`m5phet.decide` gained `chosen_by: SEARCH` this round (M5PHET branch `satoshi/wp06-search-20260925`): the same
absences as a person's record — no probabilities, no backend, no checkpoint — and a `why` that **must** name the
objective it minimised and the budget it spent, through fixed markers `validate_decision` checks. A search is neither
a person nor Laya: it produced no distribution to calibrate, and nobody deliberated.

- `search_decision.json` — one record, `representation` / `searched_candidate`, 65 options (every point that was
  *evaluated*; a refused point is not an option, because no fit of it exists), chosen `searched_b373275495e2`;
- `winner_decisions.json` — WP23's clause applied to the rank-1 stage: the 7 `feature_preprocessing` records a
  **person** is responsible for (the search chose the representation and nothing else), and **three refusals**:
  `feature_grouping/grouping_cut` (no `k=1` among the declared cuts), `group_extractor/extractor` (`tcn` is not a
  feature-extractor plugin) and — new this round — `representation/candidate`, because the winning representation is
  not one of the five that question declares and **a search may not add an option to a set the executing repository
  declared**. That refusal is exactly why the search needed a question of its own;
- `link_outcomes.json` — 58 linked (10 LAYA, 47 HUMAN, 1 SEARCH), 16 refused `NOT_COMPARABLE` (the records of the two
  candidates that have no measured row). Outcomes live in
  `~/.local/state/m5phet/decision_outcomes-wp06-stage5-20260925`, a directory of their own, because the previous
  round's outcomes bind rows of the table this one supersedes.

### What the calibration report now says, and why it says less than before

`table/calibration.md`, over the new table: every (kind, question) is `NO_NEW_MEASUREMENT` — the minimum is 30
**scorable** linked outcomes and the most any group has is 7. Nothing about Laya is measured here, as before.

What changed is the labels, and the change is the rule working. A **searched** stage is now ranked first, and it can
answer only two of the five questions: `feature_preprocessing/preprocessing` (label `normalizer`, kept) and
`representation/searched_candidate` (label `searched_b373275495e2`). The other three — `feature_grouping/grouping_cut`,
`group_extractor/extractor` and `representation/candidate` — report `NO_BEST_RANKED_OPTION` naming
`searched_b373275495e2` as the rank-1 stage that carries no record for them. `representation/candidate` **lost** the
label `hand_household_w60` it had last round for exactly the honest reason: the stage that is now first did not choose
among those five options at all.

Only the rank-1 searched stage carries human records. The other 64 searched stages are
`COMPARABLE_BUT_NO_DECISION_RECORD`, printed per stage in the report: comparable in the table, unusable for
calibration — WP23's own distinction, stated rather than left as a gap.

## Why this optimiser and not DOIN

`doin-core`'s `OptimizationPlugin` is a three-method interface and would have been easy to implement; the run around
it is not, and it would change what the number means. A DOIN domain (`doin_core.models.domain.DomainConfig`) needs an
optimization plugin **and an inference plugin** so evaluators can re-verify a reported optimum, and the predictor
domain's verification path is `doin_plugins.predictor.synthetic` — a pre-trained HMM generator producing deterministic
**synthetic** rows, precisely so that evaluators need not share the corpus. A sealed holdout is the opposite premise:
the number in the closure table is the number on *those* 9824 rows. Running this search through DOIN would have put a
second scoring path beside the seal, plus a node, a chain and a consensus threshold, for a search that runs on one
host in 25 minutes of fitting.

So: **predictor's own DEAP path**, as the order allows. `optimizer_plugins/default_optimizer.py` could not be called
as a function — its `optimize(predictor_plugin, preprocessor_plugin, config)` builds its own datasets through the
pipeline and its fitness is predictor's own validation metric, not a sealed-holdout MAE read from a report — so what
is reused is its **operator set and its conventions**, identically: integer genes over declared bounds, `cxTwoPoint`,
per-gene uniform redraw at `indpb`, `selTournament(tournsize=3)`, elitism of one, and a resume ledger in the spirit of
`optimizer_plugins/modules/resume_operations.py` (restarting the same command re-reads every finished evaluation and
loses no fit). No third optimizer was written.

## Files

| path | what |
|---|---|
| `ledger_search.json`, `ledger_search_short.json` | the two rounds: the declared space, the budget, the operators, every evaluation and every refusal with its genome |
| `specs/` | the `m5phet.pipeline.v1` spec of every point that was fitted |
| `stages/<stage>/` | per stage: `report.json` (`m5phet-evaluation-report/1`), `config.json`, `fit_manifest.json` (the representation by value), `history.json`, `artifact_digests.sha256` |
| `seal.json`, `protocol.json` | one copy: they are identical for every stage, which is what makes this one comparison |
| `table/table.{json,md}` | the closure table over 74 stages |
| `table/calibration.{json,md}` | the WP23 calibration report |
| `search_decision.json`, `winner_decisions.json`, `link_outcomes.json` | the decision records and their outcomes |
| `collect.py`, `build_table.py`, `search_decision.py`, `winner_decisions.py`, `link_outcomes.py` | the scripts that produced the four files above, in order |

`predictions.csv` (9824 rows per stage) and the fitted graphs stay in the run directory; their sha256 digests are in
each stage's `artifact_digests.sha256`, so a copy can be checked against what was scored.
