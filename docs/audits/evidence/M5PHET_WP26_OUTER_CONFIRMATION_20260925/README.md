# WP26: the searched window, re-fitted and scored once on rows no search ever ranked (2026-09-25)

The round of 2026-09-25 (`M5PHET_WP06_STAGE5_20260925`) scored **72 stages on one seal** (`33820b552ddf`, 9,824
origins) and then published the best of them: window 21, MAE 0.512925 kW, **4.50 % better** than `baseline_hand`
(0.537108 kW). That is selection on the holdout. The seal stopped being a held-out set the moment it ranked
candidates, and the 4.50 % could have been, in whole or in part, the search fitting that population's noise.

This package answers the only question that settles it: **does the advantage survive on rows that ranked nothing?**

**It does.** On a frozen outer holdout the search never read as an objective, the search's winner beats the hand
window by **0.034412 kW — 5.73 % of the hand error — mean over five seeds**, with a paired 95 % t-interval over the
seeds of **[−0.047875, −0.020948] kW** which **excludes zero**, and a row-level paired bootstrap of
**[−0.038550, −0.030221] kW** which also excludes zero. Every one of the fifteen searched fits ranks above every one
of the five hand fits in the closure table: ranks 1–15 against 16–20, with no overlap.

What did **not** survive is the *ordering inside the search's top three*. The stage the search ranked second
(`searched_0588f5747b62`) has the lowest outer-seal mean here, the winner is second and the third is third, and the
three sit inside each other's seed spread. So the confirmed statement is about the **family**, not the champion: on
this series, at this horizon, a memory of 21–34 rows beats the hand's 60. "Window 21 with `Voltage`,
`Global_intensity` and `Sub_metering_3` is the best representation" is **not** confirmed and is not claimed.

## The nested split, declared and frozen before any fit

`nested_split.json` was written by `tools/nested_split.py` **before the first weight of this round moved**, and the
tool refuses to overwrite it.

| | rows | source rows | period | what it is |
|---|---|---|---|---|
| inner | 30,240 | 0 … 30,239 | 2009-08-23T12:45 … 2009-09-13T12:44 | fitted on, validated on; never scored on here |
| **outer** | 10,080 | 30,240 … 40,319 | 2009-09-13T12:45 … 2009-09-20T12:44 | **9,567 sealed origins**, scored once per fit |
| earlier round's seal | 10,080 | 40,320 … 50,399 | 2009-09-20T12:45 … 2009-09-27T12:44 | 9,824 origins, seal `33820b552ddf` — the population the 4.50 % was measured on |

- outer seal `d4ac73f0adde05d1…`, protocol `48b783f88f541e33…`, sealing window 197, horizon 60, sealed at
  2026-09-25T18:00:00Z, naive reference `last_value` at **0.667173 kW** on those rows;
- 257 origins of the outer block were dropped before sealing because one source row (index 30590) is non-finite and
  their window or horizon covers it. That is why the outer population is 9,567 and not 9,824;
- disjointness is **checked, not asserted**: no outer origin is an origin of the earlier seal, and the highest source
  row the outer population reads is 40,319, below the earlier holdout's first row 40,320. The tool also recomputes the
  earlier seal from its own rule and refuses if it does not reproduce `33820b552ddf`;
- every fit was passed `--expect-seal d4ac73f0adde`, which refuses **before a weight is fitted** a run that would seal
  another population.

### The outer block is not the final block by time, and that costs two things

The earlier round's seal already consumes the tail. So the outer block is the last contiguous block **before** it,
and the frozen file says what that costs rather than leaving it to be discovered:

1. **a different week.** The outer block is 2009-09-13 … 09-20, one week earlier than the population the published
   4.50 % was measured on. Nothing here shows the two weeks are the same regime — and they visibly are not equally
   easy: the naive reference is 0.667173 kW on the outer week against 0.599327 kW on the earlier one;
2. **less training data.** The inner region holds 30,240 rows against the 40,320 the earlier fits trained on, 25 %
   fewer. Every number confirmed here is confirmed under a smaller fit than the number it confirms. It is the
   conservative direction — the confirmation is not helped by extra data — but it is a difference and it is named.

### The residual this construction cannot remove

`OUTER_ROWS_WERE_TRAINING_ROWS_OF_THE_EARLIER_ROUND`. The outer rows **ranked nothing** and were **scored on by
nothing**: the search's objective was the MAE on `33820b552ddf`, which is disjoint from them. But the earlier round's
fits did read them as *training* rows. The re-fits this package dispatched never read them at all — the derived file's
holdout begins where the outer block begins, and the scaler, the training origins and the validation split all stop
there. A block untouched in *every* sense does not exist inside this 50,400-row slice, and one taken from outside it
would be another file and another provenance. The residual is written into the frozen declaration, not into a
footnote.

## What was re-fitted, and what was held fixed

`baseline_hand` and the search's top three, from **their own specs** (`specs/`, copied byte-for-byte from the
stage-5 round and from WP18 step 7), **five seeds each**, 20 fits.

Held identical across all 20: core `fused_branches` with the `tcn` encoder, one block, 200 epochs, patience 15, batch
256, `enable_op_determinism`, horizon 60, sealing window 197, the same inner rows, the same derived file. **Only the
representation and the seed differ.** No hyper-parameter was moved for any stage.

All 20 fits ran on the coordinator's own RTX 4070 under `crispdm-run -m 10G -t 3600 -n wp26 --`, 305 s of fitting in
total, 15.2 s on average, 18.7 s at the longest. The card was asked **before every one of the 20 dispatches** by
`device_for` (`tools/search_representation.py`), which falls back to the CPU and records why when another job holds it;
it reported 761–770 MiB held, below the declared 1536 MiB, on every dispatch, and all 20 ran on the GPU.

## The outer-seal closure table

`table/table.md` — 20 rows, one per (stage, seed), built by `python -m evaluation.compare_stages` from the 20
reports. All 20 are `COMPARABLE`: same seal, same metric, same target, same horizon, same naive reference, 9,567 of
9,567 sealed rows scored. A mean is not a measurement, so the table shows what each fit measured; the means and the
intervals are in `confirmation.json`.

| stage | window | outer MAE, mean of 5 seeds (kW) | sd | best–worst rank | mean skill |
|---|---|---|---|---|---|
| `searched_0588f5747b62` | 21 | **0.563236** | 0.005856 | 1 – 15 | 0.155787 |
| `searched_b373275495e2` (the search's winner) | 21 | **0.566487** | 0.003412 | 4 – 13 | 0.150915 |
| `searched_b367cba9b237` | 34 | 0.568555 | 0.003141 | 7 – 14 | 0.147816 |
| `baseline_hand` | 60 | 0.600899 | 0.009188 | 16 – 20 | 0.099337 |

Naive reference on the identical 9,567 rows: `last_value`, **0.667173 kW**. Literature value: `NOT_CARRIED` — no
report in this comparison carries one, which is a different statement from "there is none".

## The paired differences, with both intervals named

Per stage, `MAE(stage, seed) − MAE(baseline_hand, seed)` over the five seeds, in kW. Negative is better than the hand
window.

| stage | mean difference | % of hand error | 95 % t-interval over seeds | excludes 0 | 95 % bootstrap over rows | excludes 0 |
|---|---|---|---|---|---|---|
| `searched_0588f5747b62` | −0.037662 | −6.27 % | [−0.053257, −0.022068] | **yes** | [−0.042033, −0.033218] | **yes** |
| `searched_b373275495e2` | −0.034412 | −5.73 % | [−0.047875, −0.020948] | **yes** | [−0.038550, −0.030221] | **yes** |
| `searched_b367cba9b237` | −0.032344 | −5.38 % | [−0.044834, −0.019854] | **yes** | [−0.035663, −0.029039] | **yes** |

**Two intervals, two questions, both declared** — because one of them alone would be quoted as the other:

- **paired Student-t over the seeds**, two-sided 95 %, df = 4. Pairing is *by seed*: the two stages are fitted with the
  same seed, the same inner rows and the same declared epochs, patience, batch size and deterministic ops; only the
  representation differs. It measures **fit-to-fit variability under this one split and this one outer holdout**. It is
  **not** a sampling interval over datasets, weeks or households, and a seed is **not** a draw from a population of
  problems. n is 5;
- **percentile bootstrap over the 9,567 sealed rows** of the seed-averaged absolute error difference, 10,000 draws,
  generator seed 20260925. It measures the **row-sampling** uncertainty of the same difference. The rows are a
  contiguous week of one household and are **not** independent, so this interval is optimistic by an amount this tool
  does not estimate.

## The honest sentence

> The representation the search selected still beats the hand window on rows that ranked nothing: **0.034412 kW,
> 5.73 % of the hand error**, mean over five seeds, with the paired 95 % interval over seeds **[−0.047875, −0.020948]
> kW excluding zero** and the row-level bootstrap **[−0.038550, −0.030221] kW** also excluding zero. The published
> 4.50 % was therefore **not** an artefact of selecting on the seal; on this earlier, harder week the margin is if
> anything larger. What does **not** carry over is the ranking inside the top three — the search's rank-2 candidate has
> the lowest mean here, and the three are inside each other's seed spread — so what is confirmed is that a **short
> window (21–34) beats 60** on this series at this horizon, not that any one of the three is the best representation.

And what this does **not** establish, said before anyone else has to say it: one household, one 35-day slice, one
horizon, one core, one contiguous week as the outer holdout, and an outer block that the earlier round's fits had read
as training rows. It is a confirmation on untouched-for-ranking rows, not a generalisation to other series.

## Files

| path | what |
|---|---|
| `nested_split.json` | the frozen declaration: the rule, both blocks by row and by timestamp, the outer seal and protocol digest, the disjointness check, the two costs and the named residual |
| `specs/` | the four `m5phet.pipeline.v1` specs, copied unchanged from the rounds that produced them |
| `stages/<stage>__seed<n>/` | per fit: `report.json` (`m5phet-evaluation-report/1`), `config.json`, `fitted/fit_manifest.json`, `history.json`, `artifact_digests.sha256` |
| `seal.json`, `protocol.json` | one copy: they are byte-identical for all 20 fits, which is what makes this one comparison (checked, not assumed) |
| `confirmation.json` | every fit's record (device, the reason the device was chosen, wall time, MAE, RMSE, skill), the per-stage mean and sd, and both intervals with their declared meanings |
| `table/table.{json,md}` | the outer-seal closure table over the 20 fits |
| `build_table.py` | the one command that built it, `python -m evaluation.compare_stages` over the 20 reports |

`predictions.csv` (9,567 rows per fit) and the fitted graphs stay in the run directory; their sha256
digests are in each fit's `artifact_digests.sha256`, so a copy can be checked against what was scored.

The tools are `predictor/tools/nested_split.py` (declares and freezes the split) and
`predictor/tools/confirm_representation.py` (re-fits and scores, chooses nothing).

## What is NOT done here

- **no search ran.** The stages are the ones the earlier round selected; nothing in `confirm_representation.py` can
  promote a candidate;
- **no hyper-parameter was tuned**, on the inner split or anywhere else;
- **the outer seal was used once per fit**, after the fit, and is not read by anything that chooses;
- **nothing was re-ranked into the earlier table.** That table stands as the record of what the earlier rows measured;
  it now carries a pointer to this one.
