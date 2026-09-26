# The retained contrast closed offline, on five completed boundaries

This is an **additive** closure. It re-closes the nine cells of `ecl-modular-contrast-20260924b`
through the real parent closing path over evidence that already existed, under the five
boundaries of [Musashi's RP157 review](../work_plan/MUSASHI_RP157_REVIEW_2026_09_24.md)
once each of them was completed. Nothing was fitted, replayed or recomputed from arrays,
and every published number is the one RP156, RP157 and RP158 published.

## Scope

    what ran        tools/df_ecl_modular.py::close_retained_run over a run directory assembled
                    from retained files only
    what it read    the accepted run record (RP155/CONTRAST.json), the design retained beside
                    it (RP159/DESIGN.json) and the child records of the previous scoring pass
                    (RP158_SCORING.json)
    what it did NOT do
                    load a checkpoint, build a dataset, start a process, or touch the outer
                    TEST split. subprocess.run/Popen/check_output/call/check_call, score_cell
                    and author_datasets were all replaced by RAISING doubles for the whole
                    call, so a closure that reached inference would have failed here

## Cost

    closure CPU     0.002 s          model replays   0
    closure wall    0.002 s          GPU seconds     0
    not included    the contrast's own fit cost, which stands unchanged in CONTRAST.json
                    (11,415.44 CPU s) and the earlier scoring replays, which are separate and
                    are not re-spent by this closure

## What the boundaries certify, and what they do not

    design          AUTHENTICATED. The retained design's canonical content hashes to
                    b5d5eee1b5fce981… which is the digest the run recorded when it produced
                    these cells; its declared horizon (96) and seeds (2021/2022/2023) are the
                    run's own
    population      AUTHENTICATED FROM THE RUN RECORD. 2,537 available validation windows
                    behind a 640-window monitor derive 1,802 label-disjoint origins at
                    horizon 96, by the same pure rule the scoring used. The scoring file's own
                    restatement is a cross-check that must agree, never the authority
    checkpoint      AUTHENTICATED FROM THE RUN RECORD. Each child's recorded and observed
                    digests are present, equal to each other, and equal to the digest the
                    accepted record holds for that cell
    metrics         SELF-REPORTED BY THE CHILD RECORDS. Both model reductions, the matched
                    naive reduction and the derived skill are typed, finite, non-negative and
                    internally consistent — and a coherent rewrite of a metric with its
                    baseline cannot be detected from records alone. Detecting that needs
                    custody over the child records, which this closure does not have and does
                    not claim

## The result, unchanged

Status COMPLETE, nine expected and nine scored, no identity failures, no problems.
On the 1,802 label-disjoint windows, author float32 MAE:

| regime | MAE mean | SD (n=3) | matched persistence, same rows |
|---|---:|---:|---:|
| R0, no pre-training | 0.371174 | 0.001534 | 0.868283 |
| R1, frozen pre-trained detector | 0.374584 | 0.000733 | 0.868283 |
| R2, pre-trained and adjustable | 0.368596 | 0.000649 | 0.868283 |

R2 is better than R0 in three of three paired seeds, R1 worse in three of three; the paired
mean change is −0.002577 MAE, −0.694 % relative. This remains a development observation on
the validation split. It decides no hypothesis, establishes no statistical independence, is
not a financial utility estimate, and still has no matched TimeFilter score on these origins.

## References

    record          ../RP159_CLOSURE_20260925/RP159_CLOSURE.json (publication record, with the
                    input digests and the measured cost)
                    ../RP159_CLOSURE_20260925/CLOSURE.json (the closure exactly as the parent
                    path wrote it)
    script          ../RP159_CLOSURE_20260925/close_retained.py
    tests           tests/test_df_ecl_closure_boundaries.py — the five boundaries, each
                    positive case and every mandatory refusal, plus the offline closure of
                    this retained fixture with inference blocked
    before/after    ../RP159_CLOSURE_20260925/boundaries_before_repair.txt (13 failing
                    boundary cases before the completion) and suites_after_repair.txt
                    (180 passing across the three ECL suites)
    prior           RP155/RESULT.md, RP156_RESULT.md, RP157_CLOSURE.md, RP158_SCORING.json
