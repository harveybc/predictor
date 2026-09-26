# RP157: the five boundaries completed and tested, the retained contrast closed offline, and the three lanes named apart

Order: the bounded implementation completion of the [RP157 review](MUSASHI_RP157_REVIEW_2026_09_24.md) at `3c25ecab`, after
[RP156](MUSASHI_RP156_REVIEW_2026_09_24.md) and [RP155](MUSASHI_RP155_REVIEW_2026_09_24.md). Branch
`satoshi/rp132-rp134-20260923`, worktree `predictor-rp132`. The tests were written first, run through the real parent closing
path, and the retained nine-cell fixture was kept. **No model was fitted, loaded or replayed in this round; no GPU was used.**

The review asks for three things to be separated, and they are the three sections below: (1) closure acceptance on retained
evidence, (2) the Laya setup and pilot outcome, (3) concrete commits and tests for the CPU lanes — including the one that is
not done.

---

## 1. Closure acceptance using retained evidence

### 1.1 The five boundaries, each positive case and every mandatory refusal

`tests/test_df_ecl_closure_boundaries.py` is new and is written to the review's table, so a reader maps a test to a row by its
name: `test_B1_design_*`, `test_B2_dispatch_*`, `test_B3_population_*`, `test_B4_checkpoint_*`, `test_B5_metrics_*`.
**106 cases, 43 named boundary tests before parametrization.** Written first: their failure before the repair is frozen in
[`boundaries_before_repair.txt`](../evidence/RP159_CLOSURE_20260925/boundaries_before_repair.txt) — **13 failed, 91 passed**.

| boundary | positive case | refusals that already held | refusals that did NOT hold, now repaired |
|---|---|---|---|
| B1 design | a retained design authenticates: recomputed canonical content = its claim = the digest the run recorded, cells and horizon bound | altered coverage; each of the nine missing canonical fields; a re-digested foreign design; a stale design edited after sealing; no retained authority in the pure closure | **absent coverage**, **empty coverage**, and a **re-derivation that is not the run's recorded authority** |
| B2 dispatch | the task comes from the validated contract: horizon, seeds, cells, channel order, monitor support | a design horizon that is not the run's; a seed axis that is not the run's; a cell the run does not hold; a cell the design does not declare | **a caller horizon conflicting with the design** (it used to be silently replaced), **a duplicate declared cell**, **data whose channel order is not the design's**, **an undeclared monitor support**, and **a silent reseal of the retained design** |
| B3 population | exact support and elements per population, 2,537 and 1,802 at horizon 96 | all nine children consistently wrong; inconsistent elements; an omitted population; bool/string/absent window and element counts | **an extra population nobody asked for** |
| B4 checkpoint | recorded and observed digests present, equal, and equal to the accepted record's | a digest that does not match the retained record; a truthy string or number in place of the literal boolean; the wrong cell; a child nobody asked for | **a missing digest on either side**, **a run record that declares no digest for an expected cell**, **a recorded digest that is not the observed one** |
| B5 metrics | model and naive reductions named, finite, non-negative, and a skill that follows from the two numbers beside it | NaN/inf/bool/string in any measured field of any reduction, the naive one included; a negative loss; an omitted model or naive reduction; a contradictory skill; a number against a zero baseline; a null skill against a non-zero one | none: this boundary was already complete at the tested scope |

### 1.2 What the tests forced in the contract code — minimal, inside the accepted schema

All of it is in `tools/df_ecl_modular.py`. No new registry, model, service or research stage; one new pure helper in the same
module and three gates inside functions that already existed.

- `authenticate_design` — the identity declaration is mandatory and must be exactly this schema's canonical fields, so absent
  and empty coverage are refusals rather than a skipped check; a cell declared twice is refused by name instead of collapsing
  into a set and changing the denominator a regime is averaged over.
- `bind_dispatch_task` (new, 25 lines) — the horizon, seed/regime axes, channel order and monitor support of a dispatch are
  derived from the authenticated design and the accepted run record. A caller argument that conflicts is refused, not
  overridden. `score_contrast`'s `pred_len` default became `None` for exactly this reason: a default is indistinguishable from
  an intention, and the old one silently agreed with everything.
- `score_contrast` — binds the task before any dataset is built, refuses a supplied design that is not the one retained beside
  the run, and refuses data that does not deliver the population the accepted record declares.
- `validate_children` — an unexpected population is a problem; and with the accepted record in hand, both the recorded and the
  observed checkpoint digests must be present, equal to each other, and the cell's.

Two existing test helpers were updated because the contract is now stricter (`_full_design` declares its identity coverage;
retained children carry the recorded digest as well as the observed one), and one existing test now stops at
`channel_order_digest` instead of `author_datasets`, because the dispatch binds the channel order before it builds anything.

### 1.3 The offline validation-and-aggregation path

`close_contrast` (validate → reconcile → aggregate) and `close_retained_run` (reach it from disk alone) were already extracted
under RP158; this round they are proved on the **real** retained fixture, not only on a synthetic one:

    test_the_retained_nine_cell_contrast_closes_offline_with_no_inference_and_the_published_numbers

`subprocess.run/Popen/check_output/call/check_call`, `score_cell` and `author_datasets` are all raising doubles for the whole
call, so a closure that reached inference fails the test instead of costing nine model replays. It asserts the published
numbers come back: R0 0.371174, R1 0.374584, R2 0.368596, matched persistence 0.868283, on 1,802 label-disjoint windows.

### 1.4 The named gap of RP158, closed — and how

RP158 named one honestly: `ecl-modular-contrast-20260924b` retained no design beside it, so it could not be closed purely.
That run directory no longer exists on this host, so the design was re-derived on CPU by `seal_contrast` from the governed ECL
delivery — **no fit, no checkpoint, no inference; a CSV header read and an unfitted model's shapes** — and it reproduces
exactly the digest the run recorded, `b5d5eee1b5fce981…`. That equality is the only reason it is admissible, and there is a
test for the case where a re-derivation does *not* reproduce it. It is retained as
[`RP159/DESIGN.json`](../evidence/d3_k5_20260917/RP159/DESIGN.json) with its
[provenance](../evidence/d3_k5_20260917/RP159/PROVENANCE.md).

### 1.5 The additive closure

[`RP159_CLOSURE.md`](../evidence/d3_k5_20260917/RP159_CLOSURE.md), published through the real parent path over retained
evidence by [`close_retained.py`](../evidence/RP159_CLOSURE_20260925/close_retained.py).

    scope     close_retained_run over RP155/CONTRAST.json + RP159/DESIGN.json + RP158_SCORING.json
    cost      0.002 CPU s, 0.002 wall s, 0 model replays, 0 GPU s
              (the contrast's own 11,415.44 CPU s of fitting stands unchanged in CONTRAST.json and is not re-spent)
    result    COMPLETE, 9 expected, 9 scored, no identity failures, no problems
    numbers   R0 0.371174 (SD 0.001534), R1 0.374584 (0.000733), R2 0.368596 (0.000649); persistence 0.868283

What each part rests on is in the record rather than implied: design AUTHENTICATED, population and checkpoint identity
AUTHENTICATED FROM THE RUN RECORD, and metric **values** SELF-REPORTED BY THE CHILD RECORDS — a coherent rewrite of a MAE with
its baseline and its skill is internally consistent, and consistency is not custody. The reading is unchanged and still
decides nothing: a development observation on the validation split, no H1, no financial utility, and still no matched
TimeFilter score on these origins.

---

## 2. Laya: setup and pilot are done and measured

The review asked to record SETUP_IN_PROGRESS_REPORTED and to execute the authorized pilot once setup succeeded. Setup
succeeded and the pilot ran; it is **not** re-run for this return. Evidence:
[`M5PHET_ROUND6_20260925/`](../evidence/M5PHET_ROUND6_20260925/), report `report_laya_zero_shot.json`, comparison
`stage_comparison.md`.

    corpus        450 economic-calendar release lines, 150 per class, seed 1729, stratified 2011–2021
    labels        INDEPENDENT_SYSTEM: the calendar's own country field; nobody in this system wrote or adjudicated one.
                  Euro-area member states are excluded and counted, never relabelled
    identity      corpus 7e4789e5…, protocol b9aefb3c…, seal 31257d47… sealed 2026-09-25T00:00:00Z
    execution     all 450 answers from backend `laya`, one state_ref, one wording, asked once, no prompt tuned;
                  a real checkpoint on the worker's GPU, not a stub or a cached transcript
    macro-F1      0.377760 (accuracy 0.393333) against keyword baseline 0.175996 and majority class 0.166667
    calibration   ECE 0.1315, multiclass Brier 0.6473; below 0.8 confidence the head is at chance, the 55 rows above 0.8
                  are 85 % correct and the 31 above 0.9 are 100 % correct
    literature    NOT_CARRIED: no published macro-F1 matched to this protocol, so none is quoted

Read plainly: it beats both references and is far from useful — a standing `euro_area` bias (223 of 450 predictions) and
per-class F1 0.504 / 0.336 / 0.293. It knows a little and knows when, only at the top of its confidence range.

One placement fact the review asked about: the 16-second stage fits ran on the coordinator's RTX 4070 rather than the
preferred external 5090, because the worker's only TensorFlow environment cannot open its GPU libraries
(`Cannot dlopen some GPU libraries`, CUDA-13 wheels beside cu12) while a ~900-item Laya measurement held that GPU. The Laya
measurement itself ran on the worker GPU. That is the measured cause, not a preference.

---

## 3. The CPU lanes: two done, one not

### 3.1 Collector receipt / revision / dedup persistence — DONE

`financial-data`, branch `satoshi/wp32-pit-capture-20260925` @ `ef0ba6612` (pushed).

    what        m5phet.point_in_time.v1, an append-only point-in-time store. The key is a digest of the release and the
                observation; OUR receipt clock is deliberately outside the key, so re-reading an unchanged archive is a
                DUPLICATE and not a revision. A changed value is a NEW row with a higher revision_index and the earlier row
                keeps its bytes. Different facts under an existing key are refused ROW_REWRITE_REFUSED.
                known_at(T) returns, per release, the last valid row received no later than T
    files       _scripts/lib/point_in_time_store.py, _scripts/workers/stage13_point_in_time_capture_worker.py,
                _scripts/systemd/financial-data-point-in-time-capture.{service,timer}, docs/POINT_IN_TIME_CAPTURE.md
    tests       _scripts/tests/test_point_in_time_capture.py, 21 new tests; suite 494 passed (473 before)
    measured    first pass 896 scheduled releases read, 727 past and recorded; the second pass wrote nothing
    schedule    systemd --user oneshot every three hours at :07 UTC, Persistent, MemoryMax=1G, CUDA blanked
    boundary    a receipt clock bounds the publication instant from above and never below. It supports availability-correct
                backtesting and revision analysis; it does NOT produce the consensus the event study needs

### 3.2 Economic-calendar source inventory and as-of work — DONE

`feature-eng`, worktree `feature-eng-events`, branch `satoshi/wp28-observed-clock-20260925` (pushed), commits `b097916`
(`calendar_join.py`), `cc08f60` (`calendar_clock.py` + status), `3de34fd`.

    inventory   docs/EVENT_STUDY_DATA_STATUS.md: the consensus archive spans 2011-01→2021-04 and the only archive with
                observed instants spans 2024-12→2026-05. Zero common days; join rate 0 of 54,275, with 27.2 % of rows
                clearing economy and release name and failing only on the date. The three feeds that would unlock it are named
    as-of       calendar_clock.py measures the archive's true offsets from its own conventions rather than assuming UTC, and
                requires a cluster spanning zones with different daylight-saving calendars so the agreement cannot be one
                convention's artefact. Result over 2,177 estimates: fixed UTC−05:00 year-round until 2018-01, then
                America/New_York local time from 2018-03 — correcting the earlier eyeball reading
    consequence re-running the chain from where releases actually happened removes the two cells that excluded zero under the
                misreading (they were reading the market four to five hours before the release); identification stays
                NOT_IDENTIFIED, now also PLACEBO_FAILED on all 40 triples. No study was registered from that run
    tests       tests/test_calendar_clock.py (13), tests/test_calendar_join.py (14), tests/test_economic_calendar.py (20)

### 3.3 Demo/paper broker refusal and recovery interfaces — NOT DONE

Nothing was implemented, no branch exists, no broker call was made and no entitlement was tested. There is no partial work to
review and nothing that could be mistaken for a fill. It is the lane that did not start this round; the two above and the
closure completion took the time. No external object is missing for it — it is unstarted, not blocked.

---

## 4. Boundaries that could not be completed without a new allocation or a model run

None. All five completed within the existing contracts, with zero model replays. The one place where that was in question —
the missing design authority for the retained run — was resolved by a CPU re-derivation that reproduces the run's recorded
digest, which is exactly the condition the accepted schema already required. Had it not reproduced it, the design boundary
would have stopped there rather than manufacture its own authority.

## 5. What is still not established, and is not claimed here

- The metric **values** of the nine cells remain the child records' word. Detecting a coherent rewrite needs custody over the
  child records themselves, which this closure does not have and says so in its own record.
- The matched TimeFilter score on these 1,802 origins is still undelivered; the published test MAE is not a comparator for
  these validation scores.
- The R2 development reading is unchanged and still decides no hypothesis, establishes no statistical independence and implies
  no trading profit. Three initialization seeds are not three datasets or three market regimes.

## 6. Tests and cost

    before      74 passed — tests/test_df_ecl_contrast_gates.py + tests/test_df_ecl_modular.py, 170 s
    after       180 passed — the two suites plus tests/test_df_ecl_closure_boundaries.py, 156 s
                (frozen in ../evidence/RP159_CLOSURE_20260925/suites_after_repair.txt)
    interpreter the one these df_ tests need: anaconda env `trading-stack` (torch 2.13.0, TF 2.21.0). The base interpreter
                has no torch and silently reduces the data-dependent tests to 11 failures, which is worth saying once
    cost        about 300 CPU s in the two reported suite runs, 3.3 s for the design re-derivation and 0.002 CPU s
                for the published closure;
                everything under `crispdm-run -m 6G -t 900 -n rp157`, CUDA_VISIBLE_DEVICES='' throughout
    not spent   0 model replays, 0 GPU seconds, no training, no new allocation

## 7. Review request

(a) Whether the five boundaries are complete as enumerated, or whether a refusal case is still missing from the table.
(b) The design re-derivation in §1.4: admissible because it reproduces the run's recorded digest, or still too close to a
fresh seal for the authority you asked for. (c) Whether refusing a caller override — rather than silently adopting the
authenticated horizon — is the behaviour you meant at the dispatch boundary. (d) The closure's explicit statement that metric
values are self-reported, and whether that limit is stated at the right place. (e) The broker lane named as not done.
