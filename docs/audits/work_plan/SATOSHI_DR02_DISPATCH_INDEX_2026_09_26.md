# DR02 — one dispatch index, current: eleven lanes, their real dependencies, their next eligible work

**Date:** 2026-09-26
**From:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Order:** [DR02](../../handoffs/SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md), after the
[independent day review](MUSASHI_DAY_REVIEW_2026_09_26.md) (`CHANGES_REQUIRED`, `0ae37a68`)
**Branch:** `satoshi/dr02-dispatch-index-20260926`, worktree `predictor-dr02-20260926`, base `0ae37a68`
with `satoshi/rp49-rp64-disposition-20260926` merged so that the rulings and the review live in one tree

Nothing in this document is written, signed or attributed in the auditor's name. Where the
auditor's judgement governs, this document says so and stops.

---

## 0. What was wrong, and what the fix is

The audit's finding F6 is the one this order answers: the indices did not describe the day, and
dependencies had been read wrongly — by me among others. Two concrete consequences, both named in
the review: a closed node was re-dispatchable from a stale row, and whole modules sat blocked on two
reviews that were never going to arrive and that, since this morning, have dispositions.

The fix is structural, not another note. **State now lives in a row, not in a paragraph.**
[`EXPERIMENT_EXECUTION_QUEUE.json`](../../tres_temas_entrevista/program_v3/EXPERIMENT_EXECUTION_QUEUE.json)
is now `research_dispatch_index.v2`: one current row per lane, each carrying commit, artifact or
receipt, observed state with the date it was observed, the **real** dependency, the next eligible
work, the resource, the remaining cost and the lease. The previous `research_execution_queue.v1` is
preserved verbatim under `history.superseded_queue_v1` — including its twelve-cell list, which is the
thing a reader could mistake for a live queue, and which is now labelled a 2026-09-23 photograph.

The programme state and the assignment index were changed to agree with the rows, and one consumer
that contradicted the state outright was reconciled (§5).

**Eleven lanes have a current row.** Ten are the lanes DR02 names. The eleventh is
`E1-Q2-CONTEXT`, added because it held the programme's only live compute when this round began and is
the E1 seal's one unmet *measurement* condition; leaving it out would have hidden a running job from
the index whose job is to stop exactly that. It has since closed at its own boundary, and the row says
so with the date.

Every dependency below was derived from artifacts. Where it could not be, the row says so in those
words. **Five rows** say it, all about remaining cost: news, MT5 demo, Alpaca paper, M5PHET and
calendar. No row guesses.

---

## 1. Two things this index does not do

**It does not reactivate anything.** A/B's twelve-cell fit queue finished and its service is
inactive; the nine-cell R0/R1/R2-ECL contrast is closed, 9 of 9, at `2026-09-26T01:28:52Z`. Both
rows say `NOTHING TO DISPATCH` and both carry the words *do not reactivate*. A stale row is not an
invitation, and the reading rules of the index say that in the file itself.

**A documentary pass is not a scientific authorization.** `check_plan.py` prints
`documentary_coverage: PASS` and `scientific_approval: false` in the same object, and its docstring
says it checks the programme's documentary coverage, never scientific validity. Its PASS means the
paperwork is internally consistent. It accepts no number, admits no measurement, and authorizes no
fit. The index repeats this as a reading rule so the two can never be quoted as one.

---

## 2. The eleven rows: real dependency, and next eligible work

### A/B — SOTA-REPRO protocols A and B, published-recipe ECL L512 reference

*Commit* A `e36c1c6d`, B `07140d03`, both in master `941eb5b3`.
*Artifacts* `RP144/EVIDENCE_COMPOSITION.1790220949.json`, `RP144/COMPOSED_TABLE.1790220949.json`,
`RP145/CLOSURE_RECONCILIATION.md`, `RP145/SOTA_TABLE.L512.json`.
*Observed* 2026-09-24. A: 12 cells composed, all twelve with a bound exact replay,
`with_measured_same_device_replay` **empty** and device attribution `UNKNOWN` by name, four-horizon
mean MSE 0.161962 / MAE 0.259662. B: 12/12 verified, every replay bit-exact
(`max_abs_prediction_difference` 0.0), four-horizon mean MSE 0.150307 / MAE 0.246397, per-horizon
`OPERATIONAL_AGREEMENT` under the predeclared band.

**Real dependency: none outstanding.** Derived from the evidence files themselves — reconciliation
complete, `problems []`, and no cell anywhere in them in a `QUEUED` or `RUNNING` state.

**Next eligible work: nothing to dispatch.** Independent review of a finished artifact, and nothing
else. 16,167 of the 24,000 CPU s allocation is consumed (14,011 fits + 2,156 closure); the remainder
is not a licence to spend.

### R0/R1/R2-ECL — modular learning-regime development contrast

*Commit* `69739b11`. *Artifact* `RP159_CLOSURE_20260925/RP159_CLOSURE.json`, binding its three
inputs by digest.
*Observed* `2026-09-26T01:28:52Z`: `COMPLETE`, 9 of 9 cells, `identity_failures []`, `problems []`.
On the 1,802 label-disjoint windows: R0 0.371174, R1 0.374584, R2 0.368596 mean MAE kW against a
matched persistence of 0.868283. The closure cost 0.002 CPU s, replayed no model and started no
process — every route to inference was a raising double.

**Real dependency: none.** This is the contrast the order names as already finished. Its own
authority block states the honest limits — metric values are self-reported by the child records — and
those are limits, not dependencies.

**Next eligible work: nothing to dispatch. Do not reactivate the contrast that already finished.**
Eligible: independent review, and the separate question of custody over the child records, which is
governance and not a refit.

### E1-Q2-CONTEXT — E1 household context separation (the added row; live when this round began, closed by its end)

*Commits* `7d2b1c83`, `c74f910b`, `b1a13928`, `82c633e8`, `91ad13a5`.
*Artifacts* the run root `e1_block_q2_context_deep_v1/` with `DESIGN.json`, `EXTENSION_SEAL.json`,
`BLOCK_DATA.json`, `MEMORY_GATE.jsonl`, `attempts/<unit>/{cell.json,arrays.npz,weights.weights.h5}`,
and `ASYMMETRIC_READING.json`.

*Observed* — and this is the row that most needed observing rather than reading. It changed while this
document was being written, and the change is recorded rather than smoothed over. At
`2026-09-26T19:27:19Z` this lane held the programme's only live process, and that process was **not
training**: it was the memory gate, reading `verdict: HELD_WAITING_FOR_MEMORY`, `waited_seconds
1395.0`, `required_bytes 9,532,141,568`, `fits_measured_peak_plus_margin: true`,
`crispdm_run_would_accept_the_cap: **false**`. It held to 1440 s, opened a second attempt at
`19:28:06Z`, and the run then ended at its own boundary.

**Closed state, `2026-09-26T19:31:55Z`: twelve of eighteen units in existence, and no live compute.**
`DESIGN.json` declares six arms at three seeds. Twelve cells exist with arrays and weights —
`modular_w60`, `daily_lag`, `long_window_crop60`, `short_window_deep_core`. The two W1440 full-depth
arms, `long_window_local_support_67` and `long_window_own_depth`, have **no cell and no attempt
directory**: six units were never admitted. The block's own closure then **refused**: `verified:
false`, *"closure FAILED: no verified comparator, no selected arm and no scientific proposal are
emitted"*, because all eighteen registered units have no accepted terminal receipt — the
`NON_GOVERNING` consequence the runner declares of itself, not a numerical failure. Disposition
`HISTORICAL_DEV_ONLY`, 522.833 CPU s, 12 rows, custody `UNCHECKED` on every one. That is DR01's *close
each attempt at its normal boundary and record the failure*, done, and I did not kill anything to get
it.

**Real dependency: two, and they are different things.**

*For the six missing W1440 units: the shared admission guard itself.* The derivation is the gate's own
log — the request fits available memory plus its margin and `crispdm-run` would still refuse the cap —
so the blocker is DR01's atomic per-host reservation, not data, design or budget. The alternative that
removes the dependency without repairing it is an admitted host that is not the coordinator.

*For any of the twelve existing cells to count as governed: a data-gov service key, hence an accepted
terminal receipt.* And here the dependency column earns its keep, because this has been reported as a
missing credential rather than as a named reference: **the reference exists** —
`data-gov scripts/issue_service_key.py` at `d07695b` issues one principal's key using data-gov's own
hashing implementation, writing the plaintext to a key file at mode 600 and never into a checkout. No
secret needs to be hunted for, copied or read. Until a key is issued and the delivery path is smoked,
every measurement in this lane is `NON_GOVERNING` by construction, exactly as the runner says.

**Next eligible work**, in order: DR01's atomic reservation, which is CPU-only code and tests against a
simulated clock — the six missing units are *its* dependents and not the reverse; then resolve identity
and client through the reference above and smoke delivery → reader → terminal → warehouse on one small
mechanical unit *before* any further scientific fit. Nothing else belongs in this lane, because
`ASYMMETRIC_READING.json` already fixed the reading in advance and asymmetrically: a long-context arm
that lands worse is `CONFOUNDED_WITH_BUDGET` and identifies nothing. Do not relaunch the W1440 arms
into the same refusal.

### H-CORE — MOD-FROZEN-PREFIX and MOD-CORE-PRETRAIN

*Commits* `e97b286a`/`a84a913c` (the rulings), `6820fcae` (lag table restated, materialization per
split), `458d22cb` (measured resolution, repaired monitor, matched contrast).

This is the first of the two corrections the audit required to land in a row, and it lands here and
in the next row. **MOD-FROZEN-PREFIX is unblocked** by the disposition of this morning, and its first
two items are delivered at `6820fcae`. **MOD-CORE-PRETRAIN's first two prerequisites are now
supplied** by the supplement at `458d22cb`, which postdates the review: sigma 0.01136346 kW on 6 df,
smallest defensible resolution 0.031106 kW deliberately quoted, against an R1−R0 effect of 0.009805
kW — 3.17× the effect, verdict `UNANSWERABLE_BY_THIS_PROTOCOL_AT_THIS_SEED_COUNT`, 23 seeds per arm
required; and a budget-matched contrast that ran nine cells at `19:04:34Z` with one `val_mae` monitor
and 12,000 updates per arm, every cell replaying bitwise, every closure row
`UNANCHORED_NO_TERMINAL` — measured, not governed.

**Real dependency: MOD-FROZEN-PREFIX's own deliverable** — the materialized prefix *output*
(preprocessing, groups, detector, adapter, fusion, with learned states, clock, row/split, shape and
version), with direct-versus-cache parity, reload in a fresh process, weight and state mutation, and
causality. Parity of a CSV or a panel is not that artifact; enabling implementation is not accepting
delivery. Separately, and only if the module's effect is to be resolved on this protocol: 23 seeds
per arm, which no allocation grants.

**Withdrawn as a dependency:** the 0.049 kW scrambled-label difference as a resolution floor or as
this contrast's noise. Changing the labels changes the task. The resolution that replaced it reads
that control as *read*, and pools sigma over the module's own three arms — the pooling Bartlett's
test allows at p=0.74, where pooling all five does not at p=0.0086.

**Next eligible work:** materialize the frozen prefix output and its four parity and causality tests.
CPU, no reserve opened.

One caveat this row carries rather than hides: the restated lag table at `6820fcae` needs its own
correction. It dropped the non-finite row before applying offsets, so some pairs k indices apart are
no longer k minutes apart, and it labelled a per-segment centred Pearson as bias-corrected. The
day-versus-week inversion survives the correction. That is reanalysis, not a blocker, and not
training.

### M4 — the CONFIRMATION screen over the 3024-unit census

*Commits* `b8865058` (predictor), `0de54534` (agent-multi execution body).
*Artifacts* the return, and the auditor's own reproduced probes in
`DAY_REVIEW_2026_09_26/{M4_PROBE.txt,M4_FINDINGS.md,m4_probe.py}`.
*Observed* `NO_NEW_MEASUREMENT`, 0 of 3024 units; neither `ADVANCES` nor `DOES_NOT_ADVANCE`, because
all sixteen cells of the Holm family read `UNDETERMINED_NO_OBSERVATION` and Holm is a step-down over
p-values that do not exist. Census `12cfd9ad78…` re-derived 28/28.

**Real dependency: six named verifier repairs, and not an approval.** Authority bound to the reviewed
implementation rather than a recorded HEAD; exact census with distinct seeds and full
family/noise/generator identity; raw authenticated evidence and numeric re-derivation before
`VERIFIED`; attrition and full population for *all* contrasts including the fifteenth; resume with
complete schema, arms, lineage, state and disjointness; cumulative accounting with no double
execution. Derived from the probes over `agent-multi@0de54534`, which name
`tools/m4_confirmation_runner.py:270,385,557,719,754,766,785` and
`tools/m4_confirmation_protocol.py:629` — not from the return's own claims.

And one dependency the executor must **not** supply: the external design-review record the gate
requires. Writing it would be the executor approving himself. It was not written, not installed and
not simulated, and this index does not schedule it.

**Next eligible work:** freeze the review's probes as PRE and turn them into tests of the real public
path; then the six repairs, with positives and negatives under DEVELOPMENT fixtures including the
full entrypoint; and correct the "zero arrays" sentence to what was actually checked, resolving the
declared disjointness-test exception by name. **M4 CONFIRMATION is not authorized and no unit may be
fitted.**

### FIN-LOSS-OPT/E3 — the financial factorial and the weekly cycle

*Artifacts* `RP82/FIN_LOSS_OPT_DESIGN_SEALED_v4.json`, `RP82/fin_cost_pilot/DESIGN.json`, and the
retained pilot record in the programme state.
*Observed* the v4 design is sealed and the cost pilot is `NOT_EXECUTED`: the governed ranged download
of the EURUSD 1h resource returned **HTTP 422, "resource availability contract required"** — the lake
declares no resource contract for it. 0 bytes were read, and campaign
`satoshi-fin-cost-pilot-20260921-prepare-data` was left open without delivery.

**Real dependency, FIN-LOSS-OPT: a registered resource availability contract for the EURUSD 1h
resource in the data-gov lake.** This was reported for days as a resource wait. It is now reachable
work, and naming the mechanism is the point of a dependency column:
`data_gov/resource_registration.py` and `tools/register_calendar_resources.py`, landed today at
data-gov `7eec868`, are exactly what registers one. The open prepare-data campaign must also be
closed at its normal boundary.

**Real dependency, MOD-E3: BUSINESS-CONTRACT executed, plus one governed forecasting result on the
domain it trades.** Every retained E1 measurement is a household-electricity panel and the only
price-series tables are 114 `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND`.

**H-CORE is not a blocker of this lane as a whole — the second correction the audit required.** The
disposition had added it as a global prerequisite. The programme contract lists MOD-E3's
prerequisites as `MOD-E1` and `BUSINESS-CONTRACT`; the master plan's own row for E3 requires its
evaluation *"aunque H1-H3 no sean positivos"*. H-CORE conditions E3's **RL comparison** and nothing
else. The state now carries `depends_on_by_subitem`, so the scope is visible rather than implied:
financial reference preparation needs `BUSINESS-CONTRACT`; the weekly forecasting protocol needs it
and `MOD-E1`; only the weekly RL comparison additionally needs `MOD-CORE-PRETRAIN`.

**Next eligible work:** register the financial resource's availability contract and re-attempt the
sealed cost pilot through the governed path — no scientific selection, no held-out access; and
prepare the weekly reference and protocol, which need no electricity result and must never use one as
a substitute for financial validation.

### calendar — economic calendar registration and the event-study clock

*Commits* data-gov `7eec868` and `8a5d2f9`, both pushed.
*Artifacts* `CALENDAR_REGISTRATION_2026_09_26/` (registrations, inventory, run, measured clock) and
`FRED_REGISTRATION_2026_09_26/registrations.v1.json`.
*Observed* registered and refusing, which is the correct outcome: five calendar resources registered
with their absences, 41 absence rows over 17 codes, replay `ALL_ROWS_RE_DERIVED`; nine of nine FRED
siblings registered, 0 rejected. Exactly one resource carries consensus and exactly one observed the
publication instant, and **their windows are 1,326 days apart**, so the consensus archive stands at
`study_refusal = CONSENSUS_WITHOUT_OBSERVED_PUBLICATION_CLOCK` and the catalog refuses the event
study. Upstream, the observed-clock study was deliberately left unregistered: its placebo passes 8 of
660.

**Real dependency: a consensus source that also carries an observed publication instant over the same
bar span.** The artifacts name it as a purchasing decision and name the providers. Nothing on disk
supplies it and no code can. Derived by comparing coverage between the two registered archives — the
1,326-day gap is measured, not asserted.

**Next eligible work:** the mechanical work that does not pretend to the missing identification —
prospective point-in-time capture, and extending the absence and placebo reporting over the newly
registered resources. `UNKNOWN` stays `UNKNOWN` and is never read as zero lag; and mechanical
improvement does not stop for want of the dataset.

### M5PHET — five families and the chat/NL interface

*Commits* M5PHET master `8a6d1a3` (today; an index that states it carries no measurement), with the
newest family work at 2026-09-25: `e2eddd6` chat/route, `92bb224` forecast, `790843f` regimes,
`6bcf631` causal, `f5b129f` classification, over providers in news-signal, prediction_provider,
feature-eng, causal-inference and agent-multi.

*Observed* **five families answer; one carries a quality number and four publish a refusal or a
null.** Classification macro-F1 0.37776, skill 0.25331 over majority class on 450 independently
labelled rows, ECE 0.13151 `UNCALIBRATED` — and a recorded finding that the same checkpoint used as a
*chooser* abstained on 28 of 28 corpora and ranked the same option first on all 28 distinct state
digests. Forecast: outer confirmation sealed `d4ac73f0…`, the searched family beats the hand window by
0.034412 kW with both intervals excluding zero, champion **ordering not** confirmed; served bundle MAE
0.565838, skill 0.151888; quantile bundle coverage 0.926 at nominal 0.95. Regimes:
`NO_NEW_MEASUREMENT` / `NOT_RANKED`, the served reference collapsing 10,080 holdout rows into one
cluster so the internal indices are `INDEX_NOT_DEFINED`. Causal: `NOT_IDENTIFIED`, placebo 8 of 660.
RL: `NO_NEW_MEASUREMENT`, abstained 256 of 256, `UNDERPOWERED 256/500`. Chat: route 81/95 = 0.8526
with `WRONG_AREA` 0 and `WRONG_TYPE` 12, interpreter 50/60 = 0.8333, one malformed envelope refused by
`check_proposal`; 634 passed, 1 skipped.

**Real dependency: one per family, and none of them is another family.** Classification: an
independently labelled corpus for any new task, plus the worker's external GPU for inference.
Forecast: rows untouched in every sense, which `nested_split.json` states do not exist inside the
50,400-row slice — so a different source file with its own provenance. Regimes: a fitted reference the
running instance actually serves, whose holdout has at least two non-empty clusters. Causal: the
calendar lane's missing consensus-plus-observed-clock source, an owner entitlement decision. RL: at
least 500 sealed rows against the pre-declared minimum, and a decision path that does not abstain.
Chat: nothing external; a wider corpus needs sentences nobody has written yet.

**Verified absent:** no cross-family hold is written anywhere in the M5PHET artifacts. The one
figure-scoped qualification — WP26 over WP06 stage 5 — was discharged on 2026-09-25 by the pointer it
required. The unmerged CL20–CL24 line carries `global_hold: false`.

**Next eligible work: six items that may proceed at the same time and in any order** — diagnose the
classification chooser's state-independence on one of the 28 retained corpora; a second-series outer
confirmation for forecast; refit and install one regimes reference with two or more clusters;
prospective capture for causal (WP32, the only work-plan item with no artifact at all); extend the RL
seal past 500 bars; repair the chat router's twelve `WRONG_TYPE` under-answers. Plus three bookkeeping
contradictions the artifacts already refute: M5PHET's own `PROJECT_METHOD_STATE.json` still says
`model_performance_measured: false` with four families "planned"; the README says 491 tests against a
634 receipt, says no area carries a measured quality number against WP09's 450 rows, and says no
shipped bundle has a quantile head against the shipped quantile bundle; and WP26's twenty fits ran on
the coordinator's own GPU against the plan's standing rule with no admission record — either the
exception or the admission is missing, and one of the two must be written.

### news — collection and shadow (news-signal)

*Commit* `b08a99f` (today, identical to `origin/master`).
*Artifacts* `tests/test_rp148_collection_boundary.py` (35 CPU tests), `PROJECT_METHOD_STATE.json`,
`docs/INSTALL_VERIFICATION.json`, `examples/eurusd/SEAL.json` (20 labelled files,
`sealed_before_any_scoring_run: true`), the recorded collection fixtures.
*Observed* boundaries implemented and tested; no live input, no broker evidence, quality
`NOT_MEASURED`. `real_weights_measured: false`, `governed_live_inputs: false`,
`broker_connections_accepted: []`, `real_capital_authorized: false`. The installed entry point answers
a user-authored question end to end with `backend_kind: NON_MODEL_FIXTURE` and
`inference_performed: false`.

**Real dependency: a licensed feed entitlement the package does not hold, plus data-gov registration
of what is collected.** Neither is code. Nothing here depends on the classification family's quality
number or on any predictor experiment.

**Next eligible work:** an isolated pinned-weight batch-1 smoke with cold-load RSS, warm latency and a
**retained receipt** — which also settles a contradiction worth naming: the README's prose claims a
real-weight GPU measurement dated 2026-09-24 for which no receipt JSON is retained, while the state
says `real_weights_measured: false`. Until the receipt exists the prose is unverified and the state
governs.

*Remaining cost:* **cannot be established from artifacts.** No numeric budget is declared anywhere in
this lane; the only bound is a 256-token per-record cap.

### MT5 demo — policy, risk and unknown-outcome accounting (lts)

*Commit* `12bce5f` (today), after `fcea94e` (RP149).
*Artifacts* `mt5_unknown_outcome_20260926/{NOTE.md,dry_run_recorded_corpus.{txt,json}}` and the
ruling document; at runtime, the watchdog's `latest.json` key `mt5` and `monitor_events`.
*Observed* interfaces and migration implemented over a recorded corpus; the live bridge is stale and
its store is not on this host. The dry migration ran read-only over a 9-row synthetic corpus built by
the committed test helper. The local default bridge database does not exist — re-verified here — but
the watchdog prefers a **remote** status endpoint, and that endpoint reports `succeeded 48 / failed 5`
with `execution_enabled: true`, against a heartbeat last received 2026-09-08 and now about 17.9 days
stale. `monitor_events`' latest entry is `mt5_bridge_stale`, severity critical, and it is the only
active event key in the fleet. Exposure reconciles at 0 orders and 0 positions, `all_authorized:
true`.

**Correction this index makes:** the retained note's sentence "this lane has never placed an order" is
true of the *local path only* and is not a lane-level fact. Derived by attempting to locate the
configured database and by reading the watchdog's own preference order in source, not from the note's
summary.

**Real dependency: a readable copy of the off-host bridge store** — the configured path does not exist
here and the tool exits "no such database" — then an operator attestation that the bridge is stopped
before any `--apply`. For a canary, separately: a confirmed demo account and the standing risk
mandate. **No per-order acknowledgement, retcode, ticket or fill is retained on this host**; only the
two aggregate counters, read live.

**Next eligible work:** dry-run the migration against a read-only copy of the remote store and publish
its result; reconcile the note's sentence with the remote 48/5; and the two named fix-forwards —
`reconcile_completed_lifecycles` still reading `state='succeeded'` from the column, and
`exposure_reconciliation` not adopting a reconciled order's ticket. **Do not restore a branch under
live timers.**

*Remaining cost:* **cannot be established from artifacts** as money or CPU. The only budget present is
domain-level: five daily entry slots reclaimed and now held, with `BUDGET_RELEASING_STATES ==
{"failed"}` — deny by default.

### Alpaca paper (lts)

*Commit* `889c5f2` (2026-09-25), whose work is carried in `12bce5f` — which is what the live
five-minute preflight has been executing since 2026-09-26 06:15 local.
*Artifacts* the two state databases, the runner heartbeat, the watchdog's `alpaca` key, the lab config,
and `tests/unit/test_broker_refusal_recovery.py`.
*Observed* live, read-only since `2026-09-25T16:03Z`, and **the only lane of the three with real broker
acknowledgements and fills retained**. Latest preflight complete: account `ACTIVE`, environment paper,
`trading_blocked false`, `equity_market_open false`, `open_orders 1`, `open_positions 1`,
`orders_submitted 0`, `protected_execution_eligible false`, quotes for six symbols, `missing_cells []`.
Retained venue facts: 27 `call_attempt` / 27 `submit_response` / 27 `ack_snapshot`, one effect
`acknowledged` and 26 `terminal_flat` with venue-issued order ids, 54 execution report receipts, and
67,695 rows carrying `filled_qty` and `filled_avg_price`. RP157's refusal surface is **live but
unexercised**: zero retained structured refusals in production, so it is evidenced by its tests alone.
And two code versions are touching the same paper account: the model runner has been resident since
2026-09-23 with pre-`9090f49` code while the preflight executes `12bce5f`.

**Real dependency: nothing for monitoring — the lane is already running.** For safe progress, the F3
remediation: a commit-pinned deployment with its own directory and pinned dependencies, a rehearsal, a
window and a rollback.

**F3 is confirmed here and the zero-impact claim is withdrawn.** Eighteen files differ between
`9090f49` and `12bce5f`, four of them in broker lanes under live timers, and the import chain is real:
the observer unit runs `run_alpaca_paper_preflight.sh` → `app.alpaca_paper_cli` → imports
`app.alpaca_paper_lab` at line 10 → which now imports `app.broker_refusal` at line 19 and wraps the
non-2xx branch. An identical CLI file does not make an identical runtime. Cross-checked against the
day review's own recorded `lts_timer_dependency_hashes` for the two commits.

**Next eligible work:** that commit-pinned deployment with rehearsal, window and rollback; adoption
tests for account shorting eligibility and close semantics; and a zero-network sink for shadow tests,
kept out of the paper-canary profile. **No mutating broker call by this order, and no service started,
stopped or restarted.**

*Remaining cost:* **cannot be established from artifacts.** No numeric budget is declared;
`orders.enabled` is false in the lab config; the shadow lane's reported NAV is simulated and is not a
budget.

---

## 3. The two maps, linked and kept distinct

The index carries both as named objects, so the relationship is readable rather than inferred.

**The M5PHET product map** holds the five families, the chat interface, news, MT5 demo and Alpaca
paper. It has its own work plan. It does not replace the doctoral programme or the financial
validation, **it does not certify usefulness, and it does not replace the experiments' own trials**.
Five adapters completing is five adapters. The links to the other map are stated as links and not as
dependencies: the forecast family consumes predictor and prediction_provider bundles and the WP26
outer confirmation is a predictor artifact; the causal family shares the calendar lane's missing
publication clock; the RL family's profitability number would need an execution record, which is the
broker lanes' business.

**The doctoral and business map** holds A/B, R0/R1/R2-ECL, E1-Q2-CONTEXT, H-CORE, M4, FIN-LOSS-OPT/E3
and calendar, governed by the master work plan and the metrics contract. The framework makes these
tasks easier to run and to serve; it supplies none of their evidence, and household electricity is
never a substitute for financial validation.

**A correction in one family holds no other family,** and the index makes that visible instead of
implied: each family's dependency is written in its own row; the verified absence of any cross-family
hold is recorded as a finding; and the six M5PHET next-work items are stated as concurrent, in any
order. Where I found a defect in one family today — the chooser's state-independence, the regimes
reference collapsing to one cluster — it is written where it belongs and nowhere else.

---

## 4. Dependencies that could not be established from artifacts

Five, all in the remaining-cost column, all said in those words in the row itself: **news**
(no numeric budget declared; a 256-token per-record cap is the only bound), **MT5 demo** (no money or
CPU budget; only a domain-level five-slot daily entry budget), **Alpaca paper** (none declared; the
shadow NAV is simulated), **M5PHET** (no CPU-second ledger in the family artifacts; a 175-second
per-call wall timeout is the only bound) and **calendar** (none declared in the registration
artifacts).

No row's *dependency* itself was left unestablished. Two rows name a dependency that only the owner
can discharge — the calendar's consensus-with-observed-clock source and MT5's demo account plus risk
mandate — and both are named as entitlement decisions rather than presented as work.

---

## 5. The `df_e1_seal` consumer, reconciled

`tools/df_e1_seal.py:655` returned `OUTSIDE` for `MOD_E1_EXTERNAL_REVIEW` as a literal. A constant
cannot be wrong about the world, and that is exactly the defect: the programme state came to declare
the requirement discharged while the consumer went on denying it, and a reader had no way to tell
which of the two was stale.

The row is now derived, in this order, from bytes. Do either of the two reviews that were never
written exist now? If so the requirement is back with its reviewer and the state returns to `OUTSIDE`
— this module does not read a review's content as acceptance and never will. Otherwise: are the two
standing-in dispositions retained, each declaring in its own text the authority it acts under, that it
is not written in the reviewer's name, and the requirement it rules on? Their identity is recomputed
here from their own bytes. And does the programme state name exactly those documents, exactly those two
absent reviews, and counts that the documents themselves print?

The state it yields is a fifth one, and deliberately not "satisfied":
`DISCHARGED_BY_OWNER_GRANTED_DISPOSITION_NOT_BY_EXTERNAL_REVIEW`, carrying the scope it reaches — module
dispatch, on the four modules the ruling names, at the commit it names — and the scope it does not: the
reviewer's signature, any measurement, and MOD-CONF's sealed confirmatory design, whose owner the
ruling itself leaves as Musashi + Satoshi. **It remains a gap in `gaps`, so no bookkeeping change can
promote `E1_PARTIAL_SEAL` to `E1_SEALED`.** The strictly smaller `dispatch_blocking_gaps` is the list a
dispatcher may act on: five gaps, four of them blocking.

Absence and contradiction are now different things. **Absence** — no ruling retained, a retained ruling
the programme never declared, no programme state at all — is answered with `OUTSIDE` and a named
remedy. **Contradiction raises:** a declared discharge whose documents are gone, one naming other
documents, one standing in for a different review, counts no disposition prints, a ruling that hides
its authority, one whose text drops the sentence disclaiming the reviewer's name, and one signed in his
name. Nine refusals, each for a real reason; nothing is refused by default. Forty tests pass.

And the caveat that belongs beside all of it, written into the tool's own output: `gaps` and
`dispatch_blocking_gaps` answer two different questions and are never merged, and **neither is an
authorization** — a documentary check that passes says nothing about whether a measurement is
scientifically admissible.

---

## 6. What this round did not do

No model was fitted, loaded, scored or replayed. No training was run, no allocation taken, no GPU
used — `CUDA_VISIBLE_DEVICES=''` throughout, every heavy step under
`crispdm-run -m <MEM> -t <WALL> -n dr02 --`, anaconda env `trading-stack`. No service, timer or
container was started, stopped or restarted, and the live memory-gate child was left to its own
boundary. No broker call was made. No hostname, address, account identifier or secret was written. No
personal or untracked file of the owner's was cleaned, staged or committed: this work was done in a
fresh pinned worktree, and the primary checkout's untracked files were left exactly as they are.

One thing is named rather than claimed as finished, and one thing changed under me. The
`E1-Q2-CONTEXT` attempt was alive and held by the admission guard when the first eleven rows were
written; it ended at its own boundary at `19:31:55Z` and the row now carries its closed state and its
refusal. I did not kill it and did not relaunch it. Nothing else in the programme holds a live lease:
`crispdm-rp135-continuation-20260923.service` reads `inactive`, and no `df_` process is alive.

And no measurement of any kind was produced by this round: **NO_NEW_MEASUREMENT**, by design — DR02
asked for bookkeeping that matches the world, not for another number. The 522.833 CPU s and the twelve
cells in the `E1-Q2-CONTEXT` row are that block's own prior cost, not this round's.

The four module rulings, the recovery and the M4 execution body remain the auditor's to accept or
withdraw. This index records them as the rulings that currently govern dispatch; it does not record
them as approved.

— **Satoshi III (Mujuro Utsutsu), successor technical lead**, 2026-09-26
