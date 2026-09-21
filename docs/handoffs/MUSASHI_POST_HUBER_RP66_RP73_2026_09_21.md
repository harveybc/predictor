# Satoshi RP66-RP73: correct the instrument, then measure a matched reference

Base: `4ef9f71`. Incorporate Musashi's post-Huber review, evidence and this order before work.
Owner authorization already given. Complete all independent blocks without asking to continue
after each one. One final return, with progress updates as needed. No service restart, live trade,
reserved confirmation, historical deletion or bulk retraining is part of this order.

## Goal and execution scope

Answer a useful next question: on the SAME household forecasting task and evaluation rows,
how does our modular model compare with an independently implemented literature-derived GRU,
and what part of any remaining error is associated with missing input information or history?
In parallel make the financial MAE/Huber x Adam/AdamW experiment actually runnable; electricity
does not select its winner. Preserve the later forecasting/RL/pretraining hypotheses and priorities.

This order supersedes the previous preparation-only restriction ONLY for the bounded household
DEV execution below. The finance scientific campaign and the full last-year paper reproduction
stay closed pending their concrete design review. Financial synthetic acceptance is authorized.
Budget: 14,400 aggregate CPU seconds including tests/pilots/closure; reserve 2,000 for closure.
No new owner approval for implementing tests/runners, sealing the prescribed corrections or
starting the permitted DEV block after its acceptance passes.

## RP66 - Complete result evidence, not a receipt badge

Freeze the supplied probes against the base before changing governing code. Repair closure at
the actual CLI: registered design -> expected units/roles -> attempts/terminal payload -> artifact
digests -> arrays and source DATA -> independent metrics -> accounting and warehouse content.
Use existing verifiers, not another path that trusts prior verified=True. Mandatory finite,
nonempty, exact-length arrays; unique expected origins, horizon and target identities, scaler
identity and units. Reject changed/missing arrays and missing/mismatched warehouse terminals.
NO_NEW_MEASUREMENT describes the round, never permits an expected forecast to disappear.
Preparation/AE/cost-pilot exclusions must come from the registered unit role, not filename guesses.
Make model/naive population, horizon and scale mandatory produced fields, not optional test-only
keys. Reverify retained history WITHOUT fitting. Write a successor table; preserve the old table.

## RP67 - Contract validation must constrain execution

Replace metadata-only require with typed contract and execution binding. Validate identity and
version of delivered data, target construction, time units, horizon mapping, split row identities,
missing policy, permitted features, metric formula/scale/aggregation, train-only scaler parameters,
and declared varying factors. Recompute inner and outer digests. Validate against runtime data
preparation, not just one JSON against another self-declaration. A stale digest and a consistently
re-digested but foreign task must both refuse before fit through the real runner.

Separate PLANNED_REFERENCE/protocol readiness from VERIFIED_COMPARATOR/result evidence: a pilot
may execute the reference to obtain evidence; it must not already claim a matched score. Remove
the bare reexecuted_on_our_rows boolean as authority. No matching unknown placeholders as proof.
No scalar units change can remain REPRODUCTION. Refuse unsupported affine conversions, invalid
sigma, NaN/inf and inconsistent physical times. A deliberate input/model contrast is allowed under
one declared estimand with common target/evaluation; do not require all arms to be identical.
Update the coverage inventory honestly. Apply checks to every newly used route; do not refactor
35 unused legacy routes just to delay these campaigns, and do not launch any unchecked route.

## RP68 - Repair the ML design and real feature paths

Seal a successor, preserving the 27-cell draft. Correct the clamped support to its measured 67
samples. For an exact same-information control, crop the actual raw input to the common 60 rows
before the extractor; measure gradients and perturbations at the full graph. If retaining a
67-sample arm, name it as extra context, not a null. Test padding effects and paired weights.
Keep the full long/short x shallow/deep contrast identifiable, not silently conflated.

For volume, freeze evaluation origins and a COMMON train-only input/target scaler from the shared
baseline train population for the primary volume-only contrast. Record a refitted-scaler variant
separately if desired; never compare MAE_z with different denominators in the same row. Count
unique support, train-only rows, targets, windows and exposures from actual identities, excluding
validation from train counts. Use validation cadence and patience in observed optimizer updates,
held fixed across volume arms; equalize checkpoint-selection opportunities. Budget exit remains
censored regardless of where the best epoch occurred. An extension is a predeclared new budget
tier, not a claim of known convergence or a guaranteed error improvement.

Implement calendar, lag, crop, enumeration and scaling in the actual future runner. Existing tests
that calculate a lag with local numpy inside the test are insufficient. Exercise observed
availability when present; UNKNOWN is not zero delay. Household remains retrospective/offline,
not evidence of live availability. Test nonfinite rows, time gaps, future-value perturbations,
future-label perturbations, duplicates and held-out changes through the consumer. Derive one
common admissible evaluation mask before scores. Calendar control must be deterministic from
its declared construction and stable under prefix extension; document finite-sample chance
association rather than guaranteeing "no information" merely because it was shuffled.

## RP69 - Literature comparator, faithfully adapted and measured

Correct TCN n_H=50 (unsupported in Table 4); trace every implemented layer, activation, output
strategy, regularizer and training choice to the primary paper or an explicit adaptation.
Do not call the paper fully specified. Distinguish raw/resampled samples from enumerated windows;
unavailable exact dates, optimizer or imputation details remain limitations, not invented facts.
Published Table 5 values stay in source notes, never the current comparable-number column.

First priority: independently implement the Table-4 GRU-MIMO family (one GRU layer, 50 units,
L2 0.0005, dropout 0), ADAPTED to our one 60-minute target and the same permitted inputs as the
modular arm. This is literature-derived, not an exact paper reproduction. Declare its final
readout/activations/regularized tensors and initialization; test parameter graph, target alignment,
learning, early stopping/restore and fresh-process replay. Use the common MAE+Adam continuity
recipe and common monitor/budget in the matched architecture comparison, explicitly labelled an
adaptation from the paper's objective. A native-recipe variant must be a separately costed factor.
Include persistence, daily seasonal persistence (where available) and a train-only constant
reference on identical rows; they are distinct baselines, not one ambiguous naive.

Prepare the paper's original 15-minute/day-ahead protocol as a separate lane, with its native
metrics, without reading our reserved test. It is NOT a prerequisite for obtaining a useful
matched-task reference now and not authorized for a full reserved evaluation in this order.

## RP70 - Financial task and Huber tuning that respect small errors

Interpret 6h/72h as elapsed-time horizons for this declared business task. Map targets by timestamp
and availability, not six/seventy-two next retained rows; if the exact target bar is absent,
exclude with reason and count the resulting population. An event-time horizon would be another
task, not a silent replacement. Test weekend/holiday gaps, missing intraday rows, incomplete bars,
timezone uncertainty, release cutoffs and label maturity. Do not invent producer availability.
Derive DEV week identities and reserve boundaries explicitly; 2025+ remains untouched.

Derive residual-scale candidates from those same admissible train pairs. Keep full precision,
strictly positive finite Huber deltas; flat/empty/short/zero residual scales have explicit fallback
or nonidentifiability rules declared before outcomes. No six-decimal rounding of candidates.
Freeze actual width/layers/features of both financial receivers before the cost pilot, not
"chosen at pilot". Enumerate the candidate allocation across loss, optimizer, LR, decay and delta:
equal tuning opportunities and observed budgets, default arms separate, no tautological budget
assertion. State what precision is resolvable by inference and what survives aggregation.

## RP71 - Replace the five placeholders with real acceptance

Implement the governed financial runner and executable FL01-FL08. Replace unconditional
NotImplementedError xfails with calls into that runner. Prove fold-change isolation of scaler,
delta and selection; governance before reading data; observed updates; min_delta/checkpoint
resolution; block uncertainty; and 1e-5/1e-6 preservation through real terminal/warehouse paths
on clearly labelled synthetic acceptance fixtures. Include negative controls that fail if these
properties are removed. Do not treat creation of the runner file as proof of any property.
Use bounded synthetic fixtures; no financial scientific selection or heuristic-strategy rerun.

## RP72 - Finish a bounded, genuinely comparable DEV block

Once RP66-RP69 acceptance passes, seal exact population and cost pilot settings before outcomes.
Primary execution block: three paired seeds of the modular continuity model and the adapted GRU
on the common W60/h60 task, inputs, split, transform and monitoring schedule. Reuse a preserved
modular fit only if its complete scientific/training contract matches; otherwise run the matched
pair, and explain the necessary difference. Never re-train just to repair a receipt.

Cost pilots use a declared subset inside train; they cannot read validation performance to pick
arms. Measure seconds/update, validation overhead, peak memory and closure costs, including all
children. Freeze training caps and checkpoints from the adequacy design. If the primary block
fits the remaining budget, execute it and reconcile before expanding. It is authorized here;
do not stop merely to ask permission. If the whole block does not fit, report measured limits,
finish all other tasks and deliver a costed successor; do not drop seeds after scores.

Then execute complete phase-2 question blocks only if their pre-score projections fit, in fixed
priority Q1 calendar, Q2 context, Q3 volume. Register each block before any scientific child and
preserve paired controls. Unrun blocks stay explicitly NOT_EXECUTED, never "no effect". Do not
weaken adequacy, change margins, trim cells or pick architecture/epochs from partial outcomes.
The long-window factorial is not a prerequisite for finishing the first matched reference.

Use the three available hosts for independent bounded work once the existing wrapper records
parent+child CPU and peak memory on each. Keep paired arms within seed/host blocks; matching
packages and observed numeric scope required. No arbitrary worker count and no GPU requirement
for these small models. A host that cannot meet the resource contract gets no scientific child;
finish independent work elsewhere and state its measured constraint, not a permission request.

## RP73 - One complete return, readable and independently verifiable

Publish PRE/POST, exact revisions, tests/skips, observed CPU per host, complete ledger, amended
designs, consumed sources, remaining objects and next actions. Update master/state per block,
not only conversation. Preserve prior errors with dated corrections. No deployment needed.

Owner-facing table is mandatory in the message itself: task/horizon/split; metric and fixed scale;
model error; naive on same rows; relative improvement; reference error; source and whether it is
PUBLISHED or OUR MATCHED REEXECUTION; comparability and uncertainty. Lead with common MAE_z for
this household comparison, retain native units only as a separately labelled supporting view.
Full precision in arrays and warehouse. A missing reference cell says NOT_COMPARABLE with the
specific missing object; no unrelated paper number. Label reused rows and NO_NEW_MEASUREMENT
where applicable. Do not claim all-domain enforcement, exact reproduction, economic gains,
convergence or a financial recipe winner from this bounded development block.
