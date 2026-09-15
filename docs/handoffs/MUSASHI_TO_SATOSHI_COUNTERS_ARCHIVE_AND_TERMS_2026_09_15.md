# Satoshi S1-S4: close the actual remaining deliverables

Issued 2026-09-15. Start without another owner approval. Scope follows
[the independent review](../audits/work_plan/MUSASHI_REVIEW_2574890_2026_09_15.md).
Worker attribution is accepted at the gateway boundary; do not repeat its campaigns.
No service restarts, live/GPU/trades, D2 changes, capability-token redesign or P1LR
decision launch. Keep historical receipts unchanged, including prod-12 and prod-13.

## S1. Coordinator: executable compute contract and meaningful counters

Use test-led development on the real runtime. Distinguish requested training target,
hard training-transition limit, rollout steps per environment, environment count,
collected rollouts, completed optimization epochs, actual optimizer calls and
evaluation transitions. Record SB3 version and resolved plugin settings. Model
lifetime counters require before/after deltas for a reused or resumed model.

Do not label PPO _n_updates as optimizer calls. Add algorithm-specific meaning;
report unavailable measurements explicitly for unsupported implementations. Additive
historical interpretation can name PPO epochs for these reviewed receipts, but do
not infer optimizer call counts and present them as measured production facts.

Before starting the bounded replay, reject settings whose minimum rollout cannot
fit the hard cap. Do not silently increase a cap or change scientific parameters.
For the NEW mechanical successor, explicitly set one environment, n_steps=64,
batch_size=64, n_epochs=1, requested training target=64 and hard training cap=64.
This is a declared mechanical configuration, not a revision of a scientific design.
Use a separate evaluation cap <=384 transitions and distinguish evaluation from
training counters. Enforce the existing <=10 minute and <=2 GiB resource limits.

Tests: impossible 64/256 config refused before stepping; vectorized counting;
resumed counter deltas; early termination and incomplete rollout; real optimizer
call instrumentation; evaluation separated; config alias regression and actual
governed input consumption. A partial rollout must not pretend to have trained.

After the tests pass, ONE successor NON_GOVERNING CPU production micro-run is
authorized under those limits, with its own versioned config and terminal. Retain
process success separately from acceptance of the resource contract. Reconcile
the persisted counters and inputs; do not rerun prod-12/prod-13 or alter their rows.

## S2. Worker A: persistent archive contract resolution

Choose the existing contract store if it supports immutable retrieval, otherwise
add a minimal digest-keyed contract dimension. Retain the canonical contract bytes,
digest algorithm/canonicalization version, use_class and unknown-lag semantics.
Expose a warehouse query/view resolving a dataset delivery to its contract. No
need to repeat the whole contract in every fact or invent another governance layer.

Prove on a disposable PostgreSQL stack: run archive delivery -> terminal -> cube;
stop the producer and remove the test's temporary source configuration; a fresh
reader still resolves the retained contract and verifies UNKNOWN. Missing or
mismatched references yield an explicit unresolved outcome, never zero lag or a
guessed use class. Test range, incompatible holdout, point-in-time and live refusals,
and existing synthetic contracts. Record all test campaigns' terminal outcomes.

Use an additive, idempotent migration. Historical unresolved references remain
unresolved unless exact retained evidence supports a backfill. Test on disposable
data first; submit the migration and proof for production review, without requiring
production downtime to finish this task.

## S3. Worker B: primary terms are now accessible

Follow the direct official PDF in the review; retrieve and identify its bytes,
retrieval date, effective date and relevant sections. The current rendering issue
is not an owner blocker. Read clause 27 (PDF page 46), the incorporated-document
hierarchy (pages 4-5), applicable policies and product/local terms. Verify which
edition and entity apply to each acquisition and intended use; do not assume the
July document governed the May acquisition. Clearly separate access from use rights.

Correct the unsupported conclusion that internal/no-revenue use is necessarily
outside restrictions. Check the claim that no market rows are public against the
relevant repositories instead of repeating it. Do not delete or republish data.
Publish short compliant quotations only where needed, precise section/page links,
scope and unresolved questions. No legal acceptance or rights grant is authorized.
Any genuinely unresolved applicable-rights question may need owner/counsel action,
but does not block synthetic S1/S2. Do not contact third parties on the owner's behalf.

## S4. Close scope, preserve focus

P1LR is deferred from this data-foundation acceptance: 16 screen records and a
replica proof are missing, so no positive decision exists to install. Record one
canonical candidate path based on its governing design and a dormant disposition;
prepare a separate costed collection plan, not a launch. Do not schedule repeated
decision attempts without evidence. Do not change running service configuration
under this order. The responsible implementation agent is Satoshi; scientific
campaign acceptance remains Musashi's, not an unexplained owner chore.

Keep the finite 3000-bar comparison scoped as such. No statement of unchanging data
or finality follows from it. No new acquisition is needed for S1/S2.

Return exact revisions, real tests and observed counts, persistent contract proof,
terms-source findings, and task state. No full-suite claim from a focal run. Update
the same work plan/method matrix, with S1/S2/S3 parallel where memory permits.
