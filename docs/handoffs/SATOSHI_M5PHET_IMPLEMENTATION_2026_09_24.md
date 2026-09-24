# M5PHET: implementation assignment to Satoshi

Latest candidate `d9ffc4d`: apply the
[independent POST review and continuation](../audits/work_plan/MUSASHI_M5PHET_D9FFC4D_REVIEW_2026_09_24.md).
Accepted repairs remain; exact forecast coverage, canonical population handling
and writable torn-tail recovery need narrow fixes. Real Laya integration and
independent experiments continue in parallel; no new owner permission required.

Earlier candidate `7385937`: retain the
[specialized-provider review and scoped continuation](../audits/work_plan/MUSASHI_M5PHET_7385937_REVIEW_2026_09_24.md).
The latest review above qualifies which earlier cases are now accepted.

Candidate `adba176` is reviewed: execute the scoped
[integration continuation](SATOSHI_M5PHET_INTEGRATION_CONTINUATION_2026_09_24.md)
alongside this assignment. Implementation exists but acceptance is not complete;
do not stop valid independent experiments for these software repairs.

Owner's upstream-reuse clarification: read the
[Laya architecture/reuse and fork decision](../audits/work_plan/MUSASHI_LAYA_ARCHITECTURE_2026_09_24.md)
before implementing equivalent provider features. Upstream 0.3.18 already has
schema helpers absent from our 0.3.11 pin. Prefer the SDK, not a rebuilt engine;
upgrade only after scoped compatibility tests, preserving our strict boundary.

Owner-approved implementation addendum to RP150, not another research proposal.
Satoshi owns implementation and integration; Musashi owns design review and
independent adversarial acceptance. Do not return only diagrams, scaffolds or
additional plans. Read M5PHET design revision `4508b98`, especially USE_CASES,
INTERFACES, ECONOMIC_CALENDAR and IMPLEMENTATION_PLAN (P01-P09, CAL01-CAL12).
Reconcile newer commits and running work first; preserve completed RP144-RP151.

## Approved plugin architecture: specialized ML, optional language models

Organize providers by task/capability, not by model size or an assumption that
every task needs a generative LLM. Classification, forecasting, representation,
RL and causal inference have their own typed contracts and configurable engines.
One provider may support multiple contracts, but only for declared, tested
combinations of operation, input and output semantics. Do not infer support from
the Cartesian product of independent capability lists.

Laya/Jev are candidate interchangeable classification providers only where their
capabilities match; an unavailable remote provider must not block a working local
one. Reuse their supported SDK/API rather than reimplementing their networks.
An optional language-model provider can interpret requests, generate explanations
or contribute declared features when a concrete use case needs it. It is neither
a mandatory orchestrator nor authority to invent measured outputs, change a
scientific task, execute code or authorize broker actions. No new general-purpose
LLM subsystem is required for this delivery.

Compose providers through explicit typed edges (for example event classification
-> forecast inputs -> policy observation), retaining as-of times and source/model
identities. Reusing a text representation in another family requires a compatible
adapter and its own evaluation; it is not automatic evidence for that task.

Acceptance: local specialized inference works with no LLM service/key installed;
configuration selects the named provider without hidden fallback; incompatible
provider/operation/output combinations refuse before loading; a provider swap
preserves task and evaluation population while recording changed model identity.
Use real available engines for integration evidence. Keep governance, local OLAP
and DOIN independently optional. Implement this in the existing provider registry,
not a parallel plugin system. These refinements do not restart current experiments.

## Method and parallel ownership

Apply the existing test-led, data-centric method, per bounded component:
requirements/use cases -> behavioral acceptance tests -> architecture/system
tests -> component/integration tests -> unit tests; then implement bottom-up,
verify integrations and exercise the real consumer. Specify each component's
tests before its implementation, not after observing its outputs. Freeze PRE
failures and independently derived expected results. Test structural bypasses
AND behavior; document-only tests do not prove a provider works.

Run disjoint lanes in separate worktrees with explicit owners: framework/provider
runtime; economic data/features; domain adapters; existing experiments. Delegate
bounded jobs to available Hermes agents when useful, with one integration owner.
Freeze shared interfaces before concurrent implementations. A missing dataset or
provider blocks only its dependent work; continue every independent runnable job.
Use short progress updates, not requests to continue at each step.

## Code deliveries

1. **M5PHET runtime.** Implement versioned request/result validation, provider
   capability checking and explicit fit/calibrate/infer/evaluate boundaries.
   Register external providers through the designed entry-point group, reject
   duplicate names, unsupported semantics and incompatible state before loading
   a model. Preserve distinct statuses and uncertainty types. No success stubs.
2. **Typed decisions.** Reuse Laya's pinned SDK through news-signal: implement
   supported choice, binary and ordinal mapping with complete-population checks
   and independently recomputed ordinal arithmetic. Test malformed/missing
   responses and no-order authority through the real consumer. Add hierarchical
   composition only with a declared tree and parent/child consistency tests.
   Existing choice clients must keep working; update dependency pins only after
   cross-repo integration tests. Do not rebuild Laya or require paid Jev access.
3. **Economic calendar.** Inspect the existing governed resource, then implement
   vintage-aware as-of features in its owning data/feature repository. Exercise
   CAL01-CAL12, including consensus frozen before public release, late receipts,
   revisions, missing values and unknown historical availability. Source numbers
   from records, not language-model guesses. Use deterministic fixtures while
   resolving source gaps, but label them and require a real governed sample for
   data-integration acceptance. Never fabricate missing clocks or entitlements.
4. **Domain providers.** Implement thin adapters to existing engines for
   hierarchical market-state representation, multi-horizon uncertain forecasts,
   and the actual RL observation/policy path. Add a specialized causal-analysis
   adapter with explicit identification before estimation. Reuse suitable OSS
   implementations; inspect their actual APIs and licenses. Unsupported output
   types refuse explicitly. Each claimed implemented family needs a real engine
   round trip and a negative control, not just a schema or mock. Small synthetic
   known-answer integration cases are software evidence, not financial findings.
5. **Composition and packaging.** Demonstrate calendar features reaching the
   actual forecasting and RL consumers, with clocks, axes, units, missingness and
   identities preserved. Keep causal studies separately identified. Publish
   installable packages, runnable examples, scoped tests, status and dependency
   pins. No duplicate brokers, optimizers, storage hosts or scheduler.

## Storage and DOIN integration, cross-cutting implementation

Read M5PHET `docs/INTEGRATION_AND_OPTIMIZATION.md` and implement INT01-INT12
alongside the first provider, not after all five families. A local community use
case must run without mandatory services; our governed campaigns use data-gov,
data-lake and data-warehouse through existing contracts. Preserve identical task
semantics but distinguish local provenance from accepted governance. A denied
governed delivery cannot silently fall back to a local file.

Owner clarification: governance is optional for EXAMPLE APPS too, and independent
of DOIN. Ship a local-first example with atomic manifest, durable attempt/event
records, typed metrics and artifact references. Add optional embedded DuckDB
analytical views as a rebuildable local projection, no server required; no new
warehouse host or production cube. Use the same task code for the opt-in governed
example. Recover interrupted writes, preserve metric unit/population/definition,
and test profile parity and retry idempotence. Historical local import cannot
retroactively create campaign or delivery authority. These examples do not relax
the profiles of already-sealed internal governed experiments.

Reuse `doin-core` OptimizationPlugin, InferencePlugin and SyntheticDataPlugin and
their actual entry-point loaders. The evaluator returns a scalar, not a structured
prediction or Pareto vector; seal objective/direction/constraints and retain all
metrics separately. Declare typed/conditional search spaces per family, including
empty spaces for fixed engines. Preserve data/holdout/clock/target contracts and
causal identification; do not search for significance or rewrite risk constraints.
Synthetic consensus verification and real-domain scientific evaluation remain
distinct. Start with one real candidate plus independent evaluator and governed
receipt/outbox reconciliation; extend objective adapters per implemented family.

Avoid new stores, arbitrary SQL credentials, tick-by-tick synchronous warehouse
dependencies or unlimited artifact retention. Verify actual provider upload and
registration capabilities before using them. Keep integration dependencies optional
and isolated. No changes to production databases or consensus policy by assumption.

## Acceptance and execution limits

Exercise actual entrypoints with changed future data, wrong model/calibration,
axis/horizon permutation, restart/reload and deliberate invalid provider output.
Require measured optimizer updates when a test claims training. Fitted state and
scalers must be train-only. Verify both end-to-end positive and refusal paths.
The shipped classification contract remains distinct from newly implemented
families; incomplete adapters stay explicitly incomplete.

Reuse standing lane budgets, resource admission and single-executor leases. The
external 5090 is first GPU choice; keep admitted independent experiments running.
Measure new work's cost before dispatch; no unbounded tuning or model downloads.
Do not touch sealed scientific configurations or final reserves. Implementing an
adapter does not authorize an unregistered financial campaign. Shadow, MT5 demo
and Alpaca paper limits remain; no real-capital promotion or form submission.

Return once all independently executable assigned work is complete, with per
delivery IMPLEMENTED/INTEGRATED/VERIFIED or the exact unmet object. Include test
commands, installed revisions, real-provider evidence, remaining consumer gaps
and running experiments. Scientific tables must give metric/scale, same-row naive,
matched literature reference and cost; label NO_NEW_MEASUREMENT if none occurred.
No claim that passing software tests proves calibration, causality or profit.
