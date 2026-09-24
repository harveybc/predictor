# M5PHET: implementation assignment to Satoshi

Owner-approved implementation addendum to RP150, not another research proposal.
Satoshi owns implementation and integration; Musashi owns design review and
independent adversarial acceptance. Do not return only diagrams, scaffolds or
additional plans. Read M5PHET design revision `4508b98`, especially USE_CASES,
INTERFACES, ECONOMIC_CALENDAR and IMPLEMENTATION_PLAN (P01-P09, CAL01-CAL12).
Reconcile newer commits and running work first; preserve completed RP144-RP151.

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
