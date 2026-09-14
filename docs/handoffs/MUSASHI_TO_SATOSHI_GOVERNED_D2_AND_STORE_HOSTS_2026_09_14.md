# Satoshi: governed D2, extractor compatibility and reusable store hosts

Date: 2026-09-14. Owner: Harvey. Coordinator: Musashi.

## Decision and reading order

The owner authorized updating the three governance services and accepted the
integration work. No further owner permission is needed for the bounded CPU
work below. Permission to run does not change a scientific verdict.

Read these before editing:

1. [Production deployment receipt](MUSASHI_N3_PRODUCTION_ACCEPTANCE_2026_09_14.md).
2. [Your completed return, preserved at its original revision](https://github.com/harveybc/predictor/blob/182bf89fa0754e3b7b2cea208621531331af931e/docs/audits/work_plan/SATOSHI_GOV_N1_N8_D2_R1_R8_RETURN_PACKET_2026_09_14.md).
3. [Verified data-gov README](https://github.com/harveybc/data-gov/blob/ff4503a4c7f1d950e47ab43894f176962025faa6/README.md) and [integration examples](https://github.com/harveybc/data-gov/blob/ff4503a4c7f1d950e47ab43894f176962025faa6/docs/INTEGRATION_EXAMPLES.md).
4. [Store-host architecture and migration tests](https://github.com/harveybc/data-gov/blob/ff4503a4c7f1d950e47ab43894f176962025faa6/docs/STORE_PACKAGES_DESIGN.md).
5. [Warehouse implementation now on master](https://github.com/harveybc/predictor/tree/a7a86e906fd5d4e4057237f014f8f95d0cf6ea6a/olap/lake), [financial adapter](https://github.com/harveybc/financial-data/tree/cc0f15e6ef9cf79c1fb47bd7d7731eb349831008/lake), and [README standard](../README_STANDARD.md).

Use isolated branches based on the relevant reviewed code. Do not replace a
research branch with the publication branch: only selected runtime components
were published. Preserve other agents' uncommitted work.

## Priority 0: finish the current experiment evidence

### A. Governed support re-adjudication

Consume the conserved D2 observations through the deployed governance API.
Register the evidence resources with their actual artifact format, hashes and
usage; do not invent a temporal contract for an aggregate report or label it
as raw observations. Use the current generic runner when its contract fits.
If an evidence format needs an adapter, specify and test that narrow adapter
on disposable fixtures, then run it. Do not use this as a reason to wait for
the proposed new repositories.

Run the repaired adjudicator once over the declared population. Retain the
original 3,972 units and 3,591 decisions unchanged; publish a successor with
the complete before/after table, applicability, planned/observed/complete seed
counts and exclusions. Verify population equality and report every changed
decision. The previous preview of 138 changes is a comparison target, not an
instruction to manufacture that count. Any discrepancy must be explained.

Record inputs, executed code/configuration, costs and outcome through Flow v3.
Reconcile the outbox and warehouse. No result is complete merely because it
exists in a local CSV. Failed and inconclusive outcomes are included.

### B. Bounded cross-CPU SNR diagnostic

Use the already frozen R4 subset: 16 regimes / 32 units, including AT9,
non-iterative controls and nearest-threshold cases. Preserve its selection
rule and all existing tolerances. Limits remain 6 aggregate CPU-hours,
4 hours wall time, 2 GiB per process, one process per available host and one
BLAS thread. Check live memory before assignment; a busy or unhealthy host
is not a reason to overcommit another machine.

Compare same-host and cross-host results, software/numerical-library versions,
absolute errors and distance to every relevant decision threshold. Separate
bitwise reproduction, tolerance compliance and decision stability. A stable
decision does not make a failed numeric tolerance pass. Do not adopt 0.006 dB
after seeing it. Submit any proposed future tolerance as a separate design
change with evidence, not a rewrite of the present result.

### C. Coverage and disposition

After A, apply the already rehearsed additive coverage views through the
governed reporting route. Preserve all historical matrices. An explicitly
selected run-plus-code identity defines the current view; historical rows
must not silently inflate its denominator. Verify counts and reconciliation
against the deployed warehouse. No database reset or loader restart.

D3 remains a separate design review, not an implicit consequence of this
deployment. Keep signal causality tests, missing-data regimes and calibration
limits explicit. No universal SNR or denoising-success claim is authorized.

## Priority 1: remove the extractor API mismatch

Choose a forward port to the current target-plugin API, not a downgrade of
predictor solely to make an old example execute. First reproduce the mismatch
between feature-extractor and `stl_preprocessor` in an isolated environment.
Specify the expected preprocessing outputs, target alignment, partition
boundaries and effective configuration before changing the adapter.

Then implement the smallest compatibility change and prove it with:

- Unit tests for shapes, timestamps, target alignment and missing inputs.
- Future-tail changes that cannot alter earlier fitted/transformed output.
- Installed-package tests, without manipulating `sys.path` to hide conflicts.
- A governed synthetic CPU micro-run with fresh outputs, declared seed and
  generator identity, artifact hashes, metrics and one reconciled outcome.
- Failure/refusal and duplicate-report tests; no stale artifact may count.

Do not claim the full feature-extractor algorithm is causally sound because
its wrapper works. Identify each untested operator, especially centered or
whole-series decomposition. Keep such operators outside scientific use until
their own online/prefix-invariance tests pass.

## Priority 2: reusable lake and warehouse hosts

Develop `data-lake` and `data-warehouse` as separate service hosts following the
linked architecture. `financial-data` and `predictor` provide external plugins;
configuration selects an installed entry point and distribution, not a repo
name treated as an import path. Configuration handles ordinary differences;
new plugins handle genuinely different behavior.

Start with requirements, acceptance scenarios, system contracts, component
interactions and unit tests; then implement bottom-up. Keep a persistent stage
file and requirements-to-test matrix. A context reset must not erase the
current stage, failing test or next action.

Deliver in this order:

1. Backend interfaces and explicit capabilities, with unchanged governance
   HTTP behavior. No duplicated policy/accounting kernel.
2. Two installed generic hosts and independently packaged fixture plugins.
   Use unique `src/` namespaces and wheel-based tests in clean environments.
3. External financial and OLAP plugins installed from their own repositories.
   Prove discovery, missing/duplicate plugin handling and config provenance.
4. Contract parity tests against existing adapters: inventory, delivered
   bytes, availability, holdout, receipts, outcomes and idempotency.
5. AdminLTE inventory, resource schema, settings and clear staged-configuration
   behavior, checked on desktop and mobile. Keep connection credentials out
   of rendered pages and committed examples.
6. Complete READMEs with an agent quickstart, exact installation commands,
   runnable disposable examples, limitations and cross-repository links.

These are new development repositories, not an immediate production move.
Keep the current three services available; do not move source files, clone the
cube, change store IDs or switch production launchers as part of a prototype.
Publish a tested migration and rollback procedure before that separate step.
This block must not delay A-C or require GPU time.

## Close-out

Publish one short status table distinguishing implemented, tested disposable,
deployed and production-proven. Include commit identities read from Git, test
counts from final code, governed campaign IDs, reconciliation counts and any
remaining failures. Report blockers with the missing fact and responsible
component. Do not return a code/configuration task to the owner when an agent
can complete it within this scope.

No broker actions, live activation, GPU sweep, dataset deletion, historical
rewrite or automatic promotion of financial resources belongs to this order.
