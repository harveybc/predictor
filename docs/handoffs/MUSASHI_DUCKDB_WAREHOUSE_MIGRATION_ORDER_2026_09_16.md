# Imperial authorization: replace only the OLAP warehouse with DuckDB

Owner instruction received 2026-09-16. Executor: Satoshi. Reviewer: Musashi.
This addendum extends V1-V4; it does not cancel the temporal semantic fixes.

## Authority, scope and continuous execution

The owner explicitly authorizes implementation, migration, deployment and the
necessary scoped warehouse service changes. Finish ALL blocks below without
requesting approval at each stage. Progress messages are informational, not
requests to continue. Maintain durable progress across context compaction.

The V3 instruction to wait for reviewer deployment is superseded FOR THIS
warehouse migration: after the acceptance criteria below pass, deploy and
verify the DuckDB successor. Do not manufacture reviewer records or scientific
approvals. If an existing mechanism requires an unavailable external record,
report that exact dependency rather than self-issuing it, and finish independent
blocks meanwhile. Actual tool denials must not be bypassed or described as
missing owner permission. Stop only the affected operation on a real blocker.

Replace PostgreSQL ONLY as the OLAP warehouse storage engine. Do not uninstall,
stop, migrate or change PostgreSQL used by any other project, service or the
Metabase application database. Preserve data-gov accounting and the financial
and synthetic lakes. No new scientific training, live trading, benchmark
campaign or result promotion is authorized by this migration.

## D0 - Discovery and test-led plan

Read the project test-led/data-centric methodology. Inventory actual warehouse
writers, readers, loaders, outboxes, ETLs, Metabase connections and DOIN result
routes. Distinguish deployed calls from unused code. Record concrete storage
ownership before making changes. Existing warehouse tables include more than
gov_*: enumerate df_*, dimensions, performance facts and receipts as applicable.

Create or update the initiative method state and requirement-to-test matrix.
Design acceptance/system/integration/unit tests top-down, implement bottom-up.
Persist current stage, completed evidence, next actions and dependencies.
No separate DuckDB-versus-PostgreSQL speed contest is required.

## D1 - Select evidence, not favorable outcomes

Define the current data-centric campaign from committed orders, campaign/run
identities, dataset/code/design lineage and recorded outcomes. Do NOT select
solely by date, model family, metric sign, presence of defaults or table name.
Default parameters alone do not invalidate a controlled baseline.

Produce a machine-readable per-run manifest with INCLUDED_CURRENT,
LEGACY_COMPARISON_ONLY, INVALIDATED, MECHANICAL_ONLY or UNRESOLVED, rationale
and source references. Separate campaign membership from scientific validity.
Include current negative, failed, inconclusive, refused and superseded outcomes
with their actual statuses and links. Governance smoke tests are operational
evidence, never scientific comparisons. Unresolved rows cannot enter primary
scientific views. Do not invent current campaign membership to fill a quota.

Migrate current campaign evidence and its dependency closure: contracts,
dimensions, raw-result references, metrics, costs, receipts and lineage. Preserve
old experiments in a verified, separately catalogued archive; do not import
them into default scientific views. Comparison views must explicitly opt into
admissible legacy evidence and disclose comparability limitations. No source
deletion or rewriting of PostgreSQL history is authorized.

## D2 - DuckDB provider and operational design

Implement an external DuckDB provider through data-warehouse's existing plugin
interface. Keep the host, data-gov policies, inventory, configuration dashboard
and result contracts consistent. Do not create another parallel governance API.
Use isolated namespaces and package entry points, not checkout import tricks.

One warehouse-owned process manages native database writes and reads through
the service. Workers never open a shared DuckDB file across machines. Route
actual OLAP loaders/ETLs through the owned interface, including replay of durable
pending outcomes. Batch writes where appropriate; bounded queues, explicit
backpressure, transactions, idempotency and restart recovery are required.

Use a local persistent volume, measured memory limits, bounded threads and disk
space checks. Prevent analytics from starving outcome ingestion. Preserve exact
contract bytes, hashes, UTC/time precision, identifiers, nulls and statuses;
document explicit type conversions. Unknown availability never becomes zero.
Complete V1-V2 temporal validation at both write and independent read.

## D3 - Reproducible migration tooling

Deliver scripts for inventory/selection, dry run, export, import, validation,
catch-up and rollback, with README commands. Configuration must not embed
credentials. Run the migration rehearsal against disposable destinations.

Take a consistent source snapshot with a durable watermark and a tested backup.
Export selected evidence in bounded batches with schemas and manifests. Retain
canonical contract bytes unchanged. Include large external artifacts by verified
reference or archive as appropriate, never silently drop them.

Validate source-to-destination row content at each table's actual grain, not just
counts or stored digest columns. Check referential closure, exact statuses and
contracts; explicit tolerances only for deliberately recomputed numeric outputs.
Report excluded counts by reason and every conversion. A second import must
create no duplicates. A interrupted import resumes without partial visibility.

## D4 - Acceptance and consumers

Test real host + provider + data-gov + warehouse + fresh reader with governed
deterministic inputs. Prove transfer/cache, all terminal outcomes, outbox recovery,
duplicate retry, write interruption, process restart and concurrent queries.
Test end-to-end receipt reconciliation and contract recovery after the producer
is unavailable. Apply a negative temporal contract through the same route.

Exercise actual adapters for predictor, preprocessor, feature-eng,
feature-extractor and the bounded offline DOIN replay route where implemented.
Reuse existing mechanical fixtures; do not repeat scientific campaigns.
Report requested and observed work distinctly. No synthetic test terminals may
be passed off as scientific evidence.

Provide working analytical access and representative existing cube queries.
Verify the deployed Metabase version's DuckDB integration before promising it.
If a supported driver cannot be used, implement and document usable warehouse
analytics through the existing console; do not keep PostgreSQL secretly serving
the OLAP cube or migrate Metabase's unrelated application database. Declare any
dashboard parity gaps explicitly and assign their remediation.

## D5 - Authorized deployment and rollback

Deploy only after rehearsal and acceptance pass. Build the replacement in an
isolated environment; never mutate the live shared environment in place.
Drain or durably queue in-flight warehouse writes, take the final checkpoint,
catch up from the watermark and verify no omitted/duplicated outcomes. Switch
only the required warehouse/loader configuration and services. Preserve actor
identities, policy and other service configurations. No broad process-name kills.

Keep the old warehouse source intact for rollback, not as an active second OLAP
writer. Record the cutover boundary. Rollback must preserve/replay DuckDB-era
accepted outcomes so returning to the old backend cannot lose new results.
Test this against disposable data before production. Never drop the new evidence
as a shortcut to rollback. Leave unrelated PostgreSQL workloads untouched.

Run bounded governed acceptance in production, verify query results, reconciliation,
loader progress and stability over multiple collection cycles. Report the actual
engine used by the running process and its file identity, not only configuration.

## D6 - Finish and publish

Update data-warehouse, provider and integration READMEs, agent instructions,
configuration examples, dashboard instructions and the current work plan. State
clearly DuckDB is the OLAP engine; PostgreSQL references are legacy migration or
unrelated services where applicable. Publish reviewed-scope changes normally;
never force-push or expose private data in Git. Preserve concurrent agent work.

Use the three machines for independent CPU tests/rehearsals where productive,
with separate databases and resource caps. A single production database owner
is intentional, not a failure to distribute. No GPU is needed.

Return one final packet containing commits, exact test scope and environment,
migration manifests/counts/content checks, historical archive inventory, services
changed, running-engine proof, new terminal reconciliation, rollback proof and
remaining limitations. Do not stop after D0 or a successful partial test to ask
whether to proceed. Carry through deployment unless an actual failed acceptance
criterion or unavailable prerequisite makes that unsafe; then finish everything
else and identify the exact missing action and its owner.

Success: DuckDB serves the current campaign's OLAP cube through governance;
historical evidence remains recoverable and segregated; no unrelated PostgreSQL
use changes; real consumers can record and analyze results without data loss.
