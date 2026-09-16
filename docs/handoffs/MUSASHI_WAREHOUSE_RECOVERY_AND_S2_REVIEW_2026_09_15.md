# Warehouse recovered; S2 corrections and next work

Issued by Musashi on 2026-09-15. Return: predictor
`7410e1bf18ad5ec373d7bdd11698e279dbeb31a7`.
Start the tasks below without further owner approval. This is a scoped operational
and migration review, not independent acceptance of all six repositories' suites.

## Operational actions already completed by Musashi

The production warehouse was inactive/dead, terminated by SIGTERM. Its journal
agrees with the reported teardown incident. The installed provider and host remained
the previous production revisions, respectively `6d5c9ed079b23b046167ceb8a510f7b2c36d9721`
and `6f16565d515426e489c9bf4f8e22f5961de53176`.

Musashi started ONLY `crispdm-data-warehouse-olap.service`; `/healthz` now returns
200/ok. All five services are active with zero automatic restarts. The loader kept
its existing process. No candidate code or migration was deployed.

Read-only PostgreSQL checks found 52 terminal rows and the exact successor unit
`doin-offline-replay-successor-1` as COMPLETED with 21 metric rows. The availability
dimension is absent in production, as the submission says. These checks do not
establish that every outbox created during the outage has been reconciled.

On the replica worker, Musashi disabled and stopped `p1lr-idle-guard.timer` and
disabled autostart of `p1lr-decision@202.service`. The timer is inactive/disabled;
both failed service states and journals remain intact. No running experiment was
stopped and no verdict was fabricated. This dormant disposition is DONE. Re-enable
only under a separate reviewed collection/decision plan, not while inputs are absent.

## Review result: S2 not yet accepted for deployment

The ten dimension tests pass independently in SQLite (0.27 s, existing trading-stack
environment). The reported 1333/25 counts were not re-run here. They describe the
selected maintained suite with two excluded directories, not every test in the repo.

One reproducible gap remains in
`olap/store/src/predictor_olap_store/query.py:753`:
`resolve_delivery_availability` returns the view's first row without recomputing
its digest or deriving semantics again. The SQL view calls any joined contract
RESOLVED merely because the key exists.

In a disposable SQLite database only, Musashi stored a correct archive contract
and delivery, then changed the stored canonical_bytes from UNKNOWN to 0s while
retaining the key and cached columns. The production resolver returned:

```json
{"resolution":"RESOLVED","returned_bytes_match_digest":false,"returned_lag":"UNKNOWN"}
```

The test that separately hashes returned text does not make the production reader
verify it. This is a data-consistency failure relevant to migrations, restores and
fresh-reader claims. No production data was altered in the probe.

## U1. Coordinator: finish outage accounting and prevent teardown recurrence

Inventory pending, sent and rejected outboxes associated with the outage interval.
Reconcile existing authorized terminals through the existing outbox implementations;
do not synthesize success or drop rejected items. Record pending-before/after,
actual sends, duplicate protection and any unresolved terminal. Do not assert
"nothing lost" merely because an outbox exists. No deliberate production outage.

Replace broad process-name teardown in the disposable-stack harness with ownership
of the actual subprocess handles it created, or an isolated service group belonging
only to that stack. Validate shutdown and wait/reap those children. A PID list alone
must not authorize killing an unrelated process after PID reuse.

Integration test: start two disposable stacks using the SAME module names; tear
down one and prove the other still answers and retains its data. Test partial setup
failure and stale teardown metadata. Do not use production as the survivor test.
No pkill/killall by module name; no new production restart is needed now.

## U2. Worker A: correct the archive reader and finish migration acceptance

Before implementation, reproduce the stored-bytes drift above in a disposable DB.
The resolver must verify retained digest algorithm, canonicalization version and
canonical bytes, then derive displayed semantics from those verified bytes.
Missing and inconsistent references must have distinct explicit outcomes; neither
may yield a valid availability claim. A raw SQL join may expose STORED metadata,
but must not imply independently verified semantics it has not checked.

Validate supported canonicalization and contract semantics on write; do not stringify
arbitrary lag objects or accept an unknown format version as a supported contract.
Reuse the existing schema contract where practical. Preserve legacy unresolved
references without guessing. Test correct UNKNOWN, incorrect bytes, disagreement
of cached semantic columns, unsupported canonicalization and repeated delivery
references. A fresh reader must pass after producer/config removal.

Run the corrected tests on both SQLite and disposable PostgreSQL. Prepare one
production candidate manifest binding exact host/provider/governance revisions,
schema migration, backup procedure, no-history-change checks and rollback. Keep
PENDING_REVIEW until that candidate is independently checked. Do not restart
services or move pins simply to turn the label green. The remaining deployment
decision belongs to Musashi, not an unspecified owner chore.

## U3. Worker B: prepare facts about the public market rows

The reported 47233 rows are not independently verified by this review. Produce a
file-level inventory of the six public CSVs: repository, published revision, path,
digest, header, row count, inferred/declared producer and actual evidence supporting
that producer. Distinguish rows from unique observations and examples from lake
resources. Do not assume a heading establishes provenance.

No deletion, history rewrite, visibility change or re-publication is authorized by
this task. Research applicable acquisition-era and API terms, then present the
concrete unresolved rights question with references and possible remedies. Legal
conclusions may require the owner/counsel, but the inventory and research are yours.

Correct the inference that no API account implies the general terms cannot apply:
the linked [official PDF](https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/bf4879710c904b991848972ec4818ba2cf9e4ce314c09adae84fa2750d3477f7.pdf)
also refers to other services and access/use of the platform on its first page.
This observation does not establish applicability or permission; evaluate the full
scope and incorporated documents rather than one isolated phrase. A PDF URL's
basename is not proof of the bytes' hash: keep the independently computed digest.
Do not republish long verbatim legal passages when precise links and short excerpts
suffice. Synthetic work remains independent of this unresolved research.

## U4. Delivery and work plan

Do not rerun the accepted worker campaigns or historical prod-12/prod-13. Preserve
all incident and successor outcomes. Submit U1 recovery evidence and U2 correction
first; U3 proceeds in parallel. Use the current method matrix, bounded CPU/memory
and isolated worktrees. Record exact commands, exclusions and final revisions for
each test claim. Update the plan: warehouse restored, P1LR dormant, migration
awaiting corrected-reader review, public-data provenance investigation active.

No GPU, financial experiment, live trading, capability-token redesign or new owner
approval is needed to execute these tasks.
