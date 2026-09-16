# I1-I3 operational acceptance and bounded follow-up

Reviewer: Musashi. Reviewed return: predictor 18ffa4e. Date: 2026-09-16.

## Accepted at the measured scope

Ran the shipped reconciler independently against the LIVE warehouse API and
data-gov's independently retained accepted payloads, read-only:

- accepted 55, content_matches 55;
- differing, missing, unverifiable and orphan terminals: zero;
- terminal rows 55; metric rows 533; artifact rows 97; dataset rows 92;
- metric predicate/scan 533/533, artifacts 97/97, datasets 92/92;
- verdict NO_LOSS_FOR_THE_COMPARED_POPULATION, exit 0; no repair requested.

Content digests match the reported corrected population. Private receipt:
~/.local/state/crispdm-duckdb/musashi-review-18ffa4e.json.
This is a live read, not a transactional snapshot or proof of future stability.
It establishes current agreement, not the cause of the past index/WAL incidents.

Executed tests/test_surplus_multiplicity_repair.py and
tests/test_incident_reconciler_contract.py with the isolated DuckDB test
interpreter: 64 passed in 12.62 seconds. Full-suite and three-engine counts were
not independently rerun. Earlier non-reproducible receipts stay superseded.

I1-I3 correction is accepted at this scope. Do not repeat the repair, rebuild
indexes speculatively, or hold healthy operation awaiting a root-cause story.
The four metric before-images and original quarantined WAL remain evidence.

## Orphan cleanup completed by Musashi

Verified the named September 15 disposable scope's actual cgroup membership:
one remaining process, launched with the disposable stack's governance config,
listening only on its ephemeral port and separate from all production services.
Stopped ONLY that exact user scope. Afterwards: scope inactive/dead, ephemeral
port no longer listening, all five production services active/running with
NRestarts=0. No production restart, database deletion or original WAL access.
The orphan-stop request is therefore CLOSED; Satoshi must not repeat it.

## Next work for Satoshi, already authorized

1. Prevent the STACK.json overwrite recurrence. Give every disposable invocation
   a unique immutable run identity and process manifest. Refuse reuse of an
   occupied directory before starting children, or allocate a unique child
   directory. Keep the old ownership record rather than overwriting it. Test two
   starts at the same destination, concurrent starts, partial launch failure and
   teardown after parent exit. Verify owned-process cleanup without broad name
   matching or relying on directory name alone to terminate a process. Never
   exercise cleanup tests against production. Deploy no unrelated changes.
2. Keep a bounded read-only reconciliation/monitoring check for warehouse
   multiplicity and predicate-versus-scan consistency, with measured query cost.
   Define cadence from that cost instead of running full scans on every request.
   Record discrepancies and alert; no automatic deletion, reindexing or repair.
   Ensure monitoring outcomes retain evidence and do not become scientific runs.
3. Return to the existing data-centric work-plan backlog. Inventory the next
   unfinished preprocessing/feature-engineering step from committed orders and
   identify its actual prerequisites. Prepare its data contracts, tests and
   governed execution plan; do not invent scientific acceptance, change sealed
   designs or launch a new large training campaign under this operational order.

Finish these independent blocks without asking the owner to continue between
them. Keep method state and work plan current and return exact tests/revisions.
Metabase's driver and historical terms research remain separate, nonblocking
tracks. Candidate root-cause mechanisms are excluded only for tested conditions,
not universally ruled out. No incident WAL replay is required here.
