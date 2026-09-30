# DOIN: governed block bodies outside replicated storage

Status: approved direction; implementation and disposable pilot planned, not deployed.
Owner: Musashi; execution coordinator: Satoshi. Applies to DOIN experiments,
including the doctoral programme. This is an independent infrastructure lane,
not a prerequisite for the experiments already authorized.

## Grounding and scope

Inspected doin-core a90bca2ddd937545c4321dd0d5c2d271bf8a870d:
`src/doin_core/models/block.py`: Block contains header, ordered transactions and
header hash. The header commits to previous_hash and merkle_root, as well as
proof-of-optimization fields. Do not replace this with an unvalidated empty block.

Inspected doin-node 8bfc64f5de20200b93a8f9451b4e1a7ea9742df8:
`src/doin_node/unified.py`, try_generate_block, generates through consensus,
applies rewards/difficulty, persists, updates finality and can prune transaction
bodies at snapshots. `stats/chain_metrics.py` extracts OPTIMAE_ACCEPTED records.
That extraction is NOT an inventory of every evaluated candidate. Candidate
callbacks also persist detailed metrics; inventory their complete paths before
claiming all evaluations are archived. Graph discovery plus source inspection;
no running node or live database was opened or modified.

## Requirements and use cases

- O01: preserve collection, deduplication, evaluation identities, ordering,
  verification, rewards and optimization semantics. Failed/repeated attempts
  remain distinguishable. No change to a model, target, split or metric.
- O02: support full retention, selective externalization, and eventually
  header-only local retention. Retention is not validator authority: a node
  checking consensus still retrieves all evidence required by its rules.
- O03: preserve full reconstructible bodies in the governed lake; keep stable
  content identity on-chain, not a credential, local path or server address.
- O04: rebuild the existing warehouse from authenticated archived bodies;
  preserve units, scale, population, horizon, reduction and provenance per metric.
- O05: recover from upload failure, crash, retry, missing bodies and chain reorg
  without lost pending results, duplicate rewards or double-counted metrics.
- O06: measure storage savings and resource costs on our existing machines.
  No invented capacity, new storage purchase or compression requirement.

Use cases: reconstruct an experiment; compare candidates in OLAP; verify a
specific metric against its committed record; rebuild warehouse from scratch;
resume a crashed publisher; migrate an archive provider without changing hashes.

## Proposed protocol and lifecycle

1. Freeze an immutable snapshot of pending ordered transactions and the exact
   evaluation records included. Keep a durable pending journal until commit.
   Preserve original transaction IDs and signatures; do not silently reserialize
   signed payloads. Record excluded/late arrivals for the next snapshot.
2. Serialize a versioned body deterministically. Specify encoding, numeric
   representation, order and rejection of non-finite values. Hash exact bytes.
   A manifest binds section digests, sizes, schemas, counts and ordered inventory.
   Distinguish body digest, manifest digest and existing block/header hash.
   Neither body nor manifest contains its own digest as hashed input.
3. Publish content-addressed objects through the lake's governed write path,
   verify by read-back and retain durable receipts. First implementation task:
   establish the actual write API and permissions; do not assume registration
   alone uploads bytes. Missing permission is a scoped dependency, not bypassable.
4. Bind the manifest commitment through a versioned protocol representation.
   Preferred initial implementation: preserve logical transactions/Merkle root
   while externalizing their physical bodies. If a commitment transaction or new
   header field is needed, it requires explicit protocol version and compatibility
   tests in doin-core. A one-transaction carrier must not replace the logical
   transaction count used by fees or erase proof-of-optimization evidence.
5. Validators resolve, download, hash-check and perform the existing semantic
   validation before acceptance. A hash proves bytes, not truthful metrics or
   availability. A unavailable body yields explicit unavailable/pending status,
   never a successful validation. Light clients are identified as such.
6. Journal publication and consensus transitions: PREPARED -> ARCHIVED ->
   CHAIN_ACCEPTED -> FINALIZED -> PROJECTED. Define crash recovery around every
   state and the existing reward/difficulty mutations. These stores do not share
   a transaction: use recoverable, idempotent operations, not an atomicity claim.
7. An asynchronous ETL reads verified bodies and upserts into the existing
   warehouse. Provisional rows may be visible if clearly labelled; canonical
   scientific queries filter finality/canonical membership. Keep metric record
   identity separate from block inclusion so a reorg does not duplicate a fact.
   Reorgs retract membership/projections, not retained history. Reconcile back
   from warehouse to exact committed inventory; counts alone are insufficient.
8. Prune local bodies only after finality, archival verification and a recorded
   recoverability check. Never delete the sole archive. Existing pruning needs
   this guard in the new mode. Uploads orphaned by failed block proposals require
   reference-aware garbage collection and a retention policy, not instant deletion.

The full block can be reconstructed as original header plus resolved body. A
separate post-commit archive envelope may include the block hash; it is not the
pre-commit body's self-reference. Archive locations are mutable resolver metadata
behind stable resource/content IDs. Moving servers must not rewrite chain history.

All candidate records need their own manifest coverage, not an inference from
OPTIMAE_ACCEPTED. A round with no qualifying block can still archive pending
records, explicitly UNANCHORED; later commitment binds them. Never label such
records chain-verified merely because their upload succeeded.

## Ownership and parallel implementation

Satoshi coordinates separate worktrees and agents, reconciles dependencies and
integrates incrementally. No agent changes a live service checkout.

1. Core agent: doin-core protocol/schema/serialization, validation and compatibility.
2. Storage agent: doin-node producer, pending journal, resolver, sync and pruning;
   data-lake/data-gov adapters live in their owning repositories.
3. Analytics agent: data-warehouse ETL/provenance/reorg/rebuild, using the existing
   metrics contract and row-level identity. No second competing warehouse.
4. Integration agent: disposable multi-node pilot, faults and resource accounting.

Agents 1-3 can design acceptance/component/unit tests in parallel against a
shared versioned contract; integration proceeds once interfaces are agreed.
Before implementation each owner records test designs and requirement links in
its existing method state. This plan is not a claim that those gates have passed.
No GPU training is required for this lane. Existing experiment and M5PHET lanes
continue; use an admitted CPU worker for heavier tests, never exhaust the desktop.

## Acceptance tests required before adoption

| Test | Requirement | Required evidence |
| --- | --- | --- |
| AT01 | O01/O02 | Same ordered transactions through full and external modes give identical logical validation, candidate decisions, rewards and analytical rows. |
| AT02 | O03 | Altered/missing/reordered body or forged receipt is rejected; valid read-back resolves the exact content. |
| AT03 | O01/O05 | Crash at every lifecycle boundary and concurrent retries lose no pending record and duplicate no reward/fact. |
| AT04 | O04/O05 | Reorg and re-inclusion preserve one metric identity, correct membership and no stale canonical totals. |
| AT05 | O03/O06 | Lake outage denies new external commitments safely; unrelated experiments continue; restore and replay succeed. |
| AT06 | O02/O03 | Pruned peer resync and full block reconstruction work; unsupported protocol is rejected, not interpreted as empty transactions. |
| AT07 | O01/O04 | All candidate attempts, accepted improvements and no-improvement rounds reconcile independently; missing records fail completeness. |
| AT08 | O04 | Empty disposable warehouse rebuilt from archive equals the full-mode oracle per metric and provenance, not just aggregate counts. |
| AT09 | O03/O05 | Provider migration preserves identity; expiry/deletion refuses removal of the sole retained body; access denial stays denied. |
| AT10 | O06 | Measured bytes by node/lake/warehouse/cache, peak RAM, upload/validation latency and rebuild time; no savings inferred from hashes alone. |

Use deterministic fixtures and a recorded representative optimization stream;
no new model training needed. Preserve PRE/POST for current-path defects. Pilot
budget and disk projection must be measured and admitted before execution.

## Rollout and limitations

Stage 0: contract and test design. Stage 1: shadow archive alongside existing
full mode, no pruning or consensus change. Stage 2: disposable multi-node
external-body protocol pilot with faults and ETL comparison. Stage 3: reviewed,
versioned activation for new DOIN experiment campaigns, including doctoral runs.
Full retention remains available; do not rewrite historical blocks or hashes.
Rollback restores retrieval/full retention prospectively, not historical identities.

Centralized archival storage reduces replication and introduces an availability
dependency. Hashes cannot recover deleted bytes. Record number/location class of
copies and retention policy; do not claim blockchain-level availability. Archive
integrity does not establish scientific correctness or forecasting usefulness.

Execution authority now: plan and isolated implementation/test preparation;
not live chain migration, production pruning, service restart or new training.
Return: commits per owner, test evidence, unresolved API/compatibility decisions,
resource measurements and rollout recommendation. This feature is PLANNED,
not delivered and not a scientific result.
