# I6N DOIN Island Sequence Plan

## Decision

Owner amendment 2026-10-14: restart H4 in a fresh chain/state namespace after
the proof-of-optimization repair. Retain the retired history as diagnostics.
Both new searches use the 19-feature availability-gated manifest, excluding
STLFSI4. H4 allows the previously approved input-resolution search; it is not
a separately fixed RAW control. H72 remains input-downsampled first.

Only a strict improvement over the committed parent champion can mint a
non-genesis block. All receivers append the originator's exact block. Migration
occurs only after commitment; proposal, stage and peer-best hints do not migrate.
The fresh nodes require canonical work proofs: VUW, reputation decay, reward
and header work are replayed from the committed parent history.

Use the established DOIN island topology, not the experimental centralized
shared-population path. Four nodes each evolve a local DEAP population of 20
and exchange champions through one blockchain. Run one governed H4 search to
completion, close and archive it, then automatically start the separately
identified H72 input-downsampling search. After its closure, compare H4 input
downsampling against the retained H4 raw result. The large hourly/raw H72
search remains a later control.

## Fixed Search Contract

| Setting | Value |
|---|---:|
| Local population per node | 20 |
| Maximum generations | 100 |
| Generation patience | 10 |
| Minimum relative improvement | `1e-8`, strict `>` |
| Seed | 6102026 on every node |
| Replicas | 1 |
| Folds per candidate | 4 distinct quarters |
| Direction | Minimize |
| Reported fitness | mean ratio to paired naive minus 1 |
| Search order | H4 raw, H72-DS-IN, H4-DS-IN, H72 hourly/raw control |

The values above are admission minima for this sequence. An implementation may
accept larger populations, generation ceilings, or patience only after a newly
sealed plan changes them; it may never silently lower them. The improvement
threshold is not a lower bound: it is exactly `1e-8`.

## Stage Plan

### 1. Unit implementation

Add `Plugin.optimize(predictor_plugin, preprocessor_plugin, config)` behind the
explicit `i6n_island_mode=true` flag. Keep shared methods unchanged. Validate
the complete contract before the first evaluator call.

The optimizer must:

1. construct a deterministic local population containing the minimal seed and
   small alternative;
2. evaluate each candidate on all four folds through the existing receipt
   verifier;
3. convert the established ratio to `ratio - 1` for reporting and minimization;
4. receive and inject valid network champions through existing callbacks;
5. perform deterministic selection, mutation and immigration within admitted
   complexity classes;
6. checkpoint population, fitness, RNG state, generation, patience and best
   candidate atomically;
7. emit candidate and generation callbacks with complete objective fields;
8. stop at 100 generations or after 10 completed generations without an
   improvement strictly greater than `1e-8`.

Group-safe crossover remains an open decision. It is not required to implement
the first correct island path and must not be improvised across conditional
genes.

### 2. Durable DOIN terminal

Before an island exits, persist a deterministic domain terminal containing:

- campaign/domain/config/plan identities;
- horizon and seed;
- terminal generation and reason;
- final population fingerprint and champion identity/fitness;
- last candidate ledger identity;
- canonical chain tip and finality checkpoint. An absent finality checkpoint
  is represented explicitly as height -1 and empty hash.

Restore this terminal after restart and expose it through the read-only API.

### 3. Automated successor

Implement a versioned oneshot service plus timer with this monotonic state
machine:

```text
H4_PREPARED
  -> H4_RUNNING
  -> H4_FLEET_TERMINAL
  -> H4_CHAIN_SHARED
  -> H4_ARCHIVE_VERIFIED
  -> H4_QUIESCED
  -> H72_DS_IN_RUNNING
  -> H72_DS_IN_FLEET_TERMINAL
  -> H72_DS_IN_CHAIN_SHARED
  -> H72_DS_IN_ARCHIVE_VERIFIED
  -> COMPLETE
```

Every transition is atomic and idempotent. No transition depends on a chat
agent remaining alive or reading logs.

### 4. Archive and OLAP

For every evaluated candidate, preserve configuration/genome inputs and metric
outputs, not training data or transformed feature matrices. The authoritative
analytical row includes:

- campaign, horizon, node, generation, candidate and genotype identities;
- exact repository commits and configuration digest;
- seed and admitted complexity class;
- all optimized genes and effective model parameters;
- per-fold train/inner error and naive sums, counts and support digests;
- mean ratio, `ratio - 1`, timing, resource peak and terminal disposition.

The blockchain may retain content hashes and durable references while the
object store/warehouse retain full metric payloads. H72 cannot start until H4
has chain verification and OLAP readback for the complete expected identity
set.

### 5. Fleet preflight

Query all four node APIs immediately before H4 launch and again before H72.
Refuse launch unless every node reports:

- seed 6102026;
- the same network/genesis identity;
- the same finalized chain height and hash after synchronization;
- exact matching commits for `predictor`, `doin-node`, `doin-core`, and
  `doin-plugins`;
- the same sealed plan/config identity for the active horizon;
- healthy evaluator dependencies and fresh resource admission.

Host-specific endpoints, device assignments and private addresses belong in a
private deployment spec, never in the public repository.

### 6. H72 readiness and contingency sequence

H4 remains untouched until all four terminal, chain and archive gates close.
During H4, prepare H72 without launching it:

1. pin the relay-provenance correction and rerun the disposable four-node
   gossip/fleet checks;
2. measure bytes per completed candidate on every island and project the
   remaining minimum of 600 candidates per island;
3. retain at least 20 GiB of non-campaign free space per host, plus the measured
   projected campaign bytes; refuse H72 when either term does not fit;
4. run one full-size H72 cost/admission pilot on each distinct host class, with
   the exact largest admitted window/model class and without reading TEST;
5. on gamma, keep both DOIN nodes reachable but serialize heavy evaluations
   through one host-wide lease; the 5090 has priority and the 5070 Ti consumes
   only a free slot while host RAM remains at its current capacity;
6. launch **H72-DS-IN first**, under a new network, plan, checkpoint and archive
   identity. Samples are unsmoothed and
   right-anchored at the current known tick (`t, t-k, ...`). The target is still
   the exact hourly `t+72h` endpoint, and the business mask and 72-hour naive
   remain byte-identical to the frozen H72 support. Its sealed search domain
   requires `delta_x >= 2` and endpoint-only supervision.
7. after H72-DS-IN closure, run a clean paired **H4-DS-IN versus H4-RAW**
   contrast with the same 19 features, support, naive and budget. The active H4
   campaign remains a retrospective engineering reference, not its raw control.
8. run the hourly/raw H72 identity later as the large-resource control.

Only if input-only evidence justifies it may a separate **H72-DS-INOUT**
identity change output steps. Its final head must still land exactly at
`t+72h`; its error and naive use the same original-time rows and scale.

No smoothing, interpolation across unknown future bars, target cleaning, TEST
access or identity reuse is allowed. Downsampling factors, effective window,
model capacity and repository/config digests are stored with every candidate
in the external payload and OLAP projection.

## Required Test Order

1. `tests/test_i6n_island_optimizer.py`
2. retained I6N optimizer/search tests
3. DOIN wrapper callback integration tests
4. successor and terminal falsification tests
5. four-node disposable system test
6. production read-only preflight
7. H4 launch and first-five-candidate alpha gate

## Stop Conditions

- Any identity, seed, commit, fold, support, chain, or archive mismatch refuses
  the affected transition.
- A resource refusal does not lower the scientific configuration or fall back
  to CPU.
- A partial four-fold candidate is terminally non-rankable.
- TEST access invalidates the run.
- The sequence never starts H72 from a merely local H4 completion.

## Hash-Only Candidate Archive

The node keeps legacy behavior unless `candidate_archive_mode` is explicitly
`external_payload`. Each production node uses the same external store and its
own durable outbox root; the reference spool is consumed by the existing node
spool loop.

```json
{
  "candidate_archive_mode": "external_payload",
  "candidate_archive_root": "/var/lib/doin/i6n-candidate-archive",
  "candidate_archive_store_id": "<archive-store-id>",
  "candidate_archive_store_url": "https://<archive-host>",
  "candidate_archive_token_env": "DOIN_ARCHIVE_TOKEN",
  "candidate_archive_pump_interval": 30,
  "archive_reference_spool": "/var/lib/doin/i6n-reference-spool",
  "archive_spool_interval": 30
}
```

The callback first validates and writes `i6n.archive.island_candidate.v1` to
the local SQLite outbox, then PUTs and GET-verifies body and manifest in the
content-addressed store. Only after readback does it write the hash/reference
transaction to the spool. Candidate configuration, genotype and metrics never
appear in that transaction. The ETL resolves finalized references and writes
four fold reports plus one candidate report to OLAP. The retained fold evidence
includes error sums, naive sums, counts, receipts and origin identities, so the
candidate fitness is independently recomputable. Missing or altered bytes
remain unresolved; the autonomous pump retries retained outbox items. A
separate idempotent closure timer emits the successor receipt only after every
candidate body resolves and its exact warehouse readback matches. Every
candidate is first projected as OFFCHAIN_VERIFIED, carrying no block/finality
claim. Genuine later champion blocks may upgrade a reference to FINALIZED.
Corpus closure v3 retains the count of unfinalized references rather than
minting housekeeping blocks or waiting forever for six successors. All four
optimizer terminals, full-chain equality and candidate-count/snapshot identity
still gate H72. The earlier FINALIZED-only closure v2 remains supported.

## Current Boundary

The repaired H4 fleet was started on 2026-10-14 with matching seeds and four
component pins. Its first real improvement was adopted byte-identically by all
four nodes and independently checked against its canonical work proof and
archived metric bytes. Candidate metrics were delivered and read back from the
warehouse with OFFCHAIN_VERIFIED provenance. Controller and per-peer archive
timers run without an agent. H4 completion and the live H72 handoff remain
future acceptance points. H4 scores remain development/tuning evidence; seven
feature producers/publication clocks remain gated before TEST.
