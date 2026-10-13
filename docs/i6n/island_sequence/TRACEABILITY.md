# I6N DOIN Island Sequence Traceability

## Requirement To Evidence

| Requirement | Unit-test evidence | Later integration/system evidence | Status |
|---|---|---|---|
| ISL-001 | `test_island_mode_requires_explicit_true_flag` | Predictor optimizer load with old and new configs | Unit passed; integration pending |
| ISL-002 | `test_island_admission_rejects_weaker_or_different_search_settings` | Four node statuses each report local population 20 | Unit passed; fleet evidence pending |
| ISL-003 | Same admission test | Sealed H4/H72 configs report max generations 100 | Unit passed; fleet evidence pending |
| ISL-004 | Same admission test; patience behavior tests | Terminal receipt reports generation patience 10 | Unit passed; terminal evidence pending |
| ISL-005 | Admission and both threshold semantics tests | Candidate/generation ledger preserves exact `1e-8` | Unit passed; ledger evidence pending |
| ISL-006 | Horizon rejection and acceptance tests | Sequence controller admits H4 then H72 only | Unit passed; sequence evidence pending |
| ISL-007 | `test_each_candidate_is_evaluated_on_all_four_distinct_folds` | Four receipt identities per ranked candidate | Unit and archive projection passed; production pending |
| ISL-008 | `test_reported_fitness_is_ratio_minus_one_and_negative_is_better` | OLAP readback recomputes fitness from fold ratios | Unit and disposable OLAP readback passed; production pending |
| ISL-009 | Stub test surface contains no TEST input | Future-perturbation and access-denial system tests | Partially designed |
| ISL-010 | `test_same_seed_replays_the_same_local_island` | Restart parity from atomic checkpoint | Unit passed; restart system evidence pending |
| ISL-011 | Existing `test_i6n_optimizer.py` and `test_i6n_optimizer_v2.py` | Replay retained shared evidence | Existing regression coverage |
| SEQ-001 | `test_i6n_island_successor.py` | Successor state-machine test: H72 refused before four valid H4 terminal envelopes and launched exactly once afterward | Integration passed; production transition pending |
| SEQ-002 | `test_i6n_island_terminal_collect.py` | Concurrent eight-endpoint collection rejects seed, config, network, genesis, height, tip, finality or commit divergence | Integration passed; fleet evidence pending |
| SEQ-003 | `test_external_chain_transaction_contains_only_reference_and_retry_is_idempotent`; `test_candidate_or_metric_mutation_changes_external_body_hash`; `test_missing_or_unreadable_external_payload_is_rejected` | `test_island_candidate_reaches_olap_with_exact_readback` proves durable outbox, finalized reference resolution, four fold projections, candidate projection, and exact OLAP readback on disposable stores | Component integration passed; production fleet gate pending |
| SEQ-004 | `test_i6n_island_successor.py` | Durable successor state and exactly one H72 launch across repeated invocations | Integration passed; restart system evidence pending |
| SEQ-005 | Horizon acceptance tests | Separate sealed configs and transition receipt | Partially designed |
| SEQ-006 | `test_archive_closure_requires_one_complete_outbox_per_terminal_island`; `test_archive_closure_accepts_a_verified_descendant_of_the_terminal_chain`; `test_live_wal_outbox_is_snapshotted_and_bound_to_exact_terminal_ledger`; `test_four_independent_outboxes_are_bound_to_their_four_terminals` | Four independent SQLite snapshots reconstruct and exactly match each island terminal candidate-ledger digest; closure verifies candidate bytes, peer/island identity, exact OLAP readback, and terminal-checkpoint ancestry before H72 | Integration passed; production closure pending |
| OPS-001 | Same-seed unit test; `test_i6n_island_terminal_collect.py` | API preflight rejects any commit, seed, network, height, tip or finalized-checkpoint mismatch | Integration passed; production API evidence pending |
| OPS-002 | Not in optimizer unit scope | Versioned systemd unit/timer installation and unattended replay | Pending system test |
| OPS-003 | Generation callbacks in patience tests | Read-only status schema and ETA mutation tests | Partially designed |
| OPS-004 | Same-seed determinism | Kill/restart parity after candidate and generation boundaries | Pending system test |
| OPS-005 | Configuration rejection precedes evaluator | GPU visibility, memory admission, and no-CPU-fallback probes | Partially designed |
| OPS-006 | H72 readiness checklist binds measured bytes/candidate, projected remaining candidates and reserve | Per-host production projection and archive-preserving relocation receipt | Designed; production measurement pending |
| OPS-007 | `doin-node` relay tests preserve handler address, exclude the physical transport peer, and stamp the next relay while retaining the logical author | Disposable four-node run has zero wrong-host gossip endpoint attempts | Component passed on prepared commit; system verification pending |
| H72-001 | H72-DS-IN constraint: right-anchored, no smoothing, `delta_x>=2`, endpoint-only exact `t+72h` | Unit tests reject hourly, trajectory and foreign network candidates | Implemented; production start awaits H4 closure |
| H72-002 | H72-DS-IN retains frozen H72 support and naive | Generator test compares every fold's support, origins and pinned source cells | Implemented; production readback pending |
| H72-003 | Input/output contingency contract: exact original endpoint and naive despite step representation | New identity, alignment mutation tests and paired original-scale readback | Designed; dormant |
| H72-004 | Gamma placement declares one host-wide heavy-fit lease, four reachable nodes and 5090 priority | 5070 Ti consumes free slots; fresh dual-workload admission is required before concurrency | Designed; production gate pending |
| H4DS-001 | Clean H4-DS-IN and H4-RAW follow H72-DS-IN; current H4 is only a retrospective engineering reference | Separate identities and exactly paired 19-feature H4 support/naive tests | Planned |
| H72RAW-001 | Hourly/raw H72 remains a later large-resource control | Separate identity, admission pilot and paired comparison | Prepared; not first successor |

## Acceptance Layers

1. **Unit:** focused file in this initiative passes, including all mutations of
   population, generation count, patience, tolerance, horizon, fold count,
   objective sign, seed, and strict threshold comparison.
2. **Regression:** retained I6N v1/v2 tests pass unchanged.
3. **Integration:** the DOIN predictor wrapper calls `Plugin.optimize`, relays
   callbacks, exchanges a champion, and preserves candidate identity.
4. **System:** four disposable nodes converge on one blockchain, survive a
   restart, archive by identity, and transition from H4 to H72 exactly once.
5. **Alpha:** the production preflight proves exact pins and common chain state;
   the first five candidates produce valid four-fold receipts and OLAP readback.

## Evidence Discipline

- Unit receipts are labeled `stub` and can never satisfy scientific gates.
- Every evidence artifact binds its code commit, configuration digest, input
  digests, seed, horizon, fold, candidate identity, and protocol version.
- A dashboard value is observational until read back from its authoritative
  store and reconciled by identity.
- Historical 8-by-3 results remain retained but are explicitly non-governing
  for the H4/H72 island sequence.
- A fleet closure is invalid if it sees fewer than four distinct outboxes, if
  any outbox cannot reconstruct its island's exact terminal candidate ledger,
  or if producer, island, candidate-set or terminal identities are duplicated,
  or if the closure chain does not contain the exact terminal tip and finalized
  checkpoint. Chain advancement after terminal collection is permitted only
  with an explicit descendant proof.
