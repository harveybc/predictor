# I6N DOIN Island Sequence Traceability

## Requirement To Evidence

| Requirement | Unit-test evidence | Later integration/system evidence | Status |
|---|---|---|---|
| ISL-001 | `test_island_mode_requires_explicit_true_flag` | Predictor optimizer load with old and new configs | Designed, expected red |
| ISL-002 | `test_island_admission_rejects_weaker_or_different_search_settings` | Four node statuses each report local population 20 | Designed, expected red |
| ISL-003 | Same admission test | Sealed H4/H72 configs report max generations 100 | Designed, expected red |
| ISL-004 | Same admission test; patience behavior tests | Terminal receipt reports generation patience 10 | Designed, expected red |
| ISL-005 | Admission and both threshold semantics tests | Candidate/generation ledger preserves exact `1e-8` | Designed, expected red |
| ISL-006 | Horizon rejection and acceptance tests | Sequence controller admits H4 then H72 only | Designed, expected red |
| ISL-007 | `test_each_candidate_is_evaluated_on_all_four_distinct_folds` | Four receipt identities per ranked candidate | Designed, expected red |
| ISL-008 | `test_reported_fitness_is_ratio_minus_one_and_negative_is_better` | OLAP readback recomputes fitness from sums | Designed, expected red |
| ISL-009 | Stub test surface contains no TEST input | Future-perturbation and access-denial system tests | Partially designed |
| ISL-010 | `test_same_seed_replays_the_same_local_island` | Restart parity from atomic checkpoint | Designed, expected red |
| ISL-011 | Existing `test_i6n_optimizer.py` and `test_i6n_optimizer_v2.py` | Replay retained shared evidence | Existing regression coverage |
| SEQ-001 | Not in optimizer unit scope | Successor state-machine test: H72 refused before H4 terminal | Pending implementation test |
| SEQ-002 | Not in optimizer unit scope | Four-node terminal and common finalized checkpoint falsifications | Pending implementation test |
| SEQ-003 | Not in optimizer unit scope | Ledger/outbox/chain/OLAP identity equality and readback | Pending implementation test |
| SEQ-004 | Not in optimizer unit scope | Restart between every transition; exactly one H72 launch | Pending implementation test |
| SEQ-005 | Horizon acceptance tests | Separate sealed configs and transition receipt | Partially designed |
| OPS-001 | Same-seed unit test | API preflight rejects any commit, seed, network, height, or tip mismatch | Pending integration test |
| OPS-002 | Not in optimizer unit scope | Versioned systemd unit/timer installation and unattended replay | Pending system test |
| OPS-003 | Generation callbacks in patience tests | Read-only status schema and ETA mutation tests | Partially designed |
| OPS-004 | Same-seed determinism | Kill/restart parity after candidate and generation boundaries | Pending system test |
| OPS-005 | Configuration rejection precedes evaluator | GPU visibility, memory admission, and no-CPU-fallback probes | Partially designed |

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
