# Phase-1 orchestration traceability

| Requirement | Observable acceptance | Evidence |
|---|---|---|
| P1-INV-01 Canonical denominator | Exact filters resolve 366 unique IDs; any other count refuses | `test_canonical_inventory_filter_resolves_exactly_366_features`, `test_denominator_mismatch_is_refused` |
| P1-PLC-01 Cost-aware deterministic placement | Repeated plans are byte-equivalent; the cheapest fraction belongs to the small host | `test_assignment_is_deterministic_and_keeps_smallest_on_small_host` |
| P1-EXE-01 One process per host and feature | Competing runners execute the worker once | `test_exclusive_claim_prevents_duplicate_execution` |
| P1-REC-01 Restart recovery | A stale claim is archived and its feature completes | `test_stale_claim_is_archived_and_work_is_recovered` |
| P1-ABS-01 Explicit absence | Missing input becomes `UNAVAILABLE`, never a silent omission or zero | `test_unavailable_input_gets_explicit_terminal` |
| P1-CPU-01 CPU-only phase | Worker observes no CUDA device | `test_worker_is_cpu_only_and_envelope_is_submitted` |
| P1-OLAP-01 Warehouse delivery | A terminal envelope receives a durable receipt | `test_worker_is_cpu_only_and_envelope_is_submitted` |
| P1-GATE-01 Population finalization | Finalizer cannot run before all planned IDs have terminals | `test_finalizer_does_not_run_while_any_item_lacks_a_terminal` |
| P1-GATE-02 Phase-2 prohibition | Gate refuses before closure and authenticates after finalization | `test_finalizer_and_phase2_gate_require_complete_denominator` |

Structural controls include exact schemas, content-addressed feature keys,
plan/terminal/gate digest recomputation, shell-free command execution and
separate worker/finalizer contracts. Behavioral controls cover deterministic
placement, contention, crash recovery, absence, CPU isolation, delivery and
the phase boundary.
