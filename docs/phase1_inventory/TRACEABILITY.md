# Phase-1 orchestration traceability

| Requirement | Observable acceptance | Evidence |
|---|---|---|
| P1-INV-01 Canonical denominator | Exact EURUSD filters resolve 366 unique IDs; any other count refuses | `test_canonical_eurusd_inventory_resolves_exactly_366_features`, `test_denominator_mismatch_is_refused` |
| P1-INV-02 Independent populations | EURUSD 366 and ETH 83 use separate identities and state roots | `test_population_templates_are_independent` |
| P1-PLC-01 Cost-aware deterministic placement | Repeated plans are byte-equivalent; the cheapest fraction belongs to the small host | `test_assignment_is_deterministic_and_keeps_smallest_on_small_host` |
| P1-EXE-01 One process per host and feature | Competing runners execute the worker once | `test_exclusive_claim_prevents_duplicate_execution` |
| P1-REC-01 Restart recovery | A stale claim is archived and its feature completes | `test_stale_claim_is_archived_and_work_is_recovered` |
| P1-XPORT-01 Remote-safe transport | SSH-like prefix receives JSON over stdin and no coordinator path | `test_remote_prefix_uses_stdio_and_never_coordinator_paths` |
| P1-ABS-01 Explicit absence | Missing input becomes `UNAVAILABLE`, never a silent omission or zero | `test_unavailable_input_gets_explicit_closure_terminal` |
| P1-RETRY-01 Failures are not science | Worker failure and rc75 retry boundedly and cannot close | `test_failures_retry_boundedly_and_never_satisfy_closure` |
| P1-OLAP-01 Warehouse delivery | Every completed envelope has a matching durable receipt | `test_finalization_waits_for_every_receipt_and_authenticated_readback` |
| P1-OLAP-02 Authenticated readback | Forged or incomplete reconciliation cannot open the gate | `test_forged_or_incomplete_readback_cannot_open_gate` |
| P1-GATE-01 Phase-2 prohibition | Gate refuses before full eligible closure | `test_phase2_gate_requires_complete_population` |

Structural controls include exact schemas, content-addressed feature keys,
plan/terminal/gate digest recomputation, shell-free command execution and
separate worker/finalizer contracts. Behavioral controls cover deterministic
placement, contention, crash recovery, absence, CPU isolation, delivery and
the phase boundary.
