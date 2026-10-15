# Musashi review of support return 98ffdf28

Scope: report, market_grid candidate/materialization code, experiment plans
and focused CPU regression tests. No new host/GPU inspection, warehouse query,
full scientific replay or TEST read in this review.

Command (CPU, bounded):

```bash
crispdm-run -m 3G -t 180s -n musashi-market-grid-audit -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONPATH=. python -m pytest -q tests/i6dmo/test_market_grid.py tests/i6dmo/test_dmo_hourly.py tests/i6dmo/test_strategy_instrument_config.py
```

Result on the clean 98ffdf28 worktree: **36 passed**, 2 protobuf deprecation
warnings, 53.68 s, exit 0. The 319-test full-suite count belongs to Satoshi's
return; it was not independently rerun here. Code inspection confirmed slots
come from calendar endpoints, include t, require exact observations, retain
actual offsets/closure flags and construct targets from elapsed t+h.

Accepted: preparation only, not predictive evidence. Closed exact-DS2 scope,
negative result, 2024 development designation, absent DS12 support and explicit
cost/calendar assumptions are correctly distinguished. No reason to repeat
the support study or the completed scientific baseline.

Required qualifications carried into the successor:

1. MG2 is a superset of DS2's ORIGIN SET, not proof of identical input tensors
   or preserved predictive performance. Shared origins do not remove a change
   in training population, time metadata or preprocessing treatment.
2. Publication timestamps alone do not validate revised historical values.
   Confirmatory readiness needs value-level vintage/provenance evidence as
   well as availability times. A governed key is required only for a source
   route that actually requires one, not as a universal administrative gate.
3. SUPPORT_STUDY.source_clock.status=RESOLVED does not certify every feature's
   point-in-time provenance. Its accompanying text retains declared clock and
   vintage assumptions. New status must expose unresolved per-source gates,
   not derive global TEST readiness from this field.
4. Canonical queue/current-state documents were still describing the initial
   DS2 design as unimplemented. This dispatch synchronizes the next action
   without changing measured identities, old chains or milestone percentages.

Decision: adopt MG2/MG12 for implementation and gated development execution
under MARKET_TIME_PAIR_PLAN.md. No new model is certified by this review.
No scientific process was started by publishing these instructions.
