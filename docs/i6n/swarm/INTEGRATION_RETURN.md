# Integration A return (launcher and real runtime), 2026-10-12

Orders `SATOSHI_I6N_EXECUTE_NOW_20261012.md` (sha256 b78ec898…), sections 1, 2.A, 3, 4.
Branch `satoshi/i6n-swarm-integration-A-20261012` from `23dbc81c`; code pin `68e6a7e2`
(the branch tip only adds this file and evidence). Machine-readable: `READINESS.json`.

## What was integrated

- Musashi's `2101d97e`: `5bff3b21` request relay (the request is written to a private 0600
  file whose sha256 is checked by `request_exec.py` INSIDE the admitted scope; fencing,
  orphan guard and caps unchanged), `75e4f749` trajectory-aware retained-result verification
  and all-head publication, `2df243fc` preflight and S3 buffer bounds. `e242c846`'s config
  grammar was folded into B's E-B1 path: ONE renderer, the sealed E plan read by digest and
  verified in production (`verify_plan`).
- Lane A.4 `0f05dd78`: freeze always runs with `--data-root` (v3 host-relative contexts) and
  `I6N_DATA_ROOT` set for freeze and evaluate; production refuses v2 absolute bindings;
  per-week family bindings (`freeze --family`, `--score-outer`) as `freeze_family:{short,long}`
  DAG nodes; multi-output arms built with A's `multi_output_arm_config` from inner-fold
  finalists via C's `resolve_arms`; collect reports per-head `DIRECT_HEAD` rows, pooled sums
  only as `POOLED_DIAGNOSTIC`.
- Deployment requests: class caps `ceil(1.25 × (peak + coverage_adjustment_bytes))` with a
  derivation digest (never lowers); production cell pins root-relative; per-host
  `shared_data_root`; default `shared_min_peers` = number of other peers.
- Cross-host bytes: freeze publishes FROZEN/BINDING/cell bytes to the ledger's
  content-addressed store; before any claim a host verifies each context pin against its
  own bytes or fetches them. `tools/i6n_swarm/crosshost_verify.py` compares the data root and
  code pin on every inventory host (host names never printed).

## Requirement → test → result

| Requirement | Test / evidence | Result |
| --- | --- | --- |
| Request relayed inside admission | `test_request_survives_launcher_without_inherited_stdin`, cost-pilot tests | 65 passed |
| One v2 grammar, sealed plan | `test_i6n_swarm_v2_wiring.py` (3), E reproducer r2 | passed; B1–B6 as expected |
| A.4 v3 freeze, I6N_DATA_ROOT, v2 refused | `test_freeze_is_always_host_relative…`, `test_production_refuses_a_v2_absolute_binding`, `test_evaluate_subprocess_gets_this_hosts_data_root` | red before (`PRE_a4_wiring.txt`), green |
| Direct multi-output arms, DIRECT_HEAD rows | `test_multi_output_arms_use_a4_config…`, `test_collect_reports_direct_head_rows…`, A arm suite | green; arm suite 14 passed in 85.7 s |
| Coverage adjustment | `test_class_cap_coverage_adjustment_raises_never_lowers` | green |
| Cross-host bytes | `test_frozen_context_bytes_reach_another_host…`, `test_bridge_waits_without_claiming…` | red before (`PRE_context_bytes.txt`), green |
| Root-relative cells, per-host roots | `test_production_cell_pins_are_root_relative…` | green |
| Publication | search_publish / publish / archive | 26 passed, 1 skipped |
| Wrapper (doin-plugins 566c5af) | wrapper + predictor plugins, doin-core 4a321cd enforced | 24 passed |
| Runtime (doin-node df6aad0) | runtime + shared contract + unified + offchain shadow | 103 passed |
| Two real DOIN nodes | `python -m tools.i6n_swarm.smoke` (synthetic evaluators) | PASSED: 96 units, 92 evaluator calls + 4 adopted, 0 duplicates, A 48 / B 44 |
| Late join | `… smoke --late-join 20` | PASSED: B entered generation 1, 16 fits, 0 duplicates |
| Cross-host availability | `crosshost_verify` | data root and code pin ABSENT on both workers (not yet deployed) |

The smoke exposed two runtime defects, both fixed with regressions red first: a peer's
rejected result arrives with `fitness: null` (inf is not JSON) and crashed persistence and
generation completion (`PRE_runtime_rejected_peer.txt`, doin-node `0f6e9eb`); a late joiner
created a stale genesis and waited forever at its barrier (doin-node `df6aad0`, doin-plugins
`566c5af`: it resumes the current generation from the ledger search state).

## Exact commands

All CPU (`CUDA_VISIBLE_DEVICES=""`), one job at a time; evidence in `evidence_integration_A/`.

```bash
# predictor (68e6a7e2)
crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_swarm_ledger.py tests/test_i6n_swarm_runner.py tests/test_i6n_swarm_lease_service.py tests/test_i6n_swarm_placement.py tests/test_i6n_cost_pilot.py
crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_swarm_bridge.py tests/test_i6n_swarm_review_e.py tests/test_i6n_swarm_v2_wiring.py tests/test_i6n_swarm_integration_a4.py
crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_swarm_campaign.py tests/test_i6n_swarm_agent.py
crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_multi_output_arm.py     # alone (85 s)
crispdm-run -m 2G -t 120s -- env PYTHONPATH=.:<doin-core>/src:<doin-node>/src:<data-lake>:<data-warehouse> python -m pytest -q tests/test_i6n_search_publish.py tests/test_i6n_publish.py tests/test_i6n_archive.py
crispdm-run -m 2G -t 120s -- env PYTHONPATH=.:<doin-plugins>/src:<doin-core>/src:<doin-node>/src python review_e_of_b_r2.py
# doin-plugins (566c5af)
DOIN_CORE_EXPECTED_COMMIT=4a321cd PYTHONPATH=<predictor>:src:<doin-core>/src:<doin-node>/src crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_swarm_wrapper.py tests/test_predictor_plugins.py
# doin-node (df6aad0)
PYTHONPATH=src:<doin-core>/src:<data-lake>:<data-warehouse> crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_swarm_runtime.py tests/test_shared_population_contract.py tests/test_unified.py tests/test_offchain_shadow.py
# two real DOIN nodes (bounded; synthetic evaluators)
PYTHONPATH=<doin-plugins>/src:<doin-core>/src:<doin-node>/src crispdm-run -m 3G -t 600s -- python -m tools.i6n_swarm.smoke --root <new dir> --timeout 480 [--late-join 20]
# cross-host bytes (after deployment, on the coordinator)
python -m tools.i6n_swarm.crosshost_verify --pin <deploy commit> --data-root /var/tmp/i6n-swarm-data-20261012
```

The smoke runs one 3 GiB / 600 s scope (two node processes, ledger service and synthetic
children together); it is not a 2 GiB unit suite.

## Findings for other owners

- **C (defect):** plan-v5 `PLAN.json` digest `40d3f5a0` does not seal cell bytes (0/681 cell
  digests in PLAN/ARTIFACTS), so raising the resource guards did not change it. The guards are
  in A's candidate identity and in `VIEWS.private.json` (681/681 digests), which B pins by
  sha256, so B's fit identity is safe. C should seal cell digests into the plan.
- **Deployment:** the data root and the code pins are not on either worker yet; rerun
  `crosshost_verify` after replication. Declare `VALIDATION_OUTER_WEEK` and
  `DIRECT_MULTI_OUTPUT_ARM`. Use doin-node `df6aad0` and doin-plugins `566c5af`, not the
  deps-directory copies.

## Not done

No production fit, unit, service or timer was started or touched (S3 pilot and
`i6n-launch-cost-collect.timer` untouched). Multi-output runs only once finalists exist.
