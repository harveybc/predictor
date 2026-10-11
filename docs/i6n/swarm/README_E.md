# Lane E: search space, sealed plan, identities (execution and status)

Status 2026-10-11: code and contracts only. **No fit was run, no data read,
no service installed.** Plan v1.1 is sealed before any fit (v1.0.0 superseded; no fit ran under it).

## What lane E owns

| File | Role |
|---|---|
| `tools/i6n_search_space.py` | v2 genotype, exact plan domain, canonicalizer with typed infeasibility, identities, size classes, seed/alternative, sampling, joint conditional mutation, legacy v1 mapping |
| `tools/i6n_search_plan.py` | sealed plan JSON, promotion, four-fold fitness, per-horizon search state (JSON, replayable), finalists, fit specs per h/class/fold, CLI |
| `optimizer_plugins/i6n_optimizer.py` | DOIN shared plugin: v2 path when `opt_config["i6n_plan"]` exists; v1 path unchanged |
| `docs/i6n/swarm/I6N_SEARCH_PLAN_v1_1.json` | sealed plan, `plan_sha256 = e7de41f04c8f188606dcaa71baa497e224303bdceaa8c802e71db63045ab788c` (v1.1) |
| `docs/i6n/swarm/E_LEGACY_MAPPING_v1.json` | 48 completed endpoint candidates mapped to v2 (sha256 `e3254dd6...7b3f`) |
| `docs/i6n/swarm/INTERFACE_E.json/.md` | published contract for A/B/C/D |
| `tests/test_i6n_search_space_v2.py`, `tests/test_i6n_search_plan.py`, `tests/test_i6n_optimizer_v2.py` | identity and domain tests |

`tools/i6n_search.py` is intentionally **frozen** (byte-identical to base): it
is the source of the canonical identity hash used by A, B and the legacy runs.

## Commands

```bash
# seal / verify the plan (no fits)
python -m tools.i6n_search_plan plan --output docs/i6n/swarm/I6N_SEARCH_PLAN_v1_1.json
python -m tools.i6n_search_plan verify --plan docs/i6n/swarm/I6N_SEARCH_PLAN_v1_1.json
# generation-0 configs for the 12 horizons (S1 admitted), with MANIFEST.json
python -m tools.i6n_search_plan initial-configs --plan <plan> --output <dir>
# one fit spec per candidate x fold, after C's units and A's freeze give bindings
python -m tools.i6n_search_plan fit-specs --plan <plan> --horizon 12 \
    --bindings <folds.json {fold_id: {experiment, fold_id, context_sha256, support_sha256}}> --output <dir>
# legacy mapping (read-only on controller STATE.json; private paths are not emitted)
python -m tools.i6n_search_plan legacy-map --state <STATE.json> --state <...> --output <map.json>
# tests (CPU)
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q \
    tests/test_i6n_search_space_v2.py tests/test_i6n_search_plan.py tests/test_i6n_optimizer_v2.py
```

## How B drives it

Either through the shared plugin (`create_shared_population` /
`evaluate_single_genome` / `reproduce_shared` with `i6n_plan`, `i6n_horizon`,
`i6n_folds{fold: {i6n_support, i6n_evaluation_context}}`,
`i6n_admitted_classes`), or directly with the JSON state functions
`new_horizon_state -> propose_generation -> record_fitness|record_disposition
-> close_generation -> select_finalists`, and `score_folds` for the four-fold
fitness. `WAIT_RESOURCE` (no admitted class) re-proposes later without
closing a generation twice. Exhaustion closes with a reason, never duplicates.

## Status of the 48 completed candidates

All 48 v1 identities recompute exactly and, under plan v1.1 (free per-layer
strides, ruled 2026-10-11), all 48 genotypes map into v2 with the SAME
canonical identity, including the mixed-stride ones such as `[1,2,1]`. None of their scores is reusable as a 2023 fold
score (single inner interval before week 2024-01-07); no refit is scheduled.
