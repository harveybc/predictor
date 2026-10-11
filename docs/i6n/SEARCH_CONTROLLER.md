# Bounded Endpoint Controller

`tools.i6n_search_controller` is a sequential local driver of the existing shared
DEAP adapter and historical subprocess evaluator. It is not a DOIN swarm, full
trajectory search, annual champion selector, or production certification. The
governing allocation is [INITIAL_ENDPOINT_ALLOCATION.md](INITIAL_ENDPOINT_ALLOCATION.md).
No controller command publishes results or launches another horizon.

## Exact Private Config

All keys below are required, unknown keys are refused. Paths are absolute on the
worker. Fill SHA fields with hashes of the actual files; do not commit private
configs, device identifiers, evidence or generated artifacts. The example is
the authorized h12 stage, not an instruction to launch it now.

```json
{
  "schema": "i6n.search_controller.v1",
  "scope": "ENDPOINT_ONLY_ENGINEERING_H12",
  "binding": {
    "path": "/var/tmp/i6n-historical-20261011/search-h12/BINDING.json",
    "sha256": "<binding-file-sha256>"
  },
  "plan": {
    "path": "/absolute/checkout/docs/i6n/INITIAL_ENDPOINT_ALLOCATION.md",
    "sha256": "<allocation-file-sha256>",
    "section": "Order and scope; Initial resource class"
  },
  "external_pilots": [],
  "output_root": "/var/tmp/i6n-historical-20261011/controller-h12",
  "seed": 6102026,
  "budget": {
    "max_candidates": 24,
    "population_size": 8,
    "generations": 3,
    "patience": 3,
    "max_proposals": 768
  },
  "admission": {
    "min_spacing_hours": 2,
    "max_samples": 25,
    "max_core_width": 32
  },
  "resources": {
    "child_memory_bytes": 5368709120,
    "child_wall_seconds": 3600,
    "total_wall_seconds": 90000,
    "min_mem_available_bytes": 8589934592,
    "poll_seconds": 2,
    "crispdm_run": "/absolute/path/to/crispdm-run",
    "admission_root": "/absolute/shared/host/crispdm/admission"
  },
  "device": {
    "cuda_visible_devices": "GPU-<explicit-device-uuid>",
    "required_gpu_name": "NVIDIA GeForce RTX 5090"
  }
}
```

The corresponding h72 config changes `scope` to
`ENDPOINT_ONLY_ENGINEERING_H72`, pins its own h72 BINDING, uses a separate output
root, and supplies the following exclusion (the complete `external_pilots`
list, not additional top-level keys):

```json
[
  {
    "receipt": {
      "path": "/absolute/path/PILOT_h72.json",
      "sha256": "<completed-pilot-result-file-sha256>"
    },
    "genome": {
      "horizon": 72,
      "multiplier": 2,
      "delta_x": 36,
      "input_transform": "identity",
      "output_mode": "endpoint",
      "delta_y": null,
      "core": {
        "entrance": "conv",
        "width": 16,
        "transformer_blocks": 0,
        "reducer_filters": [8, 4],
        "reducer_strides": [1, 1],
        "adapter": "none",
        "adapter_width": null,
        "n_horizons": 1
      }
    }
  }
]
```

For an h2 run, exclude the h2 pilot similarly, with horizon 2, multiplier 1,
delta_x 2 and its actual pinned RESULT. The controller checks the declared
genome against the COMPLETED pilot's input SHA, model, sample count, spacing,
horizon, training settings and evaluation week. This is conservative exclusion,
not checkpoint adoption or incorporation of the pilot's outer score.

Explicit secondary admission is also supported: GPU name
`NVIDIA GeForce RTX 4090 Laptop GPU`, its own UUID, child memory `3221225472`
(3 GiB), and fresh host minimum `6442450944` (6 GiB). This is a separate declared
probe class, never automatic fallback or a lowered refused request. No 5070 is
allowed. The controller runs one child locally; cross-host scheduling belongs
to main and must match the pinned allocation. Each controller needs a separate
binding/output root. No same-host concurrent GPU work is authorized here.

CPU verification uses the base cell's `device: "cpu"`, empty
`cuda_visible_devices`, `required_gpu_name: null`, and at most 2 GiB per child.
This mode is for tiny generated fixtures, not a historical CPU fallback.

## Integration

1. Main prepares and freezes each real historical binding using the historical
   evaluator. Keep data, clock, full four-year TRAIN/28-day inner support, seed,
   and TRAIN-only scales fixed. The base cell parameter cap must be <=1,000,000.
2. Write the private controller JSON. The base model seed must equal the
   controller seed; GPU stages use the allocated seed 6102026. Bind the current
   allocation document after finalizing placement. The admission store must
   equal the launcher's shared host store, not a new private store.
3. Check without fitting:

   ```sh
   python -m tools.i6n_search_controller check --config /absolute/private-h12.json
   ```

4. Main launches `run` in a durable service with a **512 MiB controller cap**.
   The process must be outside the child's cgroup. The installed `crispdm-run`
   moves each heavy child into its own resource-admitted scope, with its own
   fixed memory ceiling, no swap, and wall timeout. Do not set a 512 MiB ancestor
   limit on the entire slice containing the separate heavy scopes.

   ```sh
   python -m tools.i6n_search_controller run --config /absolute/private-h12.json
   ```

5. Status is read-only and never evaluates a genome:

   ```sh
   python -m tools.i6n_search_controller status --config /absolute/private-h12.json
   ```

The parent retains no TensorFlow model. A resource-admitted worker calls the
shared adapter with the historical evaluator's JSON subprocess command. Its
timeout is the cell wall setting +120 seconds; the outer child's 3600-second
hard cap still wins. The stage has a persisted absolute deadline (90000 seconds
allows a 24-hour allocation plus one hour setup allowance). It will not start a
child unless the full child wall budget plus 60 seconds cleanup fits. Restart
does not reset the deadline. No busy polling agent or unbounded resource queue
is required.

## Reservations And Receipts

Conditional core/window proposals, DEAP minimization tournaments, 80% mutation
and 20% immigrants use the existing adapter/search logic. True trajectories,
external pilot identities, and out-of-class proposals are explicitly filtered
before candidate reservation. They consume the finite proposal-attempt bound,
not the 24 unique candidate budget. Filter reasons and canonical genomes are
retained in `STATE.json`. Entrance, reducer and non-null adapter widths must
all fit `max_core_width`. No genes are silently rewritten to endpoint.

H12 admits distinct 24/48/72-hour histories; h72 admits 72/144/288/432-hour
histories when exact spacing and sample limits fit. H2 remains supported but
all its multipliers collapse to 24 hours, so h2 is not a context comparison.
This small bounded class is a resource probe, not a measured production class.

The shared tracker durably reserves the entire proposed population before any
fit. Each started candidate additionally records RUNNING before spawning and
claims an exclusive `STARTED` marker. Failed candidates consume their reservation
and have no fabricated fitness. No candidate is retried or padded into the next
generation. A crash with an unreconciled RUNNING candidate stops the stage;
the controller does not guess whether its child finished or start another fit.

Successful results must match the actual bridge identity, checkpoint/C2 artifact
digests, canonical effective genome, original-price TRAIN/inner support and
shared-adapter fitness. Retained admission evidence must match the exact child
argv SHA and resource cap. Its sampled whole-scope `observed_peak_bytes` is
retained as such, never relabeled RSS or an exact kernel peak. Missing peak,
admission refusal, timeout or resource stop prevents continuation. Caps are
never lowered, failed fits never retried, and external pilot checkpoints are
never adopted. Restart checks retained candidate artifacts and resource outcomes.

`STATUS.json` carries active bridge epoch/update progress and ETA plus an
explicitly estimated remaining-stage ETA. `STATE.json` is the digest-sealed
authoritative journal. `candidates/<effective-id>/` contains `WORKER.json`,
`RESOURCE.json`, `worker.log`, and `LOCATOR.json` pointing to native bridge
artifacts. `TERMINAL.json` binds terminal status, identity and the final state
digest. A terminal disposition describes only this bounded stage, never an
annual winner. Main's independent publisher can follow the native bridge C2
receipts; publication/readback must never call the evaluator again.

## Test Traceability

Tests were written before implementation. Unit/integration coverage includes
exact config and pins, allocation limits, pilot aliases, trajectory/cost filters,
bounded adapter reproduction, failed budgets, no-repeat terminal/interrupted
resume, memory/wall refusal, missing resource evidence, state tampering, capped
subprocess argv, explicit GPU admission, and h12/h72 contexts. A tiny generated
CPU case exercises the actual isolated worker, shared adapter and historical
bridge, including real fitting, C2 and terminal reuse. Launcher accounting is
stubbed only in that test; no historical run or GPU launch is performed by tests.
Resource integration remains a deployment responsibility, not a claimed result
from these generated fixtures. TEST, trading, and full-space claims stay closed.

Verification on 2026-10-11: controller, shared optimizer, and search tests
**159 passed in 30.74 seconds**, CPU-only under `crispdm-run -m 2G -t 120s`.
This includes the regression for a completion saved before resource validation:
resume refuses the bad outcome before starting another candidate. Method stage
is component/integration verification; operator deployment and historical stage
execution remain main's next actions. Only this controller, its tests and this
document were changed for the task. No historical search, GPU fit, publication,
network action or commit was performed.
