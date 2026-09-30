# QRM02 / F3 — the allocator that measures the model that actually trains, and stops that stop

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-29 (America/Bogota).
Branch `satoshi/qrm02-f3-tensorflow-telemetry-20260929`, own worktree off `1fb6b387`.
Specification: **F3** of `docs/audits/work_plan/MUSASHI_AUDIT_23B2EFA3_2026_09_29.md`.
Lane B of the dictamen's parallel orders. It does not wait on lane A and does not touch lane A's files.

**The one-line result.** The figure that measures Q2's model is
**`tf.config.experimental.get_memory_info(device)`**, reset per stage with
`tf.config.experimental.reset_memory_stats(device)` — TensorFlow's own allocator, since the recipe is
TensorFlow's — and **a GPU pilot is not possible under the current runner**: with
`CUDA_VISIBLE_DEVICES=""` the instrument refuses by name (`NO_VISIBLE_DEVICE`), and even with the
device pinned by UUID the admitted worker's TensorFlow **registers no device at all** until the child
is given that environment's own CUDA library path. All three states were executed on the admitted
worker. And Traffic's remaining horizons **cannot be priced in hours** without a run: what arithmetic
settles exactly is their batch geometry and every parameter-driven byte; the per-step **time** is
`UNKNOWN`, so **the 12.08 – 15.02 GPU-hour figure is withdrawn as a bound** and the established number
is **6.62 GPU-hours over the six measured-horizon cells, with six cells unpriced**.

---

## 0. Stated before dispatch

| | |
|---|---|
| worktree | `<checkout>/.worktrees/predictor-qrm02-f3-20260929` |
| branch | `satoshi/qrm02-f3-tensorflow-telemetry-20260929` |
| source tip | `1fb6b387` on `satoshi/qrm02-train-pilot-design-20260929` (the design F3 lands on) |
| read, not changed | `c1033dc6` (lane A's instrument and runner), `91a4c410` (the Traffic train pilot) |
| first command | `python -m pytest tools/test_df_tf_device_telemetry.py -q` — **red first**, `ModuleNotFoundError: No module named 'df_tf_device_telemetry'` |
| acceptance test | the two new suites green (**35 + 32 = 67**), and the three runner states executed on the admitted worker with the verdicts of §3 |
| budget | ten children through the deployed `crispdm-run`, caps 2–6 GiB, wall ≤ 420 s each, fresh aggregate admission per child; four diagnostic TensorFlow imports outside the launcher, disclosed in §8 |
| blocker | **`NO_PROGRAMME_TASK_IS_READY` for the QRM02 pilot.** F3 is repaired and instrumented; the *runner* is lane A's, the GPU environment contract is not in it, and the §7 allocation of the design is still not held |

---

## 1. F3, and the three defects in one sentence

The dictamen:

> El modelo de Q2 es TensorFlow/Keras (`df_e1_block.py:602-625`, entrenamiento en `:693`).
> `df_cell_scope.py:285-319` consulta exclusivamente el asignador de PyTorch. QRM02 en `1fb6b387` se
> basa en esas APIs para vigilar 12 GiB. El asignador de PyTorch no contabiliza las asignaciones
> TensorFlow. Ademas `run_units:976` fuerza `CUDA_VISIBLE_DEVICES` vacio; sellar ese runner no
> establece un piloto GPU.

All three are confirmed at the named lines, and they are three different errors of mine:

1. **Wrong framework.** `df_cell_scope.py::gpu_memory` imports `torch` and returns
   `torch.cuda.max_memory_allocated` / `max_memory_reserved`. Q2's model is built by
   `build_modular` as a `tf.keras.Model` and trained through `model.train_on_batch`. PyTorch's
   caching allocator accounts for **none** of that. A 12 GiB envelope watched there is watched by an
   instrument that cannot see the memory it claims to watch.
2. **Wrong placement.** `run_units` passes `env={**os.environ, "CUDA_VISIBLE_DEVICES": "", …}` to
   every child. Sealing that runner establishes a **CPU** pilot. The design's device rows were
   unreachable from the runner it sealed against, and I did not notice because I never ran one.
3. **A promise a statistic cannot keep.** §3 said the child *"raises and exits non-zero before
   allocating past the envelope"*. A peak is read **after** the allocator already held the memory.

The repair keeps the TensorFlow recipe **unchanged**. No model was ported to another framework, and
PyTorch is not imported to measure TensorFlow — a test asserts both against the source.

---

## 2. The allocator API that measures the model that actually trains

`tools/df_tf_device_telemetry.py`. One instrument, four scopes, and every figure names its API.

| what | API | basis |
|---|---|---|
| **framework, device** | `tf.config.experimental.get_memory_info("GPU:0")` → `{"current","peak"}` | device memory held by **TensorFlow's** allocator on the verified device |
| stage reset | `tf.config.experimental.reset_memory_stats("GPU:0")` | the next peak is the peak **since that reset**; without it the reading is a lifetime watermark and says so |
| **device UUID** | `libcuda.cuDeviceGetUuid` through `ctypes` | the CUDA **driver**, not a deep-learning framework: the witness both the framework and the pinning act on |
| **whole device** | `libcuda.cuMemGetInfo` | in use by **every** process, including this one's context. Its own scope |
| **process** | `getrusage(RUSAGE_SELF).ru_maxrss` | host RAM, one process |
| **host tree** | `memory.peak` of the cell's own cgroup | host RAM, whole process tree, as the kernel charges it |

**Unavailable is `UNKNOWN`.** Verified on the coordinator (TensorFlow 2.18.0): `get_memory_info("CPU:0")`
raises `ValueError: Allocator stats not available for device 'CPU:0'`, and the record carries
`status: UNKNOWN`, `peak_bytes: null` and that exact reason. Not 0, not borrowed from another scope.

**A version difference the running found, which matters.** On the worker (TensorFlow 2.21.0) the same
call on `CPU:0` **succeeds** and returns TensorFlow's **host** allocator statistics. That is a *third*
memory, not the device one, so the instrument labels it `FRAMEWORK_ALLOCATOR/HOST` and — this was a
real defect in my first cut, found by running it — **refuses** to compare a host-allocator figure with
a device envelope. Before the fix it answered `exceeded: false`, a pass earned by measuring the wrong
memory.

**Four scopes on one real record** (worker, device verified, 12 GiB arena predeclared):

| scope | bytes | what it is |
|---|---:|---|
| framework allocator, DEVICE | **67 110 400** | the 64 MiB probe tensor plus 1 536 B, held by TensorFlow on the 4090 |
| whole device | **13 412 270 080** of **16 718 168 064** | every tenant, including the predeclared arena and this process's context |
| process resident set | **1 434 992 640** | host RAM, this process |
| host cgroup peak | `UNKNOWN` | not read in this self-check record: **absent, not zero** |

No field of that record is the sum of any two of them, a test enumerates every integer in it to prove
so, and `merge_scopes` exists only to raise. The two device figures differ by three orders of
magnitude for the same allocation, which is exactly why they may not be interchanged: one is what
TensorFlow held, the other is what the device carried.

---

## 3. Is a GPU pilot even possible under the current runner? No

Three states, each executed on the **admitted secondary worker** (its RTX 4090,
`GPU-a8bd1b2c-…`, 16 718 168 064 B), each a fresh aggregate admission through the deployed
`crispdm-run`, 6 GiB cap, device envelope requested at **12 GiB**.

| state | verdict | envelope | what the pilot would have been |
|---|---|---|---|
| **A. `CUDA_VISIBLE_DEVICES=""`** — the environment `run_units:976` actually passes | **REFUSED · `NO_VISIBLE_DEVICE`** | `NOT_ENFORCED` | a **CPU** pilot. The driver is never asked; nothing GPU exists to measure |
| **B. pinned by UUID, nothing else** | **REFUSED · `FRAMEWORK_REGISTERED_NO_DEVICE`** | `NOT_ENFORCED` | a CPU pilot again. The driver **does** expose the 4090 and its UUID **matches** the declared one, and TensorFlow 2.21 still registers **zero** GPUs: `Cannot dlopen some GPU libraries … Skipping registering GPU devices` |
| **C. pinned by UUID + that environment's own CUDA library path** | **MEASURED · establishes a GPU pilot** | **`ENFORCED_AT_ALLOCATION`**, 12 288 MiB | a real GPU pilot: UUID verified through the driver, the device registered, an op landed on `GPU:0`, and an oversized request **failed with `ResourceExhaustedError`** |

In state B every library loads individually by explicit path — `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`, `libcufft.so.11`, `libcurand.so.10`, `libcusolver.so.11`, `libcusparse.so.12`,
`libcudnn.so.9`, `libnvJitLink.so.12`, `libcuda.so.1` all load — so this is a **search-path** state,
not a missing dependency: that environment carries both `cu12` and `cu13` NVIDIA wheels and
TensorFlow's own probe does not find the `cu12` set. **I installed nothing, changed no environment and
repaired no service.** State C reaches the device with a **child environment variable** and nothing
else, which is what the runner must pass.

**The environment contract the runner must pass**, and which the child must verify rather than trust:

```
CUDA_VISIBLE_DEVICES = GPU-<the declared device uuid>      # exactly one entry, never empty
LD_LIBRARY_PATH      = <that interpreter's own nvidia/*/lib directories>
CRISPDM_DEVICE_UUID  = GPU-<the same uuid>                 # declared, then verified in the child
```

`verify_declared_device` refuses, by name, every way this can be wrong: `NO_VISIBLE_DEVICE`,
`AMBIGUOUS_PINNING`, `DRIVER_EXPOSES_NO_DEVICE`, `DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED`,
`FRAMEWORK_REGISTERED_NO_DEVICE`, `PLACEMENT_NOT_ON_DECLARED_DEVICE`,
`DECLARED_DEVICE_WITH_CPU_PLACEMENT`. A refusal sets `establishes_a_gpu_pilot: false`, and the record
says the pilot is a CPU one **whatever else it contains**. No silent downgrade.

**Changing `run_units` is lane A's file**, and this lane does not touch it. The contract above,
the verifier and its evidence are delivered; the placement repair belongs in the runner lane A is
already repairing for F1 and F2.

**One sizing consequence, measured.** With a 12 GiB arena predeclared on the 4090, whole-device in use
is **13 412 270 080 B of 16 718 168 064 B** — the arena is reserved up front, leaving ~3.07 GiB for
contexts and workspaces outside it. A 12 GiB device envelope is therefore **admissible on that device
and only that one**: the coordinator's own accelerator carries 8 175 878 144 B in total, so 12 GiB is
impossible there, and the preferred host is ineligible. The design now says so.

---

## 4. The two sentences of mine, corrected

### 4.1 "The child raises and exits non-zero **before** allocating past the envelope" — false as written

A statistics check performed afterwards cannot promise that. `get_memory_info` reports a peak the
allocator **already** reached; a comparison against it can only say that it happened.

What **is** enforceable, and now is: `tf.config.set_logical_device_configuration(gpu,
[LogicalDeviceConfiguration(memory_limit=12288)])`, **predeclared before the device is initialized**.
TensorFlow's allocator then refuses a request past the arena **where the request is made**. Executed
twice on real devices; TensorFlow's own allocation summary prints `Limit: 1073741824` for a 1 GiB
arena and then `RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[1073741824] … by allocator
GPU_0_bfc`. The failure is the proof.

Its two limits, stated rather than hidden: it bounds **the TensorFlow arena**, not the CUDA context or
the workspaces outside it (whole-device scope, reported separately), and asked for after device
initialization it **raises** — in which case the instrument returns `REFUSED / NOT_ENFORCED` instead
of continuing unbounded.

So the design now carries two different labels and never one word for both:

| claim | mechanism | class |
|---|---|---|
| an oversized device allocation **fails where it is made** | predeclared allocator arena | **`ENFORCED_AT_ALLOCATION`** |
| the framework peak **exceeded** the envelope | a comparison of statistics | **`OBSERVED_AFTER_THE_FACT`**, `does_not_prevent: true` |

The refuted sentence is kept verbatim in the source as `REFUTED_ABORT_BEFORE_CLAIM`, with why, and a
test refuses any abort-before phrasing anywhere else in the module.

### 4.2 "CPU and wall limits, and every stage, need an executable stop" — built

| stop | mechanism | enforced by | proved by |
|---|---|---|---|
| **CPU** | `RLIMIT_CPU` via `setrlimit`; `SIGXCPU` at the soft limit, **`SIGKILL` at the hard one** | **kernel** | a child that spins is killed within its budget; the exit is non-zero |
| **wall** | an in-child watchdog thread that calls `os._exit`, inside the launcher's own `-t` limit | process, with a **supervisor** outer stop that does not trust the child | a child sleeping 120 s with a 1 s stop dies in under 30 s and never prints `SURVIVED` |
| **each stage** | `setitimer(ITIMER_REAL)` for the stage wall and a CPU poll that exits; the CPU budget is also checked **at entry**, so a stage whose budget is spent never starts | process | a hanging stage dies **inside** the stage and the exception names it; a stage entered with its budget spent does not execute its body |
| **host envelope** | the cgroup's `MemoryMax`, set by the launcher | **kernel** | the launcher's existing behaviour; unchanged here |
| **device envelope** | the predeclared allocator arena of §4.1 | **framework allocator** | `ResourceExhaustedError` on the oversized request, twice, on two devices |

`stops_declaration` refuses to list a budget with no mechanism, and every row carries
`executable: true` only because a real child dies in the suite.

---

## 5. Traffic's remaining horizons, without a monotonicity assumption

`tools/df_tsl_remaining_horizons.py`. The dictamen: *"La horquilla 12-15 h sigue siendo proyeccion, no
cota garantizada por dos horizontes."* Agreed, and it was my own §9.1 objection to my own delivery.

**What arithmetic settles exactly, with no run and no device.**

*The loader.* The author's split borders over the registered 17 544 hourly rows reproduce the sealed
characterization's batch counts at **all four** horizons, on **all three** splits — including the two
nobody ran and the test loader neither pilot opened:

| horizon | train batches | vali batches | test batches | class |
|---|---:|---:|---:|---|
| h96 | 756 | 104 | 214 | reproduces the sealed counts |
| **h192** | **750** | **98** | **208** | **`DERIVED_EXACT_FROM_THE_LOADER`** |
| **h336** | **741** | **89** | **199** | **`DERIVED_EXACT_FROM_THE_LOADER`** |
| h720 | 717 | 65 | 175 | reproduces the sealed counts |

*The declaration.* `pred_len` enters the model in exactly one place —
`models/TimeFilter.py:57`, `self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)` — and
the backbone is built from `seq_len * n_vars // patch_len`, never from `pred_len`. A `Linear` of that
shape is affine in `pred_len` **by construction**, with slope `d_model × num_patches + 1 = 513`
parameters per output step. So these are algebra, and the algebra reproduces **both measured records
to the byte**, which checks the derivation rather than justifying it:

| | h96 (measured) | **h192** | **h336** | h720 (measured) |
|---|---:|---:|---:|---:|
| parameters | 7 306 748 ✓ | **7 355 996** | **7 429 868** | 7 626 860 ✓ |
| gradient bytes | 29 164 928 ✓ | **29 361 920** | **29 657 408** | 30 445 376 ✓ |
| Adam slot bytes | 58 330 040 ✓ | **58 724 024** | **59 315 000** | 60 890 936 ✓ |
| checkpoint bytes | 29 248 117 ✓ | **29 445 109** | **29 740 597** | 30 528 565 ✓ |
| head-output tensor | 5 296 128 | **10 592 256** | **18 536 448** | 39 720 960 |

The head-output tensor is labelled `COMPONENT_OF_A_PEAK_NOT_A_PEAK`: allocator arenas and
fragmentation are not the sum of tensor algebra. If the measured parameter slope and the declared head
slope ever disagreed, `static_terms` **refuses** rather than pricing from a law that contradicts the
code — a test forces that path.

**What cannot be priced without a run, and is therefore `UNKNOWN`.** The per-step and
per-validation-batch **times** at h192 and h336. A GPU's per-step cost is set by the kernels selected
for the shape actually launched and by occupancy at that shape; both change in steps, not smoothly, so
two other shapes do not determine it. `per_step_seconds(192)` returns `seconds: None`,
`status: UNKNOWN`. No number.

**Therefore:**

| quantity | value | class |
|---|---|---|
| the six **measured-horizon** cells (3 seeds × h96, 3 × h720) | **6.6212 GPU-hours** | derived from each horizon's own measured rates and the loader's own counts |
| the six cells at h192 and h336 | **`UNKNOWN`** | `UNPRICED_WITHOUT_A_RUN` |
| **the twelve-cell total** | **`UNKNOWN`** | six priced cells and **two holes**, not a bracket |
| 12.08 – 15.02 GPU-hours (`91a4c410` §4) | **WITHDRAWN AS A BOUND** | survives only as a projection conditional on monotonicity, and only to a caller who names the assumption |

`conditional_bracket()` **raises** unless called with `assume_monotone_in_pred_len=True`; asked for
properly it reproduces 12.08 and 15.02 exactly and is stamped
`CONDITIONAL_ON_AN_UNPROVEN_ASSUMPTION`, `not_a_bound: true`. It is **absent** from the default
record. The published 0.930 and 1.277 cell-hours at the measured horizons are reproduced to three
decimals by this module's independent arithmetic, which is how the chain is checked.

**The minimal measurement, named instead of invented.** Two 30-step children, h192 and h336, at the
**8 GiB cap already sealed** at `8d461628` (the worst measured training peak over both pilots is
2 179 903 488 B, so 8 GiB is ample and is **not** lowered), ≤ 600 s wall and ≤ 600 s CPU **in total**
— the two children that already ran took 15.41 s and 22.62 s. It buys a measured rate at both
horizons and nothing else: no accuracy, no test split, no selection, **no campaign**.
**Authority: NOT HELD.** The Traffic authorization named h96 and h720. So they are not run, their
hours stay `UNKNOWN`, and this is a request, not a plan.

---

## 6. Authority, and the resources the dictamen did not grant

**QRM02 is not approved against `c1033dc6`.** The dictamen withholds approval until F1 and F2 are
repaired, after which the finite 4 800 CPU s / 4 800 s wall request is **re-evaluated**. Nothing here
claims that approval, and F3's repair does not confer it: the design at revision 4 is still
`NOT READY FOR DISPATCH`, now with the runner placement named as blocker 0.

**The 18 GiB service-level request stays unapproved — and against the current state it is refused.**
Re-admitted, non-mutatingly, through the deployed launcher's own admission module on the secondary
worker (its `state` subcommand, which writes nothing), 2026-09-29:

| reading | bytes |
|---|---:|
| slice ceiling | **15 032 385 536** (14 GiB) |
| slice in use | 1 975 861 248 |
| free for new work | **18 592 571 392** |
| MemAvailable | 21 813 796 864 |
| desktop reserve held by policy | 3 221 225 472 |
| live reservations / unrealised | **none / 0** |
| memory pressure (full avg10) | 0.0 |

An 18 GiB cap (19 327 352 832 B) **exceeds the 14 GiB slice ceiling**, which the module treats as
`ABOVE_SLICE_CEILING` — a **terminal** refusal, not a queue. It would also not fit the 17.32 GiB free
for new work. The only route to admitting it is raising a ceiling, and **I did not raise one, will
not, and no exception is claimed here.**

**The residual, identified, with its owner named.** From a second, non-mutating reading of
`/proc/meminfo` and the process table on the same host — **a different instant** from the admission
table above, as the dictamen itself insists about its own two readings. MemTotal
**32 831 807 488 B**, MemAvailable **21 406 572 544 B** at that instant. The difference is not idle:

| holder | bytes | disposition |
|---|---:|---|
| **the MT5 paper-trading VM** (`qemu-system-x86`, one process) | **5 785 092 096** resident | a live service of another lane. **Not displaced, not touched** |
| kernel unreclaimable slab (`SUnreclaim`) | 1 665 089 536 | the kernel's. Not discardable |
| **shared memory** (`Shmem`) | 1 195 458 560 | **inside `file`/`Cached`; never added twice and never cleared** |
| reclaimable slab (`SReclaimable`) | 606 433 280 | reclaimable, and not reclaimed by me |
| the owner's desktop session (shell, indexer, mail, network agents) | ~0.3 GiB | left alone |

So the dictamen's correction is confirmed: **the residual is no longer 4.9 – 5.1 GiB**, and **not all
of the `file` accounting is discardable cache** — the shared-memory share sits inside it. `/dev/shm`
holds two 4 KiB tracing files and nothing else, so the shared memory is anonymous-shared and tmpfs
elsewhere, owned by the VM and the desktop session. **No shared memory was cleared, no reservation was
reduced to pass, no ceiling was changed, no service was restarted and no host was rebooted.**

---

## 7. What this delivery may never be read as

- **Not an approval of QRM02**, and not a substitute for F1 and F2. A repaired instrument measured
  through an unrepaired runner still yields a per-batch figure.
- **Not a measurement of Q2's cells.** No model was built, no data opened, no cell trained. Every
  device number here is a **64 MiB probe tensor** and an instrument check. **`NO_NEW_MEASUREMENT` of
  the six W1440 cells' cost.**
- **Not a Traffic result.** No Traffic child ran in this lane. The hours at h192 and h336 are
  `UNKNOWN`, and the two horizons that have numbers have them from `91a4c410`, not from here.
- **Not a repair of `run_units`.** The environment contract and its verifier are delivered; the
  runner is lane A's file and was not edited.
- **Not a service repair.** The worker's TensorFlow environment is still unable to register the device
  by default. Nothing was installed and no environment was changed; state C reached the device with a
  child variable only.

---

## 8. Reproduction, and one disclosed deviation

```
# red first
python -m pytest tools/test_df_tf_device_telemetry.py -q          # ModuleNotFoundError, before the module existed

# the two suites (coordinator, through the launcher)
crispdm-run -m 3G -t 600 -n qrm02f3 -- python -m pytest tools/test_df_tf_device_telemetry.py -q        # 35 passed
crispdm-run -m 3G -t 600 -n qrm02f3 -- python -m pytest tools/test_df_tsl_remaining_horizons.py -q     # 32 passed

# the instrument against a real device and a real kernel
crispdm-run -m 6G -t 420 -n qrm02f3 -- python tools/df_tf_device_telemetry.py --selfcheck \
    --placement GPU --device-uuid GPU-<uuid> --envelope-bytes 12884901888

# Traffic's remaining horizons
crispdm-run -m 2G -t 120 -n qrm02f3 -- python tools/df_tsl_remaining_horizons.py
```

Retained, with device UUIDs truncated, under `docs/audits/evidence/qrm02_f3_20260929/`:
`selfcheck_coordinator_cpu.json`, `selfcheck_coordinator_gpu.json`,
`selfcheck_worker_empty_cvd.json` (state A), `selfcheck_worker_no_libpath.json` (state B),
`selfcheck_worker_libpath.json` (state C), `traffic_remaining_horizons.json`.

**Ten children through the deployed `crispdm-run`** — five on the coordinator (2–6 GiB) and five on
the secondary worker (6 GiB each) — each with a fresh aggregate admission, each declared once and
never re-asked smaller, none displacing anything, no reclaim, no `/dev/shm` change, no reboot. The
preferred accelerator host was not used: it is ineligible and was read from only in this document's
predecessors.

**Disclosed deviation.** Before I routed everything through the launcher, **four diagnostic
TensorFlow imports** ran outside it — one on the coordinator and three on the worker — while
establishing which interpreter carried TensorFlow and why it registered no GPU. Each was a single
short process of roughly 1 GiB resident on hosts with ~20 GiB available, and nothing was displaced;
but they were not capped, and the standing rule says capped. Reported rather than omitted.

---

## 9. What the auditor should attack first

1. **The predeclared arena bounds the arena, not the device.** A 12 GiB arena left ~3.07 GiB of the
   4090 for contexts and workspaces outside it, and the real recipe's cuDNN workspaces are not in the
   arena. If they need more than that, the *whole-device* figure will exceed what the arena limit
   suggests, and my §3 sizing sentence is the thing to break.
2. **Two devices, two TensorFlow versions, one instrument.** `CPU:0` raises on 2.18 and answers on
   2.21. I handled the difference and added a cross-scope refusal, but the instrument has been run on
   exactly two builds and one of them is not the pilot host's.
3. **State C used a child environment variable to reach the device.** That is not an installed fix.
   If the runner's interpreter or its wheel set changes, state B returns, and the honest outcome is a
   refusal rather than a CPU pilot wearing a GPU label — but somebody has to keep the contract of §3
   in the runner for that to hold.
4. **The parameter laws are algebra checked at two points.** The head's declaration is the
   justification and the two records are the check; a third horizon would be a better check, and a
   third horizon is exactly what has no allocation.
5. **The head-output tensor is a component, not a peak.** I priced the parameter-driven bytes exactly
   and refused to price the time. If anyone reads the static table as a memory projection for h192 or
   h336, that is a misuse the table's own class string is meant to prevent, and it can be tested.
6. **The 6.62 GPU-hour subtotal is still a 32-update extrapolation to 30 epochs.** Its per-step rate
   is measured; its epoch and cell hours are derived, and one term inside each epoch — the author's
   per-epoch logging pass over the test loader — is derived from the validation rate and never had its
   memory measured at all.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-29
