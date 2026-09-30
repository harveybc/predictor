# QRM integration — one runner, a declared placement, and the governed route proved hop by hop

Satoshi, successor technical lead. 2026-09-29. I am the single integration owner named in
`docs/handoffs/SATOSHI_QRM_INTEGRATION_AND_PARALLEL_WORK_2026_09_29.md` (`aa44bd14`), and nobody
else touched the shared runner while I held it.

Sources merged: QRM01 `2fee2fc7` (`satoshi/qrm01-f1-f2-repair-20260929`) and the telemetry
`8cbe99a3` (`satoshi/qrm02-f3-tensorflow-telemetry-20260929`), both off the common ancestor
`6de538e2`. Neither lane's files overlapped the other's; the merge carried no conflict. The lake
restoration and the classification work were **not rerun** and are unchanged.

Three answers first, because they are the three questions the dispatch leads with.

| question | answer |
|---|---|
| **Can a GPU request still fall back to the CPU unnoticed?** | **No.** A GPU declaration is verified on three independent facts and a shortfall raises `GPU_REQUEST_FELL_BACK_TO_CPU`. Measured live: the admitted worker's TensorFlow registers **zero** devices with the driver holding the card, and the request is **refused** — there is no code path that returns a CPU record for a GPU request. |
| **Does the governed route complete with a warehouse readback?** | **Yes, twice** — once on a declared CPU placement and once on a GPU placement whose three facts were verified inside the child. All eight hops true, both terminal digests read back out of the live warehouse and both agreeing with their receipts. |
| **The reconciled compute request** | **Not 4 800 CPU s / 4 800 wall s, and not on any host available today.** See §7: the reconciled per-child stops are **2 400 CPU s + 1 800 wall s**, campaign worst case **4 800 CPU s / 3 600 wall s + ~12 s of route overhead**, and the 12 GiB device envelope **has no eligible host**. Two authorizations are missing and one is *not*. |

---

## 1. What the two lanes could not do separately

The retained producer-to-supervisor run (QRM01 §6) states its own exclusions: **governed delivery,
governance terminal reporting and GPU**. The household governed run (`6de538e2`) proved the chain
with a child launched by a bare `subprocess.run` — **no scope, no reservation, no lease, no
envelope**. So each lane proved one half of the route and neither proved the join. The dispatch is
right that those exclusions are not tested, and the join is what this delivery is.

| hop | before | now |
|---|---|---|
| campaign → delivery | household lane only | both, through `df_e1_governed.acquire` |
| the actual child's byte digest | household lane only | both, `verify_consumed_bytes` in the child's own process |
| producer envelope | QRM01 lane only | both, `df_cell_scope_envelope.v1` → `df_cell_scope_record.v2` |
| supervisor + retained lease | QRM01 lane only | both, `df_cell_scope.supervise` through the deployed launcher |
| accepted terminal | household lane only | both, with the receipt |
| warehouse readback | household lane only | both, through the existing close client |
| device placement | neither | declared, and verified on three facts |

## 2. The placement, and why visibility proves nothing

`tools/df_placement_contract.py` (new). `run_units` used to hand every child
`CUDA_VISIBLE_DEVICES=""` — the `run_units:976` defect. That is replaced by a **declared**
placement, built once by the runner, passed to the **child only**, and verified **by the child**:

- **Child-only.** Nothing is written to the parent's `os.environ` and the base mapping is never
  mutated, so two siblings in one parent can hold two different placements. A process-wide override
  is the defect; a process-wide replacement would be the same defect with a nicer name.
- **Before the import.** `LD_LIBRARY_PATH` is read by the dynamic loader when TensorFlow's
  extension modules load. `enforce_before_tensorflow()` refuses `TENSORFLOW_ALREADY_IMPORTED`,
  because a contract examined after that import is narration.
- **Three facts, never one.** The driver's device UUID through `libcuda.cuDeviceGetUuid`;
  TensorFlow's own device registration; and where a real op lands. The telemetry lane measured, on
  a real device, a state where the driver held the card, the UUID matched **exactly** and
  TensorFlow registered **zero** devices. A visibility check would have sealed that as a GPU pilot.
- **No default, and no downgrade.** An undeclared placement refuses (`PLACEMENT_NOT_DECLARED`); a
  GPU declaration short of three facts raises `GPU_REQUEST_FELL_BACK_TO_CPU`. The refusal is an
  **exception** on purpose — a return value that a caller could continue on is how the downgrade
  happened in the first place.
- **CPU is a declaration**, not the absence of one. `CUDA_VISIBLE_DEVICES=""` alone cannot be told
  apart from a GPU run whose pinning was lost; the declaration is what lets the child refuse a
  later GPU claim over a CPU run.

**PyTorch is out of the TensorFlow memory path entirely.** `df_cell_scope.gpu_memory` asked
`torch.cuda.max_memory_allocated()` for the device figure of a `tf.keras` model. It now asks
`tf.config.experimental.get_memory_info`, records `pytorch_consulted: false`, and a test asserts
the string `import torch` does not appear in the file at all. A CPU placement answers `UNKNOWN`
**with its reason** — absence, never zero.

### 2.1 Two corrections against my own contract, both forced by live measurement

Neither was argued down; both were measured, and each is now a test that fails if it returns.

1. **A wheel inventory is not an outcome.** My first version refused a GPU placement whenever the
   interpreter's `nvidia/*/lib` inventory was incomplete. Run against the coordinator's real
   environment, that gate **refused a placement that works**: six of the nine library families are
   absent from its wheels and TensorFlow 2.18 registers `/physical_device:GPU:0` regardless,
   because the loader finds them elsewhere. A refusal that stops a working GPU run is the same
   class of error as a downgrade that hides a broken one. The question is now **loadability per
   family, by a named route** (`INTERPRETER_WHEEL` or `DYNAMIC_LOADER_DEFAULT_SEARCH`), which is
   what TensorFlow itself asks.
2. **A hardcoded CUDA generation is a wrong question.** The first table named only the CUDA 12
   sonames. The admitted worker's TensorFlow 2.21 environment carries **both** generations — the
   `cu12`/`cu13` mix the telemetry lane named as the reason TensorFlow's own probe came back empty
   — in **two different layouts** (`nvidia/<pkg>/lib` and a flat `nvidia/cu13/lib`). A CUDA 13
   build would have been refused for carrying exactly the libraries it is supposed to carry. A
   family is now satisfied by any of its known sonames, newest first, and the record names which
   one answered and which generation it came from.

## 3. Tests first, for the six failure modes

`tests/test_qrm_integration_placement.py`, written and run **red** before the module existed
(`FileNotFoundError: tools/df_placement_contract.py`), now **37 passed**.

| failure mode | refused by | proved by |
|---|---|---|
| **missing libraries** | `MISSING_CUDA_LIBRARIES`, per **family**, naming every soname that would have satisfied it | a fake site-packages with a loader double that finds nothing; and the same refusal **measured live** on the admitted worker |
| **wrong UUID** | `DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED` (plus `AMBIGUOUS_PINNING` for two visible devices) | a driver double returning another UUID; and **measured live** on the coordinator, where a nonexistent UUID gives `DRIVER_EXPOSES_NO_DEVICE` |
| **CPU fallback** | `GPU_REQUEST_FELL_BACK_TO_CPU`, carrying `FRAMEWORK_REGISTERED_NO_DEVICE` or `PLACEMENT_NOT_ON_DECLARED_DEVICE` | framework doubles for zero-registration and for an op landing on `CPU:0`; and **measured live** on the worker's real RTX 4090 |
| **stale attempt** | `ATTEMPT_IDENTITY_MISMATCH` / `ATTEMPT_IDENTITY_MISSING` | a record from an earlier attempt of the same cell, against a token minted before the child existed |
| **wrong envelope** | `ENVELOPE_MISSING`, `RECORD_SCHEMA_SUPERSEDED`, and an envelope pointing where no record sits | a document that **carries the peak** at a root field and declares nothing — refused, because no field is read by name alone |
| **failed delivery** | `CONSUMED_BYTES_ARE_NOT_THE_DELIVERED_BYTES`, `DELIVERED_PANEL_IS_NOT_THE_CHARACTERISED_PANEL`, and `require_delivery` refusing another unit's panel | the child's own digest of the file it opened, against the delivery's claim |

Each mode also has its **accepting** counterpart asserted in the same file, so none of the gates is
one that refuses everything.

The refusals are asserted on **doubles for the framework and the driver**, deliberately: a test that
needed a real GPU could not assert the states in which no GPU is present. The real-device evidence
is separate and retained (§5).

## 4. The stops are executed, not tabulated

| stop | mechanism | enforced by | proved by |
|---|---|---|---|
| **CPU** | `RLIMIT_CPU`, installed by the child on itself; `SIGXCPU` at the soft limit, `SIGKILL` at the hard one | **kernel** | `test_the_cpu_stop_actually_kills_a_child_that_spends_its_budget`: a real child spins, never prints `SURVIVED`, and exits on a signal |
| **wall** | the deployed launcher's own `-t`, outside the child | **the launcher's supervisor** | `test_the_wall_stop_is_enforced_by_the_launcher_outside_the_child`: a 120 s sleep under a 2 s limit dies, the log never shows `SURVIVED`, and the attempt is `usable_for_costing: false` |
| **each stage** | `setitimer` plus a CPU poll, **checked at entry** | process | a stage entered with its budget spent does not execute its body |
| **host memory** | the transient scope's `MemoryMax` | **kernel** | the kernel limit in the record equals the declared cap, to the byte, in both governed units |
| **device memory** | a predeclared allocator arena (`set_logical_device_configuration`) | **the framework allocator** | the telemetry lane's `ResourceExhaustedError` on the oversized request; unchanged here |

A budget with no mechanism is recorded as **absent**, never as a limit: the household record carries
`cpu_stop.why` when no CPU budget reached the child.

## 5. The governed route, every hop with its own evidence

Twice, on the coordinator, through the deployed `crispdm-run`, on the **small mechanical workload**
— the household probe, which verifies its own consumed bytes by sha256 and reads a parquet panel.
**No W1440 fit, no model, no gradients, no score.** Retained under
`docs/audits/evidence/QRM_INTEGRATION_20260929/GOVERNED_ROUTE.json`.

| hop | CPU unit `probe_cpu` | GPU unit `probe_gpu` |
|---|---|---|
| code identity | `c4a918ca` | `c4a918ca` |
| campaign | `483b79b16feb6fe7…` | `f503079bb897469f…` |
| delivery | `a1f7d520…`, `VERIFIED_TRANSFER`, `cached: false` | `1e87ff60…`, `VERIFIED_TRANSFER`, `cached: false` |
| **the child's own byte digest** | `b3192c0bcb117b2e…`, 10 890 295 B, matches both the delivery and the characterised panel | identical |
| producer envelope | `df_cell_scope_envelope.v1` → `df_cell_scope_record.v2` at `cell_scope` | identical |
| attempt token (minted before the child existed) | `1790733513783-3639c9e5…` | `1790733519408-85e890c7…` |
| **retained lease** | `probe_cpu-1790733513-…` | `probe_gpu-1790733519-…` |
| fresh-attempt contract | **accepted**, `refused_by: []`, costable | **accepted**, `refused_by: []`, costable |
| declared cap = kernel limit | 2 147 483 648 B = 2 147 483 648 B | 4 294 967 296 B = 4 294 967 296 B |
| accepted terminal | `COMPLETED`, receipt `202333674ff32960…` | `COMPLETED`, receipt `96691508bd38e203…` |
| **warehouse readback** | **ok**, digest `202333674ff32960…` **agrees with the receipt**, 4 metric rows, 1 artifact row | **ok**, digest `96691508bd38e203…` **agrees**, 4 metric rows, 1 artifact row |
| route verdict | **complete: true**, 8/8 hops | **complete: true**, 8/8 hops |

### 5.1 Footprints — each with its scope named, and never merged

Three scopes appear below. They have different bases, they are never added, and no one of them
bounds another.

| unit | `HOST_CGROUP` (the scope's own `memory.peak`, kernel accounting) | `PROCESS_RESIDENT_SET` (`ru_maxrss`, one process, host RAM) | `FRAMEWORK_ALLOCATOR/DEVICE` (TensorFlow's own accounting for `GPU:0`) |
|---|---:|---:|---|
| `probe_cpu` | **301 842 432 B** (lifetime 303 415 296 B) | 372 015 104 B | **`UNKNOWN`** — a CPU placement; TensorFlow carries no allocator statistics for `CPU:0`. **Absent, not zero** |
| `probe_gpu` | **910 901 248 B** (lifetime 912 736 256 B) | 1 324 044 288 B | **1 792 B peak**, `MEASURED`, `pytorch_consulted: false` |

The resident set exceeds the cgroup peak for `probe_gpu` and that is not a contradiction: they are
different quantities read at different instants on different bases. Neither is offered as the
other, and neither is added to the device figure.

**The 1 792 B is the whole device figure and it is honest about what it measures**: this workload
places no TensorFlow tensors beyond the verifier's one-element probe. It is the allocator working,
not a model's footprint, and **`NO_NEW_MEASUREMENT` of any cell's device cost** follows from it.

### 5.2 The GPU placement, verified rather than declared

`probe_gpu`'s child recorded, from inside its own scope:

```
placement: GPU
device_uuid: GPU-612d1e0c-…                       (the coordinator's own device)
facts_verified: [DRIVER_UUID, FRAMEWORK_REGISTRATION, EXECUTION_PLACEMENT]
establishes_a_gpu_pilot: true
```

The runner's own declaration carries `establishes_a_gpu_pilot: null` — a request is not evidence,
and only the child's verification fills it in.

### 5.3 The refusals, on real hardware

| state | host | verdict |
|---|---|---|
| GPU declared, `CUDA_VISIBLE_DEVICES=""` | coordinator | **`NO_VISIBLE_DEVICE`** before the import |
| GPU declared with a UUID no device has | coordinator | **`GPU_REQUEST_FELL_BACK_TO_CPU`** · `DRIVER_EXPOSES_NO_DEVICE`, `FRAMEWORK_REGISTERED_NO_DEVICE` |
| GPU declared, correct UUID, all nine families resolved from that interpreter's own wheels, both generations | **admitted worker's RTX 4090** | **`GPU_REQUEST_FELL_BACK_TO_CPU`** · `FRAMEWORK_REGISTERED_NO_DEVICE` — reproducible |
| GPU declared, correct UUID, libraries resolved | coordinator | **`MEASURED`**, three facts, a real GPU pilot |

**A finding the dispatch did not ask for and should have.** The telemetry lane's **state C** — a GPU
pilot established on the worker's 4090 with a 12 GiB enforced arena — is **not reproducible today**.
Its retained record names Python 3.12.13 / TensorFlow 2.21.0, which is
`~/anaconda3/envs/tensorflow` on that host. With the integrated contract passing that interpreter's
own CUDA library directories (cu13 first, then cu12, both layouts), TensorFlow 2.21 there still
prints `Cannot dlopen some GPU libraries … Skipping registering GPU devices` and registers zero
devices, while `libcuda.so.1` loads. The driver is fine; the framework is not.
**`STATE_C_NOT_REPRODUCIBLE`**, and the honest consequence is a refusal, which is what the runner
produces. I installed nothing, changed no environment and repaired no service.

## 6. The attribution correction

The telemetry lane charged a batch slice's residual to the **MT5 paper-trading virtual machine** on
the strength of that VM's **host-wide resident set** (`qemu-system-x86`, one process,
5 785 092 096 B). A resident set is a fact about one process's address space in host RAM. It says
nothing about which cgroup the kernel charges that memory to, so **it is not evidence of slice
membership** and cannot explain a slice's own charge. Two scopes were added that must never be
added.

`tools/df_slice_membership.py` (new) decides membership the only way it is decidable: a process is
charged to a slice **iff** its own cgroup path is the slice or lies under it. Read non-mutatingly
on the admitted worker, 2026-09-29:

| holder | figure | scope | membership | disposition |
|---|---:|---|---|---|
| `crispdm-batch.slice` | `memory.current` **1 517 969 408 B**, `memory.peak` **11 305 144 320 B**, `memory.max` 15 032 385 536 B | `SLICE_CGROUP_CHARGE` | — | the only figure here that is the slice's accounting |
| the slice's 2 members | resident sets summing to 24 649 728 B | `PROCESS_RESIDENT_SET` | **inside** | reported to say *who is in there*; their sum neither equals nor bounds `memory.current` |
| **the MT5 VM** (`qemu-system-x86`) | **5 776 457 728 B** resident | `PROCESS_RESIDENT_SET` | **OUTSIDE the slice** — its cgroup is not the slice and not under it | **`NOT_A_SLICE_CHARGE`.** Not displaced, not signalled, not touched |

**Verdict: the residual's owner is `UNDETERMINED`.** Every named holder is outside the slice by
membership evidence, so none of them can account for the slice's charge; the remainder belongs to
the slice's own members, to kernel memory charged to the slice, or to shared pages, and saying which
would need a per-member measurement this record does not have. The earlier attribution to the VM is
**withdrawn**, and it is withdrawn on evidence rather than on a preference.

On the coordinator the same tool finds **no VM at all**, so no attribution to it was even available
there.

## 7. The reconciled compute request

The QRM02 pilot design asked for **two architectures × one calibration cell, sequential, on the
worker's RTX 4090 by UUID, worst case 4 800 s wall and 4 800 CPU s, host envelope 12 GiB enforced,
device envelope 12 GiB**. Reconciled against the path that now actually executes:

### 7.1 What the executable path changes about the numbers

| term | as requested | reconciled | why |
|---|---|---|---|
| **CPU** | 4 800 CPU s total | **4 800 CPU s total, as 2 × 2 400 CPU s per child, enforced by `RLIMIT_CPU`** | unchanged in total, but it is now a **per-child kernel stop** rather than a campaign figure. The design's own per-cell CPU deadline was already 2 400 s; two children is exactly 4 800 |
| **wall** | 4 800 s total | **3 600 s total, as 2 × 1 800 s per child, enforced by the launcher's `-t`** | the design's per-cell wall deadline is 1 800 s and the children are sequential. 2 × 1 800 = 3 600. The extra 1 200 s was slack with **no mechanism behind it**, and the integrated path has no place to spend it: the only stop that exists is per child |
| **route overhead** | not priced | **+ ~12 s wall, + ~0.05 CPU s for 2 units** | measured: 3.09 s and 5.57 s wall, 0.017 and 0.023 CPU s for campaign + delivery + supervision + terminal + readback. Doubling for margin is still negligible against 3 600 s, and it is now priced rather than omitted |
| **host envelope** | 12 GiB per cell, enforced | **12 GiB per cell, enforced, and it fits** — live: slice ceiling 15 032 385 536 B, in use 1 562 873 856 B, headroom **13 469 511 680 B**, host free for new 18 254 893 056 B, `MemAvailable` 21 476 118 528 B, 0 live reservations, pressure 0.0 on the worker | 12 GiB = 12 884 901 888 B sits inside the headroom **at this instant**, sequentially. Read live through the admission module's non-mutating `state`, not quoted |
| **device envelope** | 12 GiB, enforced in the allocator | **NO ELIGIBLE HOST** | see 7.2 |

**So the reconciled ask is 4 800 CPU s and 3 600 wall s, per-child 2 400 / 1 800, plus a priced
~12 s of route overhead — and it is not yet askable, because its device envelope has nowhere to
run.** Naming 4 800 wall s now would be asking for a number no mechanism enforces.

### 7.2 The device envelope has no host, and that is the blocker

| host | device | total | 12 GiB envelope? | GPU pilot establishable? |
|---|---|---:|---|---|
| coordinator | `GPU-612d1e0c-…` | **8 585 740 288 B** (7 931 428 864 B free) | **impossible** — 12 884 901 888 B exceeds the whole card | **yes**, three facts verified live |
| admitted worker | `GPU-a8bd1b2c-…` RTX 4090 | 16 718 168 064 B | fits | **no** — `FRAMEWORK_REGISTERED_NO_DEVICE`, reproducible today (§5.3) |
| preferred external host | RTX 5090 | 32 607 MiB | fits | **INELIGIBLE** by standing ruling; read once to confirm the ruling, given no work |

The pilot as designed needs a host that has **both** a ≥12 GiB device and a TensorFlow that
registers it. **No host available to me has both.** The two ways out, neither of which I may take:

- **repair the worker's TensorFlow GPU environment** — an installation. I have no authority to
  install, and I installed nothing; or
- **redesign the device envelope downward** to fit the coordinator's 8 585 740 288 B card — a
  **scientific** change to a declared envelope, which belongs in a successor design before any
  score, not in an integration report.

I am not shrinking the envelope to make the refusal go away. That is the same move as re-asking a
cap smaller after a refusal, and it is forbidden for the same reason.

### 7.3 The missing authorizations, and the one that turns out not to be needed

| item | status |
|---|---|
| an allocation for the 2 calibration cells (2 × 12 GiB host cap, 2 400 CPU s + 1 800 wall s each, sequential, on a host whose TensorFlow registers its device) | **MISSING.** `lease: NONE HELD`. No allocation is borrowed from another lane |
| a repair of the worker's TensorFlow GPU environment, **or** a successor design with a device envelope that fits an 8 GiB card | **MISSING**, and it is the actual blocker. Owner or designer decision, not mine |
| the six full scientific W1440 cells | **UNAUTHORIZED and untouched** |
| the **18 GiB slice change** | **UNAUTHORIZED — and the pilot does not need it.** 12 GiB fits inside the existing 14 GiB ceiling's live headroom, sequentially. An 18 GiB cap would exceed the ceiling and be a terminal `ABOVE_SLICE_CEILING` refusal. **The request should be withdrawn rather than granted**, and no ceiling was moved to reach that conclusion |

### 7.4 The exact commands

The pilot is **not** run here; these are the commands the authorization would authorize, plus the
ones actually executed. On a non-interactive ssh shell `~/.local/bin` is not on `PATH`, so the
launcher is invoked by absolute path there.

```
# EXECUTED — the governed route, end to end, CPU placement (small mechanical workload)
crispdm-run -m 5G -t 900 -n qrmint-final -- python3 tools/df_q2_household_governed.py probe \
    --root <RUN_ROOT>/qrm_int_final_cpu --unit probe_cpu \
    --api-key-file <KEY> --gov-url http://127.0.0.1:5055 \
    --warehouse-url http://127.0.0.1:5057 --warehouse-token-file <WAREHOUSE_ENV> \
    --placement CPU --cell-cap-bytes 2147483648 --child-cpu-seconds 120 --child-timeout 600

# EXECUTED — the same route on a VERIFIED GPU placement
crispdm-run -m 5G -t 900 -n qrmint-final -- python3 tools/df_q2_household_governed.py probe \
    --root <RUN_ROOT>/qrm_int_final_gpu --unit probe_gpu \
    --api-key-file <KEY> --gov-url http://127.0.0.1:5055 \
    --warehouse-url http://127.0.0.1:5057 --warehouse-token-file <WAREHOUSE_ENV> \
    --placement GPU --device-uuid GPU-612d1e0c-... \
    --cell-cap-bytes 4294967296 --child-cpu-seconds 240 --child-timeout 600

# EXECUTED — the placement contract against real devices
crispdm-run -m 5G -t 300 -n qrm-place -- python3 tools/df_placement_contract.py \
    --placement GPU --device-uuid GPU-612d1e0c-... --verify          # MEASURED, 3 facts
crispdm-run -m 5G -t 240 -n qrm-ref   -- python3 tools/df_placement_contract.py \
    --placement GPU --device-uuid GPU-00000000-... --verify          # REFUSED
# on the admitted worker, by absolute path, with that interpreter:
~/.local/bin/crispdm-run -m 6G -t 300 -n qrm-place -- \
    ~/anaconda3/envs/tensorflow/bin/python3 /tmp/df_placement_contract.py \
    --placement GPU --device-uuid GPU-a8bd1b2c-... --verify          # REFUSED, no registration

# EXECUTED — membership evidence, non-mutating, on both hosts
crispdm-run -m 1G -t 120 -n qrm-memb -- python3 tools/df_slice_membership.py \
    --holder qemu-system-x86
~/.local/bin/crispdm-run -m 1G -t 120 -n qrm-memb -- python3 /tmp/df_slice_membership.py \
    --holder qemu-system-x86

# NOT EXECUTED — the pilot, if and only if an allocation and an eligible device host exist
<LAUNCHER> -m 12884901888 -t 1800 -n qrm02-cal-<arch> -- python3 tools/df_e1_block.py execute \
    --root <PILOT_ROOT> --placement GPU --device-uuid GPU-<eligible-device> \
    --cell-cap-bytes 12884901888 --api-key-file <KEY> --gov-url <GOV> \
    --warehouse-url <WH> --warehouse-token-file <WAREHOUSE_ENV>
```

### 7.5 The exact request

> **Requested:** two calibration cells, one per architecture, **sequential**, each in its own
> transient scope with a **12 GiB (12 884 901 888 B) `MemoryMax` declared once and never re-asked
> smaller**, each stopped by `RLIMIT_CPU` at **2 400 CPU s** and by the launcher at **1 800 wall
> s** — **4 800 CPU s and 3 600 wall s** for the campaign, plus ~12 s of governed-route overhead.
> Host: one whose TensorFlow **registers** its device, verified by the three facts before any cell
> starts; the device envelope of 12 GiB then predeclared in the framework allocator.
> **Blocked on:** no such host is available to me today (§7.2). **Not requested:** the 18 GiB slice
> change (withdrawn — 12 GiB fits the existing ceiling), the six scientific cells, any test access,
> any accuracy claim, and any allocation borrowed from another lane.
> **This buys a costed successor proposal. It buys no fit, no accuracy and no training
> authorization.**

## 8. Installed-source identity

`docs/audits/evidence/QRM_INTEGRATION_20260929/RUNNER_IDENTITY.json`, verdict
**`DEPLOYED_MATCHES_TRACKED`**:

| component | deployed sha256 | tracked path | matches | reachable from |
|---|---|---|---|---|
| launcher | `499fdc1877750337…` | `tools/crispdm-run` | **true** | `c4a918ca` |
| admission module | `8dc2c03b17e49869…` | `tools/crispdm_admission.py` | **true** | `c4a918ca` |

A user systemd able to create a transient scope is available. The host is recorded as an opaque
per-host id; **no host name is written anywhere in this repository.**

## 9. Report

```
QRM INTEGRATION — one integrated commit, a declared placement, the governed route end to end
repo/branch/tip: predictor · satoshi/qrm-integration-20260929 · merge of 2fee2fc7 + 8cbe99a3
worktree: .worktrees/predictor-qrm-integration-20260929   (single integration owner: Satoshi)
files:
  tools/df_placement_contract.py     NEW. the child-only CPU/GPU contract, enforced before the
                                     TensorFlow import; three-fact verification; per-family CUDA
                                     loadability across both generations and both wheel layouts;
                                     no default placement and no downgrade
  tools/df_slice_membership.py       NEW. slice membership as evidence; three scopes never merged;
                                     UNDETERMINED where membership cannot be established
  tools/df_e1_block.py               run_units declares a placement instead of CUDA_VISIBLE_DEVICES="";
                                     the child verifies it before load_data imports TensorFlow;
                                     the bitwise replay stays on a DECLARED CPU placement and says why
  tools/df_cell_scope.py             gpu_memory now asks TensorFlow's allocator; PyTorch is out of
                                     the path entirely (no import, asserted by test)
  tools/df_q2_household_governed.py  the governed route's child goes through the supervisor, writes
                                     through the declared envelope, digests its own bytes, and every
                                     hop is a condition that refuses under its own name
  tests/test_qrm_integration_placement.py  NEW, written red first. 37 tests: six failure modes,
                                     their accepting counterparts, the two self-corrections, and
                                     CPU + wall stops that really kill children
  docs/audits/work_plan/SATOSHI_QRM_INTEGRATION_2026_09_29.md
  docs/audits/evidence/QRM_INTEGRATION_20260929/{GOVERNED_ROUTE,CAPACITY,RUNNER_IDENTITY,
      PLACEMENT_COORDINATOR_GPU,PLACEMENT_REFUSAL_EMPTY_CVD,PLACEMENT_REFUSAL_WRONG_UUID,
      PLACEMENT_WORKER_REFUSED,SLICE_MEMBERSHIP_COORDINATOR,SLICE_MEMBERSHIP_WORKER}.json
suites: test_qrm_integration_placement 37 · test_df_cell_scope 31 · fresh_attempt 30 ·
        qrm01_auditor_counterexamples 7 · qrm01_producer_to_supervisor_e2e 11 ·
        df_tf_device_telemetry 35 · df_tsl_remaining_horizons 32 · q2_household_closure 9 ·
        df_e1_block 20  ==  212 passed, 0 failed
acceptance:
  a GPU request CANNOT fall back to the CPU unnoticed: three facts (driver UUID, framework
    registration, execution placement), shortfall raises GPU_REQUEST_FELL_BACK_TO_CPU, no code
    path returns a CPU record for a GPU request; measured live on two devices and refused on one
  governed route COMPLETE twice, 8/8 hops each, on the small mechanical household probe:
    campaign -> delivery (VERIFIED_TRANSFER, cached=false) -> the child's own sha256 of the bytes
    it opened (b3192c0b...c8db, 10,890,295 B) -> df_cell_scope_envelope.v1/record.v2 ->
    attempt token minted before the child existed + the launcher's retained lease ->
    fresh-attempt contract accepted, refused_by [] -> terminal COMPLETED and accepted ->
    warehouse readback ok, digest agrees with the receipt, 4 metric rows + 1 artifact row
  GPU unit: facts_verified [DRIVER_UUID, FRAMEWORK_REGISTRATION, EXECUTION_PLACEMENT],
    establishes_a_gpu_pilot true, device figure 1,792 B peak from TensorFlow's own allocator,
    pytorch_consulted false
  footprints, scopes named and never merged: HOST_CGROUP 301,842,432 B (CPU unit) and
    910,901,248 B (GPU unit); PROCESS_RESIDENT_SET 372,015,104 B and 1,324,044,288 B;
    FRAMEWORK_ALLOCATOR/DEVICE UNKNOWN (CPU, absent not zero) and 1,792 B peak (GPU)
  kernel limit == declared cap to the byte in both units (2 GiB, 4 GiB); cap asked once
  CPU and wall ENFORCED: a spinning child dies on a signal inside RLIMIT_CPU; a 120 s sleeper
    under a 2 s launcher limit dies and is not costable
  installed-source identity DEPLOYED_MATCHES_TRACKED; launcher and admission module byte-identical
    to their tracked blobs, both reachable from c4a918ca
  attribution corrected: the MT5 VM is OUTSIDE crispdm-batch.slice by membership evidence, so its
    5,776,457,728 B host-wide resident set is NOT_A_SLICE_CHARGE; the residual's owner is
    UNDETERMINED
  reconciled request: 4,800 CPU s (2 x 2,400, RLIMIT_CPU) and 3,600 wall s (2 x 1,800, launcher),
    not 4,800 wall s -- the extra 1,200 s had no mechanism; + ~12 s priced route overhead
what is NOT done / refused / not measured:
  - NO allocation requested is granted here and none is borrowed. The two calibration cells have
    no lease; the six scientific W1440 cells stay UNAUTHORIZED and untouched.
  - The 18 GiB slice change stays UNAUTHORIZED and is RECOMMENDED WITHDRAWN: 12 GiB fits the
    existing 14 GiB ceiling's live headroom. No ceiling was moved to establish that.
  - NO host available to me can host the pilot as designed: the coordinator's device is
    8,585,740,288 B so a 12 GiB envelope is impossible there, the admitted worker's TensorFlow
    registers zero devices, and the preferred external host is INELIGIBLE and was given no work.
  - The telemetry lane's state C is NOT REPRODUCIBLE today on the worker: with that interpreter's
    own CUDA libraries on the child's path, both generations and both layouts, TF 2.21 still
    prints "Cannot dlopen some GPU libraries" and registers none, while libcuda.so.1 loads.
    STATE_C_NOT_REPRODUCIBLE. I installed nothing and repaired no service.
  - The 1,792 B device figure is the verifier's probe tensor. NO_NEW_MEASUREMENT of any cell's
    device cost, and no model-quality claim follows from anything here.
  - No W1440 fit, no model built, no gradients, no optimizer slots, no score, no test access.
  - The lake restoration and the classification work were NOT rerun and are unchanged.
  - Historical records are neither validated nor invalidated: every record written before the
    QRM01 repair is v1 and is refused by version.
  - The MT5 virtual machine, caches, shared memory, host limits and services were NOT touched.
    No cap shrunk, no reservation reduced, no shared memory cleared, no ceiling moved, no service
    started, stopped or restarted, no reboot.
  - Capacity was re-read LIVE through the admission module's non-mutating `state` at delivery time;
    no figure is quoted from any earlier report.
  - No credential printed, copied, logged or written to a second file. No host name, IP, token or
    account identifier is written anywhere in this repository.
  - Two commits, and why: the integrated code is ONE commit (c4a918ca). The evidence and this
    report are a second, because `governed_run.strict_code_identity` refuses a dirty checkout --
    so the governed route could only run against a committed tip, and both units' retained
    code_identity pins exactly c4a918ca.
```

## 10. What I would attack first if I were reviewing this

1. **The three facts are verified once, at the start of the child.** A device that is lost
   mid-run — an Xid, a reset, an eviction — is not re-checked, so a long cell could finish on the
   CPU with a record that says GPU. Re-verification at stage boundaries is cheap and I did not add
   it.
2. **The execution-placement probe is a one-element `tf.zeros`.** It proves that *an* op lands on
   the device. It does not prove that the cell's convolutions, its optimizer slots or its cuDNN
   workspaces do. A soft-placement fallback inside the model could still put work on the CPU, and
   the probe would not see it.
3. **`DYNAMIC_LOADER_DEFAULT_SEARCH` is the weaker of the two routes and the contract accepts it.**
   A family resolved that way stops answering if the host's system libraries change, and then the
   honest outcome is a refusal — but the pilot would have been priced on an environment that no
   longer exists. The record names the route per family precisely so this is checkable; nothing
   enforces it.
4. **The device figure of 1 792 B is not a load test of anything.** I proved the route carries a
   verified GPU placement; I did **not** prove it carries a GPU *workload*. The first real
   TensorFlow cell on this path may find something the probe cannot.
5. **The residual indicator subtracts quantities with different bases.** `memory.current` minus the
   sum of members' resident sets is an *indicator* of how much of the charge the member list does
   not explain, and the record says so — but someone will read it as a measured quantity, and the
   word "residual" invites exactly that.
6. **The reconciled wall figure is arithmetic on the design's own per-cell deadline.** 2 × 1 800 s
   is right only if the children really are sequential and nothing else contends. It is not a
   measurement, and I did not measure a calibration cell's wall time — because no allocation exists
   to measure one.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-29
