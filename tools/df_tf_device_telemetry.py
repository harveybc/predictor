"""QRM02/F3 --- the allocator that measures the model that actually trains, and stops that stop.

Repairs finding **F3** of `docs/audits/work_plan/MUSASHI_AUDIT_23B2EFA3_2026_09_29.md`.

The QRM02 pilot measures a **TensorFlow/Keras** recipe (`tools/df_e1_block.py`, `build_modular` ->
`tf.keras.Model`, trained through `model.train_on_batch`).  The instrument QRM02 revision 3 named for
its device figures, `tools/df_cell_scope.py::gpu_memory`, asks **PyTorch's** caching allocator, which
does not account for a single byte TensorFlow allocates.  A 12 GiB device envelope watched by that API
is watched by an instrument that cannot see the memory it claims to watch.

This module answers the four things F3 asks for, and nothing else:

1. **The framework figure comes from TensorFlow's own allocator API**,
   `tf.config.experimental.get_memory_info(device)` -> `{"current", "peak"}`, reset at stage
   boundaries with `tf.config.experimental.reset_memory_stats(device)`.  PyTorch is never imported;
   the recipe is never ported to another framework to suit an instrument.
2. **The device actually used is declared and verified** --- by UUID, through the CUDA driver
   (`libcuda.cuDeviceGetUuid`, which is not a deep-learning framework), by the framework's own device
   registration, and by a placement probe that shows where an op actually lands.  An empty or unset
   `CUDA_VISIBLE_DEVICES` under a declared GPU placement is the `run_units:976` defect and is
   **refused by name**: sealing that runner establishes a CPU pilot, not a GPU pilot.
3. **Four scopes, never merged.**  The framework allocator, the process resident set, the host cgroup
   and whole-device telemetry each carry their own scope string and their own basis.  `merge_scopes`
   exists only to refuse.
4. **Executable stops.**  CPU is stopped by `RLIMIT_CPU` in the kernel; wall by an in-child watchdog
   plus the launcher's own outer limit; each stage by its own timer and CPU poll; the host envelope by
   the cgroup's `MemoryMax`; and the device envelope by a **predeclared allocator limit**
   (`tf.config.set_logical_device_configuration`), which is the only device mechanism that can fail an
   oversized allocation at the moment it is requested.  A statistics comparison is
   `OBSERVED_AFTER_THE_FACT` and is labelled that way everywhere it appears.

**An unavailable API is `UNKNOWN`.**  Never 0, never a number synthesized from host RAM, from another
scope or from a parameter count.  On a CPU-only placement TensorFlow answers
`ValueError: Allocator stats not available for device 'CPU:0'` --- verified --- and that is exactly the
`UNKNOWN` this module records.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import signal
import sys
import threading
import time
from contextlib import contextmanager

SCHEMA = "df_tf_device_telemetry.v1"

MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"
REFUSED = "REFUSED"

ENFORCED_AT_ALLOCATION = "ENFORCED_AT_ALLOCATION_BY_THE_FRAMEWORK_ALLOCATOR"
OBSERVED_AFTER_THE_FACT = "OBSERVED_AFTER_THE_FACT"
NOT_ENFORCED = "NOT_ENFORCED"

FRAMEWORK_SCOPE = ("FRAMEWORK_ALLOCATOR/DEVICE: the TensorFlow allocator's own accounting for one "
                   "GPU device, from tf.config.experimental.get_memory_info. It is DEVICE memory held "
                   "by TensorFlow and it is never added to any other scope")
FRAMEWORK_SCOPE_HOST = ("FRAMEWORK_ALLOCATOR/HOST: the TensorFlow allocator's own accounting for CPU:0, "
                        "from tf.config.experimental.get_memory_info where that build exposes it. It is "
                        "HOST RAM held by TensorFlow's own allocator -- not device memory, not the "
                        "cgroup's watermark, and it is never added to any other scope")
PROCESS_SCOPE = ("PROCESS_RESIDENT_SET: getrusage(RUSAGE_SELF).ru_maxrss of this process alone. Host "
                 "RAM, one process, no children, no kernel accounting, and it is never added to any "
                 "other scope")
HOST_SCOPE = ("HOST_CGROUP: memory.peak of this cell's own cgroup -- the whole process tree's host "
              "RAM high-watermark as the kernel charges it, and it is never added to any other scope")
DEVICE_SCOPE = ("WHOLE_DEVICE: memory in use on the physical device by every process, as the driver "
                "reports it. It includes contexts and other tenants and it is never added to any "
                "other scope")

SCOPE_RULE = ("its own scope, its own basis, its own name; two scopes are never added and "
              "merge_scopes refuses to")


REFUTED_CLAIM = "REFUTED_CLAIM"
REFUTED_ABORT_BEFORE_CLAIM = {
    "marker": REFUTED_CLAIM,
    "claim": ("revision 3 of the QRM02 design, section 3: \"the child raises and exits non-zero "
              "before allocating past the envelope\", where the child's evidence was "
              "max_memory_allocated / max_memory_reserved"),
    "status": "REFUTED",
    "why": ("those are allocator STATISTICS. A statistic is read after the allocator already held "
            "the memory, so a check on it can report an exceedance and can never have prevented it. "
            "Two further defects sat in the same sentence: the statistics named belonged to PyTorch, "
            "which does not account for TensorFlow's allocations, and the runner the design sealed "
            "against gives every child an empty CUDA_VISIBLE_DEVICES"),
    "what_is_enforceable": ("a PREDECLARED allocator arena, " + ENFORCED_AT_ALLOCATION +
                            " via tf.config.set_logical_device_configuration: an oversized request "
                            "then fails with ResourceExhaustedError where it is made"),
    "what_is_only_observed": ("a comparison of the framework peak with the envelope, " +
                              OBSERVED_AFTER_THE_FACT),
}


class ScopeRefusal(SystemExit):
    """Two scopes are never added; a refusal is the only answer to a request to add them."""


class DeviceRefusal(SystemExit):
    """The device actually used is not the device declared, so nothing measured here is attributable."""


class StageStopped(BaseException):
    """A stage's own executable stop fired. BaseException so a bare `except Exception` cannot eat it."""


# --- the framework figure: TensorFlow's own allocator ---------------------------------------------

def _tf():
    import tensorflow as tf                                                   # noqa: PLC0415
    return tf


def tf_allocator_memory(device: str = "GPU:0", *, tf=None) -> dict:
    """TensorFlow's allocator statistics for one logical device.

    `get_memory_info` answers `{"current", "peak"}` in bytes for the device's own allocator.  It is
    available for devices that carry a TensorFlow allocator with statistics; for `CPU:0` it raises
    `ValueError: Allocator stats not available`, and that is UNKNOWN -- an absent figure, not zero.
    """
    tf = tf if tf is not None else _tf()
    on_device = "GPU" in str(device).upper()
    base = {"framework": "tensorflow", "api": "tf.config.experimental.get_memory_info",
            "device": device, "memory": "DEVICE" if on_device else "HOST",
            "scope": FRAMEWORK_SCOPE if on_device else FRAMEWORK_SCOPE_HOST,
            "scope_rule": SCOPE_RULE}
    try:
        info = tf.config.experimental.get_memory_info(device)
    except Exception as e:                                                    # noqa: BLE001
        return {**base, "current_bytes": None, "peak_bytes": None, "status": UNKNOWN,
                "why": f"{type(e).__name__}: {e}",
                "reading": "this device carries no TensorFlow allocator statistics: the figure is "
                           "ABSENT. It is not zero and it is not taken from another scope"}
    cur, peak = info.get("current"), info.get("peak")
    if not isinstance(cur, int) or not isinstance(peak, int):
        return {**base, "current_bytes": None, "peak_bytes": None, "status": UNKNOWN,
                "why": f"the allocator answered {info!r}, which carries no integer peak"}
    return {**base, "current_bytes": int(cur), "peak_bytes": int(peak), "status": MEASURED,
            "peak_basis": ("the maximum this device's TensorFlow allocator has held since the process "
                           "started or since the last reset_memory_stats on this device")}


def reset_tf_allocator_peak(device: str = "GPU:0", *, tf=None) -> dict:
    """Reset the TensorFlow allocator's peak for one device, so a stage figure means that stage."""
    tf = tf if tf is not None else _tf()
    out = {"framework": "tensorflow", "api": "tf.config.experimental.reset_memory_stats",
           "device": device, "at": time.time()}
    try:
        tf.config.experimental.reset_memory_stats(device)
    except Exception as e:                                                    # noqa: BLE001
        return {**out, "reset": False, "status": UNKNOWN, "why": f"{type(e).__name__}: {e}",
                "reading": "the peak could not be reset, so the next reading is a LIFETIME figure "
                           "and is labelled as one"}
    return {**out, "reset": True, "status": MEASURED,
            "reset_basis": "the following peak on this device is the peak SINCE THIS RESET"}


# --- the device actually used ---------------------------------------------------------------------

def driver_device_uuid(index: int = 0) -> dict:
    """The UUID of the device the CUDA driver exposes at `index`, read through `libcuda`.

    Deliberately not a deep-learning framework: the driver is what both the framework and the
    environment's pinning act on, so it is the right witness for `which device is this really`.
    Unavailable driver -> UNKNOWN with the reason.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return {"status": UNKNOWN, "count": None, "uuid": None, "source": "libcuda",
                "why": f"libcuda.so.1 could not be loaded: {e}"}
    rc = lib.cuInit(0)
    if rc != 0:
        return {"status": UNKNOWN, "count": None, "uuid": None, "source": "libcuda.cuInit",
                "why": f"cuInit returned {rc}: the driver exposes no usable device to this process"}
    n = ctypes.c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value <= index:
        return {"status": UNKNOWN, "count": int(n.value), "uuid": None,
                "source": "libcuda.cuDeviceGetCount",
                "why": f"the driver exposes {n.value} device(s), so index {index} is absent"}
    dev = ctypes.c_int(0)
    if lib.cuDeviceGet(ctypes.byref(dev), index) != 0:
        return {"status": UNKNOWN, "count": int(n.value), "uuid": None,
                "source": "libcuda.cuDeviceGet", "why": "cuDeviceGet failed"}
    buf = (ctypes.c_char * 16)()
    fn = getattr(lib, "cuDeviceGetUuid_v2", None) or getattr(lib, "cuDeviceGetUuid", None)
    if fn is None or fn(buf, dev) != 0:
        return {"status": UNKNOWN, "count": int(n.value), "uuid": None,
                "source": "libcuda.cuDeviceGetUuid", "why": "the driver would not give a UUID"}
    h = bytes(buf).hex()
    uuid = f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    return {"status": MEASURED, "count": int(n.value), "uuid": uuid,
            "source": "libcuda.cuDeviceGetUuid", "index": int(index)}


def _visible(env: dict | None) -> str | None:
    env = os.environ if env is None else env
    return env.get("CUDA_VISIBLE_DEVICES")


def verify_declared_device(*, declared_uuid: str | None, placement: str,
                           env: dict | None = None, tf=None, driver=None) -> dict:
    """Declare the device, then verify that it is the device this process will actually use.

    Refusals, each by its own name:

      NO_VISIBLE_DEVICE                      a GPU placement with CUDA_VISIBLE_DEVICES empty or unset
                                             -- the `run_units:976` state, which is a CPU pilot
      AMBIGUOUS_PINNING                      more than one device is exposed, so a figure cannot be
                                             attributed to one
      DRIVER_EXPOSES_NO_DEVICE               the pinning names a device the driver will not give
      DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED the driver's UUID is not the declared one
      FRAMEWORK_REGISTERED_NO_DEVICE         the driver has it and TensorFlow did not register it
      PLACEMENT_NOT_ON_DECLARED_DEVICE       an op landed somewhere else
      DECLARED_DEVICE_WITH_CPU_PLACEMENT     a device was declared for a CPU pilot
    """
    driver = driver or driver_device_uuid
    vis = _visible(env)
    out = {"schema": SCHEMA, "placement": placement.upper(), "declared_uuid": declared_uuid,
           "visible_devices": vis, "refusals": [], "device_uuid_verified": None,
           "uuid_source": None, "framework_devices": None, "placement_probe": None,
           "establishes_a_gpu_pilot": False}

    if out["placement"] == "CPU":
        if declared_uuid:
            out["refusals"].append("DECLARED_DEVICE_WITH_CPU_PLACEMENT")
            out["status"] = REFUSED
            return out
        out["status"] = MEASURED
        out["reading"] = ("a CPU placement. The framework allocator carries no statistics for CPU:0, "
                          "so the framework device figure is UNKNOWN BY CONSTRUCTION here -- and this "
                          "is NOT a GPU pilot, whatever else the record contains")
        return out

    if vis is None or str(vis).strip() == "":
        out["refusals"].append("NO_VISIBLE_DEVICE")
        out["status"] = REFUSED
        out["reading"] = ("a GPU placement was declared and no device is visible to this process. "
                          "`run_units` passes CUDA_VISIBLE_DEVICES=\"\" to every child, so a pilot "
                          "sealed against that runner is a CPU pilot and may not be reported as a "
                          "GPU one")
        return out

    entries = [e for e in str(vis).split(",") if e.strip()]
    if len(entries) != 1:
        out["refusals"].append("AMBIGUOUS_PINNING")

    d = driver(0)
    out["driver"] = d
    if d.get("status") != MEASURED or not d.get("uuid"):
        out["refusals"].append("DRIVER_EXPOSES_NO_DEVICE")
    else:
        if int(d.get("count") or 0) > 1:
            if "AMBIGUOUS_PINNING" not in out["refusals"]:
                out["refusals"].append("AMBIGUOUS_PINNING")
        if declared_uuid and d["uuid"].lower() != str(declared_uuid).lower():
            out["refusals"].append("DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED")
        else:
            out["device_uuid_verified"] = d["uuid"]
            out["uuid_source"] = d.get("source")

    tf = tf if tf is not None else _tf()
    try:
        gpus = list(tf.config.list_physical_devices("GPU"))
    except Exception as e:                                                    # noqa: BLE001
        gpus = []
        out["framework_error"] = f"{type(e).__name__}: {e}"
    out["framework_devices"] = [getattr(g, "name", str(g)) for g in gpus]
    if not gpus:
        out["refusals"].append("FRAMEWORK_REGISTERED_NO_DEVICE")
    else:
        try:
            with tf.device("/GPU:0"):
                t = tf.zeros([1], dtype=getattr(tf, "float32", None))
            dev = str(getattr(t, "device", ""))
            out["placement_probe"] = {"device": dev, "api": "tf.device + a one-element tensor"}
            if "GPU:0" not in dev:
                out["refusals"].append("PLACEMENT_NOT_ON_DECLARED_DEVICE")
        except Exception as e:                                                # noqa: BLE001
            out["placement_probe"] = {"device": None, "why": f"{type(e).__name__}: {e}"}
            out["refusals"].append("PLACEMENT_NOT_ON_DECLARED_DEVICE")

    if out["refusals"]:
        out["status"] = REFUSED
        return out
    out["status"] = MEASURED
    out["establishes_a_gpu_pilot"] = True
    out["reading"] = ("one device is pinned, the driver confirms its UUID, the framework registered "
                      "it and an op landed on it: a figure from this process's TensorFlow allocator "
                      "is attributable to this device")
    return out


# --- the device envelope: what is enforceable, and what is only observed --------------------------

def install_device_envelope(limit_bytes: int, *, tf=None, device_index: int = 0) -> dict:
    """Cap the TensorFlow allocator's arena on one device, BEFORE the device is initialized.

    `set_logical_device_configuration(memory_limit=MiB)` makes TensorFlow's own allocator refuse to
    grow past the limit: an oversized request fails with `ResourceExhaustedError` where it is made.
    That is the one device mechanism in this framework that is `ENFORCED_AT_ALLOCATION`.

    Two limits of the mechanism, stated rather than hidden: it bounds what the TensorFlow allocator
    holds, not everything this process puts on the device (a CUDA context, cuDNN and cuBLAS
    workspaces outside the arena are whole-device scope), and it must be set before the device is
    initialized -- afterwards the call raises and this function REFUSES rather than continuing
    unbounded.
    """
    tf = tf if tf is not None else _tf()
    mib = int(limit_bytes) // (1 << 20)
    out = {"api": "tf.config.set_logical_device_configuration", "limit_bytes": int(limit_bytes),
           "limit_mib": mib, "device_index": int(device_index), "framework": "tensorflow",
           "scope": FRAMEWORK_SCOPE,
           "bounds": "the TensorFlow allocator's arena on this device",
           "does_not_bound": ("the CUDA context, cuDNN/cuBLAS workspaces outside the arena, and any "
                              "other tenant of the device: those are WHOLE_DEVICE scope")}
    try:
        gpus = list(tf.config.list_physical_devices("GPU"))
    except Exception as e:                                                    # noqa: BLE001
        gpus = []
        out["why"] = f"{type(e).__name__}: {e}"
    if len(gpus) <= device_index:
        return {**out, "status": REFUSED, "enforcement": NOT_ENFORCED,
                "why": out.get("why", f"the framework registered {len(gpus)} device(s), so index "
                                      f"{device_index} cannot be capped")}
    ldc = getattr(tf.config, "LogicalDeviceConfiguration", None) or \
        getattr(tf.config.experimental, "LogicalDeviceConfiguration", None)
    if ldc is None:
        return {**out, "status": REFUSED, "enforcement": NOT_ENFORCED,
                "why": "this TensorFlow exposes no LogicalDeviceConfiguration to cap with"}
    try:
        tf.config.set_logical_device_configuration(gpus[device_index], [ldc(memory_limit=mib)])
    except Exception as e:                                                    # noqa: BLE001
        return {**out, "status": REFUSED, "enforcement": NOT_ENFORCED, "why": f"{type(e).__name__}: {e}",
                "reading": "the envelope was asked for too late to be enforced, so it is NOT in "
                           "force. Nothing here may be reported as an enforced device limit"}
    return {**out, "status": MEASURED, "enforcement": ENFORCED_AT_ALLOCATION,
            "reading": ("the allocator will fail an oversized request with ResourceExhaustedError at "
                        "the request. This is the enforceable device mechanism; an allocator "
                        "STATISTIC compared later is not")}


def envelope_observation(framework: dict, *, envelope_bytes: int,
                         expected_device: str = "GPU:0") -> dict:
    """Compare a framework peak with the envelope. This is a REPORT, and it prevents nothing.

    See `REFUTED_ABORT_BEFORE_CLAIM` for the design sentence this replaces and why it was wrong.
    The enforceable mechanism is `install_device_envelope`; this function only says what happened.
    """
    peak = framework.get("peak_bytes")
    out = {"envelope_bytes": int(envelope_bytes), "framework_peak_bytes": peak,
           "api": framework.get("api"), "scope": framework.get("scope"),
           "figure_device": framework.get("device"), "envelope_device": expected_device,
           "enforcement": OBSERVED_AFTER_THE_FACT, "does_not_prevent": True,
           "reading": ("a comparison made AFTER the allocator already held the memory. It reports an "
                       "exceedance; it never prevents one. The enforceable limit is the predeclared "
                       "allocator configuration, and the two are different claims")}
    fig_dev = str(framework.get("device") or "")
    if ("GPU" in str(expected_device).upper()) != ("GPU" in fig_dev.upper()):
        return {**out, "exceeded": None, "status": REFUSED,
                "why": (f"a {expected_device} envelope may not be compared with a figure measured on "
                        f"{fig_dev or 'no device'}: one is device memory and the other is host RAM "
                        f"held by the framework. Two scopes, and a comparison across them passes "
                        f"nothing")}
    if not isinstance(peak, int):
        return {**out, "exceeded": None, "status": UNKNOWN,
                "why": framework.get("why", "the framework peak is UNKNOWN, so nothing is compared "
                                            "and nothing passes")}
    return {**out, "exceeded": bool(peak > int(envelope_bytes)), "status": MEASURED,
            "headroom_bytes": int(envelope_bytes) - int(peak)}


# --- four scopes, never merged --------------------------------------------------------------------

def _scoped(d: dict | None, scope: str, basis: str) -> dict:
    if d is None:
        return {"bytes": None, "status": UNKNOWN, "scope": scope, "scope_rule": SCOPE_RULE,
                "basis": basis, "why": "this scope was not read for this record: ABSENT, not zero"}
    out = dict(d)
    out.setdefault("status", UNKNOWN)
    out["scope"] = scope
    out["scope_rule"] = SCOPE_RULE
    out.setdefault("basis", basis)
    if out.get("status") == UNKNOWN:
        out["bytes"] = out.get("bytes")
    return out


def scope_record(*, cell_id: str, stage: str, framework: dict | None = None,
                 host_cgroup: dict | None = None, process_rss: dict | None = None,
                 whole_device: dict | None = None, extra: dict | None = None) -> dict:
    """One cell, one stage, four scopes side by side and never added."""
    fw = dict(framework) if framework else None
    if fw is not None:
        fw.setdefault("scope", FRAMEWORK_SCOPE)          # the figure's OWN scope: device or host
        fw["scope_rule"] = SCOPE_RULE
    else:
        fw = _scoped(None, FRAMEWORK_SCOPE, "tf.config.experimental.get_memory_info")
    rec = {
        "schema": SCHEMA, "cell_id": cell_id, "stage": stage, "at": time.time(),
        "framework_allocator": fw,
        "host_cgroup": _scoped(host_cgroup, HOST_SCOPE, "memory.peak of this cell's own cgroup"),
        "process_rss": _scoped(process_rss, PROCESS_SCOPE, "getrusage(RUSAGE_SELF).ru_maxrss"),
        "whole_device": _scoped(whole_device, DEVICE_SCOPE, "the driver's whole-device report"),
        "merge_rule": ("four scopes, four bases, four names. No field in this record is the sum of "
                       "two of them, and merge_scopes refuses to produce one"),
        "missing_rule": "a figure that could not be read is UNKNOWN. Never 0 and never a success",
    }
    if extra:
        rec["extra"] = extra
    return rec


def merge_scopes(record: dict, keys) -> None:
    """There is no merged figure. This function exists to refuse the request."""
    raise ScopeRefusal(
        "REFUSED: " + " + ".join(keys) + " is not a quantity. The TensorFlow allocator's device "
        "bytes, the process resident set, the cgroup's host-RAM watermark and the driver's "
        "whole-device report measure four different things on two different memories; adding any two "
        "of them produces a number that measures nothing")


# --- executable stops -----------------------------------------------------------------------------

def install_cpu_stop(cpu_seconds: float, *, exit_code: int = 68) -> dict:
    """A CPU deadline the kernel enforces: RLIMIT_CPU.

    At the soft limit the kernel delivers SIGXCPU; the handler prints the terminal line and leaves
    non-zero.  At the hard limit the kernel delivers SIGKILL, which no handler can decline -- so this
    stop does not depend on the child's cooperation.
    """
    soft = int(max(1, round(float(cpu_seconds))))
    hard = soft + 5
    cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_CPU)
    if cur_hard != resource.RLIM_INFINITY:
        hard = min(hard, cur_hard)
        soft = min(soft, hard)

    def _on_xcpu(signum, frame):
        sys.stderr.write(f"CPU_STOP: RLIMIT_CPU soft limit of {soft}s reached; leaving non-zero\n")
        sys.stderr.flush()
        os._exit(exit_code)

    signal.signal(signal.SIGXCPU, _on_xcpu)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    return {"name": "cpu", "mechanism": "RLIMIT_CPU (setrlimit) + SIGXCPU, SIGKILL at the hard limit",
            "enforced_by": "kernel", "executable": True, "soft_seconds": soft, "hard_seconds": hard,
            "hard_stop": "SIGKILL at the hard limit, which the child cannot decline",
            "exit_code": exit_code}


def install_wall_stop(wall_seconds: float, *, exit_code: int = 69,
                      outer: str = "the launcher's own -t wall limit and the supervisor's timeout") -> dict:
    """A wall deadline inside the child, plus the outer one that does not trust the child."""
    deadline = time.monotonic() + float(wall_seconds)

    def _watch():
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                sys.stderr.write(f"WALL_STOP: {wall_seconds}s wall deadline reached; leaving non-zero\n")
                sys.stderr.flush()
                os._exit(exit_code)
            time.sleep(min(0.2, max(0.01, left)))

    threading.Thread(target=_watch, name="wall_stop", daemon=True).start()
    return {"name": "wall", "mechanism": "an in-child watchdog thread that calls os._exit at the deadline",
            "enforced_by": "process", "executable": True, "wall_seconds": float(wall_seconds),
            "outer_stop": outer, "exit_code": exit_code}


@contextmanager
def stage_stop(stage: str, *, wall_seconds: float | None = None, cpu_seconds: float | None = None,
               poll_seconds: float = 0.05):
    """One stage's own executable stop: a timer for its wall and a poll for its CPU.

    The CPU budget is checked at entry, so a stage whose budget is already spent never starts, and
    polled while it runs.  The wall deadline is a real timer: a stage that hangs dies inside it.
    """
    st = {"stage": stage, "wall_seconds": None, "cpu_seconds": None, "stopped": False,
          "wall_budget_seconds": wall_seconds, "cpu_budget_seconds": cpu_seconds,
          "mechanism": "setitimer(ITIMER_REAL) for the stage wall; a CPU poll thread for the stage CPU",
          "enforced_by": "process", "executable": True}
    t0, c0 = time.monotonic(), time.process_time()

    if cpu_seconds is not None and c0 > float(cpu_seconds):
        st["stopped"] = True
        st["stop"] = "CPU_BUDGET_ALREADY_SPENT"
        raise StageStopped(f"STAGE_STOP {stage}: the CPU budget of {cpu_seconds}s was already spent "
                           f"({c0:.3f}s) before this stage began; it does not start")

    prev = None
    stop_poll = threading.Event()

    def _on_alarm(signum, frame):
        raise StageStopped(f"STAGE_STOP {stage}: the stage wall deadline of {wall_seconds}s fired")

    def _poll():
        while not stop_poll.wait(poll_seconds):
            if cpu_seconds is not None and (time.process_time() - c0) > float(cpu_seconds):
                sys.stderr.write(f"STAGE_STOP {stage}: stage CPU budget of {cpu_seconds}s exceeded\n")
                sys.stderr.flush()
                os._exit(71)

    poller = None
    try:
        if wall_seconds is not None:
            prev = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, float(wall_seconds))
        if cpu_seconds is not None:
            poller = threading.Thread(target=_poll, name=f"stage_cpu:{stage}", daemon=True)
            poller.start()
        yield st
    except StageStopped:
        st["stopped"] = True
        st.setdefault("stop", "STAGE_WALL_DEADLINE")
        raise
    finally:
        stop_poll.set()
        if wall_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if prev is not None:
                signal.signal(signal.SIGALRM, prev)
        st["wall_seconds"] = time.monotonic() - t0
        st["cpu_seconds"] = time.process_time() - c0


def stops_declaration(*, cpu_seconds: float, wall_seconds: float, stages,
                      host_envelope_bytes: int, device_envelope_bytes: int) -> dict:
    """Every stop this pilot declares, each with the mechanism that executes it.

    F3's second sentence: CPU, wall and every stage need an executable stop, not a row in a table.
    Each row here names the code that stops the run, and `executable` is true only because the test
    beside this module kills a real child with it.
    """
    stops = [
        {"name": "cpu", "budget_seconds": float(cpu_seconds),
         "mechanism": "RLIMIT_CPU (setrlimit) + SIGXCPU, SIGKILL at the hard limit",
         "enforced_by": "kernel", "executable": True},
        {"name": "wall", "budget_seconds": float(wall_seconds),
         "mechanism": "an in-child watchdog thread that calls os._exit, inside the launcher's own -t limit",
         "enforced_by": "process", "executable": True,
         "outer_stop": "the launcher's -t wall limit, which does not depend on the child"},
        {"name": "host_envelope", "budget_bytes": int(host_envelope_bytes),
         "mechanism": "the cgroup's MemoryMax, set by the launcher; the kernel kills the tree",
         "enforced_by": "kernel", "executable": True},
        {"name": "device_envelope", "budget_bytes": int(device_envelope_bytes),
         "mechanism": "tf.config.set_logical_device_configuration, predeclared before device init: "
                      "an oversized request fails with ResourceExhaustedError where it is made",
         "enforced_by": "framework_allocator", "executable": True,
         "enforcement": ENFORCED_AT_ALLOCATION,
         "not_this": "an allocator statistic compared later, which is " + OBSERVED_AFTER_THE_FACT},
    ]
    for s in stages:
        stops.append({"name": f"stage:{s}",
                      "mechanism": "setitimer(ITIMER_REAL) for the stage wall and a CPU poll that exits",
                      "enforced_by": "process", "executable": True})
    return {"schema": SCHEMA, "stops": stops,
            "reading": ("each row names the mechanism that stops the run. A budget with no mechanism "
                        "is not a stop and does not appear here")}


# --- process and whole-device scopes, kept apart --------------------------------------------------

def process_rss_peak() -> dict:
    return {"bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "status": MEASURED, "basis": "getrusage(RUSAGE_SELF).ru_maxrss"}


def whole_device_memory(index: int = 0) -> dict:
    """Whole-device memory from the driver. Its own scope: other tenants and contexts are inside it."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) != 0:
            raise OSError("cuInit failed")
    except OSError as e:
        return {"bytes": None, "total_bytes": None, "status": UNKNOWN, "source": "libcuda",
                "why": f"the driver could not be interrogated: {e}"}
    dev, ctx = ctypes.c_int(0), ctypes.c_void_p()
    free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
    if lib.cuDeviceGet(ctypes.byref(dev), index) != 0 or \
       lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
        return {"bytes": None, "total_bytes": None, "status": UNKNOWN, "source": "libcuda",
                "why": "no context could be created on this device to ask for its memory"}
    try:
        if lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
            return {"bytes": None, "total_bytes": None, "status": UNKNOWN, "source": "libcuda.cuMemGetInfo",
                    "why": "cuMemGetInfo failed"}
        return {"bytes": int(total.value - free.value), "total_bytes": int(total.value),
                "free_bytes": int(free.value), "status": MEASURED, "source": "libcuda.cuMemGetInfo",
                "note": "IN USE BY EVERY PROCESS on this device, including this one's context"}
    finally:
        lib.cuCtxDestroy_v2(ctx)


# --- self-check: the instrument, against a real device and a real kernel --------------------------

def selfcheck(*, declared_uuid: str | None, placement: str, envelope_bytes: int,
              probe_bytes: int = 64 << 20) -> dict:
    """Run the instrument end to end on this host and report what each API actually answered.

    It is an instrument check, not the pilot: it builds no model, opens no data and trains nothing.
    """
    out = {"schema": SCHEMA, "kind": "SELFCHECK", "placement": placement.upper(),
           "declared_uuid": declared_uuid, "python": sys.version.split()[0],
           "visible_devices": _visible(None)}
    stops = stops_declaration(cpu_seconds=120, wall_seconds=240, stages=["probe"],
                              host_envelope_bytes=2 << 30, device_envelope_bytes=envelope_bytes)
    out["stops_declared"] = stops
    out["cpu_stop"] = install_cpu_stop(120)
    out["wall_stop"] = install_wall_stop(240)
    try:
        tf = _tf()
        out["tensorflow_version"] = tf.__version__
    except Exception as e:                                                    # noqa: BLE001
        out["tensorflow_version"] = None
        out["status"] = UNKNOWN
        out["why"] = f"TensorFlow could not be imported: {type(e).__name__}: {e}"
        return out

    if placement.upper() == "GPU":
        out["device_envelope"] = install_device_envelope(envelope_bytes, tf=tf)
    out["device_verification"] = verify_declared_device(declared_uuid=declared_uuid,
                                                        placement=placement, tf=tf)
    device = "GPU:0" if out["device_verification"].get("establishes_a_gpu_pilot") else "CPU:0"
    out["measured_device"] = device
    out["reset_before"] = reset_tf_allocator_peak(device, tf=tf)
    with stage_stop("probe", wall_seconds=120, cpu_seconds=120) as st:
        try:
            with tf.device("/GPU:0" if device == "GPU:0" else "/CPU:0"):
                t = tf.zeros([int(probe_bytes) // 4], dtype=tf.float32)
                out["probe_tensor_device"] = str(getattr(t, "device", None))
                out["probe_tensor_bytes"] = int(probe_bytes)
            del t
        except Exception as e:                                                # noqa: BLE001
            out["probe_error"] = f"{type(e).__name__}: {e}"
    out["probe_stage"] = {k: st[k] for k in ("stage", "stopped", "wall_seconds", "cpu_seconds")}
    fw = tf_allocator_memory(device, tf=tf)
    out["framework_allocator"] = fw
    out["envelope_observation"] = envelope_observation(
        fw, envelope_bytes=envelope_bytes,
        expected_device="GPU:0" if placement.upper() == "GPU" else "CPU:0")
    out["record"] = scope_record(cell_id="SELFCHECK", stage="probe", framework=fw,
                                 process_rss=process_rss_peak(),
                                 whole_device=whole_device_memory() if device == "GPU:0" else None)

    if placement.upper() == "GPU" and out.get("device_envelope", {}).get("enforcement") == ENFORCED_AT_ALLOCATION:
        # the enforceable mechanism, exercised: a request past the predeclared arena must FAIL where
        # it is made, and the failure is the proof the statistics comparison cannot give.
        try:
            with tf.device("/GPU:0"):
                big = tf.zeros([int(envelope_bytes // 4) * 4], dtype=tf.float32)
            del big
            out["oversized_request"] = {"failed": False, "status": UNKNOWN,
                                        "why": "the oversized request did not fail; the envelope is "
                                               "not demonstrated to be in force"}
        except Exception as e:                                                # noqa: BLE001
            out["oversized_request"] = {"failed": True, "status": MEASURED,
                                        "error": type(e).__name__,
                                        "enforcement": ENFORCED_AT_ALLOCATION,
                                        "reading": "the framework allocator refused a request past "
                                                   "the predeclared arena, at the request"}
    out["status"] = MEASURED
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--placement", default="CPU", choices=["CPU", "GPU", "cpu", "gpu"])
    p.add_argument("--device-uuid", default=None)
    p.add_argument("--envelope-bytes", type=int, default=2 << 30)
    p.add_argument("--probe-bytes", type=int, default=64 << 20)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if not a.selfcheck:
        p.error("nothing to do: pass --selfcheck")
    rec = selfcheck(declared_uuid=a.device_uuid, placement=a.placement,
                    envelope_bytes=a.envelope_bytes, probe_bytes=a.probe_bytes)
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
