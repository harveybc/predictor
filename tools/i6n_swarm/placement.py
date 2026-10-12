"""Device admission policy (plan section 7, orders section 4). Pure functions over host facts.

- rtx5090: preferred; one GPU job per host; host must keep its reserve.
- rtx4090: only after a FRESH healthcheck taken in the CURRENT boot showing no
  NV_ERR_NO_MEMORY, bounded unreclaimable slab and a passing GPU probe.
- rtx4070 (coordinator desktop): small pilots only, leaving 8 GiB RAM and
  2 GiB VRAM free, never beside heavy CPU work.
- rtx5070ti: never by default; only with an aggregate proof from MEASURED
  peaks (1.25 x peak of both GPU jobs + wrappers + reserve fits host and
  slice). Drains (no new assignment) under pressure. Never learn by OOM.
Caps: 1.25 x measured class peak when measured and it fits, else the admitted
class cap. A cap is never lowered to make a job fit.
"""
import math

GIB = 1024**3
HOST_RESERVE = 2 * GIB
DESKTOP_RAM = 8 * GIB
DESKTOP_VRAM = 2 * GIB
AGGREGATE_RESERVE = 6 * GIB
WRAPPER_BYTES = GIB // 2
HEALTH_MAX_AGE = 6 * 3600
SLAB_LIMIT = 6 * GIB
AUTHORIZED = ("rtx5090", "rtx4090", "rtx4070", "rtx5070ti", "cpu_fixture")


def _cap(class_spec, measured):
    peak = (measured or {}).get(class_spec["class_id"], {}).get("peak_bytes")
    return math.ceil(1.25 * peak) if peak else int(class_spec["child_memory_bytes"])


def _no(cause, **evidence):
    return dict(admit=False, cause=cause, exclusive_host=True, cap_bytes=None, evidence=evidence)


def admit(device_class, facts, class_spec, measured, sibling_class=None):
    if device_class not in AUTHORIZED:
        return _no("DEVICE_CLASS_NOT_AUTHORIZED")
    cap = _cap(class_spec, measured)
    if device_class == "cpu_fixture":
        return dict(admit=True, cause=None, exclusive_host=False, cap_bytes=cap, evidence={})
    if facts.get("draining"):
        return _no("HOST_DRAINING")
    if device_class == "rtx4090":
        health = facts.get("healthcheck")
        if not health:
            return _no("HEALTHCHECK_REQUIRED")
        if health.get("boot_id") != facts.get("boot_id"):
            return _no("HEALTHCHECK_STALE_BOOT")
        if facts["now"] - health.get("checked_at", 0) > HEALTH_MAX_AGE:
            return _no("HEALTHCHECK_STALE")
        if health.get("nv_err_no_memory", 1) != 0:
            return _no("NV_ERR_NO_MEMORY", count=health.get("nv_err_no_memory"))
        if health.get("sunreclaim_bytes", SLAB_LIMIT + 1) > SLAB_LIMIT:
            return _no("SLAB_UNRECLAIMABLE_HIGH", sunreclaim_bytes=health.get("sunreclaim_bytes"))
        if not health.get("gpu_probe_ok"):
            return _no("GPU_PROBE_FAILED")
    if device_class == "rtx4070":
        governed = bool(class_spec.get("desktop_workload"))
        if not class_spec.get("small_pilot") and not governed:
            return _no("NOT_A_SMALL_PILOT")
        if governed and not (measured or {}).get(class_spec["class_id"], {}).get("peak_bytes"):
            return _no("NO_MEASURED_PEAK")
        if facts.get("cpu_heavy"):
            return _no("HOST_CPU_HEAVY")
        if facts["mem_available_bytes"] - cap < DESKTOP_RAM:
            return _no("DESKTOP_RAM_RESERVE", mem_available_bytes=facts["mem_available_bytes"], cap_bytes=cap)
        if facts.get("gpu_free_bytes", 0) - class_spec.get("vram_bytes", 0) < DESKTOP_VRAM:
            return _no("DESKTOP_VRAM_RESERVE", gpu_free_bytes=facts.get("gpu_free_bytes"))
        return dict(admit=True, cause=None, exclusive_host=True, cap_bytes=cap, evidence={})
    if device_class == "rtx5070ti":
        if facts.get("pressure"):
            return _no("DRAINING_UNDER_PRESSURE")
        own = (measured or {}).get(class_spec["class_id"], {}).get("peak_bytes")
        if not own:
            return _no("NO_MEASURED_PEAK")
        if sibling_class is None:
            return _no("NO_SIBLING_CLASS_DECLARED")
        sib = (measured or {}).get(sibling_class["class_id"], {}).get("peak_bytes")
        if not sib:
            return _no("NO_MEASURED_PEAK")
        required = int(1.25 * own + 1.25 * sib + 2 * WRAPPER_BYTES + AGGREGATE_RESERVE)
        limit = min(facts["mem_total_bytes"], facts.get("slice_limit_bytes", facts["mem_total_bytes"]))
        evidence = dict(required_bytes=required, limit_bytes=limit, own_peak=own, sibling_peak=sib)
        if required > limit or facts["mem_available_bytes"] < int(1.25 * own) + HOST_RESERVE:
            return dict(_no("AGGREGATE_ADMISSION_FAILED"), evidence=evidence)
        return dict(admit=True, cause=None, exclusive_host=False, cap_bytes=int(1.25 * own), evidence=evidence)
    if facts["mem_available_bytes"] < cap + HOST_RESERVE:
        return _no("HOST_MEMORY_BELOW_GATE", mem_available_bytes=facts["mem_available_bytes"], cap_bytes=cap)
    return dict(admit=True, cause=None, exclusive_host=True, cap_bytes=cap, evidence={})


def host_facts(now, boot_id, healthcheck=None, meminfo_path="/proc/meminfo"):
    """Local, read-only facts (no GPU process launched here)."""
    values = {}
    with open(meminfo_path) as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    facts = dict(now=now, boot_id=boot_id, mem_total_bytes=values.get("MemTotal", 0),
                 mem_available_bytes=values.get("MemAvailable", 0), sunreclaim_bytes=values.get("SUnreclaim", 0),
                 healthcheck=healthcheck, pressure=False, draining=False, cpu_heavy=False)
    if healthcheck and type(healthcheck.get("slice_limit_bytes")) is int:
        facts["slice_limit_bytes"] = healthcheck["slice_limit_bytes"]
    return facts
