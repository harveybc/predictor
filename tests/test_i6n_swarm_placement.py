"""Lane B placement/admission policy (pure functions, injected host facts)."""
import pytest

from tools.i6n_swarm import placement as P

GIB = 1024**3
NOW = 2_000_000.0


def facts(**kw):
    base = dict(now=NOW, boot_id="boot-2", mem_total_bytes=32 * GIB, mem_available_bytes=20 * GIB,
                gpu_free_bytes=20 * GIB, pressure=False, draining=False, cpu_heavy=False,
                healthcheck=dict(boot_id="boot-2", checked_at=NOW - 60, nv_err_no_memory=0,
                                 sunreclaim_bytes=1 * GIB, gpu_probe_ok=True))
    base.update(kw)
    return base


CLASS = dict(class_id="small", child_memory_bytes=5 * GIB, vram_bytes=3 * GIB, small_pilot=True)


def test_5090_preferred_and_exclusive():
    d = P.admit("rtx5090", facts(), CLASS, measured={})
    assert d["admit"] and d["exclusive_host"] and d["cap_bytes"] == 5 * GIB


def test_5090_refused_when_host_memory_short():
    d = P.admit("rtx5090", facts(mem_available_bytes=6 * GIB), CLASS, measured={})
    assert not d["admit"] and d["cause"] == "HOST_MEMORY_BELOW_GATE"


def test_production_cap_is_125_percent_of_measured_peak_only_if_it_fits():
    measured = {"small": dict(peak_bytes=2 * GIB, n=3)}
    d = P.admit("rtx5090", facts(), CLASS, measured=measured)
    assert d["cap_bytes"] == int(2.5 * GIB)
    big = {"small": dict(peak_bytes=30 * GIB, n=3)}
    assert P.admit("rtx5090", facts(), CLASS, measured=big)["cause"] == "HOST_MEMORY_BELOW_GATE"


@pytest.mark.parametrize("health,cause", [
    (None, "HEALTHCHECK_REQUIRED"),
    (dict(boot_id="boot-1", checked_at=NOW - 60, nv_err_no_memory=0, sunreclaim_bytes=GIB, gpu_probe_ok=True), "HEALTHCHECK_STALE_BOOT"),
    (dict(boot_id="boot-2", checked_at=NOW - 7 * 3600, nv_err_no_memory=0, sunreclaim_bytes=GIB, gpu_probe_ok=True), "HEALTHCHECK_STALE"),
    (dict(boot_id="boot-2", checked_at=NOW - 60, nv_err_no_memory=3, sunreclaim_bytes=GIB, gpu_probe_ok=True), "NV_ERR_NO_MEMORY"),
    (dict(boot_id="boot-2", checked_at=NOW - 60, nv_err_no_memory=0, sunreclaim_bytes=12 * GIB, gpu_probe_ok=True), "SLAB_UNRECLAIMABLE_HIGH"),
    (dict(boot_id="boot-2", checked_at=NOW - 60, nv_err_no_memory=0, sunreclaim_bytes=GIB, gpu_probe_ok=False), "GPU_PROBE_FAILED"),
])
def test_4090_requires_fresh_clean_healthcheck(health, cause):
    d = P.admit("rtx4090", facts(healthcheck=health), dict(CLASS, child_memory_bytes=3 * GIB), measured={})
    assert not d["admit"] and d["cause"] == cause


def test_4090_admitted_after_fresh_healthcheck():
    assert P.admit("rtx4090", facts(), dict(CLASS, child_memory_bytes=3 * GIB), measured={})["admit"]


def test_4070_small_pilots_only_with_desktop_reserve():
    assert P.admit("rtx4070", facts(mem_available_bytes=14 * GIB, gpu_free_bytes=6 * GIB), CLASS, measured={})["admit"]
    assert P.admit("rtx4070", facts(), dict(CLASS, small_pilot=False), measured={})["cause"] == "NOT_A_SMALL_PILOT"
    assert P.admit("rtx4070", facts(mem_available_bytes=12 * GIB), CLASS, measured={})["cause"] == "DESKTOP_RAM_RESERVE"
    assert P.admit("rtx4070", facts(gpu_free_bytes=4 * GIB), CLASS, measured={})["cause"] == "DESKTOP_VRAM_RESERVE"
    assert P.admit("rtx4070", facts(cpu_heavy=True, mem_available_bytes=14 * GIB), CLASS, measured={})["cause"] == "HOST_CPU_HEAVY"


def test_4070_governed_workload_requires_measured_peak_and_keeps_reserves():
    governed = dict(CLASS, small_pilot=False, desktop_workload=True)
    assert P.admit("rtx4070", facts(), governed, measured={})["cause"] == "NO_MEASURED_PEAK"
    measured = {"small": dict(peak_bytes=2 * GIB, n=4)}
    admitted = P.admit("rtx4070", facts(mem_available_bytes=16 * GIB, gpu_free_bytes=6 * GIB),
                       governed, measured=measured)
    assert admitted["admit"] and admitted["exclusive_host"]
    assert admitted["cap_bytes"] == int(2.5 * GIB)
    assert P.admit("rtx4070", facts(mem_available_bytes=10 * GIB), governed,
                   measured=measured)["cause"] == "DESKTOP_RAM_RESERVE"


def test_5070ti_needs_aggregate_proof_from_measured_peaks():
    assert P.admit("rtx5070ti", facts(), CLASS, measured={})["cause"] == "NO_MEASURED_PEAK"
    measured = {"small": dict(peak_bytes=4 * GIB, n=2)}
    tight = facts(mem_total_bytes=16 * GIB, slice_limit_bytes=16 * GIB)
    assert P.admit("rtx5070ti", tight, CLASS, measured=measured, sibling_class=CLASS)["cause"] == "AGGREGATE_ADMISSION_FAILED"
    roomy = facts(mem_total_bytes=64 * GIB, slice_limit_bytes=48 * GIB, mem_available_bytes=40 * GIB)
    d = P.admit("rtx5070ti", roomy, CLASS, measured=measured, sibling_class=CLASS)
    assert d["admit"] and d["exclusive_host"] is False
    assert d["evidence"]["required_bytes"] <= 48 * GIB


def test_5070ti_drains_under_pressure():
    measured = {"small": dict(peak_bytes=4 * GIB, n=2)}
    roomy = facts(mem_total_bytes=64 * GIB, slice_limit_bytes=48 * GIB, mem_available_bytes=40 * GIB, pressure=True)
    assert P.admit("rtx5070ti", roomy, CLASS, measured=measured, sibling_class=CLASS)["cause"] == "DRAINING_UNDER_PRESSURE"


def test_unknown_device_class_refused():
    assert P.admit("rtx5070", facts(), CLASS, measured={})["cause"] == "DEVICE_CLASS_NOT_AUTHORIZED"


def test_host_facts_carries_verified_slice_limit_from_healthcheck(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 32768 kB\nMemAvailable: 16384 kB\nSUnreclaim: 1024 kB\n")
    got = P.host_facts(NOW, "boot-2", healthcheck={"slice_limit_bytes": 12345}, meminfo_path=meminfo)
    assert got["slice_limit_bytes"] == 12345
