# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for pretrainer.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
# import pytest
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../feature-extractor"))
)


# ---------------------------------------------------------------- DR01 (order 2026-09-26)
# Every launch path now takes an atomic reservation from tools/crispdm_admission.py.  A test
# must never be decided by the coordinator's live memory, and the order forbids validating any
# of this by pressuring real RAM, so each test gets its OWN lease directory and a SIMULATED,
# generous host.  A test that wants to exercise admission itself overrides both variables (see
# tests/test_crispdm_admission.py, which builds its own readings file per case).
import json as _json

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _crispdm_admission_sandbox(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("crispdm_admission")
    readings = sandbox / "readings.json"
    readings.write_text(_json.dumps({
        "mem_available_bytes": 512 * (1 << 30),
        "mem_total_bytes": 1024 * (1 << 30),
        "slice_memory_max": None,
        "slice_memory_current": 0,
        "pressure_some_avg10": 0.0,
        "alive": {}, "cgroup_current": {}, "cgroup_peak": {},
    }))
    monkeypatch.setenv("CRISPDM_ADMISSION_DIR", str(sandbox / "admission"))
    monkeypatch.setenv("CRISPDM_ADMISSION_RESOURCES_JSON", str(readings))
    yield


# ---------------------------------------------------------------- RR02 (order 2026-09-26)
# A fit path now PROVES a live reservation covers it: tools/df_benchmark_contract.bind() asks
# tools/df_admission_guard, which asks crispdm_admission inside-scope.  On the fleet that coverage
# comes from crispdm-run, which is how every heavy command of this campaign is started.  A test
# declares the same coverage inside its OWN sandbox lease store, over this process's own cgroup --
# the mechanism tests/test_df_mod_e0_close.py::_reserved_scope already used for the closure CLI.
#
# What this does and does not mean, stated plainly because it bounds what the suite proves:
#   * it allocates nothing, starts no fit, reads no real capacity and grants nothing outside the
#     per-test sandbox directory;
#   * it means the guard is SATISFIED during tests, so the suite does not prove the refusal
#     direction here.  That direction is proved directly, both ways, in
#     tests/test_df_admission_guard.py and tests/test_crispdm_rr02_monitor.py.
#
# A test whose subject IS the absence of coverage opts out with
# `pytestmark = pytest.mark.crispdm_uncovered` (tests/test_df_admission_guard.py does).
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "crispdm_uncovered: this test's subject is the ABSENCE of a covering reservation; the "
        "session-level declared coverage must not be written for it")


@_pytest.fixture(autouse=True)
def _crispdm_declared_coverage(request, _crispdm_admission_sandbox):
    if request.node.get_closest_marker("crispdm_uncovered"):
        yield
        return
    import importlib.util as _ilu
    import os as _os
    from pathlib import Path as _Path

    tools = _Path(__file__).resolve().parents[1] / "tools"
    spec = _ilu.spec_from_file_location("crispdm_admission", tools / "crispdm_admission.py")
    A = _ilu.module_from_spec(spec)
    spec.loader.exec_module(A)
    store, res = A.Store(), A.resources_from_env()
    d = A.acquire(store, res, A.Request(name="pytest-declared-coverage", cap_bytes=2 << 30,
                                        wall_seconds=1800), A.now_from_env())
    if d["verdict"] == A.ADMITTED:
        try:
            cg = _Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::", 1)[1]
        except (OSError, IndexError):
            cg = None
        A.arm(store, res, d["lease_id"], A.now_from_env(), pid=_os.getpid(),
              cgroup=(cg or "").lstrip("/") or None)
    yield
