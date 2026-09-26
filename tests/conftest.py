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
