"""K5: the delta between two verified matrices attributes every movement to a declared cause
or leaves it visibly UNATTRIBUTED; an unverified matrix is refused."""
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


delta = _load("df_d3_matrix_delta")


def matrix(run, freeze, verdicts, tests, verified=True):
    return {"schema": "d3_mechanics_matrix_verified.v1", "run_id": run, "freeze_sha256": freeze,
            "verified": verified,
            "population": {"units": 2, "variables": 2, "operators": 1, "tests": 12,
                           "design_sha256": "x"},
            "units": {"expected": 2, "completed": 2, "failed": 0, "missing": []},
            "operators": {"q": {"verdicts": verdicts, "tests": tests}}}


def freeze(design, spec):
    return {"design_sha256": design, "operators": [{"kind": "q", "spec_sha256": spec}]}


def test_probe_movements_under_the_amendment_are_attributed_and_others_are_not():
    before = matrix("v1", "f1", {"MECHANICALLY_REFUSED": 2},
                    {"response_probe": {"FAILED": 2}, "prefix_all_available": {"PASSED": 2}})
    after = matrix("v2", "f2", {"MECHANICALLY_ACCEPTED": 2},
                   {"response_probe": {"PASSED": 2}, "prefix_all_available": {"PASSED": 1, "FAILED": 1}})
    d = delta.delta(before, after, freeze("d1", "s1"), freeze("d2", "s2"))
    q = d["operators"]["q"]
    assert q["tests"]["response_probe"]["causes"][0].startswith("07B")
    assert q["tests"]["prefix_all_available"]["causes"] == ["UNATTRIBUTED"]
    assert d["all_attributed"] is False
    assert d["unattributed"] == [{"operator": "q", "test": "prefix_all_available",
                                  "delta": {"FAILED": {"before": 0, "after": 1, "delta": 1},
                                            "PASSED": {"before": 2, "after": 1, "delta": -1}}}]


def test_a_changed_design_digest_alone_is_not_a_population_change():
    m = matrix("v1", "f1", {"MECHANICALLY_ACCEPTED": 2}, {"response_probe": {"PASSED": 2}})
    n = dict(m, run_id="v2", population=dict(m["population"], design_sha256="y"))
    d = delta.delta(m, n, freeze("d1", "s1"), freeze("d2", "s1"))
    assert d["population_changed"] is False and d["design_changed"] is True


def test_no_movement_is_all_attributed_and_an_unverified_matrix_is_refused():
    m = matrix("v1", "f1", {"MECHANICALLY_ACCEPTED": 2}, {"response_probe": {"PASSED": 2}})
    d = delta.delta(m, dict(m, run_id="v2"), freeze("d1", "s1"), freeze("d1", "s1"))
    assert d["all_attributed"] is True and d["operators"]["q"]["tests"] == {}
    with pytest.raises(SystemExit, match="not a verified matrix"):
        delta.delta(m, dict(m, verified=False), freeze("d1", "s1"), freeze("d1", "s1"))
