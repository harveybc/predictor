"""C162: the campaign is distributed by planned memory, deterministically, with
nothing dropped; a role's jobs file resolves on the host's own paths and never
repeats a dataset."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CP = _load("df_campaign_plan")
RUN = _load("df_profile_run")
GIB = 1 << 30


def items(peaks):
    return [{"bank": "FINANCIAL", "selector": f"FINANCIAL:d{i}", "planned_peak_bytes": p} for i, p in enumerate(peaks)]


def test_every_dataset_gets_exactly_one_role_heavy_ones_only_where_they_fit():
    its = items([9 * GIB, 7 * GIB, 4 * GIB, 3 * GIB, GIB, GIB // 2, GIB // 4, 100, None])
    CP.assign(its)
    assert all(i["role"] in CP.ROLE_CAPS for i in its)
    for i in its:
        assert (i["planned_peak_bytes"] or 0) <= CP.ROLE_CAPS[i["role"]][0]
    assert {i["role"] for i in its if (i["planned_peak_bytes"] or 0) > 5 * GIB} == {"WORKER_A"}
    assert {i["role"] for i in its if (i["planned_peak_bytes"] or 0) > 2 * GIB} <= {"WORKER_A", "WORKER_B"}


def test_assignment_is_deterministic_and_marks_what_no_cap_admits():
    a, b = items([3 * GIB, 11 * GIB, GIB, GIB]), items([3 * GIB, 11 * GIB, GIB, GIB])
    CP.assign(a)
    CP.assign(list(reversed(b)))
    assert [i["role"] for i in a] == [i["role"] for i in b]
    over = [i for i in a if i.get("assignment_note")]
    assert len(over) == 1 and over[0]["role"] == "WORKER_A" and over[0]["planned_peak_bytes"] == 11 * GIB


def _contracts(tmp_path):
    doc = {"contracts": [{"dataset_id": "fin.a", "files": [{"name": "a.parquet", "sha256": "0" * 64}]},
                         {"dataset_id": "fin.b", "files": [{"name": "b.parquet", "sha256": "0" * 64}]}]}
    p = tmp_path / "contracts.json"
    p.write_text(json.dumps(doc))
    return p


def test_jobs_file_selectors_resolve_on_this_hosts_paths(tmp_path):
    import argparse
    cf = _contracts(tmp_path)
    a = argparse.Namespace(public_panels=None, synthetic_bank=None, financial_contracts=cf, financial_root=tmp_path / "r")
    job = RUN.select_job("FINANCIAL:fin.b", a)
    assert job["index"] == 1 and job["root"] == str(tmp_path / "r")
    with pytest.raises(SystemExit, match="matches 0 datasets"):
        RUN.select_job("FINANCIAL:fin.missing", a)


def test_a_jobs_file_that_repeats_a_dataset_is_refused(tmp_path):
    cf = _contracts(tmp_path)
    jobs = tmp_path / "JOBS.txt"
    jobs.write_text("FINANCIAL:fin.a\nFINANCIAL:fin.a\n")
    with pytest.raises(SystemExit, match="repeats a selector"):
        RUN.main(["--out", str(tmp_path / "out"), "--financial-contracts", str(cf), "--financial-root",
                  str(tmp_path), "--jobs-file", str(jobs)])
    assert not (tmp_path / "out").exists()
