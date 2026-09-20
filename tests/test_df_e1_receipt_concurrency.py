"""RP55: a run's receipts survive children that write them at the same time.

The defect, observed in production. With `--parallel 3`, `DELIVERIES.json` and `TERMINAL_RECEIPTS.json`
were each written by read-modify-write: a child read the whole document, added its own unit and wrote
the whole document back. Three children acquiring at once therefore erased each other's entries, and
the run stopped with

    REFUSED: unit 'R0_s2' has no delivery, so it has no terminal to report

after seven children had already completed — four units' campaigns and verified deliveries existed on
the governance service while the local receipt no longer named them.

These rules drive the two writes from real concurrent PROCESSES (threads alone would not exercise the
file lock across the process boundary the children actually cross), and then check the repair tool
against the service's own record. The accounting database here is a synthetic fixture with the
deployed service's schema, declared as such; nothing in this file is an experiment.
"""
import importlib.util
import json
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
UNITS = [f"u{i:02d}" for i in range(24)]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("df_e1_governed")
RC = _load("df_e1_receipts")
RV = _load("df_e1_recover")
DESIGN_SHA = "d" * 64


def _spawn(script: str, args: list[str]) -> list[subprocess.Popen]:
    body = textwrap.dedent(script)
    return [subprocess.Popen([sys.executable, "-c", body, str(TOOLS), arg]) for arg in args]


def _join(procs):
    for p in procs:
        assert p.wait(timeout=120) == 0


def test_RP55_concurrent_children_never_erase_each_others_deliveries(tmp_path):
    path = tmp_path / "DELIVERIES.json"
    path.write_text(json.dumps({"schema": G.SCHEMA, "design_sha256": DESIGN_SHA, "units": {}}))
    script = """
        import importlib.util, json, sys
        tools, unit = sys.argv[1], sys.argv[2]
        spec = importlib.util.spec_from_file_location("g", tools + "/df_e1_governed.py")
        g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
        path = __import__("pathlib").Path(tools + "/../DELIVERIES.json")
    """
    # the file lives with the test, not the tools: pass it through the environment of the child
    script = script.replace('path = __import__("pathlib").Path(tools + "/../DELIVERIES.json")',
                            f'path = __import__("pathlib").Path({str(path)!r})')
    script += """
        base = json.loads(path.read_text())
        g._merge_write(path, unit, {"campaign_key": unit, "campaign_sha256": "c" * 64, "cached": False,
                                    "bytes": 7}, base)
    """
    _join(_spawn(script, UNITS))
    doc = json.loads(path.read_text())
    assert sorted(doc["units"]) == sorted(UNITS), "a concurrent writer erased another unit's delivery"
    assert doc["transfer"]["transferred_units"] == len(UNITS)


def test_RP55_concurrent_children_never_erase_each_others_terminals(tmp_path):
    script = f"""
        import importlib.util, json, sys
        tools, unit = sys.argv[1], sys.argv[2]
        spec = importlib.util.spec_from_file_location("rc", tools + "/df_e1_receipts.py")
        rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
        rc.record_accepted({str(tmp_path)!r}, unit, campaign_sha256="c" * 64, campaign_key=unit,
                           terminal={{"status": "COMPLETED", "generation": 1, "metrics": [], "deliveries": []}},
                           receipt={{"terminal_sha256": "t" * 64}},
                           reconciliation={{"http": 200, "missing_units": [], "accounting_only": [], "lake_only": []}},
                           design_sha256={DESIGN_SHA!r})
    """
    _join(_spawn(script, UNITS))
    doc = json.loads((tmp_path / "TERMINAL_RECEIPTS.json").read_text())
    assert sorted(doc["units"]) == sorted(UNITS), "a concurrent writer erased another unit's terminal"


def test_RP55_a_partial_write_is_never_left_behind(tmp_path):
    """Every reader saw a whole document: the write goes through a temporary file and a rename."""
    path = tmp_path / "DELIVERIES.json"
    base = {"schema": G.SCHEMA, "design_sha256": DESIGN_SHA, "units": {}}
    path.write_text(json.dumps(base))
    for unit in UNITS[:4]:
        G._merge_write(path, unit, {"campaign_key": unit, "cached": True, "bytes": 1}, json.loads(path.read_text()))
        json.loads(path.read_text())                     # parses at every step, never truncated
    assert not (tmp_path / "DELIVERIES.json.tmp").exists()


# ---------------------------------------------------------------- the repair of a damaged root

def _service_db(path: Path, design_sha: str, *, units, state="VERIFIED_CACHE", sha="a" * 64, bytes_=11):
    """A SYNTHETIC accounting database with the deployed service's schema (governed_campaigns,
    governed_deliveries, governed_terminals). Declared: it is a fixture, not the live service."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE governed_campaigns (campaign_sha256 TEXT PRIMARY KEY, campaign_key TEXT NOT NULL,
            actor TEXT, classification TEXT, terminal_lake TEXT, body_json TEXT NOT NULL, created_at TEXT);
        CREATE TABLE governed_deliveries (delivery_id TEXT PRIMARY KEY, campaign_sha256 TEXT, unit_id TEXT,
            actor TEXT, lake_id TEXT, resource_id TEXT, role TEXT, range_from TEXT, range_to TEXT,
            sha256 TEXT, bytes INTEGER, source_sha256 TEXT, delivery_kind TEXT, time_column TEXT,
            availability_contract_sha256 TEXT, state TEXT, cached INTEGER, created_at TEXT, verified_at TEXT);
        CREATE TABLE governed_terminals (terminal_sha256 TEXT PRIMARY KEY, campaign_sha256 TEXT,
            unit_id TEXT, generation INTEGER, status TEXT, terminal_lake TEXT, body_json TEXT, created_at TEXT);
    """)
    for i, unit in enumerate(units):
        key, csha = f"run-{unit}-data", f"{i:064d}"
        body = {"campaign_key": key, "config_sha256": design_sha, "units": [unit],
                "code_identity": {"kind": "git_commit", "value": "f" * 40}}
        con.execute("INSERT INTO governed_campaigns VALUES (?,?,?,?,?,?,?)",
                    (csha, key, "predictor", "NON_GOVERNING", "olap_cube", json.dumps(body), "2026-09-20T00:00:00Z"))
        con.execute("INSERT INTO governed_deliveries (delivery_id, campaign_sha256, unit_id, sha256, bytes, "
                    "state, cached, created_at, verified_at, availability_contract_sha256) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f"d{i:031d}", csha, unit, sha, bytes_, state, 1, "2026-09-20T00:00:00Z",
                     "2026-09-20T00:01:00Z", "b" * 64))
    con.commit()
    con.close()


@pytest.fixture
def damaged(tmp_path):
    design = {"design_sha256": DESIGN_SHA, "pilots": [{"cell_id": "pilot"}],
              "cells": [{"cell_id": "lost"}, {"cell_id": "kept"}, {"cell_id": "never_run"}]}
    root = tmp_path / "root"
    (root / "cache" / "public_panels").mkdir(parents=True)
    (root / "DESIGN.json").write_text(json.dumps(design))
    payload = b"panel bytes"
    sha = __import__("hashlib").sha256(payload).hexdigest()
    (root / "cache" / "public_panels" / f"{sha}.parquet").write_bytes(payload)
    (root / "DELIVERIES.json").write_text(json.dumps({
        "schema": G.SCHEMA, "design_sha256": DESIGN_SHA, "run_id": "run", "lake": "public_panels", "host": "omega",
        "units": {"kept": {"campaign_key": "run-kept-data", "campaign_sha256": "k" * 64, "delivery_id": "keep",
                           "sha256": sha, "cached": True}}}))
    db = tmp_path / "accounting.db"
    _service_db(db, DESIGN_SHA, units=["prepare", "pilot", "lost", "kept"], sha=sha, bytes_=len(payload))
    return {"root": root, "db": db, "sha": sha}


def test_RP55_the_repair_restores_exactly_what_the_service_holds(damaged):
    out = RV.apply(damaged["root"], damaged["db"])
    assert sorted(out["applied"]) == ["lost", "pilot", "prepare"]
    assert out["refused"] == {} and out["unknown_to_the_service"] == ["never_run"]
    doc = json.loads((damaged["root"] / "DELIVERIES.json").read_text())
    assert doc["units"]["kept"]["delivery_id"] == "keep"                  # an existing entry is untouched
    restored = doc["units"]["lost"]
    assert restored["sha256"] == damaged["sha"] and restored["bytes_on_disk_sha256"] == damaged["sha"]
    assert restored["verification_state"] == "VERIFIED_CACHE" and restored["restored"] == RV.SOURCE
    assert restored["code_identity"] == {"kind": "git_commit", "value": "f" * 40}
    assert not (damaged["root"] / "TERMINAL_RECEIPTS.json").exists()      # no terminal is ever invented


@pytest.mark.parametrize("case", ["unverified_state", "bytes_absent", "bytes_differ", "other_design"])
def test_RP55_the_repair_refuses_what_it_cannot_prove(damaged, tmp_path, case):
    db = tmp_path / "other.db"
    if case == "other_design":
        _service_db(db, "e" * 64, units=["lost"], sha=damaged["sha"])
    elif case == "unverified_state":
        _service_db(db, DESIGN_SHA, units=["lost"], state="PENDING", sha=damaged["sha"])
    else:
        _service_db(db, DESIGN_SHA, units=["lost"], sha="9" * 64)
        if case == "bytes_differ":
            (damaged["root"] / "cache" / "public_panels" / f"{'9' * 64}.parquet").write_bytes(b"not the bytes")
    out = RV.plan(damaged["root"], db)
    assert "lost" not in out["restore"] and "lost" in out["refused"], out
    expected = {"unverified_state": "not verified", "bytes_absent": "not on this host",
                "bytes_differ": "not the bytes the service delivered",
                "other_design": "another design"}[case]
    assert expected in out["refused"]["lost"]
