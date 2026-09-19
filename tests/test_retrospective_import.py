"""RP34: the warehouse's additive `provenance` block — evidence produced before its governance
existed is imported AS THAT, never as a prospective governed result, and never hidden.

The extension is additive in the strict sense proven here: an envelope without the block loads
exactly as before and its rows carry NULL (meaning UNSTATED, not "prospective"); loading the same
retrospective envelope twice changes nothing; and every contradiction a retrospective import could
carry is refused before a row is written.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CE = _load("campaign_envelope_rp34", REPO / "olap" / "campaign_envelope.py")
sqlalchemy = pytest.importorskip("sqlalchemy")


#: The loader's DDL is PostgreSQL/DuckDB (schemas, CREATE OR REPLACE VIEW, FILTER), so the storage
#: rules run on a DISPOSABLE DuckDB file when that engine is importable here, and are skipped with
#: this reason otherwise. The production cube is never opened by these tests; the same storage path
#: is additionally exercised end to end against a disposable warehouse service by
#: tools/df_e1_retrospective_import.py --rehearse, whose report is the evidence of that run.
HAS_DUCKDB = importlib.util.find_spec("duckdb_engine") is not None
needs_engine = pytest.mark.skipif(
    not HAS_DUCKDB, reason="duckdb_engine is not importable in this interpreter; the storage rules run "
                           "in the warehouse rehearsal (tools/df_e1_retrospective_import.py --rehearse)")


def _engine(tmp_path):
    return sqlalchemy.create_engine(f"duckdb:///{tmp_path}/cube.duckdb")


def _doc(**over):
    prov = over.pop("provenance", None)
    base = dict(campaign_key="rp34-import", producer="predictor", result_class="DEVELOPMENT",
                identity={"run_id": "r1", "code_identity": "c" * 40, "design_sha256": "d" * 64},
                data_consumed={"datasets": [], "variables": [], "operators": []},
                partitions={"exposure": "DEVELOPMENT_NO_RESERVE", "splits": "train/validation"},
                budget={"device": "cpu", "wall_seconds": 1.0, "cost_units": 1.0},
                terminal={"state": "COMPLETED", "adjudication": "DESCRIPTIVE"},
                artifacts={"verification": "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED"},
                units=[{"cell_key": "R0_s1", "candidate_key": "modular_A", "metric_name": "e1.mase_validation",
                        "metric_value": 0.8831, "terminal_state": "COMPLETED"}])
    base.update(over)
    return CE.build_envelope(provenance=prov, **base)


RETRO = {"mode": CE.RETROSPECTIVE, "executed_at": "2026-09-19T17:00:00Z", "imported_at": "2026-09-19T23:00:00Z",
         "governed_delivery_at_execution": False,
         "reason": "the household DEV pilot ran before the public panels were a governed resource"}


@needs_engine
def test_RP34_an_envelope_without_the_block_still_loads_and_its_rows_say_unstated(tmp_path):
    engine = _engine(tmp_path)
    CE.ensure_envelope_tables(engine)
    out = CE.load_envelope(engine, _doc())
    assert out["units"] == 1
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT provenance_mode, executed_at, governed_delivery_at_execution FROM public.fact_campaign_unit")).fetchall()
    assert rows and all(r[0] is None and r[1] is None and r[2] is None for r in rows)


@needs_engine
def test_RP34_a_retrospective_import_is_stored_as_such_and_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    CE.ensure_envelope_tables(engine)
    doc = _doc(provenance=RETRO)
    first = CE.load_envelope(engine, doc)
    second = CE.load_envelope(engine, doc)
    assert first["units"] == 1 and second["units"] == 0 and second["skipped_existing"] >= 1
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.text(
            "SELECT provenance_mode, executed_at, governed_delivery_at_execution, import_reason, count(*) "
            "FROM public.fact_campaign_unit GROUP BY 1,2,3,4")).fetchall()
    assert len(row) == 1
    mode, executed, governed, reason, n = row[0]
    assert mode == CE.RETROSPECTIVE and executed == RETRO["executed_at"] and governed is False and n == 1
    assert "before the public panels were a governed resource" in reason


@pytest.mark.parametrize("case,expect", [
    ("claims_governed_delivery", "cannot claim a governed delivery"),
    ("confirmatory_class", "may not be"),
    ("import_before_execution", "cannot precede the execution"),
    ("missing_field", "is missing"),
    ("unknown_field", "undeclared fields"),
    ("unknown_mode", "unknown provenance mode"),
    ("no_reason", "states its reason"),
    ("prospective_without_delivery", "asserts a governed delivery"),
])
def test_RP34_every_contradiction_of_a_retrospective_import_is_refused(tmp_path, case, expect):
    block = dict(RETRO)
    kw = {}
    if case == "claims_governed_delivery":
        block["governed_delivery_at_execution"] = True
    elif case == "confirmatory_class":
        kw["result_class"] = "CONFIRMATION"
    elif case == "import_before_execution":
        block["imported_at"] = "2026-09-18T00:00:00Z"
    elif case == "missing_field":
        block.pop("reason")
    elif case == "unknown_field":
        block["looks_governed"] = True
    elif case == "unknown_mode":
        block["mode"] = "GOVERNED_ENOUGH"
    elif case == "no_reason":
        block["reason"] = "   "
    else:
        block = {**RETRO, "mode": CE.PROSPECTIVE}
    with pytest.raises(SystemExit, match=expect):
        doc = _doc(provenance=block, **kw)
        CE.validate_envelope(doc)
    if HAS_DUCKDB:                                  # and nothing was written on the way to the refusal
        engine = _engine(tmp_path)
        CE.ensure_envelope_tables(engine)
        with engine.connect() as conn:
            n = conn.execute(sqlalchemy.text("SELECT count(*) FROM public.fact_campaign_unit")).scalar()
        assert n == 0, "a refused envelope wrote rows"


def test_RP34_the_block_is_inside_the_envelope_digest(tmp_path):
    a, b = _doc(), _doc(provenance=RETRO)
    assert a["envelope_sha256"] != b["envelope_sha256"]
    forged = json.loads(json.dumps(b))
    forged["provenance"]["mode"] = CE.PROSPECTIVE
    forged["provenance"]["governed_delivery_at_execution"] = True
    with pytest.raises(SystemExit, match="self-digest|digest"):
        CE.validate_envelope(forged)
