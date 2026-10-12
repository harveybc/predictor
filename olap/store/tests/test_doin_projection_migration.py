"""Additive migration rehearsal on a disposable DuckDB cube built by the DEPLOYED code.

The base providers are exported from git (5c383d17, byte-identical to the wheels the live
warehouse runs for query.py/fs4_store.py/provider.py), a throwaway cube is created and given a
governed report by that code in a subprocess, then it is reopened by this branch's provider.
Existing rows and relations must be untouched; only the doin_0001 relations may appear.
"""

import json
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = "5c383d17"

_OLD = r"""
import json, sys
sys.path[:0] = [sys.argv[1] + "/olap/store/src", sys.argv[1] + "/olap/duckdb_store/src"]
from predictor_duckdb_store.provider import PredictorDuckdbStore
from predictor_olap_store.query import report_sha256
s = PredictorDuckdbStore()
s.set_params(duckdb_path=sys.argv[2], schema="main", memory_limit="512MB", threads=1, min_free_bytes=1)
r = {"experiment_key": "rehearsal:old", "actor": "predictor", "lake": "LOCAL", "lineage": "UNVERIFIED",
     "metrics": [{"metric": "MAE", "value": 0.25, "split": "validation", "horizon": 12}]}
r["report_sha256"] = report_sha256(r)
print(json.dumps(s.write_metrics(r)))
with s.engine().connect() as c:
    rel = sorted(x[0] for x in c.exec_driver_sql(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall())
print(json.dumps(rel))
s._engine.dispose()
"""


def test_doin_0001_is_additive_over_a_cube_made_by_the_deployed_code(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", BASE, "olap/store/src", "olap/duckdb_store/src"],
                             capture_output=True, check=True).stdout
    base = tmp_path / "base"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(base, filter="data")
    cube = tmp_path / "cube.duckdb"
    out = subprocess.run([sys.executable, "-c", _OLD, str(base), str(cube)], capture_output=True, text=True,
                         check=True, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}).stdout.splitlines()
    assert json.loads(out[0])["stored"] is True
    old_relations = set(json.loads(out[1]))
    assert "gov_doin_projection" not in old_relations

    sys.path[:0] = [str(ROOT / "olap/store/src"), str(ROOT / "olap/duckdb_store/src")]
    from predictor_duckdb_store.provider import PredictorDuckdbStore
    from predictor_olap_store import doin_projection_store as dps

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(cube), schema="main", memory_limit="512MB", threads=1, min_free_bytes=1)
    try:
        with store.engine().connect() as c:
            relations = {x[0] for x in c.exec_driver_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()}
            assert old_relations <= relations
            assert relations - old_relations == set(dps.ALL_RELATIONS)
            assert c.exec_driver_sql("SELECT metric, value FROM main.gov_metric").fetchall() == [("MAE", 0.25)]
            assert c.exec_driver_sql("SELECT count(*) FROM main.gov_doin_metric_canonical").scalar() == 0
        # a second open re-applies nothing destructive
        store._engine.dispose()
        store._engine = None
        with store.engine().connect() as c:
            assert c.exec_driver_sql("SELECT count(*) FROM main.gov_report").scalar() == 1
    finally:
        if store._engine is not None:
            store._engine.dispose()
