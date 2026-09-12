#!/usr/bin/env python3
"""Independent PRE cases found while auditing C67-C86.

This script uses only temporary fixtures. It does not read or modify a
campaign root, the data lake or the OLAP database.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))

from descriptor_custody import Custody  # noqa: E402
import lake_descriptor_recompute as recompute  # noqa: E402
import verify_lake_terminals as verify  # noqa: E402


def parquet_bytes(values: np.ndarray, column: str = "close") -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({"timestamp": pa.array(range(len(values))),
                             column: pa.array(values)}), buf)
    return buf.getvalue()


def terminal_name(variable_id: str) -> str:
    return hashlib.sha256(variable_id.encode()).hexdigest()[:32] + ".json"


def fixture(root: Path):
    values = np.arange(1, 17, dtype=np.float64)
    payload = parquet_bytes(values)
    lake = root / "lake"
    (lake / "features").mkdir(parents=True)
    (lake / "features" / "source.parquet").write_bytes(payload)
    appearance = {
        "appearance_id": "app_0",
        "relative_path": "features/source.parquet",
        "physical_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    variable = {"variable_id": "var_a", "appearances": ["app_0"],
                "concept_name": "close", "entity": "entity_a"}
    body = {"appearances": [appearance], "variables": [variable]}
    census_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["census_sha256"] = census_sha
    census_dir = root / "census"
    census_dir.mkdir()
    census_path = census_dir / f"census-{census_sha}.json"
    census_path.write_text(json.dumps(body))

    state = root / "state"
    (state / "terminals").mkdir(parents=True)
    ledger = {
        "census_sha256": census_sha, "censused_at": "fixture",
        "conceptual_variables": 1, "identities": ["var_a"],
        "physical_appearances": 1, "pre_ledger_sha256": "fixture",
        "rule": "fixture", "schema": verify.LEDGER_SCHEMA,
        "written_at": "fixture",
    }
    (state / "PRE_LEDGER.json").write_text(json.dumps(ledger))
    terminal = {
        "appearance": "app_0", "batch": "fixture",
        "concept_name": "close", "descriptors": 25,
        "entity": "entity_a", "measured_at": "fixture",
        "not_identifiable": 1, "outcome": "MEASURED",
        "rows_used": 11, "variable_id": "var_a",
    }
    tpath = state / "terminals" / terminal_name("var_a")
    tpath.write_text(json.dumps(terminal))
    return state, lake, census_path, census_sha, tpath


def child_directory_replacement() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        child = root / "cell"
        child.mkdir()
        (child / "terminal.json").write_text('{"wall":1}')
        custody = Custody(root, require_owner=False)
        try:
            os.rename(child, root / "cell.original")
            (root / "cell").mkdir()
            (root / "cell" / "terminal.json").write_text('{"wall":9}')
            value = custody.walk_to("cell").read("terminal.json").json()
            print("CHILD_DIRECTORY_REPLACEMENT", "ACCEPTED", value)
        finally:
            custody.close()


def stale_census_digest() -> None:
    with tempfile.TemporaryDirectory() as td:
        state, lake, census_path, expected, _ = fixture(Path(td))
        replacement = parquet_bytes(np.arange(101, 117, dtype=np.float64))
        (lake / "features" / "replacement.parquet").write_bytes(replacement)
        census = json.loads(census_path.read_text())
        app = census["appearances"][0]
        app["relative_path"] = "features/replacement.parquet"
        app["physical_sha256"] = hashlib.sha256(replacement).hexdigest()
        app["size_bytes"] = len(replacement)
        census_path.write_text(json.dumps(census))
        result = verify.verify(state, lake, census_path,
                               expected_census_sha256=expected)
        print("STALE_CENSUS_DIGEST", result["population"]["verdict"],
              result["_bound"]["var_a"]["logical_id"])


def duplicate_json_key() -> None:
    with tempfile.TemporaryDirectory() as td:
        state, lake, census_path, expected, tpath = fixture(Path(td))
        terminal = json.loads(tpath.read_text())
        raw = json.dumps(terminal)[:-1] + \
            ',"rows_used":999,"rows_used":11}'
        tpath.write_text(raw)
        result = verify.verify(state, lake, census_path,
                               expected_census_sha256=expected)
        print("DUPLICATE_JSON_KEY", result["population"]["verdict"],
              result["_docs"]["var_a"]["rows_used"])


def numeric_datetime_sentinel() -> None:
    sentinel = np.iinfo(np.int64).min
    payload = parquet_bytes(np.full(32, sentinel, dtype=np.int64),
                            "announcement_datetime_local_utc")
    window, rows, why = verify.column_window(
        payload, "announcement_datetime_local_utc")
    result = recompute.recompute(window)
    print("NUMERIC_DATETIME_SENTINEL", "rows", rows, "why", why,
          "missing", result["missing_count"][0],
          "mean", result["mean"][0],
          "entropy", result["discrete_entropy_bits"][0])


if __name__ == "__main__":
    child_directory_replacement()
    stale_census_digest()
    duplicate_json_key()
    numeric_datetime_sentinel()
