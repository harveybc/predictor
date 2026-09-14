"""Flow-v3 terminal storage and HTTP reconciliation."""

import hashlib
import json

import pytest

from app.config import DEFAULT_VALUES
from app.main import assemble
from query_plugins.sql_query import Plugin

TOKEN = "test-lake-token"


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _terminal(status="COMPLETED", unit="u001", generation=1):
    body = {
        "schema": "governed_terminal.v1",
        "campaign_sha256": "a" * 64,
        "campaign_key": "campaign-1",
        "classification": "GOVERNING",
        "project": "predictor",
        "actor": "predictor",
        "unit_id": unit,
        "generation": generation,
        "status": status,
        "reason": None if status == "COMPLETED" else "typed outcome",
        "started_at": "2026-09-13T20:00:00Z",
        "finished_at": "2026-09-13T20:00:01Z",
        "costs": {"wall_seconds": 1.0},
        "deliveries": ["1" * 32],
        "artifacts": [],
        "metrics": [],
        "tags": {"purpose": "test"},
        "terminal_lake": "olap_cube",
        "config_sha256": "b" * 64,
        "code_identity": {"kind": "git_commit", "value": "c" * 40},
        "synthetic_spec_sha256": None,
    }
    body["verified_datasets"] = [{
        "delivery_id": "1" * 32,
        "lake_id": "financial_files",
        "resource_id": "market_data/a.csv",
        "role": "x",
        "sha256": "d" * 64,
        "bytes": 123,
        "source_sha256": "e" * 64,
        "range_from": "2024-01-01",
        "range_to": "2024-12-31",
        "delivery_kind": "CUT",
        "time_column": "available_at",
        "availability_contract_sha256": "f" * 64,
        "state": "VERIFIED_TRANSFER",
    }]
    body["terminal_sha256"] = hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
    return body


def _plugin(tmp_path):
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"), holdout_start=None)
    return plugin


def _client(tmp_path):
    config = dict(DEFAULT_VALUES)
    config.update({
        "sqlite_path": str(tmp_path / "cube.sqlite"),
        "holdout_start": None,
        "lake_service_token": TOKEN,
    })
    plugins = assemble(config)
    app = plugins["web"].create_app({"config": config, "plugins": plugins})
    app.config["TESTING"] = True
    return app.test_client()


def test_terminal_store_is_idempotent_and_generation_is_cas(tmp_path):
    plugin = _plugin(tmp_path)
    terminal = _terminal()
    first = plugin.write_terminal(terminal)
    second = plugin.write_terminal(terminal)
    conflict = _terminal("FAILED")
    with pytest.raises(ValueError, match="terminal generation conflict"):
        plugin.write_terminal(conflict)
    assert first["stored"] and second["already_stored"]
    assert plugin.terminal_digests("a" * 64) == [{
        "terminal_sha256": terminal["terminal_sha256"],
        "unit_id": "u001",
        "generation": 1,
    }]


def test_terminal_store_rejects_hash_mismatch(tmp_path):
    terminal = _terminal()
    terminal["status"] = "FAILED"
    with pytest.raises(ValueError, match="terminal_sha256 mismatch"):
        _plugin(tmp_path).write_terminal(terminal)


def test_existing_terminal_dataset_table_is_upgraded_in_place(tmp_path):
    db = tmp_path / "cube.sqlite"
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE gov_terminal_dataset (terminal_sha256 TEXT, delivery_id TEXT, "
            "lake_id TEXT, resource_id TEXT, role TEXT, sha256 TEXT, bytes INTEGER, "
            "source_sha256 TEXT, range_from TEXT, range_to TEXT, delivery_kind TEXT, "
            "time_column TEXT, verification_state TEXT)"
        )
        conn.execute(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, sha256) VALUES ('old', 'digest')"
        )
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(db), holdout_start=None)
    plugin.engine()
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(gov_terminal_dataset)")}
        assert "availability_contract_sha256" in columns
        assert conn.execute("SELECT terminal_sha256 FROM gov_terminal_dataset").fetchone()[0] == "old"


def test_terminal_api_requires_token_and_lists_campaign(tmp_path):
    client = _client(tmp_path)
    terminal = _terminal()
    assert client.post("/api/v2/terminals", json=terminal).status_code == 401
    response = client.post(
        "/api/v2/terminals", json=terminal,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 201
    inventory = client.get(
        "/api/v2/terminals", query_string={"campaign_sha256": "a" * 64},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert inventory.status_code == 200
    assert inventory.get_json()["terminals"][0]["unit_id"] == "u001"
