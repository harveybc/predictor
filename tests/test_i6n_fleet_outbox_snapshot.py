from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tests.test_i6n_outbox_snapshot import _source
from tools.i6n_fleet_outbox_snapshot import FleetSpecError, collect, load_spec


def _spec(tmp_path: Path) -> Path:
    nodes = []
    for index in range(4):
        island = f"island-{index}"
        publisher, terminal = _source(tmp_path, island, f"peer-{index}")
        remote = tmp_path / "remote" / island
        nodes.append({
            "island_id": island, "host": None,
            "source": str(publisher.outbox.path),
            "terminal_receipt": str(terminal),
            "remote_terminal": str(remote / "terminal.json"),
            "remote_snapshot": str(remote / "outbox.sqlite"),
            "remote_seal": str(remote / "SEAL.json"),
        })
    document = {
        "schema": "i6n.fleet_outbox_snapshot.spec.v1",
        "python": str(Path(sys.executable).resolve()),
        "working_directory": str(Path.cwd().resolve()),
        "collected_root": str((tmp_path / "collected").resolve()),
        "nodes": nodes,
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document))
    return path


def test_four_local_island_snapshots_close_without_an_agent(tmp_path):
    result = collect(load_spec(_spec(tmp_path)))

    assert result["state"] == "COMPLETE"
    assert set(result["completed"]) == {f"island-{index}" for index in range(4)}
    assert result["waiting"] == {}
    assert len({row["items_sha256"] for row in result["completed"].values()}) == 4


def test_spec_rejects_duplicate_island_ids(tmp_path):
    path = _spec(tmp_path)
    document = json.loads(path.read_text())
    document["nodes"][1]["island_id"] = document["nodes"][0]["island_id"]
    path.write_text(json.dumps(document))

    with pytest.raises(FleetSpecError, match="island ids must be distinct"):
        load_spec(path)
