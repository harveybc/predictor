"""Deployment config contract for the four-node I6N island sequence."""

from __future__ import annotations

import json

import pytest

from tools.i6n_doin_sequence_config import generate


def _source():
    return {
        "domains": [
            {
                "optimization_config": {
                    "i6n_horizon": horizon,
                    "i6n_folds": {"fold": {}},
                    "i6n_admitted_classes": ["S1", "S2", "S3"],
                }
            }
            for horizon in (4, 72)
        ]
    }


def _nodes():
    return [
        {
            "island_id": f"node-{index}",
            "endpoint": f"node-{index}.invalid:{18570 + index}",
            "candidate_archive_store_url": "http://127.0.0.1:15061",
        }
        for index in range(4)
    ]


def test_generated_nodes_use_hash_only_candidate_archival(tmp_path):
    generate(_source(), _nodes(), tmp_path, "/predictor", "/python")

    config = json.loads(
        (tmp_path / "h4" / "node-0" / "doin_node.json").read_text()
    )
    assert config["candidate_archive_mode"] == "external_payload"
    assert config["candidate_archive_store_id"] == "doin_archive_cas"
    assert config["candidate_archive_store_url"] == "http://127.0.0.1:15061"
    assert config["candidate_archive_token_env"] == "DATA_GOV_LAKE_TOKEN"
    assert config["component_version_scope"] == [
        "doin-node", "doin-core", "doin-plugins", "predictor"
    ]
    assert config["archive_reference_spool"].endswith("/h4/node-0")
    assert config["domains"][0]["optimization_config"]["min_relative_improvement"] == 1e-8


def test_missing_archive_route_is_rejected_before_writing_configs(tmp_path):
    nodes = _nodes()
    nodes[2].pop("candidate_archive_store_url")

    with pytest.raises(ValueError, match="candidate_archive_store_url"):
        generate(_source(), nodes, tmp_path, "/predictor", "/python")

    assert not tmp_path.exists() or not any(tmp_path.rglob("doin_node.json"))
