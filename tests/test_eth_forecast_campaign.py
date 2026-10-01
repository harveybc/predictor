"""Campaign tool: cell naming, control sizing, declaration, enqueue of modular and control cells (no TF)."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eth_forecast_campaign as fc  # noqa: E402
from tools import eth_forecast_dataset as ds  # noqa: E402
from tools import modular_search_space as ss  # noqa: E402
from tests.test_eth_forecast_dataset import FEATURES, SPLIT, make_view  # noqa: E402


def test_parse_cell_and_flat():
    assert fc.parse_cell("per_feature_mae_adam") == ("per_feature", "mae", "adam")
    assert fc.parse_cell("grouped32_huber_adamw") == ("grouped32", "huber", "adamw")
    assert fc.parse_cell("control_mlp_mae_adamw") == ("control_mlp", "mae", "adamw")
    with pytest.raises(ValueError):
        fc.parse_cell("grouped32_mse_adam")
    flat = fc.cell_flat("grouped32", "huber", "adam")
    assert flat["branch.grouping_size"] == 32 and flat["train.huber_delta"] == 1.0 and flat["train.weight_decay"] == 0.0
    assert "train.huber_delta" not in fc.cell_flat("grouped32", "mae", "adamw")
    assert fc.parse_size("6580M") == 6580 * 1024 ** 2 and fc.parse_size("2G") == 2 * 1024 ** 3


def test_control_sizing_matches_parameter_target():
    from tools.eth_control_evaluator import control_parameters
    hidden, count = fc.control_hidden_for(90726, 24, 83, 6, 1)
    assert hidden[0] == hidden[1] and control_parameters(24, 83, hidden, 6, 1) == count
    assert abs(count - 90726) / 90726 < 0.05


@pytest.fixture
def campaign(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view)
    data = tmp_path / "npz"
    ds.build(view, data, features=FEATURES, window=24, horizons=fc.HORIZONS, split=SPLIT, expected_sha=None)
    args = type("A", (), {})()
    args.data_manifest = str(data / "MANIFEST.json"); args.data_dir = str(data); args.root = str(tmp_path / "camp")
    args.campaign_id = "t"; args.predictor_checkout = "/x"; args.predictor_python = "/p"; args.predictor_revision = "a" * 40
    args.crispdm_run = "/c"; args.gpu_uuid = "GPU-x"; args.ld_library_path = "/l"; args.host_role = "worker_b"
    fc.declare(args)
    return Path(args.root), data


def test_declare_freezes_and_enqueue_modular_and_control(campaign, capsys):
    root, data = campaign
    decl = json.loads((root / "CAMPAIGN.json").read_text())
    assert decl["label"] == "DEVELOPMENT" and decl["freeze"]["rows"]["validation"] > 0
    assert decl["base"]["target_feature_indices"] == [FEATURES.index("log_return_1")]
    assert decl["resources"]["train"]["cap"] is None
    # modular cells x paired seeds
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam,per_feature_huber_adamw", "seeds": None})()
    fc.enqueue(args)
    db = sqlite3.connect(root / "queue.sqlite")
    rows = db.execute("select label, seed, status from candidates order by position").fetchall()
    assert rows == [("per_feature_mae_adam", 2021, "queued"), ("per_feature_mae_adam", 2022, "queued"),
                    ("per_feature_huber_adamw", 2021, "queued"), ("per_feature_huber_adamw", 2022, "queued")]
    # the control needs its amended hidden widths first
    args = type("A", (), {"root": str(root), "cells": "control_mlp_mae_adamw", "seeds": None})()
    with pytest.raises(ValueError, match="not amended"):
        fc.enqueue(args)
    patch = root / "patch.json"
    patch.write_text(json.dumps({"control": {"hidden": [16, 16], "parameter_target": 1000},
                                 "resources": {"train": {"cap": "2G"}, "verify": {"cap": "1G"}}}))
    fc.amend(type("A", (), {"root": str(root), "patch": str(patch), "change": "test"})())
    fc.enqueue(args)
    rows = db.execute("select label, seed, status, config_id from candidates where label like 'control%'").fetchall()
    assert len(rows) == 2 and rows[0][3] == rows[1][3] and rows[0][1] != rows[1][1]
    nested = json.loads(db.execute("select nested from candidates where label like 'control%'").fetchone()[0])
    assert nested["control"]["hidden"] == [16, 16] and nested["evaluator"]["loss"] == "mae"
    assert nested["evaluator"]["weight_decay"] == 0.0001 and "huber_delta" not in nested["evaluator"]
    # idempotent
    fc.enqueue(args)
    assert db.execute("select count(*) from candidates").fetchone()[0] == 6
    # caps are never lowered
    patch.write_text(json.dumps({"resources": {"train": {"cap": "1G"}}}))
    with pytest.raises(ValueError, match="never lowered"):
        fc.amend(type("A", (), {"root": str(root), "patch": str(patch), "change": "lower"})())
    # extra seeds for a configuration
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2023,2024"})()
    fc.enqueue(args)
    seeds = sorted(r[0] for r in db.execute("select seed from candidates where label='per_feature_mae_adam'"))
    assert seeds == [2021, 2022, 2023, 2024]
    status = fc.mdc.Campaign(root).status()
    assert status["counts"] == {"queued": 8}


def test_executor_refuses_without_measured_caps(campaign):
    root, _ = campaign
    decl = json.loads((root / "CAMPAIGN.json").read_text())
    with pytest.raises(RuntimeError, match="not amended from a measured pilot"):
        fc.LocalCrispdmExecutor(decl, "worker_b")


def test_residual_cells_inject_cumulative_seasonal_residual(campaign):
    root, _ = campaign
    assert fc.parse_cell("per_feature_mae_adamw_sres") == ("per_feature", "mae", "adamw")
    fc.enqueue(type("A", (), {"root": str(root), "cells": "per_feature_mae_adamw,per_feature_mae_adamw_sres", "seeds": None})())
    db = sqlite3.connect(root / "queue.sqlite")
    rows = db.execute("select label, seed, nested, config_id from candidates order by position").fetchall()
    assert [r[0] for r in rows] == ["per_feature_mae_adamw"] * 2 + ["per_feature_mae_adamw_sres"] * 2
    plain, res = json.loads(rows[0][2]), json.loads(rows[2][2])
    assert "target_residual" not in plain["model"]
    assert res["model"]["target_residual"] == {"kind": "seasonal_naive_cumulative", "period": 6,
                                               "target_features": ["log_return_1"]}
    assert rows[0][3] != rows[2][3] and rows[2][3] == rows[3][3]
    # the residual variant differs from the plain cell only in the residual key and its variant tag
    res["model"].pop("target_residual"); res["modular_candidate"].pop("f2_variant")
    assert res == plain
