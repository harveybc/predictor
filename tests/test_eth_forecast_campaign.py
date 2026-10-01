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
    # Screening uses one frozen seed per configuration.
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam,per_feature_huber_adamw", "seeds": None})()
    fc.enqueue(args)
    db = sqlite3.connect(root / "queue.sqlite")
    rows = db.execute("select label, seed, status from candidates order by position").fetchall()
    assert rows == [("per_feature_mae_adam", 2021, "queued"),
                    ("per_feature_huber_adamw", 2021, "queued")]
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
    assert len(rows) == 1
    nested = json.loads(db.execute("select nested from candidates where label like 'control%'").fetchone()[0])
    assert nested["control"]["hidden"] == [16, 16] and nested["evaluator"]["loss"] == "mae"
    assert nested["evaluator"]["weight_decay"] == 0.0001 and "huber_delta" not in nested["evaluator"]
    # idempotent
    fc.enqueue(args)
    assert db.execute("select count(*) from candidates").fetchone()[0] == 3
    # caps are never lowered
    patch.write_text(json.dumps({"resources": {"train": {"cap": "1G"}}}))
    with pytest.raises(ValueError, match="never lowered"):
        fc.amend(type("A", (), {"root": str(root), "patch": str(patch), "change": "lower"})())
    # Extra replicas require a predeclared reason and are capped at three total.
    no_reason = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2022",
                               "replica_justification": None})()
    with pytest.raises(ValueError, match="justification"):
        fc.enqueue(no_reason)
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2022,2023",
                          "replica_justification": "finalists tied within the observed seed spread"})()
    fc.enqueue(args)
    seeds = sorted(r[0] for r in db.execute("select seed from candidates where label='per_feature_mae_adam'"))
    assert seeds == [2021, 2022, 2023]
    request = db.execute("select label, seeds_json, justification from replica_requests").fetchone()
    assert request == ("per_feature_mae_adam", "[2022, 2023]", "finalists tied within the observed seed spread")
    fourth = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2024",
                            "replica_justification": "try another"})()
    with pytest.raises(ValueError, match="maximum three"):
        fc.enqueue(fourth)
    status = fc.mdc.Campaign(root).status()
    assert status["counts"] == {"queued": 5}


def test_executor_refuses_without_measured_caps(campaign):
    root, _ = campaign
    decl = json.loads((root / "CAMPAIGN.json").read_text())
    with pytest.raises(RuntimeError, match="not amended from a measured pilot"):
        fc.LocalCrispdmExecutor(decl, "worker_b")


def test_legacy_default_is_one_seed_and_implicit_replica_needs_reason(campaign):
    root, _ = campaign
    p = root / "CAMPAIGN.json"
    decl = json.loads(p.read_text())
    decl["paired_seeds"] = [2021, 2022]
    p.write_text(json.dumps(decl))
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": None})()
    fc.enqueue(args)
    with sqlite3.connect(root / "queue.sqlite") as db:
        assert db.execute("SELECT seed FROM candidates").fetchall() == [(2021,)]
    args.cells = "per_feature_huber_adam"
    args.seeds = "2022"
    fc.enqueue(args)
    args.seeds = None
    with pytest.raises(ValueError, match="justification"):
        fc.enqueue(args)


def test_failed_multi_cell_request_is_atomic(campaign):
    root, _ = campaign
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": None})()
    fc.enqueue(args)
    args.cells = "per_feature_mae_adam,control_mlp_mae_adam"
    args.seeds = "2022"
    args.replica_justification = "measure finalist variability"
    with pytest.raises(ValueError, match="not amended"):
        fc.enqueue(args)
    with sqlite3.connect(root / "queue.sqlite") as db:
        assert db.execute("SELECT seed FROM candidates").fetchall() == [(2021,)]


def test_justification_write_failure_cannot_leave_replica(campaign):
    root, _ = campaign
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": None})()
    fc.enqueue(args)
    with sqlite3.connect(root / "queue.sqlite") as db:
        db.execute("CREATE TABLE replica_requests(id INTEGER PRIMARY KEY, requested_at TEXT, label TEXT, "
                   "seeds_json TEXT, justification TEXT)")
        db.execute("CREATE TRIGGER deny_request BEFORE INSERT ON replica_requests "
                   "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    args.seeds = "2022"
    args.replica_justification = "measure finalist variability"
    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        fc.enqueue(args)
    with sqlite3.connect(root / "queue.sqlite") as db:
        assert db.execute("SELECT seed FROM candidates").fetchall() == [(2021,)]


def test_dispatch_guards_historical_queue_and_allows_explicit_authorization(campaign):
    root, _ = campaign
    p = root / "CAMPAIGN.json"
    decl = json.loads(p.read_text())
    decl["paired_seeds"] = [2021, 2022]
    decl.pop("replication_policy")
    p.write_text(json.dumps(decl))
    legacy = fc.mdc.Campaign(root)
    legacy.enqueue(fc.cell_flat("per_feature", "mae", "adam", 4), "per_feature_mae_adam")
    legacy.db.close()
    runner = fc.F2Campaign(root)
    first = runner.claim()
    assert first[0]["seed"] == 2021
    assert runner.claim() is None  # pending legacy replica is not a new authorization
    pending = runner.db.execute("SELECT status, blocked_reason FROM candidates WHERE seed=2022").fetchone()
    assert pending[0] == "queued" and "justification" in pending[1]
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2022",
                          "replica_justification": "compare finalist seed variability"})()
    fc.enqueue(args)  # records authorization even though candidate already exists
    second = runner.claim()
    assert second[0]["seed"] == 2022
    assert runner.claim() is None
    assert runner.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
    runner.db.execute("UPDATE candidates SET status='completed' WHERE seed=2022")
    runner.db.execute("UPDATE attempts SET status='completed' WHERE cid=?", (second[0]["cid"],))
    assert runner.claim()[1] == "verify"
    runner.db.close()


def test_dispatch_requires_reason_even_for_screening_after_other_replicas(campaign):
    root, _ = campaign
    legacy = fc.mdc.Campaign(root)
    flat = fc.cell_flat("per_feature", "mae", "adam", 4)
    legacy.enqueue(flat, "per_feature_mae_adam", seeds=[2021, 2022, 2023])
    # Historical seeds need not belong to the new allowed set.
    legacy.db.execute("UPDATE candidates SET cid=?, seed=2024, status='verified' WHERE seed=2023", ("h" * 64,))
    legacy.db.execute("UPDATE candidates SET status='verified' WHERE seed=2022")
    legacy.enqueue(flat, "per_feature_mae_adam", seeds=[2023])
    legacy.db.close()
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2023",
                          "replica_justification": "finalist"})()
    # Authorizations cannot waive the three-replica cap.
    with pytest.raises(ValueError, match="maximum three"):
        fc.enqueue(args)
    runner = fc.F2Campaign(root)
    assert runner.claim() is None  # even the screening seed is now an additional replica
    runner.db.close()


def test_dispatch_maximum_counts_historical_seeds_even_with_authorization(campaign):
    root, _ = campaign
    args = type("A", (), {"root": str(root), "cells": "per_feature_mae_adam", "seeds": "2021,2022,2023",
                          "replica_justification": "compare finalist variability"})()
    fc.enqueue(args)
    runner = fc.F2Campaign(root)
    assert [runner.claim()[0]["seed"] for _ in range(3)] == [2021, 2022, 2023]
    assert runner.claim() is None
    # Three historical replicas already ran; no authorization can admit another.
    runner.db.execute("UPDATE candidates SET status='verified'")
    runner.db.execute("UPDATE candidates SET seed=2024 WHERE seed=2021")
    row = dict(runner.db.execute("SELECT * FROM candidates WHERE seed=2022").fetchone())
    row.update(cid="z" * 64, position=99, seed=2021, status="queued")
    columns = list(row)
    runner.db.execute("INSERT INTO candidates (" + ",".join(columns) + ") VALUES (" +
                      ",".join("?" for _ in columns) + ")", [row[k] for k in columns])
    runner.db.execute("INSERT INTO replica_requests(requested_at,label,seeds_json,justification) VALUES(?,?,?,?)",
                      (fc.mdc.now(), args.cells, "[2021]", "historical authorization"))
    assert runner.claim() is None
    reason = runner.db.execute("SELECT blocked_reason FROM candidates WHERE seed=2021").fetchone()[0]
    assert "maximum three" in reason
    assert runner.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 3
    runner.db.close()


def test_residual_cells_inject_cumulative_seasonal_residual(campaign):
    root, _ = campaign
    assert fc.parse_cell("per_feature_mae_adamw_sres") == ("per_feature", "mae", "adamw")
    fc.enqueue(type("A", (), {"root": str(root), "cells": "per_feature_mae_adamw,per_feature_mae_adamw_sres", "seeds": None})())
    db = sqlite3.connect(root / "queue.sqlite")
    rows = db.execute("select label, seed, nested, config_id from candidates order by position").fetchall()
    assert [r[0] for r in rows] == ["per_feature_mae_adamw", "per_feature_mae_adamw_sres"]
    plain, res = json.loads(rows[0][2]), json.loads(rows[1][2])
    assert "target_residual" not in plain["model"]
    assert res["model"]["target_residual"] == {"kind": "seasonal_naive_cumulative", "period": 6,
                                               "target_features": ["log_return_1"]}
    assert rows[0][3] != rows[1][3]
    # the residual variant differs from the plain cell only in the residual key and its variant tag
    res["model"].pop("target_residual"); res["modular_candidate"].pop("f2_variant")
    assert res == plain


def test_regime_suffix_and_donor_declaration(tmp_path):
    assert fc.regime_of("per_feature_mae_adamw_r1") == "R1" and fc.regime_of("per_feature_mae_adamw") == "R0"
    assert fc.parse_cell("per_feature_mae_adamw_r2") == ("per_feature", "mae", "adamw")
    flat = fc.cell_flat("per_feature", "mae", "adamw", 4, "R1")
    assert flat["branch.regime"] == flat["core.regime"] == "R1"
    d = tmp_path / "pretrain"
    d.mkdir()
    names = ["a", "b"]
    for n in ("branch_0", "branch_1", "core"):
        for ext in (".keras", ".manifest.json", ".provenance.json"):
            (d / (n + ext)).write_bytes(n.encode() + ext.encode())
    (d / "PRETRAIN.json").write_text("{}")
    donors, binding = fc.donor_declaration(d, names)
    assert set(donors) == {"1:branch_0", "1:branch_1", "core:1"} and binding["required_contract"] == "OPERATIONAL"
    assert set(binding["donors"][donors["core:1"]]) == {"keras", "manifest", "provenance"}


def test_materialize_regime_cell_from_declared_donors(campaign, tmp_path):
    root, data = campaign
    import contextlib, io, sys as _s
    d = tmp_path / "pre"
    d.mkdir()
    for n in ("branch_0", "branch_1", "branch_2", "branch_3", "core"):
        for ext in (".keras", ".manifest.json", ".provenance.json"):
            (d / (n + ext)).write_bytes(n.encode() + ext.encode())
    (d / "PRETRAIN.json").write_text("{}")
    decl = json.loads((root / "CAMPAIGN.json").read_text())
    donors, _ = fc.donor_declaration(d, decl["base"]["feature_names"])
    decl["base"]["donors"] = donors
    nested = fc.ss.from_flat({**fc.cell_flat("per_feature", "mae", "adamw", 4, "R1"), "train.seed": 2021}, decl["base"], decl["search_space"])
    assert nested["model"]["core"]["regime"] == "R1" and nested["model"]["core"]["donor"].endswith("core.keras")
    assert all(b["regime"] == "R1" and b["donor"] for b in nested["model"]["branches"])
