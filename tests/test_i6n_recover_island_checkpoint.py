import json

from tools import i6n_recover_island_checkpoint as recovery


def test_runtime_config_matches_doin_predictor_wrapper_overlay():
    config = recovery._predictor_runtime_config({
        "i6n_island_mode": True,
        "i6n_horizon": 4,
        "quiet": False,
        "predictor_root": "/not/scientific",
    })

    assert config["i6n_island_mode"] is True
    assert config["i6n_horizon"] == 4
    assert config["quiet"] is False
    assert config["disable_postfit_uncertainty"] is True
    assert "predictor_root" not in config


def test_patch_node_config_sets_block_threshold_and_historical_domain(tmp_path):
    path = tmp_path / "node.json"
    path.write_text(json.dumps({
        "domains": [{
            "domain_id": "i6n-island-sequence-h4",
            "optimization_config": {"i6n_island_mode": True},
        }],
    }))

    report = recovery.patch_node_config(path)
    patched = json.loads(path.read_text())

    assert patched["initial_threshold"] == 1e-6
    assert patched["domains"][0]["has_synthetic_data"] is False
    assert patched["domains"][0]["synthetic_data_validation"] is False
    assert report["backup"].endswith("node.json.pre-chain-recovery")
    assert json.loads((tmp_path / "node.json.pre-chain-recovery").read_text()) == {
        "domains": [{
            "domain_id": "i6n-island-sequence-h4",
            "optimization_config": {"i6n_island_mode": True},
        }],
    }


def test_result_rebuilds_the_optimizer_cache_shape():
    payload = {
        "fitness": -0.1,
        "effective_config": {"horizon": 4},
        "genotype_canonical_sha256": "a" * 64,
        "effective_config_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "contract_sha256": "d" * 64,
        "checkpoint_identity_sha256": "e" * 64,
        "aggregation": "unweighted mean over the 4 folds",
        "aggregate_metrics": {
            "train_mae": 1.0,
            "train_naive_mae": 1.1,
            "inner_mae": 2.0,
            "inner_naive_mae": 2.2,
        },
        "per_fold": [
            {
                "fold": fold,
                "ratio_to_naive": 0.9,
                "receipt": {"status": "ok", "fold": fold},
                "origins_binding": "VERIFIED",
                "origins_sha256": {"train": fold, "inner": fold},
            }
            for fold in ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
        ],
    }

    result = recovery._result_from_payload(payload)

    assert result["fitness"] == -0.1
    assert result["val_mae"] == 2.0
    assert result["metrics"]["per_fold"] == {
        fold: 0.9 for fold in ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
    }
    assert set(result["metrics"]["receipts"]) == {
        "2023Q1", "2023Q2", "2023Q3", "2023Q4"
    }
