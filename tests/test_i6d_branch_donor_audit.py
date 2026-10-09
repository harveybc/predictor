import hashlib
import json
from pathlib import Path

import pytest

from tools import i6d_branch_donor_audit as A


FEATURE = "feature.a"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _design() -> dict:
    return {
        "design_sha256": "d" * 64,
        "config": {"feature_groups": [{"name": FEATURE}]},
        "branch_only_contract": {
            "arms": {
                "CONV": {
                    "plugin": "causal_conv1d",
                    "params": {"channels": 16, "kernel_size": 3},
                }
            },
            "shared": {"branch_time_grid": list(range(1, 25))},
        },
    }


def _result(weights: Path, *, output_shape=(6, 8), digest=None) -> dict:
    digest = digest or hashlib.sha256(weights.read_bytes()).hexdigest()
    return {
        "status": "COMPLETE",
        "feature_id": FEATURE,
        "fold_id": "inner_2023",
        "arm": "TRAINED_ENCODER",
        "architecture": {
            "id": "fs4_causal_conv_24_12_6_v1",
            "latent_shape": list(output_shape),
            "target_input": False,
        },
        "hyper": {"window": 24, "kernel_size": 3, "filters": 16},
        "artifacts": {"chosen_weights_file_sha256": digest},
    }


def test_fs4_encoder_is_retained_as_evidence_but_not_mislabelled_as_branch_donor(tmp_path):
    task = tmp_path / "task"
    weights = task / "chosen.weights.h5"
    weights.parent.mkdir()
    weights.write_bytes(b"retained weights")
    _write_json(task / "result.json", _result(weights))

    report = A.audit_branch_donors(_design(), tmp_path, expected_folds=("inner_2023",))

    assert report["summary"] == {
        "features": 1,
        "expected_feature_folds": 1,
        "verified_extractibility_artifacts": 1,
        "compatible_branch_donors": 0,
        "missing_feature_folds": 0,
    }
    cell = report["features"][0]["folds"][0]
    assert cell["state"] == "EVIDENCE_ONLY_INCOMPATIBLE"
    assert set(cell["reasons"]) >= {
        "OUTPUT_TEMPORAL_GRID_MISMATCH",
        "NOT_A_SEALED_MODULAR_DONOR",
    }


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("digest", "WEIGHTS_DIGEST_MISMATCH"),
        ("missing", "WEIGHTS_MISSING"),
    ],
)
def test_corrupt_or_missing_weights_are_not_counted_as_extractibility_evidence(
    tmp_path, mutation, reason
):
    task = tmp_path / "task"
    weights = task / "chosen.weights.h5"
    weights.parent.mkdir()
    weights.write_bytes(b"retained weights")
    result = _result(weights, digest="0" * 64 if mutation == "digest" else None)
    _write_json(task / "result.json", result)
    if mutation == "missing":
        weights.unlink()

    report = A.audit_branch_donors(_design(), tmp_path, expected_folds=("inner_2023",))

    cell = report["features"][0]["folds"][0]
    assert cell["state"] == "INVALID_ARTIFACT"
    assert reason in cell["reasons"]
    assert report["summary"]["verified_extractibility_artifacts"] == 0


def test_absent_fold_is_named_and_suppresses_readiness(tmp_path):
    report = A.audit_branch_donors(_design(), tmp_path, expected_folds=("inner_2023",))

    assert report["features"][0]["folds"] == [
        {
            "fold_id": "inner_2023",
            "state": "MISSING",
            "reasons": ["TRAINED_ENCODER_RESULT_MISSING"],
        }
    ]
    assert report["summary"]["missing_feature_folds"] == 1


def test_duplicate_complete_result_is_rejected_instead_of_chosen_by_path_order(tmp_path):
    for name in ("a", "b"):
        task = tmp_path / name
        weights = task / "chosen.weights.h5"
        weights.parent.mkdir()
        weights.write_bytes(name.encode())
        _write_json(task / "result.json", _result(weights))

    with pytest.raises(ValueError, match="DUPLICATE_TRAINED_ENCODER_RESULT"):
        A.audit_branch_donors(_design(), tmp_path, expected_folds=("inner_2023",))
