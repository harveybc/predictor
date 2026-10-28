"""R3 scientific defaults, using synthetic receipts and real reporting, never fitting.

GPU labels below are receipt fixtures, not GPU execution or scientific evidence.
Only the existing sparse fixture's pre-2025 common DEVELOPMENT rows are scored.
"""
import copy
import json
import shutil

import numpy as np
import pytest

from tests.i6dmo.test_matched_fit_path import (
    HORIZONS, _sha, _spec, _stored_terminal, world,  # noqa: F401
)
from tests.i6dmo.test_ops_matched_runner import _metrics
from tools.i6dmo import model as M, reporting, train as T

MISSING = "missing"
STRICT_FLAGS = [MISSING, True, None, 0, ""]
NON_SCIENTIFIC = [(False, "cpu"), (True, "cpu"), (False, "gpu"), (None, None)]


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("R3 reporting tests must not build or train models")

    monkeypatch.setattr(M, "build_dmo", forbidden)
    monkeypatch.setattr(T, "train_unit", forbidden)


@pytest.fixture(scope="module")
def upstreams(world):
    """Digest-bound synthetic predictions and matched identities, with no fit."""
    paths = {}
    dev = world["pop"].splits["DEVELOPMENT"]
    for arm in ("mg1", "mg2m"):
        path = world["root"] / f"r3_receipt_{arm}"
        path.mkdir()
        (path / "predictions").mkdir()
        prices = reporting._arm_development(world["pop"], world["exps"][arm], world["input"])
        pred_path = path / "predictions" / "validation.npz"
        np.savez(pred_path, row_id=dev.origin_row_ids, horizons_hours=HORIZONS,
                 mask=dev.head_mask, origin_price=prices["origin_price"],
                 target_price=prices["target_price"], predicted_price=prices["target_price"])
        receipt = dict(status="COMPLETED", scientific=True, device="gpu",
                       identity=dict(common_population_sha256=world["digest"], matched_arm=arm),
                       matched=dict(common_population_sha256=world["digest"], arm=arm,
                                    capacity=dict(schema=M.CAPACITY_SCHEMA, arm=arm,
                                                  counts=M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]])),
                       artifacts={"predictions/validation.npz": _sha(pred_path)})
        (path / "RECEIPT.json").write_text(json.dumps(receipt))
        paths[arm] = path
    return paths


@pytest.fixture
def report_case(world, upstreams, tmp_path):
    def run(scientific, device, flag=MISSING, arm="mg1"):
        fit = tmp_path / "upstream" / "out" / "train"
        shutil.copytree(upstreams[arm], fit)
        receipt_path = fit / "RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(scientific=scientific, device=device)
        receipt_path.write_text(json.dumps(receipt))
        spec = _spec(world, arm, _stored_terminal(fit))
        spec["matched"].pop("require_scientific_fit", None)
        if flag != MISSING:
            spec["matched"]["require_scientific_fit"] = flag
        out = tmp_path / "report"
        result = reporting.run_unit(spec, out, tmp_path / "progress.json")
        metrics = json.loads((out / "METRICS.json").read_text()) if result["status"] == "COMPLETE" else None
        return result, metrics, out

    return run


@pytest.mark.parametrize("scientific,device", NON_SCIENTIFIC)
@pytest.mark.parametrize("flag", STRICT_FLAGS)
def test_report_requires_scientific_gpu_unless_explicit_false(report_case, scientific, device, flag):
    result, metrics, out = report_case(scientific, device, flag)
    assert result["status"] == "FAILED", result
    assert result["disposition"] == "NON_SCIENTIFIC_FIT", result
    assert result["artifacts"] == {} and result["reports"] == []
    assert metrics is None and not (out / "validation_predictions.npz").exists()


@pytest.mark.parametrize("scientific,device", NON_SCIENTIFIC)
def test_report_explicit_false_preserves_engineering_opt_out(report_case, scientific, device):
    result, metrics, _ = report_case(scientific, device, False)
    assert result["status"] == "COMPLETE", result
    assert metrics["matched"]["require_scientific_fit"] is False
    assert metrics["matched"]["scientific_fit"] is scientific
    assert metrics["matched"]["fit_device"] == device


@pytest.mark.parametrize("arm", ["mg1", "mg2m"])
def test_report_default_accepts_valid_scientific_gpu_receipt_and_records_strictness(report_case, arm):
    result, metrics, _ = report_case(True, "gpu", arm=arm)
    assert result["status"] == "COMPLETE", result
    assert metrics["matched"]["require_scientific_fit"] is True
    assert metrics["matched"]["scientific_fit"] is True
    assert metrics["matched"]["fit_device"] == "gpu"


def _pair():
    return [dict(m, matched=dict(m["matched"], scientific_fit=True, fit_device="gpu"))
            for m in (_metrics("mg1", -0.02), _metrics("mg2m", -0.01))]


@pytest.mark.parametrize("arm_index", [0, 1])
@pytest.mark.parametrize("scientific,device", NON_SCIENTIFIC)
@pytest.mark.parametrize("flag", STRICT_FLAGS)
def test_decision_requires_both_scientific_gpu_fits_unless_explicit_false(arm_index, scientific, device, flag):
    pair = _pair()
    pair[arm_index]["matched"].update(scientific_fit=scientific, fit_device=device)
    kwargs = {} if flag == MISSING else {"require_scientific": flag}
    with pytest.raises(reporting.ReportFailure, match="NON_SCIENTIFIC_FIT"):
        reporting.decide_matched(*pair, **kwargs)


def test_decision_default_rejects_missing_scientific_metadata():
    with pytest.raises(reporting.ReportFailure, match="NON_SCIENTIFIC_FIT"):
        reporting.decide_matched(_metrics("mg1", -0.02), _metrics("mg2m", -0.01))


def test_decision_explicit_false_preserves_engineering_opt_out():
    assert reporting.decide_matched(_metrics("mg1", -0.02), _metrics("mg2m", -0.01),
                                    require_scientific=False)["selected"] == "mg1"


def test_decision_default_accepts_valid_scientific_gpu_matches():
    pair = _pair()
    before = copy.deepcopy(pair)
    verdict = reporting.decide_matched(*pair)
    assert verdict["selected"] == "mg1" and verdict["decision"] == "LOWER_SCORE"
    assert verdict["capacity_delta"]["delta"]["heads"] == 2304
    assert pair == before
