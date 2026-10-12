"""Real tiny CPU fits on invented data: dx=3, dy=6, h=12 (lane A).

Endpoint-only and trajectory candidates of the same fold must share the
endpoint support, rows and naive exactly; intermediates are diagnostics.
Not historical, clock or cost evidence.
"""

import json

import numpy as np
import pytest

from tests._i6n_trajectory_fixture import build, genome, write_json
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_search as search
from tools.i6n_contracts import paired_metrics as c2


@pytest.fixture
def frozen(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3, drop={(10, 50), (207, 30)})
    path = tmp_path / "base.json"
    write_json(path, base)
    binding = evaluator.freeze(path, tmp_path / "bridge", experiment="unit-exp", fold="2023Q4")

    def request(**kw):
        effective = search.effective_config(genome(search, 12, dx=3, core=base["model"], **kw))
        return dict(config=effective, generation=0, context=binding["i6n_evaluation_context"],
                    support=binding["i6n_support"])
    return base, binding, request


def test_endpoint_and_trajectory_share_endpoint_rows_and_naive(frozen, monkeypatch):
    _, binding, request = frozen
    endpoint_request, trajectory_request = request(), request(dy=6)
    endpoint = evaluator.evaluate(endpoint_request)
    trajectory = evaluator.evaluate(trajectory_request)
    for key in ("counts", "naive_sums", "support_digest"):
        assert endpoint[key] == trajectory[key] == binding["i6n_support"][key]
    for receipt in (endpoint, trajectory):
        assert search._score(receipt, binding["i6n_support"])[0] == receipt["fitness"]
        assert receipt["metadata"]["checkpoint_reload_verified"]
    roots = [evaluator.Path(r["metadata"]["artifact_directory"]) for r in (endpoint, trajectory)]
    assert roots[0] != roots[1]
    with np.load(roots[0] / "scores.npz") as a, np.load(roots[1] / "scores.npz") as b:
        for split in ("train", "inner"):
            for key in ("origins", "current", "future"):
                np.testing.assert_array_equal(a[f"{split}_{key}"], b[f"{split}_{key}"])
        assert not any(k.startswith("train_h") for k in a.files)
        assert {"train_h6_prediction", "inner_h6_future"} <= set(b.files)

    ep, tr = endpoint["metadata"]["candidate_receipt"], trajectory["metadata"]["candidate_receipt"]
    assert ep["schema_version"] == tr["schema_version"] == "i6n.candidate_receipt.v1"
    assert (ep["experiment"], ep["fold"]) == ("unit-exp", "2023Q4")
    assert ep["candidate_identity_sha256"] == roots[0].name
    assert ep["genotype_canonical_sha256"] != tr["genotype_canonical_sha256"]
    assert (tr["horizon_hours"], tr["delta_x_hours"], tr["delta_y_hours"], tr["history_hours"],
            tr["samples"], tr["output_mode"]) == (12, 3, 6, 24, 9, "trajectory")
    assert ep["delta_y_hours"] is None and ep["output_mode"] == "endpoint"
    assert [(h["offset_hours"], h["head_role"], h["status"], h["loss_weight"]) for h in tr["heads"]] == [
        (6, "INTERMEDIATE", "TRAINED", 1.), (12, "ENDPOINT", "TRAINED", 1.)]
    assert tr["seed"] == 6102026 and tr["stopping"]["plan_conformant"] is True
    assert tr["stopping"]["patience"] == 5 and tr["stopping"]["max_epochs"] == 50
    assert tr["cost"]["parameters"] > 0 and tr["cost"]["process_peak_rss_bytes"] > 0
    assert tr["attempt"] == {"index": 1, "parent_status_sha256": None, "fencing_token": None, "parent": None}
    assert tr["placement"]["soft_device_placement"] is False

    keys = ("schema_version", "experiment", "candidate_identity_sha256", "fold", "horizon_hours",
            "head_offset_hours", "role", "population", "split", "metric", "unit")
    assert len({tuple(r[k] for k in keys) for r in tr["rows"]}) == len(tr["rows"]) == 4
    for split, name in (("TRAIN", "train"), ("INNER", "inner")):
        end = next(r for r in tr["rows"] if r["split"] == split and r["head_offset_hours"] == 12)
        mid = next(r for r in tr["rows"] if r["split"] == split and r["head_offset_hours"] == 6)
        assert end["comparison"] is True and mid["comparison"] is False
        assert end["n"] == trajectory["counts"][name]
        assert end["sum_abs_error"] == trajectory["error_sums"][name]
        assert end["naive_sum_abs_error"] == trajectory["naive_sums"][name]
        assert mid["status"] == "SCORED" and mid["n"] > 0
        assert mid["naive_sum_abs_error"] != end["naive_sum_abs_error"]
        for r in (end, mid):
            assert r["population"] == "S" and r["role"] == "MANAGEMENT"
            assert r["c2_role"] == "management_origin" and len(r["support_sha256"]) == 64
            assert r["unit"] == "PRICE" and r["metric"] == "ABS_AND_SQUARED_ERROR_SUMS"
            assert all(np.isfinite(r[k]) for k in ("sum_abs_error", "sum_sq_error",
                                                    "naive_sum_abs_error", "naive_sum_sq_error"))
        native = json.loads((roots[1] / f"{name}.h6.c2.json").read_text())
        c2.verify_result(native)
        assert evaluator.cell.file_hash(roots[1] / f"{name}.h6.c2.json") == mid["c2_receipt_sha256"]
        with np.load(roots[1] / "scores.npz") as b:
            pred, fut, cur = (b[f"{name}_h6_{k}"] for k in ("prediction", "future", "current"))
            assert mid["sum_abs_error"] == pytest.approx(np.abs(pred - fut).sum())
            assert mid["naive_sum_abs_error"] == pytest.approx(np.abs(cur - fut).sum())
            assert len(pred) == mid["n"]

    monkeypatch.setattr(evaluator, "fit_candidate", lambda *a: pytest.fail("refit"))
    trajectory_request["generation"] = 3
    assert evaluator.evaluate(trajectory_request) == trajectory
    adopted = dict(trajectory_request, attempt={"index": 2, "parent_status_sha256": "0" * 64,
                                                "fencing_token": "lease"})
    assert evaluator.evaluate(adopted) == trajectory
    h6 = roots[1] / "train.h6.c2.json"
    original = h6.read_bytes()
    h6.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="digest"):
        evaluator.evaluate(trajectory_request)
