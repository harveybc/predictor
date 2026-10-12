"""Real tiny CPU fit on invented data: dx=36, dy=24, h=72 (lane A).

Three heads, a missing intermediate bar, endpoint support frozen by the
endpoint-only base. Not historical, clock or cost evidence.
"""

import json

import numpy as np

from tests._i6n_trajectory_fixture import build, genome, write_json
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_search as search


def test_dx36_dy24_h72_trajectory_fit(tmp_path):
    base = build(tmp_path, horizon=72, samples=3, spacing=36, drop={(10, 40)})
    path = tmp_path / "base.json"
    write_json(path, base)
    # A weekly-refit unit: the outer week is sealed now, scored after the fit only.
    binding = evaluator.freeze(path, tmp_path / "bridge", experiment="unit-exp", fold="W2024-01-07",
                               score_outer=True)
    frozen = json.loads(evaluator.Path(binding["i6n_evaluation_context"]["frozen_support"]).read_text())
    assert frozen["outer"]["count"] > 0 and frozen["outer"]["split_id"] == "DEVELOPMENT"
    effective = search.effective_config(genome(search, 72, dx=36, dy=24, core=base["model"]))
    assert effective["output_offsets"] == [24, 48, 72] and effective["samples"] == 3
    receipt = evaluator.evaluate(dict(config=effective, generation=1,
                                      context=binding["i6n_evaluation_context"],
                                      support=binding["i6n_support"]))
    for key in ("counts", "naive_sums", "support_digest"):
        assert receipt[key] == binding["i6n_support"][key]
    cr = receipt["metadata"]["candidate_receipt"]
    assert (cr["delta_x_hours"], cr["delta_y_hours"], cr["history_hours"], cr["samples"]) == (36, 24, 72, 3)
    assert [h["offset_hours"] for h in cr["heads"]] == [24, 48, 72]
    assert [h["head_role"] for h in cr["heads"]] == ["INTERMEDIATE", "INTERMEDIATE", "ENDPOINT"]
    assert all(h["status"] == "TRAINED" and h["train_target_scale"] > 0 for h in cr["heads"])
    assert len(cr["rows"]) == 9 and cr["fold"] == "W2024-01-07"
    # Fitness is TRAIN/inner endpoint only; the outer week is a diagnostic.
    assert search._score(receipt, binding["i6n_support"])[0] == receipt["fitness"]
    assert set(receipt["error_sums"]) == {"train", "inner"}
    outer = [r for r in cr["rows"] if r["split"] == "VALIDATION"]
    end = next(r for r in outer if r["head_role"] == "ENDPOINT")
    assert end["n"] == frozen["outer"]["count"] and end["naive_sum_abs_error"] == frozen["outer"]["naive_sum"]
    assert end["comparison"] is True and receipt["metadata"]["outer_metrics"]["n"] == end["n"]
    assert all(r["status"] == "SCORED" and len(r["support_sha256"]) == 64 for r in cr["rows"])
    root = evaluator.Path(receipt["metadata"]["artifact_directory"])
    with np.load(root / "scores.npz") as rows:
        endpoint = set(rows["train_origins"].tolist())
        h24 = set(rows["train_h24_origins"].tolist())
        assert len(endpoint) == receipt["counts"]["train"]
        # The missing bar removes one intermediate label, never the endpoint row.
        assert endpoint - h24 and h24 - endpoint
        for split in ("train", "inner"):
            for o in (24, 48):
                assert np.isfinite(rows[f"{split}_h{o}_prediction"]).all()
    training = receipt["metadata"]["training"]
    assert training["counts"]["train"][2] == receipt["counts"]["train"]
    assert all(np.isfinite(r["monitored_loss"]) for r in training["history"])
    assert cr["final_steps"] == effective["final_steps"]
