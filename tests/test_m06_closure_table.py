"""Tests for tools/m06_closure_table.py: the closure table must refuse what it cannot prove."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import m06_closure_table as C  # noqa: E402

POP = "7" * 64


def design():
    return {"design_sha256": "d" * 64, "seeds": [2021, 2022, 2023],
            "lock": {"paper": "Paper X", "paper_read_at": "2026-09-28",
                     "published": {"table": "Table 8", "per_horizon": {"96": {"mse": 0.375, "mae": 0.251}}},
                     "agreement": {"std_paper": {"mse": 0.008, "mae": 0.004}, "k_agree": 2, "k_partial": 3,
                                   "rounding": 0.0005, "source": "Table 7", "rule": "2 x std + 0.0005"}}}


def record(seed, mse, mae, pop=POP, naive=(2.7, 1.07)):
    r = {"cell_id": f"traffic_L96_h96_s{seed}", "dataset": "traffic", "seq_len": 96, "horizon_steps": 96,
         "seed": seed, "design_sha256": "d" * 64,
         "metric": {"author_float32": {"mse": mse, "mae": mae}, "independent_float64": {"mse": mse, "mae": mae},
                    "space": "z_train", "reduction": "mean over all elements"},
         "naive": {"mse": naive[0], "mae": naive[1], "definition": "persistence",
                   "paired_on_the_same_rows_proved_by": {"target_population_sha256": pop,
                                                         "naive_target_sha256": pop, "equal": True}},
         "population": {"sha256": pop, "windows": 10, "target_channels": 3, "elements": 10 * 96 * 3,
                        "all_finite": True},
         "receipt": {"tags": {"comparison_class": "MATCHED_PUBLISHED_RECIPE_EXECUTED"}},
         "resources": {"wall_seconds": 100.0}, "checkpoint_sha256": "c" * 64}
    r["record_sha256"] = C.sha_obj(r)
    return r


def good():
    return [record(2021, 0.3757, 0.2514), record(2022, 0.3754, 0.2512), record(2023, 0.3745, 0.2508)]


def test_three_sealed_seeds_give_a_class_on_the_mean():
    t = C.build(design(), good(), 96)
    assert t["mean"]["mse"]["class"] == "OPERATIONAL_AGREEMENT"
    assert abs(t["mean"]["mse"]["mean"] - (0.3757 + 0.3754 + 0.3745) / 3) < 1e-12
    assert t["mean"]["mae"]["naive"] == 1.07 and t["comparability_class"] == "MATCHED_PUBLISHED_RECIPE_EXECUTED"
    assert "OPERATIONAL_AGREEMENT" in C.render_md(t)


def test_a_forged_record_is_refused():
    recs = good()
    recs[1]["metric"]["author_float32"]["mse"] = 0.3750      # altered, digest not recomputed
    with pytest.raises(C.ClosureRefusal, match="FORGED"):
        C.build(design(), recs, 96)


def test_a_missing_seed_is_refused():
    with pytest.raises(C.ClosureRefusal, match="MISSING_OR_EXTRA_SEED"):
        C.build(design(), good()[:2], 96)


def test_a_duplicated_seed_is_refused():
    recs = good()
    recs[2] = record(2022, 0.3754, 0.2512)
    with pytest.raises(C.ClosureRefusal, match="DUPLICATE_SEED"):
        C.build(design(), recs, 96)


def test_a_mismatched_population_sha_is_refused_even_with_a_valid_digest():
    recs = good()
    recs[2] = record(2023, 0.3745, 0.2508, pop="8" * 64)     # self-consistent, but other rows
    with pytest.raises(C.ClosureRefusal, match="POPULATION_MISMATCH"):
        C.build(design(), recs, 96)


def test_a_naive_not_proven_on_the_model_rows_is_refused():
    recs = good()
    r = copy.deepcopy(recs[0])
    r["naive"]["paired_on_the_same_rows_proved_by"]["naive_target_sha256"] = "9" * 64
    r["record_sha256"] = C.sha_obj({k: v for k, v in r.items() if k != "record_sha256"})
    recs[0] = r
    with pytest.raises(C.ClosureRefusal, match="NAIVE_NOT_PAIRED"):
        C.build(design(), recs, 96)


def test_a_record_from_another_design_is_refused():
    recs = good()
    r = copy.deepcopy(recs[0])
    r["design_sha256"] = "e" * 64
    r["record_sha256"] = C.sha_obj({k: v for k, v in r.items() if k != "record_sha256"})
    recs[0] = r
    with pytest.raises(C.ClosureRefusal, match="FOREIGN_DESIGN"):
        C.build(design(), recs, 96)


def test_outside_margin_is_classified_not_hidden():
    recs = [record(2021, 0.40, 0.2514), record(2022, 0.40, 0.2512), record(2023, 0.40, 0.2508)]
    assert C.build(design(), recs, 96)["mean"]["mse"]["class"] == "OUTSIDE_OPERATIONAL_MARGIN"


def test_cli_refuses_with_nonzero_exit_and_writes_nothing(tmp_path):
    d = tmp_path / "design.json"
    d.write_text(json.dumps(design()))
    paths = []
    for r in good()[:2]:
        p = tmp_path / f"{r['cell_id']}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))
    out = tmp_path / "out"
    assert C.main(["--design", str(d), "--horizon", "96", "--out-dir", str(out), *paths]) == 3
    assert not out.exists()
