"""Lane A A.4: direct multi-output arms and host-root-relative contexts (invented CPU data).

Not historical, clock or cost evidence. Direct multi-output heads are business
horizons of one family, never a trajectory.
"""

import json
import shutil

import numpy as np
import pytest

from tests._i6n_trajectory_fixture import build, genome, write_json
from tools import i6n_historical_cell as cell
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_search as search
from tools.i6m_data import file_hash

SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]


def finalist(h, dx, dy=None, multiplier=1, core=None):
    core = core or {"entrance": "conv", "width": 8, "transformer_blocks": 0, "reducer_filters": [4],
                    "reducer_strides": [1], "adapter": "none", "adapter_width": None, "n_horizons": 1}
    return search.effective_config(genome(search, h, dx=dx, dy=dy, multiplier=multiplier, core=core))


# --- pure mapping and identity -------------------------------------------------------------

def test_arm_config_maps_recipe_to_one_head_per_business_horizon():
    recipe = finalist(12, 3, dy=6, multiplier=4)
    arm = evaluator.multi_output_arm_config(recipe, SHORT)
    assert arm["schema"] == "i6n.multi_output_arm.v1" and arm["family"] == "short"
    assert arm["output_mode"] == "direct_multi_output" and "trajectory" not in json.dumps(arm["output_mode"])
    assert arm["output_offsets"] == arm["horizons"] == SHORT
    assert (arm["history_hours"], arm["delta_x"], arm["samples"]) == (48, 3, 17)
    assert arm["input_offsets"] == list(range(-48, 1, 3))
    assert arm["input_transform"] == recipe["input_transform"]
    core = dict(recipe["core"], n_horizons=6)
    assert arm["core"] == core and arm["final_steps"] == recipe["final_steps"]
    genome_v1 = {k: recipe[k] for k in search._GENOME_KEYS}
    assert arm["recipe"] == dict(source_horizon=12, source_output_mode="trajectory",
                                 genotype_canonical_sha256=search.effective_identity(genome_v1),
                                 effective_config_sha256=evaluator.effective_config_sha256(recipe),
                                 config=recipe)
    again = evaluator.multi_output_arm_config(json.loads(json.dumps(recipe)), list(SHORT))
    assert again == arm
    assert evaluator.arm_config_sha256(arm) == evaluator.effective_config_sha256(arm)
    assert evaluator.arm_config_sha256(arm) != evaluator.arm_config_sha256(
        evaluator.multi_output_arm_config(finalist(12, 3, multiplier=4), SHORT))


@pytest.mark.parametrize("horizons,h,match", [
    ([2, 3, 4, 5, 6], 12, "family"), ([2, 3, 4, 5, 6, 12, 18], 12, "family"),
    (LONG, 12, "family"), (SHORT, 18, "family"), ([12, 6, 5, 4, 3, 2], 12, "family"),
])
def test_arm_family_must_be_exact_and_contain_the_recipe_horizon(horizons, h, match):
    with pytest.raises(ValueError, match=match):
        evaluator.multi_output_arm_config(finalist(h, 3 if h == 12 else 2), horizons)


def test_arm_refuses_noncanonical_recipe():
    recipe = finalist(12, 3)
    recipe["samples"] += 1
    with pytest.raises(ValueError, match="effective"):
        evaluator.multi_output_arm_config(recipe, SHORT)


# --- freeze per head, masks never couple heads ------------------------------------------------

@pytest.fixture
def rooted(tmp_path):
    """A data root holding inputs, the base cell config (relative paths) and bindings."""
    root = tmp_path / "data_root"
    root.mkdir()
    base = build(root, horizon=12, samples=9, spacing=3, drop={(10, 50)})
    for key in ("input", "source_clock"):
        base[key]["path"] = evaluator.Path(base[key]["path"]).name   # relative to the config file
    path = root / "cells" / "unit_h12.json"
    path.parent.mkdir()
    for key in ("input", "source_clock"):
        shutil.move(str(root / base[key]["path"]), str(path.parent / base[key]["path"]))
    write_json(path, base)
    return root, path, base


def test_family_freeze_seals_per_head_support_equal_to_endpoint_freezes(rooted, monkeypatch):
    root, path, base = rooted
    monkeypatch.setenv("I6N_DATA_ROOT", str(root))
    binding = evaluator.freeze(path, root / "bindings" / "arm", experiment="exp", fold="W2024-01-07",
                               score_outer=True, family="short", data_root=root)
    frozen = json.loads((root / "bindings" / "arm" / "FROZEN.json").read_text())
    assert frozen["mode"] == "multi_output" and frozen["offsets"] == SHORT
    for h in SHORT:
        single = dict(base, horizon_hours=h)
        single_path = root / "cells" / f"unit_h{h}_endpoint.json"
        write_json(single_path, single)
        cfg = cell.read_config(single_path)
        endpoint = evaluator.support(cell.prepare(cfg), cfg, portable=True)
        head = frozen["heads"][str(h)]
        assert head["status"] == "TRAINED"
        assert head["support"] == endpoint, h   # identical rows, counts, naive and digests
        assert head["outer"]["count"] > 0
    pooled = binding["i6n_support"]
    assert pooled["counts"]["train"] == sum(frozen["heads"][str(h)]["support"]["counts"]["train"] for h in SHORT)


def test_long_head_without_train_support_is_typed_and_short_heads_kept(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    # TRAIN week bars only up to hour 60: 72h labels impossible, 18..48 possible.
    base = build(root, horizon=72, samples=3, spacing=36, keep=lambda w, h: w != 10 or h <= 60)
    path = root / "base.json"
    write_json(path, base)
    monkeypatch.setenv("I6N_DATA_ROOT", str(root))
    evaluator.freeze(path, root / "arm", experiment="exp", fold="W2024-01-07", family="long", data_root=root)
    frozen = json.loads((root / "arm" / "FROZEN.json").read_text())
    status = {h: frozen["heads"][str(h)]["status"] for h in LONG}
    assert status[72] == "NO_TRAIN_SUPPORT" and status[60] == "NO_TRAIN_SUPPORT"
    assert status[18] == status[24] == status[36] == "TRAINED"
    assert frozen["heads"]["72"]["support"] is None


# --- host-root-relative contexts ----------------------------------------------------------

def test_relative_context_has_no_absolute_paths_and_is_host_independent(rooted, tmp_path, monkeypatch):
    root, path, _ = rooted
    monkeypatch.setenv("I6N_DATA_ROOT", str(root))
    binding = evaluator.freeze(path, root / "bindings" / "ep", experiment="exp", fold="2023Q4", data_root=root)
    ctx = binding["i6n_evaluation_context"]
    assert ctx["schema"] == "i6n.historical_evaluator.v3"
    text = json.dumps(ctx)
    assert str(root) not in text and str(tmp_path) not in text
    assert ctx["base_config"] == {"relpath": "cells/unit_h12.json", "sha256": file_hash(path)}
    request = dict(config=finalist(12, 3, dy=6), generation=0, context=ctx, support=binding["i6n_support"])
    cfg, _ = evaluator.load_request(request)
    identity = evaluator.c1.canonical_digest(evaluator.identity(cfg, request))
    fit = evaluator.fit_id(request)
    other = tmp_path / "other_host_root"
    shutil.copytree(root, other)
    monkeypatch.setenv("I6N_DATA_ROOT", str(other))
    cfg2, _ = evaluator.load_request(request)
    assert evaluator.c1.canonical_digest(evaluator.identity(cfg2, request)) == identity
    assert evaluator.fit_id(request) == fit
    assert evaluator.candidate_root(request).is_relative_to(other)
    monkeypatch.delenv("I6N_DATA_ROOT")
    with pytest.raises(ValueError, match="I6N_DATA_ROOT"):
        evaluator.load_request(request)


@pytest.mark.parametrize("relpath", ["../escape.json", "/abs/path.json"])
def test_relative_context_refuses_escaping_paths(rooted, monkeypatch, relpath):
    root, path, _ = rooted
    monkeypatch.setenv("I6N_DATA_ROOT", str(root))
    binding = evaluator.freeze(path, root / "b", experiment="exp", fold="F", data_root=root)
    ctx = dict(binding["i6n_evaluation_context"], base_config=dict(relpath=relpath, sha256="0" * 64))
    with pytest.raises(ValueError, match="relpath"):
        evaluator.load_request(dict(config=finalist(12, 3), generation=0, context=ctx,
                                    support=binding["i6n_support"]))


def test_freeze_refuses_inputs_outside_the_data_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    base = build(tmp_path, horizon=12, samples=9, spacing=3)   # inputs outside root
    path = root / "base.json"
    write_json(path, base)
    with pytest.raises(ValueError, match="data root"):
        evaluator.freeze(path, root / "b", experiment="exp", fold="F", data_root=root)


# --- one tiny real fit of a direct multi-output arm -------------------------------------------

def test_direct_multi_output_arm_fit_receipt(rooted, monkeypatch):
    root, path, _ = rooted
    monkeypatch.setenv("I6N_DATA_ROOT", str(root))
    binding = evaluator.freeze(path, root / "bindings" / "arm", experiment="exp", fold="W2024-01-07",
                               score_outer=True, family="short", data_root=root)
    frozen = json.loads((root / "bindings" / "arm" / "FROZEN.json").read_text())
    arm = evaluator.multi_output_arm_config(finalist(12, 3, dy=6), SHORT)
    request = dict(config=arm, generation=0, context=binding["i6n_evaluation_context"],
                   support=binding["i6n_support"])
    with pytest.raises(ValueError, match="family"):   # an endpoint genome cannot use an arm binding
        evaluator.evaluate(dict(request, config=finalist(12, 3)))
    receipt = evaluator.evaluate(request)
    assert search._score(receipt, binding["i6n_support"])[0] == receipt["fitness"]
    cr = receipt["metadata"]["candidate_receipt"]
    assert cr["output_mode"] == "direct_multi_output" and cr["delta_y_hours"] is None
    assert cr["effective_config_sha256"] == evaluator.arm_config_sha256(arm)
    assert cr["genotype_canonical_sha256"] == arm["recipe"]["genotype_canonical_sha256"]
    assert [h["offset_hours"] for h in cr["heads"]] == SHORT
    assert {h["head_role"] for h in cr["heads"]} == {"DIRECT_HEAD"}
    assert len(cr["rows"]) == 6 * 3
    for row in cr["rows"]:
        assert row["head_role"] == "DIRECT_HEAD" and row["comparison"] is True
        assert row["status"] == "SCORED"
        sealed = frozen["heads"][str(row["head_offset_hours"])]
        if row["split"] == "VALIDATION":
            assert row["n"] == sealed["outer"]["count"]
            assert row["naive_sum_abs_error"] == sealed["outer"]["naive_sum"]
        else:
            split = row["split"].lower()
            assert row["n"] == sealed["support"]["counts"][split]
            assert row["naive_sum_abs_error"] == sealed["support"]["naive_sums"][split]
            assert row["support_sha256"] == sealed["support_origins_sha256"][split]
    assert set(receipt["error_sums"]) == {"train", "inner"}
    assert receipt["metadata"]["shared_receipt_scope"] == "POOLED_FAMILY_HEADS_DIAGNOSTIC_NOT_SELECTION"
    assert receipt["metadata"]["checkpoint_reload_verified"]
    monkeypatch.setattr(evaluator, "fit_candidate", lambda *a: pytest.fail("refit"))
    assert evaluator.evaluate(dict(request, generation=4)) == receipt
