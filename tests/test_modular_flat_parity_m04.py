"""Cross-lane parity: M04's tied search-space projection composes with M01's model grammar.

M04 (tools/modular_search_space.py at 3ceabfad, pinned as a verbatim fixture) maps
DOIN's flat search parameters to a nested candidate whose ``model`` is the
predictor.modular.v1 config. M01 (predictor_plugins/modular_config.py) is the
complete reversible flat encoding of that model schema. The key sets differ by
design (tied/uniform search parameters vs one key per branch); this test proves
the composition is value-exact and that the two grammars cannot shadow each other.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt

FIXTURES = Path(__file__).parent / "fixtures" / "m04_3ceabfad"


def _m04():
    source = json.loads((FIXTURES / "SOURCE.json").read_text())
    for name, digest in source["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location("m04_space", FIXTURES / "modular_search_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base(features=12):
    return {"feature_names": [f"f{i}" for i in range(features)], "window": 24, "sample_hours": 1,
            "horizons": [24], "target_feature_indices": [features - 1],
            "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False},
            "evaluator_fixed": {}}


@pytest.mark.parametrize("grouping", [1, 3])
def test_m04_projection_lands_value_exact_in_m01_grammar(grouping):
    m04 = _m04()
    space = json.loads((FIXTURES / "ecl_l24_h24_search_space_v1.json").read_text())
    flat04 = json.loads((FIXTURES / "ecl_l24_h24_default_r0_v1.json").read_text())
    flat04.update({"train.seed": 2021, "branch.grouping_size": grouping})
    # The approved architecture keeps every branch step (branch_steps == window) and reduces
    # time only in the core, 24 -> 12 -> 6 -> 6. M04's pinned default point predates it
    # (branch_steps 12, factors [2,1,1]); the corrected point is inside M04's own bounds.
    flat04.update({"model.branch_steps": 24, "core.time_factor_0": 2, "core.time_factor_1": 2,
                   "core.time_factor_2": 1})
    space["bounds"]["branch.grouping_size"]["choices"].append(grouping) \
        if grouping not in space["bounds"]["branch.grouping_size"]["choices"] else None
    if flat04["train.loss"] == "huber":
        flat04.setdefault("train.huber_delta", 1.0)
    nested = m04.from_flat(flat04, _base(), space)
    assert m04.to_flat(nested, space) == flat04                   # M04's own round trip intact
    model = nested["model"]
    flat01 = mc.flatten(model)
    assert mc.unflatten(flat01) == mt._normalize(model)             # M01 round trip on M04 output
    order = flat01["modular.branch_order"]
    assert len(order) == 12 // grouping
    for name in order:
        assert flat01[f"branches.{name}.params.channels"] == flat04["branch.channels"]
        assert flat01[f"branches.{name}.params.kernel_size"] == flat04["branch.kernel_size"]
        assert flat01[f"branches.{name}.plugin"] == flat04["branch.plugin"]
        assert flat01[f"branches.{name}.regime"] == flat04["branch.regime"]
    for key in ("d_model", "heads", "blocks", "ff_dim", "dropout", "kernel_size"):
        assert flat01[f"core.params.{key}"] == flat04[f"core.{key}"]
    stages = flat04["core.stage_count"]
    assert flat01["core.params.stage_channels"] == [
        flat04[f"core.stage_channels_{i}"] for i in range(stages - 1)] + [flat04["model.output_channels"]]
    assert flat01["core.params.time_factors"] == [flat04[f"core.time_factor_{i}"] for i in range(stages)]
    for key in ("branch_steps", "output_steps", "output_channels"):
        assert flat01[f"modular.{key}"] == flat04[f"model.{key}"]
    assert flat01["core.regime"] == flat04["core.regime"]
    # train.* is evaluator schema, outside the model grammar: no M01 key starts with it
    assert not any(k.startswith(("train.", "branch.", "model.")) for k in flat01)


def test_raw_m04_keys_are_refused_by_the_m01_grammar_not_ignored():
    model = mt.default_config(["a", "b"])
    for key in ("core.d_model", "core.stage_channels_0", "core.stage_count"):
        with pytest.raises(ValueError):
            mc.apply_flat_overrides(model, {key: 1})
    # branch./model./train. keys are not in the M01 namespace, so the facade never reads them
    assert not any(mc.is_modular_key(k) for k in ("branch.channels", "model.output_steps", "train.loss"))


def test_m04_pinned_default_point_is_the_superseded_design_and_is_refused():
    m04 = _m04()
    space = json.loads((FIXTURES / "ecl_l24_h24_search_space_v1.json").read_text())
    flat04 = json.loads((FIXTURES / "ecl_l24_h24_default_r0_v1.json").read_text())
    flat04.update({"train.seed": 2021, "train.huber_delta": 1.0})
    assert flat04["model.branch_steps"] == 12                       # finding for M04: old design
    nested = m04.from_flat(flat04, _base(), space)
    with pytest.raises(ValueError, match="branch_steps must equal window"):
        mc.flatten(nested["model"])
