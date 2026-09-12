"""C84-C85: the v3 design validates field by field, refuses deltas after
review, derives an honest population, and the score entry point stops
before reading anything."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import per_variable_design_v3 as D  # noqa: E402
import per_variable_screen as S  # noqa: E402


def fresh():
    return D.build_design()


def resign(d):
    d["design_sha256"] = D.sha_obj({k: v for k, v in d.items()
                                    if k != "design_sha256"})
    return d


def test_the_built_design_validates():
    assert D.validate(fresh()) == []


def test_a_delta_after_review_refuses():
    d = fresh()
    reviewed = d["design_sha256"]
    d["decision_rule"]["practical_margin"] = 0.005
    resign(d)
    assert any("DELTA_AFTER_REVIEW" in p for p in
               D.validate(d, reviewed_sha256=reviewed))


def test_an_edit_without_resigning_is_caught():
    d = fresh()
    d["seeds"] = [1, 2, 3]
    assert "design_sha256 does not match the content" in D.validate(d)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["arms"].pop("A3_CAPACITY_CONTROL"), "arms"),
    (lambda d: d["budget"].update(same_for_every_arm=False), "budget"),
    (lambda d: d["budget"].update(accelerator="GPU"), "accelerator"),
    (lambda d: d["operators"].append({"id": "O9_CENTRED", "params": {},
                                      "causal": False}), "non-causal"),
    (lambda d: d["operators"][3]["params"].update(min_periods=1), "window"),
    (lambda d: d["decision_rule"].update(practical_margin=1), "practical"),
    (lambda d: d["decision_rule"].update(per_panel_harm_margin=0.5), "harm"),
    (lambda d: d["panels"].update(minimum_panels=2), "minimum_panels"),
    (lambda d: d.update(seeds=[101, 101, 202]), "seeds"),
    (lambda d: d["evaluation_windows"].update(embargo_bars=3), "embargo"),
    (lambda d: d["arms"]["A3_CAPACITY_CONTROL"].update(
        information="shuffled copies of the inputs"), "A3"),
    (lambda d: d["target"].update(labels_read_by_this_document=True), "labels"),
    (lambda d: d["license"].update(scoring="GRANTED"), "scoring"),
    (lambda d: d.update(status="SEALED"), "status"),
    (lambda d: d.update(extra_field=1), "keys differ"),
])
def test_each_field_rule_refuses_its_violation(mutate, needle):
    d = copy.deepcopy(fresh())
    mutate(d)
    resign(d)
    problems = D.validate(d)
    assert any(needle.lower() in p.lower() for p in problems), problems


def test_a_missing_artifact_makes_the_population_undetermined(tmp_path):
    pop = D.derive_population(terminals_v3=None, dag_v3=tmp_path / "no.json",
                              temporal_contracts=[])
    assert pop["state"] == "UNDETERMINED" and pop["members"] == 0


def test_fewer_than_three_panels_yields_no_members(tmp_path):
    t = tmp_path / "t3"
    t.mkdir()
    (t / "a.json").write_text(json.dumps({"variable_id": "v", "layer":
                                          "INDEPENDENTLY_RECOMPUTED",
                                          "recomputation": "AGREES"}))
    dag = tmp_path / "dag.json"
    dag.write_text(json.dumps({"nodes": [
        {"dataset_id": "eth", "column": "c", "class": "CAUSAL_ACTIVE"}]}))
    con = tmp_path / "c.json"
    con.write_text(json.dumps({"gates": {"E5a": {"status": "OPEN"}},
                               "contract": {"dataset_id": "eth"}}))
    pop = D.derive_population(terminals_v3=t, dag_v3=dag,
                              temporal_contracts=[con])
    assert pop["panel_count"] == 1 and pop["members"] == 0
    assert pop["evaluable"] is False


def test_the_score_entry_point_refuses_before_reading_anything():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "before = set(sys.modules)\n"
        "import per_variable_screen as S\n"
        "try:\n    S.main()\nexcept SystemExit as e:\n"
        "    heavy = {m for m in set(sys.modules) - before if m.split('.')[0] "
        "in ('pandas','numpy','sklearn','torch','tensorflow','pyarrow')}\n"
        "    print(e.code_name if hasattr(e, 'code_name') else e, sorted(heavy))\n"
    ) % str(REPO / "tools")
    out = subprocess.run((sys.executable, "-c", code), capture_output=True,
                         text=True).stdout.strip()
    assert out == f"{S.REFUSAL} []"


def test_score_function_refuses_too():
    with pytest.raises(S.ScoreRefusal):
        S.score(panel="eth")
