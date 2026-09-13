"""C112: the four terminal scopes stay apart in executable language."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs/audits/evidence"


def _load():
    spec = importlib.util.spec_from_file_location("terminal_evidence_scope",
                                                  ROOT / "tools/terminal_evidence_scope.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load()


@pytest.mark.parametrize("text", [
    "1,501 verified variables",
    "variables verified: 1501",
    "1,501 variables verificadas",
    "all terminals verified",
    "Verified columns: 1,501",
])
def test_verified_variables_language_is_refused(text):
    assert S.language_problems(text)


def test_qualified_sentences_pass():
    assert S.describe("NUMERIC_DESCRIPTORS_RECOMPUTED", 1501) == \
        "1,501 variables with numeric descriptors recomputed from their bound bytes"
    for scope in S.SCOPES:
        assert not S.language_problems(S.describe(scope, 7))


def test_describe_refuses_unknown_scope_and_bad_counts():
    with pytest.raises(S.ScopeRefusal, match="UNKNOWN_SCOPE"):
        S.describe("VERIFIED", 1)
    for bad in (True, -1, 1.0):
        with pytest.raises(S.ScopeRefusal):
            S.describe("PHYSICAL_TYPE_KNOWN", bad)


def test_semantic_declarations_require_every_field():
    census = {"variables": [
        {"semantics": "price", "role": "input_feature", "unit": "1", "license": "CC0",
         "missing_policy": "EXCLUDE_ROW"},
        {"semantics": "price", "role": "input_feature", "unit": "1", "license": "UNKNOWN",
         "missing_policy": "EXCLUDE_ROW"},
        {"semantics": "price", "role": "input_feature", "unit": "1", "license": "CC0"},
    ]}
    assert S.semantic_declarations_known(census) == 1


def test_disposition_from_the_real_v3_evidence(tmp_path):
    doc = S.build_disposition(EVID / "LAKE_TERMINAL_VERIFICATION.v3.json",
                              EVID / "TERMINAL_SUPERSESSION.v3.json", None)
    sc = doc["scopes"]
    assert doc["decision"] == "TERMINALS_V4_ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE"
    assert sc["NUMERIC_DESCRIPTORS_RECOMPUTED"]["count"] == 1501
    assert sc["PHYSICAL_TYPE_KNOWN"]["count"] == 1505
    assert sc["PRODUCER_AUTHORITY_ONLY"]["count"] == 460
    assert sc["SEMANTICALLY_UNRESOLVED"]["count"] == 4
    assert doc["terminals"] == 1965
    assert "semantics" in sc["NUMERIC_DESCRIPTORS_RECOMPUTED"]["does_not_imply"]
    assert not S.language_problems(json.dumps(doc))


def test_disposition_refuses_unbound_or_diverging_evidence(tmp_path):
    ver = json.loads((EVID / "LAKE_TERMINAL_VERIFICATION.v3.json").read_text())
    sup = json.loads((EVID / "TERMINAL_SUPERSESSION.v3.json").read_text())
    v, s = tmp_path / "v.json", tmp_path / "s.json"
    bad = copy.deepcopy(sup)
    bad["verification_sha256"] = "0" * 64
    v.write_text(json.dumps(ver)); s.write_text(json.dumps(bad))
    with pytest.raises(S.ScopeRefusal, match="NOT_BOUND"):
        S.build_disposition(v, s, None)
    div = copy.deepcopy(ver); div["layers"]["DIVERGES"] = 1
    sup2 = copy.deepcopy(sup); sup2["layers"]["DIVERGES"] = 1
    v.write_text(json.dumps(div)); s.write_text(json.dumps(sup2))
    with pytest.raises(S.ScopeRefusal, match="DIVERGING"):
        S.build_disposition(v, s, None)
