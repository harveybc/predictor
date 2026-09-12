"""C89-C92 battery: every C89 PRE case refuses or diverges, the semantic
contract withholds non-measurable columns, and v4 is additive."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import lake_descriptor_recompute as R  # noqa: E402
import lake_semantic_contract as S  # noqa: E402
import verify_lake_terminals_v3 as V  # noqa: E402


def pbytes(values, column="close", extra=None):
    buf = io.BytesIO()
    cols = {"timestamp": pa.array(range(len(values))), column: pa.array(values)}
    cols.update(extra or {})
    pq.write_table(pa.table(cols), buf)
    return buf.getvalue()


def census_doc(apps, variables):
    top = {"appearances": apps, "availability_contract": {}, "censused_at": "t",
           "conflicts": [], "coverage": {}, "delta": {}, "dictionary_coverage": {},
           "equivalence_classes": [], "external_full_profiles": [], "gaps": {},
           "manifest_generated_at": "t", "manifest_stage": "s", "policy": {},
           "schema": V.CENSUS_SCHEMA, "value_profiles_sampled": [],
           "variables": variables}
    top["census_sha256"] = V.canonical_census_sha256(top)
    return top


def appearance(app_id, rel, payload, columns):
    return {"appearance_id": app_id, "bytes_read_for_digest": len(payload),
            "ctime_ns": 1, "declared_columns": columns, "declared_rows": 1,
            "digest_state": "PHYSICALLY_DIGESTED", "entity": "e", "frequency": "4h",
            "manifest_status": "ok", "mtime_ns": 1, "period_end": "x",
            "period_start": "x", "physical_sha256": hashlib.sha256(payload).hexdigest(),
            "presence": "PRESENT", "profile_depth": "DECLARED_SCHEMA_ONLY",
            "provenance": "p", "relative_path": rel, "size_bytes": len(payload),
            "source_class": "c"}


def variable(vid, app_ids, concept):
    return {"appearance_count": len(app_ids), "appearances": app_ids,
            "availability_contract": "UNAVAILABLE", "available_time": "UNAVAILABLE",
            "concept_name": concept, "entity": "e", "event_time": "UNAVAILABLE",
            "frequencies": ["4h"], "license": "UNKNOWN",
            "lineage": {"provenance_files": [], "source_declarations": [],
                        "upstream_join_rule": "r", "upstream_source_dirs": []},
            "physical_type": "UNKNOWN", "profile_depth": "DECLARED_SCHEMA_ONLY",
            "role": "UNKNOWN", "semantics": "UNKNOWN", "semantics_source": "UNAVAILABLE",
            "source_class": "c", "unit": "UNKNOWN", "variable_id": vid}


def world(tmp_path, *, values=None, column="close", mutate_census=None, raw_census=None):
    values = np.arange(1, 17, dtype=np.float64) if values is None else values
    payload = pbytes(values, column)
    lake = tmp_path / "lake"
    (lake / "f").mkdir(parents=True)
    (lake / "f/src.parquet").write_bytes(payload)
    doc = census_doc([appearance("app_0", "f/src.parquet", payload, ["timestamp", column])],
                     [variable("var_a", ["app_0"], column)])
    digest = doc["census_sha256"]
    if mutate_census:
        mutate_census(doc, lake)
    (tmp_path / "census").mkdir()
    cpath = tmp_path / "census" / f"census-{digest}.json"
    cpath.write_text(raw_census(doc) if raw_census else json.dumps(doc))
    state = tmp_path / "state"
    (state / "terminals").mkdir(parents=True)
    (state / "PRE_LEDGER.json").write_text(json.dumps({
        "census_sha256": digest, "censused_at": "t", "conceptual_variables": 1,
        "identities": ["var_a"], "physical_appearances": 1, "pre_ledger_sha256": "p",
        "rule": "r", "schema": V.LEDGER_SCHEMA, "written_at": "w"}))
    t = {"appearance": "app_0", "batch": "b", "concept_name": column, "descriptors": 25,
         "entity": "e", "measured_at": "t", "not_identifiable": 1, "outcome": "MEASURED",
         "rows_used": 11, "variable_id": "var_a"}
    tpath = state / "terminals" / (hashlib.sha256(b"var_a").hexdigest()[:32] + ".json")
    tpath.write_text(json.dumps(t))
    return state, lake, cpath, digest, tpath, payload


def published(values, payload, census_sha):
    win = R.window_rows(len(values))
    rec = R.recompute(np.asarray(values[:win["rows_used"]], dtype=float))
    sha = hashlib.sha256(payload).hexdigest()
    rows = {"characterization_disposition": {"value": None, "value_text": "MEASURED",
            "identifiable": True, "source_sha256": census_sha, "window_contract": win}}
    names = (V.INSUFFICIENT if "insufficient_finite_observations" in rec else V.FULL) - {"characterization_disposition"}
    for d in names:
        v, ident = rec.get(d, (1.0, True))
        rows[d] = {"value": v if ident else None, "value_text": None,
                   "identifiable": bool(ident), "source_sha256": sha, "window_contract": win}
    return {"var_a": rows}


# --------------------------------------------------------------- C90
def test_the_canonical_census_verifies_and_recomputes_like_the_producer(tmp_path):
    s, l, c, d, _, payload = world(tmp_path)
    r = V.verify(s, l, c, expected_census_sha256=d,
                 published={"rows": published(np.arange(1, 17, dtype=float), payload, d)})
    assert r["population"]["verdict"] == V.POP_VERIFIED
    ident = r["census_identity"]
    assert ident["recomputed_canonical"] == d
    assert ident["expectation_role"] == "REVIEWER_SUPPLIED_EXPECTATION"
    assert ident["raw_file_sha256"] != d
    assert r["layers"][V.INDEPENDENTLY_RECOMPUTED] == 1


def test_a_mutated_appearance_under_the_old_declared_digest_refuses(tmp_path):
    def mutate(doc, lake):
        evil = pbytes(np.arange(101, 117, dtype=np.float64))
        (lake / "f/evil.parquet").write_bytes(evil)
        doc["appearances"][0].update(relative_path="f/evil.parquet",
                                     physical_sha256=hashlib.sha256(evil).hexdigest(),
                                     size_bytes=len(evil))
    s, l, c, d, _, _ = world(tmp_path, mutate_census=mutate)
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(s, l, c, expected_census_sha256=d)
    assert e.value.code == "CENSUS_DIGEST_MISMATCH"


def test_a_census_with_a_duplicate_key_refuses(tmp_path):
    s, l, c, d, _, _ = world(tmp_path, raw_census=lambda doc: json.dumps(doc).replace(
        '"appearance_id": "app_0"', '"appearance_id": "x", "appearance_id": "app_0"', 1))
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(s, l, c, expected_census_sha256=d)
    assert e.value.code == "CENSUS_UNREADABLE" and "duplicate" in e.value.reason


def test_a_census_with_a_non_finite_constant_refuses(tmp_path):
    s, l, c, d, _, _ = world(tmp_path, raw_census=lambda doc: json.dumps(doc).replace(
        '"coverage": {}', '"coverage": {"q": NaN}', 1))
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(s, l, c, expected_census_sha256=d)
    assert e.value.code == "CENSUS_UNREADABLE"


@pytest.mark.parametrize("where", ["appearance", "variable", "top"])
def test_an_extra_key_anywhere_in_the_census_refuses(tmp_path, where):
    def mutate(doc, lake):
        target = {"appearance": doc["appearances"][0], "variable": doc["variables"][0],
                  "top": doc}[where]
        target["zzz_undeclared"] = 1
        doc["census_sha256"] = V.canonical_census_sha256(doc)
    s, l, c, d, _, _ = world(tmp_path, mutate_census=mutate)
    doc = json.loads(c.read_text())
    new = tmp_path / "census" / f"census-{doc['census_sha256']}.json"
    c.rename(new)
    led = json.loads((s / "PRE_LEDGER.json").read_text())
    led["census_sha256"] = doc["census_sha256"]
    (s / "PRE_LEDGER.json").write_text(json.dumps(led))
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(s, l, new, expected_census_sha256=doc["census_sha256"])
    assert e.value.code == "CENSUS_SCHEMA"


def test_a_terminal_with_a_duplicate_key_diverges(tmp_path):
    s, l, c, d, tpath, _ = world(tmp_path)
    tpath.write_text(tpath.read_text()[:-1] + ',"rows_used":999,"rows_used":11}')
    r = V.verify(s, l, c, expected_census_sha256=d)
    kinds = {x["kind"] for x in r["population"]["divergences"]}
    assert "TERMINAL_UNREADABLE_OR_LENIENT_JSON" in kinds


def test_no_expectation_refuses(tmp_path):
    s, l, c, d, _, _ = world(tmp_path)
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(s, l, c, expected_census_sha256="")
    assert e.value.code == "EXPECTATION_REQUIRED"


# --------------------------------------------------------------- C91
def test_the_int64_datetime_sentinel_is_semantically_unresolved(tmp_path):
    vals = np.full(32, np.iinfo(np.int64).min, dtype=np.int64)
    s, l, c, d, _, payload = world(tmp_path, values=vals,
                                   column="announcement_datetime_local_utc")
    r = V.verify(s, l, c, expected_census_sha256=d,
                 published={"rows": {"var_a": {"mean": {"value": -9.2e18, "identifiable": True}}}})
    pv = r["_per_var"]["var_a"]
    assert pv["layer"] == V.SEMANTICALLY_UNRESOLVED
    sem = pv["appearances"]["app_0"]["semantic"]
    assert sem["state"] == S.SEMANTIC_TYPE_UNRESOLVED
    assert sem["sentinel_candidate_counts"] == {"INT64_MIN": 22}
    assert "recomputation" not in pv["appearances"]["app_0"]
    assert pv["published_numeric_descriptors_withdrawn"] == ["mean"]


def test_an_undeclared_integer_sentinel_on_a_plain_column_is_missing_policy_unresolved():
    arr = pa.array(np.array([1, 2, np.iinfo(np.int32).min, 4], dtype=np.int64))
    out = S.column_contract(arr, "value", {"physical_type": "UNKNOWN"}, 4)
    assert out["state"] == S.MISSING_POLICY_UNRESOLVED
    assert out["null_policy"]["source"] == "NONE_DECLARED_IN_CENSUS"


def test_infinities_without_a_policy_are_missing_policy_unresolved():
    arr = pa.array(np.array([1.0, np.inf, 2.0]))
    assert S.column_contract(arr, "close", {}, 3)["state"] == S.MISSING_POLICY_UNRESOLVED


def test_non_numeric_types_are_never_converted():
    arr = pa.array(["a", "b"])
    assert S.column_contract(arr, "label", {}, 2)["state"] == S.NON_NUMERIC


def test_a_zero_dominated_legitimate_column_stays_measurable():
    arr = pa.array(np.zeros(100))
    assert S.column_contract(arr, "Dividends", {}, 100)["state"] == S.NUMERIC_MEASURABLE


# --------------------------------------------------------------- C92
def test_v4_is_additive_and_history_is_guarded_through_custody(tmp_path):
    s, l, c, d, _, payload = world(tmp_path)
    for sub in ("terminals_v2", "terminals_v3"):
        (s / sub).mkdir()
        (s / sub / "x.json").write_text("{}")
    before = {x: V.custody_content_digest(s, x) for x in ("terminals", "terminals_v2", "terminals_v3")}
    r = V.verify(s, l, c, expected_census_sha256=d,
                 published={"rows": published(np.arange(1, 17, dtype=float), payload, d)})
    idx = V.supersede_v4(s, r, superseded_at="t", code_identity={})
    assert idx["v4_written"] == 1
    assert {x: V.custody_content_digest(s, x) for x in before} == before
    with pytest.raises(V.VerificationRefusal):
        V.supersede_v4(s, r, superseded_at="t", code_identity={})


def test_the_v3_verifier_imports_no_producer_code():
    import ast
    src = (REPO / "tools/verify_lake_terminals_v3.py").read_text()
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert not any("characteriz" in m or "incremental_census" in m for m in mods)
