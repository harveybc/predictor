"""C69-C72 battery. Every C69 counterexample from the PRE must now
refuse or diverge; the recomputation must catch a fabricated value and
must never count an underspecified descriptor as verified."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import lake_descriptor_recompute as R  # noqa: E402
import verify_lake_terminals as V  # noqa: E402

SERIES = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0,
                   12.0, 11.0, 13.0, 15.0, 14.0])


def tname(vid):
    return hashlib.sha256(vid.encode()).hexdigest()[:32] + ".json"


def parquet_bytes(values=SERIES, column="close"):
    import io
    buf = io.BytesIO()
    pq.write_table(pa.table({"timestamp": pa.array(range(len(values))),
                             column: pa.array(values)}), buf)
    return buf.getvalue()


def published_for(vid, values=SERIES, *, sha, rel):
    win = R.window_rows(len(values))
    rec = R.recompute(np.asarray(values[:win["rows_used"]], dtype=float))
    rows = {"characterization_disposition": {
        "value": None, "value_text": "MEASURED", "identifiable": True,
        "source_id": rel, "source_sha256": sha, "window_contract": win}}
    names = (V.INSUFFICIENT_DESCRIPTORS if "insufficient_finite_observations"
             in rec else V.FULL_DESCRIPTORS) - {"characterization_disposition"}
    for d in names:
        v, ident = rec.get(d, (1.23, True))
        rows[d] = {"value": v if ident else None,
                   "value_text": None if ident else "UNAVAILABLE",
                   "identifiable": bool(ident), "source_id": rel,
                   "source_sha256": sha, "window_contract": win}
    return rows


def world(tmp_path, *, rel="features/src.parquet", payload=None,
          terminal=None, variables=None, appearances=None, ni=False,
          write_source=True):
    payload = parquet_bytes() if payload is None else payload
    lake = tmp_path / "lake"
    (lake / "features").mkdir(parents=True)
    if write_source:
        (lake / "features" / "src.parquet").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    appearances = appearances or [{"appearance_id": "app_0",
                                   "relative_path": rel,
                                   "physical_sha256": sha,
                                   "size_bytes": len(payload)}]
    variables = variables or [{"variable_id": "var_a",
                               "appearances": ["app_0"],
                               "concept_name": "close", "entity": "e"}]
    census = {"appearances": appearances, "variables": variables}
    digest = hashlib.sha256(json.dumps(census, sort_keys=True).encode()
                            ).hexdigest()
    census["census_sha256"] = digest
    cdir = tmp_path / "census"
    cdir.mkdir()
    cpath = cdir / f"census-{digest}.json"
    cpath.write_text(json.dumps(census))
    state = tmp_path / "state"
    (state / "terminals").mkdir(parents=True)
    ids = [v["variable_id"] for v in variables]
    (state / "PRE_LEDGER.json").write_text(json.dumps({
        "census_sha256": digest, "censused_at": "x",
        "conceptual_variables": len(ids), "identities": ids,
        "physical_appearances": len(appearances), "pre_ledger_sha256": "p",
        "rule": "r", "schema": V.LEDGER_SCHEMA, "written_at": "w"}))
    for v in variables:
        doc = terminal or {
            "appearance": v["appearances"][0], "batch": "b",
            "concept_name": v["concept_name"], "descriptors": 25,
            "entity": v["entity"], "measured_at": "t",
            "not_identifiable": 1, "outcome": "MEASURED", "rows_used": 10,
            "variable_id": v["variable_id"]}
        if ni:
            doc = {"batch": "b", "concept_name": v["concept_name"],
                   "entity": v["entity"], "measured_at": "t",
                   "outcome": "NOT_IDENTIFIABLE", "reason": "axis",
                   "variable_id": v["variable_id"]}
        (state / "terminals" / tname(v["variable_id"])).write_text(
            json.dumps(doc))
    return state, lake, cpath, digest, sha


def run(state, lake, cpath, digest, **kw):
    return V.verify(state, lake, cpath, expected_census_sha256=digest, **kw)


def kinds(r):
    return {d["kind"] for d in r["population"]["divergences"]}


# ------------------------------------------------ positive, fully bound
def test_a_bound_measured_terminal_recomputes_without_divergence(tmp_path):
    state, lake, cpath, digest, sha = world(tmp_path)
    pub = {"rows": {"var_a": published_for("var_a", sha=sha,
                                           rel="features/src.parquet")},
           "attempt": "t", "duplicates": []}
    r = run(state, lake, cpath, digest, published=pub)
    assert r["population"]["verdict"] == V.POPULATION_VERIFIED
    assert r["recomputation"]["verdict"] == V.RECOMPUTED_AGREES
    comp = r["_comparisons"]["var_a"]
    assert comp["layer"] == V.INDEPENDENTLY_RECOMPUTED
    assert comp["descriptors"]["mean"]["state"] == V.AGREES
    assert comp["descriptors"]["p99"]["state"] == \
        V.NOT_INDEPENDENTLY_VERIFIABLE
    assert "TERMINALS_VERIFIED_EXACT" not in json.dumps(
        {k: v for k, v in r.items() if not k.startswith("_")}).replace(
        r["retired_verdict"], "")


def test_no_published_rows_means_no_recomputation_claim(tmp_path):
    r = run(*world(tmp_path)[:4])
    assert r["recomputation"]["verdict"] == V.RECOMPUTATION_NOT_RUN


# ------------------------------------------------- C69.1 absent source
def test_an_absent_source_diverges(tmp_path):
    r = run(*world(tmp_path, write_source=False)[:4])
    assert r["population"]["verdict"] == V.POPULATION_DIVERGES
    assert "SOURCE_ABSENT" in kinds(r)


# ------------------------------------------- C69.2 fabricated descriptor
def test_a_fabricated_descriptor_block_is_a_schema_divergence(tmp_path):
    t = {"appearance": "app_0", "batch": "b", "concept_name": "close",
         "descriptors": {"fabricated": 999}, "entity": "e",
         "measured_at": "t", "not_identifiable": 1, "outcome": "MEASURED",
         "rows_used": 10, "variable_id": "var_a"}
    r = run(*world(tmp_path, terminal=t)[:4])
    assert "TERMINAL_SCHEMA" in kinds(r)


def test_a_fabricated_published_value_diverges(tmp_path):
    state, lake, cpath, digest, sha = world(tmp_path)
    rows = published_for("var_a", sha=sha, rel="features/src.parquet")
    rows["mean"]["value"] = 999.0
    r = run(state, lake, cpath, digest,
            published={"rows": {"var_a": rows}, "attempt": "t",
                       "duplicates": []})
    assert r["recomputation"]["verdict"] == V.RECOMPUTED_DIVERGES
    assert r["_comparisons"]["var_a"]["descriptors"]["mean"]["state"] == \
        V.DIVERGES


def test_a_missing_published_row_diverges(tmp_path):
    state, lake, cpath, digest, sha = world(tmp_path)
    rows = published_for("var_a", sha=sha, rel="features/src.parquet")
    del rows["std"]
    r = run(state, lake, cpath, digest,
            published={"rows": {"var_a": rows}, "attempt": "t",
                       "duplicates": []})
    probs = r["_comparisons"]["var_a"]["problems"]
    assert any(p["kind"] == "ROW_CORRESPONDENCE_DIVERGES" for p in probs)


def test_a_published_row_bound_to_other_bytes_diverges(tmp_path):
    state, lake, cpath, digest, sha = world(tmp_path)
    rows = published_for("var_a", sha="0" * 64, rel="features/src.parquet")
    r = run(state, lake, cpath, digest,
            published={"rows": {"var_a": rows}, "attempt": "t",
                       "duplicates": []})
    assert any(p["kind"] == "PUBLISHED_SOURCE_DIGEST_DIVERGES"
               for p in r["_comparisons"]["var_a"]["problems"])


# ----------------------------------------- C69.3 another appearance/file
def test_a_terminal_pointing_at_another_variables_appearance_diverges(
        tmp_path):
    payload = parquet_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    apps = [{"appearance_id": "app_0", "relative_path": "features/src.parquet",
             "physical_sha256": sha, "size_bytes": len(payload)},
            {"appearance_id": "app_1", "relative_path": "features/src.parquet",
             "physical_sha256": sha, "size_bytes": len(payload)}]
    variables = [{"variable_id": "var_a", "appearances": ["app_0"],
                  "concept_name": "close", "entity": "e"},
                 {"variable_id": "var_b", "appearances": ["app_1"],
                  "concept_name": "close", "entity": "e"}]
    t = {"appearance": "app_1", "batch": "b", "concept_name": "close",
         "descriptors": 25, "entity": "e", "measured_at": "t",
         "not_identifiable": 1, "outcome": "MEASURED", "rows_used": 10,
         "variable_id": "var_a"}
    state, lake, cpath, digest, _ = world(tmp_path, variables=variables,
                                          appearances=apps)
    (state / "terminals" / tname("var_a")).write_text(json.dumps(t))
    r = run(state, lake, cpath, digest)
    assert "APPEARANCE_NOT_OWNED_BY_VARIABLE" in kinds(r)
    assert "var_a" not in r["_bound"]


def test_a_source_whose_bytes_differ_from_the_census_diverges(tmp_path):
    state, lake, cpath, digest, _ = world(tmp_path)
    (lake / "features" / "src.parquet").write_bytes(
        parquet_bytes(SERIES + 1))
    r = run(state, lake, cpath, digest)
    assert "SOURCE_DIGEST_OR_SIZE_DIVERGES" in kinds(r)


# -------------------------------------- C69.4 containment of the source
@pytest.mark.parametrize("case", ["absolute", "traversal", "symlink",
                                  "symlinked_dir"])
def test_a_source_outside_the_lake_is_never_read(tmp_path, case):
    outside = tmp_path / "outside.parquet"
    secret = parquet_bytes(SERIES * 7)
    outside.write_bytes(secret)
    sha = hashlib.sha256(secret).hexdigest()
    rel = {"absolute": str(outside), "traversal": "../outside.parquet",
           "symlink": "features/link.parquet",
           "symlinked_dir": "linkdir/outside.parquet"}[case]
    apps = [{"appearance_id": "app_0", "relative_path": rel,
             "physical_sha256": sha, "size_bytes": len(secret)}]
    state, lake, cpath, digest, _ = world(tmp_path, appearances=apps)
    if case == "symlink":
        os.symlink(outside, lake / "features" / "link.parquet")
    if case == "symlinked_dir":
        os.symlink(tmp_path, lake / "linkdir")
    r = run(state, lake, cpath, digest)
    assert "SOURCE_OUTSIDE_ROOT_OR_LINK" in kinds(r)
    assert "var_a" not in r["_bound"]


# ------------------------------------------- C69.5 schema of a terminal
@pytest.mark.parametrize("mutate", [
    lambda t: t.update(zzz_extra=1),
    lambda t: t.update(rows_used="many"),
    lambda t: t.update(outcome="BANANA"),
    lambda t: t.update(descriptors=True),
    lambda t: t.pop("appearance"),
])
def test_a_malformed_terminal_diverges(tmp_path, mutate):
    t = {"appearance": "app_0", "batch": "b", "concept_name": "close",
         "descriptors": 25, "entity": "e", "measured_at": "t",
         "not_identifiable": 1, "outcome": "MEASURED", "rows_used": 10,
         "variable_id": "var_a"}
    mutate(t)
    r = run(*world(tmp_path, terminal=t)[:4])
    assert r["population"]["verdict"] == V.POPULATION_DIVERGES
    assert "TERMINAL_SCHEMA" in kinds(r)


def test_a_terminal_whose_identity_contradicts_the_census_diverges(
        tmp_path):
    t = {"appearance": "app_0", "batch": "b", "concept_name": "open",
         "descriptors": 25, "entity": "e", "measured_at": "t",
         "not_identifiable": 1, "outcome": "MEASURED", "rows_used": 10,
         "variable_id": "var_a"}
    r = run(*world(tmp_path, terminal=t)[:4])
    assert "TERMINAL_CENSUS_IDENTITY_MISMATCH" in kinds(r)


# --------------------------------------- C69.6 substituted population
def test_a_self_consistent_substituted_population_refuses(tmp_path):
    state, lake, cpath, digest, _ = world(tmp_path)
    with pytest.raises(V.VerificationRefusal) as e:
        V.verify(state, lake, cpath, expected_census_sha256="a" * 64)
    assert "substituted population" in str(e.value)


def test_verification_without_an_expected_authority_refuses(tmp_path):
    state, lake, cpath, digest, _ = world(tmp_path)
    with pytest.raises(V.VerificationRefusal):
        V.verify(state, lake, cpath, expected_census_sha256="")


def test_a_ledger_with_an_extra_key_refuses(tmp_path):
    state, lake, cpath, digest, _ = world(tmp_path)
    led = json.loads((state / "PRE_LEDGER.json").read_text())
    led["smuggled"] = True
    (state / "PRE_LEDGER.json").write_text(json.dumps(led))
    with pytest.raises(V.VerificationRefusal) as e:
        run(state, lake, cpath, digest)
    assert "undeclared keys" in str(e.value)


# -------------------------------------------- not identifiable terminals
def test_a_not_identifiable_terminal_stays_producer_declared(tmp_path):
    variables = [{"variable_id": "var_t", "appearances": ["app_0"],
                  "concept_name": "timestamp", "entity": "e"}]
    state, lake, cpath, digest, sha = world(tmp_path, variables=variables,
                                            ni=True)
    pub = {"rows": {"var_t": {"characterization_disposition": {
        "value": None, "value_text": "NOT_IDENTIFIABLE",
        "identifiable": False, "source_id": "NO_APPEARANCE",
        "source_sha256": digest, "window_contract": None}}},
        "attempt": "t", "duplicates": []}
    r = run(state, lake, cpath, digest, published=pub)
    c = r["_comparisons"]["var_t"]
    assert c["layer"] == V.PRODUCER_DECLARED
    assert c["axis_claim"] == "TEMPORAL_AXIS_NAME"
    assert r["population"]["producer_declared_outcomes"][
        "PRODUCER_DECLARED_NOT_IDENTIFIABLE"] == 1


# ----------------------------------------------------- v3 additivity
def test_v3_is_written_beside_v1_and_v2_which_stay_byte_intact(tmp_path):
    state, lake, cpath, digest, sha = world(tmp_path)
    (state / "terminals_v2").mkdir()
    (state / "terminals_v2" / tname("var_a")).write_text('{"v": 2}')
    before = {d: V.tree_content_digest(state / d)
              for d in ("terminals", "terminals_v2")}
    pub = {"rows": {"var_a": published_for("var_a", sha=sha,
                                           rel="features/src.parquet")},
           "attempt": "t", "duplicates": []}
    r = run(state, lake, cpath, digest, published=pub)
    idx = V.supersede_v3(state, r, superseded_at="t",
                         code_identity={"x": "y"})
    assert idx["v3_terminals_written"] == 1
    assert {d: V.tree_content_digest(state / d) for d in before} == before
    body = json.loads((state / "terminals_v3" / tname("var_a")).read_text())
    assert body["layer"] == V.INDEPENDENTLY_RECOMPUTED
    with pytest.raises(V.VerificationRefusal):
        V.supersede_v3(state, r, superseded_at="t", code_identity={})


def test_v3_refuses_without_a_recomputation(tmp_path):
    r = run(*world(tmp_path)[:4])
    with pytest.raises(V.VerificationRefusal):
        V.supersede_v3(tmp_path / "state", r, superseded_at="t",
                       code_identity={})


def test_the_verifier_imports_no_producer_code():
    tree = ast.parse((REPO / "tools/verify_lake_terminals.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    assert not any("characteriz" in m for m in mods), mods
