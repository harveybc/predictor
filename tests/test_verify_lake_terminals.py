"""C59-C61 battery: the verifier must catch what the producer cannot.

Every test builds a fixture state directory. The real terminals under
~/.local/state are never touched, never copied for writing and never
opened for writing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_lake_terminals as V  # noqa: E402


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def name_for(vid: str) -> str:
    return hashlib.sha256(vid.encode()).hexdigest()[:32] + ".json"


def build(tmp_path, *, ids, terminals=None, census_extra=None,
          ledger_extra=None, source_rows=b"a,b\n1,2\n"):
    lake = tmp_path / "lake"
    lake.mkdir()
    (lake / "src.csv").write_bytes(source_rows)

    appearances = [{"appearance_id": f"app_{i}",
                    "relative_path": "src.csv"}
                   for i, _ in enumerate(ids)]
    census = {"census_sha256": "PLACEHOLDER",
              "appearances": appearances,
              "variables": [{"variable_id": v,
                             "appearances": [f"app_{i}"]}
                            for i, v in enumerate(ids)]}
    census.update(census_extra or {})
    digest = sha256(json.dumps(census, sort_keys=True).encode())
    census["census_sha256"] = digest
    cdir = tmp_path / "census"
    cdir.mkdir()
    cpath = cdir / f"census-{digest}.json"
    cpath.write_text(json.dumps(census, sort_keys=True))

    state = tmp_path / "state"
    (state / "terminals").mkdir(parents=True)
    ledger = {"schema": "crispdm.lake_characterization_pre_ledger.v1",
              "identities": list(ids),
              "conceptual_variables": len(ids),
              "physical_appearances": len(appearances),
              "census_sha256": digest,
              "pre_ledger_sha256": "declared-by-the-producer"}
    ledger.update(ledger_extra or {})
    (state / "PRE_LEDGER.json").write_text(json.dumps(ledger))

    for vid, doc in (terminals if terminals is not None
                     else [(v, {"variable_id": v, "outcome": "MEASURED",
                                "entity": "e"}) for v in ids]):
        (state / "terminals" / name_for(vid)).write_text(
            json.dumps(doc))
    return state, lake, cpath


IDS = ["var_a", "var_b", "var_c"]


def run(state, lake, census, **kw):
    return V.verify(state, lake, census, **kw)


# ------------------------------------------------------------ passing
def test_a_complete_consistent_set_verifies_exact(tmp_path):
    r = run(*build(tmp_path, ids=IDS))
    assert r["verdict"] == "TERMINALS_VERIFIED_EXACT"
    assert r["terminals"]["read"] == 3
    assert r["terminals"][
        "naming_rules_consistent_with_every_file"] == ["sha256[:32]"]
    assert r["sources"]["distinct_files_digested"] == 1


def test_the_source_bytes_are_digested_by_the_verifier(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS,
                                source_rows=b"x,y\n7,8\n")
    r = run(state, lake, census)
    (d,) = r["sources"]["digests"].values()
    assert d["sha256"] == sha256(b"x,y\n7,8\n")


def test_descriptors_are_retained_and_all_released(tmp_path):
    r = run(*build(tmp_path, ids=IDS))
    assert r["custody"]["retained_descriptors"] == 2  # root + terminals


def test_no_physical_path_is_published(tmp_path):
    r = run(*build(tmp_path, ids=IDS))
    blob = json.dumps({k: v for k, v in r.items()
                       if not k.startswith("_")})
    assert str(tmp_path) not in blob
    assert "WITHHELD" in blob


# ------------------------------------------------------------ refusing
def test_a_missing_terminal_is_reported_not_ignored(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (state / "terminals" / name_for("var_c")).unlink()
    r = run(state, lake, census)
    assert r["verdict"] == "TERMINALS_DIVERGE"
    assert r["terminals"]["missing"] == ["var_c"]


def test_an_extra_terminal_is_reported(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (state / "terminals" / name_for("var_z")).write_text(
        json.dumps({"variable_id": "var_z", "outcome": "MEASURED"}))
    r = run(state, lake, census)
    assert r["verdict"] == "TERMINALS_DIVERGE"
    assert r["terminals"]["extra"] == ["var_z"]
    assert r["terminals"]["unknown_variable_ids"] == ["var_z"]


def test_a_terminal_filed_under_another_variables_name_is_caught(
        tmp_path):
    """The exact attack a name-based store invites."""
    state, lake, census = build(tmp_path, ids=IDS)
    p = state / "terminals" / name_for("var_a")
    p.write_text(json.dumps({"variable_id": "var_b",
                             "outcome": "MEASURED"}))
    r = run(state, lake, census)
    assert r["verdict"] == "TERMINALS_DIVERGE"
    assert r["terminals"]["naming_rule_state"] == \
        "NO_SINGLE_RULE_EXPLAINS_EVERY_FILENAME"


def test_a_census_that_is_not_the_ledgers_census_refuses(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (tmp_path / "other").mkdir()
    other = build(tmp_path / "other", ids=["var_q"])[2]
    with pytest.raises(V.VerificationRefusal) as e:
        run(state, lake, other)
    assert "NOT the census" in str(e.value)


def test_a_census_filed_under_the_wrong_name_refuses(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    renamed = census.parent / ("census-" + "f" * 64 + ".json")
    census.rename(renamed)
    with pytest.raises(V.VerificationRefusal) as e:
        run(state, lake, renamed)
    assert "filed under" in str(e.value)


def test_a_duplicate_identity_in_the_ledger_refuses(tmp_path):
    state, lake, census = build(
        tmp_path, ids=IDS,
        ledger_extra={"identities": ["var_a", "var_a", "var_b"]})
    with pytest.raises(V.VerificationRefusal) as e:
        run(state, lake, census)
    assert "duplicate identity" in str(e.value)


def test_a_ledger_whose_count_contradicts_its_list_refuses(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS,
                                ledger_extra={"conceptual_variables": 9})
    with pytest.raises(V.VerificationRefusal) as e:
        run(state, lake, census)
    assert "conceptual variables" in str(e.value)


def test_an_unreadable_terminal_is_reported_not_skipped(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (state / "terminals" / name_for("var_a")).write_text("{not json")
    r = run(state, lake, census)
    assert r["verdict"] == "TERMINALS_DIVERGE"
    assert r["terminals"]["unreadable"]


def test_an_absent_source_is_reported_not_assumed_present(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (lake / "src.csv").unlink()
    r = run(state, lake, census)
    assert r["sources"]["absent_count"] == 1
    assert r["sources"]["absent"][0]["why"] == "the file is gone"


# ------------------------------------------------- C60-C61 additivity
def _supersede(state, r, **kw):
    return V.supersede(
        state, r, v2_dirname=kw.get("v2", "terminals_v2"),
        window_contract={"definition": "full series"},
        code_identity={"verifier": "test"},
        superseded_at="2026-09-12T00:00:00Z")


def test_v2_is_written_beside_v1_which_is_untouched(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    v1 = {p.name: (p.stat().st_ino, p.read_bytes())
          for p in (state / "terminals").iterdir()}
    idx = _supersede(state, run(state, lake, census))
    assert idx["v2_terminals_written"] == 3
    assert idx["v1_terminals_unchanged"] == 3
    after = {p.name: (p.stat().st_ino, p.read_bytes())
             for p in (state / "terminals").iterdir()}
    assert after == v1, "v1 must be byte-for-byte and inode identical"


def test_every_v2_terminal_carries_the_four_missing_facts(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    _supersede(state, run(state, lake, census))
    for p in (state / "terminals_v2").iterdir():
        d = json.loads(p.read_text())
        assert d["schema"] == V.V2_SCHEMA
        assert d["source"]["sha256"] == sha256(b"a,b\n1,2\n")
        assert d["window_contract_sha256"]
        assert d["terminal_sha256"]
        assert d["supersedes"]["unchanged"] is True


def test_an_edited_v2_terminal_stops_matching_its_own_digest(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    _supersede(state, run(state, lake, census))
    p = next((state / "terminals_v2").iterdir())
    d = json.loads(p.read_text())
    recorded = d.pop("terminal_sha256")
    assert V.sha_obj(d) == recorded
    d["rows_used"] = 999_999
    assert V.sha_obj(d) != recorded


def test_a_second_supersession_refuses_rather_than_overwriting(
        tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    r = run(state, lake, census)
    _supersede(state, r)
    with pytest.raises(V.VerificationRefusal) as e:
        _supersede(state, r)
    assert "written once" in str(e.value)


def test_a_divergent_set_is_never_superseded(tmp_path):
    state, lake, census = build(tmp_path, ids=IDS)
    (state / "terminals" / name_for("var_c")).unlink()
    r = run(state, lake, census)
    assert r["verdict"] == "TERMINALS_DIVERGE"
    with pytest.raises(V.VerificationRefusal):
        V.main(["--state-dir", str(state), "--lake-root", str(lake),
                "--census", str(census), "--supersede"])


# ------------------------------------------------------- independence
def test_the_verifier_imports_nothing_from_the_producer():
    import ast
    src = (ROOT / "tools/verify_lake_terminals.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("characterize" in m or "run_characterization" in m
                   for m in imported), sorted(imported)
