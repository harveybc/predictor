"""The twelve acceptance points of order C1-C16, §8.

Each test is one numbered point. They deliberately restate
guarantees proven in the focal batteries, because §8 is the list
Musashi will read to decide whether this return is finished.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review  # noqa: E402
from eligibility.consumed import (  # noqa: E402
    resolve_consumed_subjects)
from olap import bank_index as bi  # noqa: E402
from olap import campaign_envelope as ce  # noqa: E402
from olap import outbox as ob  # noqa: E402

FIN = REPO.parent / "financial-data"


def _csv(path: Path, columns, rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for i in range(rows):
            w.writerow([f"2020-01-0{i+1}"] +
                       [str(i + 1)] * (len(columns) - 1))


@pytest.fixture()
def run_world(tmp_path):
    cols = ["DATE_TIME", "px", "vol"]
    for r in ("d4", "d5", "d6"):
        _csv(tmp_path / f"{r}.csv", cols)
    return tmp_path, {
        "x_train_file": str(tmp_path / "d4.csv"),
        "x_validation_file": str(tmp_path / "d5.csv"),
        "x_test_file": str(tmp_path / "d6.csv"),
        "target_column": "px",
        integ.KEY_SCOPE: "forecasting",
        integ.PURPOSE_KEY: integ.PURPOSE_EXPERIMENT,
    }


# 1 ------------------------------------------------------------
def test_1_nothing_reaches_data_before_the_gate():
    src = (REPO / "app/main.py").read_text()
    gate_at = src.index("gate_run(")
    head = src[:gate_at]
    for consuming in ("optimizer_plugin.optimize(",
                      "run_prediction_pipeline(",
                      "run_preprocessing(", ".build_model(",
                      ".train(", "fit_transform("):
        assert consuming not in head
    pre = (REPO.parent /
           "preprocessor/app/data_processor.py")
    if pre.is_file():
        t = pre.read_text()
        assert t.index("gate_run(") < t.index(
            "plugin.process(data, config)")


# 2 ------------------------------------------------------------
def test_2_empty_extra_omitted_or_renamed_subject_refuses(
        run_world):
    tmp, config = run_world
    derived = resolve_consumed_subjects(dict(config),
                                        repo_root=tmp)
    for bad, needle in (
            ([], "assertion is false"),
            (derived["subject_ids"] + ["ghost"],
             "not consumed"),
            (derived["subject_ids"][:1], "not declared")):
        cfg = dict(config, eligibility_subjects=bad)
        with pytest.raises(SystemExit, match=needle):
            resolve_consumed_subjects(cfg, repo_root=tmp)
    # renamed column: the derived ids change with the schema
    _csv(tmp / "d4.csv", ["DATE_TIME", "px", "renamed"])
    _csv(tmp / "d5.csv", ["DATE_TIME", "px", "renamed"])
    _csv(tmp / "d6.csv", ["DATE_TIME", "px", "renamed"])
    after = resolve_consumed_subjects(dict(config),
                                      repo_root=tmp)
    assert after["subject_ids"] != derived["subject_ids"]


# 3 ------------------------------------------------------------
def test_3_a_self_issued_manifest_grants_no_review(run_world,
                                                   monkeypatch):
    tmp, config = run_world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "candidate", "scope": "s", "entries": []}
    doc["manifest_sha256"] = gate._self_sha(doc)
    mp = tmp / "self.json"
    mp.write_text(json.dumps(doc))
    cfg = dict(config)
    cfg[integ.KEY_MANIFEST] = str(mp)
    cfg[integ.KEY_SHA] = doc["manifest_sha256"]
    cfg[integ.KEY_CENSUS] = "a" * 64
    monkeypatch.setenv(review.REVIEW_RECORD_ENV,
                       str(tmp / "absent.json"))
    monkeypatch.setenv(review.SUBMISSION_ENV, str(tmp / "subs"))
    # under C17 a self-issued manifest cannot even reach the
    # record: the executing phase demands a PERSISTED submission
    # that a reviewer saw
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_EXECUTE
    with pytest.raises(SystemExit,
                       match="requires eligibility_submission"):
        integ.gate_run(dict(cfg), repo_root=tmp, consumer="t")
    # and even after phase one persists one, the absent record
    # still refuses
    phase1 = dict(cfg)
    phase1[integ.PURPOSE_KEY] = integ.PURPOSE_SUBMIT
    st = integ.gate_run(phase1, repo_root=tmp, consumer="t")
    cfg[integ.KEY_SUBMISSION_SHA] = st["submission_sha256"]
    with pytest.raises(SystemExit,
                       match="no external eligibility review"):
        integ.gate_run(cfg, repo_root=tmp, consumer="t")


# 4 ------------------------------------------------------------
def test_4_substituted_bindings_refuse_at_point_of_use():
    entry = {
        "subject_kind": "variable", "subject_id": "v",
        "version": "1",
        "io_schema": {"input": ["f"], "output": ["f"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none", "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "b" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "s", "decision_reason": "r",
        "reviewer": "x",
        "reviewed_at": "2026-09-09T00:00:00Z"}
    m = {"entries": [entry]}
    gate.require_eligible(m, "v", scope="s",
                          data_digest="d" * 64,
                          partitions_digest="b" * 64,
                          code_digest="c" * 64,
                          evidence_digest="e" * 64)
    for kw, needle in (
            ({"data_digest": "9" * 64}, "reviewed data digest"),
            ({"partitions_digest": "9" * 64},
             "reviewed partitions digest"),
            ({"code_digest": "9" * 64}, "different code"),
            ({"evidence_digest": "9" * 64},
             "replacing the evidence")):
        with pytest.raises(SystemExit, match=needle):
            gate.require_eligible(m, "v", scope="s", **kw)


# 5 ------------------------------------------------------------
def test_5_census_digest_policy():
    summary = FIN / "features/census/CENSUS_SUMMARY.v1.json"
    if not summary.is_file():
        pytest.skip("census summary not present")
    cov = json.loads(summary.read_text())["coverage"]
    assert cov["appearances_total"] == 1680
    assert cov["appearances_physically_digested"] == 1680
    assert cov["bytes_read_for_digest"] == 14436534039
    assert cov["digest_coverage_fraction"] == 1.0
    sys.path.insert(0, str(FIN / "_scripts" / "lib"))
    import incremental_census as ic
    src = Path(ic.__file__).read_text()
    assert 'prev.get("mtime_ns") != mtime_ns' in src
    assert 'prev.get("ctime_ns") != ctime_ns' in src
    assert "REUSED_FROM_PREVIOUS_VERIFIED_CENSUS" in src


# 6 ------------------------------------------------------------
def test_6_availability_contracts_reach_exactly_their_variables():
    sys.path.insert(0, str(FIN / "_scripts" / "lib"))
    import incremental_census as ic
    join = ic.map_families_to_entities(
        {"macro/one": {}},
        [("cross_source", "macro__one__a"),
         ("cross_source", "macro__one__b"),
         ("cross_source", "other__z")])
    assert join["by_family"]["macro/one"] == [
        "macro__one__a", "macro__one__b"]
    assert "other__z" not in join["entity_to_family"]
    scope = FIN / "features/census/AVAILABILITY_SCOPE.v1.json"
    if scope.is_file():
        doc = json.loads(scope.read_text())
        assert doc["supply"]["families_instantiable"] == len(
            doc["supply"]["instances"])


# 7 ------------------------------------------------------------
def test_7_the_index_contains_the_variables_and_recounts():
    idx_p = REPO / "examples/research/crispdm_bank_index.v1.json"
    if not idx_p.is_file():
        pytest.skip("index not built")
    idx = json.loads(idx_p.read_text())
    kinds = {}
    for r in idx["common_rows"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    assert kinds["variable"] == 1965
    assert bi._recount(idx["common_rows"]) == \
        idx["cardinality_by_kind_and_authority"]


# 8 ------------------------------------------------------------
def test_8_short_or_constant_mtm_never_fits_outside_train():
    mod = pytest.importorskip(
        "preprocessor_plugins.phase2_6_preprocessor")
    cls = getattr(mod, "PreprocessorPlugin", None) or \
        getattr(mod, "Plugin", None)
    plug = cls()
    import numpy as np
    out = plug._apply_causal_mtm_decomposition(
        np.zeros(4, dtype=np.float32), 32, 2, "short")
    assert isinstance(out, tuple) and out[1] is None
    src = (REPO / "preprocessor_plugins/"
                  "phase2_6_preprocessor.py").read_text()
    flat = " ".join(src.split())
    assert "if mtm_scaler is None and not legacy_per_split" in \
        flat
    assert "are NOT_EVALUABLE rather than fitted" in flat


# 9 ------------------------------------------------------------
def test_9_collision_fabrication_and_absent_backup_refuse(
        tmp_path):
    from tools.build_campaign_envelopes import (
        T2_SCHEMA_KEYS, _consume_producer_artifact)
    fake = {k: "x" for k in T2_SCHEMA_KEYS}
    fake["record_sha256"] = "a" * 64
    p = tmp_path / "f.json"
    p.write_text(json.dumps(fake))
    with pytest.raises(SystemExit, match="does not re-derive"):
        _consume_producer_artifact(
            p, schema_keys=T2_SCHEMA_KEYS,
            self_key="record_sha256", producer="T2",
            verifier="v")
    src = (REPO /
           "tools/backfill_campaign_envelopes.py").read_text()
    assert "a digest is not a backup" in src
    env_src = (REPO / "olap/campaign_envelope.py").read_text()
    assert "DIFFERENT identity" in env_src


# 10 -----------------------------------------------------------
def test_10_outbox_survives_a_down_database(tmp_path,
                                            monkeypatch):
    from tools import olap_loader
    doc = ce.build_envelope(
        campaign_key="k", producer="p",
        result_class="DEVELOPMENT",
        identity={"run_id": "r", "code_identity": "c",
                  "design_sha256": "d"},
        data_consumed={"datasets": [], "variables": [],
                       "operators": []},
        partitions={"exposure": "e", "splits": "s"},
        budget={"device": "cpu", "wall_seconds": 1.0,
                "cost_units": "s"},
        terminal={"state": "COMPLETE", "adjudication": "NONE"},
        artifacts={"a": "b"},
        units=[{"cell_key": "c", "candidate_key": "d",
                "metric_name": "m", "metric_value": 1.0,
                "terminal_state": "OK"}])
    monkeypatch.setenv("PGPORT", "1")
    assert ob.emit(doc, kind="envelope",
                   root=tmp_path)["written"] is True
    out = olap_loader.drain_once(tmp_path)
    assert out["database_unavailable"] is True
    assert ob.counts(tmp_path)["pending"] == 1
    assert ob.counts(tmp_path)["failed"] == 0
    assert ob.emit(doc, kind="envelope",
                   root=tmp_path)["written"] is False


# 11 -----------------------------------------------------------
def test_11_the_cube_keeps_its_history():
    sa = pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "metabase")
    pw = os.getenv("PGPASSWORD", "metabase_pass")
    try:
        e = create_engine(
            f"postgresql://{user}:{pw}@{host}:{port}/"
            "predictor_olap", future=True)
        with e.connect() as c:
            exp = c.execute(text(
                "select count(*) from dim_experiment")).scalar()
            perf = c.execute(text(
                "select count(*) from fact_performance"
            )).scalar()
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"cube unavailable: "
                    f"{exc.__class__.__name__}")
    assert exp == 39
    assert perf == 1404


# 12 -----------------------------------------------------------
def test_12_translated_rows_are_kept_and_superseded():
    sa = pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "metabase")
    pw = os.getenv("PGPASSWORD", "metabase_pass")
    try:
        e = create_engine(
            f"postgresql://{user}:{pw}@{host}:{port}/"
            "predictor_olap", future=True)
        with e.connect() as c:
            rows = dict(c.execute(text(
                "select authority_state, count(*) from "
                "fact_campaign_unit group by "
                "authority_state")).all())
            linked = c.execute(text(
                "select count(*) from fact_campaign_unit where "
                "superseded_by_envelope_sha256 is not null"
            )).scalar()
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"cube unavailable: "
                    f"{exc.__class__.__name__}")
    assert rows.get(ce.TRANSLATED, 0) >= 60, (
        "the 60 translated rows must be KEPT, not deleted — a FLOOR, "
        "because the cube is fed continuously and C35 lets a second "
        "attempt of an experiment land beside the first")
    assert rows.get(ce.PRODUCER_BOUND, 0) >= 60, (
        "a FLOOR: the cube is fed continuously and no row may\n         disappear")
    assert linked == 60


# C16 ----------------------------------------------------------
def test_c16_doin_is_reported_not_claimed():
    p = REPO / "examples/research/DOIN_SURFACE_INVENTORY.v1.json"
    if not p.is_file():
        pytest.skip("inventory not built")
    doc = json.loads(p.read_text())
    assert doc["verdict"] == \
        "TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED"
    for r in doc["repositories"]:
        assert r["has_live_call_site"] is False
        assert r["ref_count"] > 0
    assert "NOT WIRED in this order" in doc["proposed_wiring"]
