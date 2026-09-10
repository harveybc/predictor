"""C1-C5: the gate after the audit.

Every test here corresponds to a defect Musashi reproduced from
the shipped code. A green suite means the defect can no longer
occur silently; it does not mean the gate is finished.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review  # noqa: E402
from eligibility import strict  # noqa: E402
from eligibility.consumed import (  # noqa: E402
    resolve_consumed_subjects, subject_id)

COLUMNS = ["DATE_TIME", "typical_price", "volume"]


def _csv(path: Path, columns=None, rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or COLUMNS
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for i in range(rows):
            w.writerow([f"2020-01-0{i+1}"] +
                       [str(i + 1)] * (len(columns) - 1))


@pytest.fixture()
def world(tmp_path):
    """A run whose contract names three real partitions."""
    for role in ("d4", "d5", "d6"):
        _csv(tmp_path / "data" / f"{role}.csv")
    config = {
        "x_train_file": str(tmp_path / "data/d4.csv"),
        "x_validation_file": str(tmp_path / "data/d5.csv"),
        "x_test_file": str(tmp_path / "data/d6.csv"),
        "target_column": "typical_price",
        integ.KEY_SCOPE: "forecasting",
        integ.PURPOSE_KEY: integ.PURPOSE_EXPERIMENT,
    }
    return tmp_path, config


def _entry(sid, digests, **over):
    e = {
        "subject_kind": "variable", "subject_id": sid,
        "version": "1",
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none", "parameters": {},
        "digests": digests,
        "measured_cost": {"fit_seconds": 0.0},
        "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "forecasting",
        "decision_reason": "reviewed against the census",
        "reviewer": "external-reviewer-double",
        "reviewed_at": "2026-09-09T00:00:00Z",
    }
    e.update(over)
    return e


def _manifest(tmp_path, consumed, **over):
    d = consumed["digests"]
    digests = {"data": d["data"], "code": "c" * 64,
               "partitions": d["partitions"],
               "evidence": "e" * 64}
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "external-reviewer-double",
           "scope": "forecasting",
           "entries": [_entry(sid, dict(digests))
                       for sid in consumed["subject_ids"]]}
    doc.update(over)
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(doc))
    return p, doc


def _install_review(tmp_path, monkeypatch, submission,
                    census_sha, scope="forecasting", **over):
    rec = {
        "schema": review.REVIEW_RECORD_SCHEMA,
        "reviewed_at": "2026-09-10T12:00:00Z",
        "reviewer": "external-reviewer-double",
        "scope": scope,
        "reviewed_submission_sha256":
            submission["submission_sha256"],
        "reviewed_manifest_sha256":
            submission["manifest_sha256"],
        "reviewed_census_sha256": census_sha,
        "reviewed_code_digest": submission["code_digest"],
        "reviewed_partitions_digest":
            submission["partitions_digest"],
        "decision": review.REVIEW_DECISION,
    }
    rec.update(over)
    rec["record_sha256"] = review._self_sha(rec, "record_sha256")
    p = tmp_path / "REVIEW_RECORD.json"
    p.write_text(json.dumps(rec))
    monkeypatch.setenv(review.REVIEW_RECORD_ENV, str(p))
    return p, rec


CENSUS = "a" * 64


def _gated(tmp_path, config, monkeypatch, submitted_at=None,
           consumer="t"):
    """Build the submission EXACTLY as gate_run will, so the
    review record under test binds the same bytes."""
    consumed = resolve_consumed_subjects(config,
                                         repo_root=tmp_path)
    mp, _ = _manifest(tmp_path, consumed)
    config[integ.KEY_MANIFEST] = str(mp)
    config[integ.KEY_CENSUS] = CENSUS
    sub = review.build_submission(
        submitted_at=submitted_at or "2026-09-10T10:00:00Z",
        submitter=consumer, scope="forecasting",
        manifest_sha256=json.loads(
            mp.read_text())["manifest_sha256"],
        census_sha256=CENSUS, consumed=consumed)
    return consumed, mp, sub


# ==============================================================
# C1: order of execution
# ==============================================================

def test_gate_precedes_the_optimizer_and_the_pipeline():
    src = (REPO / "app/main.py").read_text()
    gate_at = src.index("gate_run(")
    opt_at = src.index("optimizer_plugin.optimize(")
    pipe_at = src.index("pipeline_plugin.run_prediction_pipeline(")
    assert gate_at < opt_at, (
        "the optimizer still runs before the gate")
    assert gate_at < pipe_at


def test_no_data_consuming_call_precedes_the_gate():
    """Class loading may precede the gate; consuming data may
    not."""
    src = (REPO / "app/main.py").read_text()
    gate_at = src.index("gate_run(")
    head = src[:gate_at]
    for forbidden in ("optimizer_plugin.optimize(",
                      "run_prediction_pipeline(",
                      "run_preprocessing(",
                      ".build_model(", ".train(",
                      "fit_transform("):
        assert forbidden not in head, (
            f"{forbidden} is reached before the gate")


# ==============================================================
# C2: the universe is derived, not declared
# ==============================================================

def test_subjects_are_derived_from_the_consumed_headers(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(config, repo_root=tmp)
    assert consumed["subject_ids"] == sorted([
        subject_id(consumed["dataset_id"], "typical_price"),
        subject_id(consumed["dataset_id"], "volume")])
    assert [c["column"] for c in consumed["excluded_columns"]] \
        == ["DATE_TIME"]
    assert consumed["digests"]["data"] != \
        consumed["digests"]["partitions"]


def test_an_empty_declared_list_grants_nothing(world,
                                               monkeypatch,
                                               tmp_path):
    tmp, config = world
    config["eligibility_subjects"] = []
    with pytest.raises(SystemExit,
                       match="assertion is false"):
        resolve_consumed_subjects(config, repo_root=tmp)


def test_a_declared_list_is_only_an_assertion(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(config, repo_root=tmp)
    # exact match passes
    config["eligibility_subjects"] = list(
        consumed["subject_ids"])
    resolve_consumed_subjects(config, repo_root=tmp)
    # an extra name refuses
    config["eligibility_subjects"] = list(
        consumed["subject_ids"]) + ["var.ghost"]
    with pytest.raises(SystemExit, match="not consumed"):
        resolve_consumed_subjects(config, repo_root=tmp)
    # an omitted name refuses
    config["eligibility_subjects"] = consumed["subject_ids"][:1]
    with pytest.raises(SystemExit, match="not declared"):
        resolve_consumed_subjects(config, repo_root=tmp)


def test_partitions_with_different_schemas_refuse(world):
    tmp, config = world
    _csv(tmp / "data/d6.csv", columns=COLUMNS + ["extra"])
    with pytest.raises(SystemExit,
                       match="partition schemas differ"):
        resolve_consumed_subjects(config, repo_root=tmp)


def test_duplicate_or_unnamed_columns_refuse(world):
    tmp, config = world
    for role in ("d4", "d5", "d6"):
        _csv(tmp / "data" / f"{role}.csv",
             columns=["DATE_TIME", "px", "px"])
    with pytest.raises(SystemExit, match="duplicate columns"):
        resolve_consumed_subjects(config, repo_root=tmp)
    for role in ("d4", "d5", "d6"):
        _csv(tmp / "data" / f"{role}.csv",
             columns=["DATE_TIME", "px", " "])
    with pytest.raises(SystemExit, match="unnamed column"):
        resolve_consumed_subjects(config, repo_root=tmp)


def test_a_missing_partition_file_refuses(world):
    tmp, config = world
    (tmp / "data/d5.csv").unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        resolve_consumed_subjects(config, repo_root=tmp)


def test_a_run_with_no_partitions_refuses():
    with pytest.raises(SystemExit,
                       match="declares no input partitions"):
        resolve_consumed_subjects({}, repo_root=REPO)


# ==============================================================
# C3: separated authority
# ==============================================================

def test_a_self_issued_manifest_no_longer_grants(world,
                                                 monkeypatch):
    tmp, config = world
    consumed, mp, _ = _gated(tmp, config, monkeypatch)
    monkeypatch.delenv(review.REVIEW_RECORD_ENV, raising=False)
    monkeypatch.setattr(review, "DEFAULT_REVIEW_RECORD",
                        tmp / "absent.json")
    with pytest.raises(SystemExit,
                       match="no external eligibility review"):
        integ.gate_run(config, repo_root=tmp, consumer="t")


def test_the_config_cannot_choose_the_review_record(world,
                                                    monkeypatch):
    tmp, config = world
    _gated(tmp, config, monkeypatch)
    config["eligibility_review_record"] = str(
        tmp / "my_own_record.json")
    monkeypatch.delenv(review.REVIEW_RECORD_ENV, raising=False)
    monkeypatch.setattr(review, "DEFAULT_REVIEW_RECORD",
                        tmp / "absent.json")
    with pytest.raises(SystemExit,
                       match="no external eligibility review"):
        integ.gate_run(config, repo_root=tmp, consumer="t")
    # the structural guarantee: the resolver takes NO arguments,
    # so no configuration value can reach it
    import inspect
    assert list(inspect.signature(
        review.review_record_path).parameters) == []


def test_a_complete_review_opens_the_gate(world, monkeypatch):
    tmp, config = world
    consumed, mp, sub = _gated(tmp, config, monkeypatch)
    _install_review(tmp, monkeypatch, sub, CENSUS)
    monkeypatch.setattr(review, "utc_now_stamp",
                        lambda: "2026-09-10T10:00:00Z")
    stamp = integ.gate_run(config, repo_root=tmp, consumer="t")
    assert stamp["eligibility_status"] == integ.STATUS_GATED
    assert stamp["subjects_reviewed"] == 2
    assert stamp["reviewer"] == "external-reviewer-double"
    assert "GATED by review" in integ.describe(stamp)


def test_a_review_of_other_bytes_refuses(world, monkeypatch):
    tmp, config = world
    consumed, mp, sub = _gated(tmp, config, monkeypatch)
    monkeypatch.setattr(review, "utc_now_stamp",
                        lambda: "2026-09-10T10:00:00Z")
    for field, value, needle in (
            ("reviewed_submission_sha256", "b" * 64,
             "DIFFERENT submission"),
            ("reviewed_manifest_sha256", "b" * 64,
             "different manifest"),
            ("reviewed_census_sha256", "b" * 64,
             "different physical census"),
            ("reviewed_code_digest", "b" * 64,
             "different code"),
            ("reviewed_partitions_digest", "b" * 64,
             "different partition layout"),
            ("scope", "other_scope", "never global")):
        _install_review(tmp, monkeypatch, sub, CENSUS,
                        **{field: value})
        with pytest.raises(SystemExit, match=needle):
            integ.gate_run(dict(config), repo_root=tmp,
                           consumer="t")


def test_a_review_dated_before_the_submission_refuses(
        world, monkeypatch):
    tmp, config = world
    consumed, mp, sub = _gated(tmp, config, monkeypatch,
                               submitted_at="2026-09-10T13:00:00Z")
    monkeypatch.setattr(review, "utc_now_stamp",
                        lambda: "2026-09-10T13:00:00Z")
    _install_review(tmp, monkeypatch, sub, CENSUS)
    with pytest.raises(SystemExit, match="chronology"):
        integ.gate_run(config, repo_root=tmp, consumer="t")


def test_the_shipped_record_template_grants_nothing(tmp_path,
                                                    monkeypatch):
    tpl = review.review_record_template()
    p = tmp_path / "tpl.json"
    p.write_text(json.dumps(tpl))
    monkeypatch.setenv(review.REVIEW_RECORD_ENV, str(p))
    fake_sub = {"submission_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "code_digest": "c" * 64,
                "partitions_digest": "d" * 64,
                "submitted_at": "2026-09-10T00:00:00Z"}
    with pytest.raises(SystemExit):
        review.read_review_record(submission=fake_sub,
                                  census_sha256=CENSUS,
                                  scope="forecasting")


def test_a_submission_grants_nothing_by_construction(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(config, repo_root=tmp)
    sub = review.build_submission(
        submitted_at="2026-09-10T00:00:00Z", submitter="s",
        scope="forecasting", manifest_sha256="a" * 64,
        census_sha256=CENSUS, consumed=consumed)
    assert "not a decision" in sub["grants_nothing"]
    review.verify_submission(sub)
    src = (REPO / "eligibility/review.py").read_text()
    assert "def build_review_record" not in src
    assert "def install_review_record" not in src


# ==============================================================
# C2/C3: bindings compared at the point of use
# ==============================================================

def test_substituted_data_or_partitions_refuse(world,
                                               monkeypatch):
    tmp, config = world
    consumed, mp, sub = _gated(tmp, config, monkeypatch)
    _install_review(tmp, monkeypatch, sub, CENSUS)
    monkeypatch.setattr(review, "utc_now_stamp",
                        lambda: "2026-09-10T10:00:00Z")
    doc = json.loads(mp.read_text())
    for field, needle in (("data", "reviewed data digest"),
                          ("partitions",
                           "reviewed partitions digest")):
        mutated = json.loads(json.dumps(doc))
        for e in mutated["entries"]:
            e["digests"][field] = "9" * 64
        mutated["manifest_sha256"] = gate._self_sha(mutated)
        mp.write_text(json.dumps(mutated))
        cfg = dict(config)
        cfg[integ.KEY_MANIFEST] = str(mp)
        # the submission binds the NEW manifest, so rebuild the
        # review around it: only the DATA binding differs
        sub2 = review.build_submission(
            submitted_at="2026-09-10T10:00:00Z",
            submitter="t", scope="forecasting",
            manifest_sha256=mutated["manifest_sha256"],
            census_sha256=CENSUS, consumed=consumed)
        _install_review(tmp, monkeypatch, sub2, CENSUS)
        with pytest.raises(SystemExit, match=needle):
            integ.gate_run(cfg, repo_root=tmp, consumer="t")
    mp.write_text(json.dumps(doc))


def test_a_renamed_column_refuses(world, monkeypatch):
    tmp, config = world
    consumed, mp, sub = _gated(tmp, config, monkeypatch)
    _install_review(tmp, monkeypatch, sub, CENSUS)
    monkeypatch.setattr(review, "utc_now_stamp",
                        lambda: "2026-09-10T10:00:00Z")
    for role in ("d4", "d5", "d6"):
        _csv(tmp / "data" / f"{role}.csv",
             columns=["DATE_TIME", "typical_price", "renamed"])
    # renaming a column changes the derived subject set AND the
    # data digest, so the run is no longer the run that was
    # reviewed; the submission binding is the first guard to say
    # so, and that is the correct reason
    with pytest.raises(SystemExit,
                       match="DIFFERENT submission"):
        integ.gate_run(config, repo_root=tmp, consumer="t")
    # and the reviewed manifest genuinely does not list the new
    # column, proven directly
    fresh = resolve_consumed_subjects(config, repo_root=tmp)
    manifest = gate.load_manifest(mp)
    with pytest.raises(SystemExit, match="unlisted subject"):
        gate.require_eligible(
            manifest,
            subject_id(fresh["dataset_id"], "renamed"),
            scope="forecasting")


# ==============================================================
# C4: strict parsing
# ==============================================================

def test_duplicate_keys_refuse(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text('{"a": 1, "a": 2}')
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        strict.strict_load_file(p, what="doc")


def test_non_finite_constants_refuse(tmp_path):
    for literal in ("NaN", "Infinity", "-Infinity"):
        p = tmp_path / "nf.json"
        p.write_text('{"a": %s}' % literal)
        with pytest.raises(SystemExit, match="non-finite"):
            strict.strict_load_file(p, what="doc")


def test_non_canonical_digests_refuse():
    for bad in ("A" * 64, "abc", "g" * 64, 12345, None,
                " " + "a" * 63):
        with pytest.raises(SystemExit, match="canonical SHA-256"
                                             "|not a digest"):
            strict.require_sha256(bad, what="d")
    strict.require_sha256("a" * 64, what="d")


def test_future_and_non_canonical_timestamps_refuse():
    with pytest.raises(SystemExit, match="FUTURE"):
        strict.require_timestamp("2099-01-01T00:00:00Z",
                                 what="t")
    for bad in ("2026-09-10", "2026-09-10T00:00:00",
                "2026-09-10T00:00:00+02:00", 20260910):
        with pytest.raises(SystemExit,
                           match="canonical RFC3339|not a "
                                 "timestamp"):
            strict.require_timestamp(bad, what="t")
    strict.require_timestamp("2026-09-10T00:00:00Z", what="t")


def test_a_bool_never_passes_a_numeric_field():
    with pytest.raises(SystemExit, match="not a number"):
        strict.require_number(True, what="n")
    strict.require_number(0, what="n")


def test_the_manifest_parser_now_refuses_all_four(tmp_path):
    """The four permissive behaviours the audit reproduced."""
    base = {"schema": gate.MANIFEST_SCHEMA,
            "issued_at": "2026-09-10T00:00:00Z",
            "issuer": "r", "scope": "s", "entries": []}
    base["manifest_sha256"] = gate._self_sha(base)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(base))
    gate.load_manifest(p)                       # baseline passes

    p.write_text(json.dumps(base).replace(
        '"issuer": "r"', '"issuer": "r", "issuer": "x"'))
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        gate.load_manifest(p)

    future = dict(base, issued_at="2099-01-01T00:00:00Z")
    future["manifest_sha256"] = gate._self_sha(future)
    p.write_text(json.dumps(future))
    with pytest.raises(SystemExit, match="FUTURE"):
        gate.load_manifest(p, max_age_days=30)

    bad_digest = dict(base)
    bad_digest["manifest_sha256"] = "abc"
    p.write_text(json.dumps(bad_digest))
    with pytest.raises(SystemExit, match="re-derive|canonical"):
        gate.load_manifest(p)


# ==============================================================
# C5: archival replay is an express choice
# ==============================================================

def test_a_new_experiment_without_a_review_refuses(world):
    tmp, config = world
    with pytest.raises(SystemExit, match="no eligibility "
                                         "manifest"):
        integ.gate_run(config, repo_root=tmp, consumer="t")


def test_archival_replay_must_be_declared(world):
    tmp, config = world
    config[integ.PURPOSE_KEY] = integ.PURPOSE_ARCHIVAL
    stamp = integ.gate_run(config, repo_root=tmp, consumer="t")
    assert stamp["eligibility_status"] == integ.STATUS_LEGACY
    assert stamp["execution_purpose"] == integ.PURPOSE_ARCHIVAL
    assert stamp["subjects_derived"] == 2
    assert config[integ.KEY_STAMP] is stamp


def test_an_unknown_purpose_refuses(world):
    tmp, config = world
    config[integ.PURPOSE_KEY] = "PROBABLY_FINE"
    with pytest.raises(SystemExit, match="unknown "
                                         "execution_purpose"):
        integ.gate_run(config, repo_root=tmp, consumer="t")


def test_the_shipped_default_is_express_not_inferred():
    from app.config import DEFAULT_VALUES
    assert DEFAULT_VALUES["execution_purpose"] == \
        integ.PURPOSE_ARCHIVAL
    src = (REPO / "eligibility/integration.py").read_text()
    assert "no longer an implicit legacy run" in src \
        or "Omitting one is no longer" in src
