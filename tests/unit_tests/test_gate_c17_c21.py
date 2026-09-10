"""C17-C21: two real phases, the whole consumed set, an immutable
contract, the real code graph and exact schemas.

The two-phase test runs the phases in SEPARATE PROCESSES, because
a protocol that only works inside one interpreter is not a
protocol.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review  # noqa: E402
from eligibility.consumed import (  # noqa: E402
    CodeIdentityRefusal, code_identity,
    resolve_consumed_subjects, resolve_plugin_modules)

CENSUS = "a" * 64


def _csv(path: Path, columns, value="1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerow(["2020-01-01"] + [value] * (len(columns) - 1))


@pytest.fixture()
def world(tmp_path, monkeypatch):
    for r in ("d4", "d5", "d6"):
        _csv(tmp_path / f"x_{r}.csv", ["DATE_TIME", "px"])
        _csv(tmp_path / f"y_{r}.csv", ["DATE_TIME", "target"])
    config = {
        "x_train_file": str(tmp_path / "x_d4.csv"),
        "x_validation_file": str(tmp_path / "x_d5.csv"),
        "x_test_file": str(tmp_path / "x_d6.csv"),
        "y_train_file": str(tmp_path / "y_d4.csv"),
        "y_validation_file": str(tmp_path / "y_d5.csv"),
        "y_test_file": str(tmp_path / "y_d6.csv"),
        "target_column": "target",
        integ.KEY_SCOPE: "forecasting",
        integ.KEY_CENSUS: CENSUS,
    }
    monkeypatch.setenv(review.SUBMISSION_ENV,
                       str(tmp_path / "subs"))
    return tmp_path, config


def _manifest(tmp_path, consumed):
    d = consumed["digests"]
    digests = {"data": d["data"], "code": d["code"],
               "partitions": d["partitions"],
               "evidence": "e" * 64}
    entries = [{
        "subject_kind": "variable", "subject_id": sid,
        "version": "1",
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none", "parameters": {},
        "digests": dict(digests), "measured_cost": {},
        "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "forecasting",
        "decision_reason": "reviewed", "reviewer": "double",
        "reviewed_at": "2026-09-09T00:00:00Z"}
        for sid in consumed["subject_ids"]]
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "double", "scope": "forecasting",
           "entries": entries}
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(doc))
    return p, doc


def _install_review(tmp_path, monkeypatch, submission):
    rec = {
        "schema": review.REVIEW_RECORD_SCHEMA,
        # a reviewer cannot decide before the submission nor
        # after now; the submission's own instant satisfies both
        "reviewed_at": submission["submitted_at"],
        "reviewer": "external-double",
        "scope": "forecasting",
        "reviewed_submission_sha256":
            submission["submission_sha256"],
        "reviewed_manifest_sha256":
            submission["manifest_sha256"],
        "reviewed_census_sha256": submission["census_sha256"],
        "reviewed_code_digest": submission["code_digest"],
        "reviewed_partitions_digest":
            submission["partitions_digest"],
        "decision": review.REVIEW_DECISION}
    rec["record_sha256"] = review._self_sha(rec, "record_sha256")
    p = tmp_path / "REVIEW.json"
    p.write_text(json.dumps(rec))
    monkeypatch.setenv(review.REVIEW_RECORD_ENV, str(p))
    return rec


# ==============================================================
# C17: two phases, two processes
# ==============================================================

def test_submit_only_persists_and_executes_nothing(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    mp, _ = _manifest(tmp, consumed)
    cfg = dict(config, eligibility_manifest=str(mp))
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_SUBMIT
    stamp = integ.gate_run(cfg, repo_root=tmp, consumer="t")
    assert stamp["eligibility_status"] == integ.STATUS_SUBMITTED
    assert "grants_nothing" in stamp
    path = review.submission_path(stamp["submission_sha256"])
    assert path.is_file(), "the submission was not persisted"
    review.verify_submission(json.loads(path.read_text()))


def test_the_two_phases_work_across_separate_processes(
        world, tmp_path):
    """The protocol must survive process boundaries."""
    tmp, config = world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    mp, _ = _manifest(tmp, consumed)
    subs = tmp / "subs"
    env_common = {
        "CRISPDM_ELIGIBILITY_SUBMISSION_DIR": str(subs)}

    phase1 = tmp / "phase1.py"
    phase1.write_text(
        "import json, sys, os\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from eligibility import integration as integ\n"
        f"cfg = json.loads({json.dumps(json.dumps(dict(config, eligibility_manifest=str(mp))))})\n"
        f"cfg[integ.PURPOSE_KEY] = integ.PURPOSE_SUBMIT\n"
        f"s = integ.gate_run(cfg, repo_root={str(tmp)!r}, "
        "consumer='t')\n"
        "print(json.dumps(s))\n")
    import os
    r1 = subprocess.run([sys.executable, str(phase1)],
                        capture_output=True, text=True,
                        env={**os.environ, **env_common},
                        timeout=300)
    assert r1.returncode == 0, r1.stderr[-500:]
    stamp1 = json.loads(r1.stdout.strip().splitlines()[-1])
    sub_sha = stamp1["submission_sha256"]

    submission = json.loads(
        (subs / f"submission-{sub_sha}.json").read_text())
    rec = {
        "schema": review.REVIEW_RECORD_SCHEMA,
        "reviewed_at": submission["submitted_at"],
        "reviewer": "external-double", "scope": "forecasting",
        "reviewed_submission_sha256": sub_sha,
        "reviewed_manifest_sha256":
            submission["manifest_sha256"],
        "reviewed_census_sha256": CENSUS,
        "reviewed_code_digest": submission["code_digest"],
        "reviewed_partitions_digest":
            submission["partitions_digest"],
        "decision": review.REVIEW_DECISION}
    rec["record_sha256"] = review._self_sha(rec, "record_sha256")
    recp = tmp / "REVIEW.json"
    recp.write_text(json.dumps(rec))

    phase2 = tmp / "phase2.py"
    phase2.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from eligibility import integration as integ\n"
        f"cfg = json.loads({json.dumps(json.dumps(dict(config, eligibility_manifest=str(mp))))})\n"
        "cfg[integ.PURPOSE_KEY] = integ.PURPOSE_EXECUTE\n"
        f"cfg[integ.KEY_SUBMISSION_SHA] = {sub_sha!r}\n"
        f"s = integ.gate_run(cfg, repo_root={str(tmp)!r}, "
        "consumer='t')\n"
        "print(json.dumps(s))\n")
    env2 = {**os.environ, **env_common,
            review.REVIEW_RECORD_ENV: str(recp)}
    r2 = subprocess.run([sys.executable, str(phase2)],
                        capture_output=True, text=True,
                        env=env2, timeout=300)
    assert r2.returncode == 0, r2.stderr[-800:]
    stamp2 = json.loads(r2.stdout.strip().splitlines()[-1])
    assert stamp2["eligibility_status"] == integ.STATUS_GATED
    assert stamp2["submission_sha256"] == sub_sha
    assert stamp2["reviewer"] == "external-double"

    # one changed byte between the phases refuses
    _csv(tmp / "y_d5.csv", ["DATE_TIME", "target"],
         value="999")
    r3 = subprocess.run([sys.executable, str(phase2)],
                        capture_output=True, text=True,
                        env=env2, timeout=300)
    assert r3.returncode != 0
    assert "DIFFERENT facts" in (r3.stdout + r3.stderr)


def test_execute_without_a_submission_digest_refuses(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    mp, _ = _manifest(tmp, consumed)
    cfg = dict(config, eligibility_manifest=str(mp))
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_EXECUTE
    with pytest.raises(SystemExit, match="requires "
                                         "eligibility_submission"):
        integ.gate_run(cfg, repo_root=tmp, consumer="t")


def test_an_unpersisted_submission_refuses(world):
    tmp, config = world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    mp, _ = _manifest(tmp, consumed)
    cfg = dict(config, eligibility_manifest=str(mp))
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_EXECUTE
    cfg[integ.KEY_SUBMISSION_SHA] = "f" * 64
    with pytest.raises(SystemExit,
                       match="no persisted submission"):
        integ.gate_run(cfg, repo_root=tmp, consumer="t")


# ==============================================================
# C18: the whole consumed set
# ==============================================================

def test_a_mutated_y_changes_the_identity(world):
    tmp, config = world
    before = resolve_consumed_subjects(dict(config),
                                       repo_root=tmp)
    _csv(tmp / "y_d5.csv", ["DATE_TIME", "target"],
         value="999")
    after = resolve_consumed_subjects(dict(config),
                                      repo_root=tmp)
    assert before["digests"]["data"] != after["digests"]["data"]
    assert before["digests"]["partitions"] != \
        after["digests"]["partitions"]


def test_the_y_target_is_a_subject(world):
    tmp, config = world
    c = resolve_consumed_subjects(dict(config), repo_root=tmp)
    sides = {s["column"]: s["side"] for s in c["subjects"]}
    assert sides["target"] == "y"
    assert sides["px"] == "x"
    roles = {s["column"]: s["contract_role"]
             for s in c["subjects"]}
    assert roles["target"] == "target"


def test_an_omitted_or_swapped_y_changes_identity(world):
    tmp, config = world
    before = resolve_consumed_subjects(dict(config),
                                       repo_root=tmp)
    swapped = dict(config)
    swapped["y_validation_file"] = str(tmp / "y_d6.csv")
    after = resolve_consumed_subjects(swapped, repo_root=tmp)
    assert before["digests"]["partitions"] != \
        after["digests"]["partitions"]
    omitted = dict(config)
    del omitted["y_test_file"]
    third = resolve_consumed_subjects(omitted, repo_root=tmp)
    assert third["digests"]["partitions"] != \
        before["digests"]["partitions"]


def test_a_missing_y_file_refuses(world):
    tmp, config = world
    (tmp / "y_d5.csv").unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        resolve_consumed_subjects(dict(config), repo_root=tmp)


# ==============================================================
# C19: the contract is immutable after the gate
# ==============================================================

@pytest.mark.parametrize("key", ["x_test_file",
                                 "target_column",
                                 "preprocessor_plugin",
                                 "eligibility_manifest",
                                 "y_train_file", "plugin"])
def test_the_optimizer_cannot_change_the_contract(key):
    with pytest.raises(SystemExit,
                       match="only propose hyperparameters"):
        integ.assert_optimizer_result_is_hyperparameters_only(
            {key: "smuggled"}, consumer="t")


def test_the_optimizer_may_propose_hyperparameters():
    out = integ.assert_optimizer_result_is_hyperparameters_only(
        {"batch_size": 64, "learning_rate": 0.01},
        consumer="t")
    assert out == {"batch_size": 64, "learning_rate": 0.01}


def test_a_non_mapping_optimizer_result_refuses():
    with pytest.raises(SystemExit, match="not a mapping"):
        integ.assert_optimizer_result_is_hyperparameters_only(
            [1, 2, 3], consumer="t")


def test_a_changed_contract_after_the_gate_refuses(world,
                                                   monkeypatch):
    tmp, config = world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    mp, _ = _manifest(tmp, consumed)
    cfg = dict(config, eligibility_manifest=str(mp))
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_SUBMIT
    s1 = integ.gate_run(cfg, repo_root=tmp, consumer="t")
    sub = json.loads(review.submission_path(
        s1["submission_sha256"]).read_text())
    _install_review(tmp, monkeypatch, sub)
    cfg[integ.PURPOSE_KEY] = integ.PURPOSE_EXECUTE
    cfg[integ.KEY_SUBMISSION_SHA] = s1["submission_sha256"]
    stamp = integ.gate_run(cfg, repo_root=tmp, consumer="t")
    assert stamp["eligibility_status"] == integ.STATUS_GATED
    integ.assert_contract_unchanged(cfg, repo_root=tmp,
                                    consumer="t")
    # now change the data underneath the approved contract
    _csv(tmp / "x_d6.csv", ["DATE_TIME", "px"], value="42")
    with pytest.raises(SystemExit, match="contract CHANGED"):
        integ.assert_contract_unchanged(cfg, repo_root=tmp,
                                        consumer="t")


def test_main_enforces_both_c19_guards():
    src = (REPO / "app/main.py").read_text()
    gate_at = src.index("gate_run(")
    filt_at = src.index(
        "assert_optimizer_result_is_hyperparameters_only(\n")
    upd_at = src.index("config.update(optimal_params)")
    rederive_at = src.index("assert_contract_unchanged(")
    pipe_at = src.index(
        "pipeline_plugin.run_prediction_pipeline(")
    assert gate_at < filt_at < upd_at
    assert upd_at < rederive_at < pipe_at


# ==============================================================
# C20: the real code graph
# ==============================================================

def test_the_code_identity_covers_the_consuming_graph(world):
    tmp, config = world
    cfg = dict(config, plugin="ann",
               pipeline_plugin="stl_pipeline",
               preprocessor_plugin="stl_preprocessor")
    c = resolve_consumed_subjects(cfg, repo_root=tmp)
    kinds = {e["kind"] for e in c["code_inventory"]}
    assert kinds == {"module", "plugin"}
    ids = {e["id"] for e in c["code_inventory"]}
    for expected in ("app/main.py", "app/config_merger.py",
                     "app/plugin_loader.py",
                     "app/data_handler.py"):
        assert expected in ids
    roles = {e["role"] for e in c["code_inventory"]
             if e["kind"] == "plugin"}
    assert {"predictor", "pipeline", "preprocessor"} <= roles


def test_changing_a_plugin_changes_the_code_digest(world,
                                                   tmp_path):
    tmp, config = world
    cfg = dict(config, plugin="ann",
               pipeline_plugin="stl_pipeline")
    a = resolve_consumed_subjects(dict(cfg), repo_root=tmp)
    cfg2 = dict(cfg, pipeline_plugin="default_pipeline")
    b = resolve_consumed_subjects(cfg2, repo_root=tmp)
    assert a["digests"]["code"] != b["digests"]["code"], (
        "changing the executed pipeline plugin left the "
        "reviewed code digest unchanged")


def test_an_unresolvable_plugin_refuses(world):
    tmp, config = world
    cfg = dict(config, plugin="a_plugin_that_does_not_exist")
    with pytest.raises(SystemExit, match="not registered"):
        resolve_consumed_subjects(cfg, repo_root=tmp)


def test_a_missing_consuming_module_refuses(tmp_path):
    with pytest.raises(SystemExit, match="is absent"):
        code_identity(tmp_path, ["app/main.py"], {})


def test_the_inventory_names_what_it_bound(world):
    tmp, config = world
    c = resolve_consumed_subjects(dict(config, plugin="ann"),
                                  repo_root=tmp)
    for e in c["code_inventory"]:
        assert len(e["sha256"]) == 64
        assert e["id"]
        if e["kind"] == "plugin":
            assert e["entry_point"]


# ==============================================================
# C21: exact recursive schemas
# ==============================================================

def _base_manifest(tmp_path, **over):
    entry = {
        "subject_kind": "variable", "subject_id": "v",
        "version": "1",
        "io_schema": {"input": ["f"], "output": ["f"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "n", "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "b" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "s", "decision_reason": "r",
        "reviewer": "x",
        "reviewed_at": "2026-09-09T00:00:00Z"}
    entry.update(over.pop("entry", {}))
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "r", "scope": "s", "entries": [entry]}
    doc.update(over)
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    return p


def test_an_undeclared_top_level_field_refuses(tmp_path):
    p = _base_manifest(tmp_path, undeclared_top="smuggled")
    with pytest.raises(SystemExit, match="not the exact schema"):
        gate.load_manifest(p)


def test_an_undeclared_entry_field_refuses(tmp_path):
    p = _base_manifest(
        tmp_path, entry={"undeclared_nested": "smuggled"})
    with pytest.raises(SystemExit,
                       match="undeclared fields"):
        gate.load_manifest(p)


def test_an_undeclared_availability_field_refuses(tmp_path):
    p = _base_manifest(tmp_path, entry={
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t",
                                  "smuggled": 1}})
    with pytest.raises(SystemExit,
                       match="undeclared fields"):
        gate.load_manifest(p)


def test_a_bool_cost_refuses(tmp_path):
    p = _base_manifest(tmp_path,
                       entry={"measured_cost": {"s": True}})
    with pytest.raises(SystemExit, match="not a number"):
        gate.load_manifest(p)


def test_a_baseline_manifest_still_loads(tmp_path):
    gate.load_manifest(_base_manifest(tmp_path))
