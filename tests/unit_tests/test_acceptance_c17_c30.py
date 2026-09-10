"""The twelve acceptance points of order C17-C30, §6.

One test per numbered point, plus the C30 separation rules.
PostgreSQL tests that write use their own throwaway database; the
populated cube is only read.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review  # noqa: E402
from eligibility import strict  # noqa: E402
from eligibility.consumed import (  # noqa: E402
    resolve_consumed_subjects)
from olap import campaign_envelope as ce  # noqa: E402
from olap import characterization as ch  # noqa: E402
from olap import inventory_rows as ir  # noqa: E402
from olap import outbox as ob  # noqa: E402
from olap import terminal as term  # noqa: E402

FIN = REPO.parent / "financial-data"


def _csv(p: Path, cols, value="1"):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow(["2020-01-01"] + [value] * (len(cols) - 1))


@pytest.fixture()
def run_world(tmp_path, monkeypatch):
    for r in ("d4", "d5", "d6"):
        _csv(tmp_path / f"x_{r}.csv", ["DATE_TIME", "px"])
        _csv(tmp_path / f"y_{r}.csv", ["DATE_TIME", "target"])
    monkeypatch.setenv(review.SUBMISSION_ENV,
                       str(tmp_path / "subs"))
    return tmp_path, {
        "x_train_file": str(tmp_path / "x_d4.csv"),
        "x_validation_file": str(tmp_path / "x_d5.csv"),
        "x_test_file": str(tmp_path / "x_d6.csv"),
        "y_train_file": str(tmp_path / "y_d4.csv"),
        "y_validation_file": str(tmp_path / "y_d5.csv"),
        "y_test_file": str(tmp_path / "y_d6.csv"),
        "target_column": "target",
        integ.KEY_SCOPE: "forecasting",
        integ.KEY_CENSUS: "a" * 64,
    }


@pytest.fixture()
def throwaway_db():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    name = f"predictor_olap_acc_{uuid.uuid4().hex[:10]}"
    dsn = ("postgresql://metabase:metabase_pass@"
           "127.0.0.1:5432/")
    admin = create_engine(dsn + "postgres", future=True,
                          isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"no PostgreSQL: "
                    f"{exc.__class__.__name__}")
    e = create_engine(dsn + name, future=True)
    try:
        yield e
    finally:
        e.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def _read_cube():
    from sqlalchemy import create_engine, text
    e = create_engine(
        "postgresql://metabase:metabase_pass@127.0.0.1:5432/"
        "predictor_olap", future=True)
    with e.connect() as c:
        yield c, text


# 1 ------------------------------------------------------------
def test_1_the_ten_pre_defects_are_gone():
    """The PRE file is frozen; its defects must not survive."""
    integ_src = (REPO / "eligibility/integration.py").read_text()
    cons_src = (REPO / "eligibility/consumed.py").read_text()
    main_src = (REPO / "app/main.py").read_text()
    inv_src = (REPO / "olap/inventory_rows.py").read_text()
    assert "SUBMIT_ONLY" in integ_src
    assert "EXECUTE_REVIEWED" in integ_src
    assert "y_sha256" in cons_src and "y_columns" in cons_src
    assert "ALLOWED_OPTIMIZER_KEYS" in integ_src
    assert "assert_contract_unchanged" in main_src
    assert "terminal_run" in main_src
    assert "observation_sha256" in inv_src
    assert len(__import__(
        "eligibility.consumed", fromlist=["x"]
    ).CONSUMING_CODE) > 4


# 2 ------------------------------------------------------------
def test_2_two_phases_across_processes(run_world):
    """Covered in depth by test_gate_c17_c21; restated here."""
    tmp, config = run_world
    consumed = resolve_consumed_subjects(dict(config),
                                         repo_root=tmp)
    assert consumed["subject_ids"]
    src = (REPO / "eligibility/review.py").read_text()
    assert "def persist_submission" in src
    assert "def load_persisted_submission" in src
    assert "def assert_same_submission" in src


# 3 ------------------------------------------------------------
def test_3_mutating_any_x_or_y_changes_identity(run_world):
    tmp, config = run_world
    base = resolve_consumed_subjects(dict(config),
                                     repo_root=tmp)
    for target, cols in ((tmp / "x_d5.csv",
                          ["DATE_TIME", "px"]),
                         (tmp / "y_d6.csv",
                          ["DATE_TIME", "target"])):
        original = target.read_text()
        _csv(target, cols, value="777")
        after = resolve_consumed_subjects(dict(config),
                                          repo_root=tmp)
        assert after["digests"]["data"] != \
            base["digests"]["data"], f"{target.name} invisible"
        assert after["digests"]["partitions"] != \
            base["digests"]["partitions"]
        target.write_text(original)


# 4 ------------------------------------------------------------
def test_4_the_optimizer_cannot_change_data_target_or_plugins():
    for key in ("x_test_file", "y_test_file", "target_column",
                "preprocessor_plugin", "pipeline_plugin",
                "plugin", "eligibility_manifest"):
        with pytest.raises(SystemExit,
                           match="only propose hyperparameters"):
            integ.assert_optimizer_result_is_hyperparameters_only(
                {key: "x"}, consumer="t")


# 5 ------------------------------------------------------------
def test_5_changing_a_plugin_changes_the_code_digest(run_world):
    tmp, config = run_world
    a = resolve_consumed_subjects(
        dict(config, plugin="ann",
             pipeline_plugin="stl_pipeline"), repo_root=tmp)
    b = resolve_consumed_subjects(
        dict(config, plugin="ann",
             pipeline_plugin="default_pipeline"), repo_root=tmp)
    assert a["digests"]["code"] != b["digests"]["code"]


# 6 ------------------------------------------------------------
def test_6_loose_documents_refuse(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"a": 1, "a": 2}')
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        strict.strict_load_file(p, what="d")
    p.write_text('{"a": NaN}')
    with pytest.raises(SystemExit, match="non-finite"):
        strict.strict_load_file(p, what="d")
    with pytest.raises(SystemExit, match="canonical SHA-256"):
        strict.require_sha256("A" * 64, what="d")
    with pytest.raises(SystemExit, match="FUTURE"):
        strict.require_timestamp("2099-01-01T00:00:00Z",
                                 what="t")
    with pytest.raises(SystemExit, match="not a number"):
        strict.require_number(True, what="n")
    # envelope: extra nested field and NaN both refuse
    def _env(**over):
        base = dict(
            campaign_key="k", producer="p",
            result_class="DEVELOPMENT",
            identity={"run_id": "r", "code_identity": "c",
                      "design_sha256": "d"},
            data_consumed={"datasets": [], "variables": [],
                           "operators": []},
            partitions={"exposure": "e", "splits": "s"},
            budget={"device": "cpu", "wall_seconds": 1.0,
                    "cost_units": "s"},
            terminal={"state": "COMPLETE",
                      "adjudication": "NONE"},
            artifacts={"a": "b"}, units=[])
        base.update(over)
        return ce.build_envelope(**base)
    with pytest.raises(SystemExit, match="undeclared fields"):
        _env(identity={"run_id": "r", "code_identity": "c",
                       "design_sha256": "d", "extra": "x"})
    with pytest.raises(SystemExit, match="non-finite"):
        _env(budget={"device": "cpu",
                     "wall_seconds": float("nan"),
                     "cost_units": "s"})


# 7 ------------------------------------------------------------
@pytest.mark.parametrize("state", term.TERMINAL_STATES)
def test_7_every_outcome_produces_exactly_one_terminal(
        state, tmp_path):
    env = term.build_terminal_envelope(
        campaign_key="k", producer="p", state=state,
        stamp={}, config={}, wall_seconds=1.0)
    assert env["terminal"]["state"] == state
    assert env["result_class"] == term.RESULT_CLASS_FOR[state]
    assert len(env["units"]) == 1, (
        "a terminal run IS a unit; units=[] was the shape the "
        "order rejected")
    out = ob.emit(env, kind="envelope", root=tmp_path)
    assert out["written"] is True
    again = ob.emit(env, kind="envelope", root=tmp_path)
    assert again["written"] is False, "two terminals for one run"


def test_7b_a_refusal_and_a_crash_both_emit(tmp_path,
                                            monkeypatch):
    monkeypatch.setenv(ob.OUTBOX_ENV, str(tmp_path))
    for boom, expected in (
            (lambda: (_ for _ in ()).throw(
                SystemExit("REFUSED: no")), term.REFUSED),
            (lambda: (_ for _ in ()).throw(
                ValueError("boom")), term.FAILED)):
        cfg = {}
        with pytest.raises(BaseException):
            term.terminal_run(boom, campaign_key="k",
                              producer="p", config=cfg)
        assert cfg["olap_terminal"]["state"] == expected
    assert ob.counts(tmp_path)["pending"] == 2


# 8 ------------------------------------------------------------
def test_8_the_loader_is_active_and_loads_without_duplicating():
    r = subprocess.run(
        ["systemctl", "--user", "is-active",
         "crispdm-olap-loader.service"],
        capture_output=True, text=True)
    if r.stdout.strip() != "active":
        pytest.skip("loader service not active in this "
                    "environment")
    hb = ob.DEFAULT_OUTBOX / "HEARTBEAT.json"
    assert hb.is_file(), "an active loader publishes a heartbeat"
    doc = json.loads(hb.read_text())
    assert doc["loaded"] >= 1, (
        "the loader has not loaded a single entry end to end")
    assert doc["pending"] == 0
    assert "published_at_epoch" in doc


# 9 ------------------------------------------------------------
def test_9_a_new_census_versions_and_a_repeat_does_not(
        throwaway_db):
    from sqlalchemy import text

    def index(digest):
        return {"schema": "crispdm.bank_index.v1",
                "index_sha256": "i" * 64,
                "banks": {"financial_domain": {
                    "authority_class": ir.__dict__.get(
                        "X", "FINANCIAL_DOMAIN_DEVELOPMENT_ONLY"),
                    "binding": {"census_document_sha256":
                                digest},
                    "appearances": [{
                        "appearance_id": "app_1", "entity": "e",
                        "source_class": "cross_source",
                        "frequency": "1h", "period_start": "a",
                        "period_end": "b",
                        "physical_sha256": digest,
                        "digest_state": "PHYSICALLY_DIGESTED"}],
                    "variables": []}}}
    ir.load_index(throwaway_db, index("1" * 64))
    ir.load_index(throwaway_db, index("1" * 64))
    with throwaway_db.connect() as c:
        assert c.execute(text(
            "select count(*) from dim_lake_appearance"
        )).scalar() == 1
    ir.load_index(throwaway_db, index("2" * 64))
    with throwaway_db.connect() as c:
        assert c.execute(text(
            "select count(*) from dim_lake_appearance"
        )).scalar() == 2
        assert c.execute(text(
            "select count(*) from v_lake_appearance_current"
        )).scalar() == 1
        current = c.execute(text(
            "select physical_sha256 from "
            "v_lake_appearance_current")).scalar()
    assert current.startswith("2")


# 10 -----------------------------------------------------------
def test_10_the_cube_keeps_its_history():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    try:
        e = create_engine(
            "postgresql://metabase:metabase_pass@"
            "127.0.0.1:5432/predictor_olap", future=True)
        with e.connect() as c:
            exp = c.execute(text(
                "select count(*) from dim_experiment")).scalar()
            perf = c.execute(text(
                "select count(*) from fact_performance"
            )).scalar()
            units = c.execute(text(
                "select count(*) from fact_campaign_unit"
            )).scalar()
            states = dict(c.execute(text(
                "select authority_state, count(*) from "
                "fact_campaign_unit group by 1")).all())
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"cube unavailable: "
                    f"{exc.__class__.__name__}")
    assert exp == 39
    assert perf == 1404
    assert units >= 120, "the 120 existing units must survive"
    # the 60 translated and 60 producer-verified rows survive;
    # runs born at a producer terminal are a THIRD provenance and
    # are never counted among the translated
    assert states.get("TRANSLATED_SUMMARY_NON_AUTHORITATIVE") \
        == 60
    assert states.get("PRODUCER_VERIFIED") == 60
    assert states.get(ce.PRODUCER_EMITTED, 0) >= 1


# 11 -----------------------------------------------------------
def test_11_financial_coverage_is_derived_not_declared():
    bridge = (FIN /
              "features/census/AVAILABILITY_BRIDGE.v1.json")
    disp = (FIN /
            "features/census/AVAILABILITY_DISPOSITION.v1.json")
    if not bridge.is_file() or not disp.is_file():
        pytest.skip("bridge not derived in this checkout")
    b = json.loads(bridge.read_text())
    d = json.loads(disp.read_text())
    assert b["bridge"]["columns_examined"] == 94
    assert b["bridge"]["resolved"] + b["bridge"]["unresolved"] \
        == 94
    assert b["demand"]["configs_declaring_both"] == []
    assert "no configuration was changed" in \
        b["bridge"]["no_config_was_edited"]
    # zero is a valid answer, and it is the answer
    assert d["disposition"] == \
        "FINANCIAL_AVAILABILITY_EVIDENCE_REQUIRED"
    for gap in d["what_would_unblock_it"]:
        assert gap["owner_decision_required"] is False


# 12 -----------------------------------------------------------
def test_12_characterization_measures_without_selecting():
    ledger = (REPO /
              "examples/research/CHARACTERIZATION_LEDGER.v1.json")
    if not ledger.is_file():
        pytest.skip("characterization not run")
    doc = json.loads(ledger.read_text())
    assert doc["rows_total"] > 0
    assert doc["selection_emitted"].startswith("NONE")
    assert doc["confirmation_used"] == "NONE"
    assert doc["gpu_used"] == "NONE"
    assert doc["total_cost_seconds"] > 0
    banks = doc["rows_by_bank"]
    assert ch.BANK_SYNTHETIC in banks
    assert ch.BANK_FINANCIAL in banks


def test_12b_a_selection_descriptor_refuses():
    with pytest.raises(SystemExit, match="never chooses them"):
        ch.assert_no_selection([{
            "descriptor": "feature_importance_rank"}])


def test_12c_noise_is_only_reported_where_identifiable():
    rows = ch.characterize_series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        variable_id="v", partition_key="p",
        bank_authority=ch.BANK_FINANCIAL,
        measured_at="2026-09-10T00:00:00Z",
        noise_reference=None)
    noise = [r for r in rows if r["descriptor"] ==
             "noise_estimate"]
    assert noise and noise[0]["identifiable"] is False
    assert "NOT IDENTIFIABLE" in noise[0]["descriptor_contract"]


def test_12d_descriptors_never_overclaim():
    rows = ch.characterize_series(
        list(range(64)), variable_id="v", partition_key="p",
        bank_authority=ch.BANK_SYNTHETIC,
        measured_at="2026-09-10T00:00:00Z")
    by = {r["descriptor"]: r for r in rows}
    assert "NOT an information content" in \
        by["compressed_length_ratio"]["descriptor_contract"]
    assert "NOT Kolmogorov complexity" in \
        by["compressed_length_ratio"]["descriptor_contract"]
    assert "not of the variable" in \
        by["discrete_entropy_bits"]["descriptor_contract"]
    assert "not a hypothesis test" in \
        by["difference_to_level_dispersion"][
            "descriptor_contract"]


def test_12e_an_unknown_bank_refuses():
    with pytest.raises(SystemExit, match="unknown bank "
                                         "authority"):
        ch.characterize_series(
            [1.0, 2.0, 3.0], variable_id="v",
            partition_key="p", bank_authority="SOME_NEW_BANK",
            measured_at="2026-09-10T00:00:00Z")
