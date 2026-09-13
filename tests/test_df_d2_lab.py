"""C171-C177: sealed D2 v2 design, fresh seed tape, one-unit worker through the public API with the
wavelet audit, and adjudication from fresh unit-level rows only. Small fixtures in tmp_path."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = load("df_d2_design")
TAPE = load("df_seed_tape")
W = load("df_d2_unit_worker")
A = load("df_d2_adjudicate")
BANK = load("df_synthetic_bank")
OPS = load("df_operators")
L = load("load_data_foundation")
# the retired name comes from the naming record, the only place it is written
OLD = next(k for k, v in D.RETIRED_NAMES.items() if v == "trailing_haar_threshold")

N = 512
CELLS = [BANK._cell("sinusoid", "white", 10, N), BANK._cell("steps", "white", 0, N)]
OPERATORS = [{"spec": {"kind": "identity", "params": {}}, "arm_role": "IDENTITY_RAW_CONTROL"},
             {"spec": {"kind": "ewma", "params": {"alpha": 0.3}}, "arm_role": "CANDIDATE"},
             {"spec": {"kind": "trailing_haar_threshold", "params": {"levels": 2, "threshold_k": 3.0}},
              "arm_role": "CANDIDATE"},
             {"spec": {"kind": "trailing_median", "params": {"window": 5}}, "arm_role": "PREVIOUSLY_REJECTED_CONTROL"},
             {"spec": {"kind": "centered_mean_oracle", "params": {"window": 5}},
              "arm_role": "NON_CAUSAL_ORACLE_CONTROL"}]
R1, R2 = (D.regime_key(D.regime_of_cell(c)) for c in CELLS)


def inputs():
    ew, hr = (D.spec_sha(OPERATORS[i]["spec"]) for i in (1, 2))
    disp = [{"spec_sha256": ew, "regime_key": R1, "metric": "snr_improvement_db", "mean": 3.0, "sd": 1.0, "n": 3},
            {"spec_sha256": hr, "regime_key": R1, "metric": "snr_improvement_db", "mean": 3.0, "sd": 1.0, "n": 3},
            {"spec_sha256": ew, "regime_key": R2, "metric": "snr_improvement_db", "mean": 2.0, "sd": 1.0, "n": 3},
            {"spec_sha256": hr, "regime_key": R2, "metric": "snr_improvement_db", "mean": -1.0, "sd": 1.0, "n": 3}]
    return {"design_id": "d2v2_fixture", "cells": CELLS, "operators": OPERATORS, "dispersion": disp,
            "dispersion_source": {"root": "fixture"},
            "snr": {"estimators": ["mad_first_difference", "wavelet_mad"],
                    "bootstrap": {"B": 20, "block_length": 20, "seed": 20260912, "interval": "percentile",
                                  "alpha": 0.05}},
            "budget": {"task_memory_bytes": 3 << 30, "wall_seconds": 3600, "cpu_seconds": 3600,
                       "host_budget_bytes": 8 << 30},
            "roles": {"COORDINATOR": {"share": "all"}}}


@pytest.fixture(scope="module")
def design():
    return D.build_design(inputs())


@pytest.fixture(scope="module")
def reserve(design, tmp_path_factory):
    tape = TAPE.build_tape(design, "fixture-master-entropy-0001", {"seeds": [], "report": {}})
    root = tmp_path_factory.mktemp("reserve") / "fresh"
    manifest = TAPE.materialize_from_tape(tape, design, root)
    return tape, root, manifest


# ------------------------------------------------------------------ design
def test_design_seals_validates_and_refuses_tampering(design):
    assert D.validate_design(design, require_current_code=True) == []
    bad = copy.deepcopy(design)
    bad["design_id"] = "other"
    assert "design digest does not re-derive" in D.validate_design(bad)
    bad = copy.deepcopy(design)
    bad["operators"][4]["arm_role"] = "CANDIDATE"
    bad["operators"][4]["competes"] = True
    assert any("never competes" in p for p in D.validate_design(D.seal_design(bad)))
    bad = copy.deepcopy(design)
    bad["seeds_per_regime"][R1]["n_seeds"] = 9
    assert any("[10, 30]" in p for p in D.validate_design(D.seal_design(bad)))
    bad = copy.deepcopy(design)
    bad["unit_of_analysis"]["time_rows_are_units"] = True
    assert any("never a time row" in p for p in D.validate_design(D.seal_design(bad)))


def test_seed_count_formula_bounds_and_underpowered(design):
    big = D.required_seeds(3.0, 1.0)
    assert big["status"] == "POWERED" and big["n"] == 10
    mid = D.required_seeds(1.0, 1.5)
    assert mid["status"] == "POWERED" and 10 < mid["n"] <= 30 and mid["power_at_n"] >= 0.8
    assert D.required_seeds(1.0, 1.5, family_m=3)["n"] >= mid["n"]
    for args in ((0.1, 2.0), (-1.0, 1.0), (None, None)):
        r = D.required_seeds(*args)
        assert r["status"] == "UNDERPOWERED" and r["n"] == 30
    assert D.required_seeds(None, None)["reason"] == "NO_HISTORICAL_DISPERSION"
    hr = D.spec_sha(OPERATORS[2]["spec"])
    assert design["seeds_per_regime"][R1]["n_seeds"] == 10 and design["seeds_per_regime"][R1]["status"] == "POWERED"
    assert design["seeds_per_regime"][R2]["n_seeds"] == 30
    assert design["seeds_per_regime"][R2]["per_spec"][hr]["status"] == "UNDERPOWERED"


def test_c137_dispersion_extractor(tmp_path):
    root = tmp_path / "lab_evaluation_c137_v1"
    root.mkdir()
    regime = {"declared_snr_db": "10", "family": "sinusoid", "length": 2048, "missingness": "none",
              "perturbation": "white"}
    runs, metrics = [], []
    for i, val in enumerate((1.0, 2.0, 4.0)):
        run = {"run_id": "c137_x", "subject_id": f"sinusoid__white__snr10__none__n2048__v1__seed{11 + i}",
               "content_sha256": "0" * 64, "variable_id": "v", "regime": regime, "operator_kind": OLD,
               "operator_params": {"levels": 2, "threshold_k": 3.0}, "spec_sha256": "1" * 64, "fitted_sha256": None,
               "status": "COMPLETED", "reason": "", "code_sha256": "2" * 64, "cpu_seconds": 0.1,
               "peak_memory_bytes": None}
        runs.append(run)
        for part, v in (("confirmation", val), ("calibration", 99.0)):
            metrics.append({"run_id": "c137_x", "operator_run_sha256": L.row_sha256("df_fact_operator_run", run),
                            "component": "COMPARISON", "partition": part, "metric": "snr_improvement_db",
                            "estimator": "c137", "estimator_params": {}, "value": v, "value_text": None,
                            "status": "COMPLETED", "reason": "", "code_sha256": "2" * 64})
    (root / "df_fact_operator_run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    (root / "df_fact_operator_signal_metric.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    out = D.extract_c137_dispersion(root, metrics=("snr_improvement_db",))
    (row,) = out["rows"]
    assert row["spec"]["kind"] == "trailing_haar_threshold" and row["historical_kind"] == OLD
    assert row["n"] == 3 and row["mean"] == pytest.approx(7 / 3) and row["sd"] == pytest.approx(np.std([1, 2, 4], ddof=1))


# -------------------------------------------------------------------- tape
def test_seed_tape_is_deterministic_and_refuses_collisions(design):
    t1 = TAPE.build_tape(design, "fixture-master-entropy-0001", {"seeds": [], "report": {}})
    t2 = TAPE.build_tape(design, "fixture-master-entropy-0001", {"seeds": [], "report": {}})
    assert t1 == t2 and TAPE.verify_tape(t1, design) == []
    assert TAPE.build_tape(design, "fixture-master-entropy-0002", {"seeds": [], "report": {}})["tape_sha256"] != t1["tape_sha256"]
    first = t1["regimes"][0]["seeds"][0]
    with pytest.raises(TAPE.TapeRefusal, match="collision"):
        TAPE.build_tape(design, "fixture-master-entropy-0001", {"seeds": [first["seed"]], "report": {}})
    with pytest.raises(TAPE.TapeRefusal, match="collision"):
        TAPE.build_tape(design, "fixture-master-entropy-0001", {"seeds": [first["derived_seeds"]["noise"]], "report": {}})
    bad = copy.deepcopy(t1)
    bad["regimes"][0]["seeds"][0]["seed"] += 1
    assert "tape digest does not re-derive" in TAPE.verify_tape(bad, design)


def test_prior_seed_scan_reads_fields_and_unit_ids(tmp_path):
    bank = tmp_path / "synthetic_bank_c128_v1" / "u1"
    bank.mkdir(parents=True)
    (bank / "UNIT.json").write_text(json.dumps({"seed": 11, "derived_seeds": {"clean": 501, "noise": 502, "missing": 503},
                                               "unit_id": "x__seed11"}))
    lab = tmp_path / "lab_evaluation_c137_v1"
    lab.mkdir()
    (lab / "rows.jsonl").write_text(json.dumps({"subject_id": "a__seed12", "bootstrap_seed": 7}) + "\n"
                                    + json.dumps({"derived_seeds": {"clean": 9001}}) + "\n")
    out = TAPE.scan_prior_seeds(tmp_path, include_code_constants=False)
    assert {11, 12, 7, 501, 502, 503, 9001} <= set(out["seeds"])
    assert out["report"]["roots"]["synthetic_bank_c128_v1"]["files_scanned"] == 1
    assert out["report"]["roots"]["snr_calibration_c163_v1"]["present"] is False


def test_materialized_reserve_is_write_once_and_complete(design, reserve):
    tape, root, manifest = reserve
    assert manifest["complete"] and len(manifest["units"]) == 40 and manifest["mode"] == D.FRESH_MODE
    u = root / manifest["units"][0]["unit_id"]
    for f in ("clean_signal.npy", "additive_noise.npy", "observed_signal.npy", "missing_mask.npy", "events.json",
              "UNIT.json", "CONTRACT.json"):
        assert (u / f).is_file()
    rec = json.loads((u / "UNIT.json").read_text())
    assert set(rec["partitions"]) == {"train", "calibration", "confirmation"}
    with pytest.raises(TAPE.TapeRefusal, match="write-once"):
        TAPE.materialize_from_tape(tape, design, root)


# ------------------------------------------------------------------ worker
def _unit(root, manifest, regime_key, i=0):
    return root / [u for u in manifest["units"] if u["regime_key"] == regime_key][i]["unit_id"]


def _job(design, unit_dir, root_dir, attempt_dir, mode=D.FRESH_MODE):
    return {"design": design, "design_sha256": design["design_sha256"], "unit_dir": str(unit_dir),
            "run_id": "d2v2_test", "mode": mode, "root_dir": str(root_dir), "attempt_dir": str(attempt_dir)}


@pytest.fixture(scope="module")
def two_units(design, reserve, tmp_path_factory):
    _, root, manifest = reserve
    out = []
    for rk in (R1, R2):
        rows = []
        adir = tmp_path_factory.mktemp("attempt")
        summary = W.process_unit(_job(design, _unit(root, manifest, rk), adir, adir), rows.append)
        out.append((summary, rows))
    return out


def test_worker_two_units_public_api_oracle_detected_rows_valid(two_units):
    for summary, rows in two_units:
        assert summary["mode"] == D.FRESH_MODE and not summary["root_invalidation"], summary["root_invalidation_reasons"]
        assert [o["outcome"] for o in summary["oracle_detection"]] == ["DETECTED"]
        (audit,) = summary["wavelet_audits"]
        assert audit["failed_checks"] == [] and audit["checks"]["prefix_every_t"]["prefixes"] > 0
        assert summary["memory_estimate_bytes"] > W.BASE_PROCESS_BYTES
        for obj in rows:
            assert A.validate_proposed_row(obj["table"], obj["row"]) == [], obj
        den = [o["row"] for o in rows if o["table"] == W.DENOISING_TABLE]
        assert {r["branch"] for r in den} == {"RAW", "TRANSFORMED", "RESIDUAL", "COMPARISON", "COST"}
        assert {r["arm_role"] for r in den} == set(D.ARM_ROLES)
        assert len({r["content_sha256"] for r in den}) == 1 and len({r["seed"] for r in den}) == 1
        snr = [o["row"] for o in rows if o["table"] == W.SNR_TABLE]
        wm = [r for r in snr if r["estimator"] == "wavelet_mad"]
        assert {r["partition"]: r["status"] for r in wm if r["partition"] != "train"} == \
            {"confirmation": "NOT_APPLICABLE", "calibration": "NOT_APPLICABLE"}
        (conf,) = [r for r in snr if r["estimator"] == "mad_first_difference" and r["partition"] == "confirmation"]
        # The fresh seeds derive from the design digest, which binds the lab code digests, so the realization of
        # a unit changes whenever that code changes. The first version asserted one realization's outcome (steps at
        # 0 dB not identifiable) and broke when the guard-removal code was merged. What must hold for every
        # realization is the typing: an estimate either completes with its values or is NOT_IDENTIFIABLE with a reason.
        if conf["status"] == "COMPLETED":
            assert conf["snr_db_hat"] is not None and conf["abs_error_db"] is not None
        else:
            assert conf["status"] == "INCONCLUSIVE" and conf["identifiability"] == "NOT_IDENTIFIABLE" and conf["reason"]
        if summary["unit_id"].startswith("sinusoid"):
            assert conf["status"] == "COMPLETED"   # 10 dB on a smooth sinusoid is identifiable for any seed
        (train,) = [r for r in snr if r["estimator"] == "wavelet_mad" and r["partition"] == "train"]
        assert train["contract_state"] == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"


def _leaky(fitted, snapshot, oracle_mode=False):
    y, a, r = OPS.transform_batch(fitted, snapshot, oracle_mode=oracle_mode)
    y2 = y.copy()
    y2[:-1] = y[1:]
    return y2, a, r


def test_wavelet_audit_invalidates_the_root_on_a_leaking_stand_in(design, reserve, tmp_path):
    _, root, manifest = reserve
    ud = _unit(root, manifest, R1, 1)
    u = W.LAB.load_unit(ud)
    fit_sl = W.LAB.train_fit_slice(u["observed"], u["rec"]["partitions"]["train"])
    fitted = W.fit_public(u, OPERATORS[2]["spec"], OPS.FROZEN_PREVIOUS_PARTITION, fit_sl)
    audit = W.audit_trailing_haar(u, fitted, transform_batch=_leaky, n_random_cuts=2, chunk_sizes=(64,))
    assert audit["root_invalidation"] and "prefix_every_t" in audit["failed_checks"]
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps({"mode": D.FRESH_MODE, "run_id": "r",
                                                            "design_sha256": design["design_sha256"]}))
    summary = W.process_unit(_job(design, ud, run_root, tmp_path / "attempt"), lambda o: None, transform_batch=_leaky)
    assert summary["root_invalidation"]
    marker = W.invalidate_root(run_root, summary)
    assert marker.is_file() and json.loads(marker.read_text())["scope"] == "WHOLE_ROOT"
    with pytest.raises(A.AdjudicationRefusal, match="invalidated as a whole"):
        A.load_fresh_root(run_root, design)


def test_a_unit_refused_whole_by_the_missing_data_rule_abstains_without_invalidating(design, reserve, tmp_path,
                                                                                     monkeypatch):
    """C172 run: every MCAR unit of the C137 bank had a longest complete TRAIN stretch of 16-43 rows. All arms were
    REFUSED, and the first worker turned the unevaluated oracle into a whole-root invalidation."""
    _, root, manifest = reserve
    monkeypatch.setattr(W.LAB, "train_fit_slice", lambda observed, train: slice(0, W.MIN_FIT_ROWS - 1))
    rows = []
    summary = W.process_unit(_job(design, _unit(root, manifest, R1), tmp_path, tmp_path / "a"), rows.append)
    assert summary["unit_evaluable"] is False and not summary["root_invalidation"], summary["root_invalidation_reasons"]
    assert [o["outcome"] for o in summary["oracle_detection"]] == ["NOT_EVALUATED"]
    den = [o["row"] for o in rows if o["table"] == W.DENOISING_TABLE]
    assert den and {r["status"] for r in den} == {"REFUSED"} and {r["arm_role"] for r in den} == set(D.ARM_ROLES)


def test_an_unevaluated_oracle_on_an_evaluated_unit_still_invalidates_the_root(design, reserve, tmp_path, monkeypatch):
    _, root, manifest = reserve
    real_fit = W.fit_public

    def fit(u, spec, fit_mode, fit_sl):
        if spec["kind"] == "centered_mean_oracle":
            raise W.OPS.OperatorRefusal("REFUSED: stand-in refusal of the oracle only")
        return real_fit(u, spec, fit_mode, fit_sl)

    monkeypatch.setattr(W, "fit_public", fit)
    summary = W.process_unit(_job(design, _unit(root, manifest, R1), tmp_path, tmp_path / "a"), lambda o: None)
    assert summary["unit_evaluable"] is True
    assert summary["root_invalidation"] and "ORACLE_NOT_EVALUATED" in summary["root_invalidation_reasons"]


def test_fresh_unit_cannot_be_relabelled_historical(design, reserve, tmp_path):
    _, root, manifest = reserve
    with pytest.raises(W.UnitRefusal, match="never re-labelled"):
        W.process_unit(_job(design, _unit(root, manifest, R1), tmp_path, tmp_path, mode=D.HISTORICAL_MODE),
                       lambda o: None)


def test_run_units_isolated_process_writes_durable_terminal(design, reserve, tmp_path):
    _, root, manifest = reserve
    dfile = tmp_path / "DESIGN.json"
    D.write_design(design, dfile)
    out = tmp_path / "d2_fresh_run"
    r = W.run_units(out, dfile, [_unit(root, manifest, R1, 2)], D.FRESH_MODE, task_memory_bytes=4 << 30,
                    wall_seconds=600, cpu_seconds=600, heartbeat_seconds=1.0, mechanism="PRLIMIT_AS")
    assert [u["status"] for u in r["units"]] == ["COMPLETED"] and not r["root_invalidated"], r
    (term,) = [json.loads(p.read_text()) for p in (out / "terminals").glob("*.json")]
    assert term["status"] == "COMPLETED" and term["output_sha256"] and term["planned_peak_bytes"] > 0
    loaded = A.load_fresh_root(out, design)
    assert loaded["terminal_counts"] == {"COMPLETED": 1} and loaded["non_governing_rows_excluded"] > 0
    assert {r["partition"] for r in loaded["denoising"]} <= {"confirmation", "all"}
    decisions = A.decide_denoising(loaded["denoising"], design)
    states = {d["arm_role"]: d["decision"] for d in decisions}
    assert states["NON_CAUSAL_ORACLE_CONTROL"] == "CONTROL_NOT_AN_ARM"
    assert states["CANDIDATE"] == "NOT_IDENTIFIABLE"          # one seed < 10 required
    assert A.decision_rows(decisions, "d2v2_test")
    again = W.run_units(out, dfile, [_unit(root, manifest, R1, 2)], D.FRESH_MODE, resume=True, mechanism="PRLIMIT_AS")
    assert again["units"][0].get("resumed_skip")


# ------------------------------------------------------------ adjudication
def _cell_regime(i=0):
    return D.regime_of_cell(CELLS[i])


def srow(unit, err, covers=1.0, ident="ESTIMATED", partition="confirmation", mode=D.FRESH_MODE, var=0,
         est="mad_first_difference", regime=None, design_sha="d" * 64):
    true = 10.0
    ok = ident == "ESTIMATED"
    return {"run_id": "r", "mode": mode, "design_sha256": design_sha, "tape_sha256": "e" * 64, "unit_id": unit,
            "seed": int(unit.split("_")[-1]), "content_sha256": "f" * 64, "regime": regime or _cell_regime(),
            "variable_index": var, "estimator": est, "contract_state": "TRAIN_SEGMENT_AGGREGATE",
            "partition": partition, "segment_start": 0, "segment_end": 100, "snr_db_hat": true + err if ok else None,
            "ci_low_db": 0.0 if ok else None, "ci_high_db": 20.0 if ok else None, "ci_lower_unbounded": False,
            "ci_upper_unbounded": False, "true_snr_db": true, "error_db": err if ok else None,
            "abs_error_db": abs(err) if ok else None, "ci_covers_true": covers if ok else None,
            "identifiability": ident, "status": "COMPLETED" if ok else "INCONCLUSIVE",
            "reason": "" if ok else "not identifiable", "code_sha256": "0" * 64}


def _snr(design, errs, **kw):
    return [srow(f"u_{100 + i}", e, design_sha=design["design_sha256"], **kw) for i, e in enumerate(errs)]


def test_snr_rules_including_cancellation_and_refusals(design):
    (d,) = A.decide_snr(_snr(design, [0.2, -0.2] * 5), design)
    assert d["decision"] == "SNR_CALIBRATED_FOR_REGIME"
    (d,) = A.decide_snr(_snr(design, [3.0, -3.0] * 5), design)
    assert d["decision"] == "SNR_REJECTED" and abs(d["evidence"]["mean_signed_error_db_for_information_only"]) < 1e-12
    assert d["evidence"]["mean_abs_error_db"] == pytest.approx(3.0)
    rows = _snr(design, [0.2] * 10)
    for r in rows[:2]:
        r.update(srow(r["unit_id"], 0.0, ident="NOT_IDENTIFIABLE", design_sha=design["design_sha256"]))
    (d,) = A.decide_snr(rows, design)
    assert d["decision"] == "SNR_NOT_IDENTIFIABLE"
    rows = _snr(design, [0.2] * 10)
    for r in rows[::2]:
        r["ci_covers_true"] = 0.0
    (d,) = A.decide_snr(rows, design)
    assert d["decision"] == "SNR_REGIME_LIMITED"
    for bad, text in ((dict(partition="calibration"), "calibration is diagnostic"),
                      (dict(mode=D.HISTORICAL_MODE), "only FRESH_CONFIRMATION")):
        rows = _snr(design, [0.2] * 10)
        rows[0].update(bad)
        with pytest.raises(A.AdjudicationRefusal, match=text):
            A.decide_snr(rows, design)
    rows = _snr(design, [0.2] * 10)
    with pytest.raises(A.AdjudicationRefusal, match="duplicated unit-level grain"):
        A.decide_snr(rows + [dict(rows[0])], design)
    rows[0]["t"] = 5
    with pytest.raises(A.AdjudicationRefusal, match="time grain"):
        A.decide_snr(rows, design)


GOOD = {"snr_improvement_db": 3.0, "rmse_ratio": 0.5, "residual_signal_share": 0.01, "delay_samples": 1.0,
        "extreme_retention": 0.9, "extreme_retention_raw": 0.95, "distortion_ratio": 0.3}


def drow(design, unit, op, metric, value, partition="confirmation", branch="COMPARISON", status="COMPLETED",
         mode=D.FRESH_MODE, regime=None):
    return {"run_id": "r", "mode": mode, "design_sha256": design["design_sha256"], "tape_sha256": "e" * 64,
            "unit_id": unit, "seed": int(unit.split("_")[-1]), "content_sha256": "f" * 64,
            "regime": regime or _cell_regime(), "variable_id": "v0", "variable_index": 0, "arm_role": op["arm_role"],
            "operator_kind": op["kind"], "operator_params": op["spec"]["params"], "spec_sha256": op["spec_sha256"],
            "fit_mode": op["fit_mode"], "fitted_sha256": None, "partition": partition, "branch": branch,
            "metric": metric, "estimator": "x", "value": value if status == "COMPLETED" else None, "status": status,
            "reason": "" if status == "COMPLETED" else "x", "code_sha256": "0" * 64, "operator_code_sha256": "0" * 64}


def _den(design, n_seeds=10, override=None, regime=None):
    rows = []
    for i in range(n_seeds):
        unit = f"u_{200 + i}"
        for op in design["operators"]:
            vals = dict(GOOD, snr_improvement_db=3.0 + 0.1 * i)
            if override:
                vals.update(override(op, i) or {})
            for m, v in vals.items():
                rows.append(drow(design, unit, op, m, v, regime=regime))
            rows.append(drow(design, unit, op, "cpu_seconds_per_1000_samples", 0.01, partition="all", branch="COST",
                             regime=regime))
    return rows


def _by(decisions, kind):
    return next(d for d in decisions if d["subject"] == kind)


def test_denoising_rules_each_state_and_seed_level_checks(design):
    ds = A.decide_denoising(_den(design), design)
    assert _by(ds, "ewma")["decision"] == "LAB_CALIBRATED"
    assert _by(ds, "centered_mean_oracle")["decision"] == "CONTROL_NOT_AN_ARM"
    ds = A.decide_denoising(_den(design, override=lambda op, i: {"extreme_retention": 0.2}
                                 if op["kind"] == "ewma" and i == 0 else None), design)
    d = _by(ds, "ewma")
    assert d["decision"] == "LAB_REJECTED" and "u_200" in " ".join(d["reasons"])
    ds = A.decide_denoising(_den(design, override=lambda op, i: {"snr_improvement_db": -0.5}
                                 if op["kind"] == "ewma" and i == 0 else None), design)
    assert _by(ds, "ewma")["decision"] == "REGIME_LIMITED"
    ds = A.decide_denoising(_den(design, override=lambda op, i: {"snr_improvement_db": 0.1 + 0.01 * i}
                                 if op["kind"] == "ewma" else None), design)
    assert _by(ds, "ewma")["decision"] == "LAB_REJECTED"
    ds = A.decide_denoising(_den(design, n_seeds=9), design)
    assert _by(ds, "ewma")["decision"] == "NOT_IDENTIFIABLE"
    ds = A.decide_denoising(_den(design, n_seeds=30, regime=_cell_regime(1)), design)
    assert _by(ds, "trailing_haar_threshold")["decision"] == "UNDERPOWERED"
    assert _by(ds, "ewma")["decision"] == "LAB_CALIBRATED"
    rows = _den(design)
    rows[0] = dict(rows[0], partition="calibration")
    with pytest.raises(A.AdjudicationRefusal, match="calibration is diagnostic"):
        A.decide_denoising(rows, design)
    rows = _den(design)
    rows[0] = dict(rows[0], mode=D.HISTORICAL_MODE)
    with pytest.raises(A.AdjudicationRefusal, match="only FRESH_CONFIRMATION"):
        A.decide_denoising(rows, design)
    for d in A.decision_rows(A.decide_denoising(_den(design), design), "r"):
        assert d["externally_reviewed"] is False


def test_historical_comparison_classifies_flip_and_keeps_counts_apart(design):
    regime = {"declared_snr_db": "10", "family": "sinusoid", "length": 2048, "missingness": "none",
              "perturbation": "white"}
    op = design["operators"][2]
    runs, metrics, re_rows = [], [], []
    for i in range(3):
        subject = f"u_{300 + i}"
        run = {"run_id": "c137_x", "subject_id": subject, "content_sha256": "f" * 64, "variable_id": "v0",
               "regime": regime, "operator_kind": OLD, "operator_params": op["spec"]["params"],
               "spec_sha256": "1" * 64, "fitted_sha256": None, "status": "COMPLETED", "reason": "",
               "code_sha256": "2" * 64, "cpu_seconds": 0.1, "peak_memory_bytes": None}
        runs.append(run)
        for part in ("calibration", "confirmation"):
            for m, v in (("support", 100.0), ("snr_improvement_db", 2.0), ("rmse_ratio", 0.5)):
                metrics.append({"operator_run_sha256": L.row_sha256("df_fact_operator_run", run), "partition": part,
                                "metric": m, "value": v, "status": "COMPLETED"})
            for m, v in (("support", 90.0), ("snr_improvement_db", 0.1), ("rmse_ratio", 0.99)):
                re_rows.append(drow(design, subject, op, m, v, partition=part, mode=D.HISTORICAL_MODE, regime=regime))
    hist = [{"operator_kind": OLD, "operator_params": op["spec"]["params"], "regime": regime,
             "decision": "LAB_CALIBRATED"}]
    rep = A.compare_with_historical(hist, runs, metrics, re_rows, design)
    (row,) = rep["rows"]
    assert row["flipped"] and row["flip_cause"] == "TRANSFORMED_RANGE" and row["reanalysis_decision"] == "LAB_REJECTED"
    assert row["historical_operator_kind"] == OLD and row["operator_kind"] == "trailing_haar_threshold"
    assert rep["status_counts"]["historical"]["RESULT"] == 3 and rep["status_counts"]["reanalysis"]["RESULT"] == 3
    assert rep["units_equal"] and rep["truth_content_equal"] and rep["grants_consumption"] is False
    assert A.historical_rows(rep, "r", "c137_x")
    with pytest.raises(A.AdjudicationRefusal, match="HISTORICAL rows only"):
        A.compare_with_historical(hist, runs, metrics, [dict(re_rows[0], mode=D.FRESH_MODE)], design)
