"""O1–O3: the successor verifier re-derives every calibration from its simulations, binds each
attempt to its recorded job and the freeze, re-decides each contrast from conserved evidence,
and refuses to rebuild what is missing or altered."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("df_utility_harness")
RV = _load("df_utility_reverify")
ops = _load("df_d3_operators")
contract = _load("df_d3_contract")
T = _load("test_df_utility_harness") if False else None

CID = "fab__v0__mad_extremes_trailing__transformed"
PLAN = {"generator": "white_null", "n": 900, "bound_confidence": 0.5,
        "n_sims": H.sims_required_for_zero(0.05, 0.5)}      # 14 simulations, one contrast
MAD = ops.build("mad_extremes_trailing")


def _series(n=900, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        drive = 0.0
        if t > 20:
            w = x[t - 16:t]
            med = np.median(w)
            mad = np.median(np.abs(w - med)) or 1.0
            drive = -0.6 * np.sign(x[t - 1] - med) * min(3.0, abs(x[t - 1] - med) / mad)
        x[t] = x[t - 1] + drive + rng.normal(0, 1.0)
    return x


def _attempt(path, body, name, job, extra_result=None):
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body).hexdigest()
    (path / name).write_bytes(body)
    (path / "job.json").write_text(json.dumps({**job, "attempt_dir": str(path)}, default=str))
    (path / "result.json").write_text(json.dumps({"status": "COMPLETED", "reason": "", "output_file": name,
                                                  "output_sha256": digest, **(extra_result or {})}))
    (path / "outcome.json").write_text(json.dumps({"status": "COMPLETED", "verified": {"output_sha256": digest},
                                                   "summary": {"outcome": (extra_result or {}).get("outcome", "COMPLETED"),
                                                               "reason": "", "cost": {"cpu_seconds": 1.0}}}))


def build_root(tmp_path, monkeypatch):
    base = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0,
                      seed=7, family=(CID,), min_rows_per_block=30, calibration_plan=dict(PLAN))
    rec = H.calibrate(base, MAD, plan=PLAN, seed=11)
    p = base.with_calibration(rec)
    s = H.series(_series())
    elig = {"freeze_sha256": "f" * 64, "design_sha256": "d" * 64,
            "cells": {("fab", "v0", MAD.KIND): {"verdict": "MECHANICALLY_ACCEPTED",
                                               "spec_sha256": contract.spec_sha256(MAD.describe())}}}
    out = H.contrast(s, MAD, p, contrast_id=CID, eligibility=elig, unit="fab", variable="v0")
    assert "delta_lower" in out
    root = tmp_path / "run"
    root.mkdir()
    (root / "FREEZE.pre.json").write_text(json.dumps({"plan": PLAN, "units": [{"unit": "fab", "variable": "v0"}]}))
    (root / "FREEZE.json").write_text(json.dumps({"run_id": "t", "code_identity": {"kind": "git_commit", "value": "0" * 40},
                                                  "protocols": {MAD.KIND: p.sealed()}}))
    (root / "REPORT.json").write_text(json.dumps({"run_id": "t"}))
    cal_job = {"kind": "calibrate", "contrast_id": CID, "operator": MAD.KIND, "protocol": base.sealed(),
               "plan": PLAN, "seed": 11}
    _attempt(root / "attempts" / f"calibrate__{MAD.KIND}", json.dumps(rec, sort_keys=True, default=str).encode(),
             "calibration.json", cal_job)
    con_job = {"kind": "contrast", "contrast_id": CID, "unit": "fab", "variable": "v0", "operator": MAD.KIND,
               "protocol": p.sealed(), "series": {"values": [float(v) for v in s["values"]]}, "eligibility": "x"}
    _attempt(root / "attempts" / CID, json.dumps(out, sort_keys=True, default=H._jsonable).encode(),
             "contrast.json", con_job, {"outcome": out["outcome"]})
    # the harness that applied is this one (the fake commit resolves to the current file)
    monkeypatch.setattr(RV, "file_at", lambda repo, commit, path: (HERE / Path(path).name).read_bytes())
    return root, rec, out


def test_O1_O3_an_intact_run_is_reverified_with_derived_bounds_and_no_decision_delta(tmp_path, monkeypatch):
    root, rec, out = build_root(tmp_path, monkeypatch)
    r = RV.reverify(root, tmp_path)
    assert r["all_verified"] is True and r["decision_delta"] == []
    cal = r["calibrations"][MAD.KIND]
    assert cal["problems"] == [] and cal["derived"]["advances"] == rec["advances"]
    assert cal["derived"]["decision_bound"] == cal["derived"]["upper_bound"] and cal["supports"]["decision"] is True
    assert cal["supports"]["under_harness"] == H.harness_sha256()
    row = r["table"][0]
    assert row["reverified_outcome"] == out["outcome"] and row["rows_paired"] == out["coverage"]["rows_paired"]
    assert row["loss_raw_mean"] > 0 and row["null_scope"]["n_sims"] == PLAN["n_sims"]
    md = RV.markdown(r)
    assert "| `fab` | v0 |" in md and "Decision delta: 0" in md


def test_O1_a_calibration_whose_stated_bound_is_not_derived_supports_nothing_and_the_contrast_is_inconclusive(tmp_path, monkeypatch):
    root, rec, out = build_root(tmp_path, monkeypatch)
    attempt = root / "attempts" / f"calibrate__{MAD.KIND}"
    tampered = json.dumps({**rec, "upper_bound": 0.0}, sort_keys=True, default=str).encode()
    d = hashlib.sha256(tampered).hexdigest()
    (attempt / "calibration.json").write_bytes(tampered)          # even with matching digests
    (attempt / "result.json").write_text(json.dumps({"status": "COMPLETED", "reason": "", "output_file": "calibration.json", "output_sha256": d}))
    (attempt / "outcome.json").write_text(json.dumps({"status": "COMPLETED", "verified": {"output_sha256": d}, "summary": {}}))
    r = RV.reverify(root, tmp_path)
    assert r["all_verified"] is False
    assert any("derived" in x for x in r["calibrations"][MAD.KIND]["problems"])
    assert r["contrasts"][CID]["reverified_outcome"] == H.INCONCLUSIVE_UNCALIBRATED
    assert r["decision_delta"] and r["decision_delta"][0]["original"] == out["outcome"]


def test_O2_altered_evidence_or_a_job_not_bound_to_the_freeze_is_inconclusive_never_rebuilt(tmp_path, monkeypatch):
    root, rec, out = build_root(tmp_path, monkeypatch)
    attempt = root / "attempts" / CID
    body = (attempt / "contrast.json").read_bytes()
    (attempt / "contrast.json").write_bytes(body.replace(b'"delta_mean"', b'"delta_mean_"'))
    r = RV.reverify(root, tmp_path)
    assert r["contrasts"][CID]["reverified_outcome"] == "INCONCLUSIVE_EVIDENCE_UNVERIFIED"
    assert any("bytes" in x for x in r["contrasts"][CID]["problems"]) and r["all_verified"] is False
    (attempt / "contrast.json").write_bytes(body)
    job = json.loads((attempt / "job.json").read_text())
    job["protocol"] = {**job["protocol"], "protocol_sha256": "9" * 64}
    (attempt / "job.json").write_text(json.dumps(job))
    r = RV.reverify(root, tmp_path)
    assert any("frozen" in x for x in r["contrasts"][CID]["problems"])
    (attempt / "job.json").unlink()
    r = RV.reverify(root, tmp_path)
    assert any("no recorded job" in x for x in r["contrasts"][CID]["problems"])


def test_O2_a_run_resumed_under_another_commit_reports_the_limited_diff(tmp_path, monkeypatch):
    root, rec, out = build_root(tmp_path, monkeypatch)
    (root / "REPORT.json").write_text(json.dumps({"run_id": "t", "code_identity_now": {"kind": "git_commit", "value": "1" * 40}}))
    import subprocess
    monkeypatch.setattr(RV.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "docs/x.md\ntools/df_utility_run.py\n"})())
    r = RV.reverify(root, tmp_path)
    rd = r["resume_diff"]
    assert rd["files_changed"] == ["docs/x.md", "tools/df_utility_run.py"]
    assert rd["scientific_code_unchanged"] is True and rd["harness_unchanged"] is True
