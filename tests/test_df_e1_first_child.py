"""RP50 (dictum F1): the FIRST child of a fresh run must reach a verified outcome.

The defect: `run_isolated` asked the closure for a verdict before it had written `outcome.json`, and
the closure needs that record to find an attempt — so the first child of a fresh run was refused with
"absent: no attempt with a record", while a copy that already had an outcome was accepted. The route
tests never reached it because they replaced `run_isolated`.

These rules therefore use the REAL `run_isolated`, the REAL worker subprocess and the REAL verifier,
on a small but genuine auto-encoder and fit: weights are written, predictions are stored, the update
counter runs, and the unit survives a reload and a resume. The panel is a synthetic fixture, declared
as such; nothing here is an experiment.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_e1_pilot")
C = _load("df_e1_close")
E = _load("df_mod_e0")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A SYNTHETIC panel (software fixture) and a sealed design over it, prepared once."""
    tmp = tmp_path_factory.mktemp("first-child")
    rng = np.random.default_rng(11)
    n = 5 * P.DAY
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    base = np.sin(2 * np.pi * np.arange(n) / P.DAY) + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({c: base * (i + 1) + rng.normal(size=n) * 0.05 + i for i, c in enumerate(cols)})
    ts = pd.date_range("2009-02-01 00:00", periods=n, freq="min")
    frame.insert(0, "timestamp_label", ts.strftime("%d/%m/%Y %H:%M:%S"))
    panel = tmp / "panel.parquet"
    frame.to_parquet(panel)
    design = P.seal(window=30, horizon=10, dev_train_days=3, dev_val_days=1, seeds=(1,), max_updates=20,
                    ae_updates=20, batch=32, patience_epochs=2, pilot_updates=5,
                    declared_task={"context_physical_seconds": 1800, "horizon_physical_seconds": 600,
                                   "purge": 40, "usable_windows_all_targets_valid": {}})
    design.pop("design_sha256")
    design["governed_bytes"] = {"path": str(panel), "sha256": P.sha_file(panel)}
    design["dev_subpartition"]["rows"] = [0, 4 * P.DAY]
    design["design_sha256"] = E.sha_obj(design)
    root = tmp / "root"
    root.mkdir()
    (root / "DESIGN.json").write_text(json.dumps(design, indent=1))
    data = P.prepare(design, root)
    return {"tmp": tmp, "root": root, "design": design, "data": data, "panel": panel, "completed": None}


def _job(run, cell_id, **over):
    cell = next(c for c in run["design"]["pilots"] + run["design"]["cells"] if c["cell_id"] == cell_id)
    job = {"kind": cell["kind"], "cell_id": cell_id, "seed": cell["seed"], "max_updates": cell.get("max_updates", 0),
           "design": run["design"], "data_npz": str(run["root"] / "DATA.npz"), "data_sha256": run["data"]["data_sha256"],
           "run_id": "rp50", "role": cell.get("role", "CELL")}
    if "regime" in cell:
        job["regime"] = cell["regime"]
    if cell.get("depends_on"):
        job["pretrained_npz"] = str(run["root"] / "attempts" / cell["depends_on"] / "detector_pretrained.npz")
    job.update(over)
    return job


def _completed(run, tmp_path):
    """A root whose R0_s1 attempt really ran, built once and reused: the dependent rules must not
    depend on the ORDER in which the rules above happened to run."""
    if run.get("completed") is None:
        base = run["tmp"] / "completed-root"
        shutil.copytree(run["root"], base)
        job = _job(run, "R0_s1", data_npz=str(base / "DATA.npz"))
        out = P.run_isolated(job, attempt_dir=base / "attempts" / "R0_s1", assigned_bytes=3 << 30,
                             wall_seconds=900, cpu_seconds=900)
        assert out["outcome"] == "COMPLETED", out.get("refusal")
        run["completed"] = base
    root = tmp_path / "copy"
    shutil.copytree(run["completed"], root)
    return root


@pytest.mark.parametrize("cell_id", ["ae_s1", "R0_s1"])
def test_RP50_the_first_child_of_a_fresh_run_reaches_a_verified_outcome(run, cell_id):
    """No outcome exists before this call: that is exactly the case the defect refused."""
    attempt = run["root"] / "attempts" / cell_id
    assert not (attempt / "outcome.json").exists()
    out = P.run_isolated(_job(run, cell_id), attempt_dir=attempt, assigned_bytes=3 << 30,
                         wall_seconds=900, cpu_seconds=900)
    assert out["outcome"] == "COMPLETED", out.get("refusal") or out.get("reason")
    assert out["score"] is not None and out["cost"]["cpu_seconds"] > 0
    record = json.loads((attempt / "outcome.json").read_text())
    assert record["score_state"] == "VERIFIED"
    phases = record["phases"]
    assert phases["attempt_started_at"] <= phases["parent_record_at"] <= phases["verdict_at"]
    assert phases["verdict_phase"] == "FRESH"
    if cell_id == "ae_s1":
        assert (attempt / "detector_pretrained.npz").is_file() and (attempt / "decoder.npz").is_file()
        assert out["score"]["pretraining"]["updates"] > 0
    else:
        assert (attempt / "weights.weights.h5").is_file()
        stored = dict(np.load(attempt / "arrays.npz"))
        assert stored["validation_pred"].size > 0 and out["score"]["training"]["updates"] > 0


def test_RP50_a_provisional_parent_record_is_never_read_as_a_score(run, tmp_path):
    root = _completed(run, tmp_path)
    attempt = root / "attempts" / "R0_s1"
    record = json.loads((attempt / "outcome.json").read_text())
    record["score_state"] = "PENDING_VERDICT"
    record["phases"]["verdict_at"] = None
    (attempt / "outcome.json").write_text(json.dumps(record))
    verdict = C.verify_unit(C.register(root), "R0_s1", do_replay=False)
    assert not verdict["verified"] and verdict["metrics"] == C.REFUSED
    assert any("verdict was never recorded" in p for p in verdict["problems"])


def test_RP50_the_unit_survives_a_reload_and_a_resume_with_the_same_verdict(run, tmp_path):
    root = _completed(run, tmp_path)
    attempt = root / "attempts" / "R0_s1"
    first = json.loads((attempt / "outcome.json").read_text())
    job = json.loads((attempt / "job.json").read_text())
    job.pop("attempt_dir", None)
    job["data_npz"] = str(root / "DATA.npz")
    again = P.run_isolated(job, attempt_dir=attempt, assigned_bytes=3 << 30, wall_seconds=900, cpu_seconds=900)
    assert again["resumed"] and again["outcome"] == "COMPLETED" and again["score"] is not None
    record = json.loads((attempt / "outcome.json").read_text())
    assert record["score_state"] == "VERIFIED" and record["phases"]["verdict_phase"] == "RESUME"
    assert record["verified"]["output_sha256"] == first["verified"]["output_sha256"]
    # the closure, run afterwards, agrees with the verdict the runner recorded
    verdict = C.verify_unit(C.register(root), "R0_s1", do_replay=False)
    assert verdict["metrics"] == C.VERIFIED and verdict["verified"]


@pytest.mark.parametrize("case", ["altered_score", "incomplete_result", "child_failure"])
def test_RP50_a_broken_child_is_refused_with_its_cost(run, tmp_path, case):
    root = _completed(run, tmp_path)
    attempt = root / "attempts" / "R0_s1"
    job = _job(run, "R0_s1", data_npz=str(root / "DATA.npz"))
    if case == "altered_score":
        body = json.loads((attempt / "cell.json").read_text())
        body["scores"]["validation"]["model"]["mae_mean"] = 999.0
        text = json.dumps(body, indent=1, sort_keys=True, default=float)
        (attempt / "cell.json").write_text(text)
        digest = __import__("hashlib").sha256(text.encode()).hexdigest()
        result = json.loads((attempt / "result.json").read_text())
        result.update(output_sha256=digest, rows_written=text.count("\n") + (0 if text.endswith("\n") else 1))
        (attempt / "result.json").write_text(json.dumps(result))
        record = json.loads((attempt / "outcome.json").read_text())
        record["verified"]["output_sha256"] = digest
        (attempt / "outcome.json").write_text(json.dumps(record))
    elif case == "incomplete_result":
        (attempt / "arrays.npz").unlink()
    else:
        shutil.rmtree(attempt)
        job["max_updates"] = -1                                   # the worker refuses this job and exits non-zero
    out = P.run_isolated(job, attempt_dir=attempt, assigned_bytes=3 << 30, wall_seconds=300, cpu_seconds=300)
    assert out["score"] is None and out["outcome"] in ("SCORE_UNVERIFIED", "UNCERTAIN", "RESOURCE_EXCEEDED")
    assert out.get("cost") is not None
    record = json.loads((attempt / "outcome.json").read_text())
    assert record["score_state"] == "REFUSED" and record["phases"]["verdict_at"]


def test_RP50_the_worker_works_from_its_delivery_alone(run, tmp_path):
    """Portability: with the design's original panel path gone, a run that holds its delivered bytes
    still registers and verifies. Nothing is read from the coordinator's paths."""
    root = _completed(run, tmp_path)
    delivered = root / "delivered.parquet"
    shutil.copy2(run["panel"], delivered)
    (root / "DELIVERIES.json").write_text(json.dumps({
        "schema": "df_e1_governed_acquisition.v1", "design_sha256": run["design"]["design_sha256"],
        "lake": "public_panels", "resource": "uci_235_individual_household_power/panel.parquet",
        "units": {"prepare": {"campaign_key": "k", "campaign_sha256": "c" * 64, "delivery_id": "d" * 32,
                              "sha256": run["data"]["panel_sha256"], "at": "2026-09-20T00:00:00Z", "host": "worker",
                              "code_identity": {"kind": "git_commit", "value": "a" * 40},
                              "path": str(delivered), "cached": False, "verification_state": "VERIFIED_TRANSFER"}}}))
    moved = tmp_path / "panel-moved-away.parquet"
    shutil.move(str(run["panel"]), moved)
    try:
        register = C.register(root)
        assert not register["problems"], register["problems"]
        assert register["panel_source"] == "GOVERNED_DELIVERY"
        verdict = C.verify_unit(register, "R0_s1", do_replay=False)
        assert verdict["metrics"] == C.VERIFIED
    finally:
        shutil.move(str(moved), str(run["panel"]))
