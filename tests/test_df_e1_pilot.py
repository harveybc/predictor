"""RP30: the household DEV pilot runner on a small synthetic panel — seal before outcomes, prepare through the real
loader with a memory preflight, batched windows (never materialised), the three regimes on the SAME initial
checkpoint with the AE of the seed, controls on the common evaluation set, verification from arrays, refusal of
altered DATA, no test split anywhere, and a closing table."""
import importlib.util
import json
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
E = _load("df_mod_e0")
RG = _load("df_e1_regimes")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A synthetic household-like panel (7 columns, one-minute grid) and a small sealed design bound to it."""
    tmp = tmp_path_factory.mktemp("e1pilot")
    rng = np.random.default_rng(0)
    n = 6 * P.DAY
    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / P.DAY)
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    base = daily + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({c: base * (i + 1) + rng.normal(size=n) * 0.05 + i for i, c in enumerate(cols)})
    df.loc[2000:2005, "Voltage"] = np.nan                                                # a short gap in the inputs
    ts = pd.date_range("2009-01-01 00:00", periods=n, freq="min")
    df.insert(0, "timestamp_label", ts.strftime("%d/%m/%Y %H:%M:%S"))
    panel = tmp / "panel.parquet"
    df.to_parquet(panel)
    design = P.seal(window=30, horizon=10, dev_train_days=4, dev_val_days=1, seeds=(1, 2), max_updates=40, ae_updates=30, batch=32, patience_epochs=2, pilot_updates=10,
                    declared_task={"context_physical_seconds": 30 * 60, "horizon_physical_seconds": 10 * 60, "purge": 40, "usable_windows_all_targets_valid": {}})
    # bind the sealed design to the synthetic bytes (the real one binds E1_TASKS' governed bytes)
    design.pop("design_sha256")
    design["governed_bytes"] = {"path": str(panel), "sha256": P.sha_file(panel)}
    design["dev_subpartition"]["rows"] = [0, 5 * P.DAY]
    design["design_sha256"] = E.sha_obj(design)
    return {"tmp": tmp, "design": design, "panel": panel, "df": df}


def _job(world, root, cell, data, **extra):
    job = {"kind": cell["kind"], "cell_id": cell["cell_id"], "seed": cell["seed"], "max_updates": cell.get("max_updates", 0), "design": world["design"],
           "data_npz": str(root / "DATA.npz"), "data_sha256": data["data_sha256"], "run_id": "t", "role": "CELL", **({"regime": cell["regime"]} if "regime" in cell else {}), **extra}
    if cell.get("depends_on"):
        job["pretrained_npz"] = str(root / "attempts" / cell["depends_on"] / "detector_pretrained.npz")
    return job


def test_RP30_seal_is_before_outcomes_and_names_task_context_reach_graph_seeds_budget(world):
    d = world["design"]
    assert d["task"]["model_reach_steps"] == E.support_reach("A", "sequence", 30) and d["task"]["model_reach_physical_seconds"] == 7 * 60
    assert d["replicas"]["seeds"] == [1, 2] and d["training"]["max_updates"] == 40 and d["budget"]["pilot_updates"] == 10
    assert len(d["cells"]) == 2 * 4 + 1 and {c["cell_id"] for c in d["pilots"]} == {"pilot_ae", "pilot_fit"}
    assert d["metrics"]["test"].startswith("NOT SCORED") and "BUDGET_LIMITED" in d["budget"]["rule"]
    assert E.sha_obj({k: v for k, v in d.items() if k != "design_sha256"}) == d["design_sha256"]


def test_RP30_prepare_uses_the_real_loader_with_preflight_and_a_train_only_scaler_and_refuses_altered_bytes(world):
    root = world["tmp"] / "root"
    data = P.prepare(world["design"], root)
    assert data["scaler"]["fitted_on"] == "train windows only" and data["memory_preflight"]["slice_bytes"] > 0
    assert data["enumerator"]["train"]["withdrawn_non_finite_inputs"] > 0                    # the Voltage gap withdrew windows
    assert data["coverage"]["common_evaluation_set"] > 0 and data["coverage"]["common_evaluation_set"] <= data["coverage"]["validation_admissible"]
    assert "test" not in data["enumerator"]
    z = np.load(root / "DATA.npz")
    assert z["eval_origins"].min() >= z["train_origins"].max() + 30 + 10                    # purge between train and validation
    bad = dict(_job(world, root, world["design"]["cells"][0], data))
    bad["data_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="REFUSED"):
        P.run_unit(bad, root / "x")


def test_RP30_units_share_the_initial_checkpoint_and_the_ae_and_are_verified_from_arrays(world):
    root = world["tmp"] / "root"
    data = P.prepare(world["design"], root)
    cells = {c["cell_id"]: c for c in world["design"]["cells"]}
    out = {}
    for cid in ("ae_s1", "R0_s1", "R1_s1", "R2_s1", "controls"):
        adir = root / "attempts" / cid
        out[cid] = P.run_unit(_job(world, root, cells[cid], data), adir)
        body = (adir / "cell.json").read_bytes()
        res = {"output_file": "cell.json", "output_sha256": __import__("hashlib").sha256(body).hexdigest()}
        rec, refusal = P.verified_unit(adir, res, {"output_sha256": res["output_sha256"]})
        assert refusal is None, refusal
    fits = {r: out[f"{r}_s1"] for r in ("R0", "R1", "R2")}
    assert len({f["initial_checkpoint"]["full_digest"] for f in fits.values()}) == 1                # same initial checkpoint
    assert fits["R1"]["regime_setup"]["detector_digest_after_setup"] == fits["R2"]["regime_setup"]["detector_digest_after_setup"] == out["ae_s1"]["pretraining"]["detector_digest"]
    assert fits["R0"]["regime_setup"]["detector_digest_after_setup"] == fits["R0"]["initial_checkpoint"]["detector_digest"]
    assert fits["R1"]["detector_unchanged"] and not fits["R1"]["gradient_proof"]["detector_receives_gradient"]
    assert all(not fits[r]["detector_unchanged"] and fits[r]["gradient_proof"]["detector_receives_gradient"] for r in ("R0", "R2"))
    assert all(f["training"]["updates"] == 40 and f["exposure"] == "NO_TEST_ACCESS" and "test" not in f["scores"] for f in fits.values())
    assert all(f["common_evaluation_set_size"] == data["coverage"]["common_evaluation_set"] for f in fits.values())
    c = out["controls"]["scores"]["validation"]
    assert set(c) == set(P.CONTROLS) and all(v["status"] == "MEDIDO" for v in c.values())
    assert c["seasonal_naive_daily"]["mase_mean"] < c["persistence"]["mase_mean"]                   # the synthetic signal is daily: the seasonal control must win
    # the same evaluation origins in every unit's arrays
    evs = [np.load(root / "attempts" / cid / "arrays.npz")["eval_origins"] for cid in out]
    assert all(np.array_equal(evs[0], e) for e in evs)
    # phases are costed apart
    assert set(fits["R1"]["cost"]) >= {"reuse_seconds", "fit_seconds", "inference_seconds", "metrics_seconds"} and "pretrain_seconds" in out["ae_s1"]["cost"]
    # an altered arrays file is refused
    a = root / "attempts" / "R0_s1" / "arrays.npz"
    z = dict(np.load(a))
    z["validation_pred"] = z["validation_pred"] + 1.0
    np.savez(a, **z)
    body = (root / "attempts" / "R0_s1" / "cell.json").read_bytes()
    res = {"output_file": "cell.json", "output_sha256": __import__("hashlib").sha256(body).hexdigest()}
    rec, refusal = P.verified_unit(root / "attempts" / "R0_s1", res, res)
    assert rec is None and refusal["why"] == "arrays absent or altered"        # the DIGEST refuses, before any recomputation


def test_RP30_batched_windows_equal_the_materialised_tensor():
    WB = P._dataset_class()
    Xs = np.arange(200 * 3, dtype=np.float32).reshape(200, 3)
    Y = np.arange(200, dtype=np.float64)
    o = np.array([10, 50, 199 - 5])
    ds = WB(Xs, Y, o, 8, 5, 1, 2, scaler_mean=np.zeros(3), scaler_sd=np.ones(3), shuffle=False, seed=0)
    X0, y0 = ds[0]
    assert X0.shape == (2, 8, 3) and np.array_equal(X0[0], Xs[3:11]) and y0[0, 0] == Y[15]
    assert len(ds) == 2 and ds[1][0].shape == (1, 8, 3)


def test_RP30_close_writes_the_table_from_verified_units(world):
    root = world["tmp"] / "root"
    (root / "DESIGN.json").write_text(json.dumps(world["design"]))
    (root / "REPORT.json").write_text(json.dumps({"run_id": "t", "data": P.prepare(world["design"], root), "cells": {}, "projection": None, "cost_pilot": {}, "host": "test"}))
    for cid in ("ae_s1", "R0_s1", "R1_s1", "R2_s1", "controls"):
        adir = root / "attempts" / cid
        body = (adir / "cell.json").read_bytes()
        sha = __import__("hashlib").sha256(body).hexdigest()
        (adir / "result.json").write_text(json.dumps({"output_file": "cell.json", "output_sha256": sha}))
        (adir / "outcome.json").write_text(json.dumps({"status": "COMPLETED", "verified": {"output_sha256": sha}, "summary": {"outcome": "COMPLETED", "cost": {"cpu_seconds": 1.0}}}))
    doc = P.close(root)
    assert doc["verified_units"]["R0_s1"] is False                                                  # the altered arrays of the previous test
    assert doc["identity"][1]["R1_R2_same_imported_detector"] and doc["identity"][1]["R1_detector_frozen"]
    assert doc["identity"][1]["shared_initial_checkpoint"] is False                                 # R0_s1 is unverified: no identity claim is made for it
    assert doc["cells"]["R1_s1"]["mase"] is not None and doc["cells"]["R0_s2"]["outcome"] == "ABSENT"
    assert doc["paired"]["R1_minus_R0"]["n"] == 0 and (root / "RESULTS.md").read_text().count("| R1_s1 |") == 1
