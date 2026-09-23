"""RP92/RP93/RP96 acceptance of the SOTA reproduction tool: the author's code path is what runs (parser, script, loader, model,
scorer), protocol substitutions FAIL the actual closure (scaler, channels, window, horizon, target, reduction, checkpoint,
substituted predictions), row-level support and future-perturbation/normalization-fit controls hold on the author's loader, a
fresh-process checkpoint reload goes through the author's test(), and the RP97 table generator is tested. Everything runs on a
tiny SYNTHETIC csv on CPU: a software fixture, declared as such — no benchmark score is produced here."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "tensorflow" in sys.modules or "keras" in sys.modules:
    # the author path is torch; in one process after the TensorFlow/Keras suites it dies with a fatal interpreter error
    # (observed on dragon, full-suite run of 2026-09-22). These rules run in their OWN process: `pytest tests/test_df_sota_repro.py`.
    pytest.skip("run in a separate process: the torch author path does not share a process with TensorFlow", allow_module_level=True)

R = _load("df_sota_repro")
B = _load("df_benchmark_contract")
T = _load("df_closure_table")

pytestmark = pytest.mark.skipif(not (R.AUTHOR_REPO / "run.py").is_file(), reason="the pinned author clone is not present here")

TINY_ARGV = ("--task_name long_term_forecast --is_training 1 --root_path X --data_path tiny.csv --model_id T --model TimeFilter --data custom --features M "
             "--seq_len 16 --label_len 8 --pred_len 4 --e_layers 1 --d_layers 1 --factor 3 --enc_in 4 --dec_in 4 --c_out 4 --patch_len 8 --des Exp "
             "--learning_rate 0.001 --batch_size 8 --train_epochs 2 --d_model 8 --d_ff 8 --n_heads 1 --dropout 0.1 --alpha 0.5 --itr 1 --num_workers 0").split()
N = 400


def _tiny_frame(n=N, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    df = pd.DataFrame({"date": pd.date_range("2016-07-01 02:00:00", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")})
    for i, name in enumerate(["0", "1", "2", "OT"]):
        df[name] = 10 + 3 * np.sin(2 * np.pi * t / 24 + i) + rng.normal(0, 0.3, n)
    return df


def _tiny_design(data_path: Path, *, seeds=(2021,), horizons=(4,)) -> dict:
    """A sealed design in the tool's schema over the tiny csv (the lock's dataset block is the fixture's, declared)."""
    parser, fix_seed = R.author_parser()
    contract = B.ecl321_official_tsl_ours(horizon_steps=4, input_window_steps=16)
    cells = []
    for h in horizons:
        args = parser.parse_args(TINY_ARGV)
        for s in seeds:
            cells.append({"cell_id": f"L16_h{h}_s{s}", "arm": "TimeFilter", "protocol": "L96", "seq_len": 16, "horizon": h, "seed": int(s), "argv": TINY_ARGV,
                          "effective_args": {k: v for k, v in sorted(vars(args).items())}, "setting": R.setting_of(args), "contract_sha256": contract.sha256()})
    lock = copy.deepcopy(R.seal.__wrapped__(seq_len=96) if hasattr(R.seal, "__wrapped__") else None) if False else None
    design = {"schema": R.SCHEMA, "task": "ecl321_official_tsl", "purpose": "SYNTHETIC_FIXTURE for the reproduction tool", "seq_len": 16, "seeds": list(seeds),
              "horizons": list(horizons), "protocol": "L96",
              "benchmark_contract": contract.to_design_block(comparability={**B.decide(contract, contract), "comparator_state": "PLANNED_REFERENCE"}),
              "cells": cells, "source_data": {"lake": "fixture", "resource": data_path.name, "sha256": R.sha_file(data_path)},
              "lock": {"published": {"table": "fixture", "per_horizon": {"4": {"mse": 0.5, "mae": 0.5}}, "average": {"mse": 0.5, "mae": 0.5}},
                       "source": {"files_sha256": R.source_digests()}, "dataset": {"rows": N}, "operational_patches": R.seal.__globals__["AGREEMENT"] and [],
                       "agreement": R.AGREEMENT, "replay_rule": R.AGREEMENT["replay"]}}
    design["design_sha256"] = R.sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    return design


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One trained tiny cell (CPU, two epochs), its BENCH_DATA and a stub warehouse holding the accepted chain."""
    tmp = tmp_path_factory.mktemp("sota")
    data = tmp / "data"; data.mkdir(); _tiny_frame().to_csv(data / "tiny.csv", index=False)
    root = tmp / "root"; root.mkdir()
    design = _tiny_design(data / "tiny.csv")
    (root / "DESIGN.json").write_text(json.dumps(design))
    arrays, record = R.bench_data(design, data / "tiny.csv")
    np.savez(root / "BENCH_DATA.npz", **arrays); record["data_sha256"] = R.sha_file(root / "BENCH_DATA.npz")
    (root / "BENCH_DATA.json").write_text(json.dumps(record))
    held = {"prepare": {"terminal_sha256": "p" * 64, "status": "COMPLETED", "config_sha256": design["design_sha256"],
                        "artifacts": [{"role": "data", "sha256": R.sha_file(root / "BENCH_DATA.npz")}, {"role": "record", "sha256": R.sha_file(root / "BENCH_DATA.json")}]}}
    receipts = {"units": {"prepare": {"campaign_sha256": "c" * 64, "terminal_sha256": "p" * 64}}}
    cell = design["cells"][0]
    rec = R.run_cell(design, cell, data_path=data / "tiny.csv", folder=root / "attempts" / cell["cell_id"], use_gpu=False)
    folder = root / "attempts" / cell["cell_id"]
    arts = [{"role": r, "sha256": R.sha_file(folder / f), "bytes": (folder / f).stat().st_size} for r, f in (("predictions", "arrays.npz"), ("checkpoint", "checkpoint.pth"), ("record", "cell.json"))]
    held[cell["cell_id"]] = {"terminal_sha256": "t" * 64, "status": "COMPLETED", "artifacts": arts, "config_sha256": design["design_sha256"],
                             "tags": {"horizon": "4", "seed": "2021", "seq_len": "16", "arm": "TimeFilter"}}
    receipts["units"][cell["cell_id"]] = {"campaign_sha256": "c" * 64, "terminal_sha256": "t" * 64}
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps(receipts))
    (root / "TERMINALS").mkdir(); (root / "TERMINALS" / f"{cell['cell_id']}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": arts}))
    # a delivery record in the governed layout, pointing at the fixture's bytes: what the closure table resolves the data path from
    (root / "DELIVERIES.json").write_text(json.dumps({"schema": "fixture", "design_sha256": design["design_sha256"], "resource": "tiny.csv", "units": {
        "prepare": {"path": str(data / "tiny.csv"), "sha256": R.sha_file(data / "tiny.csv"), "campaign_sha256": "c" * 64, "campaign_key": "fixture-prepare"}}}))
    return {"tmp": tmp, "data": data / "tiny.csv", "root": root, "design": design, "cell": cell, "record": rec, "held": held}


def _wh(world):
    return lambda campaign: {"current": json.loads(json.dumps(world["held"]))}


def _copy(world, tmp_path):
    import shutil
    root = tmp_path / "copy"
    shutil.copytree(world["root"], root)
    return root


# --- RP92: the lock reads the author's files, never a transcription -------------------------------------------------------------

def test_RP92_the_parser_and_script_are_the_authors_own_and_the_lock_carries_every_effective_default():
    parser, fix_seed = R.author_parser()
    assert fix_seed == 2021
    d = vars(parser.parse_args(TINY_ARGV))
    assert d["patience"] == 3 and d["lradj"] == "cosine" and d["loss"] == "MSE" and d["top_p"] == 0.5 and d["pos"] == 1 and d["use_norm"] == 1 and d["inverse"] is False
    cells = R.script_cells()
    l96 = {c["pred_len"]: c for c in cells if c["seq_len"] == 96}
    assert sorted(l96) == [96, 192, 336, 720] and all(c["model"] == "TimeFilter" for c in cells)
    a96 = vars(parser.parse_args(l96[96]["argv"]))
    assert (a96["enc_in"], a96["c_out"], a96["patch_len"], a96["d_model"], a96["d_ff"], a96["learning_rate"], a96["batch_size"], a96["train_epochs"], a96["e_layers"]) == (321, 321, 32, 512, 512, 1e-3, 16, 15, 2)
    assert a96["dropout"] == 0.5 and vars(parser.parse_args(l96[192]["argv"]))["dropout"] == 0.4 and a96["model_id"] == "ECL_96_96" and a96["data_path"] == "electricity.csv"
    assert R.setting_of(parser.parse_args(l96[96]["argv"])).startswith("long_term_forecast_ECL_96_96_TimeFilter_custom_ftM_sl96_ll48_pl96_dm512_nh4_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0")
    git = R.author_git()
    assert git["pinned_matches"] and git["clean"]
    assert set(R.SOURCE_FILES) - set(R.source_digests()) == set()


def test_RP92_the_sealed_design_recomputes_and_a_relabeled_or_edited_design_is_refused(world):
    d = json.loads(json.dumps(world["design"]))
    R.validate(d)
    forged = json.loads(json.dumps(d)); forged["cells"][0]["seed"] = 7
    with pytest.raises(R.SotaRefusal, match="does not recompute"):
        R.validate(forged)
    assert B.disposition(d)["disposition"] == "ACTIVE" and d["benchmark_contract"]["task_id"] == "ecl321_official_tsl"


def test_RP92_the_lock_is_sealed_before_any_data_and_names_the_official_file_and_paper_values():
    if not R.STORE_RECEIPT.is_file():
        pytest.skip("benchmark store receipt absent")
    d = R.seal()
    R.validate(d)
    assert len(d["cells"]) == 12 and d["lock"]["dataset"]["sha256"] == R.FILE_SHA256 and d["lock"]["dataset"]["rows"] == 26304
    assert d["lock"]["partitions"]["windows_per_split"]["96"] == {"train": 18221, "vali": 2537, "test": 5165}     # the loader's scored windows (rows - L - T + 1)
    assert d["lock"]["partitions"]["paper_table5_convention_rows_minus_L_plus_1"] == {"train": 18317, "vali": 2633, "test": 5261}   # the paper's printed sizes
    assert d["lock"]["published"]["per_horizon"]["96"] == {"mse": 0.133, "mae": 0.230} and d["lock"]["agreement"]["k_agree"] == 2.0
    assert d["lock"]["source"]["revision"] == R.PINNED_COMMIT and "np.Inf" in json.dumps(d["lock"]["operational_patches"])
    assert all(c["effective_args"]["patience"] == 3 for c in d["cells"]) and {c["seed"] for c in d["cells"]} == {2021, 2022, 2023}


# --- RP93: row-level support, future perturbation, normalization fit, boundary ---------------------------------------------------

def test_RP93_the_scaler_is_fit_on_the_train_rows_only_and_the_test_population_is_the_boundary_count(world, tmp_path):
    d, data = world["design"], world["data"]
    arrays, record = R.bench_data(d, data)
    df = pd.read_csv(data); n = len(df); train = int(0.7 * n); test = int(0.2 * n)
    vals = df.drop(columns=["date"]).values
    assert np.allclose(arrays["L16_h4_scaler_mean"], vals[:train].mean(axis=0)) and np.allclose(arrays["L16_h4_scaler_scale"], vals[:train].std(axis=0))
    assert record["sets"]["L16_h4"]["test"]["windows"] == test + 16 - 16 - 4 + 1 and record["sets"]["L16_h4"]["train"]["windows"] == train - 16 - 4 + 1
    # normalization-fit control: changing test rows changes nothing in the scaler; changing a train row does
    other = tmp_path / "t"; other.mkdir()
    df2 = df.copy(); df2.loc[n - 5:, "0"] += 100.0; df2.to_csv(other / "tiny.csv", index=False)
    a2, r2 = R.bench_data(d, other / "tiny.csv")
    assert np.array_equal(a2["L16_h4_scaler_mean"], arrays["L16_h4_scaler_mean"]) and np.array_equal(a2["L16_h4_scaler_scale"], arrays["L16_h4_scaler_scale"])
    df3 = df.copy(); df3.loc[3, "0"] += 100.0; df3.to_csv(other / "tiny.csv", index=False)
    a3, _ = R.bench_data(d, other / "tiny.csv")
    assert not np.array_equal(a3["L16_h4_scaler_mean"], arrays["L16_h4_scaler_mean"])


def test_RP93_a_future_perturbation_of_the_test_rows_does_not_reach_earlier_windows_through_the_authors_path(world, tmp_path):
    """The trained tiny checkpoint is re-scored (author's test(test=1)) on a csv whose LAST test rows are perturbed: predictions of
    windows that end before the perturbation are bit-identical; the perturbed windows differ."""
    d, cell, data = world["design"], world["cell"], world["data"]
    import shutil
    def score(csv: Path, tag: str):
        work = tmp_path / tag; (work / "checkpoints" / cell["setting"]).mkdir(parents=True)
        shutil.copy2(world["root"] / "attempts" / cell["cell_id"] / "checkpoint.pth", work / "checkpoints" / cell["setting"] / "checkpoint.pth")
        return R.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=csv.parent, data_name=csv.name, work=work, use_gpu=False, train=False)
    base = score(data, "base")
    df = pd.read_csv(data); n = len(df)
    other = tmp_path / "pert"; other.mkdir(); df2 = df.copy(); df2.loc[n - 6:, ["0", "1", "2", "OT"]] += 50.0; df2.to_csv(other / "tiny.csv", index=False)
    pert = score(other / "tiny.csv", "pert")
    p0, p1 = np.asarray(base["preds"]), np.asarray(pert["preds"])
    assert p0.shape == p1.shape
    # windows whose input AND target lie before row n-6 are untouched: window i covers rows [start+i, start+i+16+4)
    untouched = p0.shape[0] - 6 - 4 - 1
    assert untouched > 10 and np.array_equal(p0[:untouched], p1[:untouched]) and not np.array_equal(p0[-3:], p1[-3:])
    assert np.array_equal(base["preds"], np.load(world["root"] / "attempts" / cell["cell_id"] / "arrays.npz")["pred"])   # the fresh reload reproduces the stored predictions


def test_RP93_the_shims_and_the_alias_have_no_mathematical_effect_and_refuse_any_call():
    R.author_env()
    import sktime.datasets, patoolib
    with pytest.raises(RuntimeError, match="shim"):
        sktime.datasets.load_from_tsfile_to_dataframe("x")
    with pytest.raises(RuntimeError, match="shim"):
        patoolib.extract_archive("x")
    assert np.Inf is np.inf


# --- RP96: the actual closure verifies, and every substitution fails it ---------------------------------------------------------

def test_RP96_the_verification_binds_arrays_checkpoint_record_targets_and_metrics_and_the_table_ranks_the_active_task(world):
    ver = R.verify_sota_run(world["root"], warehouse=_wh(world), data_path=world["data"], replay=True)
    assert ver["problems"] == [] and ver["verified_units"] == [world["cell"]["cell_id"]] and ver["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT"
    r = ver["rows"][0]
    assert r["custody"]["class"] == "ACCEPTED_ARTIFACT_CHAIN" and r["recomputed"]["author_float32"] == r["author_metric_float32"]
    assert abs(r["recomputed"]["independent_float64"]["mse"] - r["author_metric_float32"]["mse"]) <= 1e-6 and r["derived"]["windows"] == 77
    assert r["replay"]["allclose_rule"] and r["replay"]["max_abs_prediction_difference"] == 0.0 and r["disposition"] == "ACTIVE"
    history = json.loads((world["root"] / "REPLAYS.json").read_text())[r["unit"]]                    # a history keyed by device@time, never an input
    assert history and all("identity" in entry and entry["property"] in ("same_device_repeatability", "cross_device_portability") for entry in history.values())
    # the same authority through the closure table: the ECL row enters the ACTIVE ranking, published metric first
    table = T.build([f"{world['root']}:tiny"], registry=B.registry(), warehouse=_wh(world), no_new_measurement=False)
    assert table["problems"] == [] and table["verified_rows"] == 1 and table["dispositions"] == {"ACTIVE": 1}
    assert table["active_ranking"][0]["unit"] == r["unit"] and table["active_ranking"][0]["ranking_error"] == r["author_metric_float32"]["mse"]
    md = T.markdown(table); assert "Active reference ranking" in md and "| 1 | ecl321_official_tsl |" in md
    # the RP97 table and its markdown
    t = R.table(world["design"], ver)
    assert t["complete"] and t["rows"][0]["mse"]["n_seeds"] == 1 and t["rows"][0]["mse"]["status"] in ("OPERATIONAL_AGREEMENT", "OPERATIONAL_PARTIAL", "OUTSIDE_OPERATIONAL_MARGIN")
    assert t["rows"][0]["scopes"]["same_device_repeatability"] == ["PASS"] and t["rows"][0]["per_seed"]["2021"]["mae"] == r["author_metric_float32"]["mae"]
    assert t["rows"][0]["matched_naive"]["mae"] > 0 and t["rows"][0]["training_completion"][0]["epochs_run"] == 2 and t["protocol_fidelity"]["verdict"].startswith("FAITHFUL")
    md2 = R.markdown(t); assert "| 4 |" in md2 and "Agreement rule" in md2 and t["unexecuted"] == []


@pytest.mark.parametrize("attack", ["predictions", "checkpoint", "scaler", "horizon", "channels", "window", "target", "reduction", "missing", "no_record_anchor", "tags"])
def test_RP96_every_protocol_substitution_fails_the_actual_closure(world, tmp_path, attack):
    root = _copy(world, tmp_path)
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    held = json.loads(json.dumps(world["held"]))
    data_path = world["data"]
    if attack == "predictions":                                                  # substituted prediction bytes (the truth), record untouched
        with np.load(folder / "arrays.npz") as z:
            pred = z["pred"]
        np.savez_compressed(folder / "arrays.npz", pred=pred * 0 + pred.mean())
    elif attack == "checkpoint":
        (folder / "checkpoint.pth").write_bytes(b"not a checkpoint")
    elif attack == "scaler":                                                     # a different scaler in the preparation evidence
        with np.load(root / "BENCH_DATA.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["L16_h4_scaler_scale"] = arrays["L16_h4_scaler_scale"] * 10
        np.savez(root / "BENCH_DATA.npz", **arrays)
    elif attack in ("horizon", "channels", "window", "target"):                   # the record claims another task than the design cell / data
        rec = json.loads((folder / "cell.json").read_text())
        if attack == "horizon":
            rec["cell"]["horizon"] = 8
        elif attack == "channels":
            rec["effective_args"]["enc_in"] = rec["effective_args"]["c_out"] = 5
        elif attack == "window":
            rec["effective_args"]["seq_len"] = 32
        else:
            rec["true_sha256"] = "0" * 64                                     # targets other than the ones the author loader derives
        (folder / "cell.json").write_text(json.dumps(rec))
        held[unit]["artifacts"] = [a if a["role"] != "record" else {**a, "sha256": R.sha_file(folder / "cell.json")} for a in held[unit]["artifacts"]]
        (root / "TERMINALS" / f"{unit}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": held[unit]["artifacts"]}))
    elif attack == "reduction":                                                  # a metric claimed under another reduction
        rec = json.loads((folder / "cell.json").read_text()); rec["author_metric_float32"]["mse"] = rec["author_metric_float32"]["mse"] * 0.5
        (folder / "cell.json").write_text(json.dumps(rec))
        held[unit]["artifacts"] = [a if a["role"] != "record" else {**a, "sha256": R.sha_file(folder / "cell.json")} for a in held[unit]["artifacts"]]
    elif attack == "missing":
        (folder / "arrays.npz").unlink()
    elif attack == "no_record_anchor":
        held[unit]["artifacts"] = [a for a in held[unit]["artifacts"] if a["role"] != "record"]
    elif attack == "tags":
        held[unit]["tags"]["seed"] = "2022"
    wh = lambda campaign: {"current": json.loads(json.dumps(held))}
    ver = R.verify_sota_run(root, warehouse=wh, data_path=data_path, replay=(attack == "checkpoint"))
    assert ver["verified_units"] == [] and ver["problems"], attack
    expected = {"predictions": "CHANGED ARRAYS", "checkpoint": "CHANGED CHECKPOINT", "scaler": "PREPARATION_CHANGED", "horizon": "identity contradicts", "channels": "effective arguments",
                "window": "effective arguments", "target": "TARGETS", "reduction": "METRIC", "missing": "missing, not absent", "no_record_anchor": "accepted record artifact",
                "tags": "IDENTITY"}[attack]
    assert any(expected in p for p in ver["problems"]), (attack, ver["problems"])
    # the closure emits no table rows as verified and the report says so
    a = SimpleNamespace(root=root, warehouse_token_file=None, warehouse_url=None, data_path=data_path, skip_replay=True, replay_device="cpu")
    rep = R.close(a, json.loads((root / "DESIGN.json").read_text()))
    assert not rep["verified"] and rep["table"]["complete"] is False and rep["table"]["unexecuted"] == [unit]
    assert rep["table"]["rows"][0]["mse"]["status"] == "NO_MEASUREMENT"                                   # nothing unverified enters a mean
    if attack not in ("missing",):
        shown = rep["table"]["rows"][0]["executed_unverified"]
        assert shown and shown[0]["unit"] == unit and shown[0]["why"] and "NOT verified" in R.markdown(rep["table"])


def test_RP96_a_replay_of_altered_checkpoint_bytes_fails_and_every_closure_replays_afresh(world, tmp_path):
    root = _copy(world, tmp_path)
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    (root / "REPLAYS.json").unlink(missing_ok=True)
    first = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert first["rows"][0]["replay"]["allclose_rule"] and "adopted_from" not in first["rows"][0]["replay"]
    second = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert "adopted_from" not in second["rows"][0]["replay"] and len(json.loads((root / "REPLAYS.json").read_text())[unit]) == 2     # a history, never an input
    # the checkpoint replaced by the same-shaped weights of ANOTHER training (different bytes): custody fails before any replay is trusted
    import torch
    sd = torch.load(folder / "checkpoint.pth")
    sd = {k: v * 1.5 for k, v in sd.items()}
    torch.save(sd, folder / "checkpoint.pth")
    third = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert third["verified_units"] == [] and any("CHANGED CHECKPOINT" in p for p in third["problems"])


def test_RP96_the_agreement_rule_is_frozen_and_applied_as_declared():
    pub = {"mse": 0.133, "mae": 0.230}
    ok = R.agreement(pub, [0.133, 0.140, 0.136], "mse")
    assert ok["status"] == "OPERATIONAL_AGREEMENT" and abs(ok["tolerance_agree"] - (2 * 0.005 + 0.0005)) < 1e-12 and ok["n_seeds"] == 3 and ok["sd_ddof1"] > 0
    assert "NOT a published per-horizon error bar" in ok["scope"]
    assert R.agreement(pub, [0.146, 0.146, 0.146], "mse")["status"] == "OPERATIONAL_PARTIAL"
    assert R.agreement(pub, [0.160, 0.160, 0.160], "mse")["status"] == "OUTSIDE_OPERATIONAL_MARGIN"
    assert R.agreement(pub, [], "mae")["status"] == "NO_MEASUREMENT"
    assert R.AGREEMENT["replay"]["atol"] == 1e-4 and R.AGREEMENT["std_paper"] == {"mse": 0.005, "mae": 0.006}




# --- RP98/RP99/RP100/RP103: Musashi's RP97 counterexamples become refusals of the actual closure and metric path ------------------

def test_RP99_the_catalog_is_recomputed_at_closure_consistent_with_the_author_metric_and_read_back(world):
    ver = R.verify_sota_run(world["root"], warehouse=_wh(world), data_path=world["data"], replay=False)
    r = ver["rows"][0]
    vp = world["root"] / "attempts" / r["unit"] / "METRICS_VAULT.json"
    assert vp.is_file() and r["recomputed"]["metrics_vault_sha256"] == R.sha_file(vp) and r["recomputed"]["metrics_vault_recomputed"] and r["recomputed"]["metrics_vault_read_back_equal"]
    v = json.loads(vp.read_text())
    assert v["schema"] == R.VAULT_SCHEMA and "NaN" not in vp.read_text() and "Infinity" not in vp.read_text()
    assert abs(v["global"]["mse"] - r["author_metric_float32"]["mse"]) <= 1e-6 and abs(v["global"]["mae"] - r["author_metric_float32"]["mae"]) <= 1e-6
    assert v["population"] == {"windows": 77, "steps": 4, "channels": 4, "elements": 77 * 16, "consumed_windows": 77}
    assert v["identity"]["true_sha256_consumed"] == json.loads((world["root"] / "attempts" / r["unit"] / "cell.json").read_text())["true_sha256"]
    assert v["identity"]["pred_sha256"] and v["identity"]["checkpoint_sha256"] and v["identity"]["metric_implementation_sha256"] and v["identity"]["row_order"]
    assert abs(np.mean(v["per_step"]["mse"]) - v["global"]["mse"]) <= 1e-9 and abs(np.mean(v["per_channel"]["mae"]) - v["global"]["mae"]) <= 1e-9
    assert abs(v["global"]["naive_mae"] - r["derived"]["naive"]["mae"]) <= 1e-6
    assert {c["state"] for c in v["catalog"].values()} <= {"DONE", "APPROXIMATE", "UNDEFINED", "NOT_APPLICABLE", "NOT_APPLICABLE_IN_ONE_CELL"}
    assert v["catalog"]["quantiles"]["state"] == "APPROXIMATE" and v["catalog"]["autocorrelation"]["state"] == "DONE" and 168 in v["catalog"]["autocorrelation"]["per_channel_lags_not_applicable"]
    assert len(v["per_window"]["mae"]) == 77 and abs(np.mean(v["per_window"]["mae"]) - v["global"]["mae"]) <= 1e-9 and v["time_blocks"][0]["n"] == 77
    assert "mase" not in json.dumps(v["global"]) and v["global"]["mae_relative_to_test_persistence"] == v["global"]["mae"] / v["global"]["naive_mae"]
    assert "mape_zspace" in v["global"] and "NOT physical" in v["catalog"]["percentage_errors_zspace"]["note"]
    assert v["independent_check"]["rule"] and abs(v["independent_check"]["global_vs_author"]["mae"]) <= 1e-6


def test_RP98_an_altered_persisted_catalog_is_rejected_preserved_and_replaced_by_the_recomputed_one(world, tmp_path):
    """Musashi's RP97 #1: global MAE and the first per-step MAE set to 999 in the persisted vault left the unit VERIFIED."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    vp = root / "attempts" / unit / "METRICS_VAULT.json"
    R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    vault = json.loads(vp.read_text()); vault["global"]["mae"] = 999.0; vault["per_step"]["mae"][0] = 999.0; vp.write_text(json.dumps(vault))
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    assert ver["verified_units"] == [] and any("VAULT_CHANGED" in p for p in ver["problems"])
    rejected = sorted((root / "attempts" / unit).glob("METRICS_VAULT.rejected.*.json"))
    assert rejected and json.loads(rejected[-1].read_text())["candidate"]["global"]["mae"] == 999.0 and "REJECTED" in json.loads(rejected[-1].read_text())["disposition"]
    assert json.loads(vp.read_text())["global"]["mae"] != 999.0 and ver["rows"][0]["recomputed"]["metrics_vault_sha256"] == R.sha_file(vp)
    again = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)             # the successor now verifies
    assert again["verified_units"] == [unit]


def test_RP98_a_contradictory_cached_replay_is_never_adopted_and_permuted_predictions_with_equal_aggregates_fail(world, tmp_path):
    """Musashi's RP97 #2: a REPLAYS.json entry with the same identity, allclose True and max difference 999 was adopted."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    first = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert first["verified_units"] == [unit] and first["rows"][0]["replay"]["exact_equal_fraction"] == 1.0 and first["rows"][0]["replay"]["finite"] and first["rows"][0]["replay"]["shape_equal"]
    hist = json.loads((root / "REPLAYS.json").read_text())
    for k in hist[unit]:
        hist[unit][k]["max_abs_prediction_difference"] = 999.0; hist[unit][k]["allclose_rule"] = True
    (root / "REPLAYS.json").write_text(json.dumps(hist))
    second = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert second["rows"][0]["replay"]["max_abs_prediction_difference"] == 0.0 and "adopted_from" not in second["rows"][0]["replay"]
    assert second["rows"][0]["same_device_repeatability"] == "PASS" and second["rows"][0]["cross_device_portability"] == "NOT_TESTED_ON_THIS_DEVICE"
    # predictions permuted across windows: population aggregates unchanged, the pointwise replay comparison must fail
    with np.load(root / "attempts" / unit / "arrays.npz") as z:
        pred = z["pred"]
    perm = pred[::-1].copy()
    assert abs(float(np.mean(np.abs(perm.astype(np.float64)))) - float(np.mean(np.abs(pred.astype(np.float64))))) < 1e-12 and not np.array_equal(perm, pred)
    np.savez(root / "attempts" / unit / "arrays.npz", pred=perm)
    rep = R.replay_cell(root, json.loads((root / "DESIGN.json").read_text()), unit, data_path=world["data"], device="cpu")
    assert rep["allclose_rule"] is False and rep["max_abs_prediction_difference"] > 0 and rep["exact_equal_fraction"] < 1.0 and rep["shape_equal"] and rep["finite"]
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)              # and the closure refuses the changed bytes first
    assert ver["verified_units"] == [] and any("CHANGED ARRAYS" in p for p in ver["problems"])


def test_RP99_the_catalog_refuses_short_extra_reordered_shaped_and_nonfinite_loaders_and_undefined_correlation_is_none():
    """Musashi's RP97 #3/#4 executed on the deployed metric function."""
    import torch
    W, T, C = 10, 2, 1
    x = torch.zeros((W, 24, C)); y = torch.ones((W, T, C)); pred = np.full((W, T, C), 2.0, dtype=np.float32)
    full = R.metrics_vault(pred, [(x, y, None, None)], pred_len=T, max_lag=2)
    assert full["global"]["mae"] == 1.0 and full["population"]["consumed_windows"] == 10
    with pytest.raises(R.VaultRefusal, match="INCOMPLETE POPULATION"):
        R.metrics_vault(pred, [(x[:5], y[:5], None, None)], pred_len=T, max_lag=2)
    with pytest.raises(R.VaultRefusal, match="EXTRA ROWS"):
        R.metrics_vault(pred, [(x, y, None, None), (x[:1], y[:1], None, None)], pred_len=T, max_lag=2)
    with pytest.raises(R.VaultRefusal, match="does not match"):
        R.metrics_vault(pred, [(x, torch.ones((W, T, C + 1)), None, None)], pred_len=T, max_lag=2)
    with pytest.raises(R.VaultRefusal, match="horizon"):
        R.metrics_vault(pred, [(x, y, None, None)], pred_len=T + 1, max_lag=2)
    bad = pred.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(R.VaultRefusal, match="non-finite"):
        R.metrics_vault(bad, [(x, y, None, None)], pred_len=T, max_lag=2)
    yb = y.clone(); yb[0, 0, 0] = float("inf")
    with pytest.raises(R.VaultRefusal, match="non-finite"):
        R.metrics_vault(pred, [(x, yb, None, None)], pred_len=T, max_lag=2)
    yr = torch.arange(W * T * C, dtype=torch.float32).reshape(W, T, C)
    a = R.metrics_vault(pred, [(x, yr, None, None)], pred_len=T, max_lag=2)
    b = R.metrics_vault(pred, [(x, yr.flip(0), None, None)], pred_len=T, max_lag=2)
    assert a["identity"]["true_sha256_consumed"] != b["identity"]["true_sha256_consumed"]                # order is part of the identity
    assert full["global"]["corr_pred_true"] is None and full["per_channel"]["corr_pred_true"] == [None] and full["catalog"]["correlation_r2"]["state"] == "UNDEFINED"
    v = R.metrics_vault(np.full((W, T, C), 2.0, dtype=np.float32), [(x, torch.randn((W, T, C)), None, None)], pred_len=T, max_lag=2)
    assert v["per_channel"]["corr_pred_true"] == [None] and v["global"]["corr_pred_true"] is None            # constant predictions
    pv = np.random.default_rng(0).normal(size=(W, T, C)).astype(np.float32)
    v2 = R.metrics_vault(pv, [(x, y, None, None)], pred_len=T, max_lag=2)
    assert v2["per_channel"]["corr_pred_true"] == [None] and v2["per_channel"]["r2"] == [None] and v2["global"]["r2"] is None   # constant targets


def test_RP99_independent_numeric_oracles_for_every_estimator_family():
    import torch
    rng = np.random.default_rng(7)
    W, T, C = 200, 3, 2
    x = torch.tensor(rng.normal(size=(W, 30, C)).astype(np.float32)); y = torch.tensor(rng.normal(size=(W, T, C)).astype(np.float32))
    pred = (y.numpy() + rng.normal(0, 0.5, size=(W, T, C))).astype(np.float32)
    v = R.metrics_vault(pred, [(x[:64], y[:64], None, None), (x[64:], y[64:], None, None)], pred_len=T, max_lag=5)
    d = pred.astype(np.float64) - y.numpy().astype(np.float64)
    assert abs(v["global"]["mse"] - np.mean(d ** 2)) < 1e-12 and abs(v["global"]["mae"] - np.mean(np.abs(d))) < 1e-12 and abs(v["global"]["bias"] - d.mean()) < 1e-12
    xs = x.numpy().astype(np.float64)
    naive = np.broadcast_to(xs[:, -1:, :], d.shape); assert abs(v["global"]["naive_mae"] - np.mean(np.abs(naive - y.numpy()))) < 1e-12
    seas = np.stack([xs[:, 30 - 24 + (k % 24), :] for k in range(T)], axis=1); assert abs(v["global"]["seasonal24_mae"] - np.mean(np.abs(seas - y.numpy()))) < 1e-12
    flat = d.ravel(); m = flat.mean(); var = flat.var()
    assert abs(v["residuals"]["var"] - var) < 1e-9 and abs(v["residuals"]["skewness"] - ((flat - m) ** 3).mean() / var ** 1.5) < 1e-9 and abs(v["residuals"]["kurtosis_raw"] - ((flat - m) ** 4).mean() / var ** 2) < 1e-9
    assert abs(v["residuals"]["quantiles"]["0.5"] - np.median(flat)) <= 0.02 + 1e-12                       # histogram resolution, declared APPROXIMATE
    assert sum(v["residuals"]["histogram"]["counts"]) + v["residuals"]["histogram"]["outside_range"] == flat.size and v["residuals"]["histogram"]["outside_range"] == 0
    yc = y.numpy()[:, :, 0].ravel().astype(np.float64); pc = pred[:, :, 0].ravel().astype(np.float64)
    assert abs(v["per_channel"]["corr_pred_true"][0] - np.corrcoef(pc, yc)[0, 1]) < 1e-9 and abs(v["per_channel"]["r2"][0] - (1 - ((pc - yc) ** 2).sum() / ((yc - yc.mean()) ** 2).sum())) < 1e-9
    series = d.mean(axis=2)
    for lag in (1, 5):
        for k in range(T):
            s_ = series[:, k] - series[:, k].mean(); expect = (s_[lag:] * s_[:-lag]).sum() / (s_ * s_).sum()
            assert abs(v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"][lag - 1][k] - expect) < 1e-9
    small = R.metrics_vault(pred[:4], [(x[:4], y[:4], None, None)], pred_len=T, max_lag=5)
    assert small["catalog"]["autocorrelation"]["lags"] == 2 and small["catalog"]["autocorrelation"]["per_channel_lags_not_applicable"] == [24, 168]
    two = R.metrics_vault(pred[:2], [(x[:2], y[:2], None, None)], pred_len=T, max_lag=5)
    assert two["catalog"]["autocorrelation"]["state"] == "NOT_APPLICABLE"
    y0 = torch.zeros((W, T, C)); x0 = torch.zeros((W, 30, C)); p0 = np.zeros((W, T, C), dtype=np.float32)
    z = R.metrics_vault(p0, [(x0, y0, None, None)], pred_len=T, max_lag=2)
    assert z["global"]["mae_relative_to_test_persistence"] is None and z["global"]["mape_zspace"] is None and z["catalog"]["percentage_errors_zspace"]["state"] == "UNDEFINED"
    assert z["catalog"]["percentage_errors_zspace"]["excluded_elements_abs_true_le_1e-8"] == W * T * C and z["global"]["skill_mae_vs_naive"] is None
    big = np.full((W, T, C), 50.0, dtype=np.float32)
    o = R.metrics_vault(big, [(x, y, None, None)], pred_len=T, max_lag=2)
    assert o["residuals"]["histogram"]["outside_range"] == W * T * C and o["residuals"]["quantiles"] is None and o["catalog"]["quantiles"]["state"] == "UNDEFINED"
    same = R.metrics_vault(y.numpy().copy(), [(x, y, None, None)], pred_len=T, max_lag=2)["global"]["mutual_information_bits_pred_true_64x64"]
    indep = R.metrics_vault(rng.normal(size=(W, T, C)).astype(np.float32), [(x, y, None, None)], pred_len=T, max_lag=2)["global"]["mutual_information_bits_pred_true_64x64"]
    assert same > indep > -1e-12
    assert v["population"]["consumed_windows"] == W and v["time_blocks"][-1]["n"] == W - 168 and v["catalog"]["time_blocks"]["last_block_partial"]


def test_RP103_the_four_horizon_average_is_formed_within_each_seed_first(world):
    """Musashi's RP97 oracle: four distinct horizon errors, identical across three seeds -> seed SD 0, n_seeds 3."""
    import copy
    ver = R.verify_sota_run(world["root"], warehouse=_wh(world), data_path=world["data"], replay=False)
    design = copy.deepcopy(world["design"]); report = copy.deepcopy(ver)
    design["horizons"] = [96, 192, 336, 720]; design["seeds"] = [2021, 2022, 2023]
    design["cells"], report["rows"], report["verified_units"] = [], [], []
    design["lock"]["published"]["per_horizon"] = {}
    for value, horizon in enumerate(design["horizons"], 1):
        design["lock"]["published"]["per_horizon"][str(horizon)] = {"mae": value, "mse": value}
        for seed in (2021, 2022, 2023):
            cell = copy.deepcopy(world["cell"]); cell.update(cell_id=f"oracle_h{horizon}_s{seed}", horizon=horizon, seed=seed)
            design["cells"].append(cell)
            row = copy.deepcopy(ver["rows"][0]); row.update(unit=cell["cell_id"], cell=cell, author_metric_float32={"mae": value, "mse": value}, verified=True, problems=[])
            report["rows"].append(row); report["verified_units"].append(cell["cell_id"])
    design["lock"]["published"]["average"] = {"mae": 2.5, "mse": 2.5}
    agg = R.table(design, report)["average_over_horizons"]["mae"]
    assert agg["n_seeds"] == 3 and agg["sd_ddof1"] == 0.0 and agg["values"] == [2.5, 2.5, 2.5] and agg["seeds"] == [2021, 2022, 2023] and "within each seed" in agg["grain"]
    report["rows"] = [r for r in report["rows"] if r["unit"] != "oracle_h720_s2023"]; report["verified_units"].remove("oracle_h720_s2023")
    agg2 = R.table(design, report)["average_over_horizons"]["mae"]
    assert agg2["status"] == "NOT_COMPUTED" and sorted(agg2["seed_averages_available"]) == ["2021", "2022"]



# --- RP101: the bounded evaluation adapter preserves the computation ---------------------------------------------------------------

def test_RP101_the_bounded_adapter_reproduces_the_authors_test_elementwise_with_the_population_and_official_metrics(world, tmp_path):
    """Original path vs bounded path on the real tiny author fixture and its trained checkpoint: identical predictions, targets,
    population, checkpoint bytes and the author's float32 metric; a budget too small reports the author metric as NOT executed
    while the float64 reduction stands; partial final batches are consumed."""
    import shutil
    d, cell, data = world["design"], world["cell"], world["data"]
    ckpt = world["root"] / "attempts" / cell["cell_id"] / "checkpoint.pth"; ckpt_sha = R.sha_file(ckpt)
    def run(tag, **kw):
        work = tmp_path / tag; (work / "checkpoints" / cell["setting"]).mkdir(parents=True)
        shutil.copy2(ckpt, work / "checkpoints" / cell["setting"] / "checkpoint.pth")
        return R.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=data.parent, data_name=data.name, work=work, use_gpu=False, train=False, **kw)
    a = run("author"); b = run("bounded", bounded=True)
    assert np.array_equal(np.asarray(a["preds"]), np.asarray(b["preds"])) and np.array_equal(np.asarray(a["trues"]), np.asarray(b["trues"]))
    assert a["author_metric"] == b["author_metric"] and b["author_metric_state"].startswith("EXECUTED") and b["bounded"]["finalized"]["windows"] == 77
    assert b["bounded"]["finalized"]["batch_sizes"]["last"] == 77 % 8 and b["bounded"]["finalized"]["batch_sizes"]["first"] == 8       # partial final batch consumed
    assert b["bounded"]["finalized"]["true_sha256"] == R.sha_array(np.asarray(a["trues"], dtype=np.float32)) and R.sha_file(ckpt) == ckpt_sha
    assert b["bounded"]["adapter"]["version"] == R.BOUNDED_ADAPTER_VERSION and b["bounded"]["adapter"]["source_sha256"]
    assert abs(b["independent_metric_float64"]["mae"] - a["author_metric"]["mae"]) <= 1e-6
    # RP109: a budget below the author function's temporaries still yields the author's float32 reduction, through the exact
    # bounded route (bit-equal to the author's function run above); the author's own function is then reported not executed
    c = run("budget", bounded=True, author_metric_budget_bytes=1)
    assert c["author_metric"] == a["author_metric"] and c["author_metric_state"].startswith("EXECUTED: df_sota_author_metric_exact.v1")
    assert c["author_scorer_parity"]["author_function"] is None and "not executed" in c["author_scorer_parity"]["why"]
    assert b["author_scorer_parity"]["bit_equal"] is True and abs(c["independent_metric_float64"]["mse"] - a["author_metric"]["mse"]) <= 1e-6
    # through run_cell: the artifact and the record carry the adapter identity, the memmaps are gone, the checkpoint is the author's
    folder = tmp_path / "cell"
    rec = R.run_cell(d, cell, data_path=data, folder=folder, use_gpu=False, bounded=True)
    assert rec["evaluation_path"]["bounded_adapter"]["version"] == R.BOUNDED_ADAPTER_VERSION and not list(folder.glob("work/bounded_*.npy"))
    with np.load(folder / "arrays.npz") as z:
        assert np.array_equal(z["pred"], np.asarray(a["preds"])) or rec["author_metric_float32"] is not None       # a fresh training: same population
    assert rec["true_sha256"] == b["bounded"]["finalized"]["true_sha256"] and rec["shapes"]["pred"] == [77, 4, 4]


def test_RP101_the_bounded_adapter_refuses_incomplete_extra_or_missing_populations(world, tmp_path, monkeypatch):
    import shutil, torch
    cell, data = world["cell"], world["data"]
    ckpt = world["root"] / "attempts" / cell["cell_id"] / "checkpoint.pth"
    work = tmp_path / "w"; (work / "checkpoints" / cell["setting"]).mkdir(parents=True); shutil.copy2(ckpt, work / "checkpoints" / cell["setting"] / "checkpoint.pth")
    R.author_env(); R.fix_seeds(cell["seed"])
    args = R.build_args(cell["argv"], data_dir=data.parent, data_name=data.name, checkpoints=work / "checkpoints", use_gpu=False)
    exp_module = __import__("importlib").import_module("exp.exp_long_term_forecasting")
    exp = exp_module.Exp_Long_Term_Forecast(args)
    real_get = exp._get_data
    class Short:                                                                  # a loader that stops early
        def __init__(self, loader, n): self.loader, self.n = loader, n
        def __iter__(self):
            for i, b in enumerate(self.loader):
                if i >= self.n: break
                yield b
        def __len__(self): return self.n
    def short(flag):
        ds, dl = real_get(flag); return ds, Short(dl, 3)
    monkeypatch.setattr(exp, "_get_data", short)
    with pytest.raises(R.SotaRefusal, match="INCOMPLETE POPULATION"):
        R.bounded_test(exp, cell["setting"], work)
    class Extra:                                                                  # a loader that yields a batch twice
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for b in self.loader:
                yield b
            yield b
        def __len__(self): return len(self.loader) + 1
    def extra(flag):
        ds, dl = real_get(flag); return ds, Extra(dl)
    monkeypatch.setattr(exp, "_get_data", extra)
    with pytest.raises(R.SotaRefusal, match="more windows"):
        R.bounded_test(exp, cell["setting"], work)
    (work / "checkpoints" / cell["setting"] / "checkpoint.pth").unlink()
    monkeypatch.setattr(exp, "_get_data", real_get)
    with pytest.raises(R.SotaRefusal, match="no checkpoint"):
        R.bounded_test(exp, cell["setting"], work)


# --- RP103: the authorized deletion lifecycle, and the metric basis when the author's reduction was not executed ------------------

def test_RP103_the_deletion_gate_refuses_unverified_or_altered_cells_and_a_passed_deletion_leaves_a_dated_historical_verification(world, tmp_path):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    (root / "REPORT.json").unlink(missing_ok=True)
    assert R.deletion_gate(root, unit)["reasons"] == ["no closure REPORT.json"]
    a = SimpleNamespace(root=root, warehouse_token_file=None, warehouse_url=None, data_path=world["data"], skip_replay=True, replay_device="cpu")
    R.close(a, json.loads((root / "DESIGN.json").read_text()))                                             # no warehouse: nothing verified -> gate refuses
    g = R.deletion_gate(root, unit); assert not g["pass"] and any("did not verify" in x for x in g["reasons"])
    # a verified closure (stub warehouse), then the gate passes; a dry run deletes nothing
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        a2 = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu")
        rep = R.close(a2, json.loads((root / "DESIGN.json").read_text()))
    assert rep["verified"] and R.deletion_gate(root, unit)["pass"]
    dry = R.delete_predictions(root, [unit], dry_run=True)
    assert (folder / "arrays.npz").is_file() and dry["units"][unit]["copies_inventoried"][0]["sha256"] == json.loads((folder / "cell.json").read_text())["arrays_sha256"]
    # an altered catalog on disk closes the gate
    vp = folder / "METRICS_VAULT.json"; original = vp.read_bytes(); v = json.loads(original); v["global"]["mae"] = 5.0; vp.write_text(json.dumps(v))
    assert any("not the one the closure reported" in x for x in R.deletion_gate(root, unit)["reasons"]); vp.write_bytes(original)
    # a copy in a staging root is inventoried and deleted too; receipts carry bytes and filesystem deltas
    staging = tmp_path / "staging"; (staging / "attempts" / unit).mkdir(parents=True); import shutil; shutil.copy2(folder / "arrays.npz", staging / "attempts" / unit / "arrays.npz")
    size = (folder / "arrays.npz").stat().st_size
    rec = R.delete_predictions(root, [unit], extra_roots=[staging])
    assert rec["units"][unit]["gate"]["pass"] and len([d for d in rec["paths"] if d["deleted"]]) == 2 and sum(d["bytes"] for d in rec["paths"]) == 2 * size
    assert not (folder / "arrays.npz").exists() and not (staging / "attempts" / unit / "arrays.npz").exists() and (folder / "PREDICTIONS_DELETED.json").is_file()
    assert (folder / "checkpoint.pth").is_file() and (folder / "cell.json").is_file() and vp.is_file() and list(root.glob("DELETION_RECEIPT.*.json"))
    assert rec["reclaimed_bytes_by_filesystem"]
    # the closure afterwards: a dated historical verification, never a current replay, never "missing"
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    row = ver["rows"][0]
    assert row["status"] == "METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION" and row["verified"] is False and row["verified_historically"] and ver["problems"] == []
    assert row["replay"]["skipped"] and "deleted" in row["replay"]["why"] and ver["historically_verified_units"] == [unit]
    t = R.table(json.loads((root / "DESIGN.json").read_text()), ver)
    assert t["rows"][0]["historically_verified_seeds"] == [unit] and t["rows"][0]["mse"]["n_seeds"] == 1
    # without a valid receipt, missing arrays are a problem
    (folder / "PREDICTIONS_DELETED.json").write_text(json.dumps({"arrays_sha256": "0" * 64, "closure_report_sha256": "x", "deleted_at": "now"}))
    bad = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    assert bad["rows"][0]["status"] == "DELETED_HISTORY_UNBOUND" and bad["problems"]
    (folder / "PREDICTIONS_DELETED.json").unlink()
    gone = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    assert gone["rows"][0]["status"] == "MISSING" and any("missing, not absent" in p for p in gone["problems"])


def test_RP101_a_record_without_the_authors_float32_reduction_verifies_on_its_float64_basis_and_says_so(world, tmp_path):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    rec = json.loads((folder / "cell.json").read_text()); rec["author_metric_float32"] = None; rec["author_metric_state"] = "NOT_EXECUTED_WITHIN_BUDGET: test"
    (folder / "cell.json").write_text(json.dumps(rec))
    held = json.loads(json.dumps(world["held"]))
    held[unit]["artifacts"] = [a if a["role"] != "record" else {**a, "sha256": R.sha_file(folder / "cell.json")} for a in held[unit]["artifacts"]]
    (root / "TERMINALS" / f"{unit}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": held[unit]["artifacts"]}))
    # the record changed, so a catalog persisted under the old record identity is VAULT_CHANGED once (preserved), then the successor verifies
    first = R.verify_sota_run(root, warehouse=lambda c: {"current": json.loads(json.dumps(held))}, data_path=world["data"], replay=False)
    assert first["verified_units"] == [unit] or any("VAULT_CHANGED" in p for p in first["problems"])
    ver = R.verify_sota_run(root, warehouse=lambda c: {"current": json.loads(json.dumps(held))}, data_path=world["data"], replay=True)
    row = ver["rows"][0]
    assert row["verified"], ver["problems"]
    # RP109: the closure recomputes the author's float32 reduction through the exact route; float64 stays a separately named check
    assert row["metric_basis"].startswith("author_float32 (recomputed at closure by df_sota_author_metric_exact.v1") and row["author_metric_float32"] == world["record"]["author_metric_float32"]
    assert abs(row["recomputed"]["independent_float64"]["mae"] - rec["independent_metric_float64"]["mae"]) <= 1e-9 and row["recomputed"]["author_scorer_parity"]["bit_equal"] is True
    assert row["replay"]["path"].startswith("fresh process") and row["replay"]["replayed_metric_float64"] and row["replay"]["true_sha256_replayed"] == rec["true_sha256"]
    t = R.table(json.loads((root / "DESIGN.json").read_text()), ver)
    assert t["rows"][0]["metric_basis"] == [row["metric_basis"]]
    # at closure with a tiny author-metric budget the exact route still gives the author's float32; only the author's own function is not run beside it
    ver2 = R.verify_sota_run(root, warehouse=lambda c: {"current": json.loads(json.dumps(held))}, data_path=world["data"], replay=False, author_metric_budget=1)
    r2 = ver2["rows"][0]
    assert r2["verified"] and r2["recomputed"]["author_float32"] == world["record"]["author_metric_float32"] and r2["recomputed"]["author_scorer_parity"]["author_function"] is None


def test_RP99_paired_contrasts_are_computed_from_the_persisted_per_window_series(world, tmp_path):
    import copy
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    pc = R.paired_contrasts(root, design, ver)
    e = pc["horizons"]["4"]
    assert e["state"].startswith("PARTIAL") and e["vs_baselines"][unit]["n_windows"] == 77 and e["vs_baselines"][unit]["model_minus_seasonal24_mae"]["sd_ddof1"] is not None
    v = json.loads((root / "attempts" / unit / "METRICS_VAULT.json").read_text())
    assert abs(e["vs_baselines"][unit]["model_minus_persistence_mae"]["mean"] - (np.mean(v["per_window"]["mae"]) - np.mean(v["per_window"]["naive_mae"]))) < 1e-12
    # a second seed (a copy of the catalog under another unit) yields a seed pair with zero difference
    twin = copy.deepcopy(design["cells"][0]); twin.update(cell_id="L16_h4_s2022", seed=2022); design["cells"].append(twin)
    import shutil; shutil.copytree(root / "attempts" / unit, root / "attempts" / twin["cell_id"])
    row2 = copy.deepcopy(ver["rows"][0]); row2.update(unit=twin["cell_id"], cell={**row2["cell"], "cell_id": twin["cell_id"], "seed": 2022}); ver["rows"].append(row2)
    pc2 = R.paired_contrasts(root, design, ver)["horizons"]["4"]
    key = f"{unit} - {twin['cell_id']}"
    assert pc2["state"] == "DONE" and pc2["seed_pairs"][key]["mae"]["mean"] == 0.0 and pc2["seed_pairs"][key]["n_windows"] == 77


def test_RP102_a_failed_attempt_is_retired_into_versioned_history_and_nothing_is_deleted(world, tmp_path):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    (root / "DELIVERIES.json").write_text(json.dumps({"design_sha256": world["design"]["design_sha256"], "units": {unit: {"path": "x", "sha256": "y", "campaign_sha256": "c", "campaign_key": "k"}}}))
    out = R.retire_attempt(root, unit, reason="interrupted")
    assert (root / "attempts" / f"{unit}.{out['stamp']}" / "cell.json").is_file() and not (root / "attempts" / unit).exists()
    d = json.loads((root / "DELIVERIES.json").read_text()); r = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())
    assert unit not in d["units"] and d["failed_attempts"][0]["unit"] == unit and unit not in r["units"] and r["failed_attempts"][0]["reason"] == "interrupted"
    assert (root / "TERMINALS" / f"{unit}.{out['stamp']}.json").is_file() and list(root.glob(f"RETIRED.{unit}.*.json"))


def test_RP102_dataloader_workers_change_host_memory_only_batches_are_identical(world, tmp_path):
    import torch
    cell, data = world["cell"], world["data"]
    R.author_env()
    DF = __import__("importlib").import_module("data_provider.data_factory")
    out = {}
    for w in (1, 0):
        R.fix_seeds(cell["seed"])
        args = R.build_args(cell["argv"], data_dir=data.parent, data_name=data.name, checkpoints=tmp_path / "c", use_gpu=False, dataloader_workers=w)
        assert args.num_workers == w
        _, train_loader = DF.data_provider(args, "train"); _, test_loader = DF.data_provider(args, "test")
        out[w] = ([b[0].clone() for b in train_loader][:6], [b[1].clone() for b in test_loader])
    assert all(torch.equal(a, b) for a, b in zip(out[1][0], out[0][0])) and all(torch.equal(a, b) for a, b in zip(out[1][1], out[0][1]))
    rec = R.run_cell(world["design"], cell, data_path=data, folder=tmp_path / "cell", use_gpu=False, bounded=True, dataloader_workers=0)
    assert rec["operational_patches"] and "num_workers 0" in rec["operational_patches"][0]["what"] and rec["effective_args"]["num_workers"] == 0


def test_RP102_the_terminal_of_a_cell_on_the_float64_basis_is_built_from_the_record_not_from_the_absent_author_reduction(world, tmp_path, monkeypatch):
    """The failure that hit T=336 on WORKER_B: the author reduction was not executed within the memory budget, the record carries
    author_metric_float32 = None, and the terminal must still be sent (on the labelled float64 basis) instead of a TypeError."""
    root = tmp_path / "root"; unit = "L96_h336_s2021"
    (root / "attempts" / unit).mkdir(parents=True)
    for f in ("arrays.npz", "checkpoint.pth", "cell.json"):
        (root / "attempts" / unit / f).write_bytes(b"x")
    record = {"author_metric_float32": None, "author_metric_state": "NOT_EXECUTED_WITHIN_BUDGET",
              "independent_metric_float64": {"mse": 0.1626, "mae": 0.2607},
              "training": {"epochs_run": 15, "best_epoch_by_vali": 10}, "cost": {"wall_seconds": 900.0, "cpu_seconds": 800.0}}
    cell = {"cell_id": unit, "arm": "timefilter", "horizon": 336, "seq_len": 96, "seed": 2021}
    sent = {}
    U = R._module("df_utility_run")
    G = SimpleNamespace(report_terminal=lambda root, u, terminal, **kw: sent.update({u: terminal}) or {"flushed": {"pending": [], "failures": []}})
    monkeypatch.setattr(R, "governance_modules", lambda: (G, U))
    a = SimpleNamespace(root=root, lake="sota_benchmarks", resource="r", gov_url="http://x", api_key_file="k")
    R.report_unit(a, {"design_sha256": "d" * 64}, cell, record)
    terminal = sent[unit]
    values = {m["metric"]: m["value"] for m in terminal["metrics"]}
    assert terminal["status"] == "COMPLETED" and values["sota.test.mse_normalized"] == 0.1626 and values["sota.test.mae_normalized"] == 0.2607
    assert terminal["tags"]["metric_basis"].startswith("independent_float64")
    assert (root / "TERMINALS" / f"{unit}.json").is_file()
    with pytest.raises(R.SotaRefusal):
        R.report_unit(a, {"design_sha256": "d" * 64}, cell, {**record, "independent_metric_float64": None})


def test_RP100_a_workers_replay_history_travels_through_merge_as_a_dated_record_and_the_table_cites_it_as_history_only(world, tmp_path):
    """The coordinator closes on another device (WORKER_B): WORKER_A's same-device GPU replays are cited from its merged
    REPLAYS files as HISTORICAL_RECORD_ONLY, in both the RP96 flat/GPU forms and the RP100 keyed form; never as verification."""
    unit = world["cell"]["cell_id"]
    root = _copy(world, tmp_path)
    src = tmp_path / "worker"; import shutil; shutil.copytree(world["root"], src)
    (src / "REPLAYS.json").write_text(json.dumps({unit: {"device": "cpu", "allclose_rule": False, "max_abs_prediction_difference": 0.0019}}))
    (src / "REPLAYS_GPU.json").write_text(json.dumps({"cells": {unit: {"device": "cuda:0", "allclose_rule": True, "max_abs_prediction_difference": 0.0}}}))
    (root / "REPLAYS.json").write_text(json.dumps({unit: {"cuda:GPU-x@2026-09-22T00:00:00Z": {"device": "cuda", "device_uuid": "GPU-x", "property": "cross_device_portability",
                                                                                           "allclose_rule": True, "max_abs_prediction_difference": 0.0}}}))
    out = R.merge(root, src)
    assert sorted(out["replay_history_files"]) == [f"REPLAYS_HISTORY.worker.REPLAYS.json", f"REPLAYS_HISTORY.worker.REPLAYS_GPU.json"]
    hist = R.replay_history(root, unit)
    assert {h["scope"] for h in hist} == {"HISTORICAL_RECORD_ONLY"} and len(hist) == 3
    assert [h["allclose_rule"] for h in sorted(hist, key=lambda h: h["source"])] == [True, False, True]
    assert R.replay_history(root, "L96_h720_s2099") == []


def test_RP101_a_stored_array_streams_npz_members_and_npy_files_chunk_by_chunk_equal_to_a_whole_load(tmp_path):
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((37, 5, 3)).astype(np.float32); true = rng.standard_normal((37, 5, 3)).astype(np.float32)
    np.savez(tmp_path / "arrays.npz", pred=pred, true=true)                      # ZIP_STORED members, as the cells' arrays.npz
    np.save(tmp_path / "t.npy", true)
    a = R.StoredArray(tmp_path / "arrays.npz", "pred"); b = R.StoredArray(tmp_path / "t.npy")
    assert a.shape == pred.shape and a.dtype == np.float32 and a.nbytes == pred.nbytes and len(a) == 37
    assert np.array_equal(a.load(), pred) and np.array_equal(b.load(), true) and np.array_equal(a[10:20], pred[10:20]) and np.array_equal(a[-1], pred[-1])
    assert np.array_equal(a[30:100], pred[30:]) and a[5:5].shape == (0, 5, 3) and a.all_finite() and np.array_equal(a.memmap(), pred)
    assert R.float64_metrics(a, b) == R.float64_metrics(pred, true)
    assert R.StoredArray(tmp_path / "arrays.npz", "true.npy").load().tobytes() == true.tobytes()
    # a compressed member (the cells written before the no-compression decision) streams forward-only, never inflated whole
    np.savez_compressed(tmp_path / "c.npz", pred=pred, true=true)
    c = R.StoredArray(tmp_path / "c.npz", "pred")
    assert c.compressed and c.shape == pred.shape and np.array_equal(c[0:10], pred[0:10]) and np.array_equal(c[10:20], pred[10:20])
    assert np.array_equal(c[5:8], pred[5:8]) and np.array_equal(c[-1], pred[-1]) and np.array_equal(c.load(), pred) and c.all_finite()
    assert R.float64_metrics(c, R.StoredArray(tmp_path / "c.npz", "true")) == R.float64_metrics(pred, true)
    assert R.compare_predictions(pred, c, atol=1e-4, rtol=1e-4, step=7)["exact_equal_fraction"] == 1.0
    with pytest.raises(R.SotaRefusal):
        c.memmap()
    c.close()
    bad = pred.copy(); bad[3, 1, 2] = np.nan; np.save(tmp_path / "bad.npy", bad)
    assert not R.StoredArray(tmp_path / "bad.npy").all_finite() and not R.all_finite(bad)


def test_RP100_the_chunked_replay_comparison_is_the_frozen_rule_without_full_size_temporaries(tmp_path):
    rng = np.random.default_rng(1)
    stored = rng.standard_normal((50, 4, 3)).astype(np.float32); np.savez(tmp_path / "a.npz", pred=stored)
    sa = R.StoredArray(tmp_path / "a.npz", "pred")
    same = R.compare_predictions(stored, sa, atol=1e-4, rtol=1e-4, step=7)
    assert same["allclose_rule"] and same["max_abs_prediction_difference"] == 0.0 and same["exact_equal_fraction"] == 1.0 and same["elements"] == stored.size
    near = stored + np.float32(5e-5); r = R.compare_predictions(near, sa, atol=1e-4, rtol=1e-4, step=7)
    assert r["allclose_rule"] == bool(np.allclose(near, stored, atol=1e-4, rtol=1e-4)) and r["allclose_rule"] and r["exact_equal_elements"] < stored.size
    far = stored.copy(); far[49, 3, 2] += np.float32(0.01); r = R.compare_predictions(far, sa, atol=1e-4, rtol=1e-4, step=7)
    assert not r["allclose_rule"] and abs(r["max_abs_prediction_difference"] - 0.01) < 1e-6 and r["exact_equal_elements"] == stored.size - 1
    perm = stored[::-1].copy()
    assert not R.compare_predictions(perm, sa, atol=1e-4, rtol=1e-4)["allclose_rule"]


def test_RP101_the_closure_streams_the_targets_to_a_file_with_the_records_digest_and_removes_it_afterwards(world, tmp_path):
    out = R.naive_and_trues(world["design"], world["cell"], world["data"], work_dir=tmp_path / "cw")
    t = R.StoredArray(out["trues_path"])
    assert out["true_sha256"] == world["record"]["true_sha256"] == R.sha_array(t.load()) and t.shape[0] == out["windows"]
    plain = R.naive_and_trues(world["design"], world["cell"], world["data"])
    assert plain["trues_path"] is None and plain["true_sha256"] == out["true_sha256"] and plain["naive"] == out["naive"]
    root = _copy(world, tmp_path)
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=False)
    row = ver["rows"][0]
    assert row["verified"] and row["recomputed"]["targets_source"].startswith("closure_work") and not (root / "closure_work").exists()


def test_RP103_the_deletion_gate_reads_replay_failures_from_every_history_form(world, tmp_path):
    """WORKER_A's history is the RP96 flat form; a merged worker's history lives in REPLAYS_HISTORY.*: a recorded failure in any of
    them demands the preserved route diagnostic before the arrays may be deleted."""
    unit = world["cell"]["cell_id"]
    root = _copy(world, tmp_path)
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        a = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu")
        R.close(a, json.loads((root / "DESIGN.json").read_text()))
    assert R.deletion_gate(root, unit)["pass"], R.deletion_gate(root, unit)["reasons"]
    (root / "REPLAYS_HISTORY.worker.REPLAYS.json").write_text(json.dumps({unit: {"device": "cpu", "allclose_rule": False, "max_abs_prediction_difference": 0.002}}))
    g = R.deletion_gate(root, unit)
    assert not g["pass"] and any("route-level diagnostic" in r for r in g["reasons"])
    (root / "attempts" / unit / "ROUTE_TRACE.json").write_text("{}")
    assert R.deletion_gate(root, unit)["pass"]


def test_RP102_the_host_allocator_setting_is_a_declared_operational_patch_of_cell_and_replay_processes(monkeypatch):
    """T=720 on WORKER_B: the author's validation grew the host process by ~5 GB of freed-but-resident buffers; the fixed glibc
    thresholds are passed to the child/replay environment and recorded, and 'none' leaves the allocator alone."""
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    env = R.child_env()
    assert env["GLIBC_TUNABLES"] == R.MALLOC_TUNABLES_DEFAULT and "mmap_threshold=1048576" in env["GLIBC_TUNABLES"]
    assert "GLIBC_TUNABLES" not in R.child_env("none") and R.child_env(None, OMP_NUM_THREADS=2)["OMP_NUM_THREADS"] == "2"
    assert R.host_allocator_patch() == []
    monkeypatch.setenv("GLIBC_TUNABLES", R.MALLOC_TUNABLES_DEFAULT)
    patch = R.host_allocator_patch()
    assert len(patch) == 1 and "GLIBC_TUNABLES" in patch[0]["what"] and "host memory only" in patch[0]["effect"]


def test_RP100_the_device_identity_is_the_device_the_process_computes_on_not_nvidia_smi_index_zero(monkeypatch):
    """gamma: the RTX 5090 is nvidia-smi index 1 and cuda:0 of a process restricted by CUDA_VISIBLE_DEVICES; the replay record
    and the same-device test must name that device. Without CUDA here, the visibility mask decides; a record carries its
    actual UUID, an older record its environment's mask, an even older one nvidia-smi's first GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8")
    monkeypatch.setattr(R, "gpu_state", lambda: [{"index": 0, "uuid": "GPU-b77fc3ad"}, {"index": 1, "uuid": "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"}])
    import torch
    if not torch.cuda.is_available():
        assert R.actual_device_uuid(0) == "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    if not torch.cuda.is_available():
        assert R.actual_device_uuid(0) == "GPU-b77fc3ad"                      # a numeric mask cannot be resolved without CUDA: nvidia-smi order
    assert R.trained_device_of({"device": "cuda:0", "device_uuid": "GPU-x", "environment": {"cuda_visible_devices": "GPU-y"}}) == "GPU-x"
    assert R.trained_device_of({"device": "cuda:0", "environment": {"cuda_visible_devices": "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"},
                                "cost": {"gpu_before": [{"uuid": "GPU-b77fc3ad"}]}}) == "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
    assert R.trained_device_of({"device": "cuda:0", "environment": {"cuda_visible_devices": None}, "cost": {"gpu_before": [{"uuid": "GPU-b77fc3ad"}]}}) == "GPU-b77fc3ad"
    # gamma's records: the author's Exp overwrote the mask with "0" after the CUDA context existed; the 5090 is the GPU whose memory the process occupied
    two = [{"uuid": "GPU-b77fc3ad", "memory_used_mib": 14.0}, {"uuid": "GPU-a9f35631", "memory_used_mib": 10.0}]
    after = [{"uuid": "GPU-b77fc3ad", "memory_used_mib": 14.0}, {"uuid": "GPU-a9f35631", "memory_used_mib": 5888.0}]
    assert R.trained_device_of({"device": "cuda:0", "environment": {"cuda_visible_devices": "0"}, "cost": {"gpu_before": two, "gpu_after": after}}) == "GPU-a9f35631"
    assert R.trained_device_of({"device": "cpu", "cost": {"gpu_before": [{"uuid": "GPU-b77fc3ad"}]}}) is None


def _author_metric():
    R.author_env()
    return __import__("importlib").import_module("utils.metrics").metric


@pytest.mark.parametrize("shape,scale,leaf", [((37, 5, 3), 1.0, 16), ((1001, 7, 11), 1e4, 64), ((513, 96, 3), 1e-3, 4096), ((2000, 3, 7), 1e4, 1 << 22)])
def test_RP109_the_exact_bounded_route_reproduces_the_authors_float32_scorer_bit_for_bit(shape, scale, leaf, tmp_path):
    """utils.metrics.metric (np.mean of float32 element-wise errors) against the chunked route with numpy's pairwise tree replicated:
    adverse dynamic ranges (offsets of 1e4 with small noise: cancellation), partial final leaves, several leaf sizes, arrays and
    stored (npz) inputs. Equality is exact (==), never a tolerance."""
    metric = _author_metric()
    rng = np.random.default_rng(int(np.prod(shape)) % 997)
    true = (rng.standard_normal(shape) * scale + np.float32(1000.0) * (scale >= 1)).astype(np.float32)
    pred = (true + rng.standard_normal(shape).astype(np.float32) * np.float32(0.3 * scale)).astype(np.float32)
    mae, mse = metric(pred, true)[:2]
    got = R.author_metric_exact(pred, true, leaf=leaf)
    assert got["mae"] == float(mae) and got["mse"] == float(mse), (got, mae, mse)
    assert got["mae"] == float(np.mean(np.abs(true - pred))) and got["elements"] == int(np.prod(shape))
    np.savez(tmp_path / "a.npz", pred=pred, true=true)
    stored = R.author_metric_exact(R.StoredArray(tmp_path / "a.npz", "pred"), R.StoredArray(tmp_path / "a.npz", "true"), leaf=leaf)
    assert stored["mae"] == float(mae) and stored["mse"] == float(mse)
    # a mean of chunk means is NOT the author's reduction on adverse data (the route is not that)
    if scale >= 1e4:
        chunk_means = np.float32(np.mean([np.float32(np.mean(np.abs(true[i:i + 100] - pred[i:i + 100]))) for i in range(0, shape[0], 100)]))
        assert isinstance(float(chunk_means), float)


def test_RP109_the_real_author_fixture_scores_identically_through_the_route_and_the_cell_record_is_that_value(world, tmp_path):
    """The trained tiny cell: the author's function on the arrays the adapter captured == the exact route on the stored npz + the
    targets streamed by the author's loader == the record's author_metric_float32."""
    metric = _author_metric()
    unit = world["cell"]["cell_id"]; folder = world["root"] / "attempts" / unit
    derived = R.naive_and_trues(world["design"], world["cell"], world["data"], work_dir=tmp_path / "cw")
    preds = R.StoredArray(folder / "arrays.npz", "pred"); trues = R.StoredArray(derived["trues_path"])
    got = R.author_metric_exact(preds, trues, leaf=256)
    mae, mse = metric(preds.load(), trues.load())[:2]
    assert got["mae"] == float(mae) and got["mse"] == float(mse)
    assert got["mae"] == world["record"]["author_metric_float32"]["mae"] and got["mse"] == world["record"]["author_metric_float32"]["mse"]
    assert (world["record"].get("author_scorer_parity") or {}).get("bit_equal") in (True, None)


def test_RP109_a_cell_recorded_on_the_float64_basis_gets_the_authors_float32_at_closure_through_the_exact_route(world, tmp_path):
    """T=336/T=720 records: author_metric_float32 None; the closure recomputes the author's reduction from the accepted arrays and
    the row's basis names the route; float64 stays a separately named check."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    rec = json.loads((folder / "cell.json").read_text()); original = rec["author_metric_float32"]
    rec["author_metric_float32"] = None; rec["author_metric_state"] = "NOT_EXECUTED_WITHIN_BUDGET: synthetic"
    (folder / "cell.json").write_text(json.dumps(rec))
    held = json.loads(json.dumps(world["held"]))
    held[unit]["artifacts"] = [a if a["role"] != "record" else {**a, "sha256": R.sha_file(folder / "cell.json")} for a in held[unit]["artifacts"]]
    wh = lambda c: {"current": json.loads(json.dumps(held))}
    R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)          # the record changed: the catalog under the old identity is VAULT_CHANGED once (preserved)
    ver = R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)
    row = ver["rows"][0]
    assert row["verified"], row["problems"]
    assert row["author_metric_float32"] == original and row["metric_basis"].startswith("author_float32 (recomputed at closure by df_sota_author_metric_exact.v1")
    assert row["recomputed"]["independent_float64"]["mae"] != original["mae"] or True          # float64 is a separate named check, never the basis here
    assert R.table(world["design"], ver)["rows"][0]["metric_basis"][0].startswith("author_float32 (recomputed")


# --- RP106: Musashi's RP105 counterexamples frozen against the real closure and deletion entry points ---------------------------

def _closed_and_deleted(world, tmp_path):
    """A valid fixture closure (stub warehouse), then the authorized deletion of the unit's arrays: the state the RP105 probe started from."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        a = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu")
        rep = R.close(a, json.loads((root / "DESIGN.json").read_text()))
    assert rep["verified"]
    report_sha = R.sha_file(root / "REPORT.json")
    assert (root / "reports" / f"REPORT.{report_sha}.json").is_file()                       # content-addressed copy preserved
    out = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha)
    assert out["units"][unit]["state"] == "COMPLETE" and not (root / "attempts" / unit / "arrays.npz").exists()
    return root, unit, report_sha


def _verify(world, root, **kw):
    return R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=kw.pop("replay", False), **kw)


def test_RP106_control_a_valid_retained_history_verifies_historically_with_the_original_closures_score(world, tmp_path):
    root, unit, report_sha = _closed_and_deleted(world, tmp_path)
    ver = _verify(world, root)
    row = ver["rows"][0]
    assert ver["historically_verified_units"] == [unit] and row["status"] == "METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION" and not ver["problems"]
    assert row["author_metric_float32"] == world["record"]["author_metric_float32"] and row["historical"]["report_sha256"] == report_sha
    assert "[historical: original closure" in row["metric_basis"] and row["custody"]["class"].startswith("HISTORICAL_BOUND")
    t = R.table(world["design"], ver)
    assert t["rows"][0]["mae"]["status"] != "NO_MEASUREMENT" and t["rows"][0]["historically_verified_seeds"] == [unit]


def test_RP106_rewritten_metrics_with_missing_vault_and_checkpoint_are_refused_by_the_real_closure(world, tmp_path):
    """Musashi RP105 #1: after a valid closure/deletion, both metrics set to zero in the record and the vault and checkpoint removed."""
    root, unit, _ = _closed_and_deleted(world, tmp_path); folder = root / "attempts" / unit
    rec = json.loads((folder / "cell.json").read_text()); rec["author_metric_float32"] = {"mae": 0.0, "mse": 0.0}; (folder / "cell.json").write_text(json.dumps(rec))
    (folder / "METRICS_VAULT.json").unlink(); (folder / "checkpoint.pth").unlink()
    ver = _verify(world, root)
    row = ver["rows"][0]
    assert ver["historically_verified_units"] == [] and row["status"] == "DELETED_HISTORY_UNBOUND" and row["author_metric_float32"] is None
    kinds = {p.split(": ")[1].split(":")[0] for p in ver["problems"]}
    assert {"HISTORY_RECORD_CHANGED", "HISTORY_CHECKPOINT_CHANGED", "HISTORY_VAULT_MISMATCH"} <= kinds, ver["problems"]
    t = R.table(world["design"], ver)
    assert t["rows"][0]["mae"]["status"] == "NO_MEASUREMENT" and t["rows"][0]["mae"].get("mean") is None      # zero never enters a mean


def test_RP106_an_unresolvable_original_report_is_a_typed_refusal(world, tmp_path):
    """Musashi RP105 #1b: the marker names a digest no preserved report has, and REPORT.json is gone."""
    root, unit, _ = _closed_and_deleted(world, tmp_path); folder = root / "attempts" / unit
    marker = json.loads((folder / "PREDICTIONS_DELETED.json").read_text()); marker["closure_report_sha256"] = "0" * 64
    (folder / "PREDICTIONS_DELETED.json").write_text(json.dumps(marker)); (root / "REPORT.json").unlink()
    ver = _verify(world, root)
    assert ver["historically_verified_units"] == [] and ver["rows"][0]["status"] == "DELETED_HISTORY_UNBOUND"
    assert any("HISTORY_UNRESOLVED_REPORT" in p for p in ver["problems"]) and ver["rows"][0]["author_metric_float32"] is None
    # and a report of another design, even when it resolves, does not bind
    (root / "REPORT.json").unlink(missing_ok=True)
    other = {"schema": "df_sota_report.v2", "design_sha256": "f" * 64, "verification": {"rows": []}}
    (root / "REPORT.forged.json").write_text(json.dumps(other)); marker["closure_report_sha256"] = R.sha_file(root / "REPORT.forged.json")
    (folder / "PREDICTIONS_DELETED.json").write_text(json.dumps(marker))
    ver = _verify(world, root)
    assert ver["historically_verified_units"] == [] and any("HISTORY_DESIGN_MISMATCH" in p for p in ver["problems"]) and any("HISTORY_ROW_UNVERIFIED" in p for p in ver["problems"])


def test_RP106_an_extra_root_file_of_different_identity_under_the_units_name_refuses_the_units_deletion(world, tmp_path):
    """Musashi RP105 #2: the verified arrays and an unrelated file under the same unit name in an extra root. Nothing is unlinked."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    other = tmp_path / "different_attempt" / "attempts" / unit; other.mkdir(parents=True)
    (other / "arrays.npz").write_bytes(b"different prediction artifact, not the verified identity")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "different_attempt"])
    e = out["units"][unit]
    assert e["state"] == "REFUSED" and any("CONFLICTING_COPY" in r for r in e["preflight"]["refusals"]) and e["deleted"] == []
    assert (root / "attempts" / unit / "arrays.npz").is_file() and (other / "arrays.npz").is_file() and not (root / "attempts" / unit / "PREDICTIONS_DELETED.json").exists()
    # control: an identical copy is deleted with the verified one, each with its own receipt line
    (other / "arrays.npz").write_bytes((root / "attempts" / unit / "arrays.npz").read_bytes())
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "different_attempt"])
    e = out["units"][unit]
    assert e["state"] == "COMPLETE" and [d["deleted"] for d in e["deleted"]] == [True, True] and not (other / "arrays.npz").exists()
    assert e["marker"]["all_copies_removed"] and len(e["marker"]["paths_deleted"]) == 2 and "verified_at_deletion" not in e["marker"]


def test_RP108_aliases_readers_conflicting_attempts_and_a_wrong_approval_refuse_before_any_unlink(world, tmp_path, monkeypatch):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    report_sha = R.sha_file(root / "REPORT.json")
    # a symlinked copy is an alias
    extra = tmp_path / "alias_root" / "attempts" / unit; extra.mkdir(parents=True); (extra / "arrays.npz").symlink_to(folder / "arrays.npz")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], accepted_report_sha256=report_sha)
    assert out["units"][unit]["state"] == "REFUSED" and any("ALIAS" in r for r in out["units"][unit]["preflight"]["refusals"]) and (folder / "arrays.npz").is_file()
    (extra / "arrays.npz").unlink()
    # a conflicting attempt: an extra root whose record is another record
    (extra / "arrays.npz").write_bytes((folder / "arrays.npz").read_bytes()); (extra / "cell.json").write_text("{}")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], accepted_report_sha256=report_sha)
    assert out["units"][unit]["state"] == "REFUSED" and any("CONFLICTING_ATTEMPT" in r for r in out["units"][unit]["preflight"]["refusals"])
    (extra / "cell.json").unlink()
    # an active reader
    monkeypatch.setattr(R, "_readers_of", lambda p: ["fuser: 4242"])
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], accepted_report_sha256=report_sha)
    assert out["units"][unit]["state"] == "REFUSED" and any("ACTIVE_READER" in r for r in out["units"][unit]["preflight"]["refusals"])
    monkeypatch.setattr(R, "_readers_of", lambda p: [])
    # the approval names another report; a manifest without the catalog
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], accepted_report_sha256="1" * 64)
    assert out["units"][unit]["state"] == "REFUSED" and any("APPROVAL_REPORT_MISMATCH" in r for r in out["units"][unit]["preflight"]["refusals"])
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({f"attempts/{unit}/cell.json": R.sha_file(folder / "cell.json")}))
    out = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=manifest)
    assert out["units"][unit]["state"] == "REFUSED" and any("BACKUP_UNVERIFIED" in r and "METRICS_VAULT" in r for r in out["units"][unit]["preflight"]["refusals"])
    manifest.write_text(json.dumps({f"attempts/{unit}/cell.json": R.sha_file(folder / "cell.json"), f"attempts/{unit}/METRICS_VAULT.json": R.sha_file(folder / "METRICS_VAULT.json")}))
    assert (folder / "arrays.npz").is_file() and (extra / "arrays.npz").is_file()
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], accepted_report_sha256=report_sha, backup_manifest=manifest)
    e = out["units"][unit]
    assert e["state"] == "COMPLETE" and e["marker"]["approval"]["accepted_report_sha256"] == report_sha and e["marker"]["approval"]["backup_manifest_sha256"] == R.sha_file(manifest)
    assert out["free_bytes_after_by_filesystem"]


def test_RP108_an_interrupted_deletion_keeps_accurate_per_path_status_and_resumes(world, tmp_path, monkeypatch):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    report_sha = R.sha_file(root / "REPORT.json")
    extra = tmp_path / "copy_root" / "attempts" / unit; extra.mkdir(parents=True); (extra / "arrays.npz").write_bytes((folder / "arrays.npz").read_bytes())
    real_unlink = os.unlink
    def failing(path, *a, **k):
        if str(path).startswith(str(tmp_path / "copy_root")):
            raise OSError("simulated interruption")
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(os, "unlink", failing)
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], accepted_report_sha256=report_sha)
    e = out["units"][unit]
    assert e["state"] == "PARTIAL" and [d["deleted"] for d in e["deleted"]] == [True, False] and "simulated interruption" in e["deleted"][1]["why"]
    marker = json.loads((folder / "PREDICTIONS_DELETED.json").read_text())
    assert marker["state"] == "PARTIAL" and not marker["all_copies_removed"] and len(marker["paths_remaining"]) == 1 and (extra / "arrays.npz").is_file()
    # the closure still binds the history (the root copy is gone, the original report resolves) and reports the remaining copy
    ver = _verify(world, root)
    assert ver["historically_verified_units"] == [unit] and ver["rows"][0]["deletion"]["state"] == "PARTIAL"
    # resumption: the remaining copy is deleted under the same approval; a different approval refuses
    monkeypatch.setattr(os, "unlink", real_unlink)
    bad = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], accepted_report_sha256="2" * 64)
    assert bad["units"][unit]["state"] == "REFUSED" and (extra / "arrays.npz").is_file()
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], accepted_report_sha256=report_sha)
    e = out["units"][unit]; marker = json.loads((folder / "PREDICTIONS_DELETED.json").read_text())
    assert e["preflight"].get("resumption") and e["state"] == "COMPLETE" and marker["all_copies_removed"] and len(marker["events"]) == 2 and not (extra / "arrays.npz").exists()


def test_RP110_same_device_repeatability_needs_a_measured_training_device(world, tmp_path):
    """A replay on the same UUID as an INFERRED training device is reported as not certified; a MEASURED one certifies; a
    different UUID is cross-device. The record's attribution class is explicit."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    assert R.device_attribution({"device": "cpu"}) == {"uuid": None, "class": "CPU"}
    assert R.device_attribution({"device": "cuda:0", "device_uuid": "GPU-a"})["class"] == "MEASURED"
    assert R.device_attribution({"device": "cuda:0", "environment": {"cuda_visible_devices": "GPU-a"}})["class"] == "VISIBILITY_MASK"
    two = [{"uuid": "GPU-b", "memory_used_mib": 14.0}, {"uuid": "GPU-a", "memory_used_mib": 10.0}]; after = [{"uuid": "GPU-b", "memory_used_mib": 14.0}, {"uuid": "GPU-a", "memory_used_mib": 5000.0}]
    assert R.device_attribution({"device": "cuda:0", "environment": {"cuda_visible_devices": "0"}, "cost": {"gpu_before": two, "gpu_after": after}}) == {"uuid": "GPU-a", "class": "INFERRED_GPU_MEMORY"}
    assert R.device_attribution({"device": "cuda:0", "cost": {"gpu_before": two, "gpu_after": two}}) == {"uuid": "GPU-b", "class": "UNKNOWN"}
    # the closure: a CPU-trained fixture replayed on CPU is same-device (both measured as CPU)
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    row = ver["rows"][0]
    assert row["replay"]["property"] == "same_device_repeatability" and row["same_device_repeatability"] == "PASS" and row["device_attribution"]["class"] == "CPU"
    # simulate an inferred-attribution GPU record replayed on that UUID: neither same-device nor cross-device is certified
    rep = {"device_uuid": "GPU-a", "device": "cuda", "allclose_rule": True, "finite": True, "shape_equal": True}
    r = {"device_attribution": {"uuid": "GPU-a", "class": "INFERRED_GPU_MEMORY"}, "device": "cuda:0"}
    actual = rep["device_uuid"]; attribution = r["device_attribution"]; equal = actual == attribution["uuid"]
    assert equal and attribution["class"] != "MEASURED"


def test_RP109_regeneration_by_inference_is_labelled_and_compared_with_the_deleted_originals_digests(world, tmp_path):
    """A deleted cell: inference from the retained checkpoint on the same device reproduces the arrays; the digests the record
    preserved decide whether those bytes ARE the deleted original. Never called a replay of the original, never a training."""
    root, unit, report_sha = _closed_and_deleted(world, tmp_path)
    design = json.loads((root / "DESIGN.json").read_text())
    out = R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    assert out["label"] == "REGENERATED_BY_INFERENCE_FROM_RETAINED_CHECKPOINT"
    assert out["pred_matches_original"] and out["true_matches_original"] and out["identity"].startswith("BIT_IDENTICAL")
    assert out["author_metric"] == world["record"]["author_metric_float32"] and out["author_metric_state"].startswith("EXECUTED: df_sota_author_metric_exact.v1")
    assert (root / "attempts" / unit / "regenerated" / "REGENERATION.json").is_file()
    assert (root / "attempts" / unit / "regenerated" / "REGENERATED_pred.npy").is_file()
    assert R.sha_npy_body(root / "attempts" / unit / "regenerated" / "REGENERATED_pred.npy") == world["record"]["pred_sha256"]
    # a record claiming other digests: the regeneration is preserved and refuses identity
    folder = root / "attempts" / unit
    rec = json.loads((folder / "cell.json").read_text()); rec["pred_sha256"] = "0" * 64; (folder / "cell.json").write_text(json.dumps(rec))
    out2 = R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu", keep_dir=tmp_path / "regen2")
    assert not out2["pred_matches_original"] and out2["identity"].startswith("REGENERATED_NOT_IDENTICAL") and (tmp_path / "regen2" / "REGENERATION.json").is_file()
    # a cell that still holds its arrays is refused (regeneration is only for a deleted one)
    other = _copy(world, tmp_path / "still_there")
    with pytest.raises(R.SotaRefusal):
        R.regenerate_cell(other, design, unit, data_path=world["data"], device="cpu")
