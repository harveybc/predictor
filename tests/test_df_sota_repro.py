"""RP92/RP93/RP96 acceptance of the SOTA reproduction tool: the author's code path is what runs (parser, script, loader, model,
scorer), protocol substitutions FAIL the actual closure (scaler, channels, window, horizon, target, reduction, checkpoint,
substituted predictions), row-level support and future-perturbation/normalization-fit controls hold on the author's loader, a
fresh-process checkpoint reload goes through the author's test(), and the RP97 table generator is tested. Everything runs on a
tiny SYNTHETIC csv on CPU: a software fixture, declared as such — no benchmark score is produced here."""
import copy
import hashlib
import importlib.util
import json
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
    assert (world["root"] / "REPLAYS.json").is_file() and "identity" in json.loads((world["root"] / "REPLAYS.json").read_text())[r["unit"]]
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
    assert abs(float(np.mean(np.abs(perm))) - float(np.mean(np.abs(pred)))) < 1e-12 and not np.array_equal(perm, pred)
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
