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
    assert t["complete"] and t["rows"][0]["mse"]["n_seeds"] == 1 and t["rows"][0]["mse"]["status"] in ("NUMERICAL_AGREEMENT", "PARTIAL_AGREEMENT", "DISAGREEMENT")
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


def test_RP96_a_replay_of_altered_checkpoint_bytes_fails_and_a_cached_replay_is_reused_only_for_identical_bytes(world, tmp_path):
    root = _copy(world, tmp_path)
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    (root / "REPLAYS.json").unlink(missing_ok=True)                                     # no cached replay in this copy
    first = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert first["rows"][0]["replay"]["allclose_rule"] and "adopted_from" not in first["rows"][0]["replay"]
    second = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True)
    assert "identical checkpoint" in second["rows"][0]["replay"]["adopted_from"]
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
    assert ok["status"] == "NUMERICAL_AGREEMENT" and abs(ok["tolerance_agree"] - (2 * 0.005 + 0.0005)) < 1e-12 and ok["n_seeds"] == 3 and ok["sd_ddof1"] > 0
    assert R.agreement(pub, [0.146, 0.146, 0.146], "mse")["status"] == "PARTIAL_AGREEMENT"
    assert R.agreement(pub, [0.160, 0.160, 0.160], "mse")["status"] == "DISAGREEMENT"
    assert R.agreement(pub, [], "mae")["status"] == "NO_MEASUREMENT"
    assert R.AGREEMENT["replay"]["atol"] == 1e-4 and R.AGREEMENT["std_paper"] == {"mse": 0.005, "mae": 0.006}


def test_RP96_the_metrics_vault_is_exhaustive_consistent_with_the_author_metric_and_written_at_closure(world):
    ver = R.verify_sota_run(world["root"], warehouse=_wh(world), data_path=world["data"], replay=False)
    r = ver["rows"][0]
    vp = world["root"] / "attempts" / r["unit"] / "METRICS_VAULT.json"
    assert vp.is_file() and r["recomputed"]["metrics_vault_sha256"] == R.sha_file(vp)
    v = json.loads(vp.read_text())
    assert abs(v["global"]["mse"] - r["author_metric_float32"]["mse"]) <= 1e-6 and abs(v["global"]["mae"] - r["author_metric_float32"]["mae"]) <= 1e-6
    assert v["population"] == {"windows": 77, "steps": 4, "channels": 4, "elements": 77 * 4 * 4}
    assert abs(np.mean(v["per_step"]["mse"]) - v["global"]["mse"]) <= 1e-9 and abs(np.mean(v["per_channel"]["mae"]) - v["global"]["mae"]) <= 1e-9
    assert abs(v["global"]["naive_mae"] - r["derived"]["naive"]["mae"]) <= 1e-9                     # the same persistence the closure reports
    for k in ("rmse", "mape", "mspe", "rse", "r2", "corr_pred_true", "bias", "seasonal24_mae", "skill_mae_vs_naive", "mase_vs_naive", "mutual_information_bits_pred_true_64x64"):
        assert k in v["global"] and v["global"][k] is not None
    assert set(v["residuals"]) >= {"mean", "var", "sd", "skewness", "kurtosis", "quantiles", "fraction_abs_gt", "histogram"} and sum(v["residuals"]["histogram"]["counts"]) == v["population"]["elements"]
    assert len(v["per_channel"]["r2"]) == 4 and len(v["per_step"]["error_growth_mae_over_step1"]) == 4 and v["per_step"]["error_growth_mae_over_step1"][0] == 1.0
    acf = v["autocorrelation"]["channel_mean_residual_per_step"]
    assert len(acf["acf"]) == 4 and len(acf["acf"][0]) == len(acf["lags"]) and all(-1.0001 <= x <= 1.0001 for row in acf["acf"] for x in row if x == x)
    assert "1" in v["autocorrelation"]["per_channel_first_step"] and len(v["autocorrelation"]["per_channel_first_step"]["1"]) == 4
    assert v["pred_sha256"] == json.loads((world["root"] / "attempts" / r["unit"] / "cell.json").read_text())["pred_sha256"]
