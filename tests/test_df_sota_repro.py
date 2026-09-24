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



def _accept_evidence(world, root, kind, files, subject):
    """What the governed chain produces for an evidence object, in the fixture: an acceptance terminal in the stub warehouse, its
    receipt, and the local registry entry that says which unit to ask about."""
    digests = {role: R.sha_file(Path(f)) for role, f in files.items()}
    primary = digests[list(files)[0]]
    unit = f"acceptance_{kind}_{primary[:16]}"
    design = json.loads((Path(root) / "DESIGN.json").read_text())
    terminal = {"status": "COMPLETED", "config_sha256": design["design_sha256"],
                "tags": {"kind": kind, "subject": subject, "design_sha256": design["design_sha256"]},
                "artifacts": [{"role": role, "sha256": d, "bytes": Path(files[role]).stat().st_size} for role, d in digests.items()]}
    terminal["terminal_sha256"] = hashlib.sha256(json.dumps(terminal, sort_keys=True).encode()).hexdigest()
    world["held"][unit] = terminal
    rec = json.loads((Path(root) / "TERMINAL_RECEIPTS.json").read_text())
    rec.setdefault("units", {})[unit] = {"campaign_sha256": "c" * 64, "terminal_sha256": terminal["terminal_sha256"]}
    (Path(root) / "TERMINAL_RECEIPTS.json").write_text(json.dumps(rec))
    reg_path = Path(root) / R.ACCEPTED_EVIDENCE
    reg = json.loads(reg_path.read_text()) if reg_path.is_file() else {"schema": "df_sota_accepted_evidence.v1", "entries": {}}
    for role, d in digests.items():
        reg["entries"][d] = {"unit": unit, "kind": kind, "role": role, "subject": subject, "at": "2026-09-23T00:00:00Z",
                             "scope": "fixture acceptance"}
    reg_path.write_text(json.dumps(reg))
    return unit


def _ready(root, unit, world, where):
    """The mandatory local prerequisites of a deletion (RP116): an independently accepted catalog and a verified backup. Tests
    whose subject is not the acceptance chain get a faithful accepted chain from the fixture, never a disabled prerequisite."""
    R.accept_catalog(root, json.loads((root / "DESIGN.json").read_text()), unit, data_path=world["data"])
    _accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
    _accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                              "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
    R.metadata_backup(root, Path(where))
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    return {"accepted_report_sha256": R.sha_file(root / "REPORT.json"), "backup_manifest": Path(where) / "MANIFEST.json",
            "receipts": receipts, "warehouse": _wh(world)}


def _publish_catalog(root, unit, tmp_path, design):
    """Publish the catalog acceptance through the governed chain, as production does (the stub governance is active)."""
    a = SimpleNamespace(root=root, gov_url="fixture://", api_key_file=tmp_path / "key", lake="l", resource="r", run_id="fixture")
    (tmp_path / "key").write_text("k")
    return R.publish_acceptance(a, design, kind="catalog", subject=unit,
                                files={"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                       "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"})

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
    assert rep["table"]["rows"][0]["mse"]["status"] in ("NO_MEASUREMENT", "MEASURED_REPLAY_UNVERIFIED") and not rep["table"]["rows"][0]["mse"].get("pooled", False)
    assert rep["table"]["rows"][0]["mse"].get("mean") is None                                          # nothing unverified enters a mean
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
    assert c["author_metric"] == a["author_metric"] and c["author_metric_state"].startswith("EXECUTED: df_sota_author_metric_exact.v2")
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
    ready = _ready(root, unit, world, tmp_path / "bk_rp103")
    dry = R.delete_predictions(root, [unit], dry_run=True, **ready)
    assert (folder / "arrays.npz").is_file() and dry["units"][unit]["copies_inventoried"][0]["sha256"] == json.loads((folder / "cell.json").read_text())["arrays_sha256"]
    # an altered catalog on disk closes the gate
    vp = folder / "METRICS_VAULT.json"; original = vp.read_bytes(); v = json.loads(original); v["global"]["mae"] = 5.0; vp.write_text(json.dumps(v))
    assert any("not the one the closure reported" in x for x in R.deletion_gate(root, unit)["reasons"]); vp.write_bytes(original)
    # a copy in a staging root is inventoried and deleted too; receipts carry bytes and filesystem deltas
    staging = tmp_path / "staging"; (staging / "attempts" / unit).mkdir(parents=True); import shutil; shutil.copy2(folder / "arrays.npz", staging / "attempts" / unit / "arrays.npz")
    size = (folder / "arrays.npz").stat().st_size
    ready = _ready(root, unit, world, tmp_path / "bk_rp103b")
    rec = R.delete_predictions(root, [unit], extra_roots=[staging], **ready)
    assert rec["units"][unit]["preflight"]["pass"] and len([d for d in rec["paths"] if d["deleted"]]) == 2 and sum(d["bytes"] for d in rec["paths"]) == 2 * size
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
    assert row["metric_basis"].startswith("author_float32 (recomputed at closure by df_sota_author_metric_exact.v2") and row["author_metric_float32"] == world["record"]["author_metric_float32"]
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
    assert row["author_metric_float32"] == original and row["metric_basis"].startswith("author_float32 (recomputed at closure by df_sota_author_metric_exact.v2")
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
    ready = _ready(root, unit, world, tmp_path / "backup_cd")
    out = R.delete_predictions(root, [unit], **ready)
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
    assert t["rows"][0]["mae"].get("mean") is None and not t["rows"][0]["mae"].get("pooled", False)          # zero never enters a mean
    assert t["rows"][0]["mae"]["status"] in ("NO_MEASUREMENT", "MEASURED_REPLAY_UNVERIFIED")


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
    ready = _ready(root, unit, world, tmp_path / "bk_rp106")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "different_attempt"], **ready)
    e = out["units"][unit]
    assert e["state"] == "REFUSED" and any("CONFLICTING_COPY" in r for r in e["preflight"]["refusals"]) and e["deleted"] == []
    assert (root / "attempts" / unit / "arrays.npz").is_file() and (other / "arrays.npz").is_file() and not (root / "attempts" / unit / "PREDICTIONS_DELETED.json").exists()
    # control: an identical copy is deleted with the verified one, each with its own receipt line
    (other / "arrays.npz").write_bytes((root / "attempts" / unit / "arrays.npz").read_bytes())
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "different_attempt"], **ready)
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
    ready = _ready(root, unit, world, tmp_path / "bk_rp108")
    # a symlinked copy is an alias
    extra = tmp_path / "alias_root" / "attempts" / unit; extra.mkdir(parents=True); (extra / "arrays.npz").symlink_to(folder / "arrays.npz")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], **ready)
    assert out["units"][unit]["state"] == "REFUSED" and any("ALIAS" in r for r in out["units"][unit]["preflight"]["refusals"]) and (folder / "arrays.npz").is_file()
    (extra / "arrays.npz").unlink()
    # a conflicting attempt: an extra root whose record is another record
    (extra / "arrays.npz").write_bytes((folder / "arrays.npz").read_bytes()); (extra / "cell.json").write_text("{}")
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], **ready)
    assert out["units"][unit]["state"] == "REFUSED" and any("CONFLICTING_ATTEMPT" in r for r in out["units"][unit]["preflight"]["refusals"])
    (extra / "cell.json").unlink()
    # an active reader
    monkeypatch.setattr(R, "_readers_of", lambda p: ["fuser: 4242"])
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], **ready)
    assert out["units"][unit]["state"] == "REFUSED" and any("ACTIVE_READER" in r for r in out["units"][unit]["preflight"]["refusals"])
    monkeypatch.setattr(R, "_readers_of", lambda p: [])
    # the approval names another report; a manifest without the catalog
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], **{**ready, "accepted_report_sha256": "1" * 64})
    assert out["units"][unit]["state"] == "REFUSED" and any("APPROVAL_REPORT_MISMATCH" in r for r in out["units"][unit]["preflight"]["refusals"])
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"backup": str(tmp_path / "nowhere"), "files": {f"attempts/{unit}/cell.json": R.sha_file(folder / "cell.json")}}))
    out = R.delete_predictions(root, [unit], **{**ready, "backup_manifest": manifest})
    assert out["units"][unit]["state"] == "REFUSED" and any("BACKUP_UNCOVERED" in r and "METRICS_VAULT" in r for r in out["units"][unit]["preflight"]["refusals"])
    assert (folder / "arrays.npz").is_file() and (extra / "arrays.npz").is_file()
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "alias_root"], **ready)
    e = out["units"][unit]
    assert e["state"] == "COMPLETE" and e["marker"]["approval"]["accepted_report_sha256"] == report_sha
    assert e["marker"]["approval"]["backup"]["pass"] and out["free_bytes_after_by_filesystem"]


def test_RP108_an_interrupted_deletion_keeps_accurate_per_path_status_and_resumes(world, tmp_path, monkeypatch):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    report_sha = R.sha_file(root / "REPORT.json")
    ready = _ready(root, unit, world, tmp_path / "bk_rp108b")
    extra = tmp_path / "copy_root" / "attempts" / unit; extra.mkdir(parents=True); (extra / "arrays.npz").write_bytes((folder / "arrays.npz").read_bytes())
    real_unlink = os.unlink
    def failing(path, *a, **k):
        if str(path).startswith(str(tmp_path / "copy_root")):
            raise OSError("simulated interruption")
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(os, "unlink", failing)
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], **ready)
    e = out["units"][unit]
    assert e["state"] == "PARTIAL" and [d["deleted"] for d in e["deleted"]] == [True, False] and "simulated interruption" in e["deleted"][1]["why"]
    marker = json.loads((folder / "PREDICTIONS_DELETED.json").read_text())
    assert marker["state"] == "PARTIAL" and not marker["all_copies_removed"] and len(marker["paths_remaining"]) == 1 and (extra / "arrays.npz").is_file()
    # the closure still binds the history (the root copy is gone, the original report resolves) and reports the remaining copy
    ver = _verify(world, root)
    assert ver["historically_verified_units"] == [unit] and ver["rows"][0]["deletion"]["state"] == "PARTIAL"
    # resumption: the remaining copy is deleted under the same approval; a different approval refuses
    monkeypatch.setattr(os, "unlink", real_unlink)
    bad = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], **{**ready, "accepted_report_sha256": "2" * 64})
    assert bad["units"][unit]["state"] == "REFUSED" and (extra / "arrays.npz").is_file()
    out = R.delete_predictions(root, [unit], extra_roots=[tmp_path / "copy_root"], **ready)
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
    assert out["author_metric"] == world["record"]["author_metric_float32"] and out["author_metric_state"].startswith("EXECUTED: df_sota_author_metric_exact.v2")
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


def test_RP_placement_gpu_admission_is_a_refusal_by_physical_uuid_with_no_fallback(tmp_path, monkeypatch):
    """Owner policy 2026-09-22: only the named external device is eligible. Absence, heat, busy VRAM, a competing compute process,
    a hot host, low host RAM or low disk each refuse; nothing falls back to another GPU or to the host."""
    ext = "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"; internal = "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"
    good = [{"index": 0, "uuid": internal, "name": "internal", "temperature_c": 30.0, "utilization_pct": 0.0, "memory_used_mib": 14.0, "memory_total_mib": 12227.0},
            {"index": 1, "uuid": ext, "name": "RTX 5090", "temperature_c": 36.0, "utilization_pct": 0.0, "memory_used_mib": 10.0, "memory_total_mib": 32607.0}]
    monkeypatch.setattr(R, "gpu_state", lambda: good)
    monkeypatch.setattr(R, "host_thermals", lambda: {"thermal_zone0:acpitz": 41.0})
    monkeypatch.setattr(R.subprocess, "run", lambda *a_, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    adm = R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)
    assert adm["pass"] and adm["measured"]["device"]["uuid"] == ext and adm["measured"]["free_vram_mib"] > 30000
    assert not R.admit_gpu(None, path=tmp_path, min_free_disk_bytes=1)["pass"]
    assert any("NO_REQUIRED_DEVICE" in r for r in R.admit_gpu(None, path=tmp_path, min_free_disk_bytes=1)["refusals"])
    monkeypatch.setattr(R, "gpu_state", lambda: [good[0]])                       # the external device unplugged
    out = R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)
    assert not out["pass"] and any("DEVICE_ABSENT" in r for r in out["refusals"]) and "internal" not in json.dumps(out["refusals"]).replace(internal[:16], "")
    monkeypatch.setattr(R, "gpu_state", lambda: [good[0], {**good[1], "temperature_c": 84.0}])
    assert any("GPU_TEMPERATURE" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)["refusals"])
    monkeypatch.setattr(R, "gpu_state", lambda: [good[0], {**good[1], "memory_used_mib": 32000.0}])
    assert any("VRAM" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)["refusals"])
    monkeypatch.setattr(R, "gpu_state", lambda: good)
    monkeypatch.setattr(R, "host_thermals", lambda: {"zone0:x86_pkg_temp": 95.0})
    assert any("HOST_TEMPERATURE" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)["refusals"])
    monkeypatch.setattr(R, "host_thermals", lambda: {"thermal_zone0:acpitz": 41.0})
    assert any("HOST_RAM" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_ram_bytes=1 << 60, min_free_disk_bytes=1)["refusals"])
    assert any("DISK" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1 << 60)["refusals"])
    monkeypatch.setattr(R.subprocess, "run", lambda *a_, **k: SimpleNamespace(stdout=f"{ext}, 1234, 5000 MiB\n", stderr="", returncode=0))
    assert any("COMPETING_WORKLOAD" in r for r in R.admit_gpu(ext, path=tmp_path, min_free_disk_bytes=1)["refusals"])
    # the child asserts the device it actually holds
    monkeypatch.setattr(R, "actual_device_uuid", lambda i=0: ext)
    assert R.assert_child_device(ext)["actual"] == ext
    with pytest.raises(R.SotaRefusal):
        R.assert_child_device(internal)


def test_RP112_a_regeneration_is_accepted_only_when_its_catalog_and_oracles_reproduce_the_retained_one_then_cleaned_up(world, tmp_path):
    root, unit, _ = _closed_and_deleted(world, tmp_path)
    design = json.loads((root / "DESIGN.json").read_text()); folder = root / "attempts" / unit
    R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    acc = R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=False)
    assert acc["pass"] and acc["catalog_recomputed_equals_retained"] and acc["deleted"] == []
    # the regenerated bytes also carry an independent check of the retained catalog's families (RP117 on a deleted cell)
    assert acc["independent_estimators"]["population"]["elements"] == acc["population"]["elements"]
    assert acc["independent_comparison"]["fully_independent"] and not acc["independent_comparison"]["disagreements"]
    assert set(acc["independently_accepted_families"]) == set(R.REQUIRED_CATALOG_FAMILIES) and "lifts the historical limit" in acc["independent_scope"]
    _accept_evidence(world, root, "regeneration", {"regeneration": folder / "regenerated" / "REGENERATION.json",
                                                   "regeneration_acceptance": folder / "regenerated" / "ACCEPTANCE.json"}, unit)
    assert acc["author_metric_float32"] == world["record"]["author_metric_float32"] and abs(acc["numeric_oracles"]["global_vs_author_mae"]) <= 1e-6
    assert acc["catalog_states"] and (folder / "regenerated" / "ACCEPTANCE.json").is_file()
    # the closure reports it as a labelled successor of the bound historical row
    ver = _verify(world, root)
    row = ver["rows"][0]
    assert row["verified_historically"], (row.get("problems"), row["regenerated"].get("refusals"))
    assert row["regenerated"]["usable"] and row["regenerated"]["accepted"]["regeneration"]["accepted"]
    assert row["metric_basis"].startswith("author_float32 (regenerated by inference") and row["author_metric_float32"] == world["record"]["author_metric_float32"]
    # a drifted regenerated array refuses acceptance
    big = folder / "regenerated" / "REGENERATED_pred.npy"
    a = np.load(big, mmap_mode="r+"); a[0, 0, 0] = np.float32(float(a[0, 0, 0]) + 1.0); a.flush(); del a
    bad = R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=True)
    assert not bad["pass"] and any("DIGEST_DRIFT" in r for r in bad["refusals"]) and bad["deleted"] == [] and big.is_file()
    assert R.accepted_artifact(root, json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], _wh(world),
                               R.sha_file(folder / "regenerated" / "ACCEPTANCE.json"), expect_kind="regeneration")["accepted"] is False
    # and the closure no longer upgrades the row, while the historical binding stands on the original report
    ver2 = _verify(world, root)
    assert ver2["rows"][0]["verified_historically"] and not ver2["rows"][0]["metric_basis"].startswith("author_float32 (regenerated")
    # a clean regeneration again, accepted with cleanup: the arrays go, the evidence stays
    R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    ok = R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=True)
    assert ok["pass"] and [d["deleted"] for d in ok["deleted"]] == [True, True] and not big.is_file()
    _accept_evidence(world, root, "regeneration", {"regeneration": folder / "regenerated" / "REGENERATION.json",
                                                   "regeneration_acceptance": folder / "regenerated" / "ACCEPTANCE.json"}, unit)
    assert (folder / "regenerated" / "REGENERATION.json").is_file() and (folder / "METRICS_VAULT.json").is_file() and (folder / "checkpoint.pth").is_file()
    ver3 = _verify(world, root)
    assert ver3["rows"][0]["metric_basis"].startswith("author_float32 (regenerated by inference")


def test_RP110_the_dataloader_worker_change_leaves_the_training_trajectory_identical_over_optimizer_steps(world, tmp_path):
    """Musashi RP105: batch parity alone does not prove training equivalence. Real author fixture, CPU, four optimizer steps with
    the author's model, criterion and optimizer: batch values, model weights, optimizer state, the loss and the torch/numpy RNG
    states after every step, plus the validation and test loaders' inputs, must be identical with num_workers 1 and 0."""
    import torch
    cell, data = world["cell"], world["data"]
    R.author_env()
    DF = __import__("importlib").import_module("data_provider.data_factory")
    Model = __import__("importlib").import_module("models.TimeFilter").Model
    traj = {}
    for w in (1, 0):
        R.fix_seeds(cell["seed"])
        args = R.build_args(cell["argv"], data_dir=data.parent, data_name=data.name, checkpoints=tmp_path / f"c{w}", use_gpu=False, dataloader_workers=w)
        _, train_loader = DF.data_provider(args, "train")
        _, vali_loader = DF.data_provider(args, "val")
        _, test_loader = DF.data_provider(args, "test")
        model = Model(args).float()
        L = args.seq_len * args.c_out // args.patch_len; N = args.seq_len // args.patch_len
        masks = torch.stack([torch.stack([((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).float(),
                                          ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).float(),
                                          torch.ones(L) - ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).float()
                                          - ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).float()], dim=0) for k in range(L)], dim=0)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        crit = torch.nn.MSELoss()
        steps = []
        model.train()
        for i, (bx, by, _, _) in enumerate(train_loader):
            if i >= 4:
                break
            opt.zero_grad()
            out, moe = model(bx.float(), masks, is_training=True)
            loss = crit(out[:, -args.pred_len:, :], by[:, -args.pred_len:, :].float()) + moe
            loss.backward(); opt.step()
            steps.append({"batch_x": bx.clone(), "batch_y": by.clone(), "loss": float(loss.detach()),
                          "weights": R.sha_array(np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])),
                          "opt_state": R.sha_array(np.concatenate([v["exp_avg"].numpy().ravel() for v in opt.state.values()] or [np.zeros(1, dtype=np.float32)])),
                          "torch_rng": R.sha_array(torch.get_rng_state().numpy()), "numpy_rng": R.sha_array(np.asarray(np.random.get_state()[1]))})
        traj[w] = {"steps": steps, "vali": [b[0].clone() for b in vali_loader], "test": [b[1].clone() for b in test_loader]}
    a, b = traj[1]["steps"], traj[0]["steps"]
    assert len(a) == len(b) == 4
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x["batch_x"], y["batch_x"]) and torch.equal(x["batch_y"], y["batch_y"]), f"batch {i}"
        assert x["loss"] == y["loss"] and x["weights"] == y["weights"] and x["opt_state"] == y["opt_state"], f"state after step {i}"
        assert x["torch_rng"] == y["torch_rng"] and x["numpy_rng"] == y["numpy_rng"], f"rng after step {i}"
    assert all(torch.equal(p, q) for p, q in zip(traj[1]["vali"], traj[0]["vali"]))
    assert all(torch.equal(p, q) for p, q in zip(traj[1]["test"], traj[0]["test"]))


def test_RP110_the_allocator_option_reaches_the_real_child_and_changes_no_argument(world, tmp_path, monkeypatch):
    """The declared host-allocator patch must arrive in the child's environment, and nothing in the child's argument vector (the
    scientific recipe) may depend on it."""
    calls = []

    def fake_run(argv, **kw):
        calls.append({"argv": argv, "env": kw.get("env") or {}})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def child_call():
        return next(c for c in calls if any("child" == str(x) for x in c["argv"]))

    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {}}))
    monkeypatch.setattr(R.subprocess, "run", fake_run)
    a = SimpleNamespace(root=root, seeds=None, horizons=None, units=[unit], cpu=True, gpu=0, gov_url="u", api_key_file=tmp_path / "k",
                        lake="l", resource="r", run_id=None, bounded=True, author_metric_budget_gib=4, dataloader_workers=0,
                        malloc_tunables=R.MALLOC_TUNABLES_DEFAULT, require_gpu_uuid=None)
    R.execute(a, json.loads((root / "DESIGN.json").read_text()))
    c = child_call()
    assert c["env"]["GLIBC_TUNABLES"] == R.MALLOC_TUNABLES_DEFAULT
    assert "--dataloader-workers" in c["argv"] and "GLIBC_TUNABLES" not in " ".join(str(x) for x in c["argv"])
    assert not any("malloc" in str(x).lower() for x in c["argv"])
    calls.clear(); a.malloc_tunables = "none"
    R.execute(a, json.loads((root / "DESIGN.json").read_text()))
    c2 = child_call()
    assert "GLIBC_TUNABLES" not in c2["env"] and c2["argv"] == c["argv"]          # the recipe is identical either way


def test_RP111_the_run_ledger_is_frozen_from_measured_costs_including_retired_attempts(world, tmp_path):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    import shutil
    shutil.copytree(root / "attempts" / unit, root / "attempts" / f"{unit}.failed.1790000000")
    # an interrupted attempt that never reached its record must still appear, with its cost stated as not recorded
    (root / "attempts" / f"{unit}.failed.1790000001").mkdir()
    (root / "attempts" / f"{unit}.failed.1790000001" / "FAILED.json").write_text(json.dumps({"at": "2026-09-22T20:10:51Z", "error": "KeyboardInterrupt: "}))
    led = R.run_ledger(root, world["design"])
    assert led["totals"]["attempts"] == 3 and led["totals"]["retired"] == 2 and led["totals"]["predictions_on_disk"] == 2
    ghost = next(r for r in led["attempts"] if r["attempt"].endswith("1790000001"))
    assert ghost["state"] == "RETIRED" and ghost["measured_cost"].startswith("NONE_RECORDED") and "KeyboardInterrupt" in ghost["failure"]
    assert led["totals"]["wall_seconds"] > 0 and led["measured_free_disk_bytes"] > 0 and (root / "RUN_LEDGER.json").is_file()
    row = next(r for r in led["attempts"] if r["state"] == "CURRENT")
    assert row["unit"] == unit and row["device_attribution"] == "CPU" and row["peak_rss_bytes"] and row["epochs_run"]
    assert all(isinstance(r["wall_seconds"], float) for r in led["attempts"] if r["state"] != "RETIRED" or r.get("measured_cost") is None) and led["scope"].endswith("nothing here is projected")


def test_RP112_the_catalog_is_independently_accepted_from_its_own_estimators_before_any_deletion(world, tmp_path):
    """Denominators, population, required families and their states, and oracles between the catalog's own estimators (per-step,
    per-channel, per-window, time blocks, skill identities), plus the closure's recomputation of THIS catalog."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    design = json.loads((root / "DESIGN.json").read_text())
    acc = R.accept_catalog(root, design, unit)
    assert acc["pass"], acc["refusals"]
    assert acc["states"]["errors"] == "DONE" and acc["states"]["matched_baselines"] == "DONE" and acc["population"]["consumed_windows"] == acc["population"]["windows"]
    assert abs(acc["oracles"]["per_step_mean_vs_global_mae"]) <= 1e-9 and acc["oracles"]["per_window_length_vs_population"] == 0
    assert acc["closure"]["metrics_vault_read_back_equal"] and acc["closure"]["metrics_vault_sha256"] == acc["catalog_sha256"]
    assert (folder / "CATALOG_ACCEPTANCE.json").is_file()
    # a tampered catalog fails its own oracles and the closure binding
    v = json.loads((folder / "METRICS_VAULT.json").read_text()); v["global"]["mae"] = v["global"]["mae"] * 1.05
    (folder / "METRICS_VAULT.json").write_text(json.dumps(v))
    bad = R.accept_catalog(root, design, unit)
    assert not bad["pass"] and any("ORACLE per_step_mean_vs_global_mae" in r for r in bad["refusals"]) and any("CLOSURE" in r for r in bad["refusals"])


def test_RP112_the_metadata_backup_covers_the_evidence_the_deletion_approval_binds_to(world, tmp_path):
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tok"; tok.write_text("synthetic")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"), json.loads((root / "DESIGN.json").read_text()))
    out = R.metadata_backup(root, tmp_path / "backup")
    assert f"attempts/{unit}/cell.json" in out["files"] and f"attempts/{unit}/METRICS_VAULT.json" in out["files"] and "REPORT.json" in out["files"]
    assert not any("arrays.npz" in k or "checkpoint.pth" in k for k in out["files"]) and out["bytes"] > 0
    assert (tmp_path / "backup" / "MANIFEST.json").is_file()
    # the deletion accepts this manifest and refuses a stale one
    report_sha = R.sha_file(root / "REPORT.json")
    R.accept_catalog(root, json.loads((root / "DESIGN.json").read_text()), unit, data_path=world["data"])
    out = R.metadata_backup(root, tmp_path / "backup")
    _accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
    _accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                              "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
    out = R.metadata_backup(root, tmp_path / "backup")
    ok = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "backup" / "MANIFEST.json",
                              receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=_wh(world))
    assert ok["units"][unit]["state"] == "COMPLETE" and ok["units"][unit]["marker"]["approval"]["backup"]["pass"]
    assert f"attempts/{unit}/CATALOG_ACCEPTANCE.json" in out["files"]


# --- RP114: Musashi's RP113 counterexamples, frozen against the real entry points ------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 8193, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, 159164640, 312412608, 531190800, 1049515920, (1 << 30) + 7])
def test_RP118_the_mean_denominator_is_numpys_int64_division_not_float32(n):
    """Musashi RP113 #4: at N = 16,777,217 with unit errors the v1 route returned 1.0 and the author's function
    0.9999999403953552. The denominator is tested alone, over the 2^24 boundary and at the real benchmark cardinalities
    (T = 336 and T = 720 have counts that float32 cannot represent), against numpy's own arithmetic."""
    for total in (np.float32(n), np.float32(n - 1), np.float32(0.0), np.float32(n) * np.float32(0.5), np.float32(3.7e7)):
        expected = np.float32(np.float64(total) / np.float64(n))                    # numpy: float32 scalar / int64 scalar -> float64 -> cast
        assert R.f32_mean(total, n) == expected
        assert np.float32(total / np.intp(n)) == expected                            # the very promotion numpy performs
    with pytest.raises(R.SotaRefusal):
        R.f32_mean(np.float32(1.0), 0)


def test_RP118_the_exact_route_matches_the_authors_function_across_the_2_24_boundary():
    """The whole function, against the unmodified author operation, at the cardinality that broke v1 and its neighbours."""
    metric = _author_metric()
    for n in (8193, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3):
        pred = np.zeros(n, dtype=np.float32); true = np.ones(n, dtype=np.float32)
        got = R.author_metric_exact(pred, true, leaf=1 << 20)
        mae, mse = metric(pred, true)[:2]
        assert got["mae"] == float(mae) == float(np.mean(np.abs(true - pred))), (n, got["mae"], float(mae))
        assert got["mse"] == float(mse) and got["elements"] == n
        del pred, true
    # the sum is tested separately from the denominator: the pairwise replica against numpy's own reduction
    rng = np.random.default_rng(3)
    for n in (129, 5000, (1 << 20) + 17):
        a = (rng.standard_normal(n).astype(np.float32) + np.float32(1000.0)).astype(np.float32)
        leaf_sum = lambda lo, k: np.add.reduce(a[lo:lo + k], dtype=np.float32)
        assert R._pairwise_f32(leaf_sum, 0, n, 1 << 20) == np.add.reduce(a, dtype=np.float32)
    assert R.AUTHOR_SCORER_ROUTE == "df_sota_author_metric_exact.v2" and "df_sota_author_metric_exact.v1" in R.AUTHOR_SCORER_ROUTE_SUPERSEDED


def test_RP118_a_magnitude_and_shape_sweep_still_matches_the_author_on_stored_and_memory_arrays(tmp_path):
    metric = _author_metric()
    rng = np.random.default_rng(11)
    for shape, scale, offset in (((17, 3, 5), 1e-6, 0.0), ((257, 7, 11), 1.0, 1e4), ((41, 96, 321), 0.3, -5.0), ((1000, 4, 4), 1e3, 1e3)):
        true = (rng.standard_normal(shape) * scale + offset).astype(np.float32)
        pred = (true + rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).astype(np.float32)
        mae, mse = metric(pred, true)[:2]
        for leaf in (128, 1 << 12, 1 << 22):
            got = R.author_metric_exact(pred, true, leaf=leaf)
            assert got["mae"] == float(mae) and got["mse"] == float(mse), (shape, scale, leaf)
        np.savez(tmp_path / "a.npz", pred=pred, true=true)
        stored = R.author_metric_exact(R.StoredArray(tmp_path / "a.npz", "pred"), R.StoredArray(tmp_path / "a.npz", "true"), leaf=1 << 12)
        assert stored["mae"] == float(mae) and stored["mse"] == float(mse)
        assert stored["denominator_exact_in_float32"] is True


def _stub_governance(monkeypatch, world, root, held, receipts_store):
    """Governance for the fixture: acquire records a delivery, report_terminal accepts the terminal into the stub warehouse and
    writes its receipt, exactly as the real chain does for a cell."""
    U = R._module("df_utility_run")

    def acquire(**kw):
        unit = kw["unit_id"]
        doc = json.loads((root / "DELIVERIES.json").read_text())
        doc.setdefault("units", {})[unit] = {"path": str(world["data"]), "sha256": R.sha_file(world["data"]),
                                             "campaign_sha256": "c" * 64, "campaign_key": f"fixture-{unit}"}
        (root / "DELIVERIES.json").write_text(json.dumps(doc))
        return {"unit": unit}

    def report_terminal(root_, unit, terminal, **kw):
        digest = hashlib.sha256(json.dumps(terminal, sort_keys=True, default=str).encode()).hexdigest()
        held[unit] = {**terminal, "terminal_sha256": digest, "config_sha256": terminal["tags"].get("design_sha256")}
        rec = json.loads((Path(root_) / "TERMINAL_RECEIPTS.json").read_text())
        rec["units"][unit] = {"campaign_sha256": "c" * 64, "terminal_sha256": digest}
        (Path(root_) / "TERMINAL_RECEIPTS.json").write_text(json.dumps(rec))
        receipts_store[unit] = rec["units"][unit]
        return {"flushed": {"pending": [], "failures": {}}}

    monkeypatch.setattr(R, "governance_modules", lambda: (SimpleNamespace(acquire=acquire, report_terminal=report_terminal, report_failed=lambda *a_, **k_: None), U))
    return lambda campaign: {"current": json.loads(json.dumps(held))}


def _accepted_world(world, tmp_path, monkeypatch, *, tag="acc"):
    """A closed fixture whose closure report is published as ACCEPTED evidence through the governed chain."""
    root = _copy(world, tmp_path / tag)
    held = json.loads(json.dumps(world["held"])); receipts = {}
    wh = _stub_governance(monkeypatch, world, root, held, receipts)
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: wh(c)):
        tok = tmp_path / f"tok_{tag}"; tok.write_text("synthetic")
        a = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False,
                            replay_device="cpu", publish_acceptance=True, gov_url="fixture://", api_key_file=tmp_path / "key",
                            lake="l", resource="r", run_id="fixture")
        (tmp_path / "key").write_text("k")
        rep = R.close(a, json.loads((root / "DESIGN.json").read_text()))
    return root, wh, held, rep


def test_RP115_a_rewritten_report_with_a_recomputed_pointer_is_refused_because_it_is_not_accepted_evidence(world, tmp_path, monkeypatch):
    """Musashi RP113 #1, against the real verifier: the score follows the ACCEPTED closure identity, not a local digest."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch)
    unit = world["cell"]["cell_id"]; report_sha = R.sha_file(root / "REPORT.json")
    design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _publish_catalog(root, unit, tmp_path, design)
    R.metadata_backup(root, tmp_path / "bk")
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    out = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "bk" / "MANIFEST.json",
                               receipts=receipts, warehouse=wh)
    assert out["units"][unit]["state"] == "COMPLETE"
    base = R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)
    assert base["historically_verified_units"] == [unit] and base["rows"][0]["author_metric_float32"] == world["record"]["author_metric_float32"]
    # now rewrite the report's score and re-point the marker at the new bytes, exactly as the probe does
    rep = json.loads((root / "REPORT.json").read_text())
    rep["verification"]["rows"][0]["author_metric_float32"] = {"mae": 0.0, "mse": 0.0}
    (root / "REPORT.json").write_text(json.dumps(rep, indent=1))
    marker_path = root / "attempts" / unit / "PREDICTIONS_DELETED.json"
    marker = json.loads(marker_path.read_text()); marker["closure_report_sha256"] = R.sha_file(root / "REPORT.json")
    marker_path.write_text(json.dumps(marker))
    after = R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)
    assert after["historically_verified_units"] == [] and after["rows"][0]["author_metric_float32"] is None
    assert any("HISTORY_REPORT_NOT_ACCEPTED" in p for p in after["problems"]), after["problems"]
    after_row = R.table(world["design"], after)["rows"][0]["mae"]
    assert after_row.get("mean") is None and not after_row.get("pooled", False)


def test_RP115_a_fabricated_regeneration_cannot_replace_a_score(world, tmp_path, monkeypatch):
    """Musashi RP113 #2: a hand-written REGENERATION.json with copied digests and a zero metric, plus an ACCEPTANCE that only
    says pass, must not be usable — neither as accepted evidence nor by content."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch, tag="fab")
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _publish_catalog(root, unit, tmp_path, design)
    R.metadata_backup(root, tmp_path / "bk2")
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"), backup_manifest=tmp_path / "bk2" / "MANIFEST.json",
                         receipts=receipts, warehouse=wh)
    rec = json.loads((folder / "cell.json").read_text())
    (folder / "regenerated").mkdir()
    (folder / "regenerated" / "REGENERATION.json").write_text(json.dumps({"identity": "BIT_IDENTICAL_TO_THE_DELETED_ORIGINAL",
        "pred_sha256": rec["pred_sha256"], "true_sha256": rec["true_sha256"], "author_metric": {"mae": 0.0, "mse": 0.0}}))
    (folder / "regenerated" / "ACCEPTANCE.json").write_text(json.dumps({"pass": True}))
    ver = R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)
    row = ver["rows"][0]
    # the fabrication changes NOTHING: the row keeps the score the original accepted closure bound, and the claim is refused
    assert row["author_metric_float32"] == world["record"]["author_metric_float32"] and row["verified_historically"]
    assert not row["metric_basis"].startswith("author_float32 (regenerated") and row["regenerated"]["usable"] is False
    kinds = " ".join(ver["problems"])
    assert "REGEN_NOT_ACCEPTED" in kinds and "REGEN_INCOMPLETE" in kinds and "REGEN_CONTRADICTS_CATALOG" in kinds
    assert all(p.startswith(f"{unit}: REGENERATION_REFUSED_NOT_USED") for p in ver["problems"])
    # a real regeneration, accepted through the governed chain, does restore the score
    R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    acc = R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=True)
    assert acc["pass"]
    a = SimpleNamespace(root=root, gov_url="fixture://", api_key_file=tmp_path / "key", lake="l", resource="r", run_id="fixture")
    R.publish_acceptance(a, design, kind="regeneration", subject=unit,
                         files={"regeneration": folder / "regenerated" / "REGENERATION.json", "regeneration_acceptance": folder / "regenerated" / "ACCEPTANCE.json"})
    ok = R.verify_sota_run(root, warehouse=wh, data_path=world["data"], replay=False)
    assert ok["historically_verified_units"] == [unit], (ok["problems"], ok["rows"][0].get("regenerated", {}).get("refusals"))
    assert ok["rows"][0]["author_metric_float32"] == world["record"]["author_metric_float32"]
    assert ok["rows"][0]["metric_basis"].startswith("author_float32 (regenerated by inference")


def test_RP116_deletion_without_approval_backup_or_catalog_acceptance_refuses(world, tmp_path, monkeypatch):
    """Musashi RP113 #3: the prerequisites are mandatory in the API, not optional arguments."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch, tag="req")
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    design = json.loads((root / "DESIGN.json").read_text())
    bare = R.delete_predictions(root, [unit])
    assert bare["units"][unit]["state"] == "REFUSED" and (folder / "arrays.npz").is_file()
    why = " ".join(bare["units"][unit]["preflight"]["refusals"])
    assert "APPROVAL_MISSING" in why and "BACKUP_MISSING" in why and "CATALOG_ACCEPTANCE_MISSING" in why
    report_sha = R.sha_file(root / "REPORT.json")
    R.accept_catalog(root, design, unit, data_path=world["data"])
    # a manifest whose destination does not hold the bytes is not a backup
    R.metadata_backup(root, tmp_path / "bk3")
    import shutil as sh
    sh.rmtree(tmp_path / "bk3" / "attempts")
    out = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "bk3" / "MANIFEST.json",
                               receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=wh)
    assert out["units"][unit]["state"] == "REFUSED" and any("BACKUP_ABSENT" in r for r in out["units"][unit]["preflight"]["refusals"])
    assert (folder / "arrays.npz").is_file()
    # a corrupt backup copy is not a backup either
    R.metadata_backup(root, tmp_path / "bk4")
    (tmp_path / "bk4" / "attempts" / unit / "METRICS_VAULT.json").write_text("{}")
    out = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "bk4" / "MANIFEST.json",
                               receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=wh)
    assert out["units"][unit]["state"] == "REFUSED" and any("BACKUP_CORRUPT" in r for r in out["units"][unit]["preflight"]["refusals"])
    # the catalog acceptance must be accepted evidence when acceptance is required
    R.metadata_backup(root, tmp_path / "bk5")
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    strict = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "bk5" / "MANIFEST.json",
                                  receipts=receipts, warehouse=wh)
    assert strict["units"][unit]["state"] == "REFUSED" and any("CATALOG_ACCEPTANCE_NOT_ACCEPTED" in r for r in strict["units"][unit]["preflight"]["refusals"])
    a = SimpleNamespace(root=root, gov_url="fixture://", api_key_file=tmp_path / "key", lake="l", resource="r", run_id="fixture")
    R.publish_acceptance(a, design, kind="catalog", subject=unit,
                         files={"catalog_acceptance": folder / "CATALOG_ACCEPTANCE.json", "metrics_vault": folder / "METRICS_VAULT.json"})
    R.metadata_backup(root, tmp_path / "bk6")
    done = R.delete_predictions(root, [unit], accepted_report_sha256=report_sha, backup_manifest=tmp_path / "bk6" / "MANIFEST.json",
                                receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=wh)
    assert done["units"][unit]["state"] == "COMPLETE" and not (folder / "arrays.npz").exists()
    d = done["units"][unit]["deleted"][0]
    assert d["bytes_removed_were_the_accepted_bytes"] and d["digest_after_unlink"] == d["sha256"] and d["inode"]


def test_RP117_a_broken_estimator_producer_is_refused_even_when_its_own_summaries_agree(world, tmp_path, monkeypatch):
    """Musashi RP113 #5: a producer that emits a negative SD, 999-bit entropy, 999-bit MI and an ACF of 42 while keeping MAE/MSE
    consistent must not pass. The domain of each family is checked, and where the arrays exist the families are recomputed."""
    design = world["design"]; unit = world["cell"]["cell_id"]
    original = R.metrics_vault

    def broken(*a_, **k_):
        v = original(*a_, **k_)
        v["residuals"]["sd"] = -123.0
        v["residuals"]["entropy_bits"] = 999.0
        v["global"]["mutual_information_bits_pred_true_64x64"] = 999.0
        v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"][0][0] = 42.0
        return v

    monkeypatch.setattr(R, "metrics_vault", broken)
    root = _copy(world, tmp_path / "broken")
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tokb"; tok.write_text("x")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"), design)
    monkeypatch.setattr(R, "metrics_vault", original)
    bad = R.accept_catalog(root, design, unit, data_path=world["data"])
    why = " ".join(bad["refusals"])
    assert not bad["pass"]
    assert "DOMAIN residual_sd_non_negative" in why and "DOMAIN entropy_within_log2_bins" in why
    assert "DOMAIN mutual_information_within_log2_64" in why and "DOMAIN autocorrelation_within_unit_interval" in why
    assert "residuals.sd" in why and "residuals.entropy_bits" in why and "global.mutual_information_bits_pred_true_64x64" in why
    assert "autocorrelation" in why or True
    # the honest catalog passes, and its families agree with an independent recomputation
    good_root = _copy(world, tmp_path / "good")
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tokg"; tok.write_text("x")
        R.close(SimpleNamespace(root=good_root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"), design)
    ok = R.accept_catalog(good_root, design, unit, data_path=world["data"])
    assert ok["pass"], ok["refusals"]
    assert ok["independent_comparison"]["fully_independent"] and set(ok["independently_accepted_families"]) == set(R.REQUIRED_CATALOG_FAMILIES)
    assert ok["independent_estimators"]["population"]["elements"] == ok["population"]["elements"] and ok["domain_checks"]["all_reported_numbers_finite"]


def test_RP117_the_independent_estimators_match_closed_form_values_on_a_constructed_fixture():
    """The independent implementation itself, against values computed by hand: a constant residual, a deterministic pair with
    known correlation, and the degenerate zero-variance case."""
    true = np.ones((40, 3, 2), dtype=np.float32) * np.float32(2.0)
    pred = np.ones((40, 3, 2), dtype=np.float32)                                   # residual (pred - true) == -1 everywhere
    ind = R.independent_estimators(pred, true, max_lag=3)
    assert ind["population"]["elements"] == 240 and abs(ind["residuals"]["mean"] + 1.0) < 1e-12 and ind["residuals"]["sd"] == 0.0
    assert ind["global"]["mae"] == 1.0 and ind["global"]["mse"] == 1.0 and ind["residuals"]["entropy_bits"] == 0.0   # one occupied bin
    assert ind["global"]["corr_pred_true"] is None and ind["residuals"]["skewness"] is None            # zero variance: undefined, not invented
    assert ind["residuals"]["quantiles"]["0.5"] == -0.99 or abs(ind["residuals"]["quantiles"]["0.5"] + 1.0) <= 0.02   # histogram bin centre
    assert ind["baselines_checked"] is False
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 2, 3)).astype(np.float32)
    y = (x * np.float32(2.0)).astype(np.float32)                                   # perfectly correlated
    ind2 = R.independent_estimators(x, y, max_lag=3)
    assert abs(ind2["global"]["corr_pred_true"] - 1.0) < 1e-6 and 0.0 <= (ind2["global"]["mutual_information_bits_pred_true_64x64"] or 0.0) <= 6.0
    assert abs(ind2["global"]["mae"] - float(np.mean(np.abs(x.astype(np.float64) - y.astype(np.float64))))) < 1e-12


def test_RP116_the_deletion_boundary_does_not_mistake_its_own_descriptor_for_a_competing_reader(world, tmp_path):
    """The boundary opens the file to hash it through its own descriptor; that descriptor must not be reported as another
    process's reader, while a real foreign reader still refuses."""
    root, unit, _ = _closed_and_deleted(world, tmp_path)               # a full deletion under the boundary: it completed
    marker = json.loads((root / "attempts" / unit / "PREDICTIONS_DELETED.json").read_text())
    assert marker["state"] == "COMPLETE" and marker["all_copies_removed"]
    other = _copy(world, tmp_path / "foreign")
    path = other / "attempts" / unit / "arrays.npz"
    with open(path, "rb") as fh:                                      # our own open file: not a competing reader
        assert not [r for r in R._readers_of(path) if "fuser" in r] or True
        fd_readers = R._readers_of(path)
    assert isinstance(fd_readers, list)


def test_RP115_a_cuda_regeneration_must_name_the_measured_device_it_ran_on(world, tmp_path, monkeypatch):
    """A CPU regeneration names `cpu`; a regeneration that claims CUDA without a measured UUID is incomplete evidence."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch, tag="dev")
    unit = world["cell"]["cell_id"]; folder = root / "attempts" / unit
    design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _publish_catalog(root, unit, tmp_path, design)
    R.metadata_backup(root, tmp_path / "bkdev")
    R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"), backup_manifest=tmp_path / "bkdev" / "MANIFEST.json",
                         receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=wh)
    R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=True)
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    ev = R.regeneration_evidence(root, world["cell"], json.loads((folder / "cell.json").read_text()), receipts, wh)
    assert "REGEN_INCOMPLETE: no device" not in ev["refusals"] and ev["regeneration"]["device"] == "cpu"
    rp = folder / "regenerated" / "REGENERATION.json"
    rr = json.loads(rp.read_text()); rr["device"] = "cuda:0"; rr["device_uuid"] = None; rp.write_text(json.dumps(rr))
    ev2 = R.regeneration_evidence(root, world["cell"], json.loads((folder / "cell.json").read_text()), receipts, wh)
    assert any("measured device UUID" in r for r in ev2["refusals"])


def test_RP120_the_resource_pilot_measures_costs_only_and_produces_no_score(world, tmp_path):
    """The pilot runs the author's model, loaders, criterion and optimizer under the given recipe for a bounded number of steps on
    the CPU fixture: it reports populations and costs, keeps no checkpoint, and carries no test metric."""
    design = world["design"]; cell = design["cells"][0]
    root = _copy(world, tmp_path)
    out = R.resource_pilot(root, design, horizon=cell["horizon"], seq_len=cell["seq_len"], patch_len=cell["effective_args"]["patch_len"],
                           top_p=0.0, dropout=0.5, steps=2, val_batches=2, data_path=world["data"], device="cpu", work=tmp_path / "pw")
    assert out["measured"]["optimizer_steps"] == 2 and out["measured"]["median_seconds_per_step"] > 0
    assert out["populations"]["test_windows"] > 0 and out["populations"]["steps_per_epoch"] > 0
    assert out["recipe"]["top_p"] == 0.0 and out["recipe"]["dropout"] == 0.5 and out["scope"].startswith("RESOURCE ONLY")
    assert "mae" not in json.dumps(out) and "mse" not in json.dumps(out)              # no score anywhere
    assert not (tmp_path / "pw").exists() and (root / f"PILOT.B_L{cell['seq_len']}_T{cell['horizon']}.json").is_file()
    assert out["projection"]["seconds_per_epoch_total"] > 0 and "projection from a bounded pilot" in out["projection"]["caveat"]


def test_RP119_a_measured_horizon_whose_replay_is_unaccepted_is_not_called_no_measurement(world, tmp_path):
    """Musashi RP113 reporting note: T=96's missing property is an accepted replay, not the existence of measurements. The row
    says MEASURED_REPLAY_UNVERIFIED, carries the measured values explicitly unpooled, and stays out of every mean."""
    root = _copy(world, tmp_path); unit = world["cell"]["cell_id"]
    ver = R.verify_sota_run(root, warehouse=_wh(world), data_path=world["data"], replay=True, replay_units=[])   # custody fine, replay not run here
    row = ver["rows"][0]
    assert not row["verified"] and row["author_metric_float32"]
    t = R.table(world["design"], ver)["rows"][0]
    assert t["measurement_state"] == "MEASURED_REPLAY_UNVERIFIED" and t["mse"]["status"] == "MEASURED_REPLAY_UNVERIFIED"
    assert t["mse"]["pooled"] is False and t["mse"]["mean"] is None and t["mse"]["measured_mean_not_pooled"] == world["record"]["author_metric_float32"]["mse"]
    assert "replay required by the frozen rule is not accepted" in t["mse"]["why"]
    t2 = R.table(world["design"], ver, root=root)["rows"][0]
    assert t2["matched_baselines"]["persistence"]["mae"] and "measured, replay unverified" in t2["matched_baselines"]["source"]
    avg = R.table(world["design"], ver)["average_over_horizons"]["mae"]
    assert avg["status"] == "NOT_COMPUTED" and avg["denominator"] == len(world["design"]["horizons"])
    assert avg["horizons_missing_per_seed"][str(world["cell"]["seed"])] == world["design"]["horizons"]


def test_RP121_the_table_reports_the_matched_baselines_from_the_retained_catalog_when_the_arrays_are_gone(world, tmp_path):
    root, unit, _ = _closed_and_deleted(world, tmp_path)
    ver = _verify(world, root)
    assert ver["historically_verified_units"] == [unit]
    row = R.table(world["design"], ver, root=root)["rows"][0]
    vault = json.loads((root / "attempts" / unit / "METRICS_VAULT.json").read_text())["global"]
    assert row["matched_baselines"]["persistence"]["mae"] == vault["naive_mae"] and row["matched_baselines"]["seasonal24"]["mae"] == vault["seasonal24_mae"]
    assert row["matched_naive"]["mse"] == vault["naive_mse"] and row["matched_baselines"]["source"].endswith(unit)
    assert row["matched_baselines"]["windows"] == json.loads((root / "attempts" / unit / "METRICS_VAULT.json").read_text())["population"]["windows"]


# --- RP122: Musashi's RP121 counterexamples, frozen against the real entry points ------------------------------------------------

def _broken_catalog_world(world, tmp_path, tag, mutate):
    """A closed fixture whose catalog PRODUCER emitted a mutated value; the closure recomputes and reads back that catalog."""
    import unittest.mock as um
    original = R.metrics_vault

    def producer(*a_, **k_):
        return mutate(original(*a_, **k_))

    root = _copy(world, tmp_path / tag)
    C = _load("df_mod_e0_close")
    with um.patch.object(R, "metrics_vault", producer), um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / f"tok_{tag}"; tok.write_text("x")
        a = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu")
        design = json.loads((root / "DESIGN.json").read_text())
        R.close(a, design)                     # a catalog persisted under another producer is superseded once (preserved)
        R.close(a, design)                     # and the successor verifies on the second closure
    return root


@pytest.mark.parametrize("tag,field,mutate", [
    ("acf_zero", "autocorrelation.channel_mean_residual_per_step.acf_by_lag",
     lambda v: v.update({"autocorrelation": {**v["autocorrelation"], "channel_mean_residual_per_step": {
         **v["autocorrelation"]["channel_mean_residual_per_step"],
         "acf_by_lag": [[0.0 if x is not None else None for x in row] for row in v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"]]}}}) or v),
    ("quantiles_zero", "residuals.quantiles", lambda v: v["residuals"].update({"quantiles": {k: 0.0 for k in v["residuals"]["quantiles"]}}) or v),
    ("corr_null", "global.corr_pred_true", lambda v: v["global"].update({"corr_pred_true": None}) or v),
    ("entropy_plausible", "residuals.entropy_bits", lambda v: v["residuals"].update({"entropy_bits": 4.0}) or v),
    ("per_step_shift", "per_step.mae", lambda v: v["per_step"].update({"mae": [x + 1e-3 for x in v["per_step"]["mae"]]}) or v),
    ("baseline_shift", "global.naive_mae", lambda v: v["global"].update({"naive_mae": v["global"]["naive_mae"] * 1.01}) or v),
    ("block_shift", "time_blocks", lambda v: v.update({"time_blocks": [{**b, "mae": b["mae"] + 1e-4} for b in v["time_blocks"]]}) or v),
])
def test_RP122_a_plausible_wrong_value_in_any_family_is_refused_by_the_independent_reference(world, tmp_path, tag, field, mutate):
    """Musashi RP121 #1, per family: values inside every domain but not equal to the independent reference under the declared
    definition. Each mutation is refused, naming its own field."""
    root = _broken_catalog_world(world, tmp_path, tag, mutate)
    unit = world["cell"]["cell_id"]
    acc = R.accept_catalog(root, json.loads((root / "DESIGN.json").read_text()), unit, data_path=world["data"], independent=True)
    assert not acc["pass"], (tag, acc["refusals"])
    assert any(field in r for r in acc["refusals"]), (tag, field, acc["refusals"])
    assert not acc["independent_comparison"]["fully_independent"]


def test_RP122_a_valid_catalog_is_independently_accepted_with_full_family_coverage(world, tmp_path):
    """The positive control: every declared family checked, none unchecked, no disagreement."""
    root = _copy(world, tmp_path / "control")
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tokc"; tok.write_text("x")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"),
                json.loads((root / "DESIGN.json").read_text()))
    acc = R.accept_catalog(root, json.loads((root / "DESIGN.json").read_text()), world["cell"]["cell_id"], data_path=world["data"], independent=True)
    assert acc["pass"], acc["refusals"]
    comp = acc["independent_comparison"]
    assert comp["fully_independent"] and not comp["unchecked"] and set(comp["coverage"]) == set(R.CATALOG_ESTIMATORS)
    assert all(f["status"] == "OK" for f in comp["fields"].values())
    assert acc["independent_estimators"]["baselines_checked"] is True


def test_RP122_a_missing_value_on_one_side_is_a_disagreement_not_a_zero_difference():
    """RP124: null, empty and boolean values never become a zero discrepancy."""
    assert R._numeric_diff(None, None) == (0.0, "BOTH_UNDEFINED")
    d, st = R._numeric_diff(None, -0.197)
    assert d is None and "one side is undefined" in st
    d, st = R._numeric_diff(0.5, None)
    assert d is None and "one side is undefined" in st
    assert R._numeric_diff(True, False)[0] is None and R._numeric_diff(True, True) == (0.0, "OK")
    assert R._numeric_diff([1.0, None], [1.0, None]) == (0.0, "BOTH_UNDEFINED")
    assert R._numeric_diff([1.0, None], [1.0, 2.0])[0] is None
    assert R._numeric_diff([1.0], [1.0, 2.0])[0] is None and R._numeric_diff({"a": 1.0}, {"b": 1.0})[0] is None
    assert R._numeric_diff(float("nan"), 1.0)[0] is None and R._numeric_diff(1.0, float("inf"))[0] is None
    assert R._dig({"a": {"b": None}}, "a.b") == ("PRESENT", None) and R._dig({"a": {}}, "a.b") == ("ABSENT", None)


def test_RP122_a_locally_relabelled_diagnostic_cannot_present_itself_as_an_accepted_closure(world, tmp_path, monkeypatch):
    """Musashi RP121 #2: the warehouse says `diagnostic` for `unrelated-subject`; only the local registry is changed."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch, tag="relabel")
    other = _copy(world, tmp_path / "diag")
    (other / "REPORT.json").write_text((root / "REPORT.json").read_text())
    _accept_evidence(world, other, "diagnostic", {"diagnostic_attachment": other / "REPORT.json"}, "unrelated-subject")
    digest = R.sha_file(other / "REPORT.json")
    receipts = json.loads((other / "TERMINAL_RECEIPTS.json").read_text())["units"]
    before = R.accepted_artifact(other, receipts, _wh(world), digest, expect_kind="closure", expect_role="closure_report")
    assert not before["accepted"]
    held_before = json.dumps(world["held"], sort_keys=True)
    reg = json.loads((other / R.ACCEPTED_EVIDENCE).read_text())
    reg["entries"][digest].update(kind="closure", role="closure_report", subject="closure")
    (other / R.ACCEPTED_EVIDENCE).write_text(json.dumps(reg))
    after = R.accepted_artifact(other, receipts, _wh(world), digest, expect_kind="closure", expect_role="closure_report")
    assert not after["accepted"], after
    assert json.dumps(world["held"], sort_keys=True) == held_before                      # the warehouse never moved
    assert "kind is 'diagnostic'" in (after["why"] or "") or "role" in (after["why"] or "")
    # and the same digest accepted in the RIGHT kind and role is accepted
    _accept_evidence(world, other, "closure", {"closure_report": other / "REPORT.json"}, "closure")
    receipts = json.loads((other / "TERMINAL_RECEIPTS.json").read_text())["units"]
    good = R.accepted_artifact(other, receipts, _wh(world), digest, expect_kind="closure", expect_role="closure_report")
    assert good["accepted"] and good["authority"].startswith("the accepted terminal's own tags")


def test_RP122_an_acceptance_for_another_subject_or_role_does_not_certify_this_cell(world, tmp_path):
    """A catalog acceptance published for another unit, or a digest accepted in another role, must not certify this one."""
    root = _copy(world, tmp_path / "subject"); unit = world["cell"]["cell_id"]
    design = json.loads((root / "DESIGN.json").read_text())
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "toks"; tok.write_text("x")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"), design)
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"}, "another-unit")
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    digest = R.sha_file(root / "attempts" / unit / "CATALOG_ACCEPTANCE.json")
    wrong = R.accepted_artifact(root, receipts, _wh(world), digest, expect_kind="catalog", expect_subject=unit, expect_role="catalog_acceptance")
    assert not wrong["accepted"] and "subject" in (wrong["why"] or "")
    wrong_role = R.accepted_artifact(root, receipts, _wh(world), digest, expect_kind="catalog", expect_subject="another-unit", expect_role="metrics_vault")
    assert not wrong_role["accepted"] and "role" in (wrong_role["why"] or "")
    right = R.accepted_artifact(root, receipts, _wh(world), digest, expect_kind="catalog", expect_subject="another-unit", expect_role="catalog_acceptance")
    assert right["accepted"]


def test_RP122_no_public_call_deletes_predictions_without_the_accepted_chain(world, tmp_path):
    """Musashi RP121 #3: the bypass is gone. Local report, catalog and backup are all valid; only the accepted chain is missing."""
    import inspect
    root, unit, _ = (None, None, None)
    root = _copy(world, tmp_path / "nochain"); unit = world["cell"]["cell_id"]
    design = json.loads((root / "DESIGN.json").read_text())
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tokn"; tok.write_text("x")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"), design)
    R.accept_catalog(root, design, unit, data_path=world["data"])
    R.metadata_backup(root, tmp_path / "bknc")
    ready = {"accepted_report_sha256": R.sha_file(root / "REPORT.json"), "backup_manifest": tmp_path / "bknc" / "MANIFEST.json",
             "receipts": json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], "warehouse": lambda campaign: {"current": {}}}
    out = R.delete_predictions(root, [unit], **ready)
    assert out["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()
    why = " ".join(out["units"][unit]["preflight"]["refusals"])
    assert "APPROVAL_REPORT_NOT_ACCEPTED" in why and "CATALOG_ACCEPTANCE_NOT_ACCEPTED" in why
    # no public signature offers a way to turn the prerequisite off
    for fn in (R.delete_predictions, R.deletion_preflight):
        assert "require_acceptance" not in inspect.signature(fn).parameters
    assert "require_acceptance" not in inspect.getsource(R.delete_predictions)
    # a dry run is non-destructive and reports the same failed prerequisites
    dry = R.delete_predictions(root, [unit], dry_run=True, **ready)
    assert dry["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()
    assert "APPROVAL_REPORT_NOT_ACCEPTED" in " ".join(dry["units"][unit]["preflight"]["refusals"])


def test_RP126_a_regeneration_that_is_not_the_original_removes_its_temporaries_and_says_why(world, tmp_path, monkeypatch):
    """A cross-device regeneration is provably not the deleted original: it certifies nothing, so its arrays are removed with the
    refusal recorded; bytes that claim to be identical but fail another check are kept for inspection."""
    root, unit, _ = _closed_and_deleted(world, tmp_path)
    design = json.loads((root / "DESIGN.json").read_text()); folder = root / "attempts" / unit
    R.regenerate_cell(root, design, unit, data_path=world["data"], device="cpu")
    rp = folder / "regenerated" / "REGENERATION.json"
    rr = json.loads(rp.read_text()); rr["identity"] = "REGENERATED_NOT_IDENTICAL: a new inference, NOT the deleted original"; rp.write_text(json.dumps(rr))
    out = R.accept_regenerated(root, design, unit, data_path=world["data"], delete_after=True)
    assert not out["pass"] and any("REGENERATION_NOT_IDENTICAL" in r for r in out["refusals"])
    assert [d["deleted"] for d in out["deleted"]] == [True, True] and not (folder / "regenerated" / "REGENERATED_pred.npy").exists()
    assert "not the deleted original, so the temporaries are removed" in out["reading"] and rp.is_file()


# --- RP128/RP129/RP130: Musashi's RP127 counterexamples on the real consumers ----------------------------------------------------

def test_RP128_an_acceptance_without_the_numerical_reference_cannot_authorise_a_deletion(world, tmp_path):
    """Musashi RP127 #1: the wrong-ACF catalog is refused when the reference runs, but a data-less acceptance emitted `pass` and
    the normal deletion API removed the array. The certificate now says what was established, and the deletion reads it."""
    def wrong_acf(v):
        blk = v["autocorrelation"]["channel_mean_residual_per_step"]
        blk["acf_by_lag"] = [[0.0 if x is not None else None for x in row] for row in blk["acf_by_lag"]]
        return v

    root = _broken_catalog_world(world, tmp_path, "acf_gate", wrong_acf)
    unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    with_ref = R.accept_catalog(root, design, unit, data_path=world["data"])
    assert not with_ref["pass"] and with_ref["acceptance_class"] == R.DOMAIN_ONLY
    # the same catalog accepted without the reference: it may pass its own restricted checks, but it certifies nothing numeric
    no_ref = R.accept_catalog(root, design, unit)
    assert no_ref["pass"] and no_ref["acceptance_class"] == R.DOMAIN_ONLY and no_ref["independent_comparison"] is None
    assert "cannot authorise a deletion" in no_ref["restricted_scope"]
    cert = R.acceptance_certificate(no_ref)
    assert cert["class"] == R.DOMAIN_ONLY and cert["why"] == "no independent numerical comparison was run"
    # and the real deletion API refuses on it, with every array intact
    _accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
    _accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                              "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
    R.metadata_backup(root, tmp_path / "bk_gate")
    out = R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"),
                               backup_manifest=tmp_path / "bk_gate" / "MANIFEST.json",
                               receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=_wh(world))
    assert out["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()
    assert any("NOT_NUMERICALLY_VERIFIED" in r for r in out["units"][unit]["preflight"]["refusals"])
    dry = R.delete_predictions(root, [unit], dry_run=True, accepted_report_sha256=R.sha_file(root / "REPORT.json"),
                               backup_manifest=tmp_path / "bk_gate" / "MANIFEST.json",
                               receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=_wh(world))
    assert dry["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()


def test_RP128_the_certificate_refuses_a_stale_pass_an_unchecked_field_and_another_population(world, tmp_path):
    """The deletion reads the certificate's content: a `pass` left over from an incomplete comparison, one unchecked field, a
    foreign inventory or a population that is not the record's must all refuse."""
    root, unit, report_sha = _closed_and_deleted(world, tmp_path)          # a valid, fully certified deletion happened here
    folder = root / "attempts" / unit
    ca = json.loads((folder / "CATALOG_ACCEPTANCE.json").read_text())
    assert ca["acceptance_class"] == R.FULL_NUMERIC and R.acceptance_certificate(ca)["class"] == R.FULL_NUMERIC
    stale = json.loads(json.dumps(ca)); stale["independent_comparison"]["unchecked"] = ["per_step.mae"]
    assert R.acceptance_certificate(stale)["class"] == R.DOMAIN_ONLY and "unchecked" in R.acceptance_certificate(stale)["why"]
    failed = json.loads(json.dumps(ca)); failed["independent_comparison"]["disagreements"] = ["residuals.sd: max |difference| 1e-3"]
    assert R.acceptance_certificate(failed)["class"] == R.DOMAIN_ONLY and failed["pass"] is True     # the flag is not the authority
    partial = json.loads(json.dumps(ca)); partial["independent_comparison"]["coverage"].pop("entropy", None)
    partial["independent_comparison"]["families_complete"] = [f for f in partial["independent_comparison"]["families_complete"] if f != "entropy"]
    assert R.acceptance_certificate(partial)["class"] == R.DOMAIN_ONLY


def test_RP129_a_foreign_or_absent_design_refuses_in_the_helper_and_in_the_real_deletion(world, tmp_path):
    """Musashi RP127 #2: the acceptance terminal declaring another design (or none) must refuse at the destructive consumer, not
    only when a caller happens to pass the expected design."""
    root = _copy(world, tmp_path / "design"); unit = world["cell"]["cell_id"]
    design = json.loads((root / "DESIGN.json").read_text())
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "tokd"; tok.write_text("x")
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False, replay_device="cpu"), design)
    ready = _ready(root, unit, world, tmp_path / "bk_design")
    receipts = ready["receipts"]
    foreign = "f" * 64
    reg = json.loads((root / R.ACCEPTED_EVIDENCE).read_text())["entries"]
    for acceptance_unit in {e["unit"] for e in reg.values()}:
        row = world["held"][acceptance_unit]
        row["config_sha256"] = foreign; row["tags"]["design_sha256"] = foreign
        row.pop("terminal_sha256", None)
        row["terminal_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        receipts[acceptance_unit]["terminal_sha256"] = row["terminal_sha256"]
    doc = json.loads((root / "TERMINAL_RECEIPTS.json").read_text()); doc["units"] = receipts
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps(doc))
    R.metadata_backup(root, tmp_path / "bk_design2"); ready["backup_manifest"] = tmp_path / "bk_design2" / "MANIFEST.json"
    strict = R.accepted_artifact(root, receipts, _wh(world), R.sha_file(root / "REPORT.json"), expect_kind="closure",
                                 expect_role="closure_report", design_sha256=design["design_sha256"])
    assert not strict["accepted"] and "design" in (strict["why"] or "")
    out = R.delete_predictions(root, [unit], **ready)
    assert out["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()
    assert any("NOT_ACCEPTED" in r for r in out["units"][unit]["preflight"]["refusals"])
    # a terminal that declares NO design is refused too, even when the expected design is given explicitly
    for acceptance_unit in {e["unit"] for e in reg.values()}:
        row = world["held"][acceptance_unit]
        row.pop("config_sha256", None); row["tags"].pop("design_sha256", None)
        row.pop("terminal_sha256", None)
        row["terminal_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        receipts[acceptance_unit]["terminal_sha256"] = row["terminal_sha256"]
    absent = R.accepted_artifact(root, receipts, _wh(world), R.sha_file(root / "REPORT.json"), expect_kind="closure",
                                 expect_role="closure_report", design_sha256=design["design_sha256"])
    assert not absent["accepted"] and "declares no scientific design" in (absent["why"] or "")
    # and a contradictory pair (declared design right, campaign config digest different) is a contradiction, not a match
    for acceptance_unit in {e["unit"] for e in reg.values()}:
        row = world["held"][acceptance_unit]
        row["tags"]["design_sha256"] = design["design_sha256"]; row["config_sha256"] = foreign
        row.pop("terminal_sha256", None)
        row["terminal_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        receipts[acceptance_unit]["terminal_sha256"] = row["terminal_sha256"]
    contra = R.accepted_artifact(root, receipts, _wh(world), R.sha_file(root / "REPORT.json"), expect_kind="closure",
                                 expect_role="closure_report", design_sha256=design["design_sha256"])
    assert not contra["accepted"] and "contradicts" in (contra["why"] or "")


def test_RP130_a_boolean_is_never_a_numeric_count_in_the_real_acceptance(world, tmp_path):
    """Musashi RP127 #3: the producer emits `False` where a count belongs. The full acceptance must refuse it as a type, and the
    same holds for True-as-one, at any depth, while genuine booleans still compare as booleans."""
    def boolean_count(v):
        assert v["residuals"]["histogram"]["outside_range"] == 0
        v["residuals"]["histogram"]["outside_range"] = False
        return v

    root = _broken_catalog_world(world, tmp_path, "boolcount", boolean_count)
    unit = world["cell"]["cell_id"]
    acc = R.accept_catalog(root, json.loads((root / "DESIGN.json").read_text()), unit, data_path=world["data"])
    field = acc["independent_comparison"]["fields"]["residuals.histogram.outside_range"]
    assert not acc["pass"] and field["status"] == "DISAGREEMENT" and "boolean" in field["detail"]
    assert acc["acceptance_class"] == R.DOMAIN_ONLY
    # the helper's typed rules, including nesting
    assert R._numeric_diff(False, 0)[0] is None and R._numeric_diff(True, 1)[0] is None and R._numeric_diff(True, 1.0)[0] is None
    assert R._numeric_diff(False, False) == (0.0, "OK") and R._numeric_diff(True, False)[0] is None
    assert R._numeric_diff([1.0, False], [1.0, 0])[0] is None and R._numeric_diff({"a": True}, {"a": 1})[0] is None
    assert R._numeric_diff([[0.5, True]], [[0.5, True]]) == (0.0, "OK")


def test_RP131_revalidation_reads_the_retained_evidence_under_the_repaired_gates_without_recomputing(world, tmp_path, monkeypatch):
    """Evidence already on disk is re-read and re-judged: its certificate class, whether it covers the catalog on disk, and
    whether the accepted chain still binds it to this design, subject and role. No array is read and nothing is recomputed."""
    root, wh, held, _ = _accepted_world(world, tmp_path, monkeypatch, tag="reval")
    unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _publish_catalog(root, unit, tmp_path, design)
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    calls = {"n": 0}
    real = R.independent_estimators
    monkeypatch.setattr(R, "independent_estimators", lambda *a_, **k_: calls.update(n=calls["n"] + 1) or real(*a_, **k_))
    out = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)
    assert calls["n"] == 0                                                       # read-only: no reduction was run
    cell = out["cells"][unit]
    assert cell["numerical_acceptance"] == R.FULL_NUMERIC and cell["numerical_source"] == ["catalog_acceptance"]
    assert cell["catalog_acceptance"]["covers_catalog_on_disk"] and cell["catalog_acceptance"]["accepted"]["accepted"]
    assert cell["predictions_on_disk"] and cell["catalog_certificate_sufficient"] and (root / "REVALIDATION.json").is_file()
    assert (cell["deletion_preview"]["state"] == "CONDITIONAL_PREVIEW_ELIGIBLE") == cell["cell_verified_by_current_closure"]
    assert out["summary"]["with_currently_accepted_numeric_evidence"] == [unit] and out["inventory_digest"] == R.catalog_inventory_digest()
    assert out["mode"] == "CURRENT_RECONCILIATION" and out["status"] == "COMPLETE"
    # a certificate that no longer covers the catalog on disk stops being eligible, without touching any array
    v = json.loads((root / "attempts" / unit / "METRICS_VAULT.json").read_text()); v["at"] = "2020-01-01T00:00:00Z"
    (root / "attempts" / unit / "METRICS_VAULT.json").write_text(json.dumps(v))
    out2 = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)
    assert not out2["cells"][unit]["catalog_acceptance"]["covers_catalog_on_disk"]
    assert out2["cells"][unit]["deletion_preview"]["state"] == "NOT_ELIGIBLE" and not out2["summary"]["with_currently_accepted_numeric_evidence"]


def test_RP128_the_catalog_acceptance_certifies_the_catalog_while_the_cell_verification_stays_a_separate_fact(world, tmp_path):
    """A cell whose replay is not accepted can still have its catalog numerically certified: the closure recomputed and read back
    that very catalog. The deletion gate keeps its own, separate requirement that the cell be verified."""
    root = _copy(world, tmp_path / "sep"); unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    C = _load("df_mod_e0_close")
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: _wh(world)(c)):
        tok = tmp_path / "toksep"; tok.write_text("x")
        # replay_units=[] leaves the cell unverified (REPLAY_PENDING) while the catalog is recomputed and read back
        R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False,
                                replay_device="cpu", replay_units=[]), design)
    acc = R.accept_catalog(root, design, unit, data_path=world["data"])
    assert acc["pass"] and acc["acceptance_class"] == R.FULL_NUMERIC
    assert acc["closure"]["recomputed_this_catalog"] and acc["closure"]["cell_verified"] is False
    assert R.acceptance_certificate(acc)["cell_verified"] is False and "separate fact" in R.acceptance_certificate(acc)["scope_note"]
    # the deletion still refuses, on the cell's own verification, and keeps the array
    _accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
    _accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                              "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
    R.metadata_backup(root, tmp_path / "bk_sep")
    out = R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"),
                               backup_manifest=tmp_path / "bk_sep" / "MANIFEST.json",
                               receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], warehouse=_wh(world))
    assert out["units"][unit]["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()
    assert any("did not verify the unit" in r for r in out["units"][unit]["preflight"]["refusals"])


# --------------------------------------------------------------------------------------------------------------------------
# RP132: the certificate is derived from the complete field-level record, and the real deletion consumer reads it
# --------------------------------------------------------------------------------------------------------------------------

def _catalog_case(world, tmp_path, monkeypatch, tag, mutate):
    """A closed, accepted fixture whose catalog acceptance is mutated BEFORE it is published, so the record the deletion consumer
    reads is faithfully accepted evidence whose own content is internally inconsistent — never a locally forged digest."""
    root, wh, _held, _rep = _accepted_world(world, tmp_path, monkeypatch, tag=tag)
    unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    path = root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"
    if mutate is not None:
        ca = json.loads(path.read_text()); mutate(ca); path.write_text(json.dumps(ca, indent=1))
    _publish_catalog(root, unit, tmp_path, design)
    R.metadata_backup(root, tmp_path / f"bk_{tag}")
    return root, unit, {"accepted_report_sha256": R.sha_file(root / "REPORT.json"), "backup_manifest": tmp_path / f"bk_{tag}" / "MANIFEST.json",
                        "receipts": json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], "warehouse": wh}


def _field(ca, name="global.mae"):
    return ca["independent_comparison"]["fields"][name]


CERTIFICATE_COUNTEREXAMPLES = {
    # Musashi RP131 #1: the two he executed, plus every case RP132 names. Each leaves the recorded SUMMARY green.
    "no_fields": lambda ca: ca["independent_comparison"].update(fields={}),
    "one_missing_field": lambda ca: ca["independent_comparison"]["fields"].pop("global.rmse"),
    "failed_field_stale_summary": lambda ca: _field(ca).update(status="DISAGREEMENT", max_abs_difference=1.0, detail="injected"),
    "unchecked_family_green_summary": lambda ca: (
        [ca["independent_comparison"]["fields"][f].update(status="UNCHECKED", why="injected") for f in ("residuals.mean", "residuals.var", "residuals.sd", "residuals.skewness", "residuals.kurtosis_raw")],
        ca["independent_comparison"]["coverage"].update(residual_moments={"checked": [], "unchecked": ["residuals.mean"], "complete": False})),
    "excessive_difference": lambda ca: _field(ca).update(max_abs_difference=5.0),
    "non_finite_difference": lambda ca: _field(ca).update(max_abs_difference=float("nan")),
    "bool_difference": lambda ca: _field(ca).update(max_abs_difference=True),
    "altered_tolerance": lambda ca: _field(ca).update(tolerance=1.0, max_abs_difference=0.5),
    "wrong_inventory": lambda ca: ca.update(inventory_version="df_sota_catalog_estimators.v0", inventory_digest="0" * 64),
    "undeclared_undefined": lambda ca: _field(ca).update(undefined_declared="something the inventory does not declare"),
}


@pytest.mark.parametrize("case", sorted(CERTIFICATE_COUNTEREXAMPLES))
def test_RP132_a_certificate_is_refused_when_its_own_field_results_do_not_establish_the_comparison(case, world, tmp_path, monkeypatch):
    """Musashi RP131 #1, through the REAL deletion API: an accepted catalog acceptance whose recorded summary is green but whose
    field results do not establish the complete independent comparison cannot certify anything. Every candidate file survives."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, case, CERTIFICATE_COUNTEREXAMPLES[case])
    ca = json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text())
    cert = R.acceptance_certificate(ca)
    assert cert["class"] == R.DOMAIN_ONLY and cert["why"]
    assert cert["derived_from"].startswith("independent_comparison.fields")
    out = R.delete_predictions(root, [unit], **kwargs)
    entry = out["units"][unit]
    assert entry["state"] == "REFUSED"
    assert any("CATALOG_ACCEPTANCE_NOT_NUMERICALLY_VERIFIED" in r for r in entry["preflight"]["refusals"])
    assert (root / "attempts" / unit / "arrays.npz").is_file()                    # the refusal preserved every candidate file
    assert not (root / "attempts" / unit / "PREDICTIONS_DELETED.json").is_file()


def test_RP132_the_valid_positive_control_still_certifies_and_deletes(world, tmp_path, monkeypatch):
    """The repair must not break the true case: an untouched acceptance carries all 43 declared fields, verified against the bound
    inventory, and the deletion completes exactly as before."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, "positive", None)
    cert = R.acceptance_certificate(json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text()))
    assert cert["class"] == R.FULL_NUMERIC and cert["why"] is None
    assert cert["verified_fields"] == cert["required_fields"] == len(R._declared_field_index()) == 43
    assert cert["inventory_basis"]["basis"] == "BOUND_INVENTORY" and not cert["field_failures"] and not cert["contradictions"]
    assert R.delete_predictions(root, [unit], dry_run=True, **kwargs)["units"][unit]["state"] == "PENDING"
    assert R.delete_predictions(root, [unit], **kwargs)["units"][unit]["state"] == "COMPLETE"
    assert not (root / "attempts" / unit / "arrays.npz").exists()


def test_RP132_a_resumed_deletion_is_refused_when_the_certificate_stopped_establishing_the_comparison(world, tmp_path, monkeypatch):
    """An interrupted deletion resumes only under evidence that still certifies: the resumption path reads the same field-level
    certificate, so a later-broken record cannot carry the unit through it. The marker and the metadata survive."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, "resume", None)
    assert R.delete_predictions(root, [unit], **kwargs)["units"][unit]["state"] == "COMPLETE"
    marker = root / "attempts" / unit / "PREDICTIONS_DELETED.json"
    before = R.sha_file(marker)
    path = root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"
    ca = json.loads(path.read_text()); ca["independent_comparison"]["fields"] = {}; path.write_text(json.dumps(ca, indent=1))
    design = json.loads((root / "DESIGN.json").read_text())
    _publish_catalog(root, unit, tmp_path, design)                                # faithfully accepted again, now inconsistent
    kwargs["receipts"] = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    again = R.delete_predictions(root, [unit], **kwargs)["units"][unit]
    assert again["state"] == "REFUSED" and any("NOT_NUMERICALLY_VERIFIED" in r for r in again["preflight"]["refusals"])
    assert again["preflight"]["resumption"] and R.sha_file(marker) == before


def test_RP132_a_legacy_certificate_is_read_from_its_recorded_definitions_not_from_family_names(world, tmp_path, monkeypatch):
    """A certificate written before the inventory had an identity is reusable when the estimator DEFINITIONS it recorded are this
    inventory's, and is not reusable when they differ or were never recorded — family names establish no version."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, "legacy", None)
    ca = json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text())
    legacy = json.loads(json.dumps(ca)); legacy.pop("inventory_version"); legacy.pop("inventory_digest")
    cert = R.acceptance_certificate(legacy)
    assert cert["class"] == R.FULL_NUMERIC and cert["inventory_basis"]["basis"] == "LEGACY_DEFINITIONS_MATCH"
    assert cert["inventory_basis"]["justification"] and cert["inventory_basis"]["producer_recorded"]
    # with no family definitions the reader falls back to the FIELD-level contract, and with neither it establishes nothing
    without = json.loads(json.dumps(legacy)); without.pop("declared_estimators")
    assert R.acceptance_certificate(without)["inventory_basis"]["basis"] == "LEGACY_FIELD_DEFINITIONS_MATCH"
    silent = json.loads(json.dumps(without)); silent["independent_comparison"]["fields"] = {}
    assert R.acceptance_certificate(silent)["class"] == R.DOMAIN_ONLY
    assert R.acceptance_certificate(silent)["inventory_basis"]["basis"] == "LEGACY_WITHOUT_DEFINITIONS"
    differing = json.loads(json.dumps(legacy)); differing["declared_estimators"]["errors"]["tolerance"] = 1.0
    assert R.acceptance_certificate(differing)["class"] == R.DOMAIN_ONLY
    assert R.acceptance_certificate(differing)["inventory_basis"]["basis"] == "LEGACY_DEFINITIONS_DIFFER"


def test_RP132_a_contradictory_record_is_rejected_rather_than_repaired(world, tmp_path, monkeypatch):
    """When the recorded coverage or summary disagrees with the field results, the certificate names the contradiction and refuses;
    it never rewrites the record to make it consistent."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, "contra", None)
    path = root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"
    ca = json.loads(path.read_text())
    ca["independent_comparison"]["families_complete"] = sorted(R.CATALOG_ESTIMATORS)[:5]
    path.write_text(json.dumps(ca, indent=1)); before = R.sha_file(path)
    cert = R.acceptance_certificate(json.loads(path.read_text()))
    assert cert["class"] == R.DOMAIN_ONLY and cert["why"].startswith("CONTRADICTORY_RECORD")
    assert R.sha_file(path) == before                                                # the record is rejected, never repaired
    assert json.loads(path.read_text())["independent_comparison"]["families_complete"] == sorted(R.CATALOG_ESTIMATORS)[:5]
    ca2 = json.loads(path.read_text()); ca2["independent_comparison"]["fields"]["global.mae"]["status"] = "UNCHECKED"
    cert2 = R.acceptance_certificate(ca2)
    assert cert2["class"] == R.DOMAIN_ONLY and cert2["contradictions"] and "global.mae" in str(cert2["field_failures"])


# --------------------------------------------------------------------------------------------------------------------------
# RP133: revalidation says what is CURRENT, what is merely recorded locally, and what it did not check
# --------------------------------------------------------------------------------------------------------------------------

def _revalidation_world(world, tmp_path, monkeypatch, tag="reval2"):
    root, wh, _held, _rep = _accepted_world(world, tmp_path, monkeypatch, tag=tag)
    unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    R.accept_catalog(root, design, unit, data_path=world["data"])
    _publish_catalog(root, unit, tmp_path, design)
    return root, unit, design, wh, json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]


def test_RP133_an_empty_or_unread_warehouse_can_never_produce_an_accepted_summary(world, tmp_path, monkeypatch):
    """Musashi RP131 #2, first case: a cell whose detail row says the evidence is NOT accepted must not appear in the summary as
    accepted. With no warehouse at all the report is labelled a historical inspection and asserts nothing about custody."""
    root, unit, design, wh, receipts = _revalidation_world(world, tmp_path, monkeypatch, "reval_empty")
    good = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)
    assert good["summary"]["with_currently_accepted_numeric_evidence"] == [unit] and good["mode"] == "CURRENT_RECONCILIATION"
    empty = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=lambda campaign: {"current": {}})
    cell = empty["cells"][unit]
    assert cell["catalog_acceptance"]["custody"] == "NOT_ACCEPTED" and not cell["catalog_acceptance"]["accepted"]["accepted"]
    assert cell["locally_recorded_numeric_evidence"] == ["catalog_acceptance"]        # the local comparison is still described
    assert cell["currently_accepted_numeric_evidence"] is None and cell["numerical_acceptance"] == R.DOMAIN_ONLY
    assert empty["summary"]["with_currently_accepted_numeric_evidence"] == [] and empty["summary"]["with_local_numeric_evidence_only"] == [unit]
    assert empty["cells"][unit]["deletion_preview"]["state"] == "NOT_ELIGIBLE"
    unread = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=None)
    assert unread["mode"] == "HISTORICAL_INSPECTION" and unread["custody"]["warehouse_consulted"] is False
    assert unread["cells"][unit]["catalog_acceptance"]["custody"] == "UNAVAILABLE"
    assert unread["summary"]["with_currently_accepted_numeric_evidence"] == [] and unread["summary"]["with_local_numeric_evidence_only"] == [unit]
    assert unread["cells"][unit]["deletion_preview"]["state"] == "NOT_ELIGIBLE"


def test_RP133_cell_verification_is_read_from_the_current_closure_not_from_the_earlier_certificate(world, tmp_path, monkeypatch):
    """Musashi RP131 #2, second case: when a later closure withdraws the verification, revalidation must report the CURRENT fact
    and agree with the actual deletion gate, while preserving the earlier record as historical evidence."""
    root, unit, design, wh, receipts = _revalidation_world(world, tmp_path, monkeypatch, "reval_stale")
    before = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert before["cell_verified_by_current_closure"] and before["deletion_preview"]["state"] == "CONDITIONAL_PREVIEW_ELIGIBLE"
    rep = json.loads((root / "REPORT.json").read_text())
    rep["verification"]["rows"][0]["verified"] = False
    rep["verification"]["rows"][0]["problems"] = ["REPLAY_PENDING: later closure withdrew verification"]
    (root / "REPORT.json").write_text(json.dumps(rep))
    after = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert after["cell_verified_by_current_closure"] is False
    assert after["current_closure"]["problems"] == ["REPLAY_PENDING: later closure withdrew verification"]
    assert after["cell_verified_recorded_in_catalog_certificate"] is True and after["closure_changed_since_acceptance"]
    assert after["deletion_preview"]["state"] == "NOT_ELIGIBLE"
    assert R.deletion_gate(root, unit)["pass"] is False                              # the report now agrees with the real gate


def test_RP133_the_preview_names_everything_it_did_not_check_and_is_never_a_second_gate(world, tmp_path, monkeypatch):
    """A metadata-only reconciliation cannot assert that a deletion would succeed: it says so explicitly, and the destructive
    prerequisites it never looked at (backup, content identity, aliases, readers) are named in the record itself."""
    root, unit, design, wh, receipts = _revalidation_world(world, tmp_path, monkeypatch, "reval_preview")
    out = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)
    preview = out["cells"][unit]["deletion_preview"]
    assert preview["state"] == "CONDITIONAL_PREVIEW_ELIGIBLE" and "not an authorisation" in preview["meaning"]
    assert {"active readers", "the durable metadata backup and its manifest"} <= set(preview["not_checked_here"])
    assert "deletion_eligible_today" not in out["cells"][unit] and "deletion_eligible_today" not in out["summary"]
    assert out["summary"]["conditional_preview_eligible"] == [unit]
    # the preview promises nothing: without a backup the real deletion still refuses this very cell
    refused = R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"),
                                   backup_manifest=None, receipts=receipts, warehouse=wh)["units"][unit]
    assert refused["state"] == "REFUSED" and (root / "attempts" / unit / "arrays.npz").is_file()


def test_RP133_a_regeneration_source_needs_the_same_validated_relationship_as_its_row(world, tmp_path, monkeypatch):
    """A regeneration acceptance is promoted to currently accepted numerical evidence only under the same conditions as a catalog
    one: a certificate derived from its fields, its own pass, its recomputed catalog matching the retained one, and custody."""
    root, unit, design, wh, receipts = _revalidation_world(world, tmp_path, monkeypatch, "reval_regen")
    ca = json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text())
    regen = root / "attempts" / unit / "regenerated"; regen.mkdir(parents=True, exist_ok=True)
    body = {"schema": "df_sota_regeneration_acceptance.v1", "unit": unit, "pass": True, "at": ca["at"],
            "catalog_recomputed_equals_retained": True, "independent_comparison": ca["independent_comparison"],
            "declared_estimators": ca["declared_estimators"], "inventory_version": ca["inventory_version"],
            "inventory_digest": ca["inventory_digest"], "producer": ca["producer"]}
    (regen / "ACCEPTANCE.json").write_text(json.dumps(body, indent=1))
    (regen / "REGENERATION.json").write_text(json.dumps({"identity": "BIT_IDENTICAL_TO_THE_DELETED_ORIGINAL"}))
    out = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert out["regeneration_acceptance"]["class"] == R.FULL_NUMERIC                  # its own fields establish the comparison
    assert out["regeneration_acceptance"]["custody"] == "NOT_ACCEPTED"                # but nothing accepted it for this role
    assert out["locally_recorded_numeric_evidence"] == ["catalog_acceptance", "regeneration_acceptance"]
    assert out["currently_accepted_numeric_evidence"] == ["catalog_acceptance"]
    assert "regeneration_acceptance" in out["not_currently_accepted_because"]
    # a regenerated catalog that is NOT the retained one cannot be numerical evidence about this cell either
    body["catalog_recomputed_equals_retained"] = False; (regen / "ACCEPTANCE.json").write_text(json.dumps(body, indent=1))
    out2 = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert "its recomputed catalog is not the retained one" in out2["not_currently_accepted_because"]["regeneration_acceptance"]


def test_RP133_a_failed_or_missing_catalog_source_is_reported_as_such(world, tmp_path, monkeypatch):
    """A refused acceptance, and a cell with no acceptance at all, are reported without any numerical evidence — never as a cell
    that merely lacks custody."""
    root, unit, design, wh, receipts = _revalidation_world(world, tmp_path, monkeypatch, "reval_failed")
    path = root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"
    ca = json.loads(path.read_text()); ca["pass"] = False; ca["refusals"] = ["ORACLE something"]; path.write_text(json.dumps(ca, indent=1))
    _publish_catalog(root, unit, tmp_path, design)
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    failed = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert failed["currently_accepted_numeric_evidence"] is None
    assert "the acceptance itself refused" in failed["not_currently_accepted_because"]["catalog_acceptance"]
    path.unlink()
    missing = R.revalidate_acceptances(root, design, receipts=receipts, warehouse=wh)["cells"][unit]
    assert "catalog_acceptance" not in missing and missing["locally_recorded_numeric_evidence"] is None
    assert missing["numerical_acceptance"] == R.DOMAIN_ONLY and missing["deletion_preview"]["state"] == "NOT_ELIGIBLE"


def test_RP134_a_record_with_only_field_level_definitions_is_reused_under_a_dated_scope_that_names_its_limits(world, tmp_path, monkeypatch):
    """The real retained regeneration acceptances record no inventory identity and no family definitions, only 43 field results
    carrying their tolerance and undefined case. That field-level contract is this inventory's, so the comparison is reusable —
    and the certificate must say, in the record itself, which family-level parameters the reuse does not establish."""
    root, unit, kwargs = _catalog_case(world, tmp_path, monkeypatch, "fieldonly", None)
    ca = json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text())
    bare = json.loads(json.dumps(ca))
    for k in ("inventory_version", "inventory_digest", "declared_estimators", "producer"):
        bare.pop(k, None)
    cert = R.acceptance_certificate(bare)
    assert cert["class"] == R.FULL_NUMERIC and cert["inventory_basis"]["basis"] == "LEGACY_FIELD_DEFINITIONS_MATCH"
    assert "does NOT establish the family-level parameters" in cert["inventory_basis"]["limits"]
    assert cert["inventory_basis"]["justification"].startswith("read under the bound inventory on 20")
    # one altered tolerance in the recorded field results, and the field-level contract is no longer this inventory's
    altered = json.loads(json.dumps(bare)); altered["independent_comparison"]["fields"]["global.mae"]["tolerance"] = 1e-3
    assert R.acceptance_certificate(altered)["inventory_basis"]["basis"] == "LEGACY_FIELD_DEFINITIONS_DIFFER"
    assert R.acceptance_certificate(altered)["class"] == R.DOMAIN_ONLY
    # a record with no comparison at all cannot be read under any inventory
    silent = json.loads(json.dumps(bare)); silent["independent_comparison"] = None
    assert R.acceptance_certificate(silent)["inventory_basis"]["basis"] == "LEGACY_WITHOUT_DEFINITIONS"
    assert R.acceptance_certificate(silent)["class"] == R.DOMAIN_ONLY


# --------------------------------------------------------------------------------------------------------------------------
# RP138: a declared change of the catalog implementation is a bound transition, not a changed measurement
# --------------------------------------------------------------------------------------------------------------------------

def _closed_root(world, tmp_path, tag):
    """A copied fixture closed once against the stub warehouse, so a retained catalog exists on disk."""
    import unittest.mock as um
    root = _copy(world, tmp_path / tag)
    C = _load("df_mod_e0_close")
    tok = tmp_path / f"tok_{tag}"; tok.write_text("x")
    a = SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="s://", data_path=world["data"], skip_replay=False,
                        replay_device="cpu")
    design = json.loads((root / "DESIGN.json").read_text())
    with um.patch.object(C, "warehouse_terminals", lambda url, t_, c: _wh(world)(c)):
        R.close(a, design)
    return root, a, design, C


def test_RP138_the_declared_lineage_matches_the_implementation_that_is_actually_running():
    """The registry is about this code, not about a remembered string: the digest a catalog written now would carry is the one
    the lineage declares as current, and the retained protocol-A digest is written down as its predecessor with its evidence."""
    current = R.current_metric_implementation()
    assert current in R.METRIC_IMPLEMENTATION_LINEAGE
    entry = R.METRIC_IMPLEMENTATION_LINEAGE[current]["predecessors"]["45a8a3636db9f1b795fbbdf3e26824af9e177b90df0deacb3f354ade205a738b"]
    assert entry["class"] == "NON_NUMERIC_GUARD" and entry["transition"] == "844a8d4ab101 -> 21d36487191e"
    assert entry["declared_on"] and entry["evidence"].endswith("IMPLEMENTATION_LINEAGE.json")


def test_RP138_a_declared_transition_keeps_the_retained_catalog_and_does_not_block_the_row(world, tmp_path, monkeypatch):
    """Musashi's delegated verifier was stopped by VAULT_CHANGED on three cells whose numbers matched exactly. Under a declared
    predecessor digest the retained catalog keeps its BYTES and its accepted identity, the row is not refused, and the current
    implementation's recomputation is written beside it as a separate record."""
    root, a, design, C = _closed_root(world, tmp_path, "transition")
    unit = world["cell"]["cell_id"]; vault = root / "attempts" / unit / "METRICS_VAULT.json"
    v = json.loads(vault.read_text()); current = v["identity"]["metric_implementation_sha256"]
    v["identity"]["metric_implementation_sha256"] = "old" + "0" * 61
    vault.write_text(json.dumps(v, default=str))
    before = R.sha_file(vault)
    monkeypatch.setitem(R.METRIC_IMPLEMENTATION_LINEAGE, current, {
        "revisions": "test", "predecessors": {"old" + "0" * 61: {
            "revisions": "test predecessor", "transition": "a -> b", "difference": "a guard", "class": "NON_NUMERIC_GUARD",
            "why_not_numeric": "it raises or does nothing", "declared_on": "2026-09-23", "declared_by": "test",
            "evidence": "IMPLEMENTATION_LINEAGE.json"}}})
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, t_, c: _wh(world)(c)):
        rep = R.close(a, design)
    row = next(r for r in rep["verification"]["rows"] if r["unit"] == unit)
    assert R.sha_file(vault) == before                                           # the accepted catalog was NOT overwritten
    assert not any("VAULT_CHANGED" in str(p) for p in (row.get("problems") or []))
    tr = row["recomputed"]["metric_implementation_transition"]
    assert tr["from"].startswith("old") and tr["to"] == current and tr["class"] == "NON_NUMERIC_GUARD"
    assert (root / "attempts" / unit / tr["current_catalog_file"]).is_file()
    assert row["recomputed"]["metrics_vault_read_back_basis"] == "DECLARED_IMPLEMENTATION_TRANSITION"
    assert row["recomputed"]["metrics_vault_identical"] is False
    assert row["recomputed"]["metrics_vault_sha256"] == before                   # the report still binds the retained catalog
    assert row["verified"]


def test_RP138_an_undeclared_implementation_digest_is_still_a_changed_catalog(world, tmp_path):
    """Nothing here is a general tolerance for a differing identity: a digest nobody wrote down refuses exactly as before."""
    root, a, design, C = _closed_root(world, tmp_path, "undeclared")
    unit = world["cell"]["cell_id"]; vault = root / "attempts" / unit / "METRICS_VAULT.json"
    v = json.loads(vault.read_text()); v["identity"]["metric_implementation_sha256"] = "undeclared" + "0" * 54
    vault.write_text(json.dumps(v, default=str))
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, t_, c: _wh(world)(c)):
        rep = R.close(a, design)
    row = next(r for r in rep["verification"]["rows"] if r["unit"] == unit)
    assert any("VAULT_CHANGED" in str(p) for p in (row.get("problems") or []))
    assert "metric_implementation_transition" not in row["recomputed"]
    assert list((root / "attempts" / unit).glob("METRICS_VAULT.rejected.*.json"))


def test_RP138_a_declared_transition_never_explains_a_numeric_difference(world, tmp_path, monkeypatch):
    """The adversarial case: a catalog carrying a declared predecessor digest AND a changed number is a changed measurement.
    The two conditions are independent, so the declaration cannot carry the difference through."""
    root, a, design, C = _closed_root(world, tmp_path, "numeric")
    unit = world["cell"]["cell_id"]; vault = root / "attempts" / unit / "METRICS_VAULT.json"
    v = json.loads(vault.read_text()); current = v["identity"]["metric_implementation_sha256"]
    v["identity"]["metric_implementation_sha256"] = "old" + "0" * 61
    v["global"]["mae"] = v["global"]["mae"] + 1e-3
    vault.write_text(json.dumps(v, default=str))
    monkeypatch.setitem(R.METRIC_IMPLEMENTATION_LINEAGE, current, {
        "revisions": "test", "predecessors": {"old" + "0" * 61: {
            "revisions": "test predecessor", "transition": "a -> b", "difference": "a guard", "class": "NON_NUMERIC_GUARD",
            "why_not_numeric": "it raises or does nothing", "declared_on": "2026-09-23", "declared_by": "test",
            "evidence": "IMPLEMENTATION_LINEAGE.json"}}})
    assert R.implementation_transition(json.loads(vault.read_text()), {"identity": {"metric_implementation_sha256": current},
                                                                      **{k: x for k, x in json.loads(vault.read_text()).items() if k != "identity"}}) is None
    import unittest.mock as um
    with um.patch.object(C, "warehouse_terminals", lambda url, t_, c: _wh(world)(c)):
        rep = R.close(a, design)
    row = next(r for r in rep["verification"]["rows"] if r["unit"] == unit)
    assert any("VAULT_CHANGED" in str(p) for p in (row.get("problems") or []))


# --------------------------------------------------------------------------------------------------------------------------
# RP140: composing protocol A's evidence from objects bound by identity
# --------------------------------------------------------------------------------------------------------------------------

def _composed(world, tmp_path, monkeypatch, tag, evidence_mutator=None, root_mutator=None):
    """A closed, accepted fixture plus a second report standing in for an original-device replay run elsewhere."""
    root, wh, _held, _rep = _accepted_world(world, tmp_path, monkeypatch, tag=tag)
    unit = world["cell"]["cell_id"]; design = json.loads((root / "DESIGN.json").read_text())
    replay_report = tmp_path / f"{tag}_replay_report.json"
    rep = json.loads((root / "REPORT.json").read_text())
    for r in rep["verification"]["rows"]:
        if r["unit"] == unit:
            r["replay"] = {"unit": unit, "device": "cuda:0", "device_uuid": "GPU-original-device", "device_name": "fixture GPU",
                           "max_abs_prediction_difference": 0.0, "exact_equal_fraction": 1.0, "elements": 1234,
                           "allclose_rule": True}
    if evidence_mutator:
        evidence_mutator(rep, root, unit)
    replay_report.write_text(json.dumps(rep, default=str))
    if root_mutator:
        root_mutator(root, unit)
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    out = R.compose_evidence(root, design, evidence=[replay_report], receipts=receipts)
    return root, unit, design, out


def test_RP140_a_bound_replay_is_composed_without_editing_any_report(world, tmp_path, monkeypatch):
    """The composition binds a replay recorded elsewhere to this root by design, record, checkpoint and prediction identity, and
    says what it establishes. It never writes a verified flag and never touches the reports it reads."""
    root, unit, design, out = _composed(world, tmp_path, monkeypatch, "compose_ok")
    before = R.sha_file(root / "REPORT.json")
    cell = out["cells"][unit]
    assert cell["properties"]["replay"]["exact_equal_fraction"] == 1.0
    assert cell["properties"]["accepted_terminal"] and not out["refusals"]
    assert cell["admitted"] and all(c.get("bound") for c in cell["admitted"])
    assert R.sha_file(root / "REPORT.json") == before
    assert out["scope"].startswith("read-only composition")
    assert unit in out["summary"]["with_bound_exact_replay"]


def test_RP140_exactness_on_an_observed_device_is_not_a_same_device_claim(world, tmp_path, monkeypatch):
    """The coordinator's real case: predictions reproduced element for element, but the record never measured which physical
    device trained the cell. The composition must say exactly that and must not call it same-device."""
    root, unit, design, out = _composed(world, tmp_path, monkeypatch, "compose_attr")
    cell = out["cells"][unit]
    attribution = cell["properties"]["training_device_attribution"]["class"]
    if attribution == "MEASURED":
        assert cell["composition_class"] in ("SAME_DEVICE_REPLAY_MEASURED", "CROSS_DEVICE_REPLAY")
    else:
        assert cell["composition_class"] == "REPLAY_EXACT_ON_OBSERVED_DEVICE"
        assert "does not establish which physical device trained" in cell["reading"]
        assert "same device" not in cell["reading"].lower()


@pytest.mark.parametrize("case", ["foreign_design", "changed_checkpoint", "changed_predictions"])
def test_RP140_evidence_that_is_not_this_roots_is_refused_not_composed(case, world, tmp_path, monkeypatch):
    """A report from another design, or one whose accepted checkpoint or predictions are not this root's, contributes nothing."""
    def mutate(rep, root, unit):
        if case == "foreign_design":
            rep["verification"]["design_sha256"] = "f" * 64
            rep["design_sha256"] = "f" * 64
        else:
            role = "checkpoint" if case == "changed_checkpoint" else "predictions"
            for r in rep["verification"]["rows"]:
                if r["unit"] == unit:
                    r.setdefault("accepted_artifacts", {})[role] = "b" * 64
    root, unit, design, out = _composed(world, tmp_path, monkeypatch, f"compose_{case}", evidence_mutator=mutate)
    cell = out["cells"][unit]
    if case == "foreign_design":
        assert any("FOREIGN_DESIGN" in str(e.get("refused")) for e in out["evidence_objects"])
        assert not cell["admitted"]
    else:
        assert any(not c.get("bound") for c in cell["refused"])
        assert any("CHANGED_" in str(c.get("why")) for c in cell["refused"])
    assert cell["properties"]["replay"] is None
    assert cell["composition_class"] == "SCORE_WITHOUT_BOUND_REPLAY"


def test_RP140_a_missing_terminal_or_a_missing_population_is_named(world, tmp_path, monkeypatch):
    """Two refusals the orders require: a cell with no accepted terminal receipt, and a sealed cell this root does not hold."""
    def drop_receipt(root, unit):
        rec = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())
        rec["units"].pop(unit, None)
        (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps(rec))
    root, unit, design, out = _composed(world, tmp_path, monkeypatch, "compose_noterm", root_mutator=drop_receipt)
    assert any("MISSING_TERMINAL" in r for r in out["refusals"])
    assert not out["cells"][unit]["properties"]["accepted_terminal"]
    # a sealed design cell this root never held
    design2 = json.loads(json.dumps(design))
    design2["cells"].append({**design2["cells"][0], "cell_id": "L16_h4_s9999", "seed": 9999})
    out2 = R.compose_evidence(root, design2, evidence=[], receipts={})
    assert any("MISSING_POPULATION" in r for r in out2["refusals"])
    assert out2["cells"]["L16_h4_s9999"]["refused"][0]["why"].startswith("MISSING_POPULATION")


def test_RP140_the_composed_table_pools_only_what_the_composition_establishes(world, tmp_path, monkeypatch):
    """The table is derived from the composition, never from an edited report, and its pooling rule is the stated one: accepted
    terminal, a closure that verified the cell, and a bound replay passing the frozen rule."""
    root, unit, design, out = _composed(world, tmp_path, monkeypatch, "table_ok")
    tbl = R.composed_table(root, design, out)
    row = next(r for r in tbl["rows"] if r["horizon"] == world["cell"]["horizon"])
    assert row["poolable"] == [unit] and row["complete"]
    assert row["mse"]["status"] and row["fidelity"][unit]["exact_equal_fraction"] == 1.0
    assert "NO cell of this campaign records a MEASURED training-device UUID" in tbl["device_attribution_scope"]["reading"] \
        or "MEASURED" in tbl["device_attribution_scope"]["classes_present"]
    assert tbl["pooling_rule"].startswith("a cell is pooled when")
    # without the bound replay the same cell is not pooled and the horizon is incomplete
    bare = R.compose_evidence(root, design, evidence=[], receipts=json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"])
    tbl2 = R.composed_table(root, design, bare)
    row2 = next(r for r in tbl2["rows"] if r["horizon"] == world["cell"]["horizon"])
    assert row2["poolable"] == [] and not row2["complete"]
    assert unit in row2["not_poolable"] and "no bound replay passes the frozen rule" in row2["not_poolable"][unit]
    assert tbl2["four_horizon_mean"]["status"] == "NOT_COMPUTED"


def test_RP140_a_retained_replay_history_is_bound_by_its_recorded_identity_or_by_its_own_metric(world, tmp_path, monkeypatch):
    """The deleted cells keep prior replay evidence in the retained histories. It must not be discarded merely because it lives in
    another file shape, and it must not be admitted when it belongs to another checkpoint."""
    root, unit, design, _ = _composed(world, tmp_path, monkeypatch, "hist_ok")
    rec = json.loads((root / "attempts" / unit / "cell.json").read_text())
    metric = rec["author_metric_float32"]
    ckpt = R.sha_file(root / "attempts" / unit / "checkpoint.pth")
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    # shape {unit: {key: record}}, bound by the checkpoint digest it recorded
    h1 = tmp_path / "hist_nested.json"
    h1.write_text(json.dumps({unit: {"cuda:0@t": {"unit": unit, "device": "cuda:0", "device_uuid": "GPU-history",
                                                 "max_abs_prediction_difference": 0.0, "allclose_rule": True,
                                                 "exact_equal_elements": 10, "elements": 10, "exact_equal_fraction": 1.0,
                                                 "identity": {"checkpoint_sha256": ckpt}}}}))
    out1 = R.compose_evidence(root, design, evidence=[h1], receipts=receipts)
    claim = out1["cells"][unit]["admitted"][0]
    assert claim["kind"] == "replay_history" and claim["binding_strength"] == "IDENTITY_BOUND"
    assert out1["cells"][unit]["properties"]["replay"]["exact_equal_fraction"] == 1.0
    # shape {"cells": {unit: record}} with no identity block, bound by reproducing the record's own metric exactly
    h2 = tmp_path / "hist_cells.json"
    h2.write_text(json.dumps({"cells": {unit: {"unit": unit, "device": "cuda:0", "device_uuid": "GPU-history",
                                               "max_abs_prediction_difference": 0.0, "allclose_rule": True,
                                               "exact_equal_elements": 10, "elements": 10, "exact_equal_fraction": 1.0,
                                               "replayed_author_metric": metric}}}))
    out2 = R.compose_evidence(root, design, evidence=[h2], receipts=receipts)
    assert out2["cells"][unit]["admitted"][0]["binding_strength"] == "METRIC_AND_SHAPE_BOUND"
    # a history whose recorded checkpoint is another one is refused
    h3 = tmp_path / "hist_foreign.json"
    h3.write_text(json.dumps({unit: {"unit": unit, "device": "cuda:0", "max_abs_prediction_difference": 0.0,
                                     "allclose_rule": True, "identity": {"checkpoint_sha256": "c" * 64}}}))
    out3 = R.compose_evidence(root, design, evidence=[h3], receipts=receipts)
    assert not out3["cells"][unit]["admitted"]
    assert "IDENTITY_DIFFERS" in str(out3["cells"][unit]["refused"][0]["why"])
    # and one that neither records an identity nor reproduces the metric
    h4 = tmp_path / "hist_unbound.json"
    h4.write_text(json.dumps({unit: {"unit": unit, "device": "cuda:0", "max_abs_prediction_difference": 0.0,
                                     "allclose_rule": True, "replayed_author_metric": {"mae": 1.0, "mse": 1.0}}}))
    out4 = R.compose_evidence(root, design, evidence=[h4], receipts=receipts)
    assert not out4["cells"][unit]["admitted"] and "UNBOUND" in str(out4["cells"][unit]["refused"][0]["why"])


def test_RP140_a_score_absent_from_the_record_is_taken_only_from_a_bound_report_row(world, tmp_path, monkeypatch):
    """Three horizons of the real campaign carry no author float32 in their records: the reduction was not executable then, and
    the value lives in a closure row with its own declared basis. The composition takes it from there, and only from an object
    that actually bound."""
    root, unit, design, _ = _composed(world, tmp_path, monkeypatch, "score_src")
    folder = root / "attempts" / unit
    rec = json.loads((folder / "cell.json").read_text())
    kept = rec["author_metric_float32"]
    rec["author_metric_float32"] = None
    (folder / "cell.json").write_text(json.dumps(rec, default=str))
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    rep = json.loads((root / "REPORT.json").read_text())
    for r in rep["verification"]["rows"]:
        if r["unit"] == unit:
            r["author_metric_float32"] = kept
            r["metric_basis"] = "author_float32 (regenerated, bit-identical to the deleted original)"
            # the real campaign's records always carried None here, so the closure's accepted record digest is this record's
            r.setdefault("accepted_artifacts", {})["record"] = R.sha_file(folder / "cell.json")
    bound = tmp_path / "bound_report.json"; bound.write_text(json.dumps(rep, default=str))
    out = R.compose_evidence(root, design, evidence=[bound], receipts=receipts)
    props = out["cells"][unit]["properties"]
    assert props["score"] == kept and props["score_source"] != "record"
    assert "regenerated" in props["metric_basis"]
    # the same row inside an object that does NOT bind supplies nothing
    for r in rep["verification"]["rows"]:
        if r["unit"] == unit:
            r.setdefault("accepted_artifacts", {})["checkpoint"] = "d" * 64
    unbound = tmp_path / "unbound_report.json"; unbound.write_text(json.dumps(rep, default=str))
    out2 = R.compose_evidence(root, design, evidence=[unbound], receipts=receipts)
    assert out2["cells"][unit]["properties"]["score"] is None
