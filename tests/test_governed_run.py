"""tools/governed_run.py pure functions: results parsing, canonical config
hashing, input dedupe, resource mapping, output redirection, report shape.
No network, no predictor run."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GR = _load("governed_run")

RESULTS = """Metric,Average,Std Dev,Min,Max
Train MAE H24,0.006512,0.000716,0.005505,0.007103
Validation R2 H1,0.811183,0.041209,0.778825,0.869338
Test Uncertainty H144,0.013374,0.007056,0.003635,0.020127
Test AUC_ROC,0.71,,,
Naive_MAE,0.01,0.0,0.01,0.01
Train R2 H120,0.045317,0.201428,-0.132303,0.326995
"""


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_parse_results_rows_shapes():
    rows = GR.parse_results_rows(_rows(RESULTS))
    assert [set(r) for r in rows] == [
        {"metric", "value", "split", "horizon", "std_dev", "min_value", "max_value", "unit"}
    ] * 6
    assert rows[0] == {
        "metric": "MAE", "value": 0.006512, "split": "train", "horizon": 24,
        "std_dev": 0.000716, "min_value": 0.005505, "max_value": 0.007103, "unit": None,
    }
    assert (rows[1]["metric"], rows[1]["split"], rows[1]["horizon"]) == ("R2", "validation", 1)
    assert (rows[2]["metric"], rows[2]["split"], rows[2]["horizon"]) == ("Uncertainty", "test", 144)
    assert rows[3] == {
        "metric": "AUC_ROC", "value": 0.71, "split": "test", "horizon": None,
        "std_dev": None, "min_value": None, "max_value": None, "unit": None,
    }
    assert (rows[4]["metric"], rows[4]["split"], rows[4]["horizon"]) == ("Naive_MAE", None, None)
    assert rows[5]["min_value"] == -0.132303


def test_parse_results_csv_file_and_non_finite(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(RESULTS + "Train SNR H24,nan,inf,,\n", encoding="utf-8")
    rows = GR.parse_results_csv(path)
    assert len(rows) == 7
    assert rows[6]["metric"] == "SNR" and rows[6]["value"] is None and rows[6]["std_dev"] is None
    assert GR.parse_results_rows(_rows("Metric,Average\n,1\n")) == []


def _effective(cache, out):
    return {
        "x_train_file": f"{cache}/predictor_examples/{'1' * 64}.csv",
        "y_train_file": f"{cache}/predictor_examples/{'1' * 64}.csv",
        "x_validation_file": f"{cache}/predictor_examples/{'2' * 64}.csv",
        "results_file": f"{out}/phase_1_ann_1575_1d_results.csv",
        "output_file": f"{out}/phase_1_ann_1575_1d_prediction.csv",
        "uncertainties_file": f"{out}/phase_1_ann_1575_1d_uncertanties.csv",
        "loss_plot_file": f"{out}/loss_plot.png",
        "stl_plot_file": f"{out}/stl.png",
        "save_model": f"{out}/predictor_model.keras",
        "save_config": f"{out}/config_out.json",
        "save_log": f"{out}/debug_out.json",
        "epochs": 2,
        "predicted_horizons": [9, 12],
        "use_normalization_json": "examples/data_downsampled/phase_1/normalization_config_b.json",
    }


IDENTITIES = {
    "x_train_file": f"gov:predictor_examples/phase_1/normalized_d4.csv@{'1' * 64}",
    "y_train_file": f"gov:predictor_examples/phase_1/normalized_d4.csv@{'1' * 64}",
    "x_validation_file": f"gov:predictor_examples/phase_1/normalized_d5.csv@{'2' * 64}",
}


def test_canonical_config_invariant_to_cache_and_out_dir():
    a = GR.canonical_config(_effective("/cache-a", "/out-a"), IDENTITIES)
    b = GR.canonical_config(_effective("/somewhere/else", "/run-7"), IDENTITIES)
    assert a == b
    assert GR.sha256_text(a) == GR.sha256_text(b)
    assert len(GR.sha256_text(a)) == 64
    body = json.loads(a)
    assert body["x_train_file"] == IDENTITIES["x_train_file"]
    assert body["x_validation_file"] == IDENTITIES["x_validation_file"]
    assert body["results_file"] == "phase_1_ann_1575_1d_results.csv"
    assert body["loss_plot_file"] == "loss_plot.png"
    assert body["stl_plot_file"] == "stl.png"
    assert body["save_model"] == "predictor_model.keras"
    assert "save_config" not in body and "save_log" not in body
    assert body["epochs"] == 2 and body["predicted_horizons"] == [9, 12]
    assert list(body) == sorted(body)
    assert a == json.dumps(body, sort_keys=True, separators=(",", ":"))
    # a different dataset identity or hyperparameter is a different hash
    other = dict(IDENTITIES, x_train_file=f"gov:predictor_examples/phase_1/normalized_d4.csv@{'9' * 64}")
    assert GR.canonical_config(_effective("/cache-a", "/out-a"), other) != a
    assert GR.canonical_config(dict(_effective("/cache-a", "/out-a"), epochs=3), IDENTITIES) != a


def test_resolve_inputs_and_dedupe(tmp_path):
    config = {
        "x_train_file": "examples/data_downsampled/phase_1/normalized_d4.csv",
        "y_train_file": "examples/data_downsampled/phase_1/normalized_d4.csv",
        "x_validation_file": "examples/data_downsampled/phase_1/../phase_1/normalized_d5.csv",
        "y_validation_file": "examples/data_downsampled/phase_1/normalized_d5.csv",
        "x_test_file": str(tmp_path / "d6.csv"),
        "y_test_file": None,
    }
    inputs = GR.resolve_inputs(config, tmp_path)
    assert list(inputs) == [
        "x_train_file", "y_train_file", "x_validation_file", "y_validation_file", "x_test_file",
    ]
    assert inputs["x_train_file"] == (tmp_path / "examples/data_downsampled/phase_1/normalized_d4.csv").resolve()
    assert inputs["x_validation_file"] == inputs["y_validation_file"]
    distinct = GR.distinct_paths(inputs)
    assert len(distinct) == 3
    assert distinct[0] == inputs["x_train_file"]
    assert distinct[2] == tmp_path.resolve() / "d6.csv"


def test_resource_mapping(tmp_path):
    root = tmp_path / "examples" / "data_downsampled"
    path = root / "phase_1" / "normalized_d4.csv"
    assert GR.resource_for(path, root) == "phase_1/normalized_d4.csv"
    assert GR.resource_for(root / "phase_1" / ".." / "phase_1" / "x.csv", root) == "phase_1/x.csv"
    with pytest.raises(GR.GovernedRunError, match="outside the lake root"):
        GR.resource_for(tmp_path / "examples" / "data" / "phase_1" / "normalized_d4.csv", root)


def test_output_redirection(tmp_path):
    out = tmp_path / "run"
    config = {
        "x_train_file": "examples/data_downsampled/phase_1/normalized_d4.csv",
        "y_train_file": "examples/data_downsampled/phase_1/normalized_d4.csv",
        "output_file": "examples/results/phase_1_daily/phase_1_ann_1575_1d_prediction.csv",
        "results_file": "examples/results/phase_1_daily/phase_1_ann_1575_1d_results.csv",
        "loss_plot_file": "examples/results/phase_1_daily/phase_1_ann_1575_1d_loss_plot.png",
        "uncertainties_file": "examples/results/phase_1_daily/phase_1_ann_1575_1d_uncertanties.csv",
        "stl_plot_file": "examples/results/phase_1_daily/phase_1_1575_1d_stl_decomposition_plot.png",
        "epochs": 10000,
    }
    cached = {"x_train_file": "/cache/predictor_examples/aa.csv", "y_train_file": "/cache/predictor_examples/aa.csv"}
    governed = GR.governed_config(config, cached, out)
    assert governed["x_train_file"] == governed["y_train_file"] == "/cache/predictor_examples/aa.csv"
    assert governed["results_file"] == str(out / "phase_1_ann_1575_1d_results.csv")
    assert governed["output_file"] == str(out / "phase_1_ann_1575_1d_prediction.csv")
    assert governed["uncertainties_file"] == str(out / "phase_1_ann_1575_1d_uncertanties.csv")
    assert governed["loss_plot_file"] == str(out / "phase_1_ann_1575_1d_loss_plot.png")
    assert governed["stl_plot_file"] == str(out / "phase_1_1575_1d_stl_decomposition_plot.png")
    # keys the config left to predictor's defaults are redirected too, so
    # nothing lands in the repository root
    assert governed["save_model"] == str(out / "predictor_model.keras")
    assert governed["save_config"] == str(out / "config_out.json")
    assert governed["save_log"] == str(out / "debug_out.json")
    assert governed["model_plot_file"] == str(out / "model_plot.png")
    assert governed["predictions_plot_file"] == str(out / "predictions_plot.png")
    assert governed["epochs"] == 10000
    for key, value in governed.items():
        if key.endswith("_file") and key not in GR.INPUT_KEYS or key in ("save_model", "save_config", "save_log"):
            assert str(value).startswith(str(out)), key
    assert config["results_file"].startswith("examples/results")  # input untouched


def test_build_report_shape():
    metrics = GR.parse_results_rows(_rows(RESULTS))
    datasets = [
        {"lake": "predictor_examples", "resource": "phase_1/normalized_d4.csv", "sha256": "1" * 64, "role": "x_train_file"},
        {"lake": "predictor_examples", "resource": "phase_1/normalized_d4.csv", "sha256": "1" * 64, "role": "y_train_file"},
    ]
    report = GR.build_report(
        "olap_cube", metrics, datasets, experiment_set_key="set-1",
        config_sha256="c" * 64, code_commit="abc-dirty", project="predictor",
        phase="phase_1_daily", tags={"plugin": "ann"},
    )
    assert report["lake"] == "olap_cube"
    assert report["experiment_set_key"] == "set-1"
    assert [d["role"] for d in report["datasets"]] == ["x_train_file", "y_train_file"]
    assert report["metrics"] is metrics
    assert report["tags"] == {"plugin": "ann"}
    json.dumps(report, allow_nan=False)


def test_main_splits_extra_flags_and_fails_closed(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATA_GOV_API_KEY", raising=False)
    cfg = tmp_path / "c.json"
    cfg.write_text("{}", encoding="utf-8")
    code = GR.main([
        "--load_config", str(cfg), "--experiment-key", "k", "--out-dir", str(tmp_path / "o"),
        "--", "--epochs", "2",
    ])
    assert code == 1
    assert "governed_run: no API key" in capsys.readouterr().err
    assert not (tmp_path / "o" / "GOVERNED_RUN.json").exists()
    assert GR.main(["--load_config", str(cfg), "--experiment-key", "bad key", "--out-dir", "x"]) == 1
    assert "invalid --experiment-key" in capsys.readouterr().err
