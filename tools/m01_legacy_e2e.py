"""Run one legacy flat-config experiment through the INSTALLED predictor, outputs redirected.

    cd <checkout with examples/data>   # data paths in the config are relative
    python -I tools/m01_legacy_e2e.py --config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
        --out <dir> [--epochs 2 --max_steps_train 300 --max_steps_test 300 --mc_samples 2]

``python -I`` keeps the checkout off sys.path, so ``app.main`` is the installed one.
Every *_file output key, save_model and save_config is rewritten into --out; the
committed examples/results files are never touched. Prints a JSON summary with
the results-CSV schema (header, ordered Metric labels) so two installs can be
compared. This is a compatibility smoke, not a forecasting result.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

OUTPUT_KEYS = ("output_file", "results_file", "loss_plot_file", "model_plot_file", "uncertainties_file",
               "predictions_plot_file", "stl_plot_file", "wavelet_plot_file", "tapper_plot_file")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", default="2")
    p.add_argument("--max_steps_train", default="300")
    p.add_argument("--max_steps_test", default="300")
    p.add_argument("--mc_samples", default="2")
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(a.config).read_text())
    for key in OUTPUT_KEYS:
        if key in config:
            config[key] = str(out / Path(config[key]).name)
    config["save_model"] = str(out / "model.keras")
    config["save_config"] = str(out / "config_out.json")
    derived = out / "derived_config.json"
    derived.write_text(json.dumps(config, indent=2))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import subprocess
    import app
    # The legacy entry point is a SCRIPT (app/main.py imports its siblings as top-level
    # modules), so run the installed file as a script exactly as predictor.sh does with
    # a checkout: -E -s keep env/user site out; the script's own directory is sys.path[0]
    # and the current directory (the checkout, for data paths) is NOT on sys.path.
    main_py = Path(app.__file__).parent / "main.py"
    subprocess.run([sys.executable, "-E", "-s", str(main_py), "--load_config", str(derived),
                    "--epochs", a.epochs, "--max_steps_train", a.max_steps_train,
                    "--max_steps_test", a.max_steps_test, "--mc_samples", a.mc_samples],
                   check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    results = Path(config["results_file"])
    with results.open() as f:
        rows = list(csv.reader(f))
    summary = {"app_package": str(Path(app.__file__).parent), "results_header": rows[0],
               "metric_labels": [r[0] for r in rows[1:]],
               "outputs_present": sorted(p.name for p in out.iterdir())}
    (out / "e2e_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
