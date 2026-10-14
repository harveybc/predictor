"""DMO10 localization probe: three instances of one model, identical batch, per-head sums.

For lane 4's bounded GPU scope (CPU tests cover the logic): in ONE session it
builds a model, saves it, then scores the SAME synthetic batch through

1. ``in_session``  - the built instance itself,
2. ``rebuild``     - ``load_dmo``'s canonical rebuild + load_weights instance,
3. ``full_load``   - the Keras full-model deserialization of the archive,

computing per-head float64 sums/counts through the shared evaluation procedure
and printing which pair diverges bitwise. No fitting; nothing scientific.

Run: ``CUDA_VISIBLE_DEVICES=<scope> python -m tools.i6dmo.gpu_probe_dmo10 [--out DIR]``
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def run(out_dir):
    from tools.i6dmo import model as M
    from tools.i6dmo import train as T
    import tensorflow as tf

    M.enable_determinism()
    rng = np.random.default_rng(6102026)
    features = [f"f{i}" for i in range(19)]
    bundle = M.build_dmo(M.model_config(features))
    x = rng.normal(size=(300, 13, 19)).astype("float32")
    y = rng.normal(size=(300, 6))
    mask = rng.random((300, 6)) > 0.2
    mask[:, 0] = True
    out = Path(out_dir)
    M.save_dmo(bundle, out / "m")
    rebuilt, _ = M.load_dmo(out / "m")
    full = tf.keras.models.load_model(out / "m" / "forecast_model.keras", compile=False, safe_mode=True)
    instances = {"in_session": bundle.forecast_model, "rebuild": rebuilt.forecast_model, "full_load": full}
    sums = {}
    for name, model in instances.items():
        metrics = T.masked_split_metrics(np.where(mask, y, np.nan),
                                         M.predict_matrix(model, x, batch_size=256), mask)
        sums[name] = {"sums": metrics["sums"], "counts": metrics["counts"], "loss": metrics["loss"]}
    pairs, divergent = {}, []
    for a, b in itertools.combinations(instances, 2):
        equal = sums[a] == sums[b]
        pairs[f"{a}_vs_{b}"] = {"equal": equal,
                                "max_abs_sum_delta": max(abs(u - v) for u, v in
                                                         zip(sums[a]["sums"], sums[b]["sums"]))}
        if not equal:
            divergent.append(f"{a}_vs_{b}")
    report = {"schema": "i6dmo.dmo10_probe.v1",
              "device": "gpu" if tf.config.list_physical_devices("GPU") else "cpu",
              "determinism": dict(M.DETERMINISM), "rows": 300, "batch_size": 256,
              "per_instance": sums, "pairs": pairs, "divergent_pairs": sorted(divergent),
              "fits_run": 0, "test_rows_read": 0}
    (out / "DMO10_PROBE.json").write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.gpu_probe_dmo10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="dmo10_probe_")
    report = run(out)
    print(json.dumps({k: report[k] for k in ("device", "divergent_pairs")}, sort_keys=True))
    print(f"report: {Path(out) / 'DMO10_PROBE.json'}")
    return 0 if not report["divergent_pairs"] else 1


if __name__ == "__main__":
    sys.exit(main())
