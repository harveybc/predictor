"""Independent mathematical/data-preparation probes. No training or production writes."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("adequacy_probe", args.repo / "tools/df_adequacy_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    period = 40.0
    clean = np.sin(2 * np.pi * np.arange(128) / period)
    observed = clean.copy()
    observed[40] += 3.0
    _, y, _, oracle = m.task_series({"clean": clean, "observed": observed,
                                   "meta": {"period": period}}, "observed_increment")
    next_clean = 2 * np.cos(2 * np.pi / period) * clean[40] - clean[39]
    conditional_prediction = next_clean - observed[40]
    print(json.dumps({"past_noise_only_old_error": float(abs(y[40] - oracle[40])),
                      "conditional_oracle_error": float(abs(y[40] - conditional_prediction))}))
    print(json.dumps({str(w): m.D.boundaries(w, 768) for w in (4, 256)}, sort_keys=True))
    print(json.dumps({str(w): {"conv_layers": len(m.D.conv_dilations(w)),
                              "rf": m.D.receptive_field(3, m.D.conv_dilations(w))}
                      for w in (4, 256)}))


if __name__ == "__main__":
    main()
