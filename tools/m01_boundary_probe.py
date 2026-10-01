"""Record the legacy consumer-visible behaviour of whichever predictor is INSTALLED.

Run it once with master installed and once with the M01 tip installed, each in an
isolated environment and from a working directory OUTSIDE any checkout, then
diff the two JSON documents: every legacy boundary must be identical and the
only permitted difference is the added opt-in entry points.

    python tools/m01_boundary_probe.py --repo <checkout for example configs/models>
        --prediction-provider <prediction_provider checkout> --out probe.json

Boundaries recorded:
  * installed entry points of every group this distribution publishes;
  * app.plugin_resolver resolution (value + module path relative to the
    installed package) of every legacy predictor/pipeline/preprocessor/target
    name used by every examples/config JSON;
  * prediction_provider's DirectionPredictor._load_direction_model on the
    committed direction .keras + _metadata.json pairs: which route it took
    (rebuild through predictor.plugins vs fallback) and a digest of the
    predictions on a fixed input.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from importlib import metadata
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

GROUPS = ("predictor.plugins", "optimizer.plugins", "pipeline.plugins", "preprocessor.plugins",
          "target.plugins", "modular.branch", "modular.fusion", "modular.core", "modular.head")


def entry_point_table():
    out = {}
    for group in GROUPS:
        rows = sorted((ep.name, ep.value, ep.dist.name if ep.dist else None)
                      for ep in metadata.entry_points(group=group))
        out[group] = [list(r) for r in rows]
    return out


def config_resolutions(repo):
    from app.plugin_resolver import PLUGIN_ROLES, canonical_name, resolve
    out = {}
    for path in sorted((repo / "examples" / "config").rglob("*.json")):
        try:
            config = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(config, dict):
            continue
        row = {}
        for role in sorted(PLUGIN_ROLES):
            try:
                name = canonical_name(config, role)
                if name:
                    w = resolve(role, name)
                    row[role] = [name, w["entry_point_value"], Path(w["origin"]).name]
            except SystemExit as exc:          # the resolver refuses by raising SystemExit
                row[role] = ["REFUSED", str(exc)[:160]]
        if row:
            out[str(path.relative_to(repo))] = row
    return out


def direction_models(repo, provider):
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "pp_direction", provider / "plugins_predictor" / "direction_predictor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._QUIET = False
    out = {}
    for keras_path in sorted((repo / "examples" / "results").rglob("*direction*_model.keras")):
        if not keras_path.with_name(keras_path.stem + "_metadata.json").exists():
            continue
        predictor = module.DirectionPredictor()
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                model = predictor._load_direction_model(str(keras_path), "probe")
        except Exception as exc:          # record the consumer's behaviour, do not hide it
            lines = [l for l in log.getvalue().splitlines() if "DirectionPredictor" in l]
            out[str(keras_path.relative_to(repo))] = {
                "route": "consumer_error", "error": f"{type(exc).__name__}: {exc}"[:300],
                "consumer_log": [l[:300] for l in lines[-3:]]}
            continue
        shape = tuple(int(d) for d in model.input_shape[1:])
        x = np.random.default_rng(0).normal(size=(4, *shape)).astype("float32")
        y = np.asarray(model(x, training=False), dtype="float32")
        route = ("rebuilt_via_predictor_plugins" if "Rebuilt via plugin" in log.getvalue()
                 else "fallback_direct_load" if "Loaded model directly" in log.getvalue() else "other")
        out[str(keras_path.relative_to(repo))] = {
            "route": route, "input_shape": list(shape),
            "prediction_sha256_1e-6": hashlib.sha256(np.round(y, 6).tobytes()).hexdigest(),
            "weights": len(model.weights)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--prediction-provider", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import app
    dist = metadata.distribution("predictor")
    document = {
        "installed": {"version": dist.version, "app_package": str(Path(app.__file__).parent),
                      "checkout_on_path": any(Path(p).resolve() == args.repo.resolve()
                                              for p in __import__("sys").path if p)},
        "entry_points": entry_point_table(),
        "config_resolutions": config_resolutions(args.repo.resolve()),
        "direction_models": direction_models(args.repo.resolve(), args.prediction_provider.resolve()),
    }
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: len(v) for k, v in document.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
